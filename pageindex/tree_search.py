"""PageIndex JSON 结构的 LLM 树检索工具。

本模块专注于检索阶段的树遍历与节点筛选。
不直接依赖某个模型 SDK；由调用方注入 `llm_caller`。
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class IndexedNode:
    node_id: str
    title: str
    summary: str
    start_index: int | None
    end_index: int | None
    parent_id: str | None
    children: list[str]
    depth: int


def _normalize_tree_input(tree_json: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将输入统一为根节点列表。"""
    if isinstance(tree_json, dict):
        if "structure" in tree_json and isinstance(tree_json["structure"], list):
            return tree_json["structure"]
        if "nodes" in tree_json and isinstance(tree_json["nodes"], list):
            return tree_json["nodes"]
        raise ValueError("Unsupported JSON format. Expected a root with `structure` or `nodes`.")
    if isinstance(tree_json, list):
        return tree_json
    raise ValueError("Unsupported tree_json type. Expected dict or list.")


def build_tree_index(tree_json: dict[str, Any] | list[dict[str, Any]]) -> dict[str, IndexedNode]:
    """把树结构展开为 node_id -> IndexedNode 映射，便于快速访问。"""
    root_nodes = _normalize_tree_input(tree_json)
    index: dict[str, IndexedNode] = {}

    def walk(nodes: list[dict[str, Any]], parent_id: str | None, depth: int) -> None:
        for node in nodes:
            node_id = str(node["node_id"])
            children = [str(child["node_id"]) for child in node.get("nodes", [])]
            index[node_id] = IndexedNode(
                node_id=node_id,
                title=str(node.get("title", "")),
                summary=str(node.get("summary", "")),
                start_index=node.get("start_index"),
                end_index=node.get("end_index"),
                parent_id=parent_id,
                children=children,
                depth=depth,
            )
            if node.get("nodes"):
                walk(node["nodes"], node_id, depth + 1)

    walk(root_nodes, None, 0)
    return index


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """解析模型输出 JSON；若包含包裹文本则使用正则兜底提取。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("LLM response does not contain JSON object.")
        return json.loads(match.group(0))


def _clean_node_list(candidate_ids: list[Any], allowed_ids: set[str], limit: int | None = None) -> list[str]:
    """按合法性/顺序/去重清洗节点 id，并可按 `limit` 截断。"""
    output: list[str] = []
    seen = set()
    for item in candidate_ids:
        node_id = str(item)
        if node_id in allowed_ids and node_id not in seen:
            seen.add(node_id)
            output.append(node_id)
            if limit is not None and len(output) >= limit:
                break
    return output


def _estimate_subtree_tokens(root_id: str, index: dict[str, IndexedNode]) -> int:
    """粗略估算子树 token 规模，用于递归终止条件判断。"""
    queue = [root_id]
    token_total = 0
    while queue:
        current = queue.pop(0)
        node = index[current]
        token_total += max(1, math.ceil(len((node.title + "\n" + node.summary).strip()) / 4))
        queue.extend(node.children)
    return token_total


def _build_candidate_selection_prompt(
    stage_name: str,
    query: str,
    candidate_ids: list[str],
    index: dict[str, IndexedNode],
    preference: str | None,
    max_select: int,
) -> str:
    """为某一遍历阶段构造候选筛选提示词（带上限）。"""
    candidates = [
        {
            "node_id": index[nid].node_id,
            "title": index[nid].title,
            "summary": index[nid].summary,
            "depth": index[nid].depth,
            "child_count": len(index[nid].children),
            "start_index": index[nid].start_index,
            "end_index": index[nid].end_index,
        }
        for nid in candidate_ids
    ]

    prompt = f"""
你正在执行文档检索的树搜索。
当前阶段：{stage_name}
请从候选节点中选择所有与问题相关的节点。

问题：{query}
候选节点：{json.dumps(candidates, ensure_ascii=False)}
"""
    if preference:
        prompt += f"\n相关章节的专家知识：{preference}\n"
    prompt += f"""
请用 JSON 格式回复：
{{
  "thinking": <简要推理>,
  "node_list": [node_id1, node_id2, ...]
}}
最多返回 {max_select} 个节点 id。
"""
    return prompt.strip()


def _llm_select_nodes(
    stage_name: str,
    query: str,
    candidate_ids: list[str],
    index: dict[str, IndexedNode],
    llm_caller: Callable[[str], str],
    preference: str | None,
    max_select: int,
    chunk_size: int,
) -> tuple[list[str], str]:
    """从候选集合中选择相关节点；候选过大时自动分块。"""
    if len(candidate_ids) <= chunk_size:
        parsed = _extract_json_from_text(
            llm_caller(_build_candidate_selection_prompt(stage_name, query, candidate_ids, index, preference, max_select))
        )
        selected = _clean_node_list(parsed.get("node_list", []), set(candidate_ids), limit=max_select)
        return selected, str(parsed.get("thinking", ""))

    chunk_count = math.ceil(len(candidate_ids) / chunk_size)
    per_chunk_select = max(1, math.ceil(max_select / chunk_count))
    merged: list[str] = []
    thinking_parts: list[str] = []

    for i in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[i:i + chunk_size]
        parsed = _extract_json_from_text(
            llm_caller(_build_candidate_selection_prompt(stage_name, query, chunk, index, preference, per_chunk_select))
        )
        chunk_selected = _clean_node_list(parsed.get("node_list", []), set(chunk), limit=per_chunk_select)
        for node_id in chunk_selected:
            if node_id not in merged:
                merged.append(node_id)
        thinking = str(parsed.get("thinking", ""))
        if thinking:
            thinking_parts.append(f"chunk[{i // chunk_size + 1}]: {thinking}")

    return merged[:max_select], " | ".join(thinking_parts)


def _default_tree_search_result(index: dict[str, IndexedNode], limit: int | None) -> dict[str, Any]:
    """未提供 LLM 调用器时使用的确定性兜底结果。"""
    ordered = sorted(index.values(), key=lambda n: (n.depth, n.node_id))
    ids = [n.node_id for n in ordered]
    if limit is not None:
        ids = ids[:limit]
    return {"thinking": "未提供 LLM 调用器，回退到确定性遍历结果。", "node_list": ids}


def search_tree(
    query: str,
    tree_json: dict[str, Any] | list[dict[str, Any]],
    top_k: int | None = None,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
    stage1_max_roots: int | None = 6,
    candidate_chunk_size: int = 20,
    max_traversal_depth: int = 4,
    subtree_token_budget: int = 1600,
    per_level_max_select: int | None = 8,
) -> dict[str, Any]:
    """递归式 LLM 树检索。

    参数说明：
      - top_k=None：返回全部已筛中的相关节点。
      - stage1_max_roots：限制一级目录探索宽度；None 表示不限制。
      - per_level_max_select：限制每层递归筛选宽度；None 表示不限制。
    """
    root_nodes = _normalize_tree_input(tree_json)
    index = build_tree_index(tree_json)
    if not index:
        return {"thinking": "空树", "node_list": [], "scored_nodes": []}

    # 最终输出上限；为 None 时表示返回所有已筛中的相关节点。
    output_limit = top_k if top_k is not None else None

    if llm_caller is None:
        result = _default_tree_search_result(index=index, limit=output_limit)
        final_node_ids = result["node_list"]
        final_thinking = result["thinking"]
    else:
        root_ids = [str(node.get("node_id")) for node in root_nodes]
        # 第一阶段控制一级目录的探索宽度。
        stage1_limit = stage1_max_roots if stage1_max_roots is not None else len(root_ids)

        stage1_selected, stage1_thinking = _llm_select_nodes(
            stage_name="stage-1 top-level prefilter",
            query=query,
            candidate_ids=root_ids,
            index=index,
            llm_caller=llm_caller,
            preference=preference,
            max_select=stage1_limit,
            chunk_size=candidate_chunk_size,
        )
        if not stage1_selected:
            stage1_selected = root_ids[:stage1_limit]
            stage1_thinking = f"{stage1_thinking} | 第一阶段回退到确定性顺序。".strip(" |")

        thinking_parts = [f"stage1: {stage1_thinking}"]
        frontier = stage1_selected
        terminal_candidates: list[str] = []

        # 递归下钻：持续细化 frontier，直到命中终止条件。
        for depth in range(1, max_traversal_depth + 1):
            if not frontier:
                break
            # 每层宽度上限，用于控制分支数和 token 成本。
            level_limit = per_level_max_select if per_level_max_select is not None else len(frontier)

            selected, level_thinking = _llm_select_nodes(
                stage_name=f"recursive-level-{depth}",
                query=query,
                candidate_ids=frontier,
                index=index,
                llm_caller=llm_caller,
                preference=preference,
                max_select=level_limit,
                chunk_size=candidate_chunk_size,
            )
            if not selected:
                selected = frontier[:level_limit]
                level_thinking = f"{level_thinking} | 当前层回退到确定性顺序。".strip(" |")
            thinking_parts.append(f"level{depth}: {level_thinking}")

            next_frontier: list[str] = []
            for node_id in selected:
                node = index[node_id]
                if not node.children:
                    terminal_candidates.append(node_id)
                    continue
                # 若子树已足够小，或达到最大深度，则停止继续下钻。
                if _estimate_subtree_tokens(node_id, index) <= subtree_token_budget or depth >= max_traversal_depth:
                    terminal_candidates.append(node_id)
                else:
                    next_frontier.extend(node.children)

            frontier = next_frontier

        if not terminal_candidates:
            terminal_candidates = stage1_selected.copy()

        terminal_candidates = _clean_node_list(terminal_candidates, set(index.keys()))

        # 最终重排在 top_k=None 时也可不设上限。
        final_select_limit = output_limit if output_limit is not None else len(terminal_candidates)
        final_node_ids, final_thinking = _llm_select_nodes(
            stage_name="final-rerank",
            query=query,
            candidate_ids=terminal_candidates,
            index=index,
            llm_caller=llm_caller,
            preference=preference,
            max_select=final_select_limit,
            chunk_size=candidate_chunk_size,
        )
        if not final_node_ids:
            final_node_ids = terminal_candidates[:final_select_limit]
            final_thinking = f"{final_thinking} | 最终回退到终止候选顺序。".strip(" |")

        thinking_parts.append(f"final: {final_thinking}")
        final_thinking = "\n".join(thinking_parts)

    scored_nodes = []
    for node_id in final_node_ids:
        node = index.get(node_id)
        if node is None:
            continue
        scored_nodes.append({
            "node_id": node.node_id,
            "title": node.title,
            "start_index": node.start_index,
            "end_index": node.end_index,
            "parent_id": node.parent_id,
        })

    return {
        "thinking": final_thinking,
        "node_list": [node["node_id"] for node in scored_nodes],
        "scored_nodes": scored_nodes,
    }


def search_tree_from_json_file(
    query: str,
    json_path: str | Path,
    top_k: int | None = None,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
    stage1_max_roots: int | None = 6,
    candidate_chunk_size: int = 20,
    max_traversal_depth: int = 4,
    subtree_token_budget: int = 1600,
    per_level_max_select: int | None = 8,
) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        tree_json = json.load(f)
    return search_tree(
        query=query,
        tree_json=tree_json,
        top_k=top_k,
        preference=preference,
        llm_caller=llm_caller,
        stage1_max_roots=stage1_max_roots,
        candidate_chunk_size=candidate_chunk_size,
        max_traversal_depth=max_traversal_depth,
        subtree_token_budget=subtree_token_budget,
        per_level_max_select=per_level_max_select,
    )
