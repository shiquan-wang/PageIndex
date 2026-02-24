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
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("LLM response does not contain JSON object.")
        return json.loads(match.group(0))


def _clean_node_list(candidate_ids: list[Any], allowed_ids: set[str], top_k: int | None = None) -> list[str]:
    output: list[str] = []
    seen = set()
    for item in candidate_ids:
        node_id = str(item)
        if node_id in allowed_ids and node_id not in seen:
            seen.add(node_id)
            output.append(node_id)
            if top_k is not None and len(output) >= top_k:
                break
    return output


def _build_stage1_prompt(query: str, top_level_nodes: list[dict[str, Any]], preference: str | None = None, max_select: int = 8) -> str:
    candidates = []
    for node in top_level_nodes:
        candidates.append(
            {
                "node_id": str(node.get("node_id")),
                "title": str(node.get("title", "")),
                "summary": str(node.get("summary", "")),
                "child_count": len(node.get("nodes", [])),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
            }
        )

    prompt = f"""
You are given a query and candidate top-level sections from a document tree.
Please select the top-level nodes that are most relevant to finding the answer.

Query: {query}
Top-level candidate nodes: {json.dumps(candidates, ensure_ascii=False)}
"""

    if preference:
        prompt += f"\nExpert Knowledge of relevant sections: {preference}\n"

    prompt += f"""
Reply in JSON format:
{{
  "thinking": <brief reasoning>,
  "node_list": [node_id1, node_id2, ...]
}}
Return at most {max_select} node ids.
"""
    return prompt.strip()


def _build_stage2_prompt(query: str, subtree_nodes: list[dict[str, Any]], preference: str | None = None, top_k: int = 8) -> str:
    prompt = f"""
You are given a query and a filtered document tree structure.
You need to find all nodes that are likely to contain the answer.

Query: {query}
Document tree structure: {json.dumps(subtree_nodes, ensure_ascii=False)}
"""

    if preference:
        prompt += f"\nExpert Knowledge of relevant sections: {preference}\n"

    prompt += f"""
Reply in JSON format:
{{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}
Return at most {top_k} node ids.
"""
    return prompt.strip()


def _collect_subtree_ids(root_id: str, index: dict[str, IndexedNode]) -> list[str]:
    queue = [root_id]
    result: list[str] = []
    while queue:
        current = queue.pop(0)
        result.append(current)
        queue.extend(index[current].children)
    return result


def _extract_subtrees(root_nodes: list[dict[str, Any]], selected_root_ids: set[str]) -> list[dict[str, Any]]:
    return [node for node in root_nodes if str(node.get("node_id")) in selected_root_ids]


def _default_tree_search_result(index: dict[str, IndexedNode], top_k: int) -> dict[str, Any]:
    ordered_nodes = sorted(index.values(), key=lambda n: (n.depth, n.node_id))
    node_list = [node.node_id for node in ordered_nodes[:top_k]]
    return {
        "thinking": "LLM caller is not provided, fallback to deterministic traversal result.",
        "node_list": node_list,
    }


def _llm_select_top_level_nodes(
    query: str,
    root_nodes: list[dict[str, Any]],
    all_node_ids: set[str],
    llm_caller: Callable[[str], str],
    preference: str | None,
    stage1_max_roots: int,
    stage1_chunk_size: int,
) -> tuple[list[str], str]:
    if len(root_nodes) <= stage1_chunk_size:
        prompt = _build_stage1_prompt(query=query, top_level_nodes=root_nodes, preference=preference, max_select=stage1_max_roots)
        parsed = _extract_json_from_text(llm_caller(prompt))
        selected = _clean_node_list(parsed.get("node_list", []), all_node_ids, top_k=stage1_max_roots)
        thinking = str(parsed.get("thinking", ""))
        return selected, thinking

    chunk_count = math.ceil(len(root_nodes) / stage1_chunk_size)
    per_chunk_select = max(1, math.ceil(stage1_max_roots / chunk_count))

    merged: list[str] = []
    thinking_parts: list[str] = []
    for i in range(0, len(root_nodes), stage1_chunk_size):
        chunk = root_nodes[i:i + stage1_chunk_size]
        prompt = _build_stage1_prompt(query=query, top_level_nodes=chunk, preference=preference, max_select=per_chunk_select)
        parsed = _extract_json_from_text(llm_caller(prompt))
        thinking = str(parsed.get("thinking", ""))
        if thinking:
            thinking_parts.append(f"chunk[{i // stage1_chunk_size + 1}]: {thinking}")
        chunk_selected = _clean_node_list(parsed.get("node_list", []), all_node_ids, top_k=per_chunk_select)
        for node_id in chunk_selected:
            if node_id not in merged:
                merged.append(node_id)

    return merged[:stage1_max_roots], " | ".join(thinking_parts)


def search_tree(
    query: str,
    tree_json: dict[str, Any] | list[dict[str, Any]],
    top_k: int = 8,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
    stage1_max_roots: int = 6,
    stage1_chunk_size: int = 20,
) -> dict[str, Any]:
    """Two-stage LLM-based tree retrieval.

    Stage 1: select relevant top-level roots (supports chunked selection when root count is large).
    Stage 2: rerank/choose final nodes only inside the selected subtrees.

    If llm_caller is None, a deterministic fallback result is returned for testing.
    """
    root_nodes = _normalize_tree_input(tree_json)
    index = build_tree_index(tree_json)
    if not index:
        return {"thinking": "empty tree", "node_list": [], "scored_nodes": [], "stage1_node_list": []}

    if llm_caller is None:
        result = _default_tree_search_result(index=index, top_k=top_k)
        stage1_node_list = []
        final_thinking = result["thinking"]
    else:
        all_node_ids = set(index.keys())
        stage1_selected, stage1_thinking = _llm_select_top_level_nodes(
            query=query,
            root_nodes=root_nodes,
            all_node_ids=all_node_ids,
            llm_caller=llm_caller,
            preference=preference,
            stage1_max_roots=stage1_max_roots,
            stage1_chunk_size=stage1_chunk_size,
        )

        if not stage1_selected:
            stage1_selected = [str(node.get("node_id")) for node in root_nodes[:stage1_max_roots]]
            stage1_thinking = f"{stage1_thinking} | stage1 fallback to first top-level nodes.".strip(" |")

        selected_subtrees = _extract_subtrees(root_nodes, set(stage1_selected))
        prompt_stage2 = _build_stage2_prompt(query=query, subtree_nodes=selected_subtrees, preference=preference, top_k=top_k)
        parsed_stage2 = _extract_json_from_text(llm_caller(prompt_stage2))

        candidate_ids = _clean_node_list(parsed_stage2.get("node_list", []), all_node_ids, top_k=top_k)
        if not candidate_ids:
            # deterministic local fallback inside selected roots
            candidate_ids = []
            for root_id in stage1_selected:
                candidate_ids.extend(_collect_subtree_ids(root_id, index))
            candidate_ids = _clean_node_list(candidate_ids, all_node_ids, top_k=top_k)

        stage2_thinking = str(parsed_stage2.get("thinking", ""))
        final_thinking = f"stage1: {stage1_thinking}\nstage2: {stage2_thinking}".strip()
        result = {"node_list": candidate_ids}
        stage1_node_list = stage1_selected

    scored_nodes = []
    for node_id in result["node_list"]:
        node = index.get(node_id)
        if node is None:
            continue
        scored_nodes.append(
            {
                "node_id": node.node_id,
                "title": node.title,
                "start_index": node.start_index,
                "end_index": node.end_index,
                "parent_id": node.parent_id,
            }
        )

    return {
        "thinking": final_thinking,
        "stage1_node_list": stage1_node_list,
        "node_list": [node["node_id"] for node in scored_nodes],
        "scored_nodes": scored_nodes,
    }


def search_tree_from_json_file(
    query: str,
    json_path: str | Path,
    top_k: int = 8,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
    stage1_max_roots: int = 6,
    stage1_chunk_size: int = 20,
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
        stage1_chunk_size=stage1_chunk_size,
    )
