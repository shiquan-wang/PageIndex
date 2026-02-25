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


def _estimate_node_tokens(node: IndexedNode) -> int:
    text = f"{node.title}\n{node.summary}".strip()
    # rough estimate, enough for recursion stop heuristic
    return max(1, math.ceil(len(text) / 4))


def _estimate_subtree_tokens(root_id: str, index: dict[str, IndexedNode]) -> int:
    queue = [root_id]
    token_total = 0
    while queue:
        current = queue.pop(0)
        node = index[current]
        token_total += _estimate_node_tokens(node)
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
    candidates = []
    for node_id in candidate_ids:
        node = index[node_id]
        candidates.append(
            {
                "node_id": node.node_id,
                "title": node.title,
                "summary": node.summary,
                "depth": node.depth,
                "child_count": len(node.children),
                "start_index": node.start_index,
                "end_index": node.end_index,
            }
        )

    prompt = f"""
You are doing tree search for document retrieval.
Current stage: {stage_name}
Select the candidate nodes most relevant to answer the query.

Query: {query}
Candidate nodes: {json.dumps(candidates, ensure_ascii=False)}
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
    if len(candidate_ids) <= chunk_size:
        prompt = _build_candidate_selection_prompt(stage_name, query, candidate_ids, index, preference, max_select)
        parsed = _extract_json_from_text(llm_caller(prompt))
        selected = _clean_node_list(parsed.get("node_list", []), set(candidate_ids), top_k=max_select)
        return selected, str(parsed.get("thinking", ""))

    chunk_count = math.ceil(len(candidate_ids) / chunk_size)
    per_chunk_select = max(1, math.ceil(max_select / chunk_count))

    merged: list[str] = []
    thinking_parts: list[str] = []
    for i in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[i:i + chunk_size]
        prompt = _build_candidate_selection_prompt(stage_name, query, chunk, index, preference, per_chunk_select)
        parsed = _extract_json_from_text(llm_caller(prompt))
        thinking = str(parsed.get("thinking", ""))
        if thinking:
            thinking_parts.append(f"chunk[{i // chunk_size + 1}]: {thinking}")

        chunk_selected = _clean_node_list(parsed.get("node_list", []), set(chunk), top_k=per_chunk_select)
        for node_id in chunk_selected:
            if node_id not in merged:
                merged.append(node_id)

    return merged[:max_select], " | ".join(thinking_parts)


def _default_tree_search_result(index: dict[str, IndexedNode], top_k: int) -> dict[str, Any]:
    ordered_nodes = sorted(index.values(), key=lambda n: (n.depth, n.node_id))
    node_list = [node.node_id for node in ordered_nodes[:top_k]]
    return {
        "thinking": "LLM caller is not provided, fallback to deterministic traversal result.",
        "node_list": node_list,
    }


def search_tree(
    query: str,
    tree_json: dict[str, Any] | list[dict[str, Any]],
    top_k: int = 8,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
    stage1_max_roots: int = 6,
    candidate_chunk_size: int = 20,
    max_traversal_depth: int = 4,
    subtree_token_budget: int = 1600,
    per_level_max_select: int = 8,
) -> dict[str, Any]:
    """Recursive LLM tree retrieval with stop conditions.

    Traversal recursively descends relevant nodes until stop conditions are met:
      - node is leaf
      - estimated subtree token size <= subtree_token_budget
      - traversal level reaches max_traversal_depth
    """
    root_nodes = _normalize_tree_input(tree_json)
    index = build_tree_index(tree_json)
    if not index:
        return {"thinking": "empty tree", "node_list": [], "scored_nodes": [], "stage1_node_list": []}

    if llm_caller is None:
        result = _default_tree_search_result(index=index, top_k=top_k)
        final_node_ids = result["node_list"]
        stage1_node_list: list[str] = []
        final_thinking = result["thinking"]
    else:
        root_ids = [str(node.get("node_id")) for node in root_nodes]
        stage1_node_list, stage1_thinking = _llm_select_nodes(
            stage_name="stage-1 top-level prefilter",
            query=query,
            candidate_ids=root_ids,
            index=index,
            llm_caller=llm_caller,
            preference=preference,
            max_select=stage1_max_roots,
            chunk_size=candidate_chunk_size,
        )
        if not stage1_node_list:
            stage1_node_list = root_ids[:stage1_max_roots]
            stage1_thinking = f"{stage1_thinking} | stage1 fallback to first top-level nodes.".strip(" |")

        traversal_thinking_parts = [f"stage1: {stage1_thinking}"]

        frontier = stage1_node_list
        terminal_candidates: list[str] = []

        for depth in range(1, max_traversal_depth + 1):
            if not frontier:
                break

            selected, level_thinking = _llm_select_nodes(
                stage_name=f"recursive-level-{depth}",
                query=query,
                candidate_ids=frontier,
                index=index,
                llm_caller=llm_caller,
                preference=preference,
                max_select=per_level_max_select,
                chunk_size=candidate_chunk_size,
            )
            if not selected:
                selected = frontier[:per_level_max_select]
                level_thinking = f"{level_thinking} | level fallback to deterministic order.".strip(" |")

            traversal_thinking_parts.append(f"level{depth}: {level_thinking}")

            next_frontier: list[str] = []
            for node_id in selected:
                node = index[node_id]
                if not node.children:
                    terminal_candidates.append(node_id)
                    continue

                subtree_tokens = _estimate_subtree_tokens(node_id, index)
                if subtree_tokens <= subtree_token_budget or depth >= max_traversal_depth:
                    terminal_candidates.append(node_id)
                else:
                    next_frontier.extend(node.children)

            if not next_frontier:
                break
            frontier = next_frontier

        if not terminal_candidates:
            terminal_candidates = stage1_node_list.copy()

        terminal_candidates = _clean_node_list(terminal_candidates, set(index.keys()))

        final_node_ids, final_thinking = _llm_select_nodes(
            stage_name="final-rerank",
            query=query,
            candidate_ids=terminal_candidates,
            index=index,
            llm_caller=llm_caller,
            preference=preference,
            max_select=top_k,
            chunk_size=candidate_chunk_size,
        )
        if not final_node_ids:
            final_node_ids = terminal_candidates[:top_k]
            final_thinking = f"{final_thinking} | final fallback to terminal candidate order.".strip(" |")

        final_thinking = "\n".join(traversal_thinking_parts + [f"final: {final_thinking}"])

    scored_nodes = []
    for node_id in final_node_ids:
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
    candidate_chunk_size: int = 20,
    max_traversal_depth: int = 4,
    subtree_token_budget: int = 1600,
    per_level_max_select: int = 8,
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
