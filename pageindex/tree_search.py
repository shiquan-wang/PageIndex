import json
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


def _build_llm_prompt(query: str, tree_json: dict[str, Any] | list[dict[str, Any]], preference: str | None = None) -> str:
    prompt = f"""
You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {json.dumps(tree_json, ensure_ascii=False)}
"""

    if preference:
        prompt += f"\nExpert Knowledge of relevant sections: {preference}\n"

    prompt += """
Reply in the following JSON format:
{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}
"""
    return prompt.strip()


def _default_tree_search_result(index: dict[str, IndexedNode], top_k: int) -> dict[str, Any]:
    ordered_nodes = sorted(index.values(), key=lambda n: (n.depth, n.node_id))
    node_list = [node.node_id for node in ordered_nodes[:top_k]]
    return {
        "thinking": "LLM caller is not provided, fallback to default top-level traversal result.",
        "node_list": node_list,
    }


def search_tree(
    query: str,
    tree_json: dict[str, Any] | list[dict[str, Any]],
    top_k: int = 8,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """LLM-based tree retrieval.

    If llm_caller is None, a deterministic fallback result is returned for testing.
    llm_caller should accept a single prompt string and return the model text response.
    """
    index = build_tree_index(tree_json)
    if not index:
        return {"thinking": "empty tree", "node_list": [], "scored_nodes": []}

    if llm_caller is None:
        result = _default_tree_search_result(index=index, top_k=top_k)
    else:
        prompt = _build_llm_prompt(query=query, tree_json=tree_json, preference=preference)
        raw_response = llm_caller(prompt)
        parsed = _extract_json_from_text(raw_response)
        node_list = [str(node_id) for node_id in parsed.get("node_list", [])]
        result = {
            "thinking": parsed.get("thinking", ""),
            "node_list": node_list[:top_k],
        }

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
        "thinking": result.get("thinking", ""),
        "node_list": [node["node_id"] for node in scored_nodes],
        "scored_nodes": scored_nodes,
    }


def search_tree_from_json_file(
    query: str,
    json_path: str | Path,
    top_k: int = 8,
    preference: str | None = None,
    llm_caller: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        tree_json = json.load(f)
    return search_tree(
        query=query,
        tree_json=tree_json,
        top_k=top_k,
        preference=preference,
        llm_caller=llm_caller,
    )
