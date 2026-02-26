import json
from pathlib import Path

from pageindex.tree_search import search_tree, search_tree_from_json_file


FIXTURE = Path("tests/results/q1-fy25-earnings_structure.json")


def test_search_tree_from_json_file_default_fallback_returns_nodes():
    result = search_tree_from_json_file(
        query="sports segment operating income",
        json_path=FIXTURE,
        top_k=5,
    )
    assert result["node_list"]
    assert len(result["node_list"]) <= 5
    assert "回退" in result["thinking"]


def test_recursive_tree_search_returns_all_when_top_k_none():
    tree = {
        "structure": [
            {
                "node_id": "r1",
                "title": "Root 1",
                "summary": "",
                "start_index": 1,
                "end_index": 1,
                "nodes": [
                    {
                        "node_id": "c1",
                        "title": "Child 1",
                        "summary": "",
                        "start_index": 2,
                        "end_index": 2,
                        "nodes": [
                            {"node_id": "l1", "title": "Leaf 1", "summary": "", "start_index": 3, "end_index": 3, "nodes": []},
                            {"node_id": "l2", "title": "Leaf 2", "summary": "", "start_index": 4, "end_index": 4, "nodes": []},
                        ],
                    }
                ],
            }
        ]
    }

    def fake_llm(prompt: str) -> str:
        if "stage-1 top-level prefilter" in prompt:
            return json.dumps({"thinking": "pick root", "node_list": ["r1"]})
        if "recursive-level-1" in prompt:
            return json.dumps({"thinking": "descend", "node_list": ["r1"]})
        if "recursive-level-2" in prompt:
            return json.dumps({"thinking": "pick child", "node_list": ["c1"]})
        if "recursive-level-3" in prompt:
            return json.dumps({"thinking": "pick leaves", "node_list": ["l1", "l2"]})
        return json.dumps({"thinking": "final", "node_list": ["l2", "l1"]})

    result = search_tree(
        query="find details",
        tree_json=tree,
        top_k=None,
        llm_caller=fake_llm,
        max_traversal_depth=4,
        subtree_token_budget=1,
        per_level_max_select=None,
    )

    assert result["node_list"] == ["l2", "l1"]


def test_recursive_candidate_chunking_when_frontier_is_large():
    tree = {
        "structure": [
            {
                "node_id": "r1",
                "title": "Root",
                "summary": "",
                "start_index": 1,
                "end_index": 1,
                "nodes": [
                    {"node_id": f"c{i:03d}", "title": f"Child {i}", "summary": "", "start_index": i, "end_index": i, "nodes": []}
                    for i in range(25)
                ],
            }
        ]
    }

    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "stage-1 top-level prefilter" in prompt:
            return json.dumps({"thinking": "r", "node_list": ["r1"]})
        if "recursive-level-1" in prompt:
            return json.dumps({"thinking": "go root", "node_list": ["r1"]})
        if "recursive-level-2" in prompt:
            if "c000" in prompt:
                return json.dumps({"thinking": "chunk1", "node_list": ["c001"]})
            if "c010" in prompt:
                return json.dumps({"thinking": "chunk2", "node_list": ["c012"]})
            return json.dumps({"thinking": "chunk3", "node_list": ["c022"]})
        return json.dumps({"thinking": "final", "node_list": ["c012", "c022"]})

    result = search_tree(
        query="find relevant leaf",
        tree_json=tree,
        llm_caller=fake_llm,
        top_k=2,
        candidate_chunk_size=10,
        per_level_max_select=3,
        subtree_token_budget=1,
    )

    recursive_level_2_prompt_count = sum("recursive-level-2" in p for p in prompts)
    assert recursive_level_2_prompt_count == 3
    assert result["node_list"] == ["c012", "c022"]
