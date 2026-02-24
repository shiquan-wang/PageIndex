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
    assert "fallback" in result["thinking"].lower()


def test_search_tree_two_stage_uses_llm_response_node_list():
    with FIXTURE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    call_count = {"n": 0}

    def fake_llm(prompt: str) -> str:
        call_count["n"] += 1
        if "Top-level candidate nodes" in prompt:
            return json.dumps(
                {
                    "thinking": "Sports-related root is relevant.",
                    "node_list": ["0016"],
                }
            )
        return json.dumps(
            {
                "thinking": "Sports segment details are likely relevant.",
                "node_list": ["0017", "0016"],
            }
        )

    result = search_tree(
        query="operating income",
        tree_json=data,
        top_k=8,
        preference="prioritize sports segment",
        llm_caller=fake_llm,
    )
    assert call_count["n"] == 2
    assert result["stage1_node_list"] == ["0016"]
    assert result["node_list"] == ["0017", "0016"]


def test_stage1_chunking_when_top_level_is_large():
    large_tree = {
        "structure": [
            {"node_id": f"r{i:03d}", "title": f"Root {i}", "summary": "", "start_index": i, "end_index": i, "nodes": []}
            for i in range(25)
        ]
    }

    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "Top-level candidate nodes" in prompt:
            # select one id from each chunk via prompt content
            if "r000" in prompt:
                return json.dumps({"thinking": "chunk1", "node_list": ["r001"]})
            if "r010" in prompt:
                return json.dumps({"thinking": "chunk2", "node_list": ["r012"]})
            return json.dumps({"thinking": "chunk3", "node_list": ["r022"]})
        return json.dumps({"thinking": "stage2", "node_list": ["r012", "r022"]})

    result = search_tree(
        query="anything",
        tree_json=large_tree,
        top_k=3,
        llm_caller=fake_llm,
        stage1_max_roots=3,
        stage1_chunk_size=10,
    )

    stage1_prompt_count = sum("Top-level candidate nodes" in p for p in prompts)
    assert stage1_prompt_count == 3
    assert result["stage1_node_list"] == ["r001", "r012", "r022"]
    assert result["node_list"] == ["r012", "r022"]
