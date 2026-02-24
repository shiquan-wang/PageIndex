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


def test_search_tree_uses_llm_response_node_list():
    with FIXTURE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def fake_llm(prompt: str) -> str:
        assert "Query: operating income" in prompt
        return json.dumps(
            {
                "thinking": "Sports segment should be relevant.",
                "node_list": ["0016", "0017"],
            }
        )

    result = search_tree(
        query="operating income",
        tree_json=data,
        top_k=8,
        preference="prioritize sports segment",
        llm_caller=fake_llm,
    )
    assert result["node_list"] == ["0016", "0017"]
