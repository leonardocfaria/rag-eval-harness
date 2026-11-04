from rag.evaluate import run_all


def test_run_all_small(tmp_path):
    res = run_all(["json", "textwrap", "heapq", "bisect", "statistics", "shutil", "string"], n_queries=30, out=tmp_path)
    assert set(res["retrievers"]) == {"bm25", "dense", "hybrid-rrf"}
    for r in res["retrievers"].values():
        assert 0 <= r["all"]["recall@1"] <= r["all"]["recall@5"] <= r["all"]["recall@10"] <= 1
    # lexical queries quote the docstring: BM25 must do well on them
    assert res["retrievers"]["bm25"]["lexical"]["recall@5"] > 0.7
    gen = res["generation"]
    assert gen["bm25 top3 mock"]["faithfulness"] == 1.0  # extractive answers are always supported
    assert gen["hybrid-rrf top3 unfaithful-mock"]["faithfulness"] < 0.2
    report = (tmp_path / "report.md").read_text()
    assert "## Ablation: fusion" in report and "| hybrid-rrf | noisy |" in report
