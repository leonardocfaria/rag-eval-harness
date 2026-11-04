"""Eval harness: recall@k, MRR, deterministic faithfulness, ablation tables. python -m rag.evaluate"""
import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

from .corpus import DEFAULT_MODULES, build_corpus, make_queries
from .llm import MockLLM, UnfaithfulMockLLM, sentences
from .retrievers import BM25, Dense, Hybrid, tokenize

KS = (1, 5, 10)
_CITE = re.compile(r"\[([^\]]+)\]")


def recall_at_k(ranked: list[int], gold: int, k: int) -> float:
    return float(gold in ranked[:k])


def mrr(ranked: list[int], gold: int, k: int = 10) -> float:
    return 1 / (ranked.index(gold) + 1) if gold in ranked[:k] else 0.0


def cited_ids(answer: str) -> list[str]:
    return _CITE.findall(answer)


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.8) -> float:
    """Share of answer sentences whose content tokens are >= threshold covered by a single context."""
    sents = [set(tokenize(s)) for s in sentences(_CITE.sub("", answer))]
    sents = [s for s in sents if s]
    if not sents:
        return 0.0
    ctx = [set(tokenize(c)) for c in contexts]
    return sum(any(len(s & c) / len(s) >= threshold for c in ctx) for s in sents) / len(sents)


def token_f1(pred: str, gold: str) -> float:
    p, g = tokenize(_CITE.sub("", pred)), tokenize(gold)
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    return 0.0 if not common else 2 * common / (len(p) + len(g))


def retrieval_metrics(retriever, queries) -> dict:
    t = time.perf_counter()
    ranked = [retriever.search(q["query"], max(KS)) for q in queries]
    ms = (time.perf_counter() - t) * 1000 / len(queries)
    out = {}
    for qtype in ("all", "lexical", "noisy"):
        idx = [i for i, q in enumerate(queries) if qtype == "all" or q["type"] == qtype]
        row = {f"recall@{k}": mean(recall_at_k(ranked[i], queries[i]["gold"], k) for i in idx) for k in KS}
        row["mrr@10"] = mean(mrr(ranked[i], queries[i]["gold"]) for i in idx)
        out[qtype] = row
    out["ms_per_query"] = ms
    return out


def generation_metrics(retriever, llm, queries, docs, top_k: int) -> dict:
    rows = []
    for q in queries:
        ctx = [(docs[i]["id"], docs[i]["text"]) for i in retriever.search(q["query"], top_k)]
        ans = llm.answer(q["query"], ctx)
        rows.append((docs[q["gold"]]["id"] in cited_ids(ans), faithfulness(ans, [t for _, t in ctx]),
                     token_f1(ans, q["answer"])))
    return {"citation_acc": mean(r[0] for r in rows), "faithfulness": mean(r[1] for r in rows),
            "answer_f1": mean(r[2] for r in rows)}


def _table(header: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(map(fmt, r)) + " |" for r in rows]) + "\n"


def run_all(modules=DEFAULT_MODULES, n_queries: int = 300, seed: int = 0, out: str | Path = "results") -> dict:
    t0 = time.perf_counter()
    docs = build_corpus(modules)
    texts = [d["text"] for d in docs]
    queries = make_queries(docs, n_queries, seed)
    bm25, dense = BM25(texts), Dense(texts)
    main = {"bm25": bm25, "dense": dense, "hybrid-rrf": Hybrid(bm25, dense)}
    res = {"n_docs": len(docs), "n_queries": len(queries), "retrievers": {}, "ablations": {}}
    for name, r in main.items():
        res["retrievers"][name] = retrieval_metrics(r, queries)

    variants = {
        "fusion": {"rrf k=60": Hybrid(bm25, dense), "rrf k=10": Hybrid(bm25, dense, k_rrf=10),
                   **{f"weighted a={a}": Hybrid(bm25, dense, "weighted", alpha=a) for a in (0.3, 0.5, 0.7)}},
        "bm25": {f"k1={k1} b={b}": BM25(texts, k1, b) for k1, b in ((0.9, 0.75), (1.5, 0.75), (2.0, 0.75), (1.5, 0.3), (1.5, 1.0))},
        "dense": {f"{an} dim={d}": Dense(texts, d, an) for an in ("char_wb", "word") for d in (64, 256)},
    }
    for group, rs in variants.items():
        res["ablations"][group] = {name: retrieval_metrics(r, queries) for name, r in rs.items()}

    gen = {}
    for name, r in main.items():
        for k in (1, 3, 5):
            gen[f"{name} top{k} mock"] = generation_metrics(r, MockLLM(), queries, docs, k)
    gen["hybrid-rrf top3 unfaithful-mock"] = generation_metrics(main["hybrid-rrf"], UnfaithfulMockLLM(), queries, docs, 3)
    res["generation"] = gen
    res["runtime_s"] = time.perf_counter() - t0
    write_report(res, Path(out))
    return res


def write_report(res: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    m = lambda r: [r["recall@1"], r["recall@5"], r["recall@10"], r["mrr@10"]]
    hdr = ["recall@1", "recall@5", "recall@10", "MRR@10"]
    parts = [f"# RAG evaluation report\n\n{res['n_docs']} documents, {res['n_queries']} queries "
             f"(half lexical, half noisy), runtime {res['runtime_s']:.0f} s.\n",
             "## Retrievers\n", _table(["retriever", "query type", *hdr, "ms/query"],
                                       [[n, t, *m(r[t]), r["ms_per_query"]] for n, r in res["retrievers"].items()
                                        for t in ("all", "lexical", "noisy")])]
    for group, rs in res["ablations"].items():
        parts += [f"\n## Ablation: {group}\n", _table(["variant", *hdr, "noisy MRR@10"],
                                                      [[n, *m(r["all"]), r["noisy"]["mrr@10"]] for n, r in rs.items()])]
    parts += ["\n## End to end (retriever + LLM)\n",
              _table(["setup", "citation accuracy", "faithfulness", "answer token-F1"],
                     [[n, g["citation_acc"], g["faithfulness"], g["answer_f1"]] for n, g in res["generation"].items()])]
    (out / "report.md").write_text("\n".join(parts))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    r = run_all(n_queries=a.queries, seed=a.seed, out=a.out)
    print(json.dumps({n: v["all"] for n, v in r["retrievers"].items()}, indent=2))
