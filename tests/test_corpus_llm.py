from rag.corpus import build_corpus, make_queries
from rag.evaluate import cited_ids, faithfulness
from rag.llm import MockLLM, UnfaithfulMockLLM

MODULES = ["json", "textwrap", "heapq", "bisect", "statistics"]


def test_corpus_and_queries_are_deterministic():
    docs = build_corpus(MODULES)
    assert docs == build_corpus(MODULES) and len(docs) > 20
    assert all(d["text"].startswith(d["id"]) for d in docs)
    q1, q2 = make_queries(docs, 10, seed=1), make_queries(docs, 10, seed=1)
    assert q1 == q2 and {q["type"] for q in q1} == {"lexical", "noisy"}
    for q in q1:  # neither the object name nor any of its parts leaks into the query
        name = docs[q["gold"]]["name"].lower()
        assert not ({name} | set(name.split("_"))) & set(q["query"].split())
        assert len(q["query"].split()) >= 3


def test_mock_llm_is_extractive_and_cites():
    ctx = [("m.a", "m.a: Sort a list. Keep it stable."), ("m.b", "m.b: Parse a JSON document into objects.")]
    out = MockLLM().answer("how do I parse json", ctx)
    assert out == "Parse a JSON document into objects. [m.b]"
    assert cited_ids(out) == ["m.b"]
    assert faithfulness(out, [t for _, t in ctx]) == 1.0


def test_faithfulness_catches_unsupported_answer():
    ctx = [("m.a", "m.a: Sort a list in place.")]
    bad = UnfaithfulMockLLM().answer("sort", ctx)
    assert faithfulness(bad, [t for _, t in ctx]) == 0.0
    assert faithfulness("Sort a list. Also it flies to the moon.", ["Sort a list in place."]) == 0.5
    assert faithfulness("", ["anything"]) == 0.0
