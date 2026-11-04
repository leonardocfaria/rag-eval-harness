import numpy as np
import pytest

from rag.evaluate import mrr, recall_at_k
from rag.retrievers import BM25, Dense, Hybrid, tokenize

DOCS = [
    "Return the square root of x.",
    "Split the string at the separator and return a list of words.",
    "Open a file and return a stream object for reading or writing.",
    "Sort the list in place using a stable algorithm.",
    "Return the current time in seconds since the epoch.",
    "Join path components into a single filesystem path string.",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("Return THE Square-root of x2") == ["square", "root", "x2"]


def test_metrics():
    assert recall_at_k([3, 1, 2], gold=1, k=1) == 0 and recall_at_k([3, 1, 2], gold=1, k=2) == 1
    assert mrr([3, 1, 2], gold=1) == 0.5 and mrr([3, 1, 2], gold=9) == 0


def test_bm25_finds_exact_doc_and_idf_favours_rare_terms():
    bm = BM25(DOCS)
    assert bm.search("square root", 1) == [0]
    assert bm.search("filesystem path", 1) == [5]
    assert bm.idf["square"] > bm.idf["string"]  # 1 doc vs 2 docs
    assert np.all(bm.scores("zzz unknown") == 0)


@pytest.mark.parametrize("mode", ["rrf", "weighted"])
def test_dense_and_hybrid_rank_all_docs(mode):
    dense = Dense(DOCS * 3, dim=8)  # repeat so min_df=2 keeps n-grams
    ranked = dense.search("square roots", 18)
    assert sorted(ranked) == list(range(18))
    h = Hybrid(BM25(DOCS * 3), dense, mode=mode)
    assert h.search("square root", 1)[0] % 6 == 0


def test_rrf_fusion_math():
    class Fixed:
        def __init__(self, s):
            self.s = np.array(s, dtype=float)

        def scores(self, q):
            return self.s

    h = Hybrid(Fixed([3, 2, 1]), Fixed([1, 3, 2]), k_rrf=60)
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert np.allclose(h.scores("q"), expected)
