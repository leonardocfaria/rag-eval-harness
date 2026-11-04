"""BM25 (hand-rolled), dense (LSA over char n-grams) and hybrid (RRF or weighted score fusion)."""
import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("""a an and are as at be by can for from has if in into is it its not of on or that the this to
was were will with which when where who return returns returned none true false""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


class Retriever:
    def scores(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[int]:
        s = self.scores(query)
        return np.argsort(-s, kind="stable")[:k].tolist()


class BM25(Retriever):
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.n = len(docs)
        toks = [tokenize(d) for d in docs]
        self.dl = np.array([len(t) for t in toks], dtype=float)
        self.avgdl = self.dl.mean()
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)  # term -> [(doc, tf)]
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query):
        s = np.zeros(self.n)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idx, tf = map(np.array, zip(*self.postings[term]))
            norm = self.k1 * (1 - self.b + self.b * self.dl[idx] / self.avgdl)
            s[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return s


class Dense(Retriever):
    """LSA embeddings: TF-IDF over char n-grams, reduced with SVD, cosine similarity.
    shortcut: no pretrained sentence encoder (downloads blocked); swap in one by replacing embed()."""

    def __init__(self, docs: list[str], dim: int = 256, analyzer: str = "char_wb", seed: int = 0):
        ngrams = (3, 5) if analyzer == "char_wb" else (1, 2)
        self.vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True, min_df=2)
        X = self.vec.fit_transform(docs)
        self.svd = TruncatedSVD(min(dim, X.shape[1] - 1), random_state=seed)
        self.emb = normalize(self.svd.fit_transform(X))

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vec.transform(texts)))

    def scores(self, query):
        return self.emb @ self.embed([query])[0]


class Hybrid(Retriever):
    """mode='rrf': sum of 1/(k_rrf + rank) over retrievers; mode='weighted': alpha*minmax(a) + (1-alpha)*minmax(b)."""

    def __init__(self, a: Retriever, b: Retriever, mode: str = "rrf", k_rrf: int = 60, alpha: float = 0.5, depth: int = 100):
        self.a, self.b, self.mode, self.k_rrf, self.alpha, self.depth = a, b, mode, k_rrf, alpha, depth

    def scores(self, query):
        sa, sb = self.a.scores(query), self.b.scores(query)
        if self.mode == "weighted":
            mm = lambda s: (s - s.min()) / (np.ptp(s) or 1.0)
            return self.alpha * mm(sa) + (1 - self.alpha) * mm(sb)
        fused = np.zeros_like(sa)
        for s in (sa, sb):
            top = np.argsort(-s, kind="stable")[: self.depth]
            top = top[s[top] > 0]  # docs with no match get no fusion credit
            fused[top] += 1.0 / (self.k_rrf + np.arange(1, len(top) + 1))
        return fused
