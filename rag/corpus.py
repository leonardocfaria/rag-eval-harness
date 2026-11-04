"""Corpus = docstrings of stdlib modules (ships with Python, PSF licence); seeded synthetic QA set."""
import importlib
import inspect
import random
import re
from collections import Counter

from .retrievers import tokenize

DEFAULT_MODULES = """abc argparse array asyncio base64 bisect calendar collections contextlib copy csv dataclasses datetime
decimal difflib email.message enum fnmatch fractions functools glob gzip hashlib heapq hmac html http.client inspect io
ipaddress itertools json logging math operator os os.path pathlib pickle platform pprint queue random re secrets shlex
shutil socket sqlite3 statistics string struct subprocess tarfile tempfile textwrap threading time timeit traceback
typing unittest urllib.parse uuid warnings weakref zipfile zlib""".split()
MAX_WORDS = 200  # one document per object, truncated: a chunk


def build_corpus(modules=DEFAULT_MODULES) -> list[dict]:
    """[{id, name, text, first}] for public functions/classes with a docstring of >= 8 words. Deterministic."""
    docs, seen = [], set()
    for m in modules:
        mod = importlib.import_module(m)
        for name in sorted(dir(mod)):
            obj = getattr(mod, name)
            if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.isclass(obj)):
                continue
            doc = inspect.getdoc(obj)
            if not doc or len(doc.split()) < 8 or doc in seen:  # skip re-exports and shared docstrings
                continue
            seen.add(doc)
            body = " ".join(doc.split()[:MAX_WORDS])
            docs.append({"id": f"{m}.{name}", "name": name, "text": f"{m}.{name}: {body}", "first": first_sentence(body)})
    return docs


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]


def _name_tokens(d: dict) -> set[str]:
    return set(tokenize(d["name"].replace("_", " "))) | {d["name"].lower()}


def make_queries(docs: list[dict], n: int = 300, seed: int = 0) -> list[dict]:
    """Two queries per sampled doc: 'lexical' (first sentence minus the name) and 'noisy'
    (6 shuffled content words from the whole docstring; words over 5 letters cut to a 4-letter prefix with p=0.5:
    partial/misspelled keywords, a vocabulary mismatch exact-term matching cannot bridge)."""
    rng = random.Random(seed)
    firsts = Counter(d["first"] for d in docs)
    lexical_q = [" ".join(t for t in tokenize(d["first"]) if t not in _name_tokens(d)) for d in docs]
    # unambiguous gold only: unique first sentence, and >= 4 tokens left once the name is removed
    candidates = [i for i, d in enumerate(docs) if firsts[d["first"]] == 1 and len(lexical_q[i].split()) >= 4]
    queries = []
    for i in rng.sample(candidates, min(n, len(candidates))):
        d = docs[i]
        name_toks = _name_tokens(d)
        lexical = lexical_q[i]
        words = sorted({t for t in tokenize(d["text"].split(": ", 1)[1]) if len(t) > 2 and t not in name_toks})
        words = rng.sample(words, min(6, len(words)))
        words = [w[:4] if len(w) > 5 and rng.random() < 0.5 else w for w in words]
        for qtype, q in (("lexical", lexical), ("noisy", " ".join(words))):
            queries.append({"query": q, "gold": i, "type": qtype, "answer": d["first"]})
    return queries
