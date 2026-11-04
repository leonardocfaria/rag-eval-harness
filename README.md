# rag-eval-harness

Retrieval-augmented QA over Python standard-library documentation. It has three retrievers (BM25, dense, hybrid), a pluggable LLM with offline mocks, and an evaluation harness: recall@k, MRR, a deterministic faithfulness check and ablation tables. Everything runs offline and is reproducible from a seed.

## Setup
```bash
cd ml/rag-eval-harness
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q              # 10 tests, ~4 s
python -m rag.evaluate           # full eval + ablations, ~2.5 min on 4 vCPUs → results/report.md, results/results.json
python -m rag.evaluate --queries 100 --seed 1 --out /tmp/rag
```
Use it as a library:
```python
from rag.corpus import build_corpus
from rag.retrievers import BM25, Dense, Hybrid
from rag.llm import MockLLM
docs = build_corpus(); texts = [d["text"] for d in docs]
r = Hybrid(BM25(texts), Dense(texts), mode="weighted", alpha=0.5)
hits = r.search("parse json string into python object", k=3)
print(MockLLM().answer("parse json", [(docs[i]["id"], docs[i]["text"]) for i in hits]))
```
To plug in a real LLM, write any object with `answer(question, contexts: list[(doc_id, text)]) -> str` that cites sources as `[doc_id]`.

## Architecture
```
stdlib modules ─► corpus.build_corpus (1 doc per public function/class docstring, ≤ 200 words)
                       │
       corpus.make_queries (seeded): lexical  = first sentence minus the object name
                                     noisy    = 6 shuffled docstring words, half cut to a 4-letter prefix
                       │
 retrievers: BM25 (hand-rolled) │ Dense (TF-IDF char 3-5-grams ► SVD 256 ► cosine) │ Hybrid (RRF or weighted min-max)
                       │
 evaluate: recall@1/5/10, MRR@10 per query type ─► LLM.answer(top-k contexts) ─► citation accuracy,
           faithfulness (each answer sentence ≥ 80 % token-covered by one context), answer token-F1
```
Design choices:
- **Corpus.** Docs and Wikipedia downloads are blocked here (see `.overnight/DECISIONS.md`). The corpus is therefore the stdlib docstrings that ship with the interpreter: 1011 documents from 68 modules, under the PSF licence. Results are tied to the Python version, and CI pins 3.12.
- **Eval set.** No labelled QA set is available offline, so queries are generated with a seed and each has one gold document. **Limitation:** these are known-item queries, not real user questions.
- **Dense retriever.** No pretrained sentence encoder can be downloaded, so the dense retriever is LSA (latent semantic analysis). It is a real dense-vector retriever but weaker than a neural one. To swap one in, replace `Dense.embed`.
- **Faithfulness.** The check is deterministic and needs no LLM judge. It is validated both ways: the extractive `MockLLM` scores 1.0, and `UnfaithfulMockLLM`, which invents its claim, scores 0.0.

## Results (seed 0, 600 queries, Python 3.12)
| retriever | queries | recall@1 | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | lexical | **0.857** | **0.977** | 0.997 | **0.910** |
| BM25 | noisy | 0.787 | 0.970 | 0.987 | 0.865 |
| dense (LSA) | lexical | 0.717 | 0.917 | 0.957 | 0.805 |
| dense (LSA) | noisy | 0.740 | 0.940 | 0.980 | 0.824 |
| hybrid RRF | lexical | 0.793 | 0.947 | 0.990 | 0.867 |
| hybrid RRF | noisy | 0.793 | 0.957 | **0.997** | 0.870 |
| hybrid weighted α=0.5 | noisy | 0.833 | 0.980 | **1.000** | **0.899** |

Ablations (full tables in [results/report.md](results/report.md)):
- **Fusion.** Weighted min-max fusion beats RRF here: MRR 0.895 at α=0.5 vs 0.868. RRF treats both ranked lists equally, while a weighted score lets the stronger BM25 dominate on easy queries. On noisy queries, weighted α=0.5 beats both of its components (0.899 vs 0.865 and 0.824).
- **BM25.** Lower term saturation (k1=0.9) and lower length normalisation (b=0.3) help slightly: MRR 0.898 and 0.894, vs 0.887 at the defaults. Docstrings vary a lot in length, and short ones are often the precise answer.
- **Dense.** Dimensions matter most (64 → 256: MRR 0.66 → 0.81). Char n-grams beat word n-grams because they tolerate truncated words.
- **End to end.** Citation accuracy follows recall@k: BM25 top-3 cites the gold doc 83 % of the time. More context (top-1 → top-5) helps dense the most (0.73 → 0.82). Faithfulness is 1.0 for the extractive mock and 0.0 for the unfaithful one, so the checker separates them.

**Debugging note.** The first version of the hard query type, called "paraphrase", scored *higher* than lexical queries. Sampling words from the whole docstring made them distinctive keyword queries, not paraphrases. It was renamed `noisy` and made harder (half the words truncated), which produced the expected BM25 drop and dense robustness.

## Not done (time box)
A neural encoder or cross-encoder reranker, human-written questions, an LLM-as-judge, chunking experiments (each docstring is one chunk).
