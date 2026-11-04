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
python -m pytest -q              # 11 tests, ~4 s
python -m rag.evaluate           # full eval + ablations, ~2 min on 4 vCPUs → results/report.md, results/results.json
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

## Results (seed 0, 600 queries over 300 unambiguous gold docs, Python 3.12)
| retriever | queries | recall@1 | recall@5 | recall@10 | MRR@10 | MRR 95 % CI |
|---|---|---|---|---|---|---|
| BM25 | lexical | **0.873** | **0.977** | **1.000** | **0.920** | 0.895-0.943 |
| BM25 | noisy | 0.793 | 0.940 | 0.970 | 0.859 | 0.824-0.894 |
| dense (LSA) | lexical | 0.737 | 0.943 | 0.990 | 0.834 | 0.801-0.866 |
| dense (LSA) | noisy | 0.750 | **0.960** | **0.997** | 0.838 | 0.804-0.870 |
| hybrid RRF | lexical | 0.820 | 0.963 | 0.997 | 0.889 | 0.861-0.915 |
| hybrid RRF | noisy | **0.797** | **0.960** | 0.987 | 0.867 | 0.833-0.899 |
| hybrid weighted α=0.3 | all | 0.827 | 0.967 | 0.997 | 0.890 | 0.869-0.909 |
| hybrid weighted α=0.3 | noisy | | | | **0.881** | |

The CIs are 95 % percentile bootstraps over queries. With 300 queries per type, an MRR gap under about 0.03 is within noise. Read the findings with that in mind (full tables in [results/report.md](results/report.md)):
- **Significant: BM25 beats LSA dense on lexical queries** (0.920 vs 0.834, CIs don't overlap). Those queries reuse the docstring's own words, which exact term matching handles best.
- **Noisy queries shift the balance.** BM25 drops (0.920 → 0.859) while dense is unaffected (0.834 → 0.838), because its char n-grams match truncated words. Dense recall@10 on noisy queries (0.997) is the best of any retriever.
- **Hybrid fusion is the robust default.** It is never the worst and is close to the best on both query types. On noisy queries, weighted fusion (0.881) and RRF (0.867) are ahead of both components, but within the CIs. Weighted vs RRF and the α sweep are also within noise.
- **BM25 parameters** (k1, b) move MRR by ≤ 0.02, within noise. The defaults are fine.
- **Significant: dense needs capacity and char n-grams.** 64 → 256 dimensions takes MRR from 0.710 to 0.836, and char beats word n-grams (0.836 vs 0.789). Both gaps are outside the CIs.
- **End to end.** Citation accuracy follows recall@k (BM25 top-3: 0.837). Extra context helps dense the most (top-1 → top-5: 0.740 → 0.847). **Limits of these numbers:**
  - The extractive `MockLLM` copies a context sentence, so its faithfulness is 1.0 by construction, even when the retrieved doc is wrong.
  - Answer-F1 mostly re-measures retrieval.
  - The unfaithful mock's citation accuracy is just recall@1, because it always cites the first context.
  - The checker is validated only at the extremes (1.0 extractive, 0.0 invented). Its real job is scoring a real LLM plugged in through the `LLM` protocol.

Eval-set hygiene:
- Gold documents with a duplicated first sentence (e.g. `bisect` pairs, `os.exec*` families) are excluded.
- A query must keep ≥ 4 tokens after the object name is removed.
- RRF gives no credit to documents a retriever scores 0.
- ms/query is a single timing pass on a shared VM, so treat it as indicative.

**Debugging note.** The first version of the hard query type, called "paraphrase", scored *higher* than lexical queries. Sampling words from the whole docstring made them distinctive keyword queries, not paraphrases. It was renamed `noisy` and made harder (half the words truncated), which produced the expected BM25 drop and dense robustness.

## Not done (time box)
A neural encoder or cross-encoder reranker, human-written questions, an LLM-as-judge, chunking experiments (each docstring is one chunk).
