# rag-eval-harness Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** A RAG QA system whose main value is the *evaluation*: retrieval metrics per retriever, answer faithfulness, ablations, all reproducible offline.
- **Corpus:** "slice of Python docs". docs.python.org is blocked (DECISIONS #3), but the same text ships with the interpreter as docstrings. The corpus is the public functions/classes of ~30 stdlib modules (PSF licence), one document per object, built deterministically from the local interpreter. Results are therefore tied to the Python version (3.12 is pinned in CI).
- **Eval set:** no human-labelled QA exists offline, so it is generated with a seed. Each question targets one gold document. Two query types:
  - `lexical`: the docstring's first sentence with the object name removed (easy for BM25);
  - `noisy`: 6 content words sampled from the whole docstring, shuffled, each cut to a 4-letter prefix with p=0.5. These are partial or misspelled keywords, a vocabulary mismatch. (A first version called this "paraphrase" with 15 % truncation; it was easier than `lexical`, so it was renamed and made noisier.)
  The gold answer is the docstring's first sentence.
- **Retrievers:** BM25 (hand-rolled, numpy, k1/b params); dense = LSA embeddings (TF-IDF over char n-grams + TruncatedSVD, cosine), because no pretrained embedding model can be downloaded; hybrid = reciprocal-rank fusion (and a weighted-score variant for the ablation).
- **LLM:** `LLM` protocol `answer(question, contexts) -> str`. `MockLLM` is extractive: it picks the context sentence with the highest overlap and cites `[doc_id]`. `UnfaithfulMockLLM` invents a sentence, to prove the checker catches it.
- **Faithfulness:** deterministic. Each answer sentence (citation stripped) is supported if ≥ 80 % of its content tokens appear in one retrieved context. Score = supported / total.
- **Non-goals:** a real LLM call, rerankers, chunk overlap tuning.

## Files
- `rag/corpus.py`: `build_corpus(modules)`, `make_queries(docs, n, seed)`.
- `rag/retrievers.py`: `tokenize`, `BM25`, `Dense`, `Hybrid` (`rrf` | `weighted`). Each has `search(query, k) -> [(doc_idx, score)]`.
- `rag/llm.py`: `LLM` protocol, `MockLLM`, `UnfaithfulMockLLM`.
- `rag/evaluate.py`: `recall_at_k`, `mrr`, `faithfulness`, `evaluate_retriever`, `run_all()` → `results/report.md` + json.

## Tasks (TDD)
- [x] 1. metrics on hand-made rankings; tokenizer.
- [x] 2. BM25 ranks the exact doc first on a toy corpus; dense/hybrid return valid rankings; RRF math.
- [x] 3. corpus/query determinism; mock LLM cites a retrieved doc; faithfulness 1.0 for extractive, < 0.5 for the unfaithful mock.
- [x] 4. run_all on a small module list in tests; full report committed; README; CI.
