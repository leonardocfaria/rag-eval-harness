# rag-eval-harness

## Goal
Build "RAG system with a rigorous evaluation harness". Build a retrieval-augmented QA system over a public corpus (for example a slice of Wikipedia or Python docs) with BM25, dense and hybrid retrieval, then an evaluation harness: recall@k, MRR, answer faithfulness via a deterministic checker or a local mock judge, and ablation tables. Pluggable LLM with an offline mock for tests. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [ ] three retrievers implemented
- [ ] eval harness outputs recall@k/MRR tables for each
- [ ] ablation report committed
- [ ] Tests pass with one command; CI workflow runs them
- [ ] README with setup, usage, architecture and results
- [ ] Self code review done
- [ ] Draft PR open

