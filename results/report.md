# RAG evaluation report

1011 documents, 600 queries (half lexical, half noisy), runtime 141 s.

## Retrievers

| retriever | query type | recall@1 | recall@5 | recall@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|---|
| bm25 | all | 0.822 | 0.973 | 0.992 | 0.887 | 0.128 |
| bm25 | lexical | 0.857 | 0.977 | 0.997 | 0.910 | 0.128 |
| bm25 | noisy | 0.787 | 0.970 | 0.987 | 0.865 | 0.128 |
| dense | all | 0.728 | 0.928 | 0.968 | 0.815 | 15.896 |
| dense | lexical | 0.717 | 0.917 | 0.957 | 0.805 | 15.896 |
| dense | noisy | 0.740 | 0.940 | 0.980 | 0.824 | 15.896 |
| hybrid-rrf | all | 0.793 | 0.952 | 0.993 | 0.868 | 14.132 |
| hybrid-rrf | lexical | 0.793 | 0.947 | 0.990 | 0.867 | 14.132 |
| hybrid-rrf | noisy | 0.793 | 0.957 | 0.997 | 0.870 | 14.132 |


## Ablation: fusion

| variant | recall@1 | recall@5 | recall@10 | MRR@10 | noisy MRR@10 |
|---|---|---|---|---|---|
| rrf k=60 | 0.793 | 0.952 | 0.993 | 0.868 | 0.870 |
| rrf k=10 | 0.792 | 0.963 | 0.993 | 0.870 | 0.871 |
| weighted a=0.3 | 0.807 | 0.960 | 0.987 | 0.874 | 0.879 |
| weighted a=0.5 | 0.832 | 0.972 | 0.997 | 0.895 | 0.899 |
| weighted a=0.7 | 0.840 | 0.977 | 0.995 | 0.899 | 0.893 |


## Ablation: bm25

| variant | recall@1 | recall@5 | recall@10 | MRR@10 | noisy MRR@10 |
|---|---|---|---|---|---|
| k1=0.9 b=0.75 | 0.835 | 0.978 | 0.992 | 0.898 | 0.879 |
| k1=1.5 b=0.75 | 0.822 | 0.973 | 0.992 | 0.887 | 0.865 |
| k1=2.0 b=0.75 | 0.810 | 0.970 | 0.990 | 0.879 | 0.853 |
| k1=1.5 b=0.3 | 0.830 | 0.975 | 0.990 | 0.894 | 0.856 |
| k1=1.5 b=1.0 | 0.797 | 0.958 | 0.990 | 0.868 | 0.843 |


## Ablation: dense

| variant | recall@1 | recall@5 | recall@10 | MRR@10 | noisy MRR@10 |
|---|---|---|---|---|---|
| char_wb dim=64 | 0.542 | 0.818 | 0.903 | 0.661 | 0.634 |
| char_wb dim=256 | 0.728 | 0.928 | 0.968 | 0.815 | 0.824 |
| word dim=64 | 0.468 | 0.767 | 0.848 | 0.591 | 0.521 |
| word dim=256 | 0.648 | 0.895 | 0.958 | 0.756 | 0.712 |


## End to end (retriever + LLM)

| setup | citation accuracy | faithfulness | answer token-F1 |
|---|---|---|---|
| bm25 top1 mock | 0.818 | 1.000 | 0.712 |
| bm25 top3 mock | 0.827 | 1.000 | 0.723 |
| bm25 top5 mock | 0.830 | 1.000 | 0.725 |
| dense top1 mock | 0.725 | 1.000 | 0.673 |
| dense top3 mock | 0.800 | 1.000 | 0.712 |
| dense top5 mock | 0.818 | 1.000 | 0.721 |
| hybrid-rrf top1 mock | 0.790 | 1.000 | 0.700 |
| hybrid-rrf top3 mock | 0.825 | 1.000 | 0.722 |
| hybrid-rrf top5 mock | 0.823 | 1.000 | 0.724 |
| hybrid-rrf top3 unfaithful-mock | 0.793 | 0.000 | 0.008 |
