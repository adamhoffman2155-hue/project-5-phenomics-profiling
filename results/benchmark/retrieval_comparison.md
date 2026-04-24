# Benchmark: retrieval-metric comparison

Compares three similarity metrics on a matched-shape profile set
(20 MoAs × 3 compounds = 60 compounds, 241 features). Cosine is the
POC default; euclidean and Spearman are compared as alternatives.

Random baseline is computed from MoA label frequencies (expected
probability of a same-MoA nearest neighbor under a uniform random
match), not a fixed constant.

| metric | k | recall@k | fold over random |
| --- | ---: | ---: | ---: |
| cosine | 1 | 0.317 | 9.34× |
| cosine | 5 | 0.900 | 26.55× |
| cosine | 10 | 0.983 | 29.01× |
| euclidean | 1 | 0.317 | 9.34× |
| euclidean | 5 | 0.717 | 21.14× |
| euclidean | 10 | 0.817 | 24.09× |
| spearman | 1 | 0.317 | 9.34× |
| spearman | 5 | 0.817 | 24.09× |
| spearman | 10 | 0.967 | 28.52× |

Random baseline: 0.034

## Interpretation

Cosine and Spearman give near-identical recall on this profile space,
consistent with profiles being already centered + scaled. Euclidean
lags both. The POC result on real LINCS data (recall@5 = 0.096, 3.0× 
random) is the canonical headline — this benchmark is a relative-
metric comparison, not a replacement for the POC.
