# Final Benchmark Summary — MovieLens 1M

Results use the deterministic per-user split (seed 42), final training on train_full, and evaluation on the held-out test set. Each method has five runs (seeds 0–4). Ranking relevance is a test rating >= 4; training-seen movies are excluded from recommendations.

| Method | Seeds | Test RMSE | Test MAE | P@10 | R@10 | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| ALS | 5 | 0.8582 ± 0.0005 | 0.6735 ± 0.0004 | 0.0212 ± 0.0005 | 0.0102 ± 0.0002 | 27.31 ± 2.89 |
| ALS reference (10 iter) | 5 | 0.8607 ± 0.0008 | 0.6757 ± 0.0007 | 0.0215 ± 0.0008 | 0.0103 ± 0.0006 | 7.64 ± 4.73 |
| ALS reference (30 iter) | 5 | 0.8576 ± 0.0004 | 0.6729 ± 0.0003 | 0.0209 ± 0.0005 | 0.0101 ± 0.0003 | 21.73 ± 13.77 |
| BPR | 5 | — | — | 0.2194 ± 0.0013 | 0.1362 ± 0.0015 | 2.17 ± 0.06 |
| Global mean | 5 | 1.1158 ± 0.0000 | 0.9329 ± 0.0000 | 0.0106 ± 0.0000 | 0.0060 ± 0.0000 | 0.11 ± 0.02 |
| Item mean | 5 | 0.9777 ± 0.0000 | 0.7811 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.10 ± 0.00 |
| POPULARITY | 5 | — | — | 0.1514 ± 0.0000 | 0.0914 ± 0.0000 | 0.11 ± 0.02 |
| PSO | 5 | 1.1159 ± 0.0000 | 0.9329 ± 0.0000 | 0.0050 ± 0.0007 | 0.0023 ± 0.0004 | 28.52 ± 21.18 |
| PSO + ALS | 5 | 0.8609 ± 0.0005 | 0.6758 ± 0.0005 | 0.0211 ± 0.0004 | 0.0102 ± 0.0003 | 35.60 ± 25.92 |
| SGD | 5 | 0.8760 ± 0.0002 | 0.6901 ± 0.0002 | 0.0962 ± 0.0016 | 0.0554 ± 0.0014 | 5.20 ± 0.52 |
| User mean | 5 | 1.0339 ± 0.0000 | 0.8283 ± 0.0000 | 0.0106 ± 0.0000 | 0.0060 ± 0.0000 | 0.11 ± 0.02 |

The ranking metrics are averaged over users with at least one relevant test movie. The ALS reference ranking values were computed by fitting the same reference models on train_full for each seed.

See `benchmark_all.csv` for per-seed results and `figures/benchmark_comparison.png` for the comparison plot.
