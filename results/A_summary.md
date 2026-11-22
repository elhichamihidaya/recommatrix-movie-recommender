# Benchmark Summary (MovieLens 1M)

Evaluated on Test Set (200,016 ratings, 6,040 users, 3,706 items) across 3 seeds (0, 1, 2) with latent dimension $d = 10$.

| Method | Seeds | Test RMSE | Test MAE | Fit Time (s) | Peak Mem (MB) | Evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| pso | 3 | 1.1159 ± 0.0000 | 0.9329 ± 0.0000 | 13.09 ± 0.39 | 102.3 ± 30.7 | 6000 |
| pso+als | 3 | 0.8610 ± 0.0005 | 0.6758 ± 0.0005 | 16.71 ± 0.60 | 95.6 ± 24.6 | 6000 |
| als_reference_10it | 3 | 0.8604 ± 0.0010 | 0.6754 ± 0.0009 | 4.21 ± 0.74 | 97.0 ± 14.3 | 0 |
| als_reference_30it | 3 | 0.8575 ± 0.0002 | 0.6728 ± 0.0002 | 11.72 ± 1.93 | 96.5 ± 5.3 | 0 |

*Note: Precision@k and Recall@k columns are computed from the saved factor matrices in `results/models/`.*
