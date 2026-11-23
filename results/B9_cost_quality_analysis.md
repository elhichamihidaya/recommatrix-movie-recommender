# B9 — Analyse du coût computationnel et de la qualité

## Protocole et lecture des résultats

Les mesures ci-dessous proviennent du benchmark MovieLens 1M : 1 000 209 notes au total, 6 040 utilisateurs et 3 706 films. Le split par utilisateur utilise la seed 42 ; le réglage utilise la validation, les entraînements finaux sont faits sur `train_full`, et le jeu de test contient 200 016 notes. Toutes les méthodes finales sont présentes sur cinq seeds (0–4). Les valeurs sont présentées en moyenne ± écart-type. Les temps et la mémoire reflètent cette exécution sur une machine Windows/Anaconda ; ils dépendent du matériel, du système, des bibliothèques et de la charge en arrière-plan.

Pour le ranking, une note de test ≥ 4 est pertinente, les films déjà vus en entraînement sont exclus et la moyenne porte sur les utilisateurs ayant au moins un film pertinent dans le test. BPR et Popularity sont des méthodes de classement : leur RMSE/MAE n'est pas défini. Les métriques ranking des deux ALS de référence sont calculées à partir de modèles réentraînés sur le même `train_full`.

## Résultats observés

| Méthode | RMSE test | P@10 | Temps moyen (s) | Mémoire de crête moyenne (MB) | Cycles configurés |
|---|---:|---:|---:|---:|---:|
| ALS | 0.8582 ± 0.0005 | 0.0212 ± 0.0005 | 27.31 ± 2.89 | 110.2 ± 2.1 | 20 |
| ALS ref. 10 | 0.8607 ± 0.0008 | 0.0215 ± 0.0008 | 7.64 ± 4.73 | 98.8 ± 10.6 | 10 |
| ALS ref. 30 | 0.8576 ± 0.0004 | 0.0209 ± 0.0005 | 21.73 ± 13.77 | 102.0 ± 8.4 | 30 |
| BPR | — | 0.2194 ± 0.0013 | 2.17 ± 0.06 | 33.0 ± 0.5 | 10 |
| Global mean | 1.1158 ± 0.0000 | 0.0106 ± 0.0000 | 0.11 ± 0.02 | 6.2 ± 0.0 | 0 |
| Item mean | 0.9777 ± 0.0000 | 0.0001 ± 0.0000 | 0.10 ± 0.00 | 6.2 ± 0.0 | 0 |
| Popularity | — | 0.1514 ± 0.0000 | 0.11 ± 0.02 | 6.8 ± 0.3 | 0 |
| PSO | 1.1159 ± 0.0000 | 0.0050 ± 0.0007 | 28.52 ± 21.18 | 104.2 ± 27.2 | 299 |
| PSO + ALS | 0.8609 ± 0.0005 | 0.0211 ± 0.0004 | 35.60 ± 25.92 | 97.3 ± 18.2 | 309 |
| SGD | 0.8760 ± 0.0002 | 0.0962 ± 0.0016 | 5.20 ± 0.52 | 197.8 ± 11.8 | 10 |
| User mean | 1.0339 ± 0.0000 | 0.0106 ± 0.0000 | 0.11 ± 0.02 | 7.6 ± 2.7 | 0 |

The fit-time plots use a logarithmic time axis to show both the very short mean/popularity baselines and the longer latent-factor methods. The memory plots compare measured peak RSS above each process baseline. Cycle counts use the model's own unit: SGD/BPR epochs, ALS sweeps, PSO generations, and the sum of PSO generations plus ALS sweeps for the hybrid. Those units describe workload within each algorithm and should not be interpreted as identical operations. The zero-cycle simple baselines are closed-form aggregation methods.

### Theoretical work per iteration

Let Ω be the training ratings, with |Ω| = 800,193, users n_u = 6,040, items n_i = 3,706, latent dimension d, and PSO population N. ALS-WR solves one d-by-d system per user and item: approximately O(|Ω| d² + (n_u+n_i)d³) per full ALS sweep; storage is O(|Ω|+(n_u+n_i)d). This matches its relatively high measured wall time for d=10 and 20 sweeps (27.31 s mean for the tuned Part B ALS).

SGD visits each observed rating and updates two d-dimensional vectors: O(|Ω|d) per epoch, with O(|Ω|+(n_u+n_i)d) storage. Its measured 10-epoch fit averages 5.20 s and 197.8 MB. It has a good rating RMSE (0.8760) and stronger P@10 (0.0962) than the non-personalized rating baselines, while using more measured memory than the other fitted models in this run.

BPR samples positive/negative user-item triplets. For S sampled triplets per epoch, its work is O(Sd) per epoch and its factor/storage requirement is O((n_u+n_i)d), plus sampling/index structures. It is ranking-only by design. At the tuned configuration its mean P@10 is 0.2194, above Popularity at 0.1514, with 2.17 s mean fit time and 33.0 MB peak memory. Thus it is the strongest top-10 ranking method in these measurements, at substantially lower time than ALS or PSO. Its quality should be judged only by ranking metrics.

PSO evaluates a population of candidate factor matrices. A full fitness evaluation costs O(|Ω|d); E evaluations therefore cost O(E|Ω|d). The population, velocities and best-so-far states require O(N(n_u+n_i)d) memory. At E=6,000 and N=20, the pure PSO averages 28.52 s and 104.2 MB, yet its RMSE is 1.1159 and P@10 is 0.0050. It has a high computational cost without competitive predictive or ranking quality on this large search space.

The PSO+ALS hybrid combines the PSO evaluation cost with ten ALS sweeps, as confirmed by the saved per-seed histories. It averages 35.60 s, 97.3 MB and RMSE 0.8609, with P@10 0.0212. ALS alone averages RMSE 0.8582 and P@10 0.0212 at 27.31 s. On these metrics, the PSO stage adds cost without a measurable quality advantage over tuned ALS. The hybrid is far better than PSO alone, showing that ALS refinement accounts for most of its rating and ranking quality.

The global/user/item means and popularity are closed-form baselines with one pass over ratings or counts, O(|Ω|+n_u+n_i) work and small state. They fit in about 0.10–0.11 s and under 8 MB peak memory here. They are useful cost floors. Global mean reaches RMSE 1.1158, while popularity reaches P@10 0.1514; a baseline can therefore be weak on one task and competitive on another.

## Pareto and statistical interpretation

On the RMSE/time plane, the Pareto front contains: ALS ref. 10, ALS ref. 30, Item mean, SGD. On the P@10/time plane, it contains: BPR, Global mean, Item mean, Popularity. A method is on the front if no other measured method is both faster and at least as good on that metric. This is a descriptive comparison of five-seed averages, not a guarantee for every seed. The separate time and memory figures show that the trade-off changes with the selected objective: BPR is particularly effective for ranking, while ALS/SGD are stronger choices for calibrated rating prediction.

Paired t-tests are exploratory because they use only five paired seeds and assume the seed-wise differences are approximately normal:
- Test t apparié als vs pso+als sur `rmse_test` : n=5, t=-19.047, p=4.476e-05.
- Test t apparié als vs als_reference_30it sur `rmse_test` : n=5, t=5.985, p=0.003918.
- Test t apparié bpr vs popularity sur `prec@10` : n=5, t=118.547, p=3.037e-08.
- Test t apparié als vs sgd sur `rmse_test` : n=5, t=-61.256, p=4.254e-07.

The test results do not replace effect sizes or confidence intervals, and a non-significant result at n=5 is not evidence of equivalence. A practical choice follows the product objective: use BPR when top-k ranking is primary; use ALS or SGD for rating prediction; use simple baselines as cheap reference points. The current measurements do not justify pure PSO for the full MovieLens 1M factorization, and do not show a quality gain from adding PSO before ALS. A metaheuristic's extra cost could be justified when it yields a robust improvement on the chosen metric, solves constraints unavailable to the baseline optimizer, or works better on a smaller search domain; none of those advantages is established by this benchmark.

## Generated artifacts

- `results/figures/B9_quality_vs_time.png`
- `results/figures/B9_quality_vs_memory.png`
- `results/figures/B9_quality_vs_cycles.png`
- `results/figures/B9_pareto_fronts.png`
- `results/B9_cost_quality_summary.csv`
- `results/B9_paired_tests.csv`

The cycle counts for PSO+ALS and the recorded configuration metadata were reconciled against the saved convergence histories. The pre-correction input CSV is retained as `results/metaheuristics_before_b9_correction.csv`.
