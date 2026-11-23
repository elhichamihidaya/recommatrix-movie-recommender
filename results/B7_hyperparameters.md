# B7 — Hyperparamètres sélectionnés

Les configurations ont été sélectionnées sur le jeu de validation, avant les expériences finales sur le test. Les grilles complètes sont conservées dans `tuning_als.csv`, `tuning_sgd.csv` et `tuning_bpr.csv` (et leurs résumés).

| Méthode | Configurations testées | Configuration retenue | Critère de validation |
|---|---:|---|---|
| ALS | 24 | `d=10`, `lam=0.05`, `n_iter=20` | `rmse_val` = 0.86064 |
| SGD | 27 | `d=20`, `eta=0.01`, `lam=0.02`, `lr_decay=0.05`, `n_epochs=10`, `patience=3` | `rmse_val` = 0.88168 |
| BPR | 27 | `d=20`, `eta=0.05`, `lam=0.001`, `lr_decay=0.02`, `n_epochs=10`, `relevant_threshold=4.0` | `precision@10` = 0.07648 |

Pour ALS et SGD, la sélection minimise la RMSE de validation. Pour BPR, modèle de classement implicite, elle maximise Precision@10 de validation. Le seuil de pertinence BPR est 4 étoiles.
