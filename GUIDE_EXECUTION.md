# Guide d'Exécution & Étapes d'Implémentation du Projet

**Auteurs :** Mohamed AASOUM & Hidaya EL HICHAMI  
**Encadrant :** Pr. R. AMATTOUCH  

Ce document constitue le guide de référence pour **exécuter le projet**, comprendre les **étapes d'implémentation** et présenter les résultats du benchmark.

---

## 1. Environnement & Démarrage Rapide

### 1.1. Activation de l'environnement Python
Le projet utilise l'environnement Conda configuré **`optimisation_env`** :

```powershell
& "C:\miniforge3\condabin\conda.bat" activate optimisation_env
```
*(Alternative directe : exécuter les scripts directement avec `& "C:\miniforge3\envs\optimisation_env\python.exe"`).*

### 1.2. Données requises
1. Téléchargez l’archive officielle [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) depuis GroupLens.
2. Décompressez le dossier `ml-1m` dans `data/`.
3. Vérifiez que `data/ml-1m/ratings.dat` existe (1 000 209 évaluations, séparées par `::`). Le chargeur accepte aussi `data/ratings.dat`.

Le dossier `data/` est ignoré par Git et n’est pas fourni dans le dépôt. Chaque personne doit télécharger le dataset avant de lancer les scripts; le benchmark utilise `ratings.dat` (les fichiers `users.dat` et `movies.dat` accompagnent l’archive). Suivez les conditions indiquées par GroupLens.

---

## 2. Étapes d'Implémentation du Système

### Étape 1 — Fondations Communes & Préparation des Données
| Fichier | Rôle dans le projet |
|---|---|
| `src/data.py` | Chargement de MovieLens 1M (`load_movielens_1m`) et découpage vectorisé déterministe par utilisateur (`get_split`, graine 42). Garantit que chaque utilisateur a $\ge 1$ note dans `train`. |
| `src/metrics.py` | Calcul de l'erreur quadratique moyenne (`rmse`) et de l'erreur absolue moyenne (`mae`). |
| `src/protocol.py` | Constantes globales (`SEEDS = (0, 1, 2, 3, 4)`, $d = 10$), enregistrement strict des résultats (`append_result`) et mesure précise du temps et de la RAM crête (`measure`). |
| `src/base.py` | Classe abstraite `BaseRecommender` : interface unifiée pour tous les modèles (`fit`, `predict`, `score_all`, `recommend`). |

### Étape 2 — Métaheuristique PSO & Hybridation ALS
| Fichier | Rôle dans le projet |
|---|---|
| `src/fitness.py` | Formulation du problème d'optimisation `MFProblem` avec calcul SSE compilé en C natif via Numba parallèle (0.65 ms / évaluation). |
| `src/pso.py` | Recommandeur par Essaim Particulaire (`PSORecommender`) avec inertie décroissante $w \in [0.9, 0.4]$ et bornage des facteurs dans $[-1.0, 1.0]$. |
| `src/als_reference.py` | Solveur ALS de référence alternant la résolution des moindres carrés exacts sur matrices creuses CSR/CSC. |
| `src/hybrid.py` | Recommandeur hybride (`HybridRecommender`) qui explore d'abord l'espace avec le PSO, puis affine localement avec l'ALS. |
| `experiments/a_tuning.py` | Réglage des hyperparamètres sur le jeu de validation ($\lambda_{\text{ALS}} = 0.05$, PSO `init_std = 0.03`). |
| `experiments/a_final.py` | Entraînement final sur `train_full` et évaluation sur `test` pour 5 graines (0, 1, 2, 3, 4). Sauvegarde des facteurs `.npz`. |
| `experiments/a_plots.py` | Génération automatique des graphiques de convergence et du résumé Markdown `results/A_summary.md`. |
| `notebooks/main.ipynb` | Notebook interactif complet prêt pour la démonstration. |

### Étape 3 — Modèles de Référence (SGD, BPR, ALS) & Évaluation du Ranking
Le projet intègre ensuite les modèles comparatifs et les métriques de ranking :
1. **`src/als.py` :** Implémentation complète de production d'ALS (compatible avec `HybridRecommender` via l'argument `init=(U0, V0)`).
2. **`src/sgd.py` & `src/bpr.py` :** Modèles de descente de gradient stochastique (SGD) et Bayesian Personalized Ranking (BPR).
3. **Extension de `src/metrics.py` :** Implémentation des métriques `precision_at_k` et `recall_at_k` pour $k \in \{5, 10, 20\}$ sur notes pertinentes $\ge 4.0$.
4. **Évaluation du ranking :** Chargement des facteurs précalculés dans `results/models/*.npz` (`pso_d10_seed*.npz` et `pso+als_d10_seed*.npz`) pour compléter les colonnes `prec@k` et `rec@k` de `results/metaheuristics.csv`.
5. **`src/benchmark.py` :** Fusion comparative globale unifiée et analyse coût computationnel vs qualité.

---

## 3. Ordre Chronologique d'Exécution des Scripts

```mermaid
flowchart TD
    T["1. Tests unitaires\npytest -q"] --> S1["2. Tuning Partie A\npython experiments/a_tuning.py"]
    S1 --> S2["3. Expériences finales Partie A\npython experiments/a_final.py"]
    S2 --> B1["4. Tuning Partie B\npython experiments/b_tuning.py"]
    B1 --> B2["5. Baselines Partie B\npython experiments/b_final.py"]
    B2 --> R["6. Benchmark et analyse\nc_final_report.py + b_cost_quality.py"]
    R --> N["7. Démonstration Notebook\nnotebooks/main.ipynb"]
```

### Étape 1 : Valider tous les tests unitaires
```powershell
pytest -q
```
- **Durée :** ~17 secondes.
- **Résultat attendu :** `20 passed` (100% de succès sur les données, la fitness, le PSO et l'ALS).

---

### Étape 2 : Lancer le réglage des hyperparamètres (Validation)
```powershell
python experiments/a_tuning.py
```
- **Durée :** ~1 minute 30 secondes.
- **Résultats produits :**
  - Sélectionne $\lambda_{\text{ALS}} = 0.05$ $\implies$ écrit `configs/als_best.json`.
  - Sélectionne la configuration PSO optimale $\implies$ écrit `configs/pso_best.json`.
  - Sauvegarde `results/tuning_pso.csv`, `results/figures/tuning_pso.png` et `results/figures/hybrid_vs_als_val.png`.

---

### Étape 3 : Lancer les expériences finales sur le jeu de Test
```powershell
python experiments/a_final.py
```
- **Durée :** ~1 minute 40 secondes.
- **Résultats produits :**
  - Entraîne PSO et l'Hybride PSO+ALS sur `train_full` pour les 5 graines (0, 1, 2, 3, 4).
  - Évalue sur le jeu de test et remplit [results/metaheuristics.csv](results/metaheuristics.csv).
  - Sauvegarde les facteurs latents dans [results/models/](results/models/).
  - Sauvegarde les historiques de convergence dans [results/convergence/](results/convergence/).

---

### Étape 4 : Exécuter la Partie B — réglage et baselines
```powershell
python experiments/backfill_a_ranking.py
python experiments/b_tuning.py
python experiments/b_final.py
python experiments/backfill_reference_ranking.py
```
- `b_tuning.py` sélectionne les configurations ALS, SGD et BPR sur validation et écrit les configurations retenues dans `configs/`.
- `b_final.py` évalue les baselines sur le test et enregistre les résultats dans `results/baselines.csv`.
- Les scripts `backfill_*.py` complètent les métriques de ranking des résultats de la Partie A et des ALS de référence.

---

### Étape 5 : Générer le benchmark global et l’analyse coût/qualité
```powershell
python experiments/c_final_report.py
python experiments/b_cost_quality.py
```
- Le benchmark combine 11 méthodes et 5 graines, puis crée `results/benchmark_all.csv`, `results/benchmark_summary.csv`, `results/benchmark_summary.md` et `results/figures/benchmark_comparison.png`.
- L’analyse B9 génère les tableaux de coût/qualité, les tests appariés et les figures sous `results/figures/`.

---

### Étape 6 : Exécuter et présenter le Notebook Jupyter
Le notebook interactif se trouve dans [notebooks/main.ipynb](notebooks/main.ipynb).

Pour le lancer dans le navigateur :
```powershell
jupyter notebook notebooks/main.ipynb
```
*(Ou ouvrez-le directement dans VS Code avec le kernel `optimisation_env` et cliquez sur **Run All**).*

- Le notebook utilise `FAST = True` et un sous-échantillon reproductible pour montrer le chargement des données, le split, PSO, ALS, l'hybridation et les courbes. Ses métriques de démonstration ne remplacent pas le benchmark complet.

---

## 4. Présentation des Livrables

1. **Le Notebook :** [notebooks/main.ipynb](notebooks/main.ipynb)
   Présente le code interactif, la démonstration de la métaheuristique, le raffinement local et l'affichage des graphiques.
2. **Le Benchmark :** Le tableau [results/benchmark_summary.md](results/benchmark_summary.md) et le fichier [results/metaheuristics.csv](results/metaheuristics.csv) démontrant que les runs ont été exécutés sur 5 graines avec mesures de temps et de mémoire crête.
3. **Le Fonctionnement Théorique & Résultats :** Détaillés dans [README.md](README.md) pour servir de base directe à la rédaction du rapport.
