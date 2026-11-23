# Projet n°23 — Optimisation des Facteurs Latents par Métaheuristique (MovieLens 1M)

**Auteurs :** Mohamed AASOUM & Hidaya EL HICHAMI  
**Encadrant :** Pr. R. AMATTOUCH  

Ce projet étudie l'optimisation des facteurs latents d'un système de recommandation par métaheuristique (Particle Swarm Optimization - PSO) et son hybridation avec l'algorithme des Moindres Carrés Alternés (Alternating Least Squares - ALS), en alternative ou complément aux approches par descente de gradient (SGD) ou factorisation matricielle classique.

## Jeu de données MovieLens 1M

Téléchargez l’archive officielle [MovieLens 1M sur GroupLens](https://grouplens.org/datasets/movielens/1m/) et décompressez le dossier `ml-1m` dans `data/`. Le fichier attendu est `data/ml-1m/ratings.dat`. Le benchmark utilise `ratings.dat`; `users.dat` et `movies.dat` sont aussi fournis dans l’archive. Le dossier `data/` n’est pas stocké sur GitHub : chaque personne qui exécute le projet doit télécharger les données et suivre les conditions indiquées par GroupLens.

---

## 1. Formulation Mathématique & Fonctionnement

### 1.1. Modèle de Factorisation Matricielle
Le modèle prédit la note $\hat{r}_{ui}$ attribuée par l'utilisateur $u$ à l'élément $i$ via une factorisation matricielle non biaisée centrée sur la moyenne globale :
$$\hat{r}_{ui} = \mu + U_u \cdot V_i^T$$
- $\mu = 3.5807$ est la moyenne globale calculée sur le jeu d'entraînement (`train_full`).
- $U \in \mathbb{R}^{n_u \times d}$ est la matrice des facteurs latents utilisateurs ($n_u = 6\,040$).
- $V \in \mathbb{R}^{n_i \times d}$ est la matrice des facteurs latents films ($n_i = 3\,706$).
- La dimension latente principale est fixée à $d = 10$.
- Les prédictions finales sont écrêtées dans l'intervalle $[1.0, 5.0]$ pour le calcul des métriques d'erreur (RMSE et MAE).

### 1.2. Encodage d'une Particule (PSO)
Dans l'algorithme PSO, une solution candidate (une particule) est un vecteur continu à une dimension formé par la concaténation aplatie des matrices $U$ et $V$ :
$$x = [\operatorname{vec}(U), \operatorname{vec}(V)] \in \mathbb{R}^D$$
La dimension de l'espace de recherche est :
$$D = (n_u + n_i) \cdot d = (6\,040 + 3\,706) \times 10 = 97\,460$$
Les positions de chaque facteur sont strictement contraintes dans l'intervalle $[-1.0, 1.0]$.

### 1.3. Fonction Objectif (Fitness) et Normalisation
La fitness minimise la racine de l'erreur quadratique moyenne (RMSE) sur l'ensemble des évaluations observées $\Omega$, pondérée par une régularisation de Tikhonov :
$$f(U, V) = \operatorname{RMSE}_\Omega(U, V) + \lambda \cdot \frac{\|U\|_F^2 + \|V\|_F^2}{D}$$
**Point clé de normalisation :** La division de la somme des normes de Frobenius par la dimension $D$ permet de maintenir la pénalité de régularisation à la même échelle que le RMSE, évitant ainsi d'écraser la mesure d'erreur.

### 1.4. Accélération Haute Performance (Numba)
Le calcul de l'erreur sur $|\Omega| = 720\,105$ évaluations a une complexité de $\mathcal{O}(|\Omega| \cdot d)$. L'implémentation s'appuie sur une fonction Numba parallélisée en code machine natif (`_sse` dans `src/fitness.py`) :
- **Temps d'une évaluation complète :** **0.65 milliseconde**.
- Cette rapidité permet d'évaluer des milliers d'individus en quelques secondes sans avoir recours à du mini-batching bruité.

---

## 2. Fonctionnement des Méthodes

### 2.1. Optimisation par Essaim Particulaire (PSO)
L'essaim est composé de $N = 20$ particules :
1. **Inertie adaptative :** Le coefficient d'inertie $w$ décroît linéairement de $w_{\text{start}} = 0.9$ à $w_{\text{end}} = 0.4$ selon la progression du budget d'évaluations (favorisant l'exploration globale au départ, puis l'exploitation locale).
2. **Vitesse :** Mise à jour via $v = w \cdot v + c_1 r_1 (p_{\text{best}} - x) + c_2 r_2 (g_{\text{best}} - x)$ avec $c_1 = c_2 = 1.5$ et écrêtage par $v_{\max} = 0.10$.
3. **Gestion des bornes :** Lorsqu'une composante dépasse $[-1.0, 1.0]$, elle est ramenée sur la frontière et sa vitesse correspondante est annulée.

### 2.2. Algorithme des Moindres Carrés Alternés (ALS)
ALS résout alternativement deux problèmes de moindres carrés linéaires exacts :
- Étape films : pour chaque film $i$, $V_i = (U_{\Omega_i}^T U_{\Omega_i} + \lambda n_i I)^{-1} U_{\Omega_i}^T (r_{\Omega_i} - \mu)$.
- Étape utilisateurs : pour chaque utilisateur $u$, $U_u = (V_{\Omega_u}^T V_{\Omega_u} + \lambda n_u I)^{-1} V_{\Omega_u}^T (r_{\Omega_u} - \mu)$.
Chaque itération nécessite la résolution de systèmes linéaires $10 \times 10$ très rapides ($0.80$ s par itération complète sur MovieLens 1M).

### 2.3. Hybridation (PSO $\to$ ALS)
L'hybridation applique le principe d'optimisation en deux temps :
1. **Exploration globale :** Le PSO explore le paysage non convexe pendant un budget de 6 000 évaluations et produit une solution $(U^*, V^*)$.
2. **Raffinement local :** Les matrices $(U^*, V^*)$ sont transmises comme point de départ explicite (`init=(U*, V*)`) à l'algorithme ALS, qui réalise 10 itérations de descente locale vers un minimum stable.

---

## 3. Résultats du Benchmark MovieLens 1M

Les expériences finales utilisent le split déterministe (seed 42), entraînent les modèles sur `train_full`, puis évaluent sur le jeu de test indépendant (200 016 notes). Les résultats finaux couvrent cinq graines (0 à 4). Les métriques Precision@k et Recall@k considèrent comme pertinentes les notes de test ≥ 4 et excluent les films déjà vus dans l'entraînement.

Les résultats complets par seed sont dans [`results/benchmark_all.csv`](results/benchmark_all.csv), le résumé moyenne ± écart-type dans [`results/benchmark_summary.md`](results/benchmark_summary.md), et le graphique dans [`results/figures/benchmark_comparison.png`](results/figures/benchmark_comparison.png).

## 4. Analyse des résultats

- Le PSO seul reste proche de la moyenne globale pour la prédiction de notes (RMSE moyen **1.1159**) et obtient une Precision@10 de **0.0050**.
- Le raffinement ALS améliore nettement le classement des recommandations : PSO + ALS atteint une Precision@10 de **0.0212** et un RMSE de **0.8609**.
- ALS accordé obtient un RMSE moyen de **0.8582** et une Precision@10 de **0.0212**. Le PSO + ALS donne donc des résultats proches, mais n'améliore pas ALS seul sur ces mesures et ajoute le coût du PSO.
- ALS de référence avec 30 itérations obtient le meilleur RMSE (**0.8576**), avec P@10 = **0.0209** et R@10 = **0.0101**. BPR obtient les meilleurs scores de classement (P@10 = **0.2194**, R@10 = **0.1362**).

Tous les résultats par méthode, y compris les baselines, sont disponibles dans le résumé global référencé ci-dessus.

## Baselines et analyse coût/qualité

- Hyperparamètres retenus après validation : [`results/B7_hyperparameters.md`](results/B7_hyperparameters.md).
- Analyse B9 (complexités, temps/mémoire/cycles, Pareto et tests appariés) : [`results/B9_cost_quality_analysis.md`](results/B9_cost_quality_analysis.md).
- Figures : `results/figures/B9_quality_vs_time.png`, `B9_quality_vs_memory.png`, `B9_quality_vs_cycles.png`, `B9_pareto_fronts.png`.
- Notebook : [`notebooks/main.ipynb`](notebooks/main.ipynb).

Régénérer les tableaux et l’analyse :
```powershell
python experiments/c_final_report.py
python experiments/b_cost_quality.py
```
