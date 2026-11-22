# Projet n°23 — Optimisation des Facteurs Latents par Métaheuristique (MovieLens 1M)

**Auteurs :** Mohamed AASOUM & Hidaya EL HICHAMI  
**Encadrant :** Pr. R. AMATTOUCH  

Ce projet étudie l'optimisation des facteurs latents d'un système de recommandation par métaheuristique (Particle Swarm Optimization - PSO) et son hybridation avec l'algorithme des Moindres Carrés Alternés (Alternating Least Squares - ALS), en alternative ou complément aux approches par descente de gradient (SGD) ou factorisation matricielle classique.

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

## 3. Résultats Détaillés du Benchmark

Les évaluations sont menées sur le jeu de test indépendant ($200\,016$ évaluations) à travers 3 graines aléatoires déterministes (`SEEDS = (0, 1, 2)`).

| Méthode | Graines | RMSE Test | MAE Test | Temps Ajustement (s) | Mémoire Crête (MB) | Évaluations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Moyenne Globale (Baseline)** | - | 1.1159 | 0.9329 | < 0.01 | - | 0 |
| **PSO (seul, accordé)** | 3 | $1.1159 \pm 0.0000$ | $0.9329 \pm 0.0000$ | $13.09 \pm 0.39$ | $102.3 \pm 30.7$ | 6 000 |
| **PSO + ALS (Hybride H1)** | 3 | **$0.8610 \pm 0.0005$** | **$0.6758 \pm 0.0005$** | $16.71 \pm 0.60$ | $95.6 \pm 24.6$ | 6 000 |
| *ALS référence (10 itér)* | 3 | $0.8604 \pm 0.0010$ | $0.6754 \pm 0.0009$ | $4.21 \pm 0.74$ | $97.0 \pm 14.3$ | 0 |
| *ALS référence (30 itér)* | 3 | $0.8575 \pm 0.0002$ | $0.6728 \pm 0.0002$ | $11.72 \pm 1.93$ | $96.5 \pm 5.3$ | 0 |

---

## 4. Analyse Critique & Enseignements

1. **La limite de la grande dimension ($D \approx 10^5$) pour les métaheuristiques pures :**  
   L'algorithme PSO seul stagne autour d'un RMSE de $1.1159$, quasiment identique à la moyenne globale. En dimension $97\,460$, l'espace de recherche est bien trop vaste pour qu'un essaim de 20 particules trouve sans gradient les directions de descente efficaces. C'est un résultat scientifique honnête et représentatif.
   
2. **Puissance du raffinement local :**  
   Dès que les facteurs trouvés par le PSO sont transmis à ALS, le RMSE chute spectaculairement de $1.1159$ à **$0.8610$** en 10 itérations.
   
3. **Gain de l'hybridation vs ALS seul :**  
   L'ALS démarré d'une initialisation aléatoire gaussienne atteint $0.8604$ en 4.2 secondes, contre $0.8610$ en 16.7 secondes pour l'hybride. L'étape d'exploration par PSO ne procure donc pas d'avantage de précision par rapport à une initialisation aléatoire standard pour ALS, tout en ajoutant un coût computationnel modéré.
