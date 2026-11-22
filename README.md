# Projet n°23 — Optimisation des Facteurs Latents par Métaheuristique (MovieLens 1M)

**Auteurs :** Mohamed AASOUM & Hidaya EL HICHAMI  
**Encadrant :** Pr. R. AMATTOUCH

Ce projet étudie l'optimisation des facteurs latents d'un système de recommandation par métaheuristique (*Particle Swarm Optimization - PSO*) et son hybridation avec l'algorithme des Moindres Carrés Alternés (*Alternating Least Squares - ALS*), en alternative ou en complément aux approches par descente de gradient (SGD) ou factorisation matricielle classique.

---

## 1. Formulation Mathématique et Fonctionnement

### 1.1. Modèle de Factorisation Matricielle

Le modèle prédit la note estimée attribuée par l'utilisateur \(u\) au film \(i\), à travers une factorisation matricielle non biaisée centrée sur la moyenne globale :

$$
\hat{r}_{ui} = \mu + U_u \cdot V_i^T
$$

Avec :

- \(\mu = 3.5807\) : moyenne globale calculée sur le jeu d'entraînement (`train_full`).
- \(U \in \mathbb{R}^{n_u \times d}\) : matrice des facteurs latents utilisateurs, avec \(n_u = 6\,040\).
- \(V \in \mathbb{R}^{n_i \times d}\) : matrice des facteurs latents films, avec \(n_i = 3\,706\).
- \(d = 10\) : dimension latente principale.
- Les prédictions finales sont limitées à l'intervalle \([1.0, 5.0]\) pour le calcul des métriques d'erreur RMSE et MAE.

### 1.2. Encodage d'une Particule (PSO)

Dans l'algorithme PSO, une solution candidate (particule) est représentée par un vecteur continu unidimensionnel obtenu par la concaténation des matrices \(U\) et \(V\), après aplatissement :

$$
x = [\mathrm{vec}(U), \mathrm{vec}(V)] \in \mathbb{R}^{D}
$$

La dimension totale de l'espace de recherche est :

$$
\begin{aligned}
D &= (n_u + n_i) \cdot d \\
  &= (6\,040 + 3\,706) \times 10 \\
  &= 97\,460
\end{aligned}
$$

Les positions de chaque facteur sont strictement contraintes dans l'intervalle :

$$
x_j \in [-1.0, 1.0]
$$

### 1.3. Fonction Objectif (Fitness) et Normalisation

La fonction de fitness cherche à minimiser la racine de l'erreur quadratique moyenne (RMSE) sur l'ensemble des évaluations observées \(\Omega\), tout en intégrant une régularisation de Tikhonov :

$$
f(U,V) =
\mathrm{RMSE}_{\Omega}(U,V)
+
\lambda \cdot
\frac{\|U\|_F^2 + \|V\|_F^2}{D}
$$

**Point clé de normalisation :**

La division de la somme des normes de Frobenius par la dimension \(D\) permet de maintenir la pénalité de régularisation à une échelle comparable à celle du RMSE, évitant ainsi que cette pénalité ne domine la mesure d'erreur.

### 1.4. Accélération Haute Performance (Numba)

Le calcul de l'erreur sur les \(720\,105\) évaluations possède une complexité de :

$$
\mathcal{O}(|\Omega| \cdot d)
$$

L'implémentation repose sur une fonction Numba parallélisée et compilée en code machine natif (`_sse` dans `src/fitness.py`).

- **Temps d'une évaluation complète :** **0.65 milliseconde**.
- Cette rapidité permet d'évaluer des milliers d'individus en quelques secondes, sans avoir recours à un mini-batching bruité.

---

## 2. Fonctionnement des Méthodes

### 2.1. Optimisation par Essaim Particulaire (PSO)

L'essaim est composé de \(N = 20\) particules.

1. **Inertie adaptative :** Le coefficient d'inertie \(w\) décroît linéairement de \(w_{\text{start}} = 0.9\) à \(w_{\text{end}} = 0.4\), selon la progression du budget d'évaluations. Cette stratégie favorise l'exploration globale au début, puis l'exploitation locale.

2. **Mise à jour de la vitesse :**

   $$
   v = w \cdot v
   + c_1 r_1 (p_{\text{best}} - x)
   + c_2 r
