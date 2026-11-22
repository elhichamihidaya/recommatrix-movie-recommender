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
   + c_2 r_2 (g_{\text{best}} - x)
   $$

   Avec :

   - \(c_1 = 1.5\) : coefficient cognitif.
   - \(c_2 = 1.5\) : coefficient social.
   - \(r_1\) et \(r_2\) : nombres aléatoires.
   - \(v_{\max} = 0.10\) : limite de la vitesse.

3. **Gestion des bornes :** Lorsqu'une composante dépasse l'intervalle \([-1.0, 1.0]\), elle est ramenée à la frontière correspondante et sa vitesse est annulée.

### 2.2. Algorithme des Moindres Carrés Alternés (ALS)

L'algorithme ALS résout alternativement deux problèmes de moindres carrés linéaires.

**Étape 1 — Optimisation des facteurs films :**

Pour chaque film \(i\) :

$$
V_i =
\left(
U_{\Omega_i}^{T} U_{\Omega_i}
+ \lambda n_i I
\right)^{-1}
U_{\Omega_i}^{T}
(r_{\Omega_i} - \mu)
$$

**Étape 2 — Optimisation des facteurs utilisateurs :**

Pour chaque utilisateur \(u\) :

$$
U_u =
\left(
V_{\Omega_u}^{T} V_{\Omega_u}
+ \lambda n_u I
\right)^{-1}
V_{\Omega_u}^{T}
(r_{\Omega_u} - \mu)
$$

Chaque itération nécessite la résolution de systèmes linéaires de dimension \(10 \times 10\), ce qui permet une exécution rapide.

- **Temps d'une itération complète :** environ **0.80 seconde** sur MovieLens 1M.

### 2.3. Hybridation (PSO → ALS)

L'approche hybride combine l'exploration globale du PSO avec le raffinement local de l'ALS en deux étapes :

1. **Exploration globale :** Le PSO explore le paysage non convexe pendant un budget de \(6\,000\) évaluations et produit une solution candidate \((U^*, V^*)\).

2. **Raffinement local :** Les matrices \((U^*, V^*)\) sont transmises explicitement comme initialisation à l'algorithme ALS :

   ```python
   init=(U_star, V_star)
   ```

   L'ALS effectue ensuite 10 itérations de raffinement local afin de converger vers une solution plus stable.

---

## 3. Résultats Détaillés du Benchmark

Les évaluations sont réalisées sur un jeu de test indépendant contenant \(200\,016\) évaluations, avec trois graines aléatoires déterministes :

```python
SEEDS = (0, 1, 2)
```

Les résultats présentés correspondent aux moyennes et écarts-types obtenus sur les trois exécutions.

| Méthode | Graines | RMSE Test | MAE Test | Temps Ajustement (s) | Mémoire Crête (MB) | Évaluations |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Moyenne Globale (Baseline)** | - | 1.1159 | 0.9329 | < 0.01 | - | 0 |
| **PSO (seul, accordé)** | 3 | \(1.1159 \pm 0.0000\) | \(0.9329 \pm 0.0000\) | \(13.09 \pm 0.39\) | \(102.3 \pm 30.7\) | 6 000 |
| **PSO + ALS (Hybride H1)** | 3 | **\(0.8610 \pm 0.0005\)** | **\(0.6758 \pm 0.0005\)** | \(16.71 \pm 0.60\) | \(95.6 \pm 24.6\) | 6 000 |
| *ALS référence (10 itérations)* | 3 | \(0.8604 \pm 0.0010\) | \(0.6754 \pm 0.0009\) | \(4.21 \pm 0.74\) | \(97.0 \pm 14.3\) | 0 |
| *ALS référence (30 itérations)* | 3 | \(0.8575 \pm 0.0002\) | \(0.6728 \pm 0.0002\) | \(11.72 \pm 1.93\) | \(96.5 \pm 5.3\) | 0 |

---

## 4. Analyse Critique et Enseignements

### 4.1. Limites des métaheuristiques pures en grande dimension

L'algorithme PSO seul stagne autour d'un RMSE de \(1.1159\), quasiment identique à celui de la moyenne globale.

Avec un espace de recherche de \(97\,460\) dimensions et un essaim limité à 20 particules, le PSO rencontre des difficultés importantes pour identifier les directions de descente efficaces sans information de gradient.

Ce résultat illustre les limites des métaheuristiques populationnelles classiques lorsqu'elles sont appliquées directement à des problèmes de très grande dimension.

### 4.2. Importance du raffinement local

Lorsque les facteurs latents obtenus par le PSO sont transmis à l'algorithme ALS, le RMSE passe de \(1.1159\) à **\(0.8610\)** après 10 itérations.

Cette amélioration importante montre l'efficacité du raffinement local par ALS pour optimiser les facteurs latents.

### 4.3. Comparaison entre l'approche hybride et ALS seul

L'ALS initialisé aléatoirement atteint un RMSE de \(0.8604\) en environ 4.21 secondes, tandis que l'approche hybride PSO + ALS atteint \(0.8610\) en environ 16.71 secondes.

Les résultats montrent donc que :

- L'hybridation PSO + ALS n'apporte pas de gain de précision par rapport à une initialisation aléatoire standard de l'ALS.
- L'étape d'exploration globale par PSO augmente le temps de calcul.
- L'ALS seul constitue ici une solution plus efficace en termes de compromis entre précision et temps d'exécution.

**Conclusion :** Malgré son intérêt théorique pour l'exploration globale de l'espace de recherche, le PSO seul présente des limites importantes dans un espace de très grande dimension. Son hybridation avec ALS permet d'obtenir de bonnes performances, mais les résultats expérimentaux montrent que l'ALS classique reste plus efficace dans cette configuration.
