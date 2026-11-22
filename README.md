# Projet n°23 — Optimisation des Facteurs Latents par Métaheuristique (MovieLens 1M)

**Auteurs :** Mohamed AASOUM & Hidaya EL HICHAMI  
**Encadrant :** Pr. R. AMATTOUCH  

Ce projet étudie l'optimisation des facteurs latents d'un système de recommandation par métaheuristique (Particle Swarm Optimization - PSO) et son hybridation avec l'algorithme des Moindres Carrés Alternés (Alternating Least Squares - ALS), en alternative ou complément aux approches par descente de gradient (SGD) ou factorisation matricielle classique.

---

## 1. Formulation Mathématique & Fonctionnement

### 1.1. Modèle de Factorisation Matricielle

Le modèle prédit la note $\hat{r}_{ui}$ attribuée par l'utilisateur $u$ à l'élément $i$ via une factorisation matricielle non biaisée centrée sur la moyenne globale :

$$
\hat{r}_{ui} = \mu + U_u \cdot V_i^T
