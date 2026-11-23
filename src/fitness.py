"""Numba-accelerated matrix factorization fitness function and problem definition."""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _sse(U: np.ndarray, V: np.ndarray, users: np.ndarray, items: np.ndarray, r: np.ndarray) -> float:
    """Parallel sum of squared errors between centered ratings and U @ V.T."""
    s = 0.0
    for n in prange(r.shape[0]):
        acc = 0.0
        for f in range(U.shape[1]):
            acc += U[users[n], f] * V[items[n], f]
        e = r[n] - acc
        s += e * e
    return s


class MFProblem:
    """Optimization problem formulation for latent factors U and V."""

    def __init__(self, train, d: int = 10, lam: float = 0.01, bound: float = 1.0):
        self.train = train
        self.d = int(d)
        self.lam = float(lam)
        self.bound = float(bound)

        self.mu = float(np.mean(train.ratings))
        self.r_c = (train.ratings - self.mu).astype(np.float32)
        self.users = train.users.astype(np.int32)
        self.items = train.items.astype(np.int32)

        self.n_users = train.n_users
        self.n_items = train.n_items
        self.dim = (self.n_users + self.n_items) * self.d

        self._u_size = self.n_users * self.d

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack 1D particle into views U and V without copying."""
        x_flat = np.asarray(x, dtype=np.float32)
        U = x_flat[:self._u_size].reshape(self.n_users, self.d)
        V = x_flat[self._u_size:].reshape(self.n_items, self.d)
        return U, V

    def pack(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Pack U and V into a single float32 1D vector."""
        return np.concatenate([U.ravel(), V.ravel()]).astype(np.float32)

    def make_batch(self, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a random mini-batch of centered ratings."""
        n = len(self.r_c)
        k = max(1, int(round(frac * n)))
        idx = rng.choice(n, size=k, replace=False)
        return self.users[idx], self.items[idx], self.r_c[idx]

    def rmse(self, x: np.ndarray, batch: tuple | None = None, backend: str = "numba") -> float:
        """Compute RMSE against (centered) ratings."""
        if batch is None:
            users, items, r = self.users, self.items, self.r_c
        else:
            users, items, r = batch

        U, V = self.unpack(x)

        if backend == "numba":
            sse_val = _sse(U, V, users, items, r)
            return float(np.sqrt(sse_val / len(r)))
        elif backend == "numpy":
            preds = np.sum(U[users] * V[items], axis=1)
            diff = r - preds
            return float(np.sqrt(np.mean(diff * diff)))
        elif backend == "naive":
            s = 0.0
            for idx in range(len(r)):
                acc = 0.0
                u = users[idx]
                i = items[idx]
                for f in range(self.d):
                    acc += float(U[u, f]) * float(V[i, f])
                diff = float(r[idx]) - acc
                s += diff * diff
            return float(np.sqrt(s / len(r)))
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def fitness(self, x: np.ndarray, batch: tuple | None = None) -> float:
        """Compute fitness = RMSE_Omega(U, V) + lam * (||U||_F^2 + ||V||_F^2) / dim."""
        U, V = self.unpack(x)
        r_rmse = self.rmse(x, batch=batch, backend="numba")
        norm_sq = float(np.sum(np.square(U, dtype=np.float64)) + np.sum(np.square(V, dtype=np.float64)))
        reg = self.lam * norm_sq / self.dim
        return float(r_rmse + reg)

    def fitness_naive(self, x: np.ndarray, batch: tuple | None = None) -> float:
        """Explicit Python loops for fitness testing."""
        r_rmse = self.rmse(x, batch=batch, backend="naive")
        norm_sq = float(np.sum(np.square(x, dtype=np.float64)))
        reg = self.lam * norm_sq / self.dim
        return float(r_rmse + reg)

    def fitness_swarm(self, X: np.ndarray, batch: tuple | None = None) -> np.ndarray:
        """Compute fitness for a population (N, dim) -> (N,)."""
        N = X.shape[0]
        scores = np.empty(N, dtype=np.float64)
        for i in range(N):
            scores[i] = self.fitness(X[i], batch=batch)
        return scores

    def init_population(self, n: int, std: float, rng: np.random.Generator) -> np.ndarray:
        """Initialize population ~ N(0, std^2), clipped to [-bound, bound]."""
        X = rng.normal(0.0, std, size=(n, self.dim)).astype(np.float32)
        np.clip(X, -self.bound, self.bound, out=X)
        return X
