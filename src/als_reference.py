# Implémentation de référence Alternating Least Squares (ALS) pour la factorisation matricielle.

import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from src.base import BaseRecommender
from src.metrics import rmse


class ALSReference(BaseRecommender):
    """Temporary simple Alternating Least Squares (ALS) solver for Part A."""

    def __init__(self, d: int = 10, lam: float = 0.05, n_iter: int = 10, seed: int = 0):
        super().__init__()
        self.d = int(d)
        self.lam = float(lam)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        if self.d < 1:
            raise ValueError("d must be a positive integer.")
        if self.lam <= 0:
            raise ValueError("lam must be positive so the ALS systems stay regularized.")
        if self.n_iter < 0:
            raise ValueError("n_iter cannot be negative.")

    def fit(self, train, val=None, init=None, n_iter: int | None = None) -> "ALSReference":
        max_iter = self.n_iter if n_iter is None else int(n_iter)
        if max_iter < 0:
            raise ValueError("n_iter cannot be negative.")
        if len(train.ratings) == 0:
            raise ValueError("train must contain at least one rating.")

        rng = np.random.default_rng(self.seed)

        self.mu = float(np.mean(train.ratings))
        rc_train = (train.ratings - self.mu).astype(np.float32)

        n_users = train.n_users
        n_items = train.n_items
        d = self.d

        # CSR (user -> items) and CSC (item -> users) of centered ratings
        csr = csr_matrix((rc_train, (train.users, train.items)), shape=(n_users, n_items), dtype=np.float32)
        csc = csc_matrix((rc_train, (train.users, train.items)), shape=(n_users, n_items), dtype=np.float32)
        self._train_csr = csr

        # Initialization
        if init is not None:
            if len(init) != 2:
                raise ValueError("init must be a pair (U0, V0).")
            init_U, init_V = init
            U = np.asarray(init_U, dtype=np.float64).copy()
            V = np.asarray(init_V, dtype=np.float64).copy()
            expected_u_shape = (n_users, d)
            expected_v_shape = (n_items, d)
            if U.shape != expected_u_shape or V.shape != expected_v_shape:
                raise ValueError(
                    "init factors have incorrect shapes: expected "
                    f"U{expected_u_shape} and V{expected_v_shape}, got U{U.shape} and V{V.shape}."
                )
            if not np.isfinite(U).all() or not np.isfinite(V).all():
                raise ValueError("init factors must contain only finite values.")
        else:
            U = rng.normal(0.0, 0.1, size=(n_users, d)).astype(np.float64)
            V = rng.normal(0.0, 0.1, size=(n_items, d)).astype(np.float64)

        self.history = []
        t0 = time.perf_counter()

        # Log iteration 0 before any updates
        self.U = U.astype(np.float32)
        self.V = V.astype(np.float32)
        preds_train = self.predict(train.users, train.items, clip=True)
        rec0 = {
            "iteration": 0,
            "n_evals": 0,
            "time_s": float(time.perf_counter() - t0),
            "rmse_train": float(rmse(train.ratings, preds_train)),
        }
        if val is not None:
            preds_val = self.predict(val.users, val.items, clip=True)
            rec0["rmse_val"] = float(rmse(val.ratings, preds_val))
        self.history.append(rec0)

        I_d = np.eye(d, dtype=np.float64)

        for it in range(1, max_iter + 1):
            # 1. Update items V
            for i in range(n_items):
                start = csc.indptr[i]
                end = csc.indptr[i + 1]
                n_i = end - start
                if n_i == 0:
                    V[i, :] = 0.0
                    continue
                users_i = csc.indices[start:end]
                r_i = csc.data[start:end].astype(np.float64)
                U_sub = U[users_i, :]  # (n_i, d)
                A = U_sub.T @ U_sub + (self.lam * n_i) * I_d
                b = U_sub.T @ r_i
                V[i, :] = np.linalg.solve(A, b)

            # 2. Update users U
            for u in range(n_users):
                start = csr.indptr[u]
                end = csr.indptr[u + 1]
                n_u = end - start
                if n_u == 0:
                    U[u, :] = 0.0
                    continue
                items_u = csr.indices[start:end]
                r_u = csr.data[start:end].astype(np.float64)
                V_sub = V[items_u, :]  # (n_u, d)
                A = V_sub.T @ V_sub + (self.lam * n_u) * I_d
                b = V_sub.T @ r_u
                U[u, :] = np.linalg.solve(A, b)

            # Log metrics
            self.U = U.astype(np.float32)
            self.V = V.astype(np.float32)
            preds_train = self.predict(train.users, train.items, clip=True)
            rec = {
                "iteration": it,
                "n_evals": 0,
                "time_s": float(time.perf_counter() - t0),
                "rmse_train": float(rmse(train.ratings, preds_train)),
            }
            if val is not None:
                preds_val = self.predict(val.users, val.items, clip=True)
                rec["rmse_val"] = float(rmse(val.ratings, preds_val))
            self.history.append(rec)

        self.n_evals = 0
        return self


# Public name used by the Part B implementation. Keep ALSReference as a
# backwards-compatible name because the existing hybrid code imports it.
ALS = ALSReference
