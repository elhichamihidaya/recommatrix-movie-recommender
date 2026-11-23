"""Stochastic gradient descent baseline for explicit MovieLens ratings."""

import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from numba import njit

from src.base import BaseRecommender
from src.data import Ratings
from src.metrics import rmse


@njit(cache=True)
def _sgd_epoch(users, items, ratings, order, U, V, mu, eta, lam):
    """Perform one shuffled SGD epoch in compiled code."""
    d = U.shape[1]
    for pos in range(len(order)):
        idx = order[pos]
        user = users[idx]
        item = items[idx]

        prediction = mu
        for factor in range(d):
            prediction += U[user, factor] * V[item, factor]
        error = ratings[idx] - prediction

        # Use the same pre-update factors for both gradients.
        for factor in range(d):
            old_u = U[user, factor]
            old_v = V[item, factor]
            U[user, factor] = old_u + eta * (error * old_v - lam * old_u)
            V[item, factor] = old_v + eta * (error * old_u - lam * old_v)


class SGDRecommender(BaseRecommender):
    """Matrix-factorization baseline trained by shuffled SGD over ratings."""

    def __init__(
        self,
        d: int = 10,
        eta: float = 0.01,
        lam: float = 0.02,
        n_epochs: int = 20,
        seed: int = 0,
        lr_decay: float = 0.05,
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        super().__init__()
        self.d = int(d)
        self.eta = float(eta)
        self.lam = float(lam)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.lr_decay = float(lr_decay)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        if self.d < 1:
            raise ValueError("d must be positive.")
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.lam < 0:
            raise ValueError("lam cannot be negative.")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1.")
        if self.lr_decay < 0:
            raise ValueError("lr_decay cannot be negative.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")

    def fit(self, train: Ratings, val: Ratings | None = None, init=None, **kwargs):
        """Fit on explicit ratings, decaying the learning rate each epoch.

        If validation data is supplied, stop after ``patience`` epochs without
        improvement and restore the factors with the best validation RMSE.
        """
        if len(train.ratings) == 0:
            raise ValueError("train must contain at least one rating.")

        rng = np.random.default_rng(self.seed)
        self.mu = float(np.mean(train.ratings))
        if init is None:
            U = rng.normal(0.0, 0.1, size=(train.n_users, self.d)).astype(np.float32)
            V = rng.normal(0.0, 0.1, size=(train.n_items, self.d)).astype(np.float32)
        else:
            if len(init) != 2:
                raise ValueError("init must be a pair (U0, V0).")
            U = np.asarray(init[0], dtype=np.float32).copy()
            V = np.asarray(init[1], dtype=np.float32).copy()
            if U.shape != (train.n_users, self.d) or V.shape != (train.n_items, self.d):
                raise ValueError("init factors have shapes incompatible with the data and d.")
            if not np.isfinite(U).all() or not np.isfinite(V).all():
                raise ValueError("init factors must contain only finite values.")

        self._train_csr = train.to_csr()
        self.history = []
        self.n_evals = 0
        best_val = np.inf
        best_factors = None
        epochs_without_improvement = 0
        t0 = time.perf_counter()

        for epoch in range(self.n_epochs + 1):
            self.U, self.V = U, V
            record = {
                "iteration": epoch,
                "n_evals": self.n_evals,
                "time_s": float(time.perf_counter() - t0),
                "rmse_train": rmse(train.ratings, self.predict(train.users, train.items)),
            }
            if val is not None:
                val_rmse = rmse(val.ratings, self.predict(val.users, val.items))
                record["rmse_val"] = val_rmse
                if val_rmse < best_val - self.min_delta:
                    best_val = val_rmse
                    best_factors = (U.copy(), V.copy())
                    epochs_without_improvement = 0
                elif epoch > 0:
                    epochs_without_improvement += 1
            self.history.append(record)

            if epoch == self.n_epochs or (val is not None and epochs_without_improvement >= self.patience):
                break

            order = rng.permutation(len(train.ratings)).astype(np.int64)
            eta_epoch = self.eta / (1.0 + self.lr_decay * epoch)
            _sgd_epoch(
                train.users,
                train.items,
                train.ratings,
                order,
                U,
                V,
                self.mu,
                eta_epoch,
                self.lam,
            )
            self.n_evals += len(train.ratings)

        if val is not None and best_factors is not None:
            U, V = best_factors
            self.U, self.V = U, V
        return self


SGD = SGDRecommender
