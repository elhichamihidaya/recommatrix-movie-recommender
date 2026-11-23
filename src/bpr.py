"""Bayesian Personalized Ranking baseline for implicit movie preferences."""

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


@njit(cache=True)
def _seed_numba(seed: int) -> None:
    """Seed Numba's random generator for repeatable negative sampling."""
    np.random.seed(seed)


@njit(cache=True)
def _bpr_epoch(positive_users, positive_items, positive_mask, U, V, eta, lam, n_samples):
    """Sample user/positive/negative triples and apply pairwise SGD updates."""
    d = U.shape[1]
    total_loss = 0.0
    for _ in range(n_samples):
        pair_idx = np.random.randint(0, len(positive_users))
        user = positive_users[pair_idx]
        positive_item = positive_items[pair_idx]

        negative_item = np.random.randint(0, V.shape[0])
        while positive_mask[user, negative_item]:
            negative_item = np.random.randint(0, V.shape[0])

        difference = 0.0
        for factor in range(d):
            difference += U[user, factor] * (V[positive_item, factor] - V[negative_item, factor])

        # sigmoid(-difference), evaluated without overflow.
        if difference >= 0.0:
            exp_term = np.exp(-difference)
            gradient = exp_term / (1.0 + exp_term)
            total_loss += np.log1p(exp_term)
        else:
            exp_term = np.exp(difference)
            gradient = 1.0 / (1.0 + exp_term)
            total_loss += -difference + np.log1p(exp_term)

        for factor in range(d):
            old_u = U[user, factor]
            old_pos = V[positive_item, factor]
            old_neg = V[negative_item, factor]
            U[user, factor] = old_u + eta * (gradient * (old_pos - old_neg) - lam * old_u)
            V[positive_item, factor] = old_pos + eta * (gradient * old_u - lam * old_pos)
            V[negative_item, factor] = old_neg + eta * (-gradient * old_u - lam * old_neg)

    return total_loss / n_samples


class BPRRecommender(BaseRecommender):
    """Learn to rank positively rated movies ahead of non-positive movies."""

    def __init__(
        self,
        d: int = 10,
        eta: float = 0.05,
        lam: float = 0.01,
        n_epochs: int = 20,
        n_samples_per_epoch: int | None = None,
        seed: int = 0,
        relevant_threshold: float = 4.0,
        lr_decay: float = 0.02,
    ):
        super().__init__()
        self.d = int(d)
        self.eta = float(eta)
        self.lam = float(lam)
        self.n_epochs = int(n_epochs)
        self.n_samples_per_epoch = n_samples_per_epoch
        self.seed = int(seed)
        self.relevant_threshold = float(relevant_threshold)
        self.lr_decay = float(lr_decay)

        if self.d < 1:
            raise ValueError("d must be positive.")
        if self.eta <= 0:
            raise ValueError("eta must be positive.")
        if self.lam < 0:
            raise ValueError("lam cannot be negative.")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1.")
        if self.n_samples_per_epoch is not None and self.n_samples_per_epoch < 1:
            raise ValueError("n_samples_per_epoch must be positive when provided.")
        if self.lr_decay < 0:
            raise ValueError("lr_decay cannot be negative.")

    def fit(self, train: Ratings, val: Ratings | None = None, init=None, **kwargs):
        """Fit BPR from positive ratings; validation is reserved for evaluation."""
        del val, kwargs  # BPR is evaluated with ranking metrics outside the fit loop.
        positive = np.asarray(train.ratings) >= self.relevant_threshold
        positive_users = np.asarray(train.users[positive], dtype=np.int32)
        positive_items = np.asarray(train.items[positive], dtype=np.int32)
        if len(positive_users) == 0:
            raise ValueError("BPR needs at least one positive rating in train.")

        positive_mask = np.zeros((train.n_users, train.n_items), dtype=np.bool_)
        positive_mask[positive_users, positive_items] = True
        # A positive pair is usable when that user has at least one non-positive item.
        positive_counts = positive_mask.sum(axis=1)
        usable = positive_counts[positive_users] < train.n_items
        positive_users = positive_users[usable]
        positive_items = positive_items[usable]
        if len(positive_users) == 0:
            raise ValueError("Every user with a positive rating has rated every movie positively.")

        rng = np.random.default_rng(self.seed)
        self.mu = 0.0  # A constant offset does not affect BPR rankings.
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
        n_samples = self.n_samples_per_epoch or len(positive_users)
        _seed_numba(self.seed)
        t0 = time.perf_counter()

        for epoch in range(self.n_epochs + 1):
            self.U, self.V = U, V
            record = {
                "iteration": epoch,
                "n_evals": self.n_evals,
                "time_s": float(time.perf_counter() - t0),
            }
            if epoch == self.n_epochs:
                self.history.append(record)
                break

            eta_epoch = self.eta / (1.0 + self.lr_decay * epoch)
            record["pairwise_loss"] = float(
                _bpr_epoch(
                    positive_users,
                    positive_items,
                    positive_mask,
                    U,
                    V,
                    eta_epoch,
                    self.lam,
                    n_samples,
                )
            )
            self.n_evals += n_samples
            self.history.append(record)

        self.U, self.V = U, V
        return self


BPR = BPRRecommender
