"""Simple rating and ranking baselines for Part B."""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from src.base import BaseRecommender
from src.data import Ratings
from src.protocol import RATING_MAX, RATING_MIN


class MeanRecommender(BaseRecommender):
    """Global, per-user, or per-item mean baseline for rating prediction."""

    def __init__(self, mode: str = "global"):
        super().__init__()
        if mode not in {"global", "user", "item"}:
            raise ValueError("mode must be 'global', 'user', or 'item'.")
        self.mode = mode
        self.means: np.ndarray | None = None

    def fit(self, train: Ratings, val=None, init=None, **kwargs):
        del val, init, kwargs
        if len(train.ratings) == 0:
            raise ValueError("train must contain at least one rating.")
        self.mu = float(np.mean(train.ratings))
        self._train_csr = train.to_csr()

        if self.mode == "user":
            sums = np.bincount(train.users, weights=train.ratings, minlength=train.n_users)
            counts = np.bincount(train.users, minlength=train.n_users)
        elif self.mode == "item":
            sums = np.bincount(train.items, weights=train.ratings, minlength=train.n_items)
            counts = np.bincount(train.items, minlength=train.n_items)
        else:
            self.means = None
            self.U = np.empty((train.n_users, 0), dtype=np.float32)
            self.V = np.empty((train.n_items, 0), dtype=np.float32)
            self.history = [{"iteration": 0, "n_evals": 0, "rmse_train": None}]
            return self

        self.means = np.full(len(counts), self.mu, dtype=np.float32)
        observed = counts > 0
        self.means[observed] = (sums[observed] / counts[observed]).astype(np.float32)
        self.U = np.empty((train.n_users, 0), dtype=np.float32)
        self.V = np.empty((train.n_items, 0), dtype=np.float32)
        self.history = [{"iteration": 0, "n_evals": 0}]
        return self

    def predict(self, users: np.ndarray, items: np.ndarray, clip: bool = True) -> np.ndarray:
        if self.U is None:
            raise ValueError("Model is not fitted yet.")
        users = np.asarray(users, dtype=np.int32)
        items = np.asarray(items, dtype=np.int32)
        if self.mode == "global":
            predictions = np.full(users.shape, self.mu, dtype=np.float32)
        elif self.mode == "user":
            predictions = self.means[users]
        else:
            predictions = self.means[items]
        if clip:
            predictions = np.clip(predictions, RATING_MIN, RATING_MAX)
        return np.asarray(predictions, dtype=np.float32)

    def score_all(self, users: np.ndarray) -> np.ndarray:
        if self.U is None:
            raise ValueError("Model is not fitted yet.")
        users = np.asarray(users, dtype=np.int32)
        if self.mode == "global":
            row_scores = np.full(len(users), self.mu, dtype=np.float32)
            n_items = self._train_csr.shape[1]
            return np.repeat(row_scores[:, None], n_items, axis=1)
        if self.mode == "user":
            row_scores = self.means[users]
            n_items = self._train_csr.shape[1]
            return np.repeat(row_scores[:, None], n_items, axis=1)
        return np.repeat(self.means[None, :], len(users), axis=0)


class PopularityRecommender(BaseRecommender):
    """Ranking-only baseline that ranks movies by positive-rating count."""

    def __init__(self, relevant_threshold: float = 4.0):
        super().__init__()
        self.relevant_threshold = float(relevant_threshold)
        self.popularity: np.ndarray | None = None

    def fit(self, train: Ratings, val=None, init=None, **kwargs):
        del val, init, kwargs
        if len(train.ratings) == 0:
            raise ValueError("train must contain at least one rating.")
        positive = np.asarray(train.ratings) >= self.relevant_threshold
        self.popularity = np.bincount(
            train.items[positive], minlength=train.n_items
        ).astype(np.float32)
        self.mu = 0.0
        self._train_csr = train.to_csr()
        # Non-empty placeholders mark this model as fitted; it has no latent factors.
        self.U = np.empty((train.n_users, 0), dtype=np.float32)
        self.V = np.empty((train.n_items, 0), dtype=np.float32)
        self.history = [{"iteration": 0, "n_evals": 0}]
        return self

    def predict(self, users: np.ndarray, items: np.ndarray, clip: bool = True) -> np.ndarray:
        del users, items, clip
        raise NotImplementedError("Popularity is a ranking baseline; use ranking metrics, not RMSE/MAE.")

    def score_all(self, users: np.ndarray) -> np.ndarray:
        if self.popularity is None:
            raise ValueError("Model is not fitted yet.")
        users = np.asarray(users, dtype=np.int32)
        return np.repeat(self.popularity[None, :], len(users), axis=0)


class GlobalMeanRecommender(MeanRecommender):
    def __init__(self):
        super().__init__("global")


class UserMeanRecommender(MeanRecommender):
    def __init__(self):
        super().__init__("user")


class ItemMeanRecommender(MeanRecommender):
    def __init__(self):
        super().__init__("item")
