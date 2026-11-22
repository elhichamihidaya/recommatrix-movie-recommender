"""Abstract base recommender interface for matrix factorization models."""

import sys
from pathlib import Path

# Ensure project root is in sys.path if run directly as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.protocol import RATING_MIN, RATING_MAX


class BaseRecommender(ABC):
    """Abstract base class for all matrix factorization recommender models."""

    def __init__(self):
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.mu: float = 0.0
        self.history: list[dict] = []
        self._train_csr: csr_matrix | None = None
        self.n_evals: int = 0

    @abstractmethod
    def fit(self, train, val=None, init=None, **kwargs):
        """Fit model on train data, optionally evaluating on val and using (U0, V0) init."""
        pass

    def predict(self, users: np.ndarray, items: np.ndarray, clip: bool = True) -> np.ndarray:
        """Vectorized prediction: mu + sum(U[u] * V[i]), optionally clipped to [1, 5]."""
        if self.U is None or self.V is None:
            raise ValueError("Model is not fitted yet.")

        u_arr = np.asarray(users, dtype=np.int32)
        i_arr = np.asarray(items, dtype=np.int32)

        # Vectorized dot product along factor dimension
        pred = self.mu + np.sum(self.U[u_arr] * self.V[i_arr], axis=1)

        if clip:
            pred = np.clip(pred, RATING_MIN, RATING_MAX)

        return pred.astype(np.float32)

    def score_all(self, users: np.ndarray) -> np.ndarray:
        """Compute scores for given users against all items: shape (len(users), n_items)."""
        if self.U is None or self.V is None:
            raise ValueError("Model is not fitted yet.")

        u_arr = np.asarray(users, dtype=np.int32)
        scores = self.mu + (self.U[u_arr] @ self.V.T)
        return scores.astype(np.float32)

    def recommend(self, user: int, k: int, exclude_seen: bool = True) -> np.ndarray:
        """Return top-k item indices for a single user, excluding seen items if requested."""
        scores = self.score_all(np.array([user], dtype=np.int32))[0]

        if exclude_seen and self._train_csr is not None:
            seen_items = self._train_csr[user].indices
            scores[seen_items] = -np.inf

        top_k = np.argsort(scores)[::-1][:k]
        return top_k

    def history_df(self) -> pd.DataFrame:
        """Return history records as a pandas DataFrame."""
        return pd.DataFrame(self.history)


if __name__ == "__main__":
    print("BaseRecommender module loaded successfully.")
