"""Hybrid Recommender combining Metaheuristic (PSO) exploration with ALS local refinement."""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd
from src.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender running a metaheuristic (PSO) first, then refining with ALS."""

    def __init__(self, meta: BaseRecommender, refiner: BaseRecommender):
        super().__init__()
        self.meta = meta
        self.refiner = refiner

    def fit(self, train, val=None, **kwargs) -> "HybridRecommender":
        """Fit metaheuristic, then pass its factors as init to the refiner."""
        self.meta.fit(train, val=val, **kwargs)
        return self.refine(train, val=val, **kwargs)

    def refine(self, train, val=None, **kwargs) -> "HybridRecommender":
        """Refine using the already-fitted metaheuristic factors without re-running meta."""
        if self.meta.U is None or self.meta.V is None:
            raise ValueError("Metaheuristic has not been fitted yet. Call fit() first.")

        init = (self.meta.U, self.meta.V)
        self.refiner.fit(train, val=val, init=init, **kwargs)

        self.U = self.refiner.U
        self.V = self.refiner.V
        self.mu = self.refiner.mu
        self._train_csr = self.refiner._train_csr
        self.n_evals = self.meta.n_evals
        self.history = self.refiner.history

        return self

    def history_df(self) -> pd.DataFrame:
        """Merged cumulative history across meta and refiner phases."""
        df_meta = self.meta.history_df().copy()
        df_refiner = self.refiner.history_df().copy()

        df_meta["phase"] = "pso"
        df_refiner["phase"] = "als"

        if not df_meta.empty:
            meta_time_offset = df_meta["time_s"].iloc[-1]
            meta_evals = self.meta.n_evals
        else:
            meta_time_offset = 0.0
            meta_evals = 0

        # Refiner history starts after meta time and keeps meta's n_evals
        df_refiner["time_s"] = df_refiner["time_s"] + meta_time_offset
        df_refiner["n_evals"] = meta_evals

        merged = pd.concat([df_meta, df_refiner], ignore_index=True)
        return merged
