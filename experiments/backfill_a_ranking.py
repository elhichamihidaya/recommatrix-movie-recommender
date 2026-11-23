"""Backfill ranking metrics for saved Part A PSO and PSO+ALS models."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data import get_split, load_movielens_1m
from src.metrics import ranking_metrics
from src.als_reference import ALSReference


class SavedFactorModel:
    """Minimal scoring adapter for factors saved by a_final.py."""

    def __init__(self, path: Path, method: str, seed: int, split, als_lam: float):
        with np.load(path, allow_pickle=True) as saved:
            U = saved["U"]
            V = saved["V"]
            mu = float(saved["mu"])

        expected_u = (split.train_full.n_users, 10)
        expected_v = (split.train_full.n_items, 10)
        valid = (
            isinstance(U, np.ndarray)
            and isinstance(V, np.ndarray)
            and U.shape == expected_u
            and V.shape == expected_v
            and np.isfinite(U).all()
            and np.isfinite(V).all()
        )
        if not valid:
            if method != "pso+als":
                raise ValueError(f"Invalid saved factors in {path}")
            print(f"Rebuilding invalid hybrid factors for seed {seed} from saved PSO factors...")
            pso = SavedFactorModel(
                path.with_name(f"pso_d10_seed{seed}.npz"), "pso", seed, split, als_lam
            )
            rebuilt = ALSReference(d=10, lam=als_lam, n_iter=10, seed=seed)
            rebuilt.fit(split.train_full, init=(pso.U, pso.V))
            U, V, mu = rebuilt.U, rebuilt.V, rebuilt.mu
            np.savez(path, U=U, V=V, mu=mu)
        self.U = np.asarray(U)
        self.V = np.asarray(V)
        self.mu = mu

    def score_all(self, users: np.ndarray) -> np.ndarray:
        return self.mu + self.U[users] @ self.V.T


def main() -> None:
    results_dir = ROOT / "results"
    models_dir = results_dir / "models"
    csv_path = results_dir / "metaheuristics.csv"
    backup_path = results_dir / "metaheuristics_before_ranking.csv"
    seeds = (0, 1, 2, 3, 4)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing results file: {csv_path}")
    results = pd.read_csv(csv_path)

    model_paths = {
        (method, seed): models_dir / f"{method}_d10_seed{seed}.npz"
        for method in ("pso", "pso+als")
        for seed in seeds
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved factors:\n" + "\n".join(missing))

    ratings = load_movielens_1m(str(ROOT / "data"))
    split = get_split(ratings, seed=42)
    metric_columns = [
        "prec@5", "prec@10", "prec@20", "rec@5", "rec@10", "rec@20"
    ]
    for column in metric_columns:
        if column not in results.columns:
            results[column] = np.nan

    for (method, seed), path in model_paths.items():
        row_mask = (results["method"] == method) & (results["seed"] == seed)
        if row_mask.sum() != 1:
            raise ValueError(f"Expected one row for {method}, seed {seed}; found {row_mask.sum()}.")
        als_lam = float(pd.read_json(ROOT / "configs" / "als_best.json", typ="series")["lam"])
        model = SavedFactorModel(path, method, seed, split, als_lam)
        scores = ranking_metrics(model, split.train_full, split.test)
        values = {
            "prec@5": scores["precision@5"],
            "prec@10": scores["precision@10"],
            "prec@20": scores["precision@20"],
            "rec@5": scores["recall@5"],
            "rec@10": scores["recall@10"],
            "rec@20": scores["recall@20"],
        }
        for column, value in values.items():
            results.loc[row_mask, column] = value
        print(
            f"{method} seed {seed}: P@10={values['prec@10']:.5f}, "
            f"R@10={values['rec@10']:.5f}"
        )

    if not backup_path.exists():
        backup_path.write_bytes(csv_path.read_bytes())
    results.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")
    print(f"Original CSV backup: {backup_path}")


if __name__ == "__main__":
    main()
