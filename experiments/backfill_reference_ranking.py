"""Compute missing ranking metrics for final reference ALS models."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.als_reference import ALSReference
from src.data import get_split, load_movielens_1m
from src.metrics import ranking_metrics


def main() -> None:
    results_dir = ROOT / "results"
    csv_path = results_dir / "a_hybrid_compare.csv"
    backup_path = results_dir / "a_hybrid_compare_before_ranking.csv"
    data = pd.read_csv(csv_path)
    ratings = load_movielens_1m(str(ROOT / "data"))
    split = get_split(ratings, seed=42)
    configs = json.loads((ROOT / "configs" / "als_best.json").read_text(encoding="utf-8"))
    lam = float(configs["lam"])

    ranking_columns = ["prec@5", "prec@10", "prec@20", "rec@5", "rec@10", "rec@20"]
    for column in ranking_columns:
        if column not in data:
            data[column] = np.nan
    if not backup_path.exists():
        backup_path.write_bytes(csv_path.read_bytes())

    for iterations in (10, 30):
        method = f"als_reference_{iterations}it"
        for seed in range(5):
            mask = (data.method == method) & (data.seed == seed)
            if int(mask.sum()) != 1:
                raise ValueError(f"Expected one row for {method}, seed {seed}; found {int(mask.sum())}")
            if data.loc[mask, ranking_columns].notna().all(axis=None):
                print(f"{method} seed {seed}: already complete, skipping")
                continue

            model = ALSReference(d=10, lam=lam, n_iter=iterations, seed=seed)
            model.fit(split.train_full)
            ranked = ranking_metrics(model, split.train_full, split.test)
            values = {
                "prec@5": ranked["precision@5"],
                "prec@10": ranked["precision@10"],
                "prec@20": ranked["precision@20"],
                "rec@5": ranked["recall@5"],
                "rec@10": ranked["recall@10"],
                "rec@20": ranked["recall@20"],
            }
            for column, value in values.items():
                data.loc[mask, column] = value
            data.to_csv(csv_path, index=False)
            print(f"{method} seed {seed}: P@10={values['prec@10']:.5f}, R@10={values['rec@10']:.5f}")

    print(f"Updated {csv_path}")
    print(f"Original CSV backup: {backup_path}")


if __name__ == "__main__":
    main()
