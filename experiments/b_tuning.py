"""Tune Part B baselines using validation data only."""

import json
import sys
from itertools import product
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from src.benchmark import run_experiment, summarize_results
from src.data import get_split, load_movielens_1m


TUNING_SEEDS = (0, 1, 2)


def baseline_grids() -> dict[str, list[dict]]:
    """Return comparable search grids: 24 ALS and 27 each for SGD and BPR."""
    als = [
        {"d": d, "lam": lam, "n_iter": n_iter}
        for d, lam, n_iter in product((5, 10, 20), (0.02, 0.05, 0.1, 0.2), (10, 20))
    ]
    sgd = [
        {"d": d, "eta": eta, "lam": lam, "n_epochs": 10, "lr_decay": 0.05, "patience": 3}
        for d, eta, lam in product((5, 10, 20), (0.002, 0.005, 0.01), (0.005, 0.02, 0.05))
    ]
    bpr = [
        {
            "d": d,
            "eta": eta,
            "lam": lam,
            "n_epochs": 10,
            "relevant_threshold": 4.0,
            "lr_decay": 0.02,
        }
        for d, eta, lam in product((5, 10, 20), (0.01, 0.03, 0.05), (0.001, 0.01, 0.05))
    ]
    return {"als": als, "sgd": sgd, "bpr": bpr}


def tune_one_baseline(method: str, configs: list[dict], split, configs_dir: Path, results_dir: Path) -> dict:
    """Evaluate each config on validation with three seeds and save its winner."""
    raw_path = results_dir / f"tuning_{method}.csv"
    summary_path = results_dir / f"tuning_{method}_summary.csv"
    best_path = configs_dir / f"{method}_best.json"
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    total = len(configs)
    expected_variants = {f"{method}_cfg_{index:02d}" for index in range(1, total + 1)}
    if not existing.empty:
        stale_variants = set(existing["variant"].astype(str)) - expected_variants
        if stale_variants:
            raise ValueError(
                f"{raw_path} contains configs no longer in this grid: {sorted(stale_variants)}. "
                "Move that CSV aside before starting a different grid."
            )
    print(f"\n--- Tuning {method.upper()}: {total} configs x {len(TUNING_SEEDS)} seeds ---")
    for index, values in enumerate(configs, start=1):
        config = {**values, "variant": f"{method}_cfg_{index:02d}"}
        print(f"[{index:02d}/{total}] {config['variant']}: {values}")
        expected_hyperparams = json.dumps(values, sort_keys=True)
        matching_variant = existing[existing["variant"] == config["variant"]] if not existing.empty else existing
        if not matching_variant.empty and not matching_variant["hyperparams"].eq(expected_hyperparams).all():
            raise ValueError(
                f"{raw_path} contains a different config for {config['variant']}; "
                "move that CSV aside before starting a different grid."
            )
        if matching_variant.empty:
            completed_seeds = set()
        else:
            required_metric = "prec@10" if method == "bpr" else "rmse_val"
            completed_rows = matching_variant[matching_variant[required_metric].notna()]
            completed_seeds = set(completed_rows["seed"].astype(int))
        for seed in TUNING_SEEDS:
            if seed in completed_seeds:
                continue
            run_experiment(
                method,
                config,
                seed,
                split,
                csv_path=raw_path,
                phase="validation",
            )
        existing = pd.read_csv(raw_path)

    raw = existing
    summary = summarize_results(raw)
    summary.to_csv(summary_path, index=False)

    if method == "bpr":
        ranked = summary.sort_values(
            ["prec@10_mean", "rec@10_mean", "fit_time_s_mean"],
            ascending=[False, False, True],
        )
        selection_metric = "precision@10"
        best_score = float(ranked.iloc[0]["prec@10_mean"])
    else:
        ranked = summary.sort_values(["rmse_val_mean", "fit_time_s_mean"], ascending=[True, True])
        selection_metric = "rmse_val"
        best_score = float(ranked.iloc[0]["rmse_val_mean"])

    best_row = ranked.iloc[0]
    best_config = json.loads(best_row["hyperparams"])
    best_config["variant"] = str(best_row["variant"])
    best_config["selection_metric"] = selection_metric
    best_config["selection_score"] = best_score
    with open(best_path, "w", encoding="utf-8") as stream:
        json.dump(best_config, stream, indent=2)

    print(f"[{method.upper()}] Best {selection_metric}: {best_score:.5f}; saved to {best_path}")
    return best_config


def tune_baselines(split, configs_dir: Path = Path("configs"), results_dir: Path = Path("results")) -> dict:
    """Tune ALS, SGD, and BPR on the shared validation set."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    grids = baseline_grids()
    winners = {}
    for method in ("als", "sgd", "bpr"):
        winners[method] = tune_one_baseline(method, grids[method], split, configs_dir, results_dir)
    return winners


def main() -> None:
    ratings = load_movielens_1m("data")
    split = get_split(ratings, seed=42)
    winners = tune_baselines(split)
    with open("configs/baselines_best.json", "w", encoding="utf-8") as stream:
        json.dump(winners, stream, indent=2)
    print("\nBaseline tuning complete. The test set was not used.")


if __name__ == "__main__":
    main()
