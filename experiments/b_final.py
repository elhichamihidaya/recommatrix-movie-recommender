"""Run final Part B baseline experiments on train_full and the held-out test set."""

import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from src.benchmark import run_experiment, summarize_results
from src.data import get_split, load_movielens_1m
from src.protocol import SEEDS


def load_best_baselines(configs_dir: Path) -> dict[str, dict]:
    """Load B7 winners, requiring tuning to finish before final runs begin."""
    combined_path = configs_dir / "baselines_best.json"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"{combined_path} is missing. Finish experiments/b_tuning.py first; "
            "it writes the selected validation configs there."
        )
    with open(combined_path, encoding="utf-8") as stream:
        best = json.load(stream)
    required = {"als", "sgd", "bpr"}
    missing = required - set(best)
    if missing:
        raise ValueError(f"Best-config file is missing methods: {sorted(missing)}")
    return best


def run_final_baselines(
    split,
    best_configs: dict[str, dict],
    results_dir: Path = Path("results"),
    seeds=SEEDS,
) -> pd.DataFrame:
    """Run all B baselines for each seed, resume completed rows, and save histories."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "baselines.csv"
    history_dir = results_dir / "histories"

    method_configs = {
        **{method: best_configs[method] for method in ("als", "sgd", "bpr")},
        "global_mean": {},
        "user_mean": {},
        "item_mean": {},
        "popularity": {},
    }
    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    for method, config in method_configs.items():
        variant = config.get("variant", "final")
        expected_hyperparams = json.dumps(
            {key: value for key, value in config.items() if key != "variant"},
            sort_keys=True,
        )
        for seed in seeds:
            if not existing.empty:
                completed = existing[
                    (existing["method"] == method)
                    & (existing["variant"] == variant)
                    & (existing["seed"] == seed)
                ]
                if not completed.empty:
                    if not completed["hyperparams"].eq(expected_hyperparams).all():
                        raise ValueError(
                            f"{csv_path} has different parameters for {method}, seed {seed}. "
                            "Move the old results aside before changing the selected configs."
                        )
                    continue

            print(f"[{method}] final run, seed={seed}")
            run_experiment(
                method,
                config,
                seed,
                split,
                csv_path=csv_path,
                phase="final",
                history_dir=history_dir,
            )
            existing = pd.read_csv(csv_path)

    results = pd.read_csv(csv_path)
    summary = summarize_results(results)
    summary.to_csv(results_dir / "baselines_summary.csv", index=False)
    return results


def main() -> None:
    configs_dir = Path("configs")
    best_configs = load_best_baselines(configs_dir)
    ratings = load_movielens_1m("data")
    split = get_split(ratings, seed=42)
    run_final_baselines(split, best_configs)
    print("\nFinal baseline runs complete; test results and histories are saved in results/.")


if __name__ == "__main__":
    main()
