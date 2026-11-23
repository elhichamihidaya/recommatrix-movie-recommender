"""Hyperparameter tuning on validation set only for ALS, PSO, and Hybrid comparison."""

import json
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.als_reference import ALSReference
from src.data import get_split, load_movielens_1m
from src.hybrid import HybridRecommender
from src.metrics import rmse
from src.pso import PSOConfig, PSORecommender


def tune_als(train, val, configs_dir: Path) -> dict:
    """Tune ALS regularization lambda on validation set."""
    out_file = configs_dir / "als_best.json"
    if out_file.exists():
        with open(out_file) as f:
            best_cfg = json.load(f)
        print(f"[ALS Tuning] Loaded existing best config: {best_cfg}")
        return best_cfg

    lambdas = [0.02, 0.05, 0.1, 0.2]
    best_lam = None
    best_val_rmse = float("inf")
    results = []

    print("\n--- Tuning ALS lambda (10 iters, seed 0) ---")
    for lam in lambdas:
        als = ALSReference(d=10, lam=lam, n_iter=10, seed=0)
        als.fit(train, val=val)
        val_rmse = als.history[-1]["rmse_val"]
        print(f"lambda={lam:0.2f} -> Val RMSE: {val_rmse:.4f}")
        results.append({"lam": lam, "val_rmse": val_rmse})
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_lam = lam

    best_cfg = {"d": 10, "lam": best_lam, "n_iter": 10, "val_rmse": best_val_rmse}
    with open(out_file, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"[ALS Tuning] Best config saved: {best_cfg}")
    return best_cfg


def tune_pso(train, val, configs_dir: Path, results_dir: Path, budget_tuning: int) -> dict:
    """Tune PSO hyperparameter grid around baseline using validation set."""
    best_file = configs_dir / "pso_best.json"
    csv_file = results_dir / "tuning_pso.csv"

    # Base configuration
    base = {
        "name": "base",
        "d": 10,
        "n_particles": 20,
        "max_evals": budget_tuning,
        "lam": 0.01,
        "init_std": 0.1,
        "bound": 1.0,
        "w_start": 0.9,
        "w_end": 0.4,
        "c1": 1.5,
        "c2": 1.5,
        "vmax_frac": 0.05,
        "seed": 0,
    }

    # One-at-a-time variations
    grid = [base]
    variations = [
        ("n_particles=10", {"n_particles": 10}),
        ("n_particles=40", {"n_particles": 40}),
        ("vmax_frac=0.01", {"vmax_frac": 0.01}),
        ("vmax_frac=0.1", {"vmax_frac": 0.1}),
        ("c1=c2=1.0", {"c1": 1.0, "c2": 1.0}),
        ("c1=c2=2.0", {"c1": 2.0, "c2": 2.0}),
        ("init_std=0.03", {"init_std": 0.03}),
        ("init_std=0.3", {"init_std": 0.3}),
        ("lam=0.0", {"lam": 0.0}),
        ("lam=0.1", {"lam": 0.1}),
    ]

    for label, diff in variations:
        cfg_dict = base.copy()
        cfg_dict["name"] = label
        cfg_dict.update(diff)
        grid.append(cfg_dict)

    df_existing = pd.DataFrame()
    if csv_file.exists():
        df_existing = pd.read_csv(csv_file)

    results = []
    print("\n--- Tuning PSO (11 configurations, seed 0) ---")
    for cfg_dict in grid:
        name = cfg_dict["name"]
        if not df_existing.empty and name in df_existing["name"].values:
            row = df_existing[df_existing["name"] == name].iloc[0].to_dict()
            results.append(row)
            print(f"[{name}] Loaded from CSV -> Val RMSE: {row['val_rmse']:.4f}")
            continue

        pso_cfg = PSOConfig(
            d=cfg_dict["d"],
            n_particles=cfg_dict["n_particles"],
            max_evals=cfg_dict["max_evals"],
            lam=cfg_dict["lam"],
            init_std=cfg_dict["init_std"],
            bound=cfg_dict["bound"],
            w_start=cfg_dict["w_start"],
            w_end=cfg_dict["w_end"],
            c1=cfg_dict["c1"],
            c2=cfg_dict["c2"],
            vmax_frac=cfg_dict["vmax_frac"],
            seed=cfg_dict["seed"],
        )

        pso = PSORecommender(pso_cfg)
        pso.fit(train, val=val)

        final_rec = pso.history[-1]
        row = {
            "name": name,
            "val_rmse": final_rec["rmse_val"],
            "final_fitness": final_rec["gbest_fitness"],
            "n_evals": pso.n_evals,
            "time_s": final_rec["time_s"],
            "config_json": pso_cfg.to_json(),
        }
        results.append(row)
        print(f"[{name}] Done -> Val RMSE: {row['val_rmse']:.4f}, Fitness: {row['final_fitness']:.4f}, Time: {row['time_s']:.1f}s")

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_file, index=False)

    # Sort by val RMSE
    df_sorted = df_res.sort_values("val_rmse")
    best_row = df_sorted.iloc[0]
    best_cfg = json.loads(best_row["config_json"])

    with open(best_file, "w") as f:
        json.dump(best_cfg, f, indent=2)

    # Plot bar chart
    fig_path = results_dir / "figures" / "tuning_pso.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.barh(df_sorted["name"], df_sorted["val_rmse"], color="skyblue", edgecolor="navy")
    plt.axvline(1.1149, color="red", linestyle="--", label="Global Mean (1.115)")
    plt.xlabel("Validation RMSE")
    plt.title("PSO Tuning: Validation RMSE across Configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"[PSO Tuning] Best config saved: {best_row['name']} (Val RMSE: {best_row['val_rmse']:.4f})")
    return best_cfg


def compare_hybrid_vs_als(train, val, als_best: dict, pso_best: dict, results_dir: Path, budget_final: int):
    """Compare ALS (10 and 30 iters) with Hybrid (PSO + ALS) on validation set."""
    csv_file = results_dir / "a_hybrid_compare_val.csv"
    fig_file = results_dir / "figures" / "hybrid_vs_als_val.png"

    print("\n--- Comparing Hybrid vs ALS on Validation (Seed 0) ---")

    # (a) ALS random init, 10 iterations
    print("Running (a) ALS 10 iterations...")
    als10 = ALSReference(d=10, lam=als_best["lam"], n_iter=10, seed=0)
    als10.fit(train, val=val)
    df_als10 = als10.history_df()
    df_als10["run"] = "ALS (10 iters)"

    # (b) PSO with budget_final, then ALS 10 iterations
    print(f"Running (b) Hybrid: PSO ({budget_final} evals) + ALS (10 iters)...")
    pso_cfg = PSOConfig(**pso_best)
    pso_cfg.max_evals = budget_final
    meta = PSORecommender(pso_cfg)
    refiner = ALSReference(d=10, lam=als_best["lam"], n_iter=10, seed=0)
    hybrid = HybridRecommender(meta=meta, refiner=refiner)
    hybrid.fit(train, val=val)
    df_hybrid = hybrid.history_df()
    df_hybrid["run"] = "Hybrid PSO+ALS"

    # (c) ALS random init, 30 iterations
    print("Running (c) ALS 30 iterations...")
    als30 = ALSReference(d=10, lam=als_best["lam"], n_iter=30, seed=0)
    als30.fit(train, val=val)
    df_als30 = als30.history_df()
    df_als30["run"] = "ALS (30 iters)"

    # Combine histories
    combined = pd.concat([df_als10, df_hybrid, df_als30], ignore_index=True)
    combined.to_csv(csv_file, index=False)

    # Plot curves: Val RMSE vs cumulative time
    plt.figure(figsize=(9, 6))
    for run_name, grp in combined.groupby("run"):
        plt.plot(grp["time_s"], grp["rmse_val"], marker="o", label=run_name)
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Validation RMSE")
    plt.title("Convergence Comparison on Validation Set (MovieLens 1M)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    plt.close()

    print(f"[Comparison] Results saved to {csv_file} and {fig_file}")


def main():
    ratings = load_movielens_1m("data")
    split = get_split(ratings, seed=42)

    configs_dir = Path("configs")
    results_dir = Path("results")
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(configs_dir / "budgets.json") as f:
        budgets = json.load(f)
    budget_tuning = budgets.get("EVAL_BUDGET_TUNING", 2000)
    budget_final = budgets.get("EVAL_BUDGET_FINAL", 6000)

    # 1. ALS tuning
    als_best = tune_als(split.train, split.val, configs_dir)

    # 2. PSO tuning
    pso_best = tune_pso(split.train, split.val, configs_dir, results_dir, budget_tuning)

    # 3. Hybrid vs ALS
    compare_hybrid_vs_als(split.train, split.val, als_best, pso_best, results_dir, budget_final)
    print("\n--- Tuning completed successfully! ---")


if __name__ == "__main__":
    main()
