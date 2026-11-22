"""Final experiments on test set for PSO, Hybrid, and reference ALS across 3 seeds."""

import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from src.als_reference import ALSReference
from src.data import get_split, load_movielens_1m
from src.hybrid import HybridRecommender
from src.metrics import mae, rmse
from src.protocol import SEEDS, append_result, measure
from src.pso import PSOConfig, PSORecommender


def main():
    ratings = load_movielens_1m("data")
    split = get_split(ratings, seed=42)

    configs_dir = Path("configs")
    results_dir = Path("results")
    models_dir = results_dir / "models"
    conv_dir = results_dir / "convergence"

    models_dir.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)

    with open(configs_dir / "budgets.json") as f:
        budgets = json.load(f)
    budget_final = budgets.get("EVAL_BUDGET_FINAL", 6000)

    with open(configs_dir / "als_best.json") as f:
        als_best = json.load(f)

    with open(configs_dir / "pso_best.json") as f:
        pso_best = json.load(f)

    meta_csv = results_dir / "metaheuristics.csv"
    als_ref_csv = results_dir / "a_hybrid_compare.csv"

    # Existing completed runs check
    existing_meta = set()
    if meta_csv.exists() and meta_csv.stat().st_size > 0:
        df_m = pd.read_csv(meta_csv)
        if not df_m.empty and "method" in df_m.columns and "seed" in df_m.columns:
            for _, r in df_m.iterrows():
                existing_meta.add((r["method"], int(r["seed"])))

    print("=========================================================")
    print("  Running Final Experiments on Test Set (MovieLens 1M)   ")
    print(f"  Seeds: {SEEDS}, Budget Final: {budget_final} evals     ")
    print("=========================================================")

    for seed in SEEDS:
        print(f"\n>>> PROCESSING SEED {seed} <<<")

        # 1. Fit PSO
        pso_cfg = PSOConfig(**pso_best)
        pso_cfg.seed = seed
        pso_cfg.max_evals = budget_final

        pso_model = PSORecommender(pso_cfg)
        pso_key = ("pso", seed)

        if pso_key not in existing_meta:
            print(f"Fitting PSO on train_full (seed {seed})...")
            with measure() as m_pso:
                pso_model.fit(split.train_full)

            # Evaluate on test
            preds_pso = pso_model.predict(split.test.users, split.test.items, clip=True)
            pso_rmse = rmse(split.test.ratings, preds_pso)
            pso_mae = mae(split.test.ratings, preds_pso)

            row_pso = {
                "method": "pso",
                "variant": "tuned",
                "seed": seed,
                "d": 10,
                "hyperparams": pso_cfg.to_dict(),
                "rmse_val": np.nan,
                "rmse_test": pso_rmse,
                "mae_test": pso_mae,
                "fit_time_s": m_pso["time_s"],
                "peak_mem_mb": m_pso["peak_mem_mb"],
                "n_evals": pso_model.n_evals,
            }
            append_result(meta_csv, row_pso)
            print(f"[PSO Seed {seed}] Test RMSE: {pso_rmse:.4f}, MAE: {pso_mae:.4f}, Time: {m_pso['time_s']:.2f}s")
        else:
            print(f"[PSO Seed {seed}] Found in results CSV, loading model...")
            # If skipping, still load saved factors for hybrid
            saved_npz = models_dir / f"pso_d10_seed{seed}.npz"
            if saved_npz.exists():
                data = np.load(saved_npz)
                pso_model.U = data["U"]
                pso_model.V = data["V"]
                pso_model.mu = float(data["mu"])
                pso_model.n_evals = budget_final
            else:
                pso_model.fit(split.train_full)

        # Save factors and history of PSO
        np.savez(models_dir / f"pso_d10_seed{seed}.npz", U=pso_model.U, V=pso_model.V, mu=pso_model.mu)
        pso_model.history_df().to_csv(conv_dir / f"pso_seed{seed}.csv", index=False)

        # 2. Refine with Hybrid (PSO -> ALS 10 iterations)
        hybrid_key = ("pso+als", seed)
        refiner = ALSReference(d=10, lam=als_best["lam"], n_iter=10, seed=seed)
        hybrid_model = HybridRecommender(meta=pso_model, refiner=refiner)

        if hybrid_key not in existing_meta:
            print(f"Refining with ALS 10 iters from PSO solution (seed {seed})...")
            with measure() as m_als:
                hybrid_model.refine(split.train_full)

            # Evaluate hybrid on test
            preds_hyb = hybrid_model.predict(split.test.users, split.test.items, clip=True)
            hyb_rmse = rmse(split.test.ratings, preds_hyb)
            hyb_mae = mae(split.test.ratings, preds_hyb)

            # Total fit time = pso time + als time
            pso_time = pso_model.history[-1]["time_s"] if pso_model.history else 0.0
            total_time = pso_time + m_als["time_s"]

            row_hyb = {
                "method": "pso+als",
                "variant": "H1",
                "seed": seed,
                "d": 10,
                "hyperparams": {"pso": pso_cfg.to_dict(), "als": als_best},
                "rmse_val": np.nan,
                "rmse_test": hyb_rmse,
                "mae_test": hyb_mae,
                "fit_time_s": total_time,
                "peak_mem_mb": m_als["peak_mem_mb"],
                "n_evals": pso_model.n_evals,
            }
            append_result(meta_csv, row_hyb)
            print(f"[Hybrid Seed {seed}] Test RMSE: {hyb_rmse:.4f}, MAE: {hyb_mae:.4f}, Total Time: {total_time:.2f}s")

        # Save factors and history of Hybrid
        np.savez(models_dir / f"pso+als_d10_seed{seed}.npz", U=hybrid_model.U, V=hybrid_model.V, mu=hybrid_model.mu)
        hybrid_model.history_df().to_csv(conv_dir / f"pso+als_seed{seed}.csv", index=False)

        # 3. Reference temporary ALS runs (10 and 30 iters)
        for iters in [10, 30]:
            ref_als = ALSReference(d=10, lam=als_best["lam"], n_iter=iters, seed=seed)
            with measure() as m_ref:
                ref_als.fit(split.train_full)
            preds_ref = ref_als.predict(split.test.users, split.test.items, clip=True)
            ref_rmse = rmse(split.test.ratings, preds_ref)
            ref_mae = mae(split.test.ratings, preds_ref)

            row_ref = {
                "method": f"als_reference_{iters}it",
                "variant": "ref_temporary",
                "seed": seed,
                "d": 10,
                "hyperparams": {"lam": als_best["lam"], "n_iter": iters},
                "rmse_val": np.nan,
                "rmse_test": ref_rmse,
                "mae_test": ref_mae,
                "fit_time_s": m_ref["time_s"],
                "peak_mem_mb": m_ref["peak_mem_mb"],
                "n_evals": 0,
            }
            append_result(als_ref_csv, row_ref)
            print(f"[Reference ALS {iters}it Seed {seed}] Test RMSE: {ref_rmse:.4f}, Time: {m_ref['time_s']:.2f}s")

    print("\n--- Final experiments completed! ---")


if __name__ == "__main__":
    main()
