"""Tests for PSORecommender on synthetic problem and MovieLens 1M."""

import json
from pathlib import Path
import numpy as np
import pytest
from src.data import Ratings, load_movielens_1m, get_split
from src.pso import PSORecommender, PSOConfig


def test_pso_synthetic_low_rank():
    # Synthetic 40 users x 30 items, d=3, rank 3
    rng = np.random.default_rng(42)
    n_users, n_items, d = 40, 30, 3
    U_true = rng.normal(0, 0.5, size=(n_users, d)).astype(np.float32)
    V_true = rng.normal(0, 0.5, size=(n_items, d)).astype(np.float32)
    R_full = 3.5 + U_true @ V_true.T

    # 50% observed
    total_pairs = n_users * n_items
    all_u, all_i = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    all_u, all_i = all_u.ravel(), all_i.ravel()
    chosen_idx = rng.choice(total_pairs, size=total_pairs // 2, replace=False)

    users = all_u[chosen_idx].astype(np.int32)
    items = all_i[chosen_idx].astype(np.int32)
    ratings = R_full[users, items].astype(np.float32)

    synth_ratings = Ratings(users=users, items=items, ratings=ratings, n_users=n_users, n_items=n_items)

    cfg = PSOConfig(d=3, n_particles=20, max_evals=1000, lam=0.001, seed=42)
    pso = PSORecommender(cfg)
    pso.fit(synth_ratings)

    history = pso.history
    initial_fitness = history[0]["gbest_fitness"]
    final_fitness = history[-1]["gbest_fitness"]

    # Final fitness at least 20% lower than initial
    assert final_fitness <= 0.80 * initial_fitness

    # gbest_fitness is non-increasing
    gbest_series = [rec["gbest_fitness"] for rec in history]
    for i in range(len(gbest_series) - 1):
        assert gbest_series[i + 1] <= gbest_series[i] + 1e-6

    # Determinism with same seed
    pso2 = PSORecommender(cfg)
    pso2.fit(synth_ratings)
    np.testing.assert_allclose(pso.U, pso2.U, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pso.V, pso2.V, rtol=1e-5, atol=1e-5)


def test_pso_real_data_short_run():
    ml1m = load_movielens_1m("data")
    split = get_split(ml1m, seed=42)

    # 500 evals short run on ML-1M train -> val
    cfg = PSOConfig(d=10, n_particles=20, max_evals=500, seed=0)
    pso = PSORecommender(cfg)
    pso.fit(split.train, val=split.val)

    init_fit = pso.history[0]["gbest_fitness"]
    final_fit = pso.history[-1]["gbest_fitness"]
    init_val_rmse = pso.history[0]["rmse_val"]
    final_val_rmse = pso.history[-1]["rmse_val"]

    print(f"\n[S4 REAL DATA SHORT RUN REPORT]")
    print(f"Initial fitness:   {init_fit:.4f}")
    print(f"Final fitness:     {final_fit:.4f}")
    print(f"Initial val RMSE:  {init_val_rmse:.4f}")
    print(f"Final val RMSE:    {final_val_rmse:.4f}")
    print(f"Below global mean (1.115)? {'YES' if final_val_rmse < 1.1149 else 'NO'}")

    assert final_fit <= init_fit + 1e-5


def test_create_budgets_file():
    # Save budgets in configs/budgets.json
    # Since 1 evaluation is ~0.65ms:
    # EVAL_BUDGET_TUNING ~ 2000 evals (a few seconds)
    # EVAL_BUDGET_FINAL ~ 6000 evals
    budgets = {
        "EVAL_BUDGET_TUNING": 2000,
        "EVAL_BUDGET_FINAL": 6000,
        "note": "Calibrated based on Numba fitness throughput of ~0.65 ms/eval on ML-1M"
    }
    cfg_dir = Path("configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_dir / "budgets.json", "w") as f:
        json.dump(budgets, f, indent=2)

    assert (cfg_dir / "budgets.json").exists()
