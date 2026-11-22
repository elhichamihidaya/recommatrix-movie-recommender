"""Tests for ALSReference and HybridRecommender."""

import numpy as np
import pytest
from src.data import load_movielens_1m, get_split, make_subsample
from src.als_reference import ALSReference
from src.pso import PSORecommender, PSOConfig
from src.hybrid import HybridRecommender
from src.metrics import rmse


@pytest.fixture(scope="session")
def ml1m_data():
    ratings = load_movielens_1m("data")
    split = get_split(ratings, seed=42)
    return split


def test_als_subsample_decreases_rmse(ml1m_data):
    sub = make_subsample(ml1m_data.train, n_users=150, n_items=150, seed=42)
    als = ALSReference(d=5, lam=0.05, n_iter=5, seed=0)
    als.fit(sub)

    train_rmses = [h["rmse_train"] for h in als.history]
    assert len(train_rmses) == 6  # iters 0..5
    # Train RMSE after 5 iterations should be clearly lower than iteration 0
    assert train_rmses[-1] < train_rmses[0]


def test_hybrid_uses_init_and_cumulative_history(ml1m_data):
    sub = make_subsample(ml1m_data.train, n_users=100, n_items=100, seed=42)
    sub_split = get_split(sub, seed=0)

    pso_cfg = PSOConfig(d=4, n_particles=10, max_evals=100, seed=0)
    meta = PSORecommender(pso_cfg)
    refiner = ALSReference(d=4, lam=0.05, n_iter=3, seed=0)

    hybrid = HybridRecommender(meta=meta, refiner=refiner)
    hybrid.fit(sub_split.train, val=sub_split.val)

    # PSO solution predictions on val
    meta_preds_val = meta.predict(sub_split.val.users, sub_split.val.items, clip=True)
    expected_val_rmse = float(rmse(sub_split.val.ratings, meta_preds_val))

    # Refiner's iteration 0 val RMSE must match meta's val RMSE
    refiner_it0_val_rmse = refiner.history[0]["rmse_val"]
    assert abs(refiner_it0_val_rmse - expected_val_rmse) < 1e-4

    # Merged history check
    df_hist = hybrid.history_df()
    assert "phase" in df_hist.columns
    assert set(df_hist["phase"].unique()) == {"pso", "als"}

    # Time must be non-decreasing
    times = df_hist["time_s"].to_numpy()
    assert np.all(np.diff(times) >= -1e-6)


def test_als_full_data_reference_benchmark(ml1m_data):
    # Full ML-1M train -> val with d=10, 10 iterations
    als = ALSReference(d=10, lam=0.05, n_iter=10, seed=0)
    als.fit(ml1m_data.train, val=ml1m_data.val)

    final_val_rmse = als.history[-1]["rmse_val"]
    total_time = als.history[-1]["time_s"]
    time_per_iter = total_time / 10.0

    print(f"\n[S5 ALS REFERENCE BENCHMARK REPORT]")
    print(f"ALS 10 iters Val RMSE: {final_val_rmse:.4f}")
    print(f"Total time:            {total_time:.2f} s")
    print(f"Time per iteration:    {time_per_iter:.2f} s")

    # Order of magnitude should be about 0.85 - 0.90
    assert 0.80 <= final_val_rmse <= 0.95
