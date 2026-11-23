"""Tests for BaseRecommender, protocol utilities, metrics and global mean RMSE."""

import os
import time
import numpy as np
import pytest
from src.base import BaseRecommender
from src.data import load_movielens_1m, get_split, Ratings
from src.metrics import rmse, mae
from src.protocol import append_result, measure, RESULT_COLUMNS


class DummyRecommender(BaseRecommender):
    """Simple concrete subclass for testing BaseRecommender interface."""

    def fit(self, train, val=None, init=None, **kwargs):
        rng = np.random.default_rng(0)
        self.U = rng.normal(0, 0.1, size=(train.n_users, 5)).astype(np.float32)
        self.V = rng.normal(0, 0.1, size=(train.n_items, 5)).astype(np.float32)
        self.mu = float(np.mean(train.ratings))
        self._train_csr = train.to_csr()
        self.history.append({"iteration": 0, "n_evals": 0, "time_s": 0.0})
        return self


def test_base_predict_and_score_all():
    n_users, n_items = 50, 40
    train = Ratings(
        users=np.array([0, 1, 2], dtype=np.int32),
        items=np.array([0, 1, 2], dtype=np.int32),
        ratings=np.array([3.0, 4.0, 5.0], dtype=np.float32),
        n_users=n_users,
        n_items=n_items,
    )
    model = DummyRecommender().fit(train)

    rng = np.random.default_rng(42)
    test_u = rng.integers(0, n_users, size=100, dtype=np.int32)
    test_i = rng.integers(0, n_items, size=100, dtype=np.int32)

    # Vectorized predict
    preds_vec = model.predict(test_u, test_i, clip=False)

    # Naive loop
    preds_loop = []
    for u, i in zip(test_u, test_i):
        val = model.mu + float(np.dot(model.U[u], model.V[i]))
        preds_loop.append(val)
    preds_loop = np.array(preds_loop, dtype=np.float32)

    np.testing.assert_allclose(preds_vec, preds_loop, rtol=1e-5, atol=1e-5)

    # Compare with score_all
    unique_test_u = np.unique(test_u)
    scores = model.score_all(unique_test_u)
    for idx_u, u in enumerate(unique_test_u):
        for i in range(n_items):
            expected = model.mu + float(np.dot(model.U[u], model.V[i]))
            assert abs(scores[idx_u, i] - expected) < 1e-5


def test_recommend_excludes_seen():
    train = Ratings(
        users=np.array([0, 0, 0], dtype=np.int32),
        items=np.array([1, 3, 5], dtype=np.int32),
        ratings=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        n_users=2,
        n_items=10,
    )
    model = DummyRecommender().fit(train)
    # Give high scores to seen items specifically
    model.V[1] = 10.0
    model.V[3] = 10.0
    model.V[5] = 10.0

    recs = model.recommend(user=0, k=5, exclude_seen=True)
    assert 1 not in recs
    assert 3 not in recs
    assert 5 not in recs
    assert len(recs) == 5


def test_append_result(tmp_path):
    csv_file = tmp_path / "test_results.csv"

    # Valid append
    row = {
        "method": "dummy",
        "variant": "test",
        "seed": 0,
        "d": 10,
        "hyperparams": {"lr": 0.01},
        "rmse_val": 1.05,
    }
    append_result(csv_file, row)
    assert csv_file.exists()

    # Append second row
    row2 = {"method": "dummy", "seed": 1}
    append_result(csv_file, row2)

    import pandas as pd
    df = pd.read_csv(csv_file)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["method"] == "dummy"
    assert df.iloc[0]["rmse_val"] == 1.05

    # Invalid key raises ValueError
    with pytest.raises(ValueError, match="Unknown keys"):
        append_result(csv_file, {"invalid_col": 123})


def test_measure_memory():
    with measure() as m:
        # Allocate ~200 MB array
        arr = np.ones((25_000_000,), dtype=np.float64)  # 25M * 8 bytes = 200 MB
        arr += 1.0
        time.sleep(0.15)
        del arr

    assert m["time_s"] >= 0.1
    # Plausible peak memory detection
    assert m["peak_mem_mb"] >= 100.0


def test_global_mean_rmse():
    ml1m = load_movielens_1m("data")
    split = get_split(ml1m, seed=42)

    mu_train = float(np.mean(split.train.ratings))
    val_preds = np.full_like(split.val.ratings, fill_value=mu_train)
    val_rmse = rmse(split.val.ratings, val_preds)
    val_mae = mae(split.val.ratings, val_preds)

    print(f"\n[GLOBAL MEAN EVALUATION]")
    print(f"Train mu:  {mu_train:.4f}")
    print(f"Val RMSE:  {val_rmse:.4f}")
    print(f"Val MAE:   {val_mae:.4f}")

    # Should be about 1.12
    assert 1.10 <= val_rmse <= 1.14
