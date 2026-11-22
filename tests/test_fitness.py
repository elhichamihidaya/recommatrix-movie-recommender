"""Tests for Numba fitness function and MFProblem on MovieLens 1M."""

import time
import numpy as np
import pytest
from src.data import load_movielens_1m, get_split, make_subsample
from src.fitness import MFProblem


@pytest.fixture(scope="session")
def ml1m_split():
    ratings = load_movielens_1m("data")
    return get_split(ratings, seed=42)


def test_fitness_implementations_match(ml1m_split):
    # Subsample 2000 ratings for numerical verification
    sub = ml1m_split.train.subset(np.arange(2000))
    prob = MFProblem(sub, d=5, lam=0.01)

    rng = np.random.default_rng(123)
    x = prob.init_population(1, std=0.1, rng=rng)[0]

    # Warm-up Numba compilation
    rmse_numba = prob.rmse(x, backend="numba")
    rmse_numpy = prob.rmse(x, backend="numpy")
    rmse_naive = prob.rmse(x, backend="naive")

    assert abs(rmse_numba - rmse_numpy) / rmse_numpy < 1e-4
    assert abs(rmse_numba - rmse_naive) / rmse_naive < 1e-4

    fit_numba = prob.fitness(x)
    fit_naive = prob.fitness_naive(x)
    assert abs(fit_numba - fit_naive) / fit_naive < 1e-4


def test_unpack_returns_views(ml1m_split):
    prob = MFProblem(ml1m_split.train, d=10, lam=0.01)
    x = np.zeros(prob.dim, dtype=np.float32)
    U, V = prob.unpack(x)

    # Modifying view modifies x
    U[0, 0] = 42.0
    V[0, 0] = 84.0
    assert x[0] == 42.0
    assert x[prob._u_size] == 84.0


def test_fitness_swarm_matches_loop(ml1m_split):
    sub = ml1m_split.train.subset(np.arange(5000))
    prob = MFProblem(sub, d=4, lam=0.02)
    rng = np.random.default_rng(42)
    X = prob.init_population(5, std=0.1, rng=rng)

    swarm_fit = prob.fitness_swarm(X)
    loop_fit = np.array([prob.fitness(x) for x in X])
    np.testing.assert_allclose(swarm_fit, loop_fit, rtol=1e-5, atol=1e-5)


def test_zero_solution_matches_global_mean(ml1m_split):
    prob = MFProblem(ml1m_split.train, d=10, lam=0.01)
    x_zero = np.zeros(prob.dim, dtype=np.float32)

    # With x = 0, predictions are 0 (relative to mu), so RMSE equals std of centered ratings
    expected_rmse = float(np.sqrt(np.mean(prob.r_c ** 2)))
    actual_rmse = prob.rmse(x_zero, backend="numba")
    assert abs(actual_rmse - expected_rmse) < 1e-5


def test_make_batch(ml1m_split):
    prob = MFProblem(ml1m_split.train, d=5, lam=0.01)
    rng = np.random.default_rng(0)
    batch = prob.make_batch(frac=0.1, rng=rng)
    users, items, r = batch
    expected_size = int(round(0.1 * len(prob.r_c)))
    assert len(users) == expected_size
    assert len(items) == expected_size
    assert len(r) == expected_size

    x = np.zeros(prob.dim, dtype=np.float32)
    fit_batch = prob.fitness(x, batch=batch)
    assert fit_batch > 0


def test_full_evaluation_timing(ml1m_split):
    prob = MFProblem(ml1m_split.train, d=10, lam=0.01)
    rng = np.random.default_rng(0)
    x = prob.init_population(1, std=0.1, rng=rng)[0]

    # Warm-up call (JIT compilation)
    _ = prob.fitness(x)

    # 20 calls timing
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = prob.fitness(x)
        timings.append(time.perf_counter() - t0)

    median_t = float(np.median(timings))
    print(f"\n[S3 TIMING REPORT]")
    print(f"One full fitness evaluation on train (720k ratings, d=10): {median_t*1000:.2f} ms")
    assert median_t < 1.0  # Should be well under 1 second with Numba parallel
