"""Tests for data loading, splitting and subsampling (MovieLens 1M)."""

import numpy as np
import pytest
from src.data import load_movielens_1m, get_split, make_subsample, Ratings


@pytest.fixture(scope="session")
def ml1m():
    return load_movielens_1m("data")


def test_load_movielens_1m(ml1m):
    assert ml1m.n_users == 6040
    assert ml1m.n_items == 3706
    assert len(ml1m) == 1000209
    assert ml1m.users.dtype == np.int32
    assert ml1m.items.dtype == np.int32
    assert ml1m.ratings.dtype == np.float32
    assert set(np.unique(ml1m.ratings)).issubset({1.0, 2.0, 3.0, 4.0, 5.0})
    assert ml1m.users.min() == 0 and ml1m.users.max() == 6039
    assert ml1m.items.min() == 0 and ml1m.items.max() == 3705


def test_get_split(ml1m):
    split = get_split(ml1m, seed=42, test_frac=0.2, val_frac=0.1)

    n_total = len(ml1m)
    n_train = len(split.train)
    n_val = len(split.val)
    n_tf = len(split.train_full)
    n_test = len(split.test)

    # Print exact split sizes for Member B alignment
    print(f"\n[EXACT SPLIT SIZES]")
    print(f"Total:      {n_total}")
    print(f"Train:      {n_train}")
    print(f"Val:        {n_val}")
    print(f"Train_full: {n_tf}")
    print(f"Test:       {n_test}")

    # Sizes approximate 720k / 80k / 200k
    assert 700000 < n_train < 740000
    assert 70000 < n_val < 90000
    assert 190000 < n_test < 210000

    # Union and partitions
    assert n_train + n_val == n_tf
    assert n_tf + n_test == n_total

    # Global dimensions preserved
    assert split.train.n_users == ml1m.n_users
    assert split.val.n_users == ml1m.n_users
    assert split.train_full.n_users == ml1m.n_users
    assert split.test.n_users == ml1m.n_users

    # Every user has >= 1 rating in train
    train_users_count = np.bincount(split.train.users, minlength=ml1m.n_users)
    assert np.all(train_users_count >= 1)

    # Determinism: same seed gives identical split
    split2 = get_split(ml1m, seed=42, test_frac=0.2, val_frac=0.1)
    np.testing.assert_array_equal(split.train.users, split2.train.users)
    np.testing.assert_array_equal(split.train.items, split2.train.items)
    np.testing.assert_array_equal(split.test.users, split2.test.users)
    np.testing.assert_array_equal(split.val.users, split2.val.users)


def test_make_subsample(ml1m):
    sub = make_subsample(ml1m, n_users=200, n_items=150, seed=42)
    assert sub.n_users <= 200
    assert sub.n_items <= 150
    assert len(sub) > 0
    assert sub.users.max() < sub.n_users
    assert sub.items.max() < sub.n_items
    csr = sub.to_csr()
    assert csr.shape == (sub.n_users, sub.n_items)
