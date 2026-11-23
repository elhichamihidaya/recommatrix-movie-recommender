"""Data loading, representation, splitting and subsampling for MovieLens 1M."""

import sys
from pathlib import Path

# Ensure project root is in sys.path if run directly as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Ratings:
    users: np.ndarray      # int32, contiguous ids 0..n_users-1
    items: np.ndarray      # int32, contiguous ids 0..n_items-1
    ratings: np.ndarray    # float32
    n_users: int
    n_items: int

    def __len__(self) -> int:
        return len(self.users)

    def subset(self, idx: np.ndarray) -> "Ratings":
        """Return a subset of ratings keeping the same global n_users and n_items."""
        return Ratings(
            users=self.users[idx],
            items=self.items[idx],
            ratings=self.ratings[idx],
            n_users=self.n_users,
            n_items=self.n_items,
        )

    def to_csr(self) -> csr_matrix:
        """Return a scipy CSR sparse matrix of shape (n_users, n_items)."""
        return csr_matrix(
            (self.ratings, (self.users, self.items)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )


def load_movielens_1m(data_dir: str = "data") -> Ratings:
    """Auto-detect and load the MovieLens 1M ratings file, re-indexing users and items."""
    path = Path(data_dir)
    candidate_paths = [
        path / "ratings.dat",
        path / "ml-1m" / "ratings.dat",
        path / "ratings.csv",
        path / "ml-1m" / "ratings.csv",
    ]

    found_path = None
    for cand in candidate_paths:
        if cand.is_file():
            found_path = cand
            break

    if found_path is None:
        raise FileNotFoundError(f"MovieLens 1M ratings file not found in {data_dir}")

    if found_path.suffix == ".dat":
        df = pd.read_csv(
            found_path,
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            usecols=["user_id", "item_id", "rating"],
        )
    else:
        df = pd.read_csv(
            found_path,
            usecols=["userId", "movieId", "rating"] if "userId" in pd.read_csv(found_path, nrows=1).columns else ["user_id", "item_id", "rating"],
        )
        if "userId" in df.columns:
            df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})

    # Re-index with sorted original ids to guarantee deterministic contiguous indexing 0..N-1
    unique_users, user_indices = np.unique(df["user_id"].to_numpy(), return_inverse=True)
    unique_items, item_indices = np.unique(df["item_id"].to_numpy(), return_inverse=True)

    users = user_indices.astype(np.int32)
    items = item_indices.astype(np.int32)
    ratings = df["rating"].to_numpy(dtype=np.float32)

    return Ratings(
        users=users,
        items=items,
        ratings=ratings,
        n_users=len(unique_users),
        n_items=len(unique_items),
    )


def _split_per_user(users: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask selecting about `frac` of each user's ratings (at least 1)."""
    n = len(users)
    order = np.lexsort((rng.random(n), users))        # sorted by user, random order inside each user
    counts = np.bincount(users)
    starts = np.cumsum(counts) - counts
    rank = np.arange(n) - np.repeat(starts, counts)   # rank inside the user's block
    take = rank < np.repeat(np.maximum(1, np.round(frac * counts)).astype(int), counts)
    mask = np.zeros(n, dtype=bool)
    mask[order[take]] = True
    return mask


@dataclass(frozen=True)
class Split:
    train: Ratings
    val: Ratings
    train_full: Ratings
    test: Ratings


def get_split(ratings: Ratings, seed: int = 42, test_frac: float = 0.2, val_frac: float = 0.1) -> Split:
    """Per-user deterministic train / val / train_full / test split."""
    # Step 1: select test set from all ratings
    rng_test = np.random.default_rng(seed)
    test_mask = _split_per_user(ratings.users, test_frac, rng_test)

    train_full_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Step 2: select val set from train_full ratings
    rng_val = np.random.default_rng(seed + 1)
    val_submask = _split_per_user(ratings.users[train_full_idx], val_frac, rng_val)

    val_idx = train_full_idx[val_submask]
    train_idx = train_full_idx[~val_submask]

    return Split(
        train=ratings.subset(train_idx),
        val=ratings.subset(val_idx),
        train_full=ratings.subset(train_full_idx),
        test=ratings.subset(test_idx),
    )


def make_subsample(ratings: Ratings, n_users: int = 500, n_items: int = 500, seed: int = 0) -> Ratings:
    """Create a smaller subsample for fast tests / demos: n_items most-rated + random users, re-indexed."""
    # Find n_items most-rated items
    item_counts = np.bincount(ratings.items, minlength=ratings.n_items)
    k_items = min(n_items, ratings.n_items)
    top_items = np.argsort(item_counts)[::-1][:k_items]
    top_items_set = set(top_items)

    # Random users
    unique_users = np.unique(ratings.users)
    k_users = min(n_users, len(unique_users))
    rng = np.random.default_rng(seed)
    chosen_users = rng.choice(unique_users, size=k_users, replace=False)
    chosen_users_set = set(chosen_users)

    # Filter mask
    mask = np.array([u in chosen_users_set and i in top_items_set for u, i in zip(ratings.users, ratings.items)], dtype=bool)
    sub_users_raw = ratings.users[mask]
    sub_items_raw = ratings.items[mask]
    sub_ratings = ratings.ratings[mask]

    # Re-index to contiguous 0..N-1
    u_unq, u_idx = np.unique(sub_users_raw, return_inverse=True)
    i_unq, i_idx = np.unique(sub_items_raw, return_inverse=True)

    return Ratings(
        users=u_idx.astype(np.int32),
        items=i_idx.astype(np.int32),
        ratings=sub_ratings.astype(np.float32),
        n_users=len(u_unq),
        n_items=len(i_unq),
    )


if __name__ == "__main__":
    print("Loading MovieLens 1M data...")
    ratings = load_movielens_1m()
    print(f"Loaded: {len(ratings)} ratings, {ratings.n_users} users, {ratings.n_items} items.")
    split = get_split(ratings)
    print(f"Split sizes - Train: {len(split.train)}, Val: {len(split.val)}, Test: {len(split.test)}")

