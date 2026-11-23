# Métriques d'évaluation pour la factorisation matricielle (RMSE, MAE, et ranking).

import numpy as np
from scipy.sparse import csr_matrix


def _rating_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate rating arrays and clip predictions to the MovieLens scale."""
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}.")
    if true.size == 0:
        raise ValueError("Rating metrics need at least one observation.")
    return true, np.clip(pred, 1.0, 5.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE after clipping predicted ratings to the [1, 5] scale."""
    true, pred = _rating_arrays(y_true, y_pred)
    diff = true - pred
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE after clipping predicted ratings to the [1, 5] scale."""
    true, pred = _rating_arrays(y_true, y_pred)
    diff = true - pred
    return float(np.mean(np.abs(diff)))


def ranking_metrics(
    model,
    train,
    test,
    k_values: tuple[int, ...] = (5, 10, 20),
    relevant_threshold: float = 4.0,
    user_batch_size: int = 256,
) -> dict[str, float]:
    """Compute precision@k and recall@k, ranking only unseen movies.

    Users without a relevant test rating are omitted from both averages. A
    relevant rating is one whose value is at least ``relevant_threshold``.
    ``train`` and ``test`` are the shared Ratings objects from ``src.data``.
    """
    ks = tuple(int(k) for k in k_values)
    if not ks or any(k < 1 for k in ks):
        raise ValueError("k_values must contain positive integers.")
    if user_batch_size < 1:
        raise ValueError("user_batch_size must be positive.")
    if train.n_users != test.n_users or train.n_items != test.n_items:
        raise ValueError("train and test must use the same user and item index spaces.")

    train_csr = csr_matrix(
        (np.ones(len(train.users), dtype=np.uint8), (train.users, train.items)),
        shape=(train.n_users, train.n_items),
    )
    relevant_by_user: dict[int, set[int]] = {}
    relevant_mask = np.asarray(test.ratings) >= relevant_threshold
    for user, item in zip(test.users[relevant_mask], test.items[relevant_mask]):
        relevant_by_user.setdefault(int(user), set()).add(int(item))

    totals = {f"precision@{k}": 0.0 for k in ks}
    totals.update({f"recall@{k}": 0.0 for k in ks})
    eligible_users = sorted(relevant_by_user)
    if not eligible_users:
        return totals

    for start in range(0, len(eligible_users), user_batch_size):
        users = np.asarray(eligible_users[start : start + user_batch_size], dtype=np.int32)
        scores_batch = np.asarray(model.score_all(users))
        if scores_batch.shape != (len(users), train.n_items):
            raise ValueError("model.score_all(users) must return one score per user and item.")

        for row, user in enumerate(users):
            scores = scores_batch[row].copy()
            seen = train_csr.indices[train_csr.indptr[user] : train_csr.indptr[user + 1]]
            scores[seen] = -np.inf
            candidate_count = train.n_items - len(seen)
            relevant = relevant_by_user[int(user)]

            for k in ks:
                effective_k = min(k, candidate_count)
                if effective_k == 0:
                    continue
                top_items = np.argpartition(scores, train.n_items - effective_k)[-effective_k:]
                hits = sum(int(item) in relevant for item in top_items)
                totals[f"precision@{k}"] += hits / effective_k
                totals[f"recall@{k}"] += hits / len(relevant)

    return {name: value / len(eligible_users) for name, value in totals.items()}


def precision_at_k(model, train, test, k: int = 10, **kwargs) -> float:
    """Convenience wrapper returning precision at one k."""
    return ranking_metrics(model, train, test, k_values=(k,), **kwargs)[f"precision@{k}"]


def recall_at_k(model, train, test, k: int = 10, **kwargs) -> float:
    """Convenience wrapper returning recall at one k."""
    return ranking_metrics(model, train, test, k_values=(k,), **kwargs)[f"recall@{k}"]
