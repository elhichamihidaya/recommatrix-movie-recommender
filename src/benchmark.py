"""Experiment runner and result aggregation for Part B benchmarks."""

import sys
from dataclasses import fields
import json
from pathlib import Path
import re

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.als import ALS
from src.bpr import BPRRecommender
from src.hybrid import HybridRecommender
from src.metrics import mae, ranking_metrics, rmse
from src.protocol import RESULT_COLUMNS, append_result, measure
from src.pso import PSOConfig, PSORecommender
from src.sgd import SGDRecommender
from src.simple import (
    GlobalMeanRecommender,
    ItemMeanRecommender,
    PopularityRecommender,
    UserMeanRecommender,
)


RANKING_ONLY_METHODS = {"bpr", "popularity"}
METHOD_ALIASES = {"global": "global_mean", "user": "user_mean", "item": "item_mean"}


def _build_model(method: str, config: dict, seed: int):
    """Create one model from a method name and its hyperparameters."""
    method = METHOD_ALIASES.get(method.lower(), method.lower())
    params = dict(config)
    params.pop("variant", None)
    params["seed"] = seed

    if method == "als":
        return ALS(**{k: params[k] for k in ("d", "lam", "n_iter", "seed") if k in params})
    if method == "sgd":
        allowed = {"d", "eta", "lam", "n_epochs", "seed", "lr_decay", "patience", "min_delta"}
        return SGDRecommender(**{k: v for k, v in params.items() if k in allowed})
    if method == "bpr":
        allowed = {"d", "eta", "lam", "n_epochs", "n_samples_per_epoch", "seed", "relevant_threshold", "lr_decay"}
        return BPRRecommender(**{k: v for k, v in params.items() if k in allowed})
    if method == "global_mean":
        return GlobalMeanRecommender()
    if method == "user_mean":
        return UserMeanRecommender()
    if method == "item_mean":
        return ItemMeanRecommender()
    if method == "popularity":
        return PopularityRecommender(relevant_threshold=params.get("relevant_threshold", 4.0))
    if method == "pso":
        allowed = {field.name for field in fields(PSOConfig)}
        return PSORecommender(PSOConfig(**{k: v for k, v in params.items() if k in allowed}))
    if method == "hybrid":
        pso_params = dict(config.get("pso", {}))
        als_params = dict(config.get("als", {}))
        pso_params["seed"] = seed
        als_params["seed"] = seed
        allowed_pso = {field.name for field in fields(PSOConfig)}
        pso = PSORecommender(PSOConfig(**{k: v for k, v in pso_params.items() if k in allowed_pso}))
        als = ALS(**{k: als_params[k] for k in ("d", "lam", "n_iter", "seed") if k in als_params})
        return HybridRecommender(meta=pso, refiner=als)
    raise ValueError(f"Unknown benchmark method: {method}")


def run_experiment(
    method: str,
    config: dict,
    seed: int,
    split,
    csv_path=None,
    phase: str = "final",
    history_dir=None,
) -> dict:
    """Fit and evaluate one method, returning one row in the common CSV schema.

    ``phase="validation"`` trains on ``split.train`` and evaluates on
    ``split.val`` for tuning. ``phase="final"`` retrains on ``split.train_full``
    and evaluates on ``split.test``. This keeps validation out of final training
    metrics and keeps the test set out of tuning.
    """
    if phase not in {"validation", "final"}:
        raise ValueError("phase must be 'validation' or 'final'.")
    method_key = METHOD_ALIASES.get(method.lower(), method.lower())
    model = _build_model(method_key, config, seed)
    fit_data = split.train if phase == "validation" else split.train_full
    evaluation_data = split.val if phase == "validation" else split.test
    with measure() as measurements:
        model.fit(fit_data, val=split.val if phase == "validation" else None)

    if history_dir is not None:
        history_path = Path(history_dir)
        history_path.mkdir(parents=True, exist_ok=True)
        variant_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.get("variant", "default")))
        history = model.history_df() if hasattr(model, "history_df") else pd.DataFrame(model.history)
        history.to_csv(history_path / f"{method_key}_{variant_slug}_seed{seed}.csv", index=False)

    if method_key in RANKING_ONLY_METHODS:
        rmse_val = np.nan
        rmse_test = np.nan
        mae_test = np.nan
    else:
        eval_predictions = model.predict(evaluation_data.users, evaluation_data.items)
        eval_rmse = rmse(evaluation_data.ratings, eval_predictions)
        if phase == "validation":
            rmse_val = eval_rmse
            rmse_test = np.nan
            mae_test = np.nan
        else:
            selected_val = config.get("selection_score", config.get("val_rmse", np.nan))
            rmse_val = selected_val if config.get("selection_metric", "rmse_val") == "rmse_val" else np.nan
            rmse_test = eval_rmse
            mae_test = mae(evaluation_data.ratings, eval_predictions)

    ranking_values = ranking_metrics(model, fit_data, evaluation_data)
    ranking = {}
    for k in (5, 10, 20):
        ranking[f"prec@{k}"] = ranking_values[f"precision@{k}"]
        ranking[f"rec@{k}"] = ranking_values[f"recall@{k}"]
    saved_config = {k: v for k, v in config.items() if k != "variant"}
    row = {
        "method": method_key,
        "variant": config.get("variant", "tuning" if phase == "validation" else "final"),
        "seed": int(seed),
        "d": getattr(model, "d", config.get("d", config.get("pso", {}).get("d", np.nan))),
        "hyperparams": json.dumps(saved_config, sort_keys=True),
        "rmse_val": rmse_val,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        **ranking,
        "fit_time_s": measurements["time_s"],
        "peak_mem_mb": measurements["peak_mem_mb"],
        "n_evals": int(getattr(model, "n_evals", 0)),
    }
    row = {column: row.get(column, np.nan) for column in RESULT_COLUMNS}
    if csv_path is not None:
        append_result(csv_path, row)
    return row


def run_seeds(
    method: str,
    config: dict,
    split,
    seeds=(0, 1, 2, 3, 4),
    csv_path=None,
    phase: str = "final",
    history_dir=None,
) -> pd.DataFrame:
    """Run the same configuration for the five assignment seeds by default."""
    rows = [
        run_experiment(
            method,
            config,
            seed,
            split,
            csv_path=csv_path,
            phase=phase,
            history_dir=history_dir,
        )
        for seed in seeds
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def merge_result_files(baselines_csv, metaheuristics_csv, output_csv) -> pd.DataFrame:
    """Merge the raw Part B and Part A CSV files into one benchmark table."""
    frames = [pd.read_csv(path) for path in (baselines_csv, metaheuristics_csv)]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.reindex(columns=RESULT_COLUMNS)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined


def build_benchmark_summary(
    baselines_csv,
    metaheuristics_csv,
    output_csv="results/benchmark_summary.csv",
    raw_output_csv="results/benchmark_all.csv",
) -> pd.DataFrame:
    """Merge both parts' raw results and save a mean/std summary CSV."""
    combined = merge_result_files(baselines_csv, metaheuristics_csv, raw_output_csv)
    summary = summarize_results(combined)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs as mean and sample standard deviation."""
    group_columns = ["method", "variant", "d", "hyperparams"]
    metric_columns = [
        "rmse_val", "rmse_test", "mae_test", "prec@5", "prec@10", "prec@20",
        "rec@5", "rec@10", "rec@20", "fit_time_s", "peak_mem_mb", "n_evals",
    ]
    available_groups = [column for column in group_columns if column in results.columns]
    available_metrics = [column for column in metric_columns if column in results.columns]
    grouped = results.groupby(available_groups, dropna=False)[available_metrics]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    return pd.concat([means, stds], axis=1).reset_index()


def paired_seed_ttest(results: pd.DataFrame, method_a: str, method_b: str, metric: str = "rmse_test") -> dict:
    """Run a paired t-test over seeds shared by two methods for one metric."""
    if metric not in results.columns:
        raise ValueError(f"Unknown result metric: {metric}")
    paired = results.loc[results["method"].isin([method_a, method_b]), ["method", "seed", metric]]
    wide = paired.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
    if method_a not in wide or method_b not in wide:
        raise ValueError("Both methods must have results for the requested comparison.")
    wide = wide[[method_a, method_b]].dropna()
    if len(wide) < 2:
        raise ValueError("At least two paired seeds are required for a paired t-test.")
    statistic, p_value = ttest_rel(wide[method_a], wide[method_b])
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_paired_seeds": int(len(wide)),
        "t_statistic": float(statistic),
        "p_value": float(p_value),
    }
