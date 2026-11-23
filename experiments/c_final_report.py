"""Build the full five-seed benchmark tables and final comparison figures."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    results = ROOT / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    files = [
        results / "baselines.csv",
        results / "metaheuristics.csv",
        results / "a_hybrid_compare.csv",
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)

    frames = [pd.read_csv(path) for path in files]
    all_results = pd.concat(frames, ignore_index=True)
    all_results = all_results.sort_values(["method", "seed"]).reset_index(drop=True)
    expected_methods = {
        "als", "sgd", "bpr", "global_mean", "user_mean", "item_mean",
        "popularity", "pso", "pso+als", "als_reference_10it", "als_reference_30it",
    }
    found_methods = set(all_results["method"].dropna())
    if found_methods != expected_methods:
        raise ValueError(f"Unexpected benchmark methods: {sorted(found_methods)}")
    counts = all_results.groupby("method")["seed"].nunique()
    if not (counts == 5).all():
        raise ValueError(f"Every method must have five seeds; found {counts.to_dict()}")

    all_path = results / "benchmark_all.csv"
    all_results.to_csv(all_path, index=False)
    metrics = [
        "rmse_test", "mae_test", "prec@5", "prec@10", "prec@20",
        "rec@5", "rec@10", "rec@20", "fit_time_s", "peak_mem_mb", "n_evals",
    ]
    grouped = all_results.groupby("method", sort=False)[metrics]
    means = grouped.mean().add_suffix("_mean")
    std_values = grouped.std(ddof=1)
    std_values = std_values.mask(grouped.count().eq(1), 0.0)
    stds = std_values.add_suffix("_std")
    summary = pd.concat([means, stds], axis=1).reset_index()
    summary.insert(1, "n_seeds", all_results.groupby("method").seed.nunique().reindex(summary.method).to_numpy())
    summary.to_csv(results / "benchmark_summary.csv", index=False)

    pretty = {
        "als_reference_10it": "ALS reference (10 iter)",
        "als_reference_30it": "ALS reference (30 iter)",
        "global_mean": "Global mean",
        "user_mean": "User mean",
        "item_mean": "Item mean",
        "pso+als": "PSO + ALS",
    }
    plot_summary = summary.dropna(subset=["rmse_test_mean"]).copy()
    plot_summary["label"] = plot_summary.method.map(lambda value: pretty.get(value, value.upper()))
    plot_summary = plot_summary.sort_values("rmse_test_mean")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    y = np.arange(len(plot_summary))
    axes[0].barh(y, plot_summary.rmse_test_mean, color="#3977a8")
    axes[0].errorbar(
        plot_summary.rmse_test_mean, y,
        xerr=plot_summary.rmse_test_std.fillna(0), fmt="none", ecolor="#222222", capsize=3,
    )
    axes[0].set_yticks(y, plot_summary.label)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test RMSE (mean ± std)")
    axes[0].set_title("Rating prediction quality")
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)

    ranking = plot_summary.dropna(subset=["prec@10_mean"]).sort_values("prec@10_mean")
    ry = np.arange(len(ranking))
    axes[1].barh(ry, ranking["prec@10_mean"], color="#55a868")
    axes[1].errorbar(
        ranking["prec@10_mean"], ry,
        xerr=ranking["prec@10_std"].fillna(0), fmt="none", ecolor="#222222", capsize=3,
    )
    axes[1].set_yticks(ry, ranking.label)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Precision@10 (mean ± std)")
    axes[1].set_title("Top-10 recommendation quality")
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)
    fig.suptitle("MovieLens 1M benchmark — five seeds (0–4)")
    fig.tight_layout()
    fig.savefig(figures / "benchmark_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Update the PSO convergence figure to include all five final seeds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for seed in range(5):
        path = results / "convergence" / f"pso_seed{seed}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        history = pd.read_csv(path)
        axes[0].plot(history["n_evals"], history["gbest_fitness"], label=f"Seed {seed}")
        axes[1].plot(history["time_s"], history["rmse_train"], label=f"Seed {seed}")
    axes[0].set(xlabel="Fitness evaluations", ylabel="Global best fitness", title="PSO convergence")
    axes[1].set(xlabel="Elapsed time (s)", ylabel="Train RMSE", title="PSO training RMSE")
    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "convergence_pso.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "# Final Benchmark Summary — MovieLens 1M",
        "",
        "Results use the deterministic per-user split (seed 42), final training on train_full, and evaluation on the held-out test set. Each method has five runs (seeds 0–4). Ranking relevance is a test rating >= 4; training-seen movies are excluded from recommendations.",
        "",
        "| Method | Seeds | Test RMSE | Test MAE | P@10 | R@10 | Fit time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        def fmt(metric: str, digits: int = 4) -> str:
            mean = row.get(f"{metric}_mean", np.nan)
            std = row.get(f"{metric}_std", np.nan)
            if pd.isna(mean):
                return "—"
            return f"{mean:.{digits}f} ± {std:.{digits}f}"

        report_lines.append(
            f"| {pretty.get(row.method, row.method.upper())} | {int(row.n_seeds)} | "
            f"{fmt('rmse_test')} | {fmt('mae_test')} | {fmt('prec@10')} | "
            f"{fmt('rec@10')} | {fmt('fit_time_s', 2)} |"
        )
    report_lines.extend([
        "",
        "The ranking metrics are averaged over users with at least one relevant test movie. The ALS reference ranking values were computed by fitting the same reference models on train_full for each seed.",
        "",
        "See `benchmark_all.csv` for per-seed results and `figures/benchmark_comparison.png` for the comparison plot.",
    ])
    report = "\n".join(report_lines) + "\n"
    (results / "benchmark_summary.md").write_text(report, encoding="utf-8")

    print(f"Combined rows: {len(all_results)}")
    print(f"Methods: {len(expected_methods)} × 5 seeds")
    print(f"Wrote {all_path}")
    print(f"Wrote {results / 'benchmark_summary.csv'}")
    print(f"Wrote {results / 'benchmark_summary.md'}")
    print(f"Wrote {figures / 'benchmark_comparison.png'} and {figures / 'convergence_pso.png'}")


if __name__ == "__main__":
    main()
