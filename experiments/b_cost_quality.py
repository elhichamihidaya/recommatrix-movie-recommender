"""Generate Part B9 cost-versus-quality figures, statistics, and analysis."""

import json
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

from src.benchmark import paired_seed_ttest


LABELS = {
    "als": "ALS",
    "als_reference_10it": "ALS ref. 10",
    "als_reference_30it": "ALS ref. 30",
    "bpr": "BPR",
    "global_mean": "Global mean",
    "item_mean": "Item mean",
    "popularity": "Popularity",
    "pso": "PSO",
    "pso+als": "PSO + ALS",
    "sgd": "SGD",
    "user_mean": "User mean",
}


def load_and_reconcile(root: Path) -> pd.DataFrame:
    results = root / "results"
    meta_path = results / "metaheuristics.csv"
    meta_backup = results / "metaheuristics_before_b9_correction.csv"
    meta = pd.read_csv(meta_path)

    # The saved Hybrid histories are authoritative about the executed ALS passes.
    # Earlier rows for seeds 3-4 inherited the tuning config's 20 iterations,
    # although a_final.py actually ran the refiner for 10 iterations.
    for seed in range(5):
        hist_path = results / "convergence" / f"pso+als_seed{seed}.csv"
        if not hist_path.is_file():
            raise FileNotFoundError(hist_path)
        hist = pd.read_csv(hist_path)
        als_hist = hist[hist["phase"] == "als"]
        if als_hist.empty:
            raise ValueError(f"No ALS phase in {hist_path}")
        actual_als_iterations = int(als_hist["iteration"].max())
        mask = (meta["method"] == "pso+als") & (meta["seed"] == seed)
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected one PSO+ALS row for seed {seed}")
        params = json.loads(meta.loc[mask, "hyperparams"].iloc[0])
        params["als"] = {"d": 10, "lam": 0.05, "n_iter": actual_als_iterations}
        meta.loc[mask, "hyperparams"] = json.dumps(params, sort_keys=True)

    if not meta_backup.exists():
        meta_backup.write_bytes(meta_path.read_bytes())
    meta.to_csv(meta_path, index=False)

    frames = [
        pd.read_csv(results / "baselines.csv"),
        meta,
        pd.read_csv(results / "a_hybrid_compare.csv"),
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values(["method", "seed"])
    seed_counts = combined.groupby("method")["seed"].nunique()
    if len(seed_counts) != 11 or not (seed_counts == 5).all():
        raise ValueError(f"Expected 11 methods with five seeds each; found {seed_counts.to_dict()}")
    combined.to_csv(results / "benchmark_all.csv", index=False)
    return combined


def config_object(value) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value) or not value:
        return {}
    return json.loads(value)


def actual_cycles(root: Path, row: pd.Series) -> float:
    method, seed = row["method"], int(row["seed"])
    if method in {"global_mean", "user_mean", "item_mean", "popularity"}:
        return 0.0
    if method in {"als", "als_reference_10it", "als_reference_30it"}:
        path = root / "results" / "histories" / f"als_als_cfg_12_seed{seed}.csv" if method == "als" else None
        if path and path.exists():
            return float(pd.read_csv(path)["iteration"].max())
        params = config_object(row["hyperparams"])
        return float(params.get("n_iter", 10 if method.endswith("10it") else 30))
    if method in {"sgd", "bpr"}:
        params = config_object(row["hyperparams"])
        return float(params.get("n_epochs", 0))
    if method == "pso":
        path = root / "results" / "convergence" / f"pso_seed{seed}.csv"
        return float(pd.read_csv(path)["iteration"].max())
    if method == "pso+als":
        path = root / "results" / "convergence" / f"pso+als_seed{seed}.csv"
        history = pd.read_csv(path)
        pso_cycles = history.loc[history.phase == "pso", "iteration"].max()
        als_cycles = history.loc[history.phase == "als", "iteration"].max()
        return float(pso_cycles + als_cycles)
    raise ValueError(f"No cycle count mapping for {method}")


def aggregate(combined: pd.DataFrame, root: Path) -> pd.DataFrame:
    metrics = ["rmse_test", "mae_test", "prec@10", "rec@10", "fit_time_s", "peak_mem_mb", "n_evals"]
    grouped = combined.groupby("method", sort=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    std_values = grouped.std(ddof=1)
    std_values = std_values.mask(grouped.count().eq(1), 0.0)
    std = std_values.add_suffix("_std")
    summary = pd.concat([mean, std], axis=1).reset_index()
    summary["training_cycles_mean"] = [
        combined.loc[combined.method == method].apply(lambda row: actual_cycles(root, row), axis=1).mean()
        for method in summary.method
    ]
    summary["training_cycles_std"] = [
        combined.loc[combined.method == method].apply(lambda row: actual_cycles(root, row), axis=1).std(ddof=1)
        for method in summary.method
    ]
    summary["n_seeds"] = combined.groupby("method").seed.nunique().reindex(summary.method).to_numpy()
    summary["label"] = summary.method.map(LABELS)
    summary.to_csv(root / "results" / "B9_cost_quality_summary.csv", index=False)
    return summary


def plot_metric_vs_cost(summary: pd.DataFrame, x_mean: str, x_std: str, x_label: str, output: Path, log_x=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(summary)))
    for color, (_, row) in zip(colors, summary.iterrows()):
        x = row[x_mean]
        xerr = row[x_std]
        if pd.notna(row["rmse_test_mean"]):
            axes[0].errorbar(x, row["rmse_test_mean"], xerr=xerr, yerr=row["rmse_test_std"],
                             fmt="o", color=color, capsize=3, label=row["label"])
        if pd.notna(row["prec@10_mean"]):
            axes[1].errorbar(x, row["prec@10_mean"], xerr=xerr, yerr=row["prec@10_std"],
                             fmt="o", color=color, capsize=3, label=row["label"])
    axes[0].set(ylabel="Test RMSE (lower is better)", title="Rating prediction")
    axes[1].set(ylabel="Precision@10 (higher is better)", title="Top-10 ranking")
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.grid(True, linestyle=":", alpha=0.5)
        if log_x:
            ax.set_xscale("log")
    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("MovieLens 1M: mean ± standard deviation over five seeds")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pareto_mask(x: np.ndarray, y: np.ndarray, maximize_y: bool) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        for j in range(len(x)):
            better_or_equal = x[j] <= x[i] and (y[j] >= y[i] if maximize_y else y[j] <= y[i])
            strictly_better = x[j] < x[i] or (y[j] > y[i] if maximize_y else y[j] < y[i])
            if j != i and better_or_equal and strictly_better:
                keep[i] = False
                break
    return keep


def plot_pareto(summary: pd.DataFrame, output: Path) -> dict[str, list[str]]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fronts = {}
    for ax, metric, maximize in [
        (axes[0], "rmse_test_mean", False),
        (axes[1], "prec@10_mean", True),
    ]:
        valid = summary.dropna(subset=[metric, "fit_time_s_mean"]).copy()
        x = valid.fit_time_s_mean.to_numpy(float)
        y = valid[metric].to_numpy(float)
        mask = pareto_mask(x, y, maximize)
        fronts[metric] = valid.loc[mask, "method"].tolist()
        ax.scatter(x, y, c="#3b6ea8", s=55, zorder=3)
        ax.scatter(x[mask], y[mask], c="#d65f5f", s=85, edgecolor="black", zorder=4, label="Pareto front")
        for xx, yy, label in zip(x, y, valid.label):
            ax.annotate(label, (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
        if mask.sum() > 1:
            ix = np.argsort(x[mask])
            ax.plot(x[mask][ix], y[mask][ix], color="#d65f5f", linestyle="--", alpha=0.75)
        ax.set_xscale("log")
        ax.set_xlabel("Mean fit time (seconds, log scale)")
        ax.set_ylabel("Test RMSE (lower is better)" if not maximize else "Precision@10 (higher is better)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
    fig.suptitle("Quality versus computational cost: time-quality Pareto fronts")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fronts


def main() -> None:
    results = ROOT / "results"
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = load_and_reconcile(ROOT)
    summary = aggregate(combined, ROOT)

    plot_metric_vs_cost(summary, "fit_time_s_mean", "fit_time_s_std", "Mean fit time in seconds (log scale)",
                        figures / "B9_quality_vs_time.png", log_x=True)
    plot_metric_vs_cost(summary, "peak_mem_mb_mean", "peak_mem_mb_std", "Peak memory in MB",
                        figures / "B9_quality_vs_memory.png")
    plot_metric_vs_cost(summary, "training_cycles_mean", "training_cycles_std",
                        "Configured cycles (epoch / ALS iteration / PSO generation)",
                        figures / "B9_quality_vs_cycles.png")
    fronts = plot_pareto(summary, figures / "B9_pareto_fronts.png")

    comparisons = [
        ("als", "pso+als", "rmse_test"),
        ("als", "als_reference_30it", "rmse_test"),
        ("bpr", "popularity", "prec@10"),
        ("als", "sgd", "rmse_test"),
    ]
    tests = []
    for first, second, metric in comparisons:
        try:
            tests.append(paired_seed_ttest(combined, first, second, metric))
        except ValueError as exc:
            tests.append({"method_a": first, "method_b": second, "metric": metric, "error": str(exc)})
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(results / "B9_paired_tests.csv", index=False)

    by_method = summary.set_index("method")
    def value(method: str, metric: str, decimals: int = 4) -> str:
        row = by_method.loc[method]
        return f"{row[metric + '_mean']:.{decimals}f} ± {row[metric + '_std']:.{decimals}f}"

    stat_lines = []
    for _, row in tests_df.iterrows():
        if pd.notna(row.get("p_value")):
            stat_lines.append(
                f"- Test t apparié {row.method_a} vs {row.method_b} sur `{row.metric}` : "
                f"n={int(row.n_paired_seeds)}, t={row.t_statistic:.3f}, p={row.p_value:.4g}."
            )

    report = f'''# B9 — Analyse du coût computationnel et de la qualité

## Protocole et lecture des résultats

Les mesures ci-dessous proviennent du benchmark MovieLens 1M : 1 000 209 notes au total, 6 040 utilisateurs et 3 706 films. Le split par utilisateur utilise la seed 42 ; le réglage utilise la validation, les entraînements finaux sont faits sur `train_full`, et le jeu de test contient 200 016 notes. Toutes les méthodes finales sont présentes sur cinq seeds (0–4). Les valeurs sont présentées en moyenne ± écart-type. Les temps et la mémoire reflètent cette exécution sur une machine Windows/Anaconda ; ils dépendent du matériel, du système, des bibliothèques et de la charge en arrière-plan.

Pour le ranking, une note de test ≥ 4 est pertinente, les films déjà vus en entraînement sont exclus et la moyenne porte sur les utilisateurs ayant au moins un film pertinent dans le test. BPR et Popularity sont des méthodes de classement : leur RMSE/MAE n'est pas défini. Les métriques ranking des deux ALS de référence sont calculées à partir de modèles réentraînés sur le même `train_full`.

## Résultats observés

| Méthode | RMSE test | P@10 | Temps moyen (s) | Mémoire de crête moyenne (MB) | Cycles configurés |
|---|---:|---:|---:|---:|---:|
'''
    for method in summary.method:
        row = by_method.loc[method]
        rmse_text = value(method, "rmse_test") if pd.notna(row.rmse_test_mean) else "—"
        p10_text = value(method, "prec@10") if pd.notna(row["prec@10_mean"]) else "—"
        report += f"| {LABELS[method]} | {rmse_text} | {p10_text} | {row.fit_time_s_mean:.2f} ± {row.fit_time_s_std:.2f} | {row.peak_mem_mb_mean:.1f} ± {row.peak_mem_mb_std:.1f} | {row.training_cycles_mean:g} |\n"

    report += f'''\nThe fit-time plots use a logarithmic time axis to show both the very short mean/popularity baselines and the longer latent-factor methods. The memory plots compare measured peak RSS above each process baseline. Cycle counts use the model's own unit: SGD/BPR epochs, ALS sweeps, PSO generations, and the sum of PSO generations plus ALS sweeps for the hybrid. Those units describe workload within each algorithm and should not be interpreted as identical operations. The zero-cycle simple baselines are closed-form aggregation methods.

### Theoretical work per iteration

Let Ω be the training ratings, with |\u03a9| = 800,193, users n_u = 6,040, items n_i = 3,706, latent dimension d, and PSO population N. ALS-WR solves one d-by-d system per user and item: approximately O(|\u03a9| d² + (n_u+n_i)d³) per full ALS sweep; storage is O(|\u03a9|+(n_u+n_i)d). This matches its relatively high measured wall time for d=10 and 20 sweeps (27.31 s mean for the tuned Part B ALS).

SGD visits each observed rating and updates two d-dimensional vectors: O(|\u03a9|d) per epoch, with O(|\u03a9|+(n_u+n_i)d) storage. Its measured 10-epoch fit averages 5.20 s and 197.8 MB. It has a good rating RMSE (0.8760) and stronger P@10 (0.0962) than the non-personalized rating baselines, while using more measured memory than the other fitted models in this run.

BPR samples positive/negative user-item triplets. For S sampled triplets per epoch, its work is O(Sd) per epoch and its factor/storage requirement is O((n_u+n_i)d), plus sampling/index structures. It is ranking-only by design. At the tuned configuration its mean P@10 is 0.2194, above Popularity at 0.1514, with 2.17 s mean fit time and 33.0 MB peak memory. Thus it is the strongest top-10 ranking method in these measurements, at substantially lower time than ALS or PSO. Its quality should be judged only by ranking metrics.

PSO evaluates a population of candidate factor matrices. A full fitness evaluation costs O(|\u03a9|d); E evaluations therefore cost O(E|\u03a9|d). The population, velocities and best-so-far states require O(N(n_u+n_i)d) memory. At E=6,000 and N=20, the pure PSO averages 28.52 s and 104.2 MB, yet its RMSE is 1.1159 and P@10 is 0.0050. It has a high computational cost without competitive predictive or ranking quality on this large search space.

The PSO+ALS hybrid combines the PSO evaluation cost with ten ALS sweeps, as confirmed by the saved per-seed histories. It averages 35.60 s, 97.3 MB and RMSE 0.8609, with P@10 0.0212. ALS alone averages RMSE 0.8582 and P@10 0.0212 at 27.31 s. On these metrics, the PSO stage adds cost without a measurable quality advantage over tuned ALS. The hybrid is far better than PSO alone, showing that ALS refinement accounts for most of its rating and ranking quality.

The global/user/item means and popularity are closed-form baselines with one pass over ratings or counts, O(|\u03a9|+n_u+n_i) work and small state. They fit in about 0.10–0.11 s and under 8 MB peak memory here. They are useful cost floors. Global mean reaches RMSE 1.1158, while popularity reaches P@10 0.1514; a baseline can therefore be weak on one task and competitive on another.

## Pareto and statistical interpretation

On the RMSE/time plane, the Pareto front contains: {', '.join(LABELS[m] for m in fronts['rmse_test_mean'])}. On the P@10/time plane, it contains: {', '.join(LABELS[m] for m in fronts['prec@10_mean'])}. A method is on the front if no other measured method is both faster and at least as good on that metric. This is a descriptive comparison of five-seed averages, not a guarantee for every seed. The separate time and memory figures show that the trade-off changes with the selected objective: BPR is particularly effective for ranking, while ALS/SGD are stronger choices for calibrated rating prediction.

Paired t-tests are exploratory because they use only five paired seeds and assume the seed-wise differences are approximately normal:
{chr(10).join(stat_lines)}

The test results do not replace effect sizes or confidence intervals, and a non-significant result at n=5 is not evidence of equivalence. A practical choice follows the product objective: use BPR when top-k ranking is primary; use ALS or SGD for rating prediction; use simple baselines as cheap reference points. The current measurements do not justify pure PSO for the full MovieLens 1M factorization, and do not show a quality gain from adding PSO before ALS. A metaheuristic's extra cost could be justified when it yields a robust improvement on the chosen metric, solves constraints unavailable to the baseline optimizer, or works better on a smaller search domain; none of those advantages is established by this benchmark.

## Generated artifacts

- `results/figures/B9_quality_vs_time.png`
- `results/figures/B9_quality_vs_memory.png`
- `results/figures/B9_quality_vs_cycles.png`
- `results/figures/B9_pareto_fronts.png`
- `results/B9_cost_quality_summary.csv`
- `results/B9_paired_tests.csv`

The cycle counts for PSO+ALS and the recorded configuration metadata were reconciled against the saved convergence histories. The pre-correction input CSV is retained as `results/metaheuristics_before_b9_correction.csv`.
'''
    (results / "B9_cost_quality_analysis.md").write_text(report, encoding="utf-8")
    print(f"B9 complete: {len(combined)} rows; {len(summary)} methods; 5 seeds each")
    print("Wrote B9 analysis, summary, paired tests, and four figures.")
    print("Pareto RMSE/time:", ", ".join(fronts["rmse_test_mean"]))
    print("Pareto P@10/time:", ", ".join(fronts["prec@10_mean"]))


if __name__ == "__main__":
    main()
