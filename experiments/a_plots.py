"""Generate convergence figures and A_summary.md table strictly from CSV results."""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import src  # Ensure DLL directory is configured for Windows OpenBLAS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_convergence_plots(conv_dir: Path, fig_dir: Path):
    """Plot PSO convergence: gbest fitness and train RMSE vs evals and vs time."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_file = fig_dir / "convergence_pso.png"

    seeds = [0, 1, 2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for s in seeds:
        csv_file = conv_dir / f"pso_seed{s}.csv"
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        # Left plot: gbest fitness vs n_evals
        axes[0].plot(df["n_evals"], df["gbest_fitness"], label=f"Seed {s}")
        # Right plot: train RMSE vs time
        axes[1].plot(df["time_s"], df["rmse_train"], label=f"Seed {s}")

    axes[0].set_xlabel("Fitness Evaluations")
    axes[0].set_ylabel("Global Best Fitness")
    axes[0].set_title("PSO Fitness Convergence")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_xlabel("Elapsed Time (s)")
    axes[1].set_ylabel("Train RMSE")
    axes[1].set_title("PSO Train RMSE vs Time")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Convergence plot saved: {out_file}")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format a pandas DataFrame as a GitHub-flavored Markdown table without tabulate."""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in df.columns) + " |"
        rows.append(row_str)
    return "\n".join([header_line, sep_line] + rows)


def generate_summary_md(results_dir: Path):
    """Generate results/A_summary.md strictly from CSVs."""
    meta_csv = results_dir / "metaheuristics.csv"
    als_csv = results_dir / "a_hybrid_compare.csv"

    dfs = []
    if meta_csv.exists() and meta_csv.stat().st_size > 0:
        dfs.append(pd.read_csv(meta_csv))
    if als_csv.exists() and als_csv.stat().st_size > 0:
        dfs.append(pd.read_csv(als_csv))

    if not dfs:
        print("No CSV results found to summarize.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    summary_rows = []
    for method, grp in df_all.groupby("method", sort=False):
        n_seeds = len(grp)
        rmse_mean, rmse_std = grp["rmse_test"].mean(), grp["rmse_test"].std(ddof=1) if n_seeds > 1 else 0.0
        mae_mean, mae_std = grp["mae_test"].mean(), grp["mae_test"].std(ddof=1) if n_seeds > 1 else 0.0
        time_mean, time_std = grp["fit_time_s"].mean(), grp["fit_time_s"].std(ddof=1) if n_seeds > 1 else 0.0
        mem_mean, mem_std = grp["peak_mem_mb"].mean(), grp["peak_mem_mb"].std(ddof=1) if n_seeds > 1 else 0.0
        evals_mean = grp["n_evals"].mean()

        summary_rows.append({
            "Method": method,
            "Seeds": n_seeds,
            "Test RMSE": f"{rmse_mean:.4f} ± {rmse_std:.4f}",
            "Test MAE": f"{mae_mean:.4f} ± {mae_std:.4f}",
            "Fit Time (s)": f"{time_mean:.2f} ± {time_std:.2f}",
            "Peak Mem (MB)": f"{mem_mean:.1f} ± {mem_std:.1f}",
            "Evaluations": f"{int(evals_mean)}",
        })

    df_summary = pd.DataFrame(summary_rows)
    md_table = format_markdown_table(df_summary)

    md_content = "# Benchmark Summary (MovieLens 1M)\n\n"
    md_content += "Evaluated on Test Set (200,016 ratings, 6,040 users, 3,706 items) across 3 seeds (0, 1, 2) with latent dimension $d = 10$.\n\n"
    md_content += md_table
    md_content += "\n\n*Note: Precision@k and Recall@k columns are computed from the saved factor matrices in `results/models/`.*\n"

    out_file = results_dir / "A_summary.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Summary saved: {out_file}")
    print("\n" + md_content)


def main():
    results_dir = Path("results")
    conv_dir = results_dir / "convergence"
    fig_dir = results_dir / "figures"

    generate_convergence_plots(conv_dir, fig_dir)
    generate_summary_md(results_dir)


if __name__ == "__main__":
    main()
