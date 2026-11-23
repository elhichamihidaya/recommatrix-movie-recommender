"""Protocol constants, measurement utilities, and results serialization."""

from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from typing import Iterator
import numpy as np
import pandas as pd
import psutil

# Experiment constants
RATING_MIN: float = 1.0
RATING_MAX: float = 5.0
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
D_MAIN: int = 10
SPLIT_SEED: int = 42

RESULT_COLUMNS: list[str] = [
    "method",
    "variant",
    "seed",
    "d",
    "hyperparams",
    "rmse_val",
    "rmse_test",
    "mae_test",
    "prec@5",
    "prec@10",
    "prec@20",
    "rec@5",
    "rec@10",
    "rec@20",
    "fit_time_s",
    "peak_mem_mb",
    "n_evals",
]


def append_result(csv_path: str | Path, row_dict: dict) -> None:
    """Validate keys and append one row to the common results CSV file."""
    # Check for unknown keys
    unknown_keys = set(row_dict.keys()) - set(RESULT_COLUMNS)
    if unknown_keys:
        raise ValueError(f"Unknown keys in row_dict: {unknown_keys}")

    # Build row in exact schema order, filling missing with NaN
    row = {}
    for col in RESULT_COLUMNS:
        val = row_dict.get(col, np.nan)
        if col == "hyperparams" and isinstance(val, (dict, list)):
            row[col] = json.dumps(val, sort_keys=True)
        else:
            row[col] = val

    df_row = pd.DataFrame([row], columns=RESULT_COLUMNS)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = not path.exists() or path.stat().st_size == 0
    df_row.to_csv(path, mode="a", index=False, header=header)


@contextmanager
def measure() -> Iterator[dict]:
    """Measure elapsed execution time (s) and peak memory usage (MB) above baseline."""
    process = psutil.Process()
    baseline_rss = process.memory_info().rss / (1024 * 1024)
    peak_rss = baseline_rss
    stopped = False

    def sample_memory():
        nonlocal peak_rss
        while not stopped:
            try:
                rss = process.memory_info().rss / (1024 * 1024)
                if rss > peak_rss:
                    peak_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.05)

    sampler_thread = threading.Thread(target=sample_memory, daemon=True)
    sampler_thread.start()

    stats = {"time_s": 0.0, "peak_mem_mb": 0.0}
    t0 = time.perf_counter()

    try:
        yield stats
    finally:
        stopped = True
        sampler_thread.join(timeout=0.2)
        elapsed = time.perf_counter() - t0
        # Final sample check
        try:
            final_rss = process.memory_info().rss / (1024 * 1024)
            if final_rss > peak_rss:
                peak_rss = final_rss
        except Exception:
            pass

        stats["time_s"] = float(elapsed)
        stats["peak_mem_mb"] = float(max(0.0, peak_rss - baseline_rss))
