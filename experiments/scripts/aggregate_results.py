"""Aggregate per-run JSON experiment results into pandas DataFrames.

Reads JSON files from experiments/results/ (produced by paper.experiments.common.save_result)
and produces summary DataFrames with mean/std per (benchmark, method) group.

Functions:
    load_single_result(path) -> dict: Load one JSON, extract flat metrics.
    aggregate_results(results_dir, benchmark) -> pd.DataFrame: Summary stats.
    get_epoch_curves(results_dir, benchmark) -> dict: Per-method convergence curves.

CLI usage:
    python experiments/scripts/aggregate_results.py experiments/results/
    python experiments/scripts/aggregate_results.py experiments/results/ --benchmark mnist
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Load a single result file
# ---------------------------------------------------------------------------

def load_single_result(path: str) -> Dict[str, Any]:
    """Load a single JSON result file and extract key fields into a flat dict.

    Extracts top-level identifiers (benchmark, method, seed) and flattens
    the metrics dict into the top level. Preserves epoch_logs for convergence
    curve plotting.

    Args:
        path: Path to a JSON result file.

    Returns:
        Flat dict with keys: benchmark, method, seed, final_accuracy,
        best_accuracy, wall_time_seconds, peak_gpu_memory_mb, function_evals,
        total_steps, epoch_logs, source_file.

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If file is not valid JSON.
        KeyError: If required keys are missing from the JSON.
    """
    with open(path, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})

    result = {
        "benchmark": data["benchmark"],
        "method": data["method"],
        "seed": data["seed"],
        "final_accuracy": metrics.get("final_accuracy", 0.0),
        "best_accuracy": metrics.get("best_accuracy", 0.0),
        "final_mse": metrics.get("final_mse"),
        "best_mse": metrics.get("best_mse"),
        "wall_time_seconds": metrics.get("wall_time_seconds", 0.0),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb", 0.0),
        "function_evals": metrics.get("function_evals", 0),
        "total_steps": metrics.get("total_steps", 0),
        "epoch_logs": data.get("epoch_logs", []),
        "source_file": os.path.basename(path),
    }

    return result


# ---------------------------------------------------------------------------
# Aggregate results into summary DataFrame
# ---------------------------------------------------------------------------

def aggregate_results(
    results_dir: str,
    benchmark: Optional[str] = None,
) -> pd.DataFrame:
    """Read all JSON result files and produce a summary DataFrame.

    Loads all matching JSON files, creates a per-run DataFrame, then
    computes grouped summary statistics per (benchmark, method).

    Summary columns:
        - benchmark, method
        - mean_accuracy, std_accuracy (from best_accuracy)
        - mean_time, std_time (from wall_time_seconds)
        - mean_memory (from peak_gpu_memory_mb)
        - mean_func_evals (from function_evals)
        - n_runs (count of seeds)

    Args:
        results_dir: Directory containing JSON result files.
        benchmark: Optional benchmark name to filter by. If None, loads all.

    Returns:
        pd.DataFrame with one row per (benchmark, method) group.
        Empty DataFrame (with correct columns) if no results found.
    """
    summary_columns = [
        "benchmark",
        "method",
        "mean_accuracy",
        "std_accuracy",
        "mean_mse",
        "std_mse",
        "mean_time",
        "std_time",
        "mean_memory",
        "mean_func_evals",
        "n_runs",
    ]

    # Find matching JSON files
    if benchmark:
        pattern = os.path.join(results_dir, f"{benchmark}_*.json")
    else:
        pattern = os.path.join(results_dir, "*.json")

    json_files = sorted(glob.glob(pattern))

    if not json_files:
        return pd.DataFrame(columns=summary_columns)

    # Load all results into flat dicts
    rows = []
    for path in json_files:
        try:
            row = load_single_result(path)
            rows.append(row)
        except (json.JSONDecodeError, KeyError, FileNotFoundError, AttributeError, TypeError) as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            continue

    if not rows:
        return pd.DataFrame(columns=summary_columns)

    # Create per-run DataFrame (drop epoch_logs for aggregation)
    per_run_cols = [
        "benchmark",
        "method",
        "seed",
        "final_accuracy",
        "best_accuracy",
        "final_mse",
        "best_mse",
        "wall_time_seconds",
        "peak_gpu_memory_mb",
        "function_evals",
        "total_steps",
    ]
    df_runs = pd.DataFrame([{k: r[k] for k in per_run_cols} for r in rows])

    # Group by (benchmark, method) and compute summary stats
    grouped = df_runs.groupby(["benchmark", "method"], sort=True)

    summary_rows = []
    for (bm, method), group in grouped:
        # MSE-aware aggregation: regression benchmarks (mse populated) report
        # mean/std MSE; classification benchmarks report mean/std accuracy.
        mse_vals = group["best_mse"].dropna()
        has_mse = len(mse_vals) > 0
        summary_rows.append(
            {
                "benchmark": bm,
                "method": method,
                "mean_accuracy": group["best_accuracy"].mean(),
                "std_accuracy": group["best_accuracy"].std(ddof=1)
                if len(group) > 1
                else 0.0,
                "mean_mse": mse_vals.mean() if has_mse else float("nan"),
                "std_mse": mse_vals.std(ddof=1)
                if has_mse and len(mse_vals) > 1
                else (0.0 if has_mse else float("nan")),
                "mean_time": group["wall_time_seconds"].mean(),
                "std_time": group["wall_time_seconds"].std(ddof=1)
                if len(group) > 1
                else 0.0,
                "mean_memory": group["peak_gpu_memory_mb"].mean(),
                "mean_func_evals": group["function_evals"].mean(),
                "n_runs": len(group),
            }
        )

    return pd.DataFrame(summary_rows, columns=summary_columns)


# ---------------------------------------------------------------------------
# Convergence curves
# ---------------------------------------------------------------------------

def get_epoch_curves(
    results_dir: str,
    benchmark: str,
) -> Dict[str, List[List[Tuple[int, float]]]]:
    """Extract per-method convergence curves for plotting.

    Returns a dict mapping method name to a list of per-seed accuracy curves.
    Each curve is a list of (epoch, accuracy) tuples from epoch_logs.

    Args:
        results_dir: Directory containing JSON result files.
        benchmark: Benchmark name to filter by.

    Returns:
        Dict mapping method -> list of curves.
        Each curve is a list of (epoch, accuracy) tuples.
        Empty dict if no results found.
    """
    pattern = os.path.join(results_dir, f"{benchmark}_*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        return {}

    curves: Dict[str, List[List[Tuple[int, float]]]] = {}

    for path in json_files:
        try:
            result = load_single_result(path)
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

        method = result["method"]
        epoch_logs = result["epoch_logs"]

        if method not in curves:
            curves[method] = []

        # Extract (epoch, accuracy) from each log entry
        curve = []
        for entry in epoch_logs:
            epoch = entry.get("epoch", 0)
            accuracy = entry.get("accuracy", 0.0)
            curve.append((epoch, accuracy))

        curves[method].append(curve)

    return curves


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: aggregate results and print summary."""
    if len(sys.argv) < 2:
        print(
            "Usage: python experiments/scripts/aggregate_results.py <results_dir> "
            "[--benchmark <name>]"
        )
        sys.exit(1)

    results_dir = sys.argv[1]
    benchmark = None

    if "--benchmark" in sys.argv:
        idx = sys.argv.index("--benchmark")
        if idx + 1 < len(sys.argv):
            benchmark = sys.argv[idx + 1]

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    df = aggregate_results(results_dir, benchmark=benchmark)

    if df.empty:
        print("No results found.")
    else:
        # Configure pandas display for readable output
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
