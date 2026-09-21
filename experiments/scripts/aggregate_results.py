"""Aggregate per-run JSON experiment results into pandas DataFrames.

Reads JSON files from experiments/results/ and produces summary DataFrames with
mean/std per (benchmark, method) group.

CLI usage:
    python experiments/scripts/aggregate_results.py experiments/results/
    python experiments/scripts/aggregate_results.py experiments/results/ --benchmark mnist
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


class LeakedResultError(RuntimeError):
    """Raised when a result file is stamped ``"leaked": true``.

    Deliberately not caught by the aggregation loop: a run whose checkpoint was
    picked on the test set must fail loudly, not be skipped with a warning.
    """


def load_single_result(path: str) -> Dict[str, Any]:
    """Load a single JSON result file and extract key fields into a flat dict.

    Flattens the metrics dict into the top level and preserves epoch_logs for
    convergence curve plotting.

    Raises:
        LeakedResultError: If the run was produced with --allow-test-leakage.
    """
    with open(path, "r") as f:
        data = json.load(f)

    if data.get("leaked", False):
        raise LeakedResultError(
            f"{path} was produced with --allow-test-leakage: its checkpoint was "
            "selected on the test set. Re-run without that flag; a leaked run "
            "must never be aggregated into a reported number."
        )

    metrics = data.get("metrics", {})

    result = {
        "benchmark": data["benchmark"],
        "method": data["method"],
        "seed": data["seed"],
        # Reported metric: test accuracy of the val-selected checkpoint. Older
        # result files lack the key; final_accuracy is the fallback (best_accuracy
        # was max-over-epochs test).
        "test_accuracy_at_selected": metrics.get("test_accuracy_at_selected", metrics.get("final_accuracy", 0.0)),
        "final_accuracy": metrics.get("final_accuracy", 0.0),
        # Distinguishes "solved nothing" from "no metric recorded"; a missing
        # final_accuracy defaults to 0.0 above and is otherwise indistinguishable.
        "has_final_accuracy": "final_accuracy" in metrics,
        # Written by the MAX-SAT exact solvers when they hit their wall clock.
        "timed_out": (data.get("hyperparameters") or {}).get("timed_out"),
        "best_accuracy": metrics.get("best_accuracy", 0.0),
        "final_mse": metrics.get("final_mse"),
        "best_mse": metrics.get("best_mse"),
        "wall_time_seconds": metrics.get("wall_time_seconds", 0.0),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb", 0.0),
        "function_evals": metrics.get("function_evals", 0),
        "total_steps": metrics.get("total_steps", 0),
        "epoch_logs": data.get("epoch_logs", []),
        "step_logs": data.get("step_logs", []),
        # Which budget this cell was matched on: evals, steps, wallclock, or None for
        # PolyStep and Adam, which set the budget rather than receive one.
        "rl_env_steps": metrics.get("rl_env_steps"),
        "match_axis": (data.get("hyperparameters") or {}).get("match_axis"),
        "deadline_s": (data.get("hyperparameters") or {}).get("deadline_s"),
        "source_file": os.path.basename(path),
    }

    return result


def aggregate_results(
    results_dir: str,
    benchmark: Optional[str] = None,
) -> pd.DataFrame:
    """Read all JSON result files and produce a summary DataFrame.

    One row per (benchmark, method) group with mean/std accuracy, MSE, time,
    memory, function evals and seed count. Empty DataFrame (with correct columns)
    if no results found.

    Raises:
        LeakedResultError: If any result file is stamped ``"leaked": true``.
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

    if benchmark:
        pattern = os.path.join(results_dir, f"{benchmark}_*.json")
    else:
        pattern = os.path.join(results_dir, "*.json")

    json_files = sorted(glob.glob(pattern))

    if not json_files:
        return pd.DataFrame(columns=summary_columns)

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

    per_run_cols = [
        "benchmark",
        "method",
        "seed",
        "test_accuracy_at_selected",
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

    grouped = df_runs.groupby(["benchmark", "method"], sort=True)

    summary_rows = []
    for (bm, method), group in grouped:
        # Regression benchmarks (mse populated) report mean/std MSE;
        # classification benchmarks report mean/std accuracy.
        mse_vals = group["best_mse"].dropna()
        has_mse = len(mse_vals) > 0
        summary_rows.append(
            {
                "benchmark": bm,
                "method": method,
                "mean_accuracy": group["test_accuracy_at_selected"].mean(),
                "std_accuracy": group["test_accuracy_at_selected"].std(ddof=1) if len(group) > 1 else 0.0,
                "mean_mse": mse_vals.mean() if has_mse else float("nan"),
                "std_mse": mse_vals.std(ddof=1)
                if has_mse and len(mse_vals) > 1
                else (0.0 if has_mse else float("nan")),
                "mean_time": group["wall_time_seconds"].mean(),
                "std_time": group["wall_time_seconds"].std(ddof=1) if len(group) > 1 else 0.0,
                "mean_memory": group["peak_gpu_memory_mb"].mean(),
                "mean_func_evals": group["function_evals"].mean(),
                "n_runs": len(group),
            }
        )

    return pd.DataFrame(summary_rows, columns=summary_columns)


def main() -> None:
    """CLI entry point: aggregate results and print summary."""
    if len(sys.argv) < 2:
        print("Usage: python experiments/scripts/aggregate_results.py <results_dir> [--benchmark <name>]")
        sys.exit(1)

    results_dir = sys.argv[1]
    benchmark = None
    out_prefix = None

    if "--benchmark" in sys.argv:
        idx = sys.argv.index("--benchmark")
        if idx + 1 < len(sys.argv):
            benchmark = sys.argv[idx + 1]

    if "--write" in sys.argv:
        idx = sys.argv.index("--write")
        if idx + 1 < len(sys.argv):
            out_prefix = sys.argv[idx + 1]

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    df = aggregate_results(results_dir, benchmark=benchmark)

    if df.empty:
        print("No results found.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))

    if out_prefix is None:
        return

    # Write both the aggregate and the per-seed rows, so the paper's numbers have
    # a regenerable artifact on disk behind them.
    summary_path = f"{out_prefix}_summary.csv"
    per_seed_path = f"{out_prefix}_per_seed.csv"
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".", exist_ok=True)
    df.to_csv(summary_path, index=False)

    # Same population as the summary above: directly in ``results_dir``, filtered
    # by ``--benchmark``, skipping files that will not parse. Recursing would pull
    # in theory-mode runs and tuning trials that share the reported
    # ``{benchmark}_{method}_{seed}`` filename pattern.
    rows: List[Dict[str, Any]] = []
    pattern = os.path.join(results_dir, f"{benchmark}_*.json") if benchmark else os.path.join(results_dir, "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            rec = load_single_result(path)
        except (json.JSONDecodeError, KeyError, FileNotFoundError, AttributeError, TypeError) as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            continue
        rec.pop("epoch_logs", None)
        rec["source_file"] = os.path.relpath(path, results_dir)
        rows.append(rec)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(per_seed_path, index=False)
    if len(per_seed) != int(df["n_runs"].sum()):
        print(
            f"Warning: per-seed rows ({len(per_seed)}) != summary n_runs ({int(df['n_runs'].sum())})",
            file=sys.stderr,
        )

    print(f"\nwrote {summary_path} ({len(df)} groups)")
    print(f"wrote {per_seed_path} ({len(per_seed)} runs, no file stamped leaked)")


if __name__ == "__main__":
    main()
