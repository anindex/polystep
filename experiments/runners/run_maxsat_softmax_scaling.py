#!/usr/bin/env python
"""MAX-SAT binary search scaling: find largest feasible variable count for softmax solver.

Uses binary search to find the largest variable count that fits in GPU memory,
recording peak memory at each scale point.

NOTE: MAX-SAT uses full-space mode (no subspace). Softmax is NOT the default
solver in full-space mode (paper Table 10 reports a 23-point accuracy gap at
P=50K). The purpose of this script is scalability/memory testing, not accuracy
parity with Sinkhorn.

Binary search procedure:
  1. Start with low=1000, high=2_000_000
  2. Probe mid=(low+high)//2 with a short run (--steps-short, default 10)
  3. If OOM: high=mid-1; if success: record memory, best=mid, low=mid+1
  4. After convergence: run full steps (--steps-full, default 100) at best_feasible
  5. Save memory profile and final result as JSON

Usage:
    python experiments/runners/run_maxsat_softmax_scaling.py
    python experiments/runners/run_maxsat_softmax_scaling.py --device cuda --seed 42
    python experiments/runners/run_maxsat_softmax_scaling.py --steps-short 5 --steps-full 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Ensure repo root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from experiments.runners.common import save_result, set_seed, track_gpu_memory
from experiments.runners.nondiff_data import generate_maxsat_instance
from experiments.runners.nondiff_models import MaxSATModel
from experiments.runners.run_maxsat import (
    CRA_ALPHA,
    CRA_LAMBDA,
    PSTORCH_CONFIG,
    PSTORCH_TURBO_1M,
    evaluate_sat_result,
    make_sat_closure,
)


# ---------------------------------------------------------------------------
# Core runner: run softmax MAX-SAT at a given scale
# ---------------------------------------------------------------------------


def run_softmax_maxsat(
    num_vars: int,
    seed: int,
    device: str,
    steps: int = 100,
) -> Dict[str, Any]:
    """Run MAX-SAT optimization with softmax solver at a given variable count.

    Args:
        num_vars: Number of Boolean variables in the 3-SAT instance.
        seed: Random seed for reproducibility.
        device: Device string ('cuda' or 'cpu').
        steps: Number of optimizer steps to run.

    Returns:
        Dict with sat_ratio, peak_memory_mb, num_vars, steps, wall_time_seconds.
    """
    from polystep.optimizer import PolyStepOptimizer

    set_seed(seed)

    # Generate instance with fixed seed for reproducibility
    instance = generate_maxsat_instance(num_vars=num_vars, seed=42)
    model = MaxSATModel(num_vars).to(device)
    clause_vars = instance["clause_vars"].to(device)
    clause_signs = instance["clause_signs"].to(device)

    # Select config based on scale
    turbo = num_vars >= 1_000_000
    if turbo:
        config = PSTORCH_TURBO_1M
        pdim = config["particle_dim"]
    else:
        config = PSTORCH_CONFIG
        pdim = 2

    # Create optimizer with solver='softmax' explicitly (full-space, not default)
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=config["epsilon"],
        step_radius=config["step_radius"],
        probe_radius=config["probe_radius"],
        num_probe=config["num_probe"],
        sinkhorn_max_iters=config["sinkhorn_max_iters"],
        chunk_size=config["chunk_size"],
        amortize_steps=config.get("amortize_steps", 3),
        amortize_ema=config.get("amortize_ema", 0.7),
        mixed_precision=config.get("mixed_precision", False),
        particle_dim=pdim,
        solver='softmax',
    )

    # Build closure
    clause_sample = config.get("clause_sample_size", 0) if turbo else 0
    closure = make_sat_closure(
        clause_vars, clause_signs,
        cra_lambda=CRA_LAMBDA, cra_alpha=CRA_ALPHA,
        clause_sample_size=clause_sample,
        model=model if turbo else None,
        particle_dim=pdim,
    )
    has_resample = hasattr(closure, 'resample')

    start_time = time.time()

    # Reset memory stats before run
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in range(steps):
        if has_resample:
            closure.resample()
        optimizer.step(closure)

    wall_time = time.time() - start_time

    # Record peak memory
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = round(peak_bytes / (1024 * 1024), 2)

    # Evaluate final result
    result = evaluate_sat_result(model, clause_vars, clause_signs)

    return {
        "sat_ratio": result["sat_ratio"],
        "num_satisfied": result["num_satisfied"],
        "num_clauses": result["num_clauses"],
        "peak_memory_mb": peak_memory_mb,
        "num_vars": num_vars,
        "steps": steps,
        "wall_time_seconds": round(wall_time, 2),
    }


# ---------------------------------------------------------------------------
# Binary search for maximum feasible scale
# ---------------------------------------------------------------------------


def binary_search_max_feasible(
    device: str = "cuda",
    seed: int = 42,
    steps_short: int = 10,
    steps_full: int = 100,
) -> Dict[str, Any]:
    """Binary search for the largest MAX-SAT variable count that fits in GPU memory.

    Uses short probe runs (steps_short) for feasibility checks, then runs a full
    evaluation (steps_full) at the best feasible scale.

    Args:
        device: Device string ('cuda' or 'cpu').
        seed: Random seed.
        steps_short: Steps for binary search probes (quick feasibility check).
        steps_full: Steps for final measurement at best feasible scale.

    Returns:
        Dict with best_feasible, memory_profile, final_result.
    """
    low = 1_000
    high = 2_000_000
    best_feasible = 0
    memory_profile: Dict[int, Any] = {}

    print(f"Binary search: finding largest feasible variable count on {device}")
    print(f"  Search range: [{low:,}, {high:,}]")
    print(f"  Probe steps: {steps_short}, full steps: {steps_full}")
    print("-" * 70)

    while low <= high:
        mid = (low + high) // 2
        print(f"\n  Probing num_vars={mid:,} ...", end=" ", flush=True)

        # Clear GPU cache before each attempt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            result = run_softmax_maxsat(mid, seed=seed, device=device, steps=steps_short)
            memory_profile[mid] = result["peak_memory_mb"]
            best_feasible = mid
            low = mid + 1
            print(f"OK (peak {result['peak_memory_mb']:.1f} MB, sat={result['sat_ratio']:.3f})")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                memory_profile[mid] = "OOM"
                high = mid - 1
                print("OOM")
            else:
                # Non-OOM RuntimeError: re-raise
                raise

        # Clear GPU after each attempt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 70}")
    print(f"  Best feasible: {best_feasible:,} variables")

    # Run full evaluation at best feasible scale
    final_result = None
    if best_feasible > 0:
        print(f"\n  Running full evaluation ({steps_full} steps) at {best_feasible:,} vars ...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            final_result = run_softmax_maxsat(
                best_feasible, seed=seed, device=device, steps=steps_full
            )
            print(f"  Final: sat_ratio={final_result['sat_ratio']:.4f}, "
                  f"peak={final_result['peak_memory_mb']:.1f} MB, "
                  f"time={final_result['wall_time_seconds']:.1f}s")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print("  OOM on full run -- reducing to last known good")
                final_result = {"error": "OOM on full run", "num_vars": best_feasible}
            else:
                raise

    return {
        "best_feasible": best_feasible,
        "memory_profile": memory_profile,
        "final_result": final_result,
        "search_range": {"low_start": 1_000, "high_start": 2_000_000},
        "config": {
            "steps_short": steps_short,
            "steps_full": steps_full,
            "seed": seed,
            "device": device,
        },
    }


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------


def print_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary table of the scaling results."""
    profile = result["memory_profile"]
    print(f"\n{'=' * 70}")
    print("MAX-SAT Softmax Scaling Results")
    print(f"{'=' * 70}")
    print(f"{'Num Vars':>12}  {'Peak Memory':>14}  {'Status':>8}")
    print(f"{'-' * 12}  {'-' * 14}  {'-' * 8}")

    for num_vars in sorted(profile.keys()):
        mem = profile[num_vars]
        if mem == "OOM":
            print(f"{num_vars:>12,}  {'--':>14}  {'OOM':>8}")
        else:
            print(f"{num_vars:>12,}  {mem:>11.1f} MB  {'OK':>8}")

    best = result["best_feasible"]
    print(f"\nBest feasible: {best:,} variables")

    if result.get("final_result") and "sat_ratio" in result["final_result"]:
        fr = result["final_result"]
        print(f"Final SAT ratio: {fr['sat_ratio']:.4f}")
        print(f"Final peak memory: {fr['peak_memory_mb']:.1f} MB")
        print(f"Final wall time: {fr['wall_time_seconds']:.1f}s")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for MAX-SAT softmax scaling binary search."""
    parser = argparse.ArgumentParser(
        description="Binary search for largest feasible MAX-SAT variable count with softmax solver"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (default: cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--results-dir", type=str, default="experiments/results/softmax",
                        help="Directory for JSON results (default: experiments/results/softmax)")
    parser.add_argument("--steps-short", type=int, default=10,
                        help="Steps for binary search probes (default: 10)")
    parser.add_argument("--steps-full", type=int, default=100,
                        help="Steps for final evaluation at best scale (default: 100)")
    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Run binary search
    result = binary_search_max_feasible(
        device=args.device,
        seed=args.seed,
        steps_short=args.steps_short,
        steps_full=args.steps_full,
    )

    # Print summary
    print_summary(result)

    # Save results as JSON
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, f"maxsat_softmax_scaling_seed{args.seed}.json")

    # Convert dict keys to strings for JSON serialization (int keys not valid JSON)
    serializable_profile = {str(k): v for k, v in result["memory_profile"].items()}
    save_data = {
        "benchmark": "maxsat_softmax_scaling",
        "best_feasible": result["best_feasible"],
        "memory_profile": serializable_profile,
        "final_result": result["final_result"],
        "search_range": result["search_range"],
        "config": result["config"],
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
