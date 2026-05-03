#!/usr/bin/env python
"""Fill missing cells in radius and epsilon ablation grids.

Scans existing results, identifies cells with < 5 seeds, and runs
only the missing seeds. Uses the same hyperparameters as the original
ablation sweep.

Usage:
    python -m experiments.runners.run_fill_ablation_grid --device cuda
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from experiments.runners.common import (
    evaluate_accuracy,
    save_result,
    set_seed,
    track_gpu_memory,
    load_mnist,
    MNISTNet,
    SEEDS,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
ABLATION_DIR = os.path.join(RESULTS_DIR, "softmax", "ablations")





# ---------------------------------------------------------------------------
# Training loop (matches original ablation)
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "rank": 4,
    "epsilon_init": 1.0,
    "epsilon_target": 0.1,
    "rotation_interval": 0,
    "absorb_interval": 20,
    "num_probe": 3,
    "sinkhorn_max_iters": 50,
    "epochs": 10,
    "batch_size": 512,
}


def run_one(pr: float, sr: float, seed: int, device: str,
            benchmark: str, out_dir: str) -> float:
    """Run a single ablation cell and save result."""
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    method_name = f"pr{pr}_sr{sr}"
    out_file = os.path.join(out_dir, f"{benchmark}_{method_name}_{seed}.json")
    if os.path.exists(out_file):
        print(f"  SKIP: {out_file} exists")
        d = json.load(open(out_file))
        return d["metrics"]["best_accuracy"]

    set_seed(seed)
    train_loader, test_loader = load_mnist(batch_size=BASE_CONFIG["batch_size"])
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()

    epochs = BASE_CONFIG["epochs"]
    total_steps = epochs * len(train_loader)
    eps_init = BASE_CONFIG["epsilon_init"]
    eps_target = BASE_CONFIG["epsilon_target"]
    eps_decay = (eps_init - eps_target) / max(1, total_steps)

    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout, rank=BASE_CONFIG["rank"],
        rotation_mode="random", rotation_interval=0,
        absorb_mode="periodic", absorb_interval=BASE_CONFIG["absorb_interval"],
    )

    optimizer = PolyStepOptimizer(
        model, compile=False, seed=seed,
        epsilon=CosineEpsilon(init=eps_init, target=eps_target, decay=eps_decay),
        step_radius=sr, probe_radius=pr,
        num_probe=BASE_CONFIG["num_probe"],
        sinkhorn_max_iters=BASE_CONFIG["sinkhorn_max_iters"],
        subspace=subspace, chunk_size=1024, amortize_steps=3,
    )
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    epoch_logs = []
    best_acc = 0.0
    step_count = 0
    fwd_count = 0
    t0 = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(epochs):
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                def closure(bp, _d=data, _t=targets):
                    nonlocal fwd_count
                    fwd_count += next(iter(bp.values())).shape[0]
                    return evaluator.evaluate(bp, _d, _t)
                optimizer.step(closure)
                step_count += 1

            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_acc = max(best_acc, test_acc)
            epoch_logs.append({
                "epoch": epoch + 1,
                "accuracy": test_acc,
                "cumulative_function_evals": fwd_count,
                "wall_time": time.time() - t0,
            })
            print(f"    [{method_name} s{seed}] ep{epoch+1}/{epochs} acc={test_acc*100:.1f}%")

    wall = time.time() - t0

    result = {
        "benchmark": benchmark,
        "method": method_name,
        "seed": seed,
        "metrics": {
            "final_accuracy": test_acc,
            "best_accuracy": best_acc,
            "wall_time_seconds": wall,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": fwd_count,
            "total_steps": step_count,
        },
        "hyperparameters": {**BASE_CONFIG, "probe_radius": pr, "step_radius": sr},
        "epoch_logs": epoch_logs,
    }
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  SAVED: {out_file} best={best_acc*100:.1f}%")
    return best_acc



def run_epsilon_one(ei: float, et: float, seed: int, device: str,
                    out_dir: str) -> float:
    """Run a single epsilon ablation cell."""
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    method_name = f"ei{ei}_et{et}"
    benchmark = "ablation_epsilon"
    out_file = os.path.join(out_dir, f"{benchmark}_{method_name}_{seed}.json")
    if os.path.exists(out_file):
        print(f"  SKIP: {out_file} exists")
        d = json.load(open(out_file))
        return d["metrics"]["best_accuracy"]

    set_seed(seed)
    train_loader, test_loader = load_mnist(batch_size=BASE_CONFIG["batch_size"])
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()

    epochs = BASE_CONFIG["epochs"]
    total_steps = epochs * len(train_loader)
    eps_decay = (ei - et) / max(1, total_steps)

    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout, rank=BASE_CONFIG["rank"],
        rotation_mode="random", rotation_interval=0,
        absorb_mode="periodic", absorb_interval=BASE_CONFIG["absorb_interval"],
    )

    optimizer = PolyStepOptimizer(
        model, compile=False, seed=seed,
        epsilon=CosineEpsilon(init=ei, target=et, decay=eps_decay),
        step_radius=4.5, probe_radius=1.0,
        num_probe=BASE_CONFIG["num_probe"],
        sinkhorn_max_iters=BASE_CONFIG["sinkhorn_max_iters"],
        subspace=subspace, chunk_size=1024, amortize_steps=3,
    )
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    epoch_logs = []
    best_acc = 0.0
    step_count = 0
    fwd_count = 0
    t0 = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(epochs):
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                def closure(bp, _d=data, _t=targets):
                    nonlocal fwd_count
                    fwd_count += next(iter(bp.values())).shape[0]
                    return evaluator.evaluate(bp, _d, _t)
                optimizer.step(closure)
                step_count += 1

            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_acc = max(best_acc, test_acc)
            epoch_logs.append({
                "epoch": epoch + 1,
                "accuracy": test_acc,
                "cumulative_function_evals": fwd_count,
                "wall_time": time.time() - t0,
            })
            print(f"    [{method_name} s{seed}] ep{epoch+1}/{epochs} acc={test_acc*100:.1f}%")

    wall = time.time() - t0

    result = {
        "benchmark": benchmark,
        "method": method_name,
        "seed": seed,
        "metrics": {
            "final_accuracy": test_acc,
            "best_accuracy": best_acc,
            "wall_time_seconds": wall,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": fwd_count,
            "total_steps": step_count,
        },
        "hyperparameters": {**BASE_CONFIG, "epsilon_init": ei, "epsilon_target": et},
        "epoch_logs": epoch_logs,
    }
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  SAVED: {out_file} best={best_acc*100:.1f}%")
    return best_acc


# ---------------------------------------------------------------------------
# Grid scan: find and fill missing cells
# ---------------------------------------------------------------------------

def scan_existing(out_dir: str, benchmark: str, pr_vals, sr_vals):
    """Scan existing results and return {(pr,sr): [seed, ...]}."""
    existing = {}
    for pr in pr_vals:
        for sr in sr_vals:
            method = f"pr{pr}_sr{sr}"
            pattern = os.path.join(out_dir, f"{benchmark}_{method}_*.json")
            files = glob.glob(pattern)
            seeds = []
            for f in files:
                try:
                    d = json.load(open(f))
                    seeds.append(d["seed"])
                except:
                    pass
            existing[(pr, sr)] = seeds
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-seeds", type=int, default=1,
                        help="Min seeds per cell (1 is fine for heatmaps; 5 for line plots)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_seeds = [42, 123, 456, 789, 1337]

    # === RADIUS GRID ===
    # Clean 5×5 grid (drop sr=2.0,8.0 and pr=0.25,1.5)
    pr_vals = [0.1, 0.5, 1.0, 2.0, 3.0]
    sr_vals = [1.0, 3.0, 4.5, 6.0, 10.0]
    radius_dir = os.path.join(ABLATION_DIR, "radius")

    print("=== RADIUS HEATMAP GRID (5×5) ===")
    existing = scan_existing(radius_dir, "ablation_radius", pr_vals, sr_vals)
    radius_todo = []
    for pr in pr_vals:
        for sr in sr_vals:
            have = existing.get((pr, sr), [])
            needed = [s for s in all_seeds[:args.target_seeds] if s not in have]
            status = f"have={len(have)}" + (f" NEED={len(needed)}" if needed else " ✓")
            print(f"  pr={pr} sr={sr}: {status}")
            for s in needed:
                radius_todo.append((pr, sr, s))

    print(f"\nTotal radius runs needed: {radius_todo}")
    if not radius_todo:
        print("Radius grid is already complete!")

    # === EPSILON GRID ===
    ei_vals = [0.5, 1.0, 2.0, 3.0]
    et_vals = [0.01, 0.1, 0.5]
    epsilon_dir = os.path.join(ABLATION_DIR, "epsilon")

    print("\n=== EPSILON HEATMAP GRID (4×3) ===")
    eps_existing = {}
    for ei in ei_vals:
        for et in et_vals:
            method = f"ei{ei}_et{et}"
            pattern = os.path.join(epsilon_dir, f"ablation_epsilon_{method}_*.json")
            files = glob.glob(pattern)
            seeds = []
            for f in files:
                try:
                    d = json.load(open(f))
                    seeds.append(d["seed"])
                except:
                    pass
            eps_existing[(ei, et)] = seeds

    eps_todo = []
    for ei in ei_vals:
        for et in et_vals:
            have = eps_existing.get((ei, et), [])
            needed = [s for s in all_seeds[:args.target_seeds] if s not in have]
            status = f"have={len(have)}" + (f" NEED={len(needed)}" if needed else " ✓")
            print(f"  ei={ei} et={et}: {status}")
            for s in needed:
                eps_todo.append((ei, et, s))

    print(f"\nTotal epsilon runs needed: {len(eps_todo)}")

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        print(f"  {len(radius_todo)} radius cells")
        print(f"  {len(eps_todo)} epsilon cells")
        return

    # === RUN RADIUS FILLS ===
    if radius_todo:
        print(f"\n{'='*60}")
        print(f"Running {len(radius_todo)} radius fills...")
        print(f"{'='*60}")
        for i, (pr, sr, seed) in enumerate(radius_todo):
            print(f"\n[{i+1}/{len(radius_todo)}] pr={pr} sr={sr} seed={seed}")
            run_one(pr, sr, seed, args.device, "ablation_radius", radius_dir)

    # === RUN EPSILON FILLS ===
    if eps_todo:
        print(f"\n{'='*60}")
        print(f"Running {len(eps_todo)} epsilon fills...")
        print(f"{'='*60}")
        for i, (ei, et, seed) in enumerate(eps_todo):
            print(f"\n[{i+1}/{len(eps_todo)}] ei={ei} et={et} seed={seed}")
            run_epsilon_one(ei, et, seed, args.device, epsilon_dir)

    print("\n=== ALL FILLS COMPLETE ===")


if __name__ == "__main__":
    main()
