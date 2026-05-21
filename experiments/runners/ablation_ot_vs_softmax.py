#!/usr/bin/env python
"""OT vs Softmax ablation: systematic comparison of update rules in PolyStep.

Determines under what conditions entropic OT diverges from simpler softmax-
weighted updates. Sweeps subspace rank, particle dimension, epsilon, and
tasks (MNIST, SNN).

Phases (select via --step N):
  1  Screening   - update_rule × rank × dp  (MNIST, 5 epochs, seed=42)
  2  Epsilon     - update_rule × epsilon     (MNIST, 5 epochs, seed=42)
  3  SNN         - update_rule × rank        (SNN, 20 epochs, seed=42)
  4  Profiling   - wall-clock per-step timing (50 steps per config)
  5  Convergence - OT vs softmax vs greedy   (MNIST, 10 epochs, per-step logs)

Results saved to: results/ot_ablation/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.nn as nn

from experiments.runners.common import (
    MNISTNet,
    evaluate_accuracy,
    load_mnist,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.nondiff_models import SpikingMNISTNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join("results", "ot_ablation")

# Solver name mapping: ablation rule name -> PolyStepOptimizer solver string
SOLVER_MAP = {
    "entropic_ot": "sinkhorn",
    "softmax": "softmax",
    "tempered_softmax": "tempered_softmax",
    "min_cost_greedy": "min_cost_greedy",
    "top_k_mean": "top_k_mean",
}

UPDATE_RULES = list(SOLVER_MAP.keys())
RANKS = [2, 4, 8, 16, 32, 64]
PARTICLE_DIMS = [2, 4, 8]
EPSILONS = [0.1, 0.5, 1.0, 3.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_optimizer(
    model: nn.Module,
    rank: int,
    dp: int,
    epsilon: float,
    solver_name: str,
    seed: int = 42,
    num_probe: int = 1,
    amortize_steps: int = 1,
    step_radius: float = 8.0,
    probe_radius: float = 1.0,
):
    """Create PolyStepOptimizer with HybridSubspace for given config."""
    from polystep.optimizer import PolyStepOptimizer
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.epsilon import LinearEpsilon

    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout,
        rank=rank,
        rotation_mode="random",
        rotation_interval=0,
        absorb_mode="periodic",
        absorb_interval=0,
    )

    solver_str = SOLVER_MAP[solver_name]

    # For tempered_softmax, set tau = epsilon (so "same sharpness" baseline)
    extra_kwargs: Dict[str, Any] = {}
    if solver_str == "tempered_softmax":
        extra_kwargs["tempered_softmax_tau"] = epsilon

    # Linear epsilon schedule for entropic_ot; fixed for others
    if solver_str == "sinkhorn":
        total_steps_est = 500  # conservative; exact doesn't matter much
        eps_decay = max((epsilon - 0.1) / total_steps_est, 1e-6)
        eps_value = LinearEpsilon(init=epsilon, target=0.1, decay=eps_decay)
    else:
        eps_value = epsilon

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=eps_value,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        subspace=subspace,
        subspace_particle_dim=dp,
        chunk_size=1024,
        amortize_steps=amortize_steps,
        solver=solver_str,
        **extra_kwargs,
    )

    # Compute P/V ratio for results metadata
    P = subspace.subspace_dim // dp if dp > 0 else 0
    V = 2 * dp  # orthoplex vertices
    pv_ratio = P / V if V > 0 else float("inf")

    return optimizer, subspace, {"P": P, "V": V, "pv_ratio": pv_ratio,
                                  "subspace_dim": subspace.subspace_dim}


def _train_loop(
    model: nn.Module,
    optimizer,
    evaluator,
    train_loader,
    test_loader,
    epochs: int,
    device: torch.device,
    per_step_log: bool = False,
) -> Dict[str, Any]:
    """Run training loop and return metrics dict."""
    loss_fn = nn.CrossEntropyLoss()
    epoch_logs: List[Dict[str, Any]] = []
    step_logs: List[Dict[str, Any]] = []
    best_acc = 0.0
    step_count = 0
    fwd_evals = 0
    start = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)

                def closure(bp, _d=data, _t=targets):
                    nonlocal fwd_evals
                    fwd_evals += next(iter(bp.values())).shape[0]
                    return evaluator.evaluate(bp, _d, _t)

                optimizer.step(closure)

                with torch.no_grad():
                    output = model(data)
                    loss = loss_fn(output, targets).item()
                    epoch_correct += (output.argmax(1) == targets).sum().item()
                    epoch_total += targets.size(0)
                epoch_loss += loss
                step_count += 1

                if per_step_log and step_count % 5 == 0:
                    test_acc = evaluate_accuracy(model, test_loader, device=device)
                    step_logs.append({
                        "step": step_count,
                        "epoch": epoch + 1,
                        "test_accuracy": test_acc,
                        "loss": loss,
                        "wall_time": time.time() - start,
                    })

            train_acc = epoch_correct / max(epoch_total, 1)
            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_acc = max(best_acc, test_acc)
            epoch_logs.append({
                "epoch": epoch + 1,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "loss": epoch_loss / max(len(train_loader), 1),
                "wall_time": time.time() - start,
            })

    wall = time.time() - start
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_acc = max(best_acc, final_acc)

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "wall_time_seconds": wall,
        "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
        "function_evals": fwd_evals,
        "total_steps": step_count,
        "epoch_logs": epoch_logs,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Step 1: Screening - update_rule × rank × dp
# ---------------------------------------------------------------------------

def run_phase1_screening(seed: int = 42, epochs: int = 5, epsilon: float = 3.0,
                         dry_run: bool = False):
    """Sweep update_rule × rank × particle_dim on MNIST."""
    from polystep.cost_nn import NNCostEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(batch_size=512)
    loss_fn = nn.CrossEntropyLoss()

    rules = UPDATE_RULES[:2] if dry_run else UPDATE_RULES  # OT+softmax in dry-run
    ranks = [8] if dry_run else RANKS
    dps = [2] if dry_run else PARTICLE_DIMS
    ep = 1 if dry_run else epochs

    results: List[Dict[str, Any]] = []
    total = len(rules) * len(ranks) * len(dps)
    idx = 0

    for rule in rules:
        for rank in ranks:
            for dp in dps:
                idx += 1
                print(f"[Step 1] {idx}/{total}: rule={rule}, rank={rank}, dp={dp}")
                set_seed(seed)
                model = MNISTNet().to(device)
                try:
                    optimizer, subspace, pv_info = _make_optimizer(
                        model, rank, dp, epsilon, rule, seed=seed,
                    )
                except Exception as e:
                    print(f"  SKIP: {e}")
                    results.append({
                        "rule": rule, "rank": rank, "dp": dp, "epsilon": epsilon,
                        "error": str(e), **{k: None for k in ["final_accuracy", "best_accuracy", "wall_time_seconds"]},
                    })
                    continue
                evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
                metrics = _train_loop(model, optimizer, evaluator,
                                      train_loader, test_loader, ep, device)
                row = {
                    "rule": rule, "rank": rank, "dp": dp, "epsilon": epsilon,
                    "seed": seed, **pv_info,
                    "final_accuracy": metrics["final_accuracy"],
                    "best_accuracy": metrics["best_accuracy"],
                    "wall_time_seconds": metrics["wall_time_seconds"],
                }
                results.append(row)
                print(f"  acc={metrics['final_accuracy']*100:.1f}%, "
                      f"best={metrics['best_accuracy']*100:.1f}%, "
                      f"time={metrics['wall_time_seconds']:.1f}s, "
                      f"P/V={pv_info['pv_ratio']:.1f}")

    _ensure_results_dir()
    path = os.path.join(RESULTS_DIR, "screening_results.json")
    with open(path, "w") as f:
        json.dump({"step": "screening", "seed": seed, "epochs": ep,
                    "epsilon": epsilon, "results": results}, f, indent=2)
    print(f"Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Step 2: Epsilon sensitivity - update_rule × epsilon
# ---------------------------------------------------------------------------

def run_phase2_epsilon(seed: int = 42, epochs: int = 5, rank: int = 8, dp: int = 2,
                       dry_run: bool = False):
    """Sweep update_rule × epsilon on MNIST at fixed rank/dp."""
    from polystep.cost_nn import NNCostEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(batch_size=512)
    loss_fn = nn.CrossEntropyLoss()

    rules = UPDATE_RULES[:2] if dry_run else UPDATE_RULES
    epsilons = [1.0] if dry_run else EPSILONS
    ep = 1 if dry_run else epochs

    results: List[Dict[str, Any]] = []
    total = len(rules) * len(epsilons)
    idx = 0

    for rule in rules:
        for eps in epsilons:
            idx += 1
            print(f"[Step 2] {idx}/{total}: rule={rule}, eps={eps}")
            set_seed(seed)
            model = MNISTNet().to(device)
            try:
                optimizer, subspace, pv_info = _make_optimizer(
                    model, rank, dp, eps, rule, seed=seed,
                )
            except Exception as e:
                print(f"  SKIP: {e}")
                results.append({"rule": rule, "rank": rank, "dp": dp, "epsilon": eps,
                                "error": str(e)})
                continue
            evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
            metrics = _train_loop(model, optimizer, evaluator,
                                  train_loader, test_loader, ep, device)
            row = {
                "rule": rule, "rank": rank, "dp": dp, "epsilon": eps,
                "seed": seed, **pv_info,
                "final_accuracy": metrics["final_accuracy"],
                "best_accuracy": metrics["best_accuracy"],
                "wall_time_seconds": metrics["wall_time_seconds"],
            }
            results.append(row)
            print(f"  acc={metrics['final_accuracy']*100:.1f}%, time={metrics['wall_time_seconds']:.1f}s")

    _ensure_results_dir()
    path = os.path.join(RESULTS_DIR, "epsilon_sweep.json")
    with open(path, "w") as f:
        json.dump({"step": "epsilon_sweep", "seed": seed, "epochs": ep,
                    "rank": rank, "dp": dp, "results": results}, f, indent=2)
    print(f"Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Step 3: SNN non-differentiable task
# ---------------------------------------------------------------------------

def run_phase3_snn(seed: int = 42, epochs: int = 20, dp: int = 2, epsilon: float = 0.5,
                   dry_run: bool = False):
    """Sweep update_rule × rank on hard-LIF SNN task."""
    from polystep.cost_nn import NNCostEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(batch_size=512)
    loss_fn = nn.CrossEntropyLoss()

    rules = UPDATE_RULES[:2] if dry_run else UPDATE_RULES
    ranks = [4] if dry_run else [2, 4, 8, 16]
    ep = 1 if dry_run else epochs

    results: List[Dict[str, Any]] = []
    total = len(rules) * len(ranks)
    idx = 0

    for rule in rules:
        for rank in ranks:
            idx += 1
            print(f"[Step 3] {idx}/{total}: rule={rule}, rank={rank} (SNN)")
            set_seed(seed)
            model = SpikingMNISTNet(num_steps=15).to(device)
            try:
                optimizer, subspace, pv_info = _make_optimizer(
                    model, rank, dp, epsilon, rule, seed=seed,
                    step_radius=2.0, probe_radius=1.0, amortize_steps=1,
                )
            except Exception as e:
                print(f"  SKIP: {e}")
                results.append({"rule": rule, "rank": rank, "dp": dp, "epsilon": epsilon,
                                "error": str(e)})
                continue
            evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
            metrics = _train_loop(model, optimizer, evaluator,
                                  train_loader, test_loader, ep, device)
            row = {
                "rule": rule, "rank": rank, "dp": dp, "epsilon": epsilon,
                "seed": seed, "task": "snn", **pv_info,
                "final_accuracy": metrics["final_accuracy"],
                "best_accuracy": metrics["best_accuracy"],
                "wall_time_seconds": metrics["wall_time_seconds"],
            }
            results.append(row)
            print(f"  acc={metrics['final_accuracy']*100:.1f}%, time={metrics['wall_time_seconds']:.1f}s")

    _ensure_results_dir()
    path = os.path.join(RESULTS_DIR, "snn_results.json")
    with open(path, "w") as f:
        json.dump({"step": "snn", "seed": seed, "epochs": ep,
                    "dp": dp, "epsilon": epsilon, "results": results}, f, indent=2)
    print(f"Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Step 4: Wall-clock profiling
# ---------------------------------------------------------------------------

def run_phase4_profiling(seed: int = 42, num_steps: int = 50, dp: int = 2,
                         epsilon: float = 3.0, dry_run: bool = False):
    """Measure per-step wall-clock time for each solver at various ranks."""
    from polystep.cost_nn import NNCostEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(batch_size=512)
    loss_fn = nn.CrossEntropyLoss()
    train_iter = iter(train_loader)

    rules = UPDATE_RULES[:2] if dry_run else UPDATE_RULES
    ranks = [8] if dry_run else [2, 8, 32, 64]
    steps = 5 if dry_run else num_steps

    results: List[Dict[str, Any]] = []
    total = len(rules) * len(ranks)
    idx = 0

    for rule in rules:
        for rank in ranks:
            idx += 1
            print(f"[Step 4] {idx}/{total}: rule={rule}, rank={rank}")
            set_seed(seed)
            model = MNISTNet().to(device)
            try:
                optimizer, subspace, pv_info = _make_optimizer(
                    model, rank, dp, epsilon, rule, seed=seed,
                )
            except Exception as e:
                print(f"  SKIP: {e}")
                results.append({"rule": rule, "rank": rank, "dp": dp,
                                "error": str(e)})
                continue
            evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

            step_times: List[float] = []
            train_iter = iter(train_loader)
            for s in range(steps):
                try:
                    data, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    data, targets = next(train_iter)
                data, targets = data.to(device), targets.to(device)

                def closure(bp, _d=data, _t=targets):
                    return evaluator.evaluate(bp, _d, _t)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                optimizer.step(closure)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                step_times.append(t1 - t0)

            mean_time = sum(step_times) / len(step_times)
            results.append({
                "rule": rule, "rank": rank, "dp": dp, "epsilon": epsilon,
                **pv_info,
                "mean_step_time_s": mean_time,
                "median_step_time_s": sorted(step_times)[len(step_times) // 2],
                "step_times": step_times,
            })
            print(f"  mean={mean_time*1000:.1f}ms/step")

    _ensure_results_dir()
    path = os.path.join(RESULTS_DIR, "profiling.json")
    with open(path, "w") as f:
        json.dump({"step": "profiling", "seed": seed, "num_steps": steps,
                    "results": results}, f, indent=2)
    print(f"Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Step 5: Convergence dynamics
# ---------------------------------------------------------------------------

def run_phase5_convergence(seed: int = 42, epochs: int = 10, rank: int = 8,
                           dp: int = 2, epsilon: float = 3.0, dry_run: bool = False):
    """Compare OT vs softmax vs greedy convergence at fixed config."""
    from polystep.cost_nn import NNCostEvaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(batch_size=512)
    loss_fn = nn.CrossEntropyLoss()

    rules = ["entropic_ot", "softmax", "min_cost_greedy"] if not dry_run else ["softmax"]
    ep = 1 if dry_run else epochs

    results: List[Dict[str, Any]] = []

    for rule in rules:
        print(f"[Step 5] rule={rule}, rank={rank}, dp={dp}, eps={epsilon}")
        set_seed(seed)
        model = MNISTNet().to(device)
        try:
            optimizer, subspace, pv_info = _make_optimizer(
                model, rank, dp, epsilon, rule, seed=seed,
            )
        except Exception as e:
            print(f"  SKIP: {e}")
            results.append({"rule": rule, "error": str(e)})
            continue
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        metrics = _train_loop(model, optimizer, evaluator,
                              train_loader, test_loader, ep, device,
                              per_step_log=True)
        results.append({
            "rule": rule, "rank": rank, "dp": dp, "epsilon": epsilon,
            "seed": seed, **pv_info,
            "final_accuracy": metrics["final_accuracy"],
            "best_accuracy": metrics["best_accuracy"],
            "wall_time_seconds": metrics["wall_time_seconds"],
            "epoch_logs": metrics["epoch_logs"],
            "step_logs": metrics["step_logs"],
        })
        print(f"  final={metrics['final_accuracy']*100:.1f}%, "
              f"best={metrics['best_accuracy']*100:.1f}%")

    _ensure_results_dir()
    path = os.path.join(RESULTS_DIR, "convergence.json")
    with open(path, "w") as f:
        json.dump({"step": "convergence", "seed": seed, "epochs": ep,
                    "rank": rank, "dp": dp, "epsilon": epsilon,
                    "results": results}, f, indent=2)
    print(f"Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OT vs Softmax ablation study")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Phase to run (1-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: 1 epoch, minimal grid")
    args = parser.parse_args()

    print(f"=== OT vs Softmax Ablation - Phase {args.step} ===")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.step == 1:
        run_phase1_screening(seed=args.seed, dry_run=args.dry_run)
    elif args.step == 2:
        run_phase2_epsilon(seed=args.seed, dry_run=args.dry_run)
    elif args.step == 3:
        run_phase3_snn(seed=args.seed, dry_run=args.dry_run)
    elif args.step == 4:
        run_phase4_profiling(seed=args.seed, dry_run=args.dry_run)
    elif args.step == 5:
        run_phase5_convergence(seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
