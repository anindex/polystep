#!/usr/bin/env python
"""Run all methods and seeds for Hard MoE (Mixture-of-Experts) benchmark.

Methods:
  - polystep: PolyStepOptimizer with HybridSubspace on hard-gated MoE
  - cmaes: pycma CMA-ES on hard-gated MoE
  - openai_es: OpenAI-ES on hard-gated MoE
  - spsa: SPSA on hard-gated MoE

Model: HardMoENet (~235K params) - top-1 argmax gating (non-differentiable)
Data: Combined MNIST + Fashion-MNIST (20 classes)

polystep config from r4_sr12t4 config (90.92% at 20ep, seed 42):
  Flat eps=0.5 (eps scheduling -> collapse), scheduled sr 12->4,
  flat pr=1.0, rank=4, advanced features (biased_rotation, anderson, adaptive_omega).

Results saved as: experiments/results/softmax/main/moe_{method}_{seed}.json
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# Ensure repo root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import torch
import torch.nn as nn

from experiments.runners.common import (
    SEEDS,
    evaluate_accuracy,
    load_flat_params,
    make_train_val_split,
    set_flat_params,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.nondiff_models import HardMoENet
from experiments.runners.nondiff_data import generate_multidomain_data
from experiments.baselines.openai_es import train_openai_es
from experiments.baselines.spsa import train_spsa


# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

BENCHMARK = "moe"
BATCH_SIZE = 512
EPOCHS = 30

OPENAI_ES_CONFIG = {
    "sigma": 0.02, "lr": 0.01, "population_size": 50, "generations": 2000,
    "lr_decay": True, "weight_decay": 0.01, "fitness_shaping": "rank",
}
CMAES_CONFIG = {"generations": 2000, "popsize": 16, "stdev_init": 0.5}
SPSA_CONFIG = {"a": 0.1, "c": 0.1, "alpha": 0.602, "gamma": 0.101, "max_iters": 10000}

# polystep config - r4_sr12t4 config (90.92% at 20ep, seed 42)
# HYBRID: flat eps + flat pr, but SCHEDULED sr only
# eps ≤ 0.5 mandatory - eps scheduling causes MoE collapse (2.66% at eps=1.5)
# sr scheduling (12->4) with rank=4 beats flat rank=8 while being 2x faster
PSTORCH_CONFIG = {
    "epsilon": 0.5,                     # FLAT - eps scheduling -> collapse
    "step_radius_init": 12.0,          # sr scheduling: 12->4
    "step_radius_target": 4.0,
    "probe_radius": 1.0,               # FLAT
    "num_probe": 1, "rank": 4,
    "chunk_size": 1024, "amortize_steps": 1,
    "rotation_interval": 0, "absorb_interval": 20,
    "biased_rotation": True,
    "anderson_depth": 0,               # Option 3: ablation showed zero effect on MoE (identical 3-epoch numerics vs depth=5)
    "adaptive_omega": True,
}


# ---------------------------------------------------------------------------
# Method: polystep
# ---------------------------------------------------------------------------

def run_polystep(seed, device, results_dir, epochs=EPOCHS, dry_run=False,
                audit_no_leakage: bool = True):
    """Train Hard MoE with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses a held-out validation
    split (honest protocol). Set ``audit_no_leakage=False`` to revert
    to legacy test-set selection.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    if dry_run:
        epochs = 1

    set_seed(seed)
    model = HardMoENet(num_experts=4).to(device)
    loss_fn = nn.CrossEntropyLoss()

    data = generate_multidomain_data(data_dir="data/", batch_size=BATCH_SIZE)
    train_loader, test_loader = data["train_loader"], data["test_loader"]

    # Leakage-free model selection: carve 10% validation split from training data
    val_loader = None
    if audit_no_leakage:
        train_loader, val_loader = make_train_val_split(
            train_loader, val_frac=0.1, seed=seed,
        )
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader

    total_steps = epochs * len(train_loader)

    # Epsilon: FLAT (no CosineEpsilon - eps scheduling destroys MoE)
    epsilon_value = PSTORCH_CONFIG["epsilon"]

    # Step radius: CosineEpsilon scheduled 12->4
    if "step_radius_init" in PSTORCH_CONFIG:
        sr_decay = (PSTORCH_CONFIG["step_radius_init"] - PSTORCH_CONFIG["step_radius_target"]) / max(1, total_steps)
        step_radius_value = CosineEpsilon(
            init=PSTORCH_CONFIG["step_radius_init"],
            target=PSTORCH_CONFIG["step_radius_target"],
            decay=sr_decay,
        )
    else:
        step_radius_value = PSTORCH_CONFIG.get("step_radius", 1.0)

    # Probe radius: FLAT
    probe_radius_value = PSTORCH_CONFIG.get("probe_radius", 1.0)

    # HybridSubspace setup
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout,
        rank=PSTORCH_CONFIG["rank"],
        rotation_mode="random",
        rotation_interval=PSTORCH_CONFIG["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=PSTORCH_CONFIG["absorb_interval"],
    )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=epsilon_value,
        step_radius=step_radius_value,
        probe_radius=probe_radius_value,
        num_probe=PSTORCH_CONFIG["num_probe"],
        subspace=subspace,
        chunk_size=PSTORCH_CONFIG["chunk_size"],
        amortize_steps=PSTORCH_CONFIG["amortize_steps"],
        biased_rotation=PSTORCH_CONFIG.get("biased_rotation", False),
        anderson_depth=PSTORCH_CONFIG.get("anderson_depth", 0),
        adaptive_omega=PSTORCH_CONFIG.get("adaptive_omega", False),
        solver="softmax",
    )

    import copy

    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
    epoch_logs = []
    step_logs = []
    best_accuracy = 0.0
    best_state_dict = None
    step_count = 0
    fwd_pass_count = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_start = time.time()

            for data_batch, targets in train_loader:
                data_batch, targets = data_batch.to(device), targets.to(device)

                def closure(batched_params, _data=data_batch, _targets=targets):
                    nonlocal fwd_pass_count
                    fwd_pass_count += next(iter(batched_params.values())).shape[0]
                    return evaluator.evaluate(batched_params, _data, _targets)

                optimizer.step(closure)

                with torch.no_grad():
                    output = model(data_batch)
                    loss = loss_fn(output, targets).item()
                    epoch_correct += (output.argmax(dim=1) == targets).sum().item()
                    epoch_total += targets.size(0)
                epoch_loss += loss
                step_count += 1

                # Per-20-step fine-grained tracking
                if step_count % 20 == 0:
                    step_test_acc = evaluate_accuracy(model, test_loader, device=device)
                    step_logs.append({
                        "step": step_count,
                        "epoch": epoch + 1,
                        "test_accuracy": step_test_acc,
                        "loss": loss,
                        "wall_time": time.time() - start_time,
                    })

            train_acc = epoch_correct / max(epoch_total, 1)
            test_acc = evaluate_accuracy(model, test_loader, device=device)
            selection_acc = (
                evaluate_accuracy(model, selection_loader, device=device)
                if selection_loader is not test_loader
                else test_acc
            )
            if selection_acc > best_accuracy:
                best_accuracy = selection_acc
                best_state_dict = copy.deepcopy(model.state_dict())
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(len(train_loader), 1)

            epoch_logs.append({
                "epoch": epoch + 1,
                "accuracy": test_acc,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "loss": avg_loss,
                "time": epoch_time,
                "wall_time": time.time() - start_time,
            })
            print(f"    Epoch {epoch+1}/{epochs} | train={train_acc*100:.1f}% | test={test_acc*100:.1f}% | loss={avg_loss:.4f}")

    wall_time = time.time() - start_time
    last_epoch_acc = test_acc
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_acc = evaluate_accuracy(model, test_loader, device=device)

    filepath = save_result(
        benchmark=BENCHMARK,
        method="polystep",
        seed=seed,
        metrics={
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "last_epoch_accuracy": last_epoch_acc,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": fwd_pass_count,
            "total_steps": step_count,
        },
        hyperparameters={
            **PSTORCH_CONFIG,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "solver": "softmax",
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: CMA-ES (pycma)
# ---------------------------------------------------------------------------

def run_cmaes(seed, device, results_dir, dry_run=False):
    """Train Hard MoE with CMA-ES (pycma)."""
    import cma

    generations = 10 if dry_run else CMAES_CONFIG["generations"]
    popsize = CMAES_CONFIG["popsize"]

    set_seed(seed)
    model = HardMoENet(num_experts=4).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    data = generate_multidomain_data(data_dir="data/", batch_size=BATCH_SIZE)
    train_loader, test_loader = data["train_loader"], data["test_loader"]

    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return batch

    n_params = sum(p.numel() for p in model.parameters())

    def eval_cost(x, gen_inputs, gen_targets):
        flat = torch.tensor(x, dtype=torch.float32, device=device)
        set_flat_params(model, flat)
        with torch.no_grad():
            outputs = model(gen_inputs)
            loss = loss_fn(outputs, gen_targets)
        return loss.item()

    x0 = load_flat_params(model).cpu().numpy()
    sigma0 = CMAES_CONFIG["stdev_init"]

    opts = {
        "maxiter": generations,
        "popsize": popsize,
        "seed": seed,
        "verbose": -9,
    }
    if n_params >= 1000:
        opts["CMA_diagonal"] = True

    best_accuracy = 0.0
    epoch_logs = []
    total_evals = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        gen = 0
        while not es.stop():
            solutions = es.ask()
            gen_batch = get_batch()
            gen_inputs = gen_batch[0].to(device)
            gen_targets = gen_batch[1].to(device)
            costs = [eval_cost(s, gen_inputs, gen_targets) for s in solutions]
            es.tell(solutions, costs)
            total_evals += len(solutions)
            gen += 1

            if gen % max(1, generations // 10) == 0 or es.stop():
                set_flat_params(model, torch.tensor(es.result.xbest, dtype=torch.float32, device=device))
                test_acc = evaluate_accuracy(model, test_loader, device=device)
                best_accuracy = max(best_accuracy, test_acc)
                elapsed = time.time() - start_time
                epoch_logs.append({
                    "epoch": gen,
                    "accuracy": test_acc,
                    "loss": es.result.fbest,
                    "time": elapsed,
                })
                print(f"    Gen {gen}/{generations} | acc={test_acc*100:.1f}% | loss={es.result.fbest:.4f}")

    wall_time = time.time() - start_time

    set_flat_params(model, torch.tensor(es.result.xbest, dtype=torch.float32, device=device))
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    filepath = save_result(
        benchmark=BENCHMARK,
        method="cmaes",
        seed=seed,
        metrics={
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": total_evals,
            "total_steps": gen,
        },
        hyperparameters={
            **CMAES_CONFIG,
            "generations_actual": gen,
            "CMA_diagonal": n_params >= 1000,
        },
        epoch_logs=epoch_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: OpenAI-ES
# ---------------------------------------------------------------------------

def run_openai_es(seed, device, results_dir, dry_run=False):
    """Train Hard MoE with OpenAI Evolution Strategy."""
    generations = 10 if dry_run else OPENAI_ES_CONFIG["generations"]

    set_seed(seed)
    model = HardMoENet(num_experts=4).to(device)

    data = generate_multidomain_data(data_dir="data/", batch_size=BATCH_SIZE)
    train_loader, test_loader = data["train_loader"], data["test_loader"]

    result = train_openai_es(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        sigma=OPENAI_ES_CONFIG["sigma"],
        lr=OPENAI_ES_CONFIG["lr"],
        population_size=OPENAI_ES_CONFIG["population_size"],
        generations=generations,
        lr_decay=OPENAI_ES_CONFIG.get("lr_decay", False),
        weight_decay=OPENAI_ES_CONFIG.get("weight_decay", 0.0),
        fitness_shaping=OPENAI_ES_CONFIG.get("fitness_shaping", "zscore"),
        device=device,
        seed=seed,
    )

    filepath = save_result(
        benchmark=BENCHMARK,
        method="openai_es",
        seed=seed,
        metrics=result["metrics"],
        hyperparameters=result["hyperparameters"],
        epoch_logs=result["epoch_logs"],
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: SPSA
# ---------------------------------------------------------------------------

def run_spsa(seed, device, results_dir, dry_run=False):
    """Train Hard MoE with SPSA."""
    max_iters = 100 if dry_run else SPSA_CONFIG["max_iters"]

    set_seed(seed)
    model = HardMoENet(num_experts=4).to(device)

    data = generate_multidomain_data(data_dir="data/", batch_size=BATCH_SIZE)
    train_loader, test_loader = data["train_loader"], data["test_loader"]

    result = train_spsa(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        a=SPSA_CONFIG["a"],
        c=SPSA_CONFIG["c"],
        alpha=SPSA_CONFIG.get("alpha", 0.602),
        gamma=SPSA_CONFIG.get("gamma", 0.101),
        max_iters=max_iters,
        device=device,
        seed=seed,
    )

    filepath = save_result(
        benchmark=BENCHMARK,
        method="spsa",
        seed=seed,
        metrics=result["metrics"],
        hyperparameters=result["hyperparameters"],
        epoch_logs=result["epoch_logs"],
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

METHOD_RUNNERS = {
    "polystep": lambda seed, device, results_dir, dry_run: run_polystep(seed, device, results_dir, dry_run=dry_run),
    "cmaes": run_cmaes,
    "openai_es": run_openai_es,
    "spsa": run_spsa,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Hard MoE (Mixture-of-Experts) benchmark: methods x seeds"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["polystep", "cmaes", "openai_es", "spsa"],
        help="Methods to run (default: all 4)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Seeds (default: {SEEDS})",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--results-dir",
        default=os.path.join("experiments", "results", "softmax", "main"),
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of polystep epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 1 epoch (polystep) / 10 generations (ES) for testing",
    )
    parser.add_argument(
        "--allow-test-leakage", action="store_true",
        help=(
            "Legacy mode: select best_state_dict on the test set instead "
            "of a held-out validation slice. Default is honest protocol "
            "(val-selected). Use only for bit-for-bit reproduction "
            "of earlier results."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("Hard MoE (Mixture-of-Experts) Benchmark")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    if args.dry_run:
        print("  Mode: DRY RUN (minimal epochs/generations)")
    print()

    os.makedirs(args.results_dir, exist_ok=True)

    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(args.results_dir, f"{BENCHMARK}_{method}_{seed}.json")
            if os.path.exists(output_file):
                print(f"  Skipping {method} seed={seed} (result exists)")
                continue
            print(f"  Running {method} seed={seed}...")
            try:
                if method == "polystep":
                    run_polystep(seed, args.device, args.results_dir, epochs=args.epochs,
                                dry_run=args.dry_run, audit_no_leakage=not args.allow_test_leakage)
                elif method in METHOD_RUNNERS:
                    METHOD_RUNNERS[method](seed, args.device, args.results_dir, args.dry_run)
                else:
                    print(f"    Unknown method: {method}")
            except Exception as e:
                print(f"    ERROR: {method} seed={seed} failed: {e}")
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\nDone! Results in {args.results_dir}")


if __name__ == "__main__":
    main()
