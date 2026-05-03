#!/usr/bin/env python
"""Run all methods and seeds for MNIST benchmark.

Methods: polystep, cmaes, openai_es, spsa, adam
Model: MNISTNet MLP (784->128->10, ~102K params) for all methods.
Data: Standard MNIST (28x28 grayscale, 10 classes)

Hyperparameters are hardcoded constants for reproducibility.
Results are saved as JSON files in experiments/results/softmax/main/.

Usage:
    python experiments/runners/run_mnist.py
    python experiments/runners/run_mnist.py --methods polystep adam --seeds 42 123
    python experiments/runners/run_mnist.py --device cpu
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from experiments.runners.common import (
    SEEDS,
    MNISTNet,
    evaluate_accuracy,
    load_mnist,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.baselines.openai_es import train_openai_es
from experiments.baselines.spsa import train_spsa
from experiments.baselines.sgd_baseline import train_sgd


# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

BENCHMARK = "mnist"
BATCH_SIZE = 512
EPOCHS = 30

# polystep hyperparameters (HybridSubspace)
# Best softmax configuration (HybridSubspace + cosine schedules)
#   rank=8 (sweet spot, 16 vertices), K=1 (K>1 useless for softmax),
#   amort=3 (EMA smoothing), eps=10.0->0.1 (higher init = better exploration)
#   Cosine-scheduled step_radius (5->1) and probe_radius (10->2)
#   absorb_interval=0 (continuous absorb), no biased_rotation (no effect on MNIST)
#   NO momentum - sweep showed no_mom (95.70%) beats with-mom (94.82%)
PSTORCH_CONFIG = {
    "rank": 8,
    "step_radius_init": 5.0,
    "step_radius_target": 1.0,
    "probe_radius_init": 10.0,
    "probe_radius_target": 2.0,
    "epsilon_init": 10.0,
    "epsilon_target": 0.1,
    "rotation_interval": 0,
    "absorb_interval": 0,
    "num_probe": 1,  # K>1 adds zero benefit for softmax; uses fused path
    "chunk_size": 1024,
    "amortize_steps": 3,
    "amortize_ema": 0.7,
}

# OpenAI ES hyperparameters (Salimans et al. 2017 + compute budget matching)
# polystep does ~2360 steps on MNIST (20 ep x 118 batches), each evaluating ~50 candidates
# Give ES comparable: 2000 gen x 50 pop = 100K evals
# Enable lr_decay (linear to 0) and weight_decay (0.01) per reference algorithm
OPENAI_ES_CONFIG = {
    "sigma": 0.02,
    "lr": 0.01,
    "population_size": 50,
    "generations": 2000,
    "lr_decay": True,
    "weight_decay": 0.01,
    "fitness_shaping": "rank",
}

# SPSA hyperparameters (compute budget matched)
# 10000 iters x 2 evals = 20K evals, gain sequences tuned for MNIST MLP
# a=0.1 (original stable value), c=0.1, alpha/gamma at Spall defaults
# a=0.1 is stable; larger values diverge on this problem
SPSA_CONFIG = {
    "a": 0.1,
    "c": 0.1,
    "alpha": 0.602,
    "gamma": 0.101,
    "max_iters": 10000,
}

# Adam hyperparameters
ADAM_CONFIG = {
    "lr": 0.001,
    "epochs": EPOCHS,
}

# CMA-ES hyperparameters (compute budget matched)
# 2000 gen x 16 pop = 32K evals
CMAES_CONFIG = {
    "generations": 2000,
    "popsize": 16,
    "stdev_init": 0.5,
}


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_polystep(seed, device, train_loader, test_loader, results_dir, solver=None,
                audit_no_leakage: bool = True, val_loader=None):
    """Train MNIST with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses a held-out validation
    split (honest protocol). Set ``audit_no_leakage=False`` to revert
    to the legacy behavior where ``best_state_dict`` was selected on
    the test set.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    set_seed(seed)
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader
    selection_label = "val" if (audit_no_leakage and val_loader is not None) else "test"

    total_steps = EPOCHS * len(train_loader)
    epsilon_decay = (
        (PSTORCH_CONFIG["epsilon_init"] - PSTORCH_CONFIG["epsilon_target"])
        / max(1, total_steps)
    )
    sr_decay = (PSTORCH_CONFIG["step_radius_init"] - PSTORCH_CONFIG["step_radius_target"]) / max(1, total_steps)
    pr_decay = (PSTORCH_CONFIG["probe_radius_init"] - PSTORCH_CONFIG["probe_radius_target"]) / max(1, total_steps)

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
        epsilon=CosineEpsilon(
            init=PSTORCH_CONFIG["epsilon_init"],
            target=PSTORCH_CONFIG["epsilon_target"],
            decay=epsilon_decay,
        ),
        step_radius=CosineEpsilon(
            init=PSTORCH_CONFIG["step_radius_init"],
            target=PSTORCH_CONFIG["step_radius_target"],
            decay=sr_decay,
        ),
        probe_radius=CosineEpsilon(
            init=PSTORCH_CONFIG["probe_radius_init"],
            target=PSTORCH_CONFIG["probe_radius_target"],
            decay=pr_decay,
        ),
        num_probe=PSTORCH_CONFIG["num_probe"],
        subspace=subspace,
        chunk_size=PSTORCH_CONFIG.get("chunk_size", 1024),
        amortize_steps=PSTORCH_CONFIG.get("amortize_steps", 0),
        amortize_ema=PSTORCH_CONFIG.get("amortize_ema", 0.0),
        use_momentum=PSTORCH_CONFIG.get("use_momentum", False),
        momentum_init=PSTORCH_CONFIG.get("momentum_init", 0.5),
        momentum_final=PSTORCH_CONFIG.get("momentum_final", 0.95),
        solver=solver,
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
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_start = time.time()

            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)

                def closure(batched_params, _data=data, _targets=targets):
                    nonlocal fwd_pass_count
                    fwd_pass_count += next(iter(batched_params.values())).shape[0]
                    return evaluator.evaluate(batched_params, _data, _targets)

                optimizer.step(closure)

                with torch.no_grad():
                    output = model(data)
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
            # Pick best_state_dict on `selection_loader` (validation
            # split when audit_no_leakage=True, test otherwise). The
            # final_accuracy reported in the JSON is always test, but
            # the model selection that drives `best_state_dict` must
            # not peek at the test set.
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
                f"{selection_label}_accuracy": selection_acc,
                "loss": avg_loss,
                "time": epoch_time,
                "wall_time": time.time() - start_time,
            })
            print(
                f"    Epoch {epoch+1}/{EPOCHS} | train={train_acc*100:.1f}% | "
                f"test={test_acc*100:.1f}% | {selection_label}-best={best_accuracy*100:.1f}% | "
                f"loss={avg_loss:.4f}"
            )

    wall_time = time.time() - start_time
    last_epoch_acc = (
        evaluate_accuracy(model, selection_loader, device=device)
        if selection_loader is not test_loader
        else evaluate_accuracy(model, test_loader, device=device)
    )
    if last_epoch_acc > best_accuracy:
        best_accuracy = last_epoch_acc
        best_state_dict = copy.deepcopy(model.state_dict())
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_acc = evaluate_accuracy(model, test_loader, device=device)

    result = save_result(
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
        hyperparameters=PSTORCH_CONFIG,
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {result}")


def run_cmaes(seed, device, train_loader, test_loader, results_dir):
    """Train MNIST with CMA-ES (EvoTorch)."""
    try:
        from polystep.benchmarks.baselines import train_cmaes, has_evotorch
    except ImportError:
        print("    Skipping cmaes (polystep.benchmarks.baselines not importable)")
        return

    if not has_evotorch():
        print("    Skipping cmaes (EvoTorch not installed)")
        return

    set_seed(seed)
    model = MNISTNet().to(device)

    # Extract data tensors for CMA-ES interface
    train_data_list, train_labels_list = [], []
    for data, labels in train_loader:
        train_data_list.append(data)
        train_labels_list.append(labels)
    train_data = torch.cat(train_data_list).to(device)
    train_labels = torch.cat(train_labels_list).to(device)

    test_data_list, test_labels_list = [], []
    for data, labels in test_loader:
        test_data_list.append(data)
        test_labels_list.append(labels)
    test_data = torch.cat(test_data_list).to(device)
    test_labels = torch.cat(test_labels_list).to(device)

    with track_gpu_memory() as mem:
        start_time = time.time()
        result = train_cmaes(
            model=model,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            generations=CMAES_CONFIG["generations"],
            popsize=CMAES_CONFIG["popsize"],
            stdev_init=CMAES_CONFIG["stdev_init"],
            device=device,
            verbose=True,
        )
        wall_time = time.time() - start_time

    filepath = save_result(
        benchmark=BENCHMARK,
        method="cmaes",
        seed=seed,
        metrics={
            "final_accuracy": result.final_accuracy,
            "best_accuracy": result.best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": result.function_evals,
            "total_steps": result.total_steps,
        },
        hyperparameters=CMAES_CONFIG,
        epoch_logs=result.epoch_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


def run_openai_es(seed, device, train_loader, test_loader, results_dir):
    """Train MNIST with OpenAI Evolution Strategy."""
    set_seed(seed)  # Seed before model init for deterministic weights
    model = MNISTNet().to(device)

    result = train_openai_es(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        sigma=OPENAI_ES_CONFIG["sigma"],
        lr=OPENAI_ES_CONFIG["lr"],
        population_size=OPENAI_ES_CONFIG["population_size"],
        generations=OPENAI_ES_CONFIG["generations"],
        lr_decay=OPENAI_ES_CONFIG.get("lr_decay", False),
        weight_decay=OPENAI_ES_CONFIG.get("weight_decay", 0.0),
        fitness_shaping=OPENAI_ES_CONFIG.get("fitness_shaping", "zscore"),
        device=device,
        seed=seed,
    )

    result["benchmark"] = BENCHMARK
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


def run_spsa(seed, device, train_loader, test_loader, results_dir):
    """Train MNIST with SPSA."""
    set_seed(seed)  # Seed before model init for deterministic weights
    model = MNISTNet().to(device)

    result = train_spsa(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        a=SPSA_CONFIG["a"],
        c=SPSA_CONFIG["c"],
        alpha=SPSA_CONFIG.get("alpha", 0.602),
        gamma=SPSA_CONFIG.get("gamma", 0.101),
        max_iters=SPSA_CONFIG["max_iters"],
        device=device,
        seed=seed,
    )

    result["benchmark"] = BENCHMARK
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


def run_adam(seed, device, train_loader, test_loader, results_dir):
    """Train MNIST with Adam (gradient-based ceiling)."""
    set_seed(seed)
    model = MNISTNet().to(device)

    result = train_sgd(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="adam",
        lr=ADAM_CONFIG["lr"],
        epochs=ADAM_CONFIG["epochs"],
        device=device,
        seed=seed,
    )

    result["benchmark"] = BENCHMARK
    filepath = save_result(
        benchmark=BENCHMARK,
        method="adam",
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
    "polystep": run_polystep,
    "cmaes": run_cmaes,
    "openai_es": run_openai_es,
    "spsa": run_spsa,
    "adam": run_adam,
}


def run_method(method, seed, device, results_dir, data_dir, solver=None,
               audit_no_leakage: bool = True):
    """Run a single method+seed combination."""
    from experiments.runners.common import make_train_val_split

    train_loader, test_loader = load_mnist(
        data_dir=data_dir, batch_size=BATCH_SIZE,
    )
    val_loader = None
    if audit_no_leakage:
        train_loader, val_loader = make_train_val_split(
            train_loader, val_frac=0.1, seed=seed,
        )
    runner = METHOD_RUNNERS.get(method)
    if runner is None:
        print(f"    Unknown method: {method}")
        return
    if method == 'polystep':
        runner(seed, device, train_loader, test_loader, results_dir,
               solver=solver, audit_no_leakage=audit_no_leakage,
               val_loader=val_loader)
    else:
        runner(seed, device, train_loader, test_loader, results_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MNIST benchmark: all methods x all seeds"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["polystep", "cmaes", "openai_es", "spsa", "adam"],
        help="Methods to run (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to run (default: 42 123 456 789 1337)",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--results-dir", default="experiments/results/softmax/main", help="Results directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--solver", choices=["softmax", "sinkhorn"], default="softmax",
        help="Solver backend: softmax (default, used with subspace) or sinkhorn (full-space).",
    )
    parser.add_argument(
        "--allow-test-leakage", action="store_true",
        help=(
            "Legacy mode: select best_state_dict on the test set instead "
            "of a held-out validation slice. Default is honest protocol "
            "(val-selected). Use this flag only for bit-for-bit reproduction "
            "of earlier results."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("MNIST Benchmark")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print()

    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(
                args.results_dir, f"{BENCHMARK}_{method}_{seed}.json"
            )
            if os.path.exists(output_file):
                print(f"Skipping {method} seed={seed} (result exists)")
                continue
            print(f"Running {method} seed={seed}...")
            try:
                run_method(
                    method, seed, args.device, args.results_dir, args.data_dir,
                    solver=args.solver, audit_no_leakage=not args.allow_test_leakage,
                )
            except Exception as e:
                print(f"  ERROR: {method} seed={seed} failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\nDone. Results in experiments/results/softmax/main/mnist_*.json")


if __name__ == "__main__":
    main()
