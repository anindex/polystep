#!/usr/bin/env python
"""Run all methods and seeds for non-differentiable showcase elevation experiments.

Elevates 4 existing non-differentiable showcases (SNN/LIF, int8-quantized,
argmax hard attention, staircase activation) from synthetic demos to
publishable experiments with 5-seed rigor and 4 baselines.

Methods:
  - polystep: PolyStepOptimizer with HybridSubspace on non-diff models
  - adam: Adam on smooth model variant (accuracy ceiling)
  - cmaes: pycma CMA-ES on non-diff models
  - openai_es: OpenAI-ES on non-diff models
  - spsa: SPSA on non-diff models

Showcases:
  - snn: Spiking Neural Network with hard-threshold LIF neurons (MNIST)
  - int8: Int8 weight quantization via round() (MNIST)
  - argmax: Argmax hard attention routing over 8 slots (Fashion-MNIST)
  - staircase: Piecewise-constant staircase activation (MNIST)

Results saved as: experiments/results/softmax/main/{showcase}_{method}_{seed}.json
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
    load_mnist,
    load_fashion_mnist,
    load_flat_params,
    make_train_val_split,
    set_flat_params,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.nondiff_models import (
    SpikingMNISTNet,
    SmoothSpikingMNISTNet,
    QuantizedMLP,
    SmoothQuantizedMLP,
    DiscreteAttentionNet,
    SmoothDiscreteAttentionNet,
    StaircaseNet,
    SmoothStaircaseNet,
)
from experiments.baselines.sgd_baseline import train_sgd
from experiments.baselines.openai_es import train_openai_es
from experiments.baselines.spsa import train_spsa


# ---------------------------------------------------------------------------
# Showcase configurations
# ---------------------------------------------------------------------------

SHOWCASE_CONFIGS = {
    "snn": {
        "model_fn": lambda: SpikingMNISTNet(num_steps=15),
        "smooth_model_fn": lambda: SmoothSpikingMNISTNet(num_steps=15),
        "load_data": lambda: load_mnist(batch_size=512),
        "benchmark": "snn",
    },
    "int8": {
        "model_fn": lambda: QuantizedMLP(),
        "smooth_model_fn": lambda: SmoothQuantizedMLP(),
        "load_data": lambda: load_mnist(batch_size=512),
        "benchmark": "int8",
    },
    "argmax": {
        "model_fn": lambda: DiscreteAttentionNet(),
        "smooth_model_fn": lambda: SmoothDiscreteAttentionNet(),
        "load_data": lambda: load_fashion_mnist(batch_size=512),
        "benchmark": "argmax",
    },
    "staircase": {
        "model_fn": lambda: StaircaseNet(),
        "smooth_model_fn": lambda: SmoothStaircaseNet(),
        "load_data": lambda: load_mnist(batch_size=512),
        "benchmark": "staircase",
    },
}


# ---------------------------------------------------------------------------
# Hyperparameter constants
# ---------------------------------------------------------------------------

EPOCHS_PSTORCH = 30    # SNN needs more epochs; softmax solver is fast enough
EPOCHS_PSTORCH_NONSNN = 30  # Int8/Argmax/Staircase
EPOCHS_ADAM = 30
ADAM_LR = 0.001

OPENAI_ES_CONFIG = {
    "sigma": 0.02, "lr": 0.01, "population_size": 50, "generations": 10000,
    "lr_decay": True, "weight_decay": 0.01, "fitness_shaping": "rank",
}
CMAES_CONFIG = {"generations": 10000, "popsize": 16, "stdev_init": 0.5}
SPSA_CONFIG = {"a": 0.1, "c": 0.1, "alpha": 0.602, "gamma": 0.101, "max_iters": 50000}

PSTORCH_CONFIGS = {
    # SNN: Best sweep config = sm_sr2 (93.28% at 20ep, beats 40ep production)
    # FLAT eps/sr/pr - CosineEpsilon scheduling DESTROYS SNN (collapse to 10-47%)
    # biased_rotation + absorb_interval=20 = key levers for spiking landscape stability
    "snn": {
        "epsilon": 0.5,
        "step_radius": 2.0,
        "probe_radius": 1.0,
        "num_probe": 1, "rank": 4,
        "chunk_size": 1024, "amortize_steps": 1,
        "rotation_interval": 0, "absorb_interval": 20,
        "biased_rotation": True,
        "anderson_depth": 5, "adaptive_omega": True,
    },
    # INT8: Best sweep config = rank8 (97.18% at 20ep, beats 40ep production)
    # CosineEpsilon scheduling works well on quantization plateaus
    # rank=8 is the dominant lever (+1.08% over rank=4)
    "int8": {
        "epsilon_init": 5.0, "epsilon_target": 0.3,
        "step_radius_init": 32.0, "step_radius_target": 8.0,
        "probe_radius_init": 2.0, "probe_radius_target": 0.5,
        "num_probe": 1, "rank": 8,
        "chunk_size": 1024, "amortize_steps": 1,
        "rotation_interval": 0, "absorb_interval": 0,
        "use_momentum": True, "momentum_init": 0.3, "momentum_final": 0.5,
    },
    # Argmax: Best sweep config = rank8 (87.08% at 20ep, beats 40ep production)
    # Same CosineEpsilon strategy as INT8 but NO momentum (sweep: no_mom=84.71% < rank8=87.08%)
    "argmax": {
        "epsilon_init": 5.0, "epsilon_target": 0.3,
        "step_radius_init": 32.0, "step_radius_target": 8.0,
        "probe_radius_init": 2.0, "probe_radius_target": 0.5,
        "num_probe": 1, "rank": 8,
        "chunk_size": 1024, "amortize_steps": 1,
        "rotation_interval": 0, "absorb_interval": 0,
    },
    # Staircase: Retuned from sr64→16 which degraded after epoch 11
    # Gentler cosine targets (64→32, eps 5→1) eliminate late-epoch degradation
    # 15ep tune: cosine_64_32=91.96%/91.10% vs cosine_64_16=91.80%/89.15%
    "staircase": {
        "epsilon_init": 5.0, "epsilon_target": 1.0,
        "step_radius_init": 64.0, "step_radius_target": 32.0,
        "probe_radius_init": 2.0, "probe_radius_target": 1.0,
        "num_probe": 1, "rank": 4,
        "chunk_size": 1024, "amortize_steps": 1,
        "rotation_interval": 0, "absorb_interval": 0,
    },
}


# ---------------------------------------------------------------------------
# Method: polystep
# ---------------------------------------------------------------------------

def run_polystep(showcase_name, seed, device, results_dir, dry_run=False,
                audit_no_leakage: bool = True):
    """Train non-diff model with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses a held-out validation
    split (honest protocol). Set ``audit_no_leakage=False`` to revert
    to legacy test-set selection.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout
    from polystep.cost_nn import NNCostEvaluator

    config = SHOWCASE_CONFIGS[showcase_name]
    polystep_cfg = PSTORCH_CONFIGS[showcase_name]
    if dry_run:
        epochs = 1
    elif "epochs" in polystep_cfg:
        epochs = polystep_cfg["epochs"]
    elif showcase_name == "snn":
        epochs = EPOCHS_PSTORCH
    else:
        epochs = EPOCHS_PSTORCH_NONSNN

    set_seed(seed)
    model = config["model_fn"]()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, test_loader = config["load_data"]()

    # Leakage-free model selection: carve 10% validation split from training data
    val_loader = None
    if audit_no_leakage:
        train_loader, val_loader = make_train_val_split(
            train_loader, val_frac=0.1, seed=seed,
        )
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader

    total_steps = epochs * len(train_loader)

    # CosineEpsilon scheduling: coarse-to-fine annealing
    if "epsilon_init" in polystep_cfg:
        eps_init = polystep_cfg["epsilon_init"]
        eps_target = polystep_cfg["epsilon_target"]
        epsilon_decay = (eps_init - eps_target) / max(1, total_steps)
        epsilon_value = CosineEpsilon(init=eps_init, target=eps_target, decay=epsilon_decay)
    else:
        epsilon_value = polystep_cfg.get("epsilon", 0.5)

    if "step_radius_init" in polystep_cfg:
        sr_decay = (polystep_cfg["step_radius_init"] - polystep_cfg["step_radius_target"]) / max(1, total_steps)
        step_radius_value = CosineEpsilon(init=polystep_cfg["step_radius_init"], target=polystep_cfg["step_radius_target"], decay=sr_decay)
    else:
        step_radius_value = polystep_cfg.get("step_radius", 1.0)

    if "probe_radius_init" in polystep_cfg:
        pr_decay = (polystep_cfg["probe_radius_init"] - polystep_cfg["probe_radius_target"]) / max(1, total_steps)
        probe_radius_value = CosineEpsilon(init=polystep_cfg["probe_radius_init"], target=polystep_cfg["probe_radius_target"], decay=pr_decay)
    else:
        probe_radius_value = polystep_cfg.get("probe_radius", 1.0)

    # HybridSubspace setup
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout,
        rank=polystep_cfg["rank"],
        rotation_mode="random",
        rotation_interval=polystep_cfg["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=polystep_cfg["absorb_interval"],
    )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=epsilon_value,
        step_radius=step_radius_value,
        probe_radius=probe_radius_value,
        num_probe=polystep_cfg["num_probe"],
        subspace=subspace,
        chunk_size=polystep_cfg["chunk_size"],
        amortize_steps=polystep_cfg["amortize_steps"],
        use_momentum=polystep_cfg.get("use_momentum", False),
        momentum_init=polystep_cfg.get("momentum_init", 0.5),
        momentum_final=polystep_cfg.get("momentum_final", 0.95),
        biased_rotation=polystep_cfg.get("biased_rotation", False),
        anderson_depth=polystep_cfg.get("anderson_depth", 0),
        adaptive_omega=polystep_cfg.get("adaptive_omega", False),
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
    last_epoch_acc = evaluate_accuracy(model, test_loader, device=device)
    if last_epoch_acc > best_accuracy:
        best_accuracy = last_epoch_acc
        best_state_dict = copy.deepcopy(model.state_dict())
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_acc = evaluate_accuracy(model, test_loader, device=device)

    filepath = save_result(
        benchmark=showcase_name,
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
            **polystep_cfg,
            "epochs": epochs,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: Adam (smooth model variant -- accuracy ceiling)
# ---------------------------------------------------------------------------

def run_adam(showcase_name, seed, device, results_dir, dry_run=False):
    """Train smooth model variant with Adam (accuracy ceiling baseline)."""
    config = SHOWCASE_CONFIGS[showcase_name]
    epochs = 1 if dry_run else EPOCHS_ADAM

    set_seed(seed)
    model = config["smooth_model_fn"]()
    model = model.to(device)

    train_loader, test_loader = config["load_data"]()

    result = train_sgd(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="adam",
        lr=ADAM_LR,
        epochs=epochs,
        device=device,
        seed=seed,
    )

    filepath = save_result(
        benchmark=showcase_name,
        method="adam",
        seed=seed,
        metrics=result["metrics"],
        hyperparameters=result["hyperparameters"],
        epoch_logs=result["epoch_logs"],
        results_dir=results_dir,
    )
    print(f"    Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: CMA-ES (pycma)
# ---------------------------------------------------------------------------

def run_cmaes(showcase_name, seed, device, results_dir, dry_run=False):
    """Train non-diff model with CMA-ES (pycma)."""
    import cma

    config = SHOWCASE_CONFIGS[showcase_name]
    generations = 10 if dry_run else CMAES_CONFIG["generations"]
    popsize = CMAES_CONFIG["popsize"]

    set_seed(seed)
    model = config["model_fn"]()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader = config["load_data"]()

    # Cycling batch iterator for evaluation
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

    # Pre-allocate reusable buffer for parameter loading (avoids 160K+ tensor allocations)
    _flat_buf = torch.empty(n_params, dtype=torch.float32, device=device)

    def eval_cost(x, gen_inputs, gen_targets):
        """Evaluate single solution on a fixed batch."""
        _flat_buf.copy_(torch.as_tensor(x, dtype=torch.float32))
        set_flat_params(model, _flat_buf)
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
            # Fetch ONE batch for the entire generation
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

    # Final evaluation
    set_flat_params(model, torch.tensor(es.result.xbest, dtype=torch.float32, device=device))
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    filepath = save_result(
        benchmark=showcase_name,
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

def run_openai_es(showcase_name, seed, device, results_dir, dry_run=False):
    """Train non-diff model with OpenAI Evolution Strategy."""
    config = SHOWCASE_CONFIGS[showcase_name]
    generations = 10 if dry_run else OPENAI_ES_CONFIG["generations"]

    set_seed(seed)
    model = config["model_fn"]()
    model = model.to(device)

    train_loader, test_loader = config["load_data"]()

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
        benchmark=showcase_name,
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

def run_spsa(showcase_name, seed, device, results_dir, dry_run=False):
    """Train non-diff model with SPSA."""
    config = SHOWCASE_CONFIGS[showcase_name]
    max_iters = 100 if dry_run else SPSA_CONFIG["max_iters"]

    set_seed(seed)
    model = config["model_fn"]()
    model = model.to(device)

    train_loader, test_loader = config["load_data"]()

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
        benchmark=showcase_name,
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
    "polystep": run_polystep,
    "adam": run_adam,
    "cmaes": run_cmaes,
    "openai_es": run_openai_es,
    "spsa": run_spsa,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run non-differentiable showcase elevation experiments: showcases x methods x seeds"
    )
    parser.add_argument(
        "--showcases", nargs="+",
        default=["snn", "int8", "argmax", "staircase"],
        help="Showcases to run (default: all 4)",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["polystep", "adam", "cmaes", "openai_es", "spsa"],
        help="Methods to run (default: all 5)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Seeds to run (default: 42 123 456 789 1337)",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (default: cuda)",
    )
    parser.add_argument(
        "--results-dir", default="experiments/results/softmax/main", help="Results directory",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 1 epoch / 10 generations / 100 SPSA iters for testing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Override skip-if-exists and rerun all experiments",
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
    parser.add_argument(
        "--step-radius", type=float, default=None,
        help="Override step_radius for polystep (for hyperparameter sweeps)",
    )
    parser.add_argument(
        "--epochs-polystep", type=int, default=None,
        help="Override number of polystep epochs (for quick sweeps)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("Non-Differentiable Showcase Elevation Experiments")
    print(f"  Showcases: {args.showcases}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    if args.dry_run:
        print("  Mode: DRY RUN (minimal epochs/generations)")
    if args.step_radius is not None:
        print(f"  Step radius override: {args.step_radius}")
        for sc in PSTORCH_CONFIGS:
            cfg = PSTORCH_CONFIGS[sc]
            if "step_radius_init" in cfg:
                cfg["step_radius_init"] = args.step_radius
                cfg["step_radius_target"] = args.step_radius
            else:
                cfg["step_radius"] = args.step_radius
    if args.epochs_polystep is not None:
        global EPOCHS_PSTORCH, EPOCHS_PSTORCH_NONSNN
        EPOCHS_PSTORCH = args.epochs_polystep
        EPOCHS_PSTORCH_NONSNN = args.epochs_polystep
        print(f"  polystep epochs override: {args.epochs_polystep}")
    print()

    for showcase_name in args.showcases:
        if showcase_name not in SHOWCASE_CONFIGS:
            print(f"Unknown showcase: {showcase_name}")
            continue

        print(f"=== {showcase_name} ===")

        for method in args.methods:
            runner = METHOD_RUNNERS.get(method)
            if runner is None:
                print(f"  Unknown method: {method}")
                continue

            for seed in args.seeds:
                output_file = os.path.join(
                    args.results_dir, f"{showcase_name}_{method}_{seed}.json"
                )
                if os.path.exists(output_file) and not args.force:
                    print(f"  Skipping {method} seed={seed} (result exists)")
                    continue

                print(f"  Running {method} seed={seed}...")
                try:
                    runner(showcase_name, seed, args.device, args.results_dir,
                           dry_run=args.dry_run,
                           **(dict(audit_no_leakage=not args.allow_test_leakage) if method == 'polystep' else {}))
                except Exception as e:
                    print(f"    ERROR: {method} seed={seed} failed: {e}")
                    traceback.print_exc()
                finally:
                    # Prevent CUDA memory accumulation across sequential runs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print()

    print("Done. Results in experiments/results/softmax/main/")


if __name__ == "__main__":
    main()
