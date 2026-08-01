#!/usr/bin/env python
"""Run all methods and seeds for non-differentiable showcase elevation experiments.

Elevates 4 existing non-differentiable showcases (SNN/LIF, int8-quantized,
argmax hard attention, staircase activation) from synthetic demos to
publishable experiments with 5-seed rigor and 4 baselines.

Methods:
  - polystep: PolyStepOptimizer with HybridSubspace on non-diff models
  - adam: Adam on smooth model variant (accuracy ceiling)
  - the six gradient-free baselines from ``polystep.baselines``: cma_es, openai_es,
    spsa, mezo, random_search, eggroll

``--fair`` gives every gradient-free method the same subspace, the same candidate
budget derived from PolyStep, and the same probe radius; ``--theory-mode`` runs the
configuration Theorem 4.2 analyses. See ``experiments/runners/fairness.py``.

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

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from experiments.runners.common import (
    SEEDS,
    evaluate_accuracy,
    load_mnist,
    load_fashion_mnist,
    make_train_val_split,
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
from experiments.runners.fairness import (
    FAIR_METHODS,
    apply_theory_mode,
    make_subspace,
    minibatch_loss,
    polystep_eval_budget,
    probe_scale_of,
    run_baseline,
    subspace_tag,
)


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


EPOCHS_PSTORCH = 30  # SNN needs more epochs; softmax solver is fast enough
EPOCHS_PSTORCH_NONSNN = 30  # Int8/Argmax/Staircase
EPOCHS_ADAM = 30
ADAM_LR = 0.001

# Free-running (non-fair) candidate budgets, preserving what the previous ad-hoc
# implementations spent: ES 10000 gen x 50 pop, CMA-ES 10000 gen x 16 pop, SPSA 50000
# iters x 2. `--fair` replaces all of these with one budget derived from PolyStep.
LEGACY_BUDGETS = {
    "openai_es": 500_000,
    "eggroll": 500_000,
    "cma_es": 160_000,
    "spsa": 100_000,
    "mezo": 100_000,
    "random_search": 100_000,
}

PSTORCH_CONFIGS = {
    # SNN: Best sweep config = sm_sr2 (93.28% at 20ep, beats 40ep production)
    # FLAT eps/sr/pr - CosineEpsilon scheduling DESTROYS SNN (collapse to 10-47%)
    # biased_rotation + absorb_interval=20 = key levers for spiking landscape stability
    "snn": {
        "epsilon": 0.5,
        "step_radius": 2.0,
        "probe_radius": 1.0,
        "num_probe": 1,
        "rank": 4,
        "chunk_size": 1024,
        "amortize_steps": 1,
        "rotation_interval": 0,
        "absorb_interval": 20,
        "biased_rotation": True,
    },
    # INT8: Best sweep config = rank8 (97.18% at 20ep, beats 40ep production)
    # CosineEpsilon scheduling works well on quantization plateaus
    # rank=8 is the dominant lever (+1.08% over rank=4)
    "int8": {
        "epsilon_init": 5.0,
        "epsilon_target": 0.3,
        "step_radius_init": 32.0,
        "step_radius_target": 8.0,
        "probe_radius_init": 2.0,
        "probe_radius_target": 0.5,
        "num_probe": 1,
        "rank": 8,
        "chunk_size": 1024,
        "amortize_steps": 1,
        "rotation_interval": 0,
        "absorb_interval": 0,
        "use_momentum": True,
        "momentum_init": 0.3,
        "momentum_final": 0.5,
    },
    # Argmax: Best sweep config = rank8 (87.08% at 20ep, beats 40ep production)
    # Same CosineEpsilon strategy as INT8 but NO momentum (sweep: no_mom=84.71% < rank8=87.08%)
    "argmax": {
        "epsilon_init": 5.0,
        "epsilon_target": 0.3,
        "step_radius_init": 32.0,
        "step_radius_target": 8.0,
        "probe_radius_init": 2.0,
        "probe_radius_target": 0.5,
        "num_probe": 1,
        "rank": 8,
        "chunk_size": 1024,
        "amortize_steps": 1,
        "rotation_interval": 0,
        "absorb_interval": 0,
    },
    # Staircase: Retuned from sr64->16 which degraded after epoch 11
    # Gentler cosine targets (64->32, eps 5->1) eliminate late-epoch degradation
    # 15ep tune: cosine_64_32=91.96%/91.10% vs cosine_64_16=91.80%/89.15%
    "staircase": {
        "epsilon_init": 5.0,
        "epsilon_target": 1.0,
        "step_radius_init": 64.0,
        "step_radius_target": 32.0,
        "probe_radius_init": 2.0,
        "probe_radius_target": 1.0,
        "num_probe": 1,
        "rank": 4,
        "chunk_size": 1024,
        "amortize_steps": 1,
        "rotation_interval": 0,
        "absorb_interval": 0,
    },
}


def _load_split(showcase_name, seed, audit_no_leakage):
    """Load the showcase data and carve the honest-protocol val split.

    Returns ``(train_loader, val_loader, test_loader)`` with ``val_loader``
    None only in legacy ``--allow-test-leakage`` mode. Every method goes
    through here, so no method can quietly select on the test set.
    """
    train_loader, test_loader = SHOWCASE_CONFIGS[showcase_name]["load_data"]()
    val_loader = None
    if audit_no_leakage:
        train_loader, val_loader = make_train_val_split(train_loader, val_frac=0.1, seed=seed)
    return train_loader, val_loader, test_loader


def _epochs_for(showcase_name, cfg, dry_run):
    if dry_run:
        return 1
    if "epochs" in cfg:
        return cfg["epochs"]
    return EPOCHS_PSTORCH if showcase_name == "snn" else EPOCHS_PSTORCH_NONSNN


def _build_polystep(model, seed, total_steps, cfg):
    """Build the subspace and optimizer from ``cfg``.

    Shared by the training run and by ``fair_eval_budget``, which needs the built
    optimizer to know how many candidates a step costs.
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon
    from polystep.transform import ParamLayout

    def sched(key, flat_default):
        # Scheduled when the config carries both endpoints; flat otherwise, which is
        # also what --theory-mode leaves behind.
        if f"{key}_init" not in cfg:
            return cfg.get(key, flat_default)
        init, target = cfg[f"{key}_init"], cfg[f"{key}_target"]
        return CosineEpsilon(init=init, target=target, decay=(init - target) / max(1, total_steps))

    layout = ParamLayout.from_module(model)
    subspace = make_subspace(
        layout,
        rank=cfg["rank"],
        seed=seed,
        rotation_mode="random",
        rotation_interval=cfg["rotation_interval"],
        absorb_mode="periodic",
        absorb_interval=cfg["absorb_interval"],
    )
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=sched("epsilon", 0.5),
        step_radius=sched("step_radius", 1.0),
        probe_radius=sched("probe_radius", 1.0),
        num_probe=cfg["num_probe"],
        subspace=subspace,
        chunk_size=cfg["chunk_size"],
        probe_radius_jitter=cfg.get("probe_radius_jitter", 0.0),
        probe_radius_jitter_dist=cfg.get("probe_radius_jitter_dist", "smooth"),
        step_radius_jitter=cfg.get("step_radius_jitter", 0.0),
        polytope_type=cfg.get("polytope_type", "simplex"),
        amortize_steps=cfg.get("amortize_steps", 1),
        amortize_ema=cfg.get("amortize_ema", 0.0),
        use_momentum=cfg.get("use_momentum", False),
        momentum_init=cfg.get("momentum_init", 0.5),
        momentum_final=cfg.get("momentum_final", 0.95),
        biased_rotation=cfg.get("biased_rotation", False),
        solver="softmax",
    )
    return layout, subspace, optimizer


def fair_eval_budget(showcase_name, seed, device, train_loader, epochs, cfg):
    """The shared candidate budget: what PolyStep spends over ``epochs`` epochs."""
    set_seed(seed)
    model = SHOWCASE_CONFIGS[showcase_name]["model_fn"]().to(device)
    _, _, opt = _build_polystep(model, seed, epochs * len(train_loader), cfg)
    return polystep_eval_budget(opt, epochs * len(train_loader))


def run_polystep(
    showcase_name,
    seed,
    device,
    results_dir,
    dry_run=False,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
):
    """Train non-diff model with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses a held-out validation
    split (honest protocol). Set ``audit_no_leakage=False`` to revert
    to legacy test-set selection.
    """
    from polystep.cost_nn import NNCostEvaluator

    config = SHOWCASE_CONFIGS[showcase_name]
    polystep_cfg = PSTORCH_CONFIGS[showcase_name]
    if theory_mode:
        polystep_cfg = apply_theory_mode(polystep_cfg)
    epochs = _epochs_for(showcase_name, polystep_cfg, dry_run)

    set_seed(seed)
    model = config["model_fn"]().to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _load_split(showcase_name, seed, audit_no_leakage)
    selection_loader = val_loader if val_loader is not None else test_loader

    total_steps = epochs * len(train_loader)
    layout, subspace, optimizer = _build_polystep(model, seed, total_steps, polystep_cfg)
    eval_budget = polystep_eval_budget(optimizer, total_steps)

    import copy

    evaluator = NNCostEvaluator(
        model,
        loss_fn=loss_fn,
        compile_vmap=polystep_cfg.get("compile_evaluator", False),
        compile_forward=polystep_cfg.get("compile_forward", False),
    )
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
                    step_logs.append(
                        {
                            "step": step_count,
                            "epoch": epoch + 1,
                            # Cumulative candidate evaluations: the x-axis of the
                            # accuracy-vs-evaluations figure, shared with the baselines.
                            "evals": fwd_pass_count,
                            "test_accuracy": step_test_acc,
                            "loss": loss,
                            "wall_time": time.time() - start_time,
                        }
                    )

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

            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "accuracy": test_acc,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                    "wall_time": time.time() - start_time,
                }
            )
            print(
                f"    Epoch {epoch + 1}/{epochs} | train={train_acc * 100:.1f}% | test={test_acc * 100:.1f}% | loss={avg_loss:.4f}"
            )

    wall_time = time.time() - start_time
    # The epoch loop already scored (and, if better, saved) the final
    # weights on the selection split. Re-scoring them on *test* here and
    # letting that overwrite best_accuracy/best_state_dict was a test peek
    # that survived audit_no_leakage=True.
    last_epoch_acc = test_acc
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
            "test_accuracy_at_selected": final_acc,
            "last_epoch_accuracy": last_epoch_acc,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": fwd_pass_count,
            "total_steps": step_count,
        },
        hyperparameters={
            **polystep_cfg,
            "epochs": epochs,
            # The four keys that let a reader check the table was matched.
            **subspace_tag(subspace, polystep_cfg["rank"]),
            "eval_budget": eval_budget,
            "evals_used": fwd_pass_count,
            "fair": fair,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
        leaked=not audit_no_leakage,
    )
    print(f"    Saved: {filepath}")


def run_adam(showcase_name, seed, device, results_dir, dry_run=False, audit_no_leakage: bool = True):
    """Train smooth model variant with Adam (accuracy ceiling baseline)."""
    config = SHOWCASE_CONFIGS[showcase_name]
    epochs = 1 if dry_run else EPOCHS_ADAM

    set_seed(seed)
    model = config["smooth_model_fn"]()
    model = model.to(device)

    train_loader, val_loader, test_loader = _load_split(showcase_name, seed, audit_no_leakage)

    result = train_sgd(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
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
        leaked=not audit_no_leakage,
    )
    print(f"    Saved: {filepath}")


def run_gradient_free(
    showcase_name,
    seed,
    device,
    results_dir,
    method=None,
    dry_run=False,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
    budget: int = None,
):
    """Run one ``polystep.baselines`` method on a non-differentiable showcase.

    Replaces the inline pycma copy and the two ``experiments/baselines`` wrappers,
    which each counted evaluations differently and searched full parameter space
    while PolyStep searched a subspace.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    config = SHOWCASE_CONFIGS[showcase_name]
    cfg = PSTORCH_CONFIGS[showcase_name]
    if theory_mode:
        cfg = apply_theory_mode(cfg)
    epochs = _epochs_for(showcase_name, cfg, dry_run)
    if dry_run and budget is None:
        budget = 2000

    set_seed(seed)
    model = config["model_fn"]().to(device)
    layout = ParamLayout.from_module(model)
    train_loader, val_loader, test_loader = _load_split(showcase_name, seed, audit_no_leakage)

    subspace = None
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        if budget is None:
            budget = fair_eval_budget(showcase_name, seed, device, train_loader, epochs, cfg)
    if budget is None:
        budget = LEGACY_BUDGETS[method]

    loss_batch = minibatch_loss(NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss()), train_loader, device)
    selection_loader = val_loader if val_loader is not None else test_loader

    out = run_baseline(
        method,
        model=model,
        layout=layout,
        loss_batch=loss_batch,
        budget=budget,
        val_fn=lambda m: evaluate_accuracy(m, selection_loader, device=device),
        test_fn=lambda m: evaluate_accuracy(m, test_loader, device=device),
        mode="max",
        seed=seed,
        subspace=subspace,
        subspace_rank=cfg["rank"] if fair else None,
        probe_scale=probe_scale_of(cfg),
    )
    out["hyperparameters"]["fair"] = fair
    filepath = save_result(
        benchmark=showcase_name,
        method=method,
        seed=seed,
        results_dir=results_dir,
        leaked=not audit_no_leakage,
        **out,
    )
    print(f"    Saved: {filepath}")


# polystep and adam have their own loops; every gradient-free method goes through
# the one shared runner.
METHOD_RUNNERS = {"polystep": run_polystep, "adam": run_adam}
ALL_METHODS = ("polystep", "adam", *FAIR_METHODS)
#: The old name for cma_es, kept so existing result files and scripts still resolve.
ALIASES = {"cmaes": "cma_es"}


def main():
    parser = argparse.ArgumentParser(
        description="Run non-differentiable showcase elevation experiments: showcases x methods x seeds"
    )
    parser.add_argument(
        "--showcases",
        nargs="+",
        default=["snn", "int8", "argmax", "staircase"],
        help="Showcases to run (default: all 4)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        help=f"Methods to run (default: all of {list(ALL_METHODS)})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to run (default: 42 123 456 789 1337)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/results/softmax/main",
        help="Results directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch / 10 generations / 100 SPSA iters for testing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override skip-if-exists and rerun all experiments",
    )
    parser.add_argument(
        "--allow-test-leakage",
        action="store_true",
        help=(
            "Legacy mode: select best_state_dict on the test set instead "
            "of a held-out validation slice. Default is honest protocol "
            "(val-selected). Use only for bit-for-bit reproduction "
            "of earlier results."
        ),
    )
    parser.add_argument(
        "--fair",
        action="store_true",
        help=(
            "Matched-budget, matched-representation table: every gradient-free "
            "method gets the same subspace (FactoredSubspace for eggroll), the same "
            "candidate budget derived from PolyStep, and the same probe radius."
        ),
    )
    parser.add_argument(
        "--theory-mode",
        action="store_true",
        help=(
            "Run the configuration Theorem 4.2 analyses: probe_radius_jitter=0.05 "
            "with the smooth density, independent rotations, flat epsilon, step "
            "radius r0*(t+1)^-(1/2+0.1), orthoplex, HybridSubspace, no momentum / "
            "amortization / Anderson."
        ),
    )
    parser.add_argument("--budget", type=int, default=None, help="Override the candidate budget (smoke runs)")
    parser.add_argument(
        "--step-radius",
        type=float,
        default=None,
        help="Override step_radius for polystep (for hyperparameter sweeps)",
    )
    parser.add_argument(
        "--epochs-polystep",
        type=int,
        default=None,
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
            method = ALIASES.get(method, method)
            gradient_free = method in FAIR_METHODS
            runner = METHOD_RUNNERS.get(method)
            if runner is None and not gradient_free:
                print(f"  Unknown method: {method}")
                continue

            for seed in args.seeds:
                output_file = os.path.join(args.results_dir, f"{showcase_name}_{method}_{seed}.json")
                if os.path.exists(output_file) and not args.force:
                    print(f"  Skipping {method} seed={seed} (result exists)")
                    continue

                print(f"  Running {method} seed={seed}...")
                try:
                    kwargs = dict(dry_run=args.dry_run, audit_no_leakage=not args.allow_test_leakage)
                    if gradient_free:
                        run_gradient_free(
                            showcase_name,
                            seed,
                            args.device,
                            args.results_dir,
                            method=method,
                            fair=args.fair,
                            theory_mode=args.theory_mode,
                            budget=args.budget,
                            **kwargs,
                        )
                    else:
                        if method == "polystep":
                            kwargs.update(fair=args.fair, theory_mode=args.theory_mode)
                        runner(showcase_name, seed, args.device, args.results_dir, **kwargs)
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
