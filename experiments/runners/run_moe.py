#!/usr/bin/env python
"""Run all methods and seeds for Hard MoE (Mixture-of-Experts) benchmark.

Methods:
  - polystep: PolyStepOptimizer with HybridSubspace on hard-gated MoE
  - the six gradient-free baselines from ``polystep.baselines``: cma_es, openai_es,
    spsa, mezo, random_search, eggroll

``--fair`` gives every gradient-free method the same subspace, the same candidate
budget derived from PolyStep, and the same probe radius; ``--theory-mode`` runs the
configuration Theorem 4.2 analyses. See ``experiments/runners/fairness.py``.

Model: HardMoENet (~235K params) - top-1 argmax gating (non-differentiable)
Data: Combined MNIST + Fashion-MNIST (20 classes)

polystep config from r4_sr12t4 config (90.92% at 20ep, seed 42):
  Flat eps=0.5 (eps scheduling -> collapse), scheduled sr 12->4,
  flat pr=1.0, rank=4, biased_rotation.

Results saved as: experiments/results/softmax/main/moe_{method}_{seed}.json
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
    make_train_val_split,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.nondiff_models import HardMoENet
from experiments.runners.nondiff_data import generate_multidomain_data
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


BENCHMARK = "moe"
BATCH_SIZE = 512
EPOCHS = 30

# Free-running (non-fair) candidate budgets, preserving what the previous ad-hoc
# implementations spent: ES 2000 gen x 50 pop, CMA-ES 2000 gen x 16 pop, SPSA 10000
# iters x 2. `--fair` replaces all of these with one budget derived from PolyStep.
LEGACY_BUDGETS = {
    "openai_es": 100_000,
    "eggroll": 100_000,
    "cma_es": 32_000,
    "spsa": 20_000,
    "mezo": 20_000,
    "random_search": 20_000,
}

# polystep config - r4_sr12t4 config (90.92% at 20ep, seed 42)
# HYBRID: flat eps + flat pr, but SCHEDULED sr only
# eps <= 0.5 mandatory - eps scheduling causes MoE collapse (2.66% at eps=1.5)
# sr scheduling (12->4) with rank=4 beats flat rank=8 while being 2x faster
PSTORCH_CONFIG = {
    "epsilon": 0.5,  # FLAT - eps scheduling -> collapse
    "step_radius_init": 12.0,  # sr scheduling: 12->4
    "step_radius_target": 4.0,
    "probe_radius": 1.0,  # FLAT
    "num_probe": 1,
    "rank": 4,
    "chunk_size": 1024,
    "amortize_steps": 1,
    "rotation_interval": 0,
    "absorb_interval": 20,
    "biased_rotation": True,
}


def _load_split(seed, audit_no_leakage):
    """Load the MoE data and carve the honest-protocol val split.

    Returns ``(train_loader, val_loader, test_loader)`` with ``val_loader``
    None only in legacy ``--allow-test-leakage`` mode. Every method goes
    through here, so no method can quietly select on the test set.
    """
    data = generate_multidomain_data(data_dir="data/", batch_size=BATCH_SIZE)
    train_loader, test_loader = data["train_loader"], data["test_loader"]
    val_loader = None
    if audit_no_leakage:
        train_loader, val_loader = make_train_val_split(train_loader, val_frac=0.1, seed=seed)
    return train_loader, val_loader, test_loader


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
        polytope_type=cfg.get("polytope_type", "simplex"),
        amortize_steps=cfg.get("amortize_steps", 1),
        amortize_ema=cfg.get("amortize_ema", 0.0),
        use_momentum=cfg.get("use_momentum", False),
        biased_rotation=cfg.get("biased_rotation", False),
        solver="softmax",
    )
    return layout, subspace, optimizer


def fair_eval_budget(seed, device, train_loader, epochs, cfg):
    """The shared candidate budget: what PolyStep spends over ``epochs`` epochs."""
    set_seed(seed)
    _, _, opt = _build_polystep(HardMoENet(num_experts=4).to(device), seed, epochs * len(train_loader), cfg)
    return polystep_eval_budget(opt, epochs * len(train_loader))


def run_polystep(
    seed,
    device,
    results_dir,
    epochs=EPOCHS,
    dry_run=False,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
):
    """Train Hard MoE with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses a held-out validation
    split (honest protocol). Set ``audit_no_leakage=False`` to revert
    to legacy test-set selection.
    """
    from polystep.cost_nn import NNCostEvaluator

    cfg = apply_theory_mode(PSTORCH_CONFIG) if theory_mode else dict(PSTORCH_CONFIG)
    if dry_run:
        epochs = 1

    set_seed(seed)
    model = HardMoENet(num_experts=4).to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _load_split(seed, audit_no_leakage)
    selection_loader = val_loader if val_loader is not None else test_loader

    total_steps = epochs * len(train_loader)
    layout, subspace, optimizer = _build_polystep(model, seed, total_steps, cfg)
    eval_budget = polystep_eval_budget(optimizer, total_steps)

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
            "test_accuracy_at_selected": final_acc,
            "last_epoch_accuracy": last_epoch_acc,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": fwd_pass_count,
            "total_steps": step_count,
        },
        hyperparameters={
            **cfg,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "solver": "softmax",
            # The four keys that let a reader check the table was matched.
            **subspace_tag(subspace, cfg["rank"]),
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


def run_gradient_free(
    method,
    seed,
    device,
    results_dir,
    dry_run=False,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
    epochs: int = EPOCHS,
    budget: int = None,
):
    """Run one ``polystep.baselines`` method on Hard MoE.

    Replaces the inline pycma copy and the two ``experiments/baselines`` wrappers,
    which each counted evaluations differently and searched full parameter space
    while PolyStep searched a subspace.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    cfg = apply_theory_mode(PSTORCH_CONFIG) if theory_mode else dict(PSTORCH_CONFIG)
    if dry_run:
        epochs, budget = 1, budget or 2000

    set_seed(seed)
    model = HardMoENet(num_experts=4).to(device)
    layout = ParamLayout.from_module(model)
    train_loader, val_loader, test_loader = _load_split(seed, audit_no_leakage)

    subspace = None
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        if budget is None:
            budget = fair_eval_budget(seed, device, train_loader, epochs, cfg)
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
        benchmark=BENCHMARK,
        method=method,
        seed=seed,
        results_dir=results_dir,
        leaked=not audit_no_leakage,
        **out,
    )
    print(f"    Saved: {filepath}")


# polystep has its own loop; every gradient-free method goes through the one
# shared runner.
METHOD_RUNNERS = {"polystep": run_polystep}
ALL_METHODS = ("polystep", *FAIR_METHODS)
#: The old name for cma_es, kept so existing result files and scripts still resolve.
ALIASES = {"cmaes": "cma_es"}


def main():
    parser = argparse.ArgumentParser(description="Run Hard MoE (Mixture-of-Experts) benchmark: methods x seeds")
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
        help=f"Seeds (default: {SEEDS})",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--results-dir",
        default=os.path.join("experiments", "results", "softmax", "main"),
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of polystep epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 1 epoch (polystep) / 10 generations (ES) for testing",
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
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("Hard MoE (Mixture-of-Experts) Benchmark")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Fair: {args.fair}  Theory mode: {args.theory_mode}")
    if args.dry_run:
        print("  Mode: DRY RUN (minimal epochs/generations)")
    print()

    os.makedirs(args.results_dir, exist_ok=True)

    for method in args.methods:
        method = ALIASES.get(method, method)
        for seed in args.seeds:
            output_file = os.path.join(args.results_dir, f"{BENCHMARK}_{method}_{seed}.json")
            if os.path.exists(output_file):
                print(f"  Skipping {method} seed={seed} (result exists)")
                continue
            print(f"  Running {method} seed={seed}...")
            try:
                if method == "polystep":
                    run_polystep(
                        seed,
                        args.device,
                        args.results_dir,
                        epochs=args.epochs,
                        dry_run=args.dry_run,
                        audit_no_leakage=not args.allow_test_leakage,
                        fair=args.fair,
                        theory_mode=args.theory_mode,
                    )
                elif method in FAIR_METHODS:
                    run_gradient_free(
                        method,
                        seed,
                        args.device,
                        args.results_dir,
                        dry_run=args.dry_run,
                        audit_no_leakage=not args.allow_test_leakage,
                        fair=args.fair,
                        theory_mode=args.theory_mode,
                        epochs=args.epochs,
                        budget=args.budget,
                    )
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
