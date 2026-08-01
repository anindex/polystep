#!/usr/bin/env python
"""Run all methods and seeds for MNIST benchmark.

Methods: polystep, adam, and the six gradient-free baselines from
``polystep.baselines`` (cma_es, openai_es, spsa, mezo, random_search, eggroll).
Model: MNISTNet MLP (784->128->10, ~102K params) for all methods.
Data: Standard MNIST (28x28 grayscale, 10 classes)

Hyperparameters are hardcoded constants for reproducibility.
Results are saved as JSON files in experiments/results/softmax/main/.

``--fair`` runs the matched-budget, matched-representation table: every
gradient-free method gets the same subspace, the same candidate budget derived
from what PolyStep spends, and the same probe radius. See
``experiments/runners/fairness.py``.

Usage:
    python experiments/runners/run_mnist.py
    python experiments/runners/run_mnist.py --methods polystep adam --seeds 42 123
    python experiments/runners/run_mnist.py --fair --device cpu
    python experiments/runners/run_mnist.py --theory-mode --methods polystep
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
from experiments.baselines.sgd_baseline import train_sgd


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
#   probe_radius_jitter=0.05: required by the convergence analysis. It forces
#   amortize_steps=1 (see PolyStepOptimizer.__init__), so the amortization the
#   sweep found is off here and the config records that rather than hiding it.
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

# Adam hyperparameters
ADAM_CONFIG = {
    "lr": 0.001,
    "epochs": EPOCHS,
}

# Free-running (non-fair) candidate budgets, preserving what the previous ad-hoc
# implementations spent: ES 2000 gen x 50 pop, SPSA 10000 iters x 2, CMA-ES 2000 gen
# x 16 pop. The new methods get the budget of the method they most resemble.
# `--fair` overrides all of these with one budget derived from PolyStep.
LEGACY_BUDGETS = {
    "openai_es": 100_000,
    "eggroll": 100_000,
    "cma_es": 32_000,
    "spsa": 20_000,
    "mezo": 20_000,
    "random_search": 20_000,
}


def _build_polystep(model, seed, total_steps, cfg, solver):
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
        chunk_size=cfg.get("chunk_size", 1024),
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
        solver=solver,
    )
    return layout, subspace, optimizer


def fair_eval_budget(seed, device, train_loader, epochs, cfg, solver):
    """The shared candidate budget: what PolyStep spends over ``epochs`` epochs."""
    set_seed(seed)
    probe_model = MNISTNet().to(device)
    _, _, opt = _build_polystep(probe_model, seed, epochs * len(train_loader), cfg, solver)
    return polystep_eval_budget(opt, epochs * len(train_loader))


def run_polystep(
    seed,
    device,
    train_loader,
    test_loader,
    results_dir,
    solver=None,
    audit_no_leakage: bool = True,
    val_loader=None,
    fair: bool = False,
    theory_mode: bool = False,
    epochs: int = None,
):
    """Train MNIST with polystep PolyStepOptimizer + HybridSubspace.

    By default, best-checkpoint selection uses a held-out validation
    split (honest protocol). Set ``audit_no_leakage=False`` to revert
    to the legacy behavior where ``best_state_dict`` was selected on
    the test set.
    """
    from polystep.cost_nn import NNCostEvaluator

    cfg = apply_theory_mode(PSTORCH_CONFIG) if theory_mode else dict(PSTORCH_CONFIG)
    epochs = EPOCHS if epochs is None else epochs

    set_seed(seed)
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader
    selection_label = "val" if (audit_no_leakage and val_loader is not None) else "test"

    total_steps = epochs * len(train_loader)
    layout, subspace, optimizer = _build_polystep(model, seed, total_steps, cfg, solver)
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

            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "accuracy": test_acc,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    f"{selection_label}_accuracy": selection_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                    "wall_time": time.time() - start_time,
                }
            )
            print(
                f"    Epoch {epoch + 1}/{epochs} | train={train_acc * 100:.1f}% | "
                f"test={test_acc * 100:.1f}% | {selection_label}-best={best_accuracy * 100:.1f}% | "
                f"loss={avg_loss:.4f}"
            )

    wall_time = time.time() - start_time
    # The epoch loop already scored (and, if better, saved) the final
    # weights on the selection split; re-scoring here only ever added a
    # second chance for the test set to pick the checkpoint.
    last_epoch_acc = test_acc
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
    print(f"    Saved: {result}")


def run_gradient_free(
    method,
    seed,
    device,
    train_loader,
    test_loader,
    results_dir,
    val_loader=None,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
    epochs: int = None,
    budget: int = None,
):
    """Run one ``polystep.baselines`` method on MNIST.

    Replaces the three ad-hoc implementations (EvoTorch CMA-ES,
    ``experiments/baselines/openai_es.py``, ``experiments/baselines/spsa.py``), which
    each counted evaluations differently and searched full parameter space while
    PolyStep searched a subspace.

    ``fair=True`` gives the method the same subspace, the same candidate budget as
    PolyStep, and the same probe radius; otherwise it runs full-space on the budget
    the old implementation spent.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    cfg = apply_theory_mode(PSTORCH_CONFIG) if theory_mode else dict(PSTORCH_CONFIG)
    epochs = EPOCHS if epochs is None else epochs

    set_seed(seed)
    model = MNISTNet().to(device)
    layout = ParamLayout.from_module(model)
    subspace = None
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        if budget is None:
            budget = fair_eval_budget(seed, device, train_loader, epochs, cfg, "softmax")
    if budget is None:
        budget = LEGACY_BUDGETS[method]

    loss_batch = minibatch_loss(NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss()), train_loader, device)
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader

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


def run_adam(seed, device, train_loader, test_loader, results_dir, val_loader=None, audit_no_leakage: bool = True):
    """Train MNIST with Adam (gradient-based ceiling)."""
    set_seed(seed)
    model = MNISTNet().to(device)

    result = train_sgd(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
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
        leaked=not audit_no_leakage,
    )
    print(f"    Saved: {filepath}")


# polystep and adam have their own loops; every gradient-free method goes through
# the one shared runner.
METHOD_RUNNERS = {"polystep": run_polystep, "adam": run_adam}
ALL_METHODS = ("polystep", "adam", *FAIR_METHODS)
#: The old name for cma_es, kept so existing result files and scripts still resolve.
ALIASES = {"cmaes": "cma_es"}


def run_method(
    method,
    seed,
    device,
    results_dir,
    data_dir,
    solver=None,
    audit_no_leakage: bool = True,
    fair: bool = False,
    theory_mode: bool = False,
    epochs: int = None,
    budget: int = None,
):
    """Run a single method+seed combination."""
    from experiments.runners.common import make_train_val_split

    train_loader, test_loader = load_mnist(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
    )
    val_loader = None
    if audit_no_leakage:
        train_loader, val_loader = make_train_val_split(
            train_loader,
            val_frac=0.1,
            seed=seed,
        )
    method = ALIASES.get(method, method)
    kwargs = dict(audit_no_leakage=audit_no_leakage, val_loader=val_loader)
    if method in FAIR_METHODS:
        run_gradient_free(
            method,
            seed,
            device,
            train_loader,
            test_loader,
            results_dir,
            fair=fair,
            theory_mode=theory_mode,
            epochs=epochs,
            budget=budget,
            **kwargs,
        )
        return
    runner = METHOD_RUNNERS.get(method)
    if runner is None:
        print(f"    Unknown method: {method}")
        return
    if method == "polystep":
        kwargs.update(solver=solver, fair=fair, theory_mode=theory_mode, epochs=epochs)
    runner(seed, device, train_loader, test_loader, results_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run MNIST benchmark: all methods x all seeds")
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
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--results-dir", default="experiments/results/softmax/main", help="Results directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--solver",
        choices=["softmax", "sinkhorn"],
        default="softmax",
        help="Solver backend: softmax (default, used with subspace) or sinkhorn (full-space).",
    )
    parser.add_argument(
        "--allow-test-leakage",
        action="store_true",
        help=(
            "Legacy mode: select best_state_dict on the test set instead "
            "of a held-out validation slice. Default is honest protocol "
            "(val-selected). Use this flag only for bit-for-bit reproduction "
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
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke runs)")
    parser.add_argument("--budget", type=int, default=None, help="Override the candidate budget (smoke runs)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("MNIST Benchmark")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print(f"  Fair: {args.fair}  Theory mode: {args.theory_mode}")
    print()

    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(args.results_dir, f"{BENCHMARK}_{ALIASES.get(method, method)}_{seed}.json")
            if os.path.exists(output_file):
                print(f"Skipping {method} seed={seed} (result exists)")
                continue
            print(f"Running {method} seed={seed}...")
            try:
                run_method(
                    method,
                    seed,
                    args.device,
                    args.results_dir,
                    args.data_dir,
                    solver=args.solver,
                    audit_no_leakage=not args.allow_test_leakage,
                    fair=args.fair,
                    theory_mode=args.theory_mode,
                    epochs=args.epochs,
                    budget=args.budget,
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
