#!/usr/bin/env python
"""Run all methods and seeds for the MNIST benchmark.

Methods: polystep, adam, and the six gradient-free baselines from
``polystep.baselines``, all on the MNISTNet MLP (784->128->10). ``--fair`` runs
the matched-budget, matched-representation table; ``--theory-mode`` runs the
unaccelerated reference configuration. Results are saved as JSON in
experiments/results/softmax/main/.

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
    METHOD_ALIASES,
    SEEDS,
    MNISTNet,
    evaluate_accuracy,
    load_mnist,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.fairness import (
    build_polystep,
    FAIR_METHODS,
    apply_point,
    apply_theory_mode,
    budget_for_method,
    TestSplitTripwire,
    apply_polystep_multipliers,
    load_selection,
    make_subspace,
    minibatch_loss,
    polystep_eval_budget,
    probe_scale_of,
    run_baseline,
    subspace_tag,
)
from experiments.baselines.sgd_baseline import train_sgd


BENCHMARK = "mnist"

#: PolyStep's own sweep writes here, not the shared gallery file, so a tuning run
#: cannot untune a headline run reading that file.
POLYSTEP_SELECTION_PATH = os.path.join("experiments", "results", "tuning", "polystep_selected.json")
BATCH_SIZE = 512
EPOCHS = 30

# polystep hyperparameters (HybridSubspace + cosine schedules), no momentum.
# No probe_radius_jitter here: setting it forces amortize_steps=1 (see
# PolyStepOptimizer.__init__) and would change the tuned headline run;
# --theory-mode runs the analysed configuration instead.
POLYSTEP_CONFIG = {
    "rank": 8,
    "step_radius_init": 5.0,
    "step_radius_target": 1.0,
    "probe_radius_init": 10.0,
    "probe_radius_target": 2.0,
    "epsilon_init": 10.0,
    "epsilon_target": 0.1,
    "rotation_interval": 0,
    "absorb_interval": 0,
    "num_probe": 1,  # K>1 adds no benefit for softmax; uses fused path
    "chunk_size": 1024,
    "amortize_steps": 3,
    "amortize_ema": 0.7,
}

ADAM_CONFIG = {
    "lr": 0.001,
    "epochs": EPOCHS,
}

# Free-running (non-fair) candidate budgets: ES 2000 gen x 50 pop, SPSA 10000
# iters x 2, CMA-ES 2000 gen x 16 pop. `--fair` overrides all of these with one
# budget derived from PolyStep.
LEGACY_BUDGETS = {
    "openai_es": 100_000,
    "eggroll": 100_000,
    "cma_es": 32_000,
    "spsa": 20_000,
    "mezo": 20_000,
    "random_search": 20_000,
}


def fair_eval_budget(seed, device, train_loader, epochs, cfg, solver):
    """The shared candidate budget: what PolyStep spends over ``epochs`` epochs."""
    set_seed(seed)
    probe_model = MNISTNet().to(device)
    _, _, opt = build_polystep(probe_model, seed, epochs * len(train_loader), cfg, solver)
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
    tuning: bool = False,
):
    """Train MNIST with polystep PolyStepOptimizer + HybridSubspace.

    Best-checkpoint selection uses a held-out validation split;
    ``audit_no_leakage=False`` selects ``best_state_dict`` on the test set.
    """
    from polystep.cost_nn import NNCostEvaluator

    # PolyStep reads its own sweep the same way the baselines read theirs; theory
    # mode reads it too and overrides the analysed knobs on top. ``tuning=True``
    # is the sweep itself and must not read a selection it is producing.
    cfg = dict(POLYSTEP_CONFIG)
    polystep_tuned_name, polystep_tuned_prov = None, None
    if not tuning:
        selected, polystep_tuned_prov = load_selection("gallery", BENCHMARK, "polystep", path=POLYSTEP_SELECTION_PATH)
        if selected:
            cfg = apply_polystep_multipliers(cfg, selected["point"])
            polystep_tuned_name = selected["name"]
            print(f"    tuned: {selected['name']} (val={selected['val']:.4f})")
    if theory_mode:
        cfg = apply_theory_mode(cfg)
    epochs = EPOCHS if epochs is None else epochs

    if tuning:
        if val_loader is None:
            raise ValueError("tuning=True needs a validation split to select on")
        test_loader = TestSplitTripwire()

    set_seed(seed)
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader
    selection_label = "val" if (audit_no_leakage and val_loader is not None) else "test"

    total_steps = epochs * len(train_loader)
    layout, subspace, optimizer = build_polystep(model, seed, total_steps, cfg, solver)
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

                # Per-20-step tracking on the selection split: the x-axis of the
                # accuracy-vs-evaluations figure; nothing selects on it, so the test
                # set stays untouched.
                if step_count % 20 == 0:
                    step_logs.append(
                        {
                            "step": step_count,
                            "epoch": epoch + 1,
                            # Cumulative candidate evaluations, shared with the baselines.
                            "evals": fwd_pass_count,
                            "val_accuracy": evaluate_accuracy(model, selection_loader, device=device),
                            "loss": loss,
                            "wall_time": time.time() - start_time,
                        }
                    )

            train_acc = epoch_correct / max(epoch_total, 1)
            # Score the selection split only (validation when audit_no_leakage=True,
            # test otherwise -- and then the run is stamped leaked=True). The test set
            # is read once, after the checkpoint has been chosen.
            selection_acc = evaluate_accuracy(model, selection_loader, device=device)
            if selection_acc > best_accuracy:
                best_accuracy = selection_acc
                best_state_dict = copy.deepcopy(model.state_dict())
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(len(train_loader), 1)

            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "accuracy": selection_acc,
                    "train_accuracy": train_acc,
                    f"{selection_label}_accuracy": selection_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                    "wall_time": time.time() - start_time,
                }
            )
            print(
                f"    Epoch {epoch + 1}/{epochs} | train={train_acc * 100:.1f}% | "
                f"{selection_label}={selection_acc * 100:.1f}% | "
                f"{selection_label}-best={best_accuracy * 100:.1f}% | "
                f"loss={avg_loss:.4f}"
            )

    wall_time = time.time() - start_time
    # The epoch loop already scored (and, if better, saved) the final
    # weights on the selection split; re-scoring here only ever added a
    # second chance for the test set to pick the checkpoint.
    last_epoch_acc = selection_acc
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_acc = float("nan") if tuning else evaluate_accuracy(model, test_loader, device=device)

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
            "tuned": polystep_tuned_name,
            "tuning_provenance": polystep_tuned_prov,
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

    ``fair=True`` gives the method the same subspace, the same candidate budget as
    PolyStep, and the same probe radius; otherwise it runs full-space on its own budget.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    cfg = apply_theory_mode(POLYSTEP_CONFIG) if theory_mode else dict(POLYSTEP_CONFIG)
    epochs = EPOCHS if epochs is None else epochs

    set_seed(seed)
    model = MNISTNet().to(device)
    layout = ParamLayout.from_module(model)
    subspace = None
    match_axis = "evals"
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        # What every other method searches. EGGROLL alone gets FactoredSubspace, whose
        # dimension is smaller by construction; recording the gap is what lets a reader
        # see that the table is matched on rank and not on dimension.
        shared_dim = make_subspace(layout, rank=cfg["rank"], seed=seed, method="polystep").subspace_dim
        if budget is None:
            budget = fair_eval_budget(seed, device, train_loader, epochs, cfg, "softmax")
            budget, match_axis = budget_for_method(method, budget, epochs * len(train_loader))
    if budget is None:
        budget = LEGACY_BUDGETS[method]

    loss_batch = minibatch_loss(NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss()), train_loader, device)
    selection_loader = val_loader if (audit_no_leakage and val_loader is not None) else test_loader

    # The picks tune_gallery.py --showcases mnist files under ("gallery", "mnist",
    # method). Without them every baseline runs at fair_hyperparams defaults, which is a
    # strawman: EGGROLL scores 18.6% untuned against 78% tuned.
    selected, provenance = load_selection("gallery", BENCHMARK, method)
    probe_scale = probe_scale_of(cfg, dim=subspace.subspace_dim if subspace else layout.total_params)
    hp = apply_point({}, selected["point"], probe_scale) if selected else None
    if selected:
        print(f"    tuned: {selected['name']} (val={selected['val']:.4f})")

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
        shared_subspace_dim=shared_dim if fair else None,
        hp=hp,
        probe_scale=probe_scale,
    )
    out["hyperparameters"]["fair"] = fair
    out["hyperparameters"]["match_axis"] = match_axis
    out["hyperparameters"]["polystep_steps"] = epochs * len(train_loader)
    out["hyperparameters"]["tuned"] = selected["name"] if selected else None
    out["hyperparameters"]["tuning_provenance"] = provenance
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
    method = METHOD_ALIASES.get(method, method)
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
            "Run the unaccelerated reference configuration: probe_radius_jitter=0.05 "
            "with the smooth density, independent rotations, flat epsilon, step "
            "radius r0*(t+1)^-(1/2+0.1), orthoplex, HybridSubspace, no momentum / "
            "amortization / Anderson."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke runs)")
    parser.add_argument("--budget", type=int, default=None, help="Override the candidate budget (smoke runs)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells whose result file already exists (default: skip them).",
    )
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

    failures: list = []
    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(
                args.results_dir, f"{BENCHMARK}_{METHOD_ALIASES.get(method, method)}_{seed}.json"
            )
            if os.path.exists(output_file) and not args.force:
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
                failures.append((method, seed, repr(e)))
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for method, seed, err in failures:
            print(f"  {method} seed={seed}: {err}")
        return 1
    print(f"\nDone. Results in {args.results_dir}/{BENCHMARK}_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
