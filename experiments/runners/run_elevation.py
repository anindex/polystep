#!/usr/bin/env python
"""Run all methods and seeds for the non-differentiable showcases (snn, int8, argmax, staircase).

Methods: polystep (HybridSubspace on the non-diff models), adam on a smooth
variant, and the six gradient-free baselines from ``polystep.baselines``.
``--fair`` gives every gradient-free method the same subspace, budget, and probe
radius; ``--theory-mode`` runs the unaccelerated reference configuration.

Results saved as: experiments/results/softmax/main/{showcase}_{method}_{seed}.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn

from experiments.runners.common import (
    METHOD_ALIASES,
    SEEDS,
    reseed_loaders,
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
    build_polystep,
    FAIR_METHODS,
    TestSplitTripwire,
    apply_point,
    apply_polystep_multipliers,
    apply_theory_mode,
    load_selection,
    make_subspace,
    minibatch_loss,
    matched_budget,
    polystep_eval_budget,
    probe_scale_of,
    run_baseline,
    subspace_tag,
)

#: tune_polystep.py writes here, not the shared baseline selection file.
POLYSTEP_SELECTION_PATH = os.path.join("experiments", "results", "tuning", "polystep_selected.json")


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


EPOCHS_POLYSTEP = 30  # SNN needs more epochs; softmax solver is fast enough
EPOCHS_POLYSTEP_NONSNN = 30
EPOCHS_ADAM = 30
ADAM_LR = 0.001

# Free-running (non-fair) candidate budgets: ES 10000 gen x 50 pop, CMA-ES 10000
# gen x 16 pop, SPSA 50000 iters x 2. `--fair` replaces all of these with one
# budget derived from PolyStep.
LEGACY_BUDGETS = {
    "openai_es": 500_000,
    "eggroll": 500_000,
    "cma_es": 160_000,
    "spsa": 100_000,
    "mezo": 100_000,
    "random_search": 100_000,
}

POLYSTEP_CONFIGS = {
    # SNN: flat eps/sr/pr; cosine scheduling collapses SNN accuracy, and
    # biased_rotation + absorb_interval=20 stabilize the spiking landscape.
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
    # INT8: cosine schedules work well on quantization plateaus; rank=8 is the
    # dominant lever.
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
    # Argmax: same cosine strategy as INT8, but no momentum.
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
    # Staircase: gentle cosine targets (sr 64->32, eps 5->1) avoid late-epoch
    # degradation.
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
    # Rewind the shuffle generators, or the Nth run in a process sees the Nth
    # minibatch stream and a reported seed only reproduces in a fresh interpreter.
    reseed_loaders(seed, train_loader, val_loader, test_loader)
    return train_loader, val_loader, test_loader


def _epochs_for(showcase_name, cfg, dry_run):
    if dry_run:
        return 1
    if "epochs" in cfg:
        return cfg["epochs"]
    return EPOCHS_POLYSTEP if showcase_name == "snn" else EPOCHS_POLYSTEP_NONSNN


def fair_eval_budget(showcase_name, seed, device, train_loader, epochs, cfg):
    """The shared candidate budget: what PolyStep spends over ``epochs`` epochs."""
    set_seed(seed)
    model = SHOWCASE_CONFIGS[showcase_name]["model_fn"]().to(device)
    _, _, opt = build_polystep(model, seed, epochs * len(train_loader), cfg)
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
    tuning: bool = False,
):
    """Train non-diff model with polystep PolyStepOptimizer + HybridSubspace.

    Best-checkpoint selection uses a held-out validation split;
    ``audit_no_leakage=False`` selects on the test set. ``tuning=True`` skips the
    selection file and swaps the test split for ``TestSplitTripwire`` so a sweep
    that reads test raises.
    """
    from polystep.cost_nn import NNCostEvaluator

    config = SHOWCASE_CONFIGS[showcase_name]
    polystep_cfg = POLYSTEP_CONFIGS[showcase_name]
    # PolyStep reads its own sweep selection the same way the baselines read
    # theirs; theory mode reads it too and then overrides the analysed knobs on
    # top, so the gap it feeds is measured with everything else at the tuned values.
    polystep_tuned_name, polystep_tuned_prov = None, None
    if not tuning:
        selected, polystep_tuned_prov = load_selection(
            "gallery", showcase_name, "polystep", path=POLYSTEP_SELECTION_PATH
        )
        if selected:
            polystep_cfg = apply_polystep_multipliers(polystep_cfg, selected["point"])
            polystep_tuned_name = selected["name"]
            print(f"    tuned: {selected['name']} (val={selected['val']:.4f})")
    if theory_mode:
        polystep_cfg = apply_theory_mode(polystep_cfg)
    # Sweep a PolyStep knob without a flag, e.g. POLYSTEP_CFG_OVERRIDE='{"rank":4,"epochs":58}',
    # e.g. to retune rank against step count at a fixed evaluation budget.
    _cfg_override = os.environ.get("POLYSTEP_CFG_OVERRIDE")
    if _cfg_override:
        polystep_cfg = {**polystep_cfg, **json.loads(_cfg_override)}
    epochs = _epochs_for(showcase_name, polystep_cfg, dry_run)

    set_seed(seed)
    model = config["model_fn"]().to(device)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _load_split(showcase_name, seed, audit_no_leakage)
    if tuning:
        if val_loader is None:
            raise ValueError("tuning=True needs a validation split to select on")
        test_loader = TestSplitTripwire()
    selection_loader = val_loader if val_loader is not None else test_loader

    total_steps = epochs * len(train_loader)
    layout, subspace, optimizer = build_polystep(model, seed, total_steps, polystep_cfg)
    eval_budget = polystep_eval_budget(optimizer, total_steps)

    import copy

    evaluator = NNCostEvaluator(
        model,
        loss_fn=loss_fn,
        compile_vmap=polystep_cfg.get("compile_evaluator", False),
        # Not False: the evaluator default None means "auto-enable wherever the
        # in-place path runs".
        compile_forward=polystep_cfg.get("compile_forward"),
    )
    # NOT registered: the site-aware / delta evaluator path register_evaluator
    # switches on is slower for these models; the cost is the delta path, not the
    # registration itself.
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
                    return evaluator.evaluate(batched_params, _data, _targets)

                optimizer.step(closure)
                # Read from the optimizer, not the closure: a registered fast path
                # never calls the closure, so counting there would undercount.
                fwd_pass_count = optimizer.candidate_evals

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
                    "val_accuracy": selection_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                    "wall_time": time.time() - start_time,
                }
            )
            print(
                f"    Epoch {epoch + 1}/{epochs} | train={train_acc * 100:.1f}% | val={selection_acc * 100:.1f}% | loss={avg_loss:.4f}"
            )

    wall_time = time.time() - start_time
    # Test is read exactly once per run, here, on the validation-selected weights.
    last_epoch_acc = selection_acc
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    final_acc = float("nan") if tuning else evaluate_accuracy(model, test_loader, device=device)

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
            # Symmetric with the baseline runner, so a reader can check both sides
            # were tuned and at what cost; None means the run used the transplanted
            # default.
            "tuned": polystep_tuned_name,
            "tuning_provenance": polystep_tuned_prov,
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
    """Run one ``polystep.baselines`` method on a non-differentiable showcase."""
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    config = SHOWCASE_CONFIGS[showcase_name]
    cfg = POLYSTEP_CONFIGS[showcase_name]
    if theory_mode:
        cfg = apply_theory_mode(cfg)
    epochs = _epochs_for(showcase_name, cfg, dry_run)
    match_axis = "evals"
    deadline_s = None
    if dry_run and budget is None:
        budget = 2000
        match_axis = "dry-run"

    set_seed(seed)
    model = config["model_fn"]().to(device)
    layout = ParamLayout.from_module(model)
    train_loader, val_loader, test_loader = _load_split(showcase_name, seed, audit_no_leakage)

    subspace = None
    if fair:
        subspace = make_subspace(layout, rank=cfg["rank"], seed=seed, method=method)
        # The dimension every method searches. EGGROLL alone gets FactoredSubspace,
        # which is smaller by construction; recording the gap shows the table is
        # matched on rank, not dimension.
        shared_dim = make_subspace(layout, rank=cfg["rank"], seed=seed, method="polystep").subspace_dim
        if budget is None:
            budget = fair_eval_budget(showcase_name, seed, device, train_loader, epochs, cfg)
            budget, match_axis, deadline_s = matched_budget(
                method,
                showcase=showcase_name,
                seed=seed,
                polystep_steps=epochs * len(train_loader),
                eval_budget=budget,
                results_dir=results_dir,
            )
    if budget is None:
        budget = LEGACY_BUDGETS[method]
        match_axis = "legacy"

    loss_batch = minibatch_loss(NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss()), train_loader, device)
    selection_loader = val_loader if val_loader is not None else test_loader

    # The config tune_gallery.py picked on validation, or None if nothing was swept.
    selected, provenance = load_selection("gallery", showcase_name, method)
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
        deadline_s=deadline_s,
    )
    out["hyperparameters"]["fair"] = fair
    out["hyperparameters"]["tuned"] = selected["name"] if selected else None
    out["hyperparameters"]["tuning_provenance"] = provenance
    # Keep comparisons with step and evaluation budgets in separate tables.
    out["hyperparameters"]["match_axis"] = match_axis
    out["hyperparameters"]["polystep_steps"] = epochs * len(train_loader)
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
            "Run the unaccelerated reference configuration: probe_radius_jitter=0.05 "
            "with the smooth density, independent rotations, flat epsilon, step "
            "radius r0*(t+1)^-(1/2+0.1), orthoplex, HybridSubspace, no momentum / "
            "amortization / Anderson."
        ),
    )
    parser.add_argument("--budget", type=int, default=None, help="Override the candidate budget (smoke runs)")
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=None,
        help=(
            "Override probe_radius in POLYSTEP_CONFIGS. Sets the probe reach "
            "r_p*eps, which is the width a plateau must stay under for a probe "
            "than for the step to be nonzero. Used by the escape sweep."
        ),
    )
    parser.add_argument(
        "--probe-jitter",
        type=float,
        default=None,
        help="Override probe_radius_jitter (eta_max). Escape sweep knob.",
    )
    parser.add_argument(
        "--step-jitter",
        type=float,
        default=None,
        help=(
            "Override step_radius_jitter. Cor. 4.15 condition (iii) needs it "
            "for the one-step law to have a density on an annulus. Escape sweep knob."
        ),
    )
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
    # Overrides apply only to the showcases actually being run; applying them to
    # every POLYSTEP_CONFIGS entry would force one radius on benchmarks tuned to
    # different ones.
    _targets = [sc for sc in args.showcases if sc in POLYSTEP_CONFIGS]

    def _override_radius(key, value):
        """Set a radius on the flat key or on both schedule endpoints.

        ``sched`` ignores the flat key whenever ``{key}_init`` is present, so
        writing only the flat key is a silent no-op on cosine-scheduled benchmarks.
        """
        for sc in _targets:
            cfg = POLYSTEP_CONFIGS[sc]
            if f"{key}_init" in cfg:
                cfg[f"{key}_init"] = value
                cfg[f"{key}_target"] = value
            else:
                cfg[key] = value

    if args.step_radius is not None:
        print(f"  Step radius override: {args.step_radius}")
        _override_radius("step_radius", args.step_radius)
    if args.epochs_polystep is not None:
        global EPOCHS_POLYSTEP, EPOCHS_POLYSTEP_NONSNN
        EPOCHS_POLYSTEP = args.epochs_polystep
        EPOCHS_POLYSTEP_NONSNN = args.epochs_polystep
        print(f"  polystep epochs override: {args.epochs_polystep}")
    if args.probe_radius is not None:
        print(f"  probe_radius override: {args.probe_radius}")
        _override_radius("probe_radius", args.probe_radius)
    # The jitters are read by cfg.get, not through sched, so the flat key is correct.
    for flag, key in (
        (args.probe_jitter, "probe_radius_jitter"),
        (args.step_jitter, "step_radius_jitter"),
    ):
        if flag is not None:
            for sc in _targets:
                POLYSTEP_CONFIGS[sc][key] = flag
            print(f"  {key} override: {flag}")
    print()

    failures: list = []
    for showcase_name in args.showcases:
        if showcase_name not in SHOWCASE_CONFIGS:
            print(f"Unknown showcase: {showcase_name}")
            continue

        print(f"=== {showcase_name} ===")

        for method in args.methods:
            method = METHOD_ALIASES.get(method, method)
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
                    failures.append((showcase_name, method, seed, repr(e)))
                finally:
                    # Prevent CUDA memory accumulation across sequential runs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print()

    print(f"Done. Results in {args.results_dir}")
    # Non-zero on any swallowed failure so the launcher does not read a half-run
    # grid as complete; the per-run except stays so one bad cell does not lose the
    # rest.
    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for showcase, method, seed, err in failures:
            print(f"  {showcase}/{method} seed={seed}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
