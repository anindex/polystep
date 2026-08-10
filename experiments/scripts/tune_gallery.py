#!/usr/bin/env python
"""Tune the gradient-free baselines on the gallery benchmarks, on validation only.

The gallery runners do not sweep :data:`fairness.TUNING_GRID` themselves; this
script does and writes the picks to ``experiments/results/tuning/selected_configs.json``.

Selection is on the validation split at a reduced budget; the headline runs then spend
the full matched budget at the selected config, on all five seeds. Test is never read.

    python experiments/scripts/tune_gallery.py --showcases snn mnist \
        --methods openai_es spsa cma_es mezo random_search eggroll \
        --budget-frac 0.05
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from experiments.runners.common import (  # noqa: E402
    evaluate_accuracy,
    reseed_loaders,
    load_mnist,
    make_train_val_split,
    set_seed,
)
from experiments.runners.fairness import (
    SEQUENTIAL_METHODS,
    step_matched_budget,
    refine_grid,  # noqa: E402
    TUNING_GRID,
    make_subspace,
    minibatch_loss,
    probe_scale_of,
    run_baseline,
    select_best,
    tuning_configs,
    tuning_cost,
    write_selection,
)
from polystep.cost_nn import NNCostEvaluator  # noqa: E402
from polystep.transform import ParamLayout  # noqa: E402

GF_METHODS = ("openai_es", "spsa", "cma_es", "mezo", "random_search", "eggroll")


def setup(showcase: str, seed: int, device, epochs: int = None):
    """``(model_fn, cfg, train_loader, val_loader, full_budget, total_steps)`` for one showcase.

    ``total_steps`` is what PolyStep takes for its configured epochs, which is the
    step budget the sequential baselines are matched to.

    The test loader is deliberately not returned: a tuning sweep has no business
    holding one.
    """
    if showcase == "mnist":
        from experiments.runners.run_mnist import (
            BATCH_SIZE,
            EPOCHS,
            POLYSTEP_CONFIG,
            MNISTNet,
            fair_eval_budget,
        )

        train, _ = load_mnist(batch_size=BATCH_SIZE)
        train, val = make_train_val_split(train, val_frac=0.1, seed=seed)
        cfg = dict(POLYSTEP_CONFIG)
        budget = fair_eval_budget(seed, device, train, epochs or EPOCHS, cfg, "softmax")
        return (lambda: MNISTNet().to(device)), cfg, train, val, budget, (epochs or EPOCHS) * len(train)

    if showcase == "moe":
        # run_moe.run_gradient_free reads load_selection("gallery", "moe", ...), so
        # this branch is what lets its baselines be tuned at all.
        from experiments.runners.run_moe import (
            EPOCHS,
            POLYSTEP_CONFIG,
            HardMoENet,
            _load_split,
            fair_eval_budget,
        )

        train, val, _ = _load_split(seed, True)
        cfg = dict(POLYSTEP_CONFIG)
        budget = fair_eval_budget(seed, device, train, epochs or EPOCHS, cfg)
        return (lambda: HardMoENet(num_experts=4).to(device)), cfg, train, val, budget, (epochs or EPOCHS) * len(train)

    from experiments.runners.run_elevation import (
        POLYSTEP_CONFIGS,
        SHOWCASE_CONFIGS,
        _epochs_for,
        _load_split,
        fair_eval_budget,
    )

    train, val, _ = _load_split(showcase, seed, True)
    cfg = dict(POLYSTEP_CONFIGS[showcase])
    epochs = epochs or _epochs_for(showcase, cfg, False)
    budget = fair_eval_budget(showcase, seed, device, train, epochs, cfg)
    model_fn = SHOWCASE_CONFIGS[showcase]["model_fn"]
    return (lambda: model_fn().to(device)), cfg, train, val, budget, epochs * len(train)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--showcases", nargs="+", default=["snn"])
    p.add_argument("--methods", nargs="+", default=list(GF_METHODS))
    p.add_argument("--seed", type=int, default=42, help="Tuning seed. Not a headline seed.")
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs the matched budget is derived from. Must match the runners' "
        "--epochs-polystep, or the config is tuned at a budget the headline "
        "runs never spend.",
    )
    p.add_argument(
        "--budget-frac",
        type=float,
        default=0.05,
        help="Fraction of the matched budget each trial gets. 0.05 separated the SNN "
        "grid by 49 points, so it ranks configurations without paying for five.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Tuning rounds. Round two recentres each method's grid on round one's "
        "winner, which matters because a winner on the grid edge means the optimum "
        "was never swept: on the SNN five of six baselines landed on an edge. Same "
        "cell count per round for every method, so the tuning budget stays equal.",
    )
    p.add_argument(
        "--selection-path",
        default=None,
        help="Where to write the picks. Defaults to the shared selected_configs.json "
        "that the runners read. Point a smoke or trial sweep somewhere else: writing "
        "the shared file while a table is running can untune a seed in flight.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trials, costs = [], {}

    for showcase in args.showcases:
        set_seed(args.seed)
        model_fn, cfg, train_loader, val_loader, full_budget, polystep_steps = setup(
            showcase, args.seed, device, args.epochs
        )
        layout = ParamLayout.from_module(model_fn())

        for method in args.methods:
            # The sequential methods are budgeted on steps, not evaluations, so they
            # are tuned at the full step-matched budget they are reported at: a cell
            # tuned at the reported budget cannot fail to transfer.
            if method in SEQUENTIAL_METHODS:
                budget = step_matched_budget(method, polystep_steps)
            else:
                budget = max(1000, int(full_budget * args.budget_frac))
            dim = make_subspace(layout, rank=cfg["rank"], seed=args.seed, method=method).subspace_dim
            scale = probe_scale_of(cfg, dim=dim)
            costs.setdefault(showcase, {})[method] = tuning_cost(method, budget, seeds=1, rounds=args.rounds)

            def sweep(points, rnd):
                """Score one round's points and append them to ``trials``."""
                configs = tuning_configs(method, probe_scale=scale, seed=args.seed, points=points)
                print(
                    f"\n[{showcase}] {method} round {rnd}/{args.rounds}: {len(configs)} "
                    f"configs x {budget:,} evals (dim={dim}, probe_scale={scale:.5g})",
                    flush=True,
                )
                for i, hp in enumerate(configs):
                    # SPSA's gain is a_k = a / (A + k)^alpha, and A defaults to 10% of
                    # the affordable iterations. Pin A to the budget the cell is
                    # reported at, so the schedule tuned is the schedule run.
                    if method == "spsa":
                        hp = {**hp, "A": 0.1 * (budget // 2)}
                    name = "_".join(f"{k}{v:g}" for k, v in sorted(points[i].items()))
                    set_seed(args.seed)
                    # setup() builds the loaders once and every cell shares them, so
                    # rewind their shuffle generators or cell i trains on the i-th
                    # minibatch stream and position leaks into the ranking.
                    reseed_loaders(args.seed, train_loader, val_loader)
                    model = model_fn()
                    sub = make_subspace(layout, rank=cfg["rank"], seed=args.seed, method=method)
                    loss_batch = minibatch_loss(
                        NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss()), train_loader, device
                    )
                    t0 = time.time()
                    try:
                        out = run_baseline(
                            method,
                            model=model,
                            layout=layout,
                            loss_batch=loss_batch,
                            budget=budget,
                            val_fn=lambda m: evaluate_accuracy(m, val_loader, device=device),
                            # A tuning sweep that can reach test is a leak waiting to happen.
                            test_fn=lambda m: float("nan"),
                            mode="max",
                            seed=args.seed,
                            subspace=sub,
                            subspace_rank=cfg["rank"],
                            hp=hp,
                            probe_scale=scale,
                        )
                    except Exception as e:  # one bad cell must not lose the sweep
                        print(f"  {i:>2} {name:<28} ERROR {type(e).__name__}: {e}", flush=True)
                        continue
                    val = out["metrics"]["best_val_accuracy"]
                    trials.append(
                        {
                            "showcase": showcase,
                            "method": method,
                            "round": rnd,
                            "index": i,
                            "name": name,
                            "point": points[i],
                            "hp": hp,
                            "val": val,
                        }
                    )
                    print(f"  {i:>2} {name:<28} val={val:.4f}  ({time.time() - t0:.0f}s)", flush=True)

            def picked_so_far():
                return [t for t in trials if t["showcase"] == showcase and t["method"] == method]

            points = TUNING_GRID[method]
            for rnd in range(1, args.rounds + 1):
                sweep(points, rnd)
                if not picked_so_far() or rnd == args.rounds:
                    break
                # Round one's winner can sit on the grid edge; recentring on it
                # extends the range outward or refines around it, at the same cell
                # count for every method.
                points = refine_grid(method, select_best(picked_so_far())["point"])

            # Count cells that actually ran, not cells the grid intended: a cell
            # that raises is dropped by the per-cell except above.
            costs.setdefault(showcase, {})[method] = tuning_cost(
                method, budget, seeds=1, rounds=args.rounds, configs=len(picked_so_far())
            )
            if picked_so_far():
                best = select_best(picked_so_far())
                print(f"  -> {method}: {best['name']} val={best['val']:.4f}", flush=True)

    path = write_selection(
        "gallery",
        trials,
        {
            "sweep": "experiments/scripts/tune_gallery.py",
            "showcases": args.showcases,
            "seed": args.seed,
            "budget_frac": args.budget_frac,
            "selection_split": "validation",
        },
        costs,
        **({"path": args.selection_path} if args.selection_path else {}),
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
