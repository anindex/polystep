#!/usr/bin/env python
"""Sweep PolyStep's own grid on a gallery benchmark, on validation only.

This is the symmetric half of ``tune_gallery.py``: same grid size, same seed, same
validation-only selection, so PolyStep and the baselines get equal tuning budget.

PolyStep's grid entries are *multipliers* on ``epsilon``/``step_radius``/``probe_radius``
(``fairness.apply_polystep_multipliers``), not the additive points the baselines use, so
this cannot route through ``run_baseline``. It drives ``run_elevation.run_polystep``
instead and reads back ``metrics.best_accuracy``, which that runner already computes on
the *selection* split. Test is never used for selection here.

    python experiments/scripts/tune_polystep.py --showcase snn --epochs 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, ".")

from experiments.runners.fairness import (  # noqa: E402
    TUNING_GRID,
    refine_grid,
    apply_polystep_multipliers,
    select_best,
    tuning_cost,
    write_selection,
)


# --- which module owns which showcase ---------------------------------------------
#
# Each adapter swaps its module's config, runs one PolyStep cell in tuning mode,
# and puts the config back.


def _elevation_adapter(showcase):
    from experiments.runners import run_elevation as R

    def get_base():
        return dict(R.POLYSTEP_CONFIGS[showcase])

    def set_cfg(cfg):
        R.POLYSTEP_CONFIGS[showcase] = cfg

    def set_epochs(epochs):
        R.EPOCHS_POLYSTEP = epochs
        R.EPOCHS_POLYSTEP_NONSNN = epochs

    def run(seed, device, cell):
        R.run_polystep(showcase, seed, device, cell, fair=True, tuning=True)

    return get_base, set_cfg, set_epochs, run


def _mnist_adapter(_showcase):
    from experiments.runners import run_mnist as R
    from experiments.runners.common import make_train_val_split

    state = {"epochs": None}

    def get_base():
        return dict(R.POLYSTEP_CONFIG)

    def set_cfg(cfg):
        R.POLYSTEP_CONFIG = cfg

    def set_epochs(epochs):
        state["epochs"] = epochs

    def run(seed, device, cell):
        train, test = R.load_mnist(batch_size=R.BATCH_SIZE)
        train, val = make_train_val_split(train, val_frac=0.1, seed=seed)
        R.run_polystep(
            seed,
            device,
            train,
            test,
            cell,
            val_loader=val,
            fair=True,
            tuning=True,
            epochs=state["epochs"],
        )

    return get_base, set_cfg, set_epochs, run


def _moe_adapter(_showcase):
    from experiments.runners import run_moe as R

    state = {"epochs": None}

    def get_base():
        return dict(R.POLYSTEP_CONFIG)

    def set_cfg(cfg):
        R.POLYSTEP_CONFIG = cfg

    def set_epochs(epochs):
        state["epochs"] = epochs

    def run(seed, device, cell):
        R.run_polystep(
            seed,
            device,
            cell,
            epochs=state["epochs"] or R.EPOCHS,
            fair=True,
            tuning=True,
        )

    return get_base, set_cfg, set_epochs, run


ADAPTERS = {"mnist": _mnist_adapter, "moe": _moe_adapter}


def adapter_for(showcase):
    """``(get_base, set_cfg, set_epochs, run)`` for ``showcase``."""
    return ADAPTERS.get(showcase, _elevation_adapter)(showcase)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--showcase", default="snn")
    p.add_argument("--seed", type=int, default=42, help="Tuning seed.")
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs per trial. Fewer than the reported run: this ranks configs, it "
        "does not produce the number. The matched budget scales with it, so every "
        "trial stays budget-matched to itself.",
    )
    p.add_argument("--out", default="experiments/results/tuning/polystep_trials")
    p.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Tuning rounds, matching tune_gallery.py. Round two recentres the "
        "multiplier grid on round one's winner, so PolyStep and the baselines get "
        "the same number of rounds at the same cell count.",
    )
    p.add_argument(
        "--step-jitter",
        type=float,
        default=None,
        help="Set step_radius_jitter on the base config before applying multipliers. "
        "Must match what the reported run passes, or the grid is swept around a "
        "different config than the one it selects for.",
    )
    p.add_argument(
        "--selection-path",
        default="experiments/results/tuning/polystep_selected.json",
        help="Deliberately not the shared selected_configs.json. Runners read that file "
        "with load_selection, which swallows a parse error and silently falls back to "
        "the untuned config, so writing it while a table is running can untune a seed "
        "without saying so. Merge into the shared file once the table is done.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_base, set_cfg, set_epochs, run_cell = adapter_for(args.showcase)
    base = get_base()
    if args.step_jitter is not None:
        base["step_radius_jitter"] = args.step_jitter
    grid = TUNING_GRID["polystep"]
    os.makedirs(args.out, exist_ok=True)

    set_epochs(args.epochs)

    trials = []

    def sweep(points, rnd):
        print(
            f"[{args.showcase}] polystep round {rnd}/{args.rounds}: {len(points)} configs "
            f"x {args.epochs} epochs, seed {args.seed}\n",
            flush=True,
        )
        for i, point in enumerate(points):
            name = "_".join(f"{k}{v:g}" for k, v in sorted(point.items()))
            cell = os.path.join(args.out, f"r{rnd}_cell{i:02d}")
            shutil.rmtree(cell, ignore_errors=True)
            os.makedirs(cell, exist_ok=True)
            set_cfg(apply_polystep_multipliers(base, point))
            t0 = time.time()
            try:
                run_cell(args.seed, device, cell)
            except Exception as e:  # one bad cell must not lose the sweep
                print(f"  {i:>2} {name:<34} ERROR {type(e).__name__}: {e}", flush=True)
                continue
            finally:
                set_cfg(base)
            hit = glob.glob(os.path.join(cell, "*.json"))
            if not hit:
                print(f"  {i:>2} {name:<34} no result written", flush=True)
                continue
            d = json.load(open(hit[0]))
            # best_accuracy is the selection-split score: the runner is called with
            # tuning=True, so test_accuracy_at_selected is NaN and the test split
            # raises if touched.
            val = d["metrics"]["best_accuracy"]
            trials.append(
                {
                    "showcase": args.showcase,
                    "method": "polystep",
                    "round": rnd,
                    "index": i,
                    "name": name,
                    "point": point,
                    "val": val,
                    "evals": d["hyperparameters"]["eval_budget"],
                }
            )
            print(f"  {i:>2} {name:<34} val={val:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    points = grid
    for rnd in range(1, args.rounds + 1):
        sweep(points, rnd)
        if not trials or rnd == args.rounds:
            break
        # Same recentring rule the baselines get, so neither side is swept over a
        # range the other was not.
        points = refine_grid("polystep", select_best(trials)["point"])

    if not trials:
        print("no trials completed")
        return 1
    best = select_best(trials)
    print(f"\n  -> polystep: {best['name']} val={best['val']:.4f}")
    print(f"     multipliers {best['point']}")
    path = write_selection(
        "gallery",
        trials,
        {
            "sweep": "experiments/scripts/tune_polystep.py",
            "showcase": args.showcase,
            "seed": args.seed,
            "epochs": args.epochs,
            "selection_split": "validation",
        },
        # Per-trial cost is the budget the runner actually spent, not 0.
        {
            args.showcase: {
                "polystep": tuning_cost(
                    "polystep",
                    max(t["evals"] for t in trials),
                    seeds=1,
                    rounds=args.rounds,
                    configs=len(trials),
                )
            }
        },
        path=args.selection_path,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
