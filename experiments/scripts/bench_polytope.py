"""Which polytope pays for itself, at a fixed forward-evaluation budget.

The budget is candidate evaluations,
not steps: the orthoplex spends ``2k`` per step against the simplex's ``k+1``, so a
fixed step count would hand it more forwards and the comparison would be meaningless.

    python experiments/scripts/bench_polytope.py
"""

import argparse
import statistics
import json
from pathlib import Path

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


SEEDS = (0, 1, 2, 3, 4, 5)
BUDGET = 300_000

CONFIGS = {
    "simplex (default)": dict(polytope_type="simplex"),
    "orthoplex alone": dict(polytope_type="orthoplex"),
    "orthoplex + multifidelity_screen": dict(polytope_type="orthoplex", multifidelity_screen=True),
    "orthoplex + use_quadratic_model + trust_region": dict(
        polytope_type="orthoplex", use_quadratic_model=True, trust_region=True, num_probe=2
    ),
    "orthoplex + use_quadratic_model + trust_region, K=1": dict(
        polytope_type="orthoplex", use_quadratic_model=True, trust_region=True, num_probe=1
    ),
    # A centred tight frame also supports the fit on d+1 vertices instead of 2d.
    # Curvature is the trace, from the shared f(X).
    "simplex + use_quadratic_model + trust_region": dict(
        polytope_type="simplex", use_quadratic_model=True, trust_region=True, num_probe=1
    ),
}


def _task(seed):
    torch.manual_seed(seed)
    x = torch.randn(256, 20)
    y = torch.randint(0, 3, (256,))
    return nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 3)), x, y


def run(name, kwargs, seed):
    model, x, y = _task(seed)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    optimizer = PolyStepOptimizer(model, epsilon=0.1, step_radius=0.3, seed=seed, compile=False, **kwargs)
    optimizer.register_evaluator(evaluator, x, y)

    def closure(params, inputs=x, targets=y):
        return evaluator.evaluate(params, inputs, targets)

    with torch.no_grad():
        start = nn.functional.cross_entropy(model(x), y).item()

    # The screen needs a cheap low-fidelity closure; without one it silently declines
    # and that row would just repeat the plain orthoplex.
    screen = optimizer.screen_closure_from(closure, x, y)

    spent, steps = 0, 0
    while spent < BUDGET:
        optimizer.step(closure, screen_closure=screen)
        steps += 1
        spent = sum(optimizer.state.evals)

    with torch.no_grad():
        end = nn.functional.cross_entropy(model(x), y).item()
    return start - end, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--suite", action="store_true", help="Run the representative algorithm pilot")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workloads", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--wall", action="store_true", help="Also compare at the baseline wall budget")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("experiments/results/benchmarks/algorithms.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.suite:
        from experiments.runners.search_suite import ARMS, WORKLOADS, run_search

        rows = []
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for name in args.workloads or WORKLOADS:
            for seed in args.seeds:
                base = run_search(
                    name,
                    "hybrid",
                    seed,
                    device=args.device,
                    budget=args.budget,
                    batch=args.batch,
                    streaming=args.streaming,
                )
                target = base["start_validation_loss"] - max(
                    0.01, 0.5 * (base["start_validation_loss"] - base["best_validation_loss"])
                )
                base["target"] = target
                base["time_to_target"] = next(
                    (h["seconds"] for h in base["history"] if h["validation_loss"] <= target), None
                )
                base["axis"] = "evaluations"
                rows.append(base)
                for arm in args.arms or ARMS:
                    for axis in ("evaluations", "wall") if args.wall else ("evaluations",):
                        if arm == "hybrid" and axis == "evaluations":
                            continue
                        row = run_search(
                            name,
                            arm,
                            seed,
                            device=args.device,
                            budget=10**9 if axis == "wall" else args.budget,
                            seconds=base["seconds"] if axis == "wall" else None,
                            batch=args.batch,
                            target=target,
                            streaming=args.streaming,
                        )
                        row["axis"] = axis
                        rows.append(row)
                        print(json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)
                    args.output.write_text(json.dumps(rows, indent=2) + "\n")
        return

    print("| config | loss reduction | steps |\n|---|---|---|")
    for name, kwargs in CONFIGS.items():
        results = [run(name, kwargs, s) for s in args.seeds]
        reduction = statistics.mean(r for r, _ in results)
        steps = statistics.mean(s for _, s in results)
        print(f"| `{name}` | {reduction:.3f} | {steps:.0f} |")


if __name__ == "__main__":
    main()
