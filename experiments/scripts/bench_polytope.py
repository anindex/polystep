"""Which polytope pays for itself, at a fixed forward-evaluation budget.

Reproduces the table in ``docs/performance.md``. The budget is candidate evaluations,
not steps: the orthoplex spends ``2k`` per step against the simplex's ``k+1``, so a
fixed step count would hand it more forwards and the comparison would be meaningless.

    python experiments/scripts/bench_polytope.py
"""

import argparse
import statistics

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator

torch.set_num_threads(max(1, (torch.get_num_threads() or 1) - 8))

SEEDS = (0, 1, 2, 3, 4, 5)
BUDGET = 300_000

CONFIGS = {
    "simplex (default)": dict(polytope_type="simplex"),
    "orthoplex alone": dict(polytope_type="orthoplex"),
    "orthoplex + multifidelity_screen": dict(polytope_type="orthoplex", multifidelity_screen=True),
    "orthoplex + use_quadratic_model + trust_region": dict(
        polytope_type="orthoplex", use_quadratic_model=True, trust_region=True, num_probe=2
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
    args = ap.parse_args()

    print("| config | loss reduction | steps |\n|---|---|---|")
    for name, kwargs in CONFIGS.items():
        results = [run(name, kwargs, s) for s in args.seeds]
        reduction = statistics.mean(r for r, _ in results)
        steps = statistics.mean(s for _, s in results)
        print(f"| `{name}` | {reduction:.3f} | {steps:.0f} |")


if __name__ == "__main__":
    main()
