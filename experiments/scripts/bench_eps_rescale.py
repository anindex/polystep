"""Benchmark: warm-start dual rescaling across an epsilon anneal.

Resolves a contested design point: SinkhornSolver rescales warm-start dual
potentials by ``eps_new/eps_old`` when epsilon changes between solves
(solvers/sinkhorn.py). This is mathematically unjustified for cost-unit
potentials (the entropic dual is not homogeneous in epsilon), yet the original
docstring claimed a 5-10x iteration reduction. This script measures it directly.

Strategies compared across a decreasing epsilon schedule with drifting cost
matrices (warm-starting each solve from the previous one):
  - rescale : f,g *= eps_new/eps_old   (current behavior; pass init_eps)
  - none    : keep f,g unchanged       (no init_eps)
  - reset   : zero the warm start when |log(eps_new/eps_old)| > jump
  - cold    : always start from zeros  (no warm-start reference)

Metric: total inner Sinkhorn iterations to reach threshold, fraction of solves
that hit max_iterations (non-convergence), mean final marginal error. Run:
    python experiments/scripts/bench_eps_rescale.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polystep.solvers import SinkhornSolver  # noqa: E402


def make_schedule(T: int, eps_hi: float = 1.0, eps_lo: float = 0.01):
    """Geometric anneal from eps_hi to eps_lo over T steps."""
    ratio = (eps_lo / eps_hi) ** (1.0 / (T - 1))
    return [eps_hi * ratio**t for t in range(T)]


def make_costs(T: int, P: int, V: int, scale: float, seed: int = 0):
    """A sequence of drifting cost matrices (simulates moving particles)."""
    g = torch.Generator().manual_seed(seed)
    C = torch.rand(P, V, generator=g) * scale
    costs = []
    for _ in range(T):
        C = C + 0.05 * scale * torch.randn(P, V, generator=g)
        costs.append(C.clamp(min=0).clone())
    return costs


def run(strategy, schedule, costs, *, threshold=1e-6, max_iters=2000, jump=0.7):
    solver = SinkhornSolver(threshold=threshold, max_iterations=max_iters)
    f = g = None
    prev_eps = None
    total_iters = 0
    nonconv = 0
    errs = []
    for eps, C in zip(schedule, costs):
        solver.epsilon = eps
        kwargs = {"cost_matrix": C, "scale_cost": None}
        if strategy != "cold":
            kwargs["init_f"], kwargs["init_g"] = f, g
            if prev_eps is not None:
                if strategy == "rescale":
                    kwargs["init_eps"] = prev_eps
                elif strategy == "reset" and abs(math.log(eps / prev_eps)) > jump:
                    kwargs["init_f"] = kwargs["init_g"] = None
        res = solver.solve(**kwargs)
        total_iters += res.n_iters
        nonconv += int(not res.converged)
        if res.errors:
            errs.append(res.errors[-1])
        f, g = res.f, res.g
        prev_eps = eps
    mean_err = sum(errs) / len(errs) if errs else float("nan")
    return total_iters, nonconv, mean_err


def main():
    T, P, V = 60, 64, 4
    schedule = make_schedule(T)
    regimes = {"scaled (|C|~1)": 1.0, "raw (|C|~50)": 50.0}
    strategies = ["cold", "none", "rescale", "reset"]

    for label, scale in regimes.items():
        costs = make_costs(T, P, V, scale)
        print(f"\n=== {label}, T={T}, P={P}, V={V}, eps 1.0->0.01 ===")
        print(f"{'strategy':>10} | {'tot_iters':>9} | {'nonconv':>7} | {'mean_err':>9}")
        print("-" * 46)
        for s in strategies:
            it, nc, err = run(s, schedule, costs)
            print(f"{s:>10} | {it:>9} | {nc:>7} | {err:>9.2e}")


if __name__ == "__main__":
    main()
