"""Ask/tell benchmark: PolyStep vs a Gaussian ES on synthetic objectives.

Both optimizers use the same ask/tell interface and the same per-generation
evaluation budget (popsize x generations), mirroring the evosax / NeuroEvoBench
protocol. This shows PolyStep as a drop-in gradient-free optimizer directly
comparable to evolution strategies. Pure torch, no extra dependencies.

Run:
    python experiments/bench_ask_tell.py
"""

from __future__ import annotations

import argparse

import torch

from polystep import PolyStepES
from polystep.objectives.synthetic import Ackley, Rastrigin, Rosenbrock, Sphere

OBJECTIVES = {"sphere": Sphere, "ackley": Ackley, "rastrigin": Rastrigin, "rosenbrock": Rosenbrock}


class GaussianES:
    """OpenAI-ES style ask/tell baseline (isotropic Gaussian, z-scored update)."""

    def __init__(self, dim, popsize, x0, sigma=0.3, lr=0.1, seed=0):
        self.dim = dim
        self.popsize = popsize
        self.sigma = sigma
        self.lr = lr
        self.mean = x0.clone()
        self.generator = torch.Generator().manual_seed(seed)
        self._eps = None
        self.best_fitness = float("inf")

    def ask(self):
        self._eps = torch.randn(self.popsize, self.dim, generator=self.generator)
        return self.mean.unsqueeze(0) + self.sigma * self._eps

    def tell(self, fitness):
        adv = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        grad = (self._eps * adv.unsqueeze(1)).mean(dim=0) / self.sigma
        self.mean = self.mean - self.lr * grad  # descend the loss
        self.best_fitness = min(self.best_fitness, fitness.min().item())


def run(optimizer, fn, generations):
    for _ in range(generations):
        optimizer.tell(fn(optimizer.ask()))
    return optimizer.best_fitness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x0 = torch.full((args.dim,), 2.0)
    popsize = 2 * args.dim  # PolyStep orthoplex population for a single particle

    header = f"{'objective':<12}{'PolyStep':>14}{'GaussianES':>14}{'evals':>10}"
    print(f"dim={args.dim}  generations={args.generations}  popsize={popsize}")
    print(header)
    print("-" * len(header))
    for name, Obj in OBJECTIVES.items():
        obj = Obj(dim=args.dim)
        ps = PolyStepES(args.dim, num_particles=1, epsilon=0.1, step_radius=0.3, x0=x0, seed=args.seed)
        es = GaussianES(args.dim, popsize=popsize, x0=x0, seed=args.seed)
        ps_best = run(ps, obj, args.generations)
        es_best = run(es, obj, args.generations)
        print(f"{name:<12}{ps_best:>14.4e}{es_best:>14.4e}{popsize * args.generations:>10}")


if __name__ == "__main__":
    main()
