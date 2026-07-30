"""OpenAI-ES ask/tell baseline, shared by the examples that compare against it."""

from __future__ import annotations

import torch


class OpenAIES:
    """OpenAI-ES (Salimans et al., 2017).

    Antithetic sampling, z-scored fitness shaping, and the gradient estimate
    ``g = (1 / (pop * sigma)) * sum(shaped * eps)``. Every caller passes its own
    ``sigma`` and ``lr``.
    """

    def __init__(self, dim, popsize, x0, sigma=0.5, lr=0.2, seed=0):
        self.dim = dim
        self.popsize = popsize + (popsize % 2)  # even, for antithetic pairs
        self.sigma = sigma
        self.lr = lr
        self.mean = x0.clone()
        self.generator = torch.Generator().manual_seed(seed)
        self._eps = None
        self.best_fitness = float("inf")

    def ask(self):
        half = torch.randn(self.popsize // 2, self.dim, generator=self.generator)
        self._eps = torch.cat([half, -half], dim=0)
        return self.mean.unsqueeze(0) + self.sigma * self._eps

    def tell(self, fitness):
        self.best_fitness = min(self.best_fitness, fitness.min().item())
        adv = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
        self.mean = self.mean - self.lr * (self._eps * adv.unsqueeze(1)).mean(dim=0) / self.sigma


def run(opt, fit_fn, generations):
    """Best-accuracy-so-far curve over ``generations`` ask/tell rounds."""
    curve = []
    for _ in range(generations):
        opt.tell(fit_fn(opt.ask()))
        curve.append(100.0 * (1.0 - opt.best_fitness))
    return curve
