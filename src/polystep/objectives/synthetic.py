"""Synthetic test functions for benchmarking optimization algorithms."""
import math
from typing import Optional

import torch

from .base import ObjectiveFn


class Ackley(ObjectiveFn):
    """Ackley test function.

    f(x) = -a*exp(-b*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(c*x_i))) + a + e

    Global minimum at x = 0 with f(x) = 0.
    """

    def __init__(
        self,
        dim: int = 2,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2 * math.pi,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[torch.Tensor] = None,
    ):
        if bounds is None:
            bounds = torch.tensor([[-6.0, 6.0]] * dim)
        optimizers = torch.zeros(1, dim)
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.linalg.norm(X, dim=-1))
        part2 = -torch.exp(torch.mean(torch.cos(c * X), dim=-1))
        return part1 + part2 + a + math.e


class Rosenbrock(ObjectiveFn):
    """Rosenbrock (banana) function. Global minimum at x = (1,...,1) with f(x) = 0."""

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-5.0, 5.0]] * dim)
        optimizers = torch.ones(1, dim)
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            dim=-1,
        )


class Rastrigin(ObjectiveFn):
    """Rastrigin function. Global minimum at x = 0 with f(x) = 0."""

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-5.12, 5.12]] * dim)
        optimizers = torch.zeros(1, dim)
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return 10.0 * self.dim + torch.sum(
            X ** 2 - 10.0 * torch.cos(2.0 * math.pi * X), dim=-1,
        )


class StyblinskiTang(ObjectiveFn):
    """Styblinski-Tang function. Global minimum at x = -2.903534 per dim."""

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-5.0, 5.0]] * dim)
        optimizers = torch.full((1, dim), -2.903534)
        optimal_value = -39.166166 * dim
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=optimal_value, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(dim=-1)


class Levy(ObjectiveFn):
    """Levy function. Global minimum at x = (1,...,1) with f(x) = 0."""

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-10.0, 10.0]] * dim)
        optimizers = torch.ones(1, dim)
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            dim=-1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
        )
        return part1 + part2 + part3


class Griewank(ObjectiveFn):
    """Griewank function. Global minimum at x = 0 with f(x) = 0."""

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-600.0, 600.0]] * dim)
        optimizers = torch.zeros(1, dim)
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        part1 = torch.sum(X ** 2 / 4000.0, dim=-1)
        i = torch.arange(1, self.dim + 1, device=X.device, dtype=X.dtype)
        part2 = -torch.prod(torch.cos(X / torch.sqrt(i)), dim=-1)
        return part1 + part2 + 1.0


class Beale(ObjectiveFn):
    """Beale function (2D). Global minimum at (3, 0.5) with f(x) = 0."""

    def __init__(self, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-4.5, 4.5], [-4.5, 4.5]])
        optimizers = torch.tensor([[3.0, 0.5]])
        super().__init__(dim=2, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return part1 + part2 + part3


class Branin(ObjectiveFn):
    """Branin function (2D). Global minimum ~0.397887."""

    def __init__(self, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-5.0, 10.0], [0.0, 15.0]])
        optimizers = torch.tensor([[-math.pi, 12.275], [math.pi, 2.275], [9.42478, 2.475]])
        super().__init__(dim=2, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.397887, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi ** 2) * X[..., 0] ** 2
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1 ** 2 + t2 + 10


class Sphere(ObjectiveFn):
    """Sphere function. Global minimum at x = 0 with f(x) = 0.

    f(x) = sum(x_i^2)

    The simplest convex test function, useful as a baseline.
    """

    def __init__(self, dim: int = 2, noise_std: Optional[float] = None,
                 negate: bool = False, bounds: Optional[torch.Tensor] = None):
        if bounds is None:
            bounds = torch.tensor([[-5.12, 5.12]] * dim)
        optimizers = torch.zeros(1, dim)
        super().__init__(dim=dim, bounds=bounds, optimizers=optimizers,
                         optimal_value=0.0, noise_std=noise_std, negate=negate)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sum(X ** 2, dim=-1)
