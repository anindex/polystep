"""Abstract base class for optimization objective functions."""
import abc
from typing import Optional

import torch


class ObjectiveFn(abc.ABC):
    """Base class for optimization objectives.

    Attributes:
        dim: Dimensionality of the problem.
        bounds: Search space bounds of shape ``(dim, 2)`` as
            ``[(min, max), ...]``.
        optimizers: Known global optimizer locations of shape
            ``(num_opts, dim)``.
        optimal_value: Known global optimum value.
        noise_std: Standard deviation of additive Gaussian observation noise.
            ``None`` or ``0`` disables noise.
        negate: If ``True``, negate the output (turn a maximization problem
            into a minimization problem).
    """

    def __init__(
        self,
        dim: int,
        bounds: Optional[torch.Tensor] = None,
        optimizers: Optional[torch.Tensor] = None,
        optimal_value: Optional[float] = None,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ):
        self.dim = dim
        self.bounds = bounds
        self.optimizers = optimizers
        self.optimal_value = optimal_value
        self.noise_std = noise_std
        self.negate = negate

    @abc.abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the raw objective value.

        Args:
            X: Input array of shape ``(..., dim)``.

        Returns:
            Cost array of shape ``(...)``.
        """
        pass

    def __call__(
        self,
        X: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Compute the final cost, applying noise and negation if configured.

        Args:
            X: Input points of shape ``(..., dim)``.
            generator: Optional ``torch.Generator`` for reproducible noise.

        Returns:
            Cost values of shape ``(...)``.
        """
        cost = self.evaluate(X)
        if self.noise_std is not None and self.noise_std > 0.0:
            noise = torch.empty_like(cost).normal_(
                mean=0.0, std=float(self.noise_std), generator=generator,
            )
            cost = cost + noise
        if self.negate:
            return -cost
        return cost
