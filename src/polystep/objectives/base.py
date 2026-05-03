"""Abstract base class for optimization objective functions."""
import abc
from typing import Optional, Tuple

import torch


class ObjectiveFn(abc.ABC):
    """Base class for optimization objectives.

    Attributes:
        dim: Dimensionality of the problem.
        bounds: Search space bounds of shape (dim, 2) as [(min, max), ...].
        optimizers: Known global optimizer locations of shape (num_opts, dim).
        optimal_value: Known global optimum value.
        noise_std: Standard deviation of additive Gaussian noise.
        negate: If True, negate the output (maximization -> minimization).
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
            X: Input array of shape (..., dim).

        Returns:
            Cost array of shape (...).
        """
        pass

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the final cost, applying negation if configured.

        Args:
            X: Input points of shape (..., dim).

        Returns:
            Cost values of shape (...).
        """
        cost = self.evaluate(X)
        if self.negate:
            return -cost
        return cost
