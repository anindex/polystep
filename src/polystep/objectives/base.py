"""Abstract base class for optimization objective functions."""

import abc
from typing import Optional

import torch


class ObjectiveFn(abc.ABC):
    """Base class for optimization objectives."""

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
        # Negate flips optimal_value too, or regret never reaches 0.
        self.optimal_value = -optimal_value if (negate and optimal_value is not None) else optimal_value
        self.noise_std = noise_std
        self.negate = negate

    @abc.abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Raw objective value: ``(..., dim)`` in, ``(...)`` out."""
        pass

    def __call__(
        self,
        X: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply noise and negation, returning the final cost."""
        cost = self.evaluate(X)
        if self.noise_std is not None and self.noise_std > 0.0:
            # normal_ needs the generator on the tensor's device, so draw there and move.
            draw_device = generator.device if generator is not None else cost.device
            noise = torch.empty(cost.shape, dtype=cost.dtype, device=draw_device).normal_(
                mean=0.0,
                std=float(self.noise_std),
                generator=generator,
            )
            cost = cost + noise.to(cost.device)
        if self.negate:
            return -cost
        return cost
