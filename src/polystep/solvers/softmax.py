"""SoftmaxSolver: direct softmax weighting for subspace modes."""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._shared import solve_softmax
from .base import SolverResult


@dataclass
class SoftmaxSolver:
    """Direct softmax weighting: transport weights via ``softmax(-C/epsilon)`` scaled to row sum ``a``."""

    epsilon: float = 0.1
    compile: bool = False

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SolverResult:
        """Compute softmax weights from the cost matrix. Only row sums are enforced; ``b`` is ignored."""
        # One-sided: a caller passing a non-uniform b probably wanted SinkhornSolver.
        if b is not None:
            uniform = torch.full_like(b, 1.0 / cost_matrix.shape[1])
            if not torch.allclose(b.to(device=uniform.device, dtype=uniform.dtype), uniform, atol=1e-6):
                warnings.warn(
                    "SoftmaxSolver ignores the target marginal `b`: it only "
                    "enforces row sums equal to the source marginal `a`. "
                    "Pass solver='sinkhorn' for two-sided OT.",
                    stacklevel=2,
                )

        return solve_softmax(self, cost_matrix, a, self.epsilon, scale_cost, "SoftmaxSolver", "epsilon")
