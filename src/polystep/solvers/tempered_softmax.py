"""Softmax weighting with a fixed temperature, independent of the optimizer's epsilon."""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._shared import solve_softmax
from .base import SolverResult


@dataclass
class TemperedSoftmaxSolver:
    """Softmax solver with a fixed temperature ``tau``; ``epsilon`` is accepted but ignored."""

    epsilon: float = 0.1
    tau: float = 1.0
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
        """Compute softmax weights using the fixed temperature ``tau``."""
        # Use tau, not self.epsilon.
        return solve_softmax(self, cost_matrix, a, self.tau, scale_cost, "TemperedSoftmaxSolver", "tau")
