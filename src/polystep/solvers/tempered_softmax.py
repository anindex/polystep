"""TemperedSoftmaxSolver: softmax weighting with a fixed temperature.

Like ``SoftmaxSolver`` but uses a separate temperature parameter ``tau``
instead of the optimizer's epsilon schedule. This decouples the softmax
sharpness from the entropic regularization so the ablation can sweep
tau independently.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._shared import solve_softmax
from .base import SolverResult


@dataclass
class TemperedSoftmaxSolver:
    """Softmax solver with a fixed temperature independent of epsilon.

    Computes transport weights via ``softmax(-C / tau)`` where ``tau`` is
    set once at construction and NOT overridden by the optimizer's per-step
    epsilon. The ``epsilon`` attribute is accepted for API compatibility
    but ignored in ``solve()``.

    Attributes:
        epsilon: Accepted for API compatibility; overridden per-step by the
            optimizer but NOT used in the softmax computation.
        tau: Fixed temperature for the softmax. Lower tau = sharper weights.
        compile: Accepted for API compatibility; unused.
    """

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
        """Compute softmax weights using fixed temperature tau.

        Args:
            cost_matrix: Cost matrix C of shape (P, V).
            a: Source marginal of shape (P,). Defaults to uniform 1/P.
            b: Target marginal (accepted but ignored).
            init_f: Warm-start dual potential (accepted but ignored).
            init_g: Warm-start dual potential (accepted but ignored).
            scale_cost: Cost scaling: 'mean', 'max_cost', or float divisor.

        Returns:
            SolverResult with transport matrix, cost, and metadata.

        Raises:
            ValueError: If tau <= 0.
        """
        # tau, not self.epsilon: that is what this solver exists for.
        return solve_softmax(self, cost_matrix, a, self.tau, scale_cost, "TemperedSoftmaxSolver", "tau")
