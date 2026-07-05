"""TemperedSoftmaxSolver: softmax weighting with a fixed temperature.

Like ``SoftmaxSolver`` but uses a separate temperature parameter ``tau``
instead of the optimizer's epsilon schedule. This decouples the softmax
sharpness from the entropic regularization so the ablation can sweep
tau independently.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ..costs import scale_cost_matrix
from ._prelude import align_marginal, sanitize_cost, validate_positive
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
        validate_positive(
            self.tau,
            "tau",
            "tau is the temperature in softmax(-C/tau).",
        )

        P, V = cost_matrix.shape
        cost_matrix = sanitize_cost(cost_matrix)  # FP32 promote + finite costs
        device, dtype = cost_matrix.device, cost_matrix.dtype
        a = align_marginal(a, P, device, dtype)

        C = scale_cost_matrix(cost_matrix, scale_cost)

        # Use fixed tau (NOT self.epsilon) for the softmax. Pin inside an
        # autocast-disabled FP32 region so an outer mixed-precision context
        # can't downcast -C/tau before softmax subtracts the row max.
        with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            W = torch.softmax(-C / self.tau, dim=-1)
            transport = W * a.unsqueeze(-1)
            ent_cost = (C * transport).sum().item()

        return SolverResult(
            matrix=transport,
            cost=ent_cost,
            f=None,
            g=None,
            converged=True,
            n_iters=1,
            ent_reg_cost=ent_cost,
        )
