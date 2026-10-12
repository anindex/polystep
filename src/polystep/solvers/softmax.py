"""SoftmaxSolver: direct softmax weighting for subspace modes."""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._shared import prepare_cost, validate_positive, warn_tiny_temperature
from .base import SolverResult


def _softmax_plan(C, a, temperature: float, cost_shift, cost_scale):
    """Return ``(transport, ent_cost)`` for the one-sided softmax plan. Shifts per row before dividing so softmax does not return NaN at tiny temperatures."""
    with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
        W = torch.softmax(-(C - C.amin(dim=-1, keepdim=True)) / temperature, dim=-1)
        transport = W * a.to(W.dtype).unsqueeze(-1)
        # Undo both frame changes so the reported cost is <C_raw, transport>.
        ent_cost = (C * transport).sum() * cost_scale + cost_shift * a.to(W.dtype).sum()
    return transport, ent_cost


@dataclass
class SoftmaxSolver:
    """Direct softmax weighting: transport weights via ``softmax(-C/epsilon)`` scaled to row sum ``a``."""

    epsilon: float = 0.1

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

        validate_positive(self.epsilon, "epsilon", "epsilon is the temperature in softmax(-C/epsilon).")
        C, a, cost_shift, cost_scale = prepare_cost(cost_matrix, a, scale_cost, "SoftmaxSolver")
        warn_tiny_temperature(self, self.epsilon, C, "SoftmaxSolver", "epsilon")
        transport, ent_cost = _softmax_plan(C, a, self.epsilon, cost_shift, cost_scale)
        return SolverResult(matrix=transport, _ent_reg=ent_cost)
