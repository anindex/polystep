"""Non-iterative greedy solvers for ablation studies."""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._shared import align_marginal, sanitize_cost, validate_cost_shape
from .base import SolverResult


@dataclass
class MinCostGreedySolver:
    """Greedy argmin: each particle moves to its single lowest-cost vertex."""

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
        """Compute the greedy argmin assignment; returns a one-nonzero-per-row transport matrix."""
        # Sanitize first: argmin over a NaN would hand all mass to a masked vertex.
        P, V = validate_cost_shape(cost_matrix, "MinCostGreedySolver")
        C_raw = sanitize_cost(cost_matrix)
        device, dtype = C_raw.device, C_raw.dtype
        a = align_marginal(a, P, device, dtype, "a")

        min_indices = C_raw.argmin(dim=-1)  # (P,)
        transport = torch.zeros(P, V, device=device, dtype=dtype)
        transport.scatter_(1, min_indices.unsqueeze(1), a.unsqueeze(1))
        # A constant row (all-infeasible) spreads uniformly: the barycentre of the centred polytope is the particle itself.
        informative = (C_raw.amax(dim=-1) > C_raw.amin(dim=-1)).unsqueeze(1)
        transport = torch.where(informative, transport, a.unsqueeze(1).expand(P, V) / V)

        # scale_cost is unused: argmin is invariant to a positive divisor.
        ent_cost = (C_raw * transport).sum().item()

        return SolverResult(
            matrix=transport,
            cost=ent_cost,
            f=None,
            g=None,
            converged=True,
            n_iters=1,
            ent_reg_cost=ent_cost,
        )


@dataclass
class TopKMeanSolver:
    """Top-K mean: each particle moves to the uniform average of its k lowest-cost vertices."""

    epsilon: float = 0.1
    compile: bool = False
    k: int = 3

    def __post_init__(self):
        if self.k < 1:
            raise ValueError(
                f"TopKMeanSolver.k must be >= 1, got {self.k}. k=0 yields an "
                f"all-zero transport plan, violating the row-mass contract "
                f"(rows must sum to the source marginal a) that the barycentric "
                f"projection assumes."
            )

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SolverResult:
        """Compute the top-k uniform assignment; returns a k-nonzeros-per-row transport matrix."""
        P, V = validate_cost_shape(cost_matrix, "TopKMeanSolver")
        C_raw = sanitize_cost(cost_matrix)
        device, dtype = C_raw.device, C_raw.dtype
        a = align_marginal(a, P, device, dtype, "a")

        k_eff = min(self.k, V)
        _, topk_indices = C_raw.topk(k_eff, dim=-1, largest=False)  # (P, k_eff)

        # topk does not honour the sanitize mask, so drop infeasible picks and spread mass over what is left; the row still sums to a.
        feasible = torch.isfinite(cost_matrix)
        keep = feasible.gather(1, topk_indices)
        keep = keep | ~keep.any(dim=1, keepdim=True)

        transport = torch.zeros(P, V, device=device, dtype=dtype)
        mass = a.unsqueeze(1) / keep.sum(dim=1, keepdim=True) * keep
        transport.scatter_(1, topk_indices, mass.to(dtype))
        informative = (C_raw.amax(dim=-1) > C_raw.amin(dim=-1)).unsqueeze(1)
        transport = torch.where(informative, transport, a.unsqueeze(1).expand(P, V) / V)

        # scale_cost is unused: topk is invariant to a positive divisor.
        ent_cost = (C_raw * transport).sum().item()

        return SolverResult(
            matrix=transport,
            cost=ent_cost,
            f=None,
            g=None,
            converged=True,
            n_iters=1,
            ent_reg_cost=ent_cost,
        )
