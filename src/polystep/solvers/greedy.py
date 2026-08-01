"""Greedy solvers: deterministic update rules for ablation studies.

Provides two non-iterative solvers that bypass entropic OT entirely:

- ``MinCostGreedySolver``: Each particle moves to its single lowest-cost vertex.
- ``TopKMeanSolver``: Each particle moves to the uniform average of its k
  lowest-cost vertices.

Both produce transport matrices compatible with the standard barycentric
projection (Eq. 7) and conform to the ``Solver`` protocol.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._shared import align_marginal, sanitize_cost, validate_cost_shape
from .base import SolverResult


@dataclass
class MinCostGreedySolver:
    """Greedy argmin solver: each particle moves to its lowest-cost vertex.

    For each row i of the cost matrix, assigns all mass ``a[i]`` to the
    single vertex with minimum cost:  ``T[i, argmin_v C[i,v]] = a[i]``,
    zero elsewhere.

    Attributes:
        epsilon: Accepted for API compatibility; unused by greedy assignment.
        compile: Accepted for API compatibility; unused.
    """

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
        """Compute greedy argmin assignment from cost matrix.

        Args:
            cost_matrix: Cost matrix C of shape (P, V).
            a: Source marginal of shape (P,). Defaults to uniform 1/P.
            b: Target marginal (accepted but ignored).
            init_f: Warm-start dual potential (accepted but ignored).
            init_g: Warm-start dual potential (accepted but ignored).
            scale_cost: Cost scaling: 'mean', 'max_cost', or float divisor.

        Returns:
            SolverResult with sparse transport matrix (one non-zero per row).
        """
        # sanitize first: argmin over a NaN is undefined and would hand all the mass
        # to a masked vertex. No recentering, the selection is shift-invariant.
        P, V = validate_cost_shape(cost_matrix, "MinCostGreedySolver")
        C_raw = sanitize_cost(cost_matrix)
        device, dtype = C_raw.device, C_raw.dtype
        a = align_marginal(a, P, device, dtype, "a")

        # Greedy: each particle picks the single lowest-cost vertex
        min_indices = C_raw.argmin(dim=-1)  # (P,)
        transport = torch.zeros(P, V, device=device, dtype=dtype)
        transport.scatter_(1, min_indices.unsqueeze(1), a.unsqueeze(1))
        # An all-infeasible row is constant after sanitize, so argmin picks vertex 0 and
        # the particle takes a full step on nothing. Spread it uniformly: the polytope is
        # centred, so the barycentre is the particle itself.
        informative = (C_raw.amax(dim=-1) > C_raw.amin(dim=-1)).unsqueeze(1)
        transport = torch.where(informative, transport, a.unsqueeze(1).expand(P, V) / V)

        # scale_cost is accepted but unused: argmin is invariant to a positive
        # divisor, and the reported cost is in the caller's frame either way.
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
    """Top-K mean solver: each particle moves to the uniform average of its k best vertices.

    For each row i, finds the k vertices with lowest cost and assigns
    equal mass ``a[i] / k`` to each: ``T[i,v] = a[i] / k`` for the top-k
    vertices, zero elsewhere.

    When V < k, gracefully falls back to using all V vertices.

    Attributes:
        epsilon: Accepted for API compatibility; unused.
        compile: Accepted for API compatibility; unused.
        k: Number of lowest-cost vertices to average over. Default 3.
    """

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
        """Compute top-k uniform assignment from cost matrix.

        Args:
            cost_matrix: Cost matrix C of shape (P, V).
            a: Source marginal of shape (P,). Defaults to uniform 1/P.
            b: Target marginal (accepted but ignored).
            init_f: Warm-start dual potential (accepted but ignored).
            init_g: Warm-start dual potential (accepted but ignored).
            scale_cost: Cost scaling: 'mean', 'max_cost', or float divisor.

        Returns:
            SolverResult with transport matrix (k non-zeros per row).
        """
        P, V = validate_cost_shape(cost_matrix, "TopKMeanSolver")
        C_raw = sanitize_cost(cost_matrix)
        device, dtype = C_raw.device, C_raw.dtype
        a = align_marginal(a, P, device, dtype, "a")

        k_eff = min(self.k, V)
        _, topk_indices = C_raw.topk(k_eff, dim=-1, largest=False)  # (P, k_eff)

        # sanitize_cost only guarantees a masked vertex ranks below every finite one,
        # which argmin honours and topk does not: a row with fewer than k finite
        # entries would hand real mass to forbidden directions. Drop those picks and
        # spread the row's mass over what is left, so the row still sums to a. A row
        # with nothing feasible keeps its picks rather than losing its mass.
        # From the caller's matrix: sanitize maps +inf to a finite penalty, so this is the
        # only place the mask survives. The screen keeps keep_v >= k_eff regardless.
        feasible = ~(torch.isnan(cost_matrix) | (cost_matrix == float("inf")))
        keep = feasible.gather(1, topk_indices)
        keep = keep | ~keep.any(dim=1, keepdim=True)

        transport = torch.zeros(P, V, device=device, dtype=dtype)
        mass = a.unsqueeze(1) / keep.sum(dim=1, keepdim=True) * keep
        transport.scatter_(1, topk_indices, mass.to(dtype))

        # scale_cost is accepted but unused: topk is invariant to a positive
        # divisor, and the reported cost is in the caller's frame either way.
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
