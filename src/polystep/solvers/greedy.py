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

from ..costs import scale_cost_matrix
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
        P, V = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        if a is None:
            a = torch.ones(P, device=device, dtype=dtype) / P

        C = scale_cost_matrix(cost_matrix.clone(), scale_cost)

        # Greedy: each particle picks the single lowest-cost vertex
        min_indices = C.argmin(dim=-1)  # (P,)
        transport = torch.zeros(P, V, device=device, dtype=dtype)
        transport.scatter_(1, min_indices.unsqueeze(1), a.unsqueeze(1))

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
        P, V = cost_matrix.shape
        device = cost_matrix.device
        dtype = cost_matrix.dtype

        if a is None:
            a = torch.ones(P, device=device, dtype=dtype) / P

        C = scale_cost_matrix(cost_matrix.clone(), scale_cost)

        # Graceful fallback when fewer vertices than k
        k_eff = min(self.k, V)

        # Top-k: find k lowest-cost vertex indices per particle
        _, topk_indices = C.topk(k_eff, dim=-1, largest=False)  # (P, k_eff)

        # Uniform mass over top-k vertices
        transport = torch.zeros(P, V, device=device, dtype=dtype)
        mass_per_vertex = a.unsqueeze(1) / k_eff  # (P, 1)
        transport.scatter_(1, topk_indices, mass_per_vertex.expand_as(topk_indices))

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
