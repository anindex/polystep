"""Shared result type for the OT / weighting solvers.

Defines the
``SolverResult`` dataclass that all solvers return. Uses structural typing
(Protocol) so any class with a matching ``.solve()`` signature qualifies
without explicit inheritance.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SolverResult:
    """Base result from any solver.

    Attributes:
        matrix: Transport plan / weight matrix of shape (P, V).
        cost: Objective value, same convention as ``ent_reg_cost``.
        f: First dual potential (Sinkhorn) or None (softmax).
        g: Second dual potential (Sinkhorn) or None (softmax).
        converged: Whether the solver converged within tolerance. In a
            fixed-iteration mode (``threshold <= 0``) no marginal residual is
            measured, so this reports only that the duals came back finite.
        n_iters: Number of iterations actually run.
        ent_reg_cost: Objective in the caller's cost frame. Solver-specific: comparable
            across steps of one solver, not across solvers. ``SinkhornSolver`` reports the
            entropic dual ``<f, a> + <g, b> - eps * sum_ij P_ij``, equal at convergence to
            the negative-entropy primal, not to the ``<C, P> + eps * KL(P || a x b)`` form
            other libraries report. Every other solver reports plain ``<C, P>``, so
            ``KLSoftmaxSolver`` omits its ``lam * KL`` and is not the objective it minimizes.
    """

    matrix: torch.Tensor
    cost: float
    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    converged: bool = True
    n_iters: int = 1
    ent_reg_cost: float = 0.0
