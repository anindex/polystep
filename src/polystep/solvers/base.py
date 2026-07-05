"""Solver abstraction layer: Protocol and base result type.

Defines the ``Solver`` protocol for pluggable OT/weighting solvers and the
``SolverResult`` dataclass that all solvers return. Uses structural typing
(Protocol) so any class with a matching ``.solve()`` signature qualifies
without explicit inheritance.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class SolverResult:
    """Base result from any solver.

    Attributes:
        matrix: Transport plan / weight matrix of shape (P, V).
        cost: Regularized cost scalar.
        f: First dual potential (Sinkhorn) or None (softmax).
        g: Second dual potential (Sinkhorn) or None (softmax).
        converged: Whether the solver converged within tolerance.
        n_iters: Number of iterations actually run.
        ent_reg_cost: Entropic regularized cost = <f, a> + <g, b>.
    """

    matrix: torch.Tensor
    cost: float
    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    converged: bool = True
    n_iters: int = 1
    ent_reg_cost: float = 0.0


class Solver:
    """Protocol for pluggable solvers.

    Any class that implements ``solve()`` with the following signature
    qualifies as a ``Solver`` via structural typing (duck typing). This
    is intentionally NOT a ``typing.Protocol`` subclass to avoid runtime
    metaclass conflicts with ``@dataclass``; conformance is checked by
    the type checker via structural subtyping.

    Attributes:
        epsilon: Entropic regularization / temperature parameter.
    """

    epsilon: float

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SolverResult:
        """Solve the weighting / transport problem.

        Args:
            cost_matrix: Cost matrix C of shape (n, m).
            a: Source marginal of shape (n,). Defaults to uniform 1/n.
            b: Target marginal of shape (m,). Defaults to uniform 1/m.
            init_f: Warm-start first dual potential of shape (n,).
            init_g: Warm-start second dual potential of shape (m,).
            scale_cost: Cost scaling: 'mean', 'max_cost', or float divisor.

        Returns:
            SolverResult with transport plan and cost.
        """
        raise NotImplementedError
