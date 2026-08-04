"""Shared ``SolverResult`` dataclass returned by all solvers."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SolverResult:
    """Result from any solver.

    ``ent_reg_cost`` is solver-specific: comparable across steps of one solver, not across solvers.
    """

    matrix: torch.Tensor
    cost: float
    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    converged: bool = True
    n_iters: int = 1
    ent_reg_cost: float = 0.0
