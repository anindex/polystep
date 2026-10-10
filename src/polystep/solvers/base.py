"""Shared ``SolverResult`` dataclass returned by all solvers."""

import functools
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SolverResult:
    """Result from any solver.

    ``ent_reg_cost`` is solver-specific: comparable across steps of one solver, not across solvers.
    Like ``SinkhornResult``, it is read from the device on first access, so a solve never waits for it.
    """

    matrix: torch.Tensor
    f: Optional[torch.Tensor] = None
    g: Optional[torch.Tensor] = None
    converged: bool = True
    n_iters: int = 1
    _ent_reg: Optional[torch.Tensor] = None

    @functools.cached_property
    def ent_reg_cost(self) -> float:
        """Transport cost <C, P> in the caller's frame, or 0.0 when the solve did not report one."""
        return 0.0 if self._ent_reg is None else self._ent_reg.item()
