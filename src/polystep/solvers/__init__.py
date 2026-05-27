"""Solver abstraction layer for polystep.

Pluggable solver implementations for the polytope step optimizer:

- ``Solver``: protocol defining the solver interface.
- ``SolverResult``: shared result dataclass.
- ``SinkhornSolver``: entropic OT solver (full- and low-rank).
- ``SinkhornResult``: result with dual potentials.
- ``SoftmaxSolver`` / ``SoftmaxResult``: one-sided softmax weighting.
- ``KLSoftmaxSolver``: KL-penalized interpolation between softmax and
  Sinkhorn.
- ``TemperedSoftmaxSolver``: softmax with a fixed temperature.
- ``MinCostGreedySolver`` / ``TopKMeanSolver``: simple non-OT baselines.
"""
from .base import Solver, SolverResult
from .greedy import MinCostGreedySolver, TopKMeanSolver
from .kl_softmax import KLSoftmaxSolver
from .sinkhorn import SinkhornSolver, SinkhornResult
from .softmax import SoftmaxSolver, SoftmaxResult
from .tempered_softmax import TemperedSoftmaxSolver

__all__ = [
    "Solver",
    "SolverResult",
    "MinCostGreedySolver",
    "TopKMeanSolver",
    "SinkhornSolver",
    "SinkhornResult",
    "SoftmaxSolver",
    "SoftmaxResult",
    "KLSoftmaxSolver",
    "TemperedSoftmaxSolver",
]
