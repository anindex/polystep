"""Solver abstraction layer for polystep.

Pluggable solver implementations for the polytope step optimizer:

- ``SolverResult``: shared result dataclass.
- ``SinkhornSolver``: full-rank log-domain entropic OT solver.
- ``SinkhornResult``: result with dual potentials.
- ``SoftmaxSolver``: one-sided softmax weighting.
- ``KLSoftmaxSolver``: KL-penalized interpolation between softmax and
  Sinkhorn.
- ``TemperedSoftmaxSolver``: softmax with a fixed temperature.
- ``MinCostGreedySolver`` / ``TopKMeanSolver``: simple non-OT baselines.
"""

from .base import SolverResult
from .greedy import MinCostGreedySolver, TopKMeanSolver
from .kl_softmax import KLSoftmaxSolver
from .sinkhorn import SinkhornSolver, SinkhornResult
from .softmax import SoftmaxSolver
from .tempered_softmax import TemperedSoftmaxSolver

__all__ = [
    "SolverResult",
    "MinCostGreedySolver",
    "TopKMeanSolver",
    "SinkhornSolver",
    "SinkhornResult",
    "SoftmaxSolver",
    "KLSoftmaxSolver",
    "TemperedSoftmaxSolver",
]
