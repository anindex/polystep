"""Solver abstraction layer for polystep.

Provides pluggable solver implementations for the polytope step optimizer:
- ``Solver``: Protocol defining the solver interface
- ``SolverResult``: Base result dataclass
- ``SinkhornSolver``: Entropic OT solver (full-rank and low-rank)
- ``SinkhornResult``: Extended result with dual potentials
"""
from .base import Solver, SolverResult
from .greedy import MinCostGreedySolver, TopKMeanSolver
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
    "TemperedSoftmaxSolver",
]
