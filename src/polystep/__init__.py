"""polystep: PyTorch PolyStep Optimizer."""
__version__ = "0.2.2"

__all__ = [
    # Core solver
    "Solver", "SolverResult",
    "SinkhornSolver", "SinkhornResult",
    "SoftmaxSolver", "SoftmaxResult",
    # Geometry
    "get_orthoplex_vertices", "get_simplex_vertices", "get_cube_vertices",
    "get_sampled_polytope_vertices", "get_random_rotation_matrices",
    "POLYTOPE_MAP", "POLYTOPE_NUM_VERTICES_MAP",
    # Cost & epsilon
    "compute_cost_matrix", "scale_cost_matrix", "LinearEpsilon", "CosineEpsilon",
    # Solver
    "PolyStep", "SolverState",
    # Transform
    "ParamEntry", "ParamLayout", "get_device", "create_generator",
    # NN cost evaluation
    "NNCostEvaluator", "compute_nn_cost_matrix", "auto_detect_chunk_size",
    # Compilation
    "CompiledFunctions", "try_compile",
    # Subspace
    "FactorSpec", "LowRankSubspace", "LinearSubspace", "ProjectionSpec",
    "AdaptiveSubspace", "CMAAdaptiveSubspace",
    # CMA
    "compute_cma_hyperparameters", "update_step_size_csa", "compute_ot_weights",
    # Blockwise
    "BlockConfig", "create_per_layer_blocks", "create_grouped_blocks",
    # Dynamics
    "apply_momentum", "update_adaptive_radius", "compute_momentum_coefficient",
    # Optimizer
    "PolyStepOptimizer", "RankSchedule",
    # High-level API
    "train", "TrainConfig", "TrainCallback",
    "LoggingCallback", "EarlyStoppingCallback", "get_diagnostics",
    # Objectives
    "ObjectiveFn", "Ackley", "Rosenbrock", "Rastrigin", "StyblinskiTang",
    "Levy", "Griewank", "Beale", "Branin", "Sphere",
    # Layers
    "VmapSafeMultiHeadAttention", "VmapSafeLSTMCell", "VmapSafeLSTM",
    # Projection
    "SparseRandomProjection",
    # Hybrid subspace
    "HybridSubspace", "LayerProjectionSpec",
]

from .solvers import Solver, SolverResult, SinkhornSolver, SinkhornResult, SoftmaxSolver, SoftmaxResult

from .geometry import (
    get_orthoplex_vertices,
    get_simplex_vertices,
    get_cube_vertices,
    get_sampled_polytope_vertices,
    get_random_rotation_matrices,
    POLYTOPE_MAP,
    POLYTOPE_NUM_VERTICES_MAP,
)

from .costs import compute_cost_matrix, scale_cost_matrix
from .epsilon import LinearEpsilon, CosineEpsilon

from .solver import PolyStep, SolverState

from .transform import ParamEntry, ParamLayout, get_device, create_generator

from .cost_nn import NNCostEvaluator, compute_nn_cost_matrix, auto_detect_chunk_size

from ._compiled import CompiledFunctions, try_compile

from .subspace import FactorSpec, LowRankSubspace, LinearSubspace, ProjectionSpec

from .adaptive_subspace import AdaptiveSubspace

from .cma_subspace import CMAAdaptiveSubspace
from .cma import (
    compute_cma_hyperparameters,
    update_step_size_csa,
    compute_ot_weights,
)

from .blockwise import BlockConfig, create_per_layer_blocks, create_grouped_blocks

from .dynamics import apply_momentum, update_adaptive_radius, compute_momentum_coefficient

from .optimizer import PolyStepOptimizer, RankSchedule

from .api import (
    train, TrainConfig, TrainCallback,
    LoggingCallback, EarlyStoppingCallback, get_diagnostics,
)

# Objectives
from .objectives import (
    ObjectiveFn,
    Ackley, Rosenbrock, Rastrigin, StyblinskiTang,
    Levy, Griewank, Beale, Branin, Sphere,
)

# Vmap-safe layers
from .layers import VmapSafeMultiHeadAttention, VmapSafeLSTMCell, VmapSafeLSTM

# Sparse projection for large-scale models
from .projection import SparseRandomProjection

# Hybrid subspace
from .hybrid_subspace import HybridSubspace, LayerProjectionSpec
