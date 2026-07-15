"""Compiled function variants for hot paths in the Sinkhorn Step solver.

These are the compiled hot paths. ``torch.compile`` is applied to individual
loop body functions (not the outer convergence loop) to avoid graph breaks
from Python control flow (iteration counting, convergence checks, early
stopping). Each function is a pure tensor operation with no Python side
effects, making it safe for ``fullgraph=True`` compilation.

Compilation strategy:
    - Per-function compilation using torch.compile with 'default' mode.
      The 'default' mode compiles with Inductor for operator fusion while
      avoiding CUDA graph tensor ownership conflicts between chained functions.
    - Each function is compiled independently so one failure does not block others.
    - Compilation is only attempted when compile=True AND CUDA is available;
      on CPU, all functions use eager mode (torch.compile overhead exceeds benefit).
    - If torch.compile raises an exception for a specific function, that function
      silently falls back to its eager implementation with a one-time UserWarning.
    - compile=False disables all compilation attempts, using raw eager functions.

Usage:
    cf = CompiledFunctions(compile=True)   # compiled on CUDA, eager on CPU
    cf = CompiledFunctions(compile=False)  # always eager

    # Pre-compile all functions to exclude JIT warmup from benchmarks:
    cf.warm_start(dim=10, batch=5, device=torch.device('cuda'))

Provides pure tensor functions suitable for torch.compile, plus a try_compile
helper that gracefully falls back to eager mode on failure. The CompiledFunctions
registry holds either compiled or eager versions depending on the compile flag.
"""

import warnings
from typing import Callable, Optional, Tuple

import torch


# Note: "reduce-overhead" uses CUDA graphs which cause tensor ownership conflicts
# when chaining multiple compiled functions (rotate->probe->barycentric) within one step.
# "default" mode still compiles with Inductor but avoids CUDA graph issues.
DEFAULT_MODE = "default"


def try_compile(
    fn: Callable,
    *,
    fullgraph: bool = True,
    mode: str = DEFAULT_MODE,
    name: Optional[str] = None,
) -> Callable:
    """Attempt to compile a function with torch.compile, falling back to eager.

    Args:
        fn: Pure tensor function to compile.
        fullgraph: Whether to require a single graph (no graph breaks).
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        name: Label for warning messages. Defaults to fn.__name__.

    Returns:
        Compiled function, or the original fn if compilation fails.
    """
    label = name if name is not None else getattr(fn, "__name__", repr(fn))
    try:
        return torch.compile(fn, fullgraph=fullgraph, mode=mode)
    except Exception as e:
        warnings.warn(
            f"torch.compile failed for '{label}': {e}. Falling back to eager mode.",
            stacklevel=2,
        )
        return fn


# ---------------------------------------------------------------------------
# Pure compiled functions (no .item(), no list ops, no shape-dependent branches)
# ---------------------------------------------------------------------------


def _sinkhorn_iteration(
    f: torch.Tensor,
    g: torch.Tensor,
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    omega: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single log-domain Sinkhorn iteration with optional overrelaxation (pure tensor ops).

    When omega=1.0 (default), this is standard Sinkhorn. When omega>1.0,
    overrelaxation accelerates convergence from O(1/t) to O(1/t^2).

    Args:
        f: First dual potential of shape (n,).
        g: Second dual potential of shape (m,).
        log_K: Log kernel matrix (-C / eps) of shape (n, m).
        log_a: Log source marginal of shape (n,).
        log_b: Log target marginal of shape (m,).
        eps: Entropic regularization strength.
        omega: Overrelaxation parameter in [0.5, 1.95]. Default 1.0 (no overrelaxation).

    Returns:
        Tuple of updated (f, g) dual potentials.
    """
    f_target = eps * (log_a - torch.logsumexp(log_K + g.unsqueeze(0) / eps, dim=1))
    f_new = (1 - omega) * f + omega * f_target

    g_target = eps * (log_b - torch.logsumexp(log_K + f_new.unsqueeze(1) / eps, dim=0))
    g_new = (1 - omega) * g + omega * g_target
    return f_new, g_new


def _rotate_and_translate(
    rot_mats: torch.Tensor,
    polytope_vertices: torch.Tensor,
    origin: torch.Tensor,
    step_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate polytope vertices and translate to particle positions (pure tensor ops).

    Args:
        rot_mats: Rotation matrices of shape (batch, dim, dim).
        polytope_vertices: Template vertices of shape (num_verts, dim).
        origin: Particle positions of shape (batch, dim).
        step_radius: Step distance multiplier.

    Returns:
        Tuple of (step_points, rotated_vertices):
            - step_points: (batch, num_verts, dim) vertex positions after rotation + translation.
            - rotated_vertices: (batch, num_verts, dim) rotated vertices before translation.
    """
    rotated = torch.einsum("bji, ni -> bnj", rot_mats, polytope_vertices)
    step_points = rotated * step_radius + origin.unsqueeze(1)
    return step_points, rotated


def _barycentric_projection(
    transport_matrix: torch.Tensor,
    a: torch.Tensor,
    X_vertices: torch.Tensor,
) -> torch.Tensor:
    """Barycentric projection: weighted average of vertices by transport plan (pure tensor ops).

    Args:
        transport_matrix: Transport plan of shape (batch, num_vertices).
        a: Source marginal weights of shape (batch,).
        X_vertices: Vertex positions of shape (batch, num_vertices, dim).

    Returns:
        Updated particle positions of shape (batch, dim).
    """
    # Divide by the realized row sum (the OT barycenter), not the target marginal
    # a. Equal for softmax rows; correct for unconverged Sinkhorn (a kept for API).
    row_sum = transport_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    weights = transport_matrix / row_sum
    # Cast to vertex dtype so the matmul works in mixed precision (no fp32/bf16 promote).
    X_new = torch.einsum("bkd,bk->bd", X_vertices, weights.to(X_vertices.dtype))
    return X_new


def _compute_probe_points(
    origin: torch.Tensor,
    directions: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
) -> torch.Tensor:
    """Generate probe points at fixed scale intervals along directions (pure tensor ops).

    Args:
        origin: Particle positions of shape (batch, dim).
        directions: Direction vectors of shape (batch, num_points, dim).
        scales: Scalar coefficients of shape (num_probe,).
        probe_radius: Maximum distance multiplier.

    Returns:
        Probe points of shape (batch, num_points, num_probe, dim).
    """
    # origin: (batch, 1, 1, dim)
    origin_exp = origin[:, None, None, :]
    # directions: (batch, num_points, 1, dim)
    directions_exp = directions[:, :, None, :]
    # scales: (1, 1, num_probe, 1)
    scales_exp = scales[None, None, :, None]

    return origin_exp + (directions_exp * probe_radius) * scales_exp


def _fused_softmax_project(
    cost_matrix: torch.Tensor,
    epsilon: float,
    a: torch.Tensor,
    polytope_verts: torch.Tensor,
    rot_mats: torch.Tensor,
    step_radius: float,
    X: torch.Tensor,
    scale_cost_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused softmax weighting + vertex-free barycentric projection (pure tensor ops).

    Combines softmax OT solve and barycentric projection into a single compiled
    pipeline. Eliminates the intermediate O(P*V*dim) rotated vertex tensor by
    computing the weighted centroid in template space then rotating, reducing
    the projection step to O(P*dim).

    Args:
        cost_matrix: Cost matrix C of shape (P, V).
        epsilon: Temperature parameter (entropic regularization strength).
        a: Source marginal of shape (P,).
        polytope_verts: Template polytope vertices of shape (V, dim) -- NOT rotated.
        rot_mats: Per-particle rotation matrices of shape (P, dim, dim).
        step_radius: Step distance multiplier.
        X: Current particle positions of shape (P, dim).
        scale_cost_mean: If True, normalize cost matrix by its mean absolute value.

    Returns:
        Tuple of (X_new, transport, ent_cost):
            - X_new: Updated particle positions of shape (P, dim).
            - transport: Transport matrix of shape (P, V) with row sums equal to a.
            - ent_cost: Entropic cost as a 0-dim tensor (kept as tensor for compile safety).
    """
    # Cost scaling (inline for compile safety -- avoids cross-module string dispatch)
    if scale_cost_mean:
        s = torch.clamp(cost_matrix.abs().mean(), min=1e-10)
        C = cost_matrix / s
    else:
        C = cost_matrix

    # Softmax weights -- PyTorch's softmax subtracts row-max internally for stability
    W = torch.softmax(-C / epsilon, dim=-1)  # (P, V)

    # Transport matrix: row sums equal source marginal a
    transport = W * a.unsqueeze(-1)  # (P, V)

    # Vertex-free centroid: O(P*dim), not O(P*V*dim) rotated vertices. Cast W to the
    # geometry dtype so the matmul works in mixed precision (W is fp32, geometry bf16).
    w_centroid = W.to(polytope_verts.dtype) @ polytope_verts  # (P, dim)
    rot_centroid = torch.einsum("bij,bj->bi", rot_mats, w_centroid)  # (P, dim)

    # Barycentric projection
    X_new = X + step_radius * rot_centroid  # (P, dim)

    # Entropic cost -- kept as tensor to avoid graph break in compiled path
    ent_cost = (C * transport).sum()

    return X_new, transport, ent_cost


# ---------------------------------------------------------------------------
# Registry: holds compiled or eager versions of each function
# ---------------------------------------------------------------------------


class CompiledFunctions:
    """Registry of compiled (or eager) pure tensor functions.

    When compile=True and CUDA is available, wraps each function with
    torch.compile via try_compile. Otherwise uses the eager versions directly.

    Attributes:
        sinkhorn_iter: Compiled or eager _sinkhorn_iteration.
        rotate_and_translate: Compiled or eager _rotate_and_translate.
        barycentric_projection: Compiled or eager _barycentric_projection.
        compute_probe_points: Compiled or eager _compute_probe_points.
        fused_softmax_project: Compiled or eager _fused_softmax_project.
    """

    DEFAULT_MODE = DEFAULT_MODE

    def __init__(self, compile: bool = True) -> None:
        self.compile = compile and torch.cuda.is_available()
        if self.compile:
            self.sinkhorn_iter = try_compile(_sinkhorn_iteration, name="sinkhorn_iteration")
            self.rotate_and_translate = try_compile(_rotate_and_translate, name="rotate_and_translate")
            self.barycentric_projection = try_compile(_barycentric_projection, name="barycentric_projection")
            self.compute_probe_points = try_compile(_compute_probe_points, name="compute_probe_points")
            self.fused_softmax_project = try_compile(_fused_softmax_project, name="fused_softmax_project")
        else:
            self.sinkhorn_iter = _sinkhorn_iteration
            self.rotate_and_translate = _rotate_and_translate
            self.barycentric_projection = _barycentric_projection
            self.compute_probe_points = _compute_probe_points
            self.fused_softmax_project = _fused_softmax_project

    def warm_start(self, dim: int = 10, batch: int = 5, device: Optional[torch.device] = None) -> None:
        """Pre-compile all hot paths by running dummy inputs.

        Triggers JIT compilation warmup so that the first real call does not
        include compilation overhead. Call before benchmarking to get accurate
        timing measurements.

        No-ops when compilation is disabled (``compile=False``).

        Args:
            dim: Dimensionality for dummy tensors. For best results, match
                the actual ``particle_dim`` of your problem.
            batch: Batch size for dummy tensors. For best results, match
                the actual ``num_particles`` of your problem.
            device: Device for dummy tensors. Defaults to CPU if None.
        """
        if not self.compile:
            return  # No JIT compilation to warm up

        if device is None:
            device = torch.device("cpu")
        num_verts = 2 * dim

        with torch.inference_mode():
            # Warm sinkhorn_iter
            f = torch.zeros(batch, device=device)
            g = torch.zeros(batch, device=device)
            log_K = torch.zeros(batch, batch, device=device)
            log_a = torch.zeros(batch, device=device)
            log_b = torch.zeros(batch, device=device)
            self.sinkhorn_iter(f, g, log_K, log_a, log_b, 0.1)

            # Warm rotate_and_translate
            rot_mats = torch.eye(dim, device=device).unsqueeze(0).expand(batch, -1, -1).contiguous()
            polytope_verts = torch.randn(num_verts, dim, device=device)
            origin = torch.zeros(batch, dim, device=device)
            self.rotate_and_translate(rot_mats, polytope_verts, origin, 1.0)

            # Warm barycentric_projection
            transport = torch.ones(batch, num_verts, device=device) / num_verts
            a = torch.ones(batch, device=device) / batch
            vertices = torch.randn(batch, num_verts, dim, device=device)
            self.barycentric_projection(transport, a, vertices)

            # Warm compute_probe_points
            directions = torch.randn(batch, num_verts, dim, device=device)
            scales = torch.linspace(0.2, 0.8, 3, device=device)
            self.compute_probe_points(origin, directions, scales, 1.0)

            # Warm fused_softmax_project
            C_warm = torch.randn(batch, num_verts, device=device)
            a_warm = torch.ones(batch, device=device) / batch
            verts_warm = torch.randn(num_verts, dim, device=device)
            rot_warm = torch.eye(dim, device=device).unsqueeze(0).expand(batch, -1, -1).contiguous()
            X_warm = torch.randn(batch, dim, device=device)
            self.fused_softmax_project(C_warm, 0.1, a_warm, verts_warm, rot_warm, 1.0, X_warm)
