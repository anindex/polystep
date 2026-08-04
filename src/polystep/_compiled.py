"""Compiled function variants for hot paths in the Sinkhorn Step solver."""

import warnings
from typing import Callable, Optional, Tuple

import torch


# "reduce-overhead" uses CUDA graphs, which conflict when chaining compiled functions; "default" avoids them.
DEFAULT_MODE = "default"


def try_compile(
    fn: Callable,
    *,
    fullgraph: bool = True,
    mode: str = DEFAULT_MODE,
    name: Optional[str] = None,
) -> Callable:
    """Wrap a function with torch.compile, falling back to eager on failure."""
    label = name if name is not None else getattr(fn, "__name__", repr(fn))
    try:
        compiled = torch.compile(fn, fullgraph=fullgraph, mode=mode)
    except Exception as e:
        warnings.warn(
            f"torch.compile failed for '{label}': {e}. Falling back to eager mode.",
            stacklevel=2,
        )
        return fn

    state = {"compiled": compiled}

    def guarded(*args, **kwargs):
        target = state["compiled"]
        if target is None:
            return fn(*args, **kwargs)
        try:
            return target(*args, **kwargs)
        except Exception as e:
            # Don't swallow OOM: an eager retry would just OOM again.
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            state["compiled"] = None
            warnings.warn(
                f"torch.compile backend failed on first call to '{label}': {e}. Falling back to eager mode.",
                stacklevel=2,
            )
            return fn(*args, **kwargs)

    return guarded


def _sinkhorn_iteration(
    f: torch.Tensor,
    g: torch.Tensor,
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    omega: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One log-domain Sinkhorn iteration on duals ``f`` and ``g``; ``omega`` over-relaxes."""
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
    """Rotate template vertices onto each particle and translate by ``step_radius``."""
    rotated = torch.einsum("bji, ni -> bnj", rot_mats, polytope_vertices)
    step_points = rotated * step_radius + origin.unsqueeze(1)
    return step_points, rotated


def _barycentric_projection(
    transport_matrix: torch.Tensor,
    X_vertices: torch.Tensor,
) -> torch.Tensor:
    """Weighted average of vertices by the transport plan."""
    # Divide by the realized row sum, not the target marginal a: correct for unconverged Sinkhorn.
    row_sum = transport_matrix.sum(dim=-1, keepdim=True)
    # Uniform on a zero or NaN row: the centred polytope's barycentre is the particle.
    # Dividing by a clamped zero would give zero weights and move it to the origin.
    weights = torch.where(
        row_sum > 0,
        transport_matrix / row_sum.clamp(min=1e-12),
        transport_matrix.new_full((), 1.0 / transport_matrix.shape[-1]),
    )
    # Cast to the vertex dtype so the matmul works in mixed precision.
    X_new = torch.einsum("bkd,bk->bd", X_vertices, weights.to(X_vertices.dtype))
    return X_new


def _compute_probe_points(
    origin: torch.Tensor,
    directions: torch.Tensor,
    scales: torch.Tensor,
    probe_radius: float,
) -> torch.Tensor:
    """Probe points at fixed scale intervals along directions."""
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax solve + vertex-free barycentric projection."""
    # Recenter to min(C)=0: softmax is shift-invariant, but this makes 'mean' scaling shift-invariant and bounds -C/epsilon (which otherwise overflows to NaN).
    C = cost_matrix - cost_matrix.amin()

    # Inline for compile safety, avoids cross-module string dispatch.
    if scale_cost_mean:
        s = torch.clamp(C.abs().mean(), min=1e-10)
        C = C / s

    # Shift per row before dividing: torch.softmax subtracts the row max only after
    # the division, so at tiny epsilon a row above the global minimum returns NaN.
    # Pinned outside autocast like the eager softmax_plan.
    with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
        W = torch.softmax(-(C - C.amin(dim=-1, keepdim=True)) / epsilon, dim=-1)  # (P, V)

    # Transport matrix: row sums equal the source marginal a.
    transport = W * a.unsqueeze(-1)  # (P, V)

    # Vertex-free centroid: O(P*dim), not O(P*V*dim). Cast W to the geometry dtype for mixed precision.
    w_centroid = W.to(polytope_verts.dtype) @ polytope_verts  # (P, dim)
    rot_centroid = torch.einsum("bij,bj->bi", rot_mats, w_centroid)  # (P, dim)

    X_new = X + step_radius * rot_centroid  # (P, dim)

    return X_new, transport


def _tensorize_scalars(fn: Callable, eager: Callable, positions: Tuple[int, ...], ref_arg: int) -> Callable:
    """Wrap ``fn`` so the float args at ``positions`` arrive as 0-d tensors."""
    if fn is eager:
        return fn

    def wrapper(*args, **kwargs):
        args = list(args)
        ref = args[ref_arg]
        for i in positions:
            if i < len(args) and not isinstance(args[i], torch.Tensor):
                args[i] = torch.as_tensor(args[i], dtype=ref.dtype, device=ref.device)
        return fn(*args, **kwargs)

    return wrapper


class CompiledFunctions:
    """Registry of compiled (or eager) pure tensor functions."""

    def __init__(self, compile: bool = True) -> None:
        self.compile = compile and torch.cuda.is_available()
        if self.compile:
            # Dynamo specializes on Python floats, so a scheduled radius would recompile every step; 0-d tensors stay dynamic.
            self.sinkhorn_iter = _tensorize_scalars(
                try_compile(_sinkhorn_iteration, name="sinkhorn_iteration"), _sinkhorn_iteration, (5, 6), 2
            )
            self.rotate_and_translate = _tensorize_scalars(
                try_compile(_rotate_and_translate, name="rotate_and_translate"), _rotate_and_translate, (3,), 0
            )
            self.barycentric_projection = try_compile(_barycentric_projection, name="barycentric_projection")
            self.compute_probe_points = _tensorize_scalars(
                try_compile(_compute_probe_points, name="compute_probe_points"), _compute_probe_points, (3,), 0
            )
            self.fused_softmax_project = _tensorize_scalars(
                try_compile(_fused_softmax_project, name="fused_softmax_project"), _fused_softmax_project, (1, 5), 0
            )
        else:
            self.sinkhorn_iter = _sinkhorn_iteration
            self.rotate_and_translate = _rotate_and_translate
            self.barycentric_projection = _barycentric_projection
            self.compute_probe_points = _compute_probe_points
            self.fused_softmax_project = _fused_softmax_project
