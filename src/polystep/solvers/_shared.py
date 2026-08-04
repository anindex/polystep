"""Shared preamble for the OT / weighting solvers: promote half precision to FP32, sanitize costs, align marginals and warm-start duals."""

import contextlib
import math
import warnings
from typing import Optional

import torch

from .base import SolverResult


def validate_positive(value: float, name: str, context: str = "") -> None:
    """Raise ValueError unless ``value > 0``."""
    if not value > 0:
        msg = f"{name} must be > 0, got {value}."
        if context:
            msg += " " + context
        raise ValueError(msg)


def solver_health(transport: torch.Tensor, displacement: torch.Tensor, step_radius: float):
    """Return ``(ess, rho)``, the two OT health numbers, as 0-dim tensors."""
    w = transport / transport.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ess = (1.0 / (w * w).sum(dim=1).clamp(min=1e-12)).mean() / transport.shape[1]
    rho = (displacement.norm(dim=1) / max(step_radius, 1e-12)).mean()
    return ess, rho


def decomposition_dtype(dtype: torch.dtype) -> torch.dtype:
    """FP32 for half precision: no QR/SVD backend has a half-precision kernel."""
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


@contextlib.contextmanager
def single_thread_cpu(device: torch.device):
    """Run a CPU decomposition on one thread, restoring the count after."""
    prev = torch.get_num_threads() if device.type == "cpu" else 1
    if prev != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if prev != 1:
            torch.set_num_threads(prev)


def thin_qr(matrix: torch.Tensor):
    """Reduced QR, pinned to one CPU thread."""
    with single_thread_cpu(matrix.device):
        return torch.linalg.qr(matrix, mode="reduced")


def loss_buffer_dtype(particle_dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for probe losses: FP32 for half precision, keeps FP64."""
    return particle_dtype if particle_dtype == torch.float64 else torch.float32


def sanitize_cost(cost_matrix: torch.Tensor) -> torch.Tensor:
    """Promote half precision to FP32 and replace non-finite costs on-device.

    Every non-finite cost is invalid, ``-inf`` included, and becomes ``2*max|finite| + 1``,
    below every finite one. ``ask_tell.tell`` and ``baselines.core`` read ``-inf`` the same way.
    """
    if cost_matrix.dtype in (torch.bfloat16, torch.float16):
        cost_matrix = cost_matrix.to(torch.float32)
    if cost_matrix.numel() == 0:
        return cost_matrix
    finite = torch.isfinite(cost_matrix)
    max_finite = torch.where(finite, cost_matrix, cost_matrix.new_zeros(())).abs().amax()
    # Clamp: near dtype max, 2*max_finite + 1 overflows and the +inf survives.
    penalty = (max_finite * 2.0 + 1.0).clamp(max=torch.finfo(cost_matrix.dtype).max)
    return torch.where(finite, cost_matrix, penalty)


def exp_plan(f: torch.Tensor, g: torch.Tensor, C: torch.Tensor, eps: float) -> torch.Tensor:
    """``P_ij = exp((f_i + g_j - C_ij) / eps)`` that cannot overflow to ``inf``.

    Exact wherever the plain exp was finite. The clamp binds only after the duals
    diverge, where inf/inf in the barycentric projection would give NaN parameters.
    """
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
    # -1 for headroom: exp(log(finfo.max)) rounds back up to inf.
    return torch.exp(log_P.clamp(max=math.log(torch.finfo(log_P.dtype).max) - 1.0))


def recenter_cost(cost_matrix: torch.Tensor):
    """Subtract the per-matrix minimum so min(C)=0. The plan is shift-invariant, and this keeps the log-sum-exp in FP32 range."""
    if cost_matrix.numel() == 0:
        return cost_matrix, cost_matrix.new_zeros(())
    shift = cost_matrix.amin()
    return cost_matrix - shift, shift


# Below this ratio of temperature to max|C|, -C/temperature saturates.
TINY_TEMPERATURE_RATIO = 1e-6


def validate_cost_shape(cost_matrix: torch.Tensor, solver: str) -> tuple[int, int]:
    """Return ``(P, V)``, rejecting an empty cost matrix."""
    if cost_matrix.dim() != 2:
        raise ValueError(f"{solver} expects a 2D cost matrix (P, V), got shape {tuple(cost_matrix.shape)}.")
    rows, cols = cost_matrix.shape
    if rows == 0 or cols == 0:
        raise ValueError(
            f"{solver} received an empty cost matrix (shape {tuple(cost_matrix.shape)}); "
            "at least one particle and one vertex are required."
        )
    return rows, cols


def warn_tiny_temperature(solver_obj, temperature: float, C: torch.Tensor, solver: str, name: str) -> None:
    """Warn once per new sharpest temperature when ``-C/temperature`` will saturate."""
    seen = getattr(solver_obj, "_min_temperature_checked", None)
    if seen is not None and temperature >= seen:
        return
    solver_obj._min_temperature_checked = temperature
    cost_max = C.detach().abs().max().item() if C.numel() > 0 else 0.0
    if cost_max > 0 and temperature < TINY_TEMPERATURE_RATIO * cost_max:
        warnings.warn(
            f"{solver} {name}={temperature:.2e} is very small relative to the cost-matrix "
            f"scale (max |C|={cost_max:.2e}); -C/{name} saturates and the plan will not "
            f"satisfy its marginals. Rescale the cost or raise {name}.",
            stacklevel=3,
        )


def align_marginal(
    a: Optional[torch.Tensor],
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "a",
) -> torch.Tensor:
    """Return a length-``n`` marginal on ``(device, dtype)``; ``None`` becomes uniform 1/n."""
    if a is None:
        return torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    a = a.to(device=device, dtype=dtype)
    if a.shape != (n,):
        raise ValueError(f"marginal {name} must have shape ({n},), got {tuple(a.shape)}.")
    if not torch.isfinite(a).all():
        raise ValueError(f"marginal {name} contains non-finite entries (NaN/Inf).")
    if (a < 0).any():
        raise ValueError(f"marginal {name} has negative entries; a transport marginal must be nonnegative.")
    if not a.sum() > 0:
        raise ValueError(f"marginal {name} has nonpositive total mass; it must sum to a positive value.")
    return a


def align_dual(
    init: Optional[torch.Tensor],
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str = "init",
) -> Optional[torch.Tensor]:
    """Move a warm-start dual onto ``(device, dtype)``, or ``None`` on a shape mismatch."""
    if init is None:
        return None
    if init.shape != (n,):
        warnings.warn(
            f"warm-start {name} shape mismatch: expected ({n},), got {tuple(init.shape)}. Falling back to zeros.",
            stacklevel=2,
        )
        return None
    return init.to(device=device, dtype=dtype).clone()


def prepare_cost(cost_matrix, a, scale_cost, solver: str):
    """Shared solver preamble: validate, sanitize, align, recenter, then scale. Recenter before scaling so mean/max scaling is shift-invariant."""
    from ..costs import resolve_cost_scale

    rows, _ = validate_cost_shape(cost_matrix, solver)
    cost_matrix = sanitize_cost(cost_matrix)
    a = align_marginal(a, rows, cost_matrix.device, cost_matrix.dtype)
    C, cost_shift = recenter_cost(cost_matrix)
    cost_scale = resolve_cost_scale(C, scale_cost)
    return C / cost_scale, a, cost_shift, cost_scale


def solve_softmax(solver_obj, cost_matrix, a, temperature: float, scale_cost, solver: str, name: str):
    """One-sided softmax solve, shared by ``SoftmaxSolver`` and ``TemperedSoftmaxSolver``."""
    validate_positive(temperature, name, f"{name} is the temperature in softmax(-C/{name}).")
    C, a, cost_shift, cost_scale = prepare_cost(cost_matrix, a, scale_cost, solver)
    warn_tiny_temperature(solver_obj, temperature, C, solver, name)
    transport, ent_cost = softmax_plan(C, a, temperature, cost_shift, cost_scale)
    return SolverResult(
        matrix=transport,
        cost=ent_cost,
        f=None,
        g=None,
        converged=True,
        n_iters=1,
        ent_reg_cost=ent_cost,
    )


def softmax_plan(C, a, temperature: float, cost_shift, cost_scale):
    """Return ``(transport, ent_cost)`` for the one-sided softmax plan. Shifts per row before dividing so softmax does not return NaN at tiny temperatures."""
    with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
        W = torch.softmax(-(C - C.amin(dim=-1, keepdim=True)) / temperature, dim=-1)
        transport = W * a.to(W.dtype).unsqueeze(-1)
        # Undo both frame changes so the reported cost is <C_raw, transport>.
        ent_cost = ((C * transport).sum() * cost_scale + cost_shift * a.to(W.dtype).sum()).item()
    return transport, ent_cost
