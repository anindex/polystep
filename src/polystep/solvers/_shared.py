"""Shared preamble for the OT / weighting solvers: promote half precision to FP32, sanitize costs, align marginals and warm-start duals."""

import contextlib
import math
import warnings
from typing import Optional

import torch

from .._compiled import normalize_rows
from ..costs import resolve_cost_scale


def validate_positive(value: float, name: str, context: str = "") -> float:
    """Return a positive finite value, or raise ValueError."""
    if not (math.isfinite(value) and value > 0):
        msg = f"{name} must be > 0 and finite, got {value}."
        if context:
            msg += " " + context
        raise ValueError(msg)
    return value


def solver_health(transport: torch.Tensor, displacement: torch.Tensor, step_radius: float):
    """Return ``(ess, rho)``, the two OT health numbers, as 0-dim tensors."""
    w = normalize_rows(transport)
    ess = (1.0 / (w * w).sum(dim=1)).mean() / transport.shape[1]
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


QR_PIN_MAX_WORK = 10**8


def thin_qr(matrix: torch.Tensor):
    """Reduced QR, pinned to one CPU thread below ``QR_PIN_MAX_WORK`` of rows * cols**2 work."""
    if matrix.numel() * matrix.shape[-1] >= QR_PIN_MAX_WORK:
        return torch.linalg.qr(matrix, mode="reduced")
    with single_thread_cpu(matrix.device):
        return torch.linalg.qr(matrix, mode="reduced")


def autocast_off(device_type: str):
    """Autocast disabled on ``device_type``, or a no-op where it has none."""
    if torch.amp.is_autocast_available(device_type):
        return torch.amp.autocast(device_type, enabled=False)
    return contextlib.nullcontext()


def orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Q of the positive-diagonal QR: CholQR2 from 8 columns, 32x tallness and 2**18 entries at full FP32 unless ill-conditioned, else Householder."""
    rows, cols = matrix.shape
    try:
        full_fp32 = torch.get_float32_matmul_precision() == "highest"
    except RuntimeError:
        full_fp32 = False
    if cols >= 8 and rows >= 32 * cols and matrix.numel() >= 2**18 and full_fp32:
        pin = single_thread_cpu(matrix.device) if matrix.numel() * cols < QR_PIN_MAX_WORK else contextlib.nullcontext()
        with pin, autocast_off(matrix.device.type):
            L, info = torch.linalg.cholesky_ex(matrix.mT @ matrix)
            Q = matrix @ torch.linalg.inv_ex(L)[0].mT
            gram = Q.mT @ Q
            Q = Q @ torch.linalg.inv_ex(torch.linalg.cholesky_ex(gram)[0])[0].mT
            gram_error = torch.linalg.matrix_norm(gram - torch.eye(cols, dtype=gram.dtype, device=gram.device))
        if bool((info == 0) & (gram_error < 0.5)):
            return Q
    Q, R = thin_qr(matrix)
    d = torch.sign(torch.diagonal(R))
    return Q * torch.where(d == 0, 1.0, d)


def loss_buffer_dtype(particle_dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for probe losses: FP32 for half precision, keeps FP64."""
    return particle_dtype if particle_dtype == torch.float64 else torch.float32


def sanitize_cost(cost_matrix: torch.Tensor) -> torch.Tensor:
    """Promote half precision to FP32 and integer costs to the default float, then replace non-finite costs on-device.

    Every non-finite cost is invalid, ``-inf`` included, and becomes ``2*max|finite| + 1``,
    clamped to the dtype limit. ``ask_tell.tell`` and ``baselines.core`` read ``-inf`` the same way.
    """
    if not cost_matrix.is_floating_point():
        cost_matrix = cost_matrix.to(torch.get_default_dtype())
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


def precision_floor(dtype: torch.dtype, eps: float, dual_max: float, cost_max: float) -> float:
    """Smallest relative marginal error ``dtype`` resolves on the costs that carry mass."""
    return 2 * torch.finfo(dtype).eps * ((dual_max + min(dual_max, cost_max)) / eps + 1)


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
            stacklevel=3,
        )
        return None
    return init.to(device=device, dtype=dtype).clone()


def recenter_and_scale(cost_matrix, raw, scale_cost):
    """Recenter a sanitized cost, then scale it.

    Data-dependent modes read only the entries finite in ``raw``, then penalize the rest
    at twice the largest scaled finite cost plus one, so a narrow finite spread cannot
    inflate the penalty.
    """
    finite = torch.isfinite(raw) if isinstance(scale_cost, str) else None
    prescale = (
        (cost_matrix.abs().amax() / (torch.finfo(cost_matrix.dtype).max / 4)).clamp(min=1) if finite is not None else 1
    )
    C, cost_shift = recenter_cost(cost_matrix / prescale if finite is not None else cost_matrix)
    cost_scale = resolve_cost_scale(C, scale_cost, finite)
    C = C / cost_scale
    if finite is not None:
        C = torch.where(finite, C, C.where(finite, 0).amax() * 2 + 1)
        cost_shift = cost_shift * prescale
        cost_scale = cost_scale.double() * prescale
    return C, cost_shift, cost_scale


def prepare_cost(cost_matrix, a, scale_cost, solver: str):
    """Shared solver preamble: validate, sanitize, align, recenter, then scale. Recenter before scaling so mean/max scaling is shift-invariant."""
    rows, _ = validate_cost_shape(cost_matrix, solver)
    safe = sanitize_cost(cost_matrix)
    a = align_marginal(a, rows, safe.device, safe.dtype)
    C, cost_shift, cost_scale = recenter_and_scale(safe, cost_matrix, scale_cost)
    return C, a, cost_shift, cost_scale
