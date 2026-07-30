"""Shared entry-point preparation for the OT / weighting solvers.

Every log-sum-exp / softmax solver needs the same preamble before it can run:
promote half precision to FP32 (BF16's 7 mantissa bits collapse the row-max
trick once the cost spread exceeds ~15 nats), replace non-finite costs with a
finite penalty, default and device/dtype-align the marginals, and coerce
warm-start duals onto the cost tensor. Centralizing it keeps the variants from
drifting apart (e.g. one solver gaining FP32 promotion while another silently
NaNs on the same input).

Design note - no per-step host syncs: ``sanitize_cost`` is branch-free (no
``.item()`` / ``.all()`` in a Python ``if``), and ``align_marginal`` /
``align_dual`` only move tensors (``.to`` is a no-op when already aligned).
Value checks that would force a device->host sync are intentionally omitted so
the hot path (a fresh solve every optimizer step) stays GPU-resident.
"""

import math
import warnings
from typing import Optional

import torch

from .base import SolverResult


def validate_positive(value: float, name: str, context: str = "") -> None:
    """Raise ValueError unless ``value > 0`` (a plain Python-float check).

    Solvers store their temperature (``epsilon`` / ``tau``) as a mutable
    attribute that schedules overwrite per step, so this is re-checked inside
    ``solve()`` rather than only at construction.
    """
    if not value > 0:
        msg = f"{name} must be > 0, got {value}."
        if context:
            msg += " " + context
        raise ValueError(msg)


def solver_health(transport: torch.Tensor, displacement: torch.Tensor, step_radius: float):
    """``(ess, rho)`` as 0-dim tensors, the two OT health numbers.

    ``ess`` is the effective sample size over the transport weights divided by the
    vertex count, so it is 1.0 when the weights are uniform and the barycenter is a
    plain mean. ``rho = ||Delta|| / step_radius`` is how far the barycenter moved as a
    fraction of the step radius. Left unreduced so the caller can batch the device sync.
    """
    w = transport / transport.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ess = (1.0 / (w * w).sum(dim=1).clamp(min=1e-12)).mean() / transport.shape[1]
    rho = (displacement.norm(dim=1) / max(step_radius, 1e-12)).mean()
    return ess, rho


def decomposition_dtype(dtype: torch.dtype) -> torch.dtype:
    """FP32 for half precision: no QR or SVD backend has a half-precision kernel.

    CPU LAPACK raises ``not implemented for 'Half'`` and cuSOLVER has no geqrf/gesvd
    for either half type, so every decomposition site upcasts and casts back.
    """
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


def thin_qr(matrix: torch.Tensor):
    """Reduced QR, pinned to one thread on CPU.

    LAPACK spreads a tall-thin QR over every core and the synchronization dominates at
    subspace shapes, costing orders of magnitude over the single-threaded run. CUDA
    tensors skip the pinning. The thread count is restored before returning.
    """
    if matrix.device.type != "cpu":
        return torch.linalg.qr(matrix, mode="reduced")
    prev = torch.get_num_threads()
    if prev == 1:
        return torch.linalg.qr(matrix, mode="reduced")
    torch.set_num_threads(1)
    try:
        return torch.linalg.qr(matrix, mode="reduced")
    finally:
        torch.set_num_threads(prev)


def loss_buffer_dtype(particle_dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for probe losses.

    Half precision is promoted to FP32 for log-sum-exp stability, but FP64 is kept:
    a double objective whose probe costs differ below FP32 resolution collapses to a
    constant cost matrix, and the particle stops moving.
    """
    return particle_dtype if particle_dtype == torch.float64 else torch.float32


def sanitize_cost(cost_matrix: torch.Tensor) -> torch.Tensor:
    """Promote half precision to FP32 and replace non-finite costs, on-device.

    A hard-constraint ``+inf`` or an upstream NaN becomes ``2 * max|finite| + 1`` so the
    masked vertex ranks below every finite one without sending ``-C/eps`` to ``-inf`` and
    NaN-ing the whole row. ``-inf`` is the opposite case: for a minimization it is the
    best possible value, so it maps to the finite minimum instead of the penalty.
    Branch-free: no host sync on the finite path. How strongly the masked vertex is
    suppressed depends on epsilon and the cost scale, which are not visible here.

    The penalty is relative to the finite scale, not an absolute floor. An absolute
    floor of 1e6 survives into the ``'mean'`` and ``'max_cost'`` reductions in
    :func:`~polystep.costs.scale_cost_matrix`, which run after this, and divides every
    real cost difference down to ~1e-6 of the scale, flattening the plan to uniform for
    that step. With no finite entry the scale collapses to ``max_finite = 0``, so
    ``+inf`` and NaN map to 1 and ``-inf`` to 0: an all-``+inf`` matrix comes out
    constant (the uniform plan), a mixed one still ranks ``-inf`` best.
    """
    if cost_matrix.dtype in (torch.bfloat16, torch.float16):
        cost_matrix = cost_matrix.to(torch.float32)
    if cost_matrix.numel() == 0:
        return cost_matrix
    finite = torch.isfinite(cost_matrix)
    max_finite = torch.where(finite, cost_matrix, cost_matrix.new_zeros(())).abs().amax()
    # Clamp: near dtype max, 2 * max_finite + 1 overflows and the +inf we came to
    # remove survives.
    penalty = (max_finite * 2.0 + 1.0).clamp(max=torch.finfo(cost_matrix.dtype).max)
    # Fill non-finite slots with a value no smaller than any finite entry so the
    # reduction returns the finite minimum.
    min_finite = torch.where(finite, cost_matrix, max_finite).amin()
    replacement = torch.where(cost_matrix == float("-inf"), min_finite, penalty)
    return torch.where(finite, cost_matrix, replacement)


def exp_plan(f: torch.Tensor, g: torch.Tensor, C: torch.Tensor, eps: float) -> torch.Tensor:
    """``P_ij = exp((f_i + g_j - C_ij) / eps)`` that cannot overflow to ``inf``.

    Exact wherever the plain exp was finite. The clamp binds only after the duals
    diverge, where inf/inf in the barycentric projection would give NaN parameters.
    """
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / eps
    # -1 for headroom: exp(log(finfo.max)) rounds back up to inf.
    return torch.exp(log_P.clamp(max=math.log(torch.finfo(log_P.dtype).max) - 1.0))


def recenter_cost(cost_matrix: torch.Tensor):
    """Subtract the per-matrix minimum so ``|C|`` stays bounded.

    The entropic-OT plan is invariant to a constant shift of the cost:
    ``softmax(-C/eps)`` is shift-invariant, and a shift only translates the
    dual potentials, leaving the plan ``P`` unchanged. Keeping ``min(C)=0``
    preserves FP32 precision in the log-sum-exp and the ``exp((f+g-C)/eps)``
    reconstruction when ``|C|`` is much larger than ``eps``, and removes the
    ``+inf`` logit that would NaN a softmax at tiny ``eps``.

    Returns ``(shifted_cost, shift)`` with ``shift = C.min()``. Add
    ``shift * a.sum()`` back to any reported ``<C, P>`` or dual value (plan
    mass ``sum(P)`` equals ``a.sum()``). One reduction, no host sync.
    """
    if cost_matrix.numel() == 0:
        return cost_matrix, cost_matrix.new_zeros(())
    shift = cost_matrix.amin()
    return cost_matrix - shift, shift


# Below this ratio of temperature to max|C|, -C/temperature saturates before any
# row-max subtraction can help. Empirical for FP32 and BF16.
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
    """Warn once per new sharpest temperature when ``-C/temperature`` will saturate.

    Only fires on a decrease, so the reduction (which host-syncs) stays off the hot
    path when the schedule is flat or rising.
    """
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
    """Return a length-``n`` marginal on ``(device, dtype)``.

    ``None`` -> uniform ``1/n``. A provided marginal is moved onto the cost
    tensor (a no-op when already aligned), shape-checked, and value-checked
    (finite, nonnegative, positive total mass) so a malformed marginal fails
    loudly instead of being silently clamped to an infeasible plan before the
    ``log``. The value checks host-sync, so they run *only* on the user-supplied
    path: the integrated optimizer passes ``a=None`` for its uniform marginal
    (see :func:`solver.PolyStep.init_state`), keeping the per-step solve
    sync-free.
    """
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
    """Move a warm-start dual onto ``(device, dtype)``, or ``None`` on mismatch.

    Returns a fresh (cloned) tensor the caller may mutate in place, or ``None``
    when the shape does not match (caller falls back to a zero init, after a
    warning). The ``.to`` is a no-op when the dual is already aligned.
    """
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
    """Shared solver preamble: validate, sanitize, align, recenter, scale.

    Recenter before scaling. The plan is invariant to a constant cost shift but
    'mean'/'max_cost' are not, so scaling first would tie the temperature to the
    arbitrary absolute loss level. ``min(C) = 0`` also keeps the log-sum-exp in
    FP32 range and removes the ``+inf`` logit that NaNs a softmax at tiny epsilon.

    Returns ``(C, a, cost_shift, cost_scale)``.
    """
    from ..costs import resolve_cost_scale

    rows, _ = validate_cost_shape(cost_matrix, solver)
    cost_matrix = sanitize_cost(cost_matrix)
    a = align_marginal(a, rows, cost_matrix.device, cost_matrix.dtype)
    C, cost_shift = recenter_cost(cost_matrix)
    cost_scale = resolve_cost_scale(C, scale_cost)
    return C / cost_scale, a, cost_shift, cost_scale


def solve_softmax(solver_obj, cost_matrix, a, temperature: float, scale_cost, solver: str, name: str):
    """One-sided softmax solve, shared by ``SoftmaxSolver`` and ``TemperedSoftmaxSolver``.

    ``name`` is the attribute the temperature came from, so the validation and the
    warning name the parameter the caller actually set.
    """
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
    """``(transport, ent_cost)`` for the one-sided softmax plan.

    Shifts per row before dividing: ``torch.softmax`` subtracts the row max only
    after the division, so at a small temperature a row sitting above the global
    minimum sends every logit to -inf and comes back NaN. Pinned outside autocast
    so an outer mixed-precision region cannot downcast the logits.
    """
    with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
        W = torch.softmax(-(C - C.amin(dim=-1, keepdim=True)) / temperature, dim=-1)
        transport = W * a.to(W.dtype).unsqueeze(-1)
        # Undo both frame changes so the reported cost is <C_raw, transport>.
        ent_cost = ((C * transport).sum() * cost_scale + cost_shift * a.to(W.dtype).sum()).item()
    return transport, ent_cost
