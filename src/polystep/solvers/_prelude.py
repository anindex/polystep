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

import warnings
from typing import Optional

import torch


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


def sanitize_cost(cost_matrix: torch.Tensor) -> torch.Tensor:
    """Promote half precision to FP32 and replace non-finite costs, on-device.

    Non-finite entries (a hard-constraint ``+inf`` or an upstream NaN) are
    replaced with ``2 * max|finite| + 1`` (floored at ``1e6``) so a masked
    vertex still gets near-zero weight without sending ``-C/eps`` to ``-inf``
    and NaN-ing the whole row. Branch-free: no host sync on the finite path.
    """
    if cost_matrix.dtype in (torch.bfloat16, torch.float16):
        cost_matrix = cost_matrix.to(torch.float32)
    if cost_matrix.numel() == 0:
        return cost_matrix
    finite = torch.isfinite(cost_matrix)
    max_finite = torch.where(finite, cost_matrix, cost_matrix.new_zeros(())).abs().amax()
    penalty = torch.clamp(max_finite * 2.0 + 1.0, min=1e6)
    return torch.where(finite, cost_matrix, penalty)


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
