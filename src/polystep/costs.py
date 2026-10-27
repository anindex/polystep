"""Cost matrix computation for PolyStep."""

import math
from typing import Callable, Optional, Union

import torch


def compute_cost_matrix(
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    X_probe: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Evaluate the objective at probe points and average over the probe dimension."""
    batch, num_verts, num_probe, dim = X_probe.shape
    flat_X = X_probe.reshape(-1, dim)

    if chunk_size is not None and chunk_size > 0:
        raw_costs_flat = torch.cat([objective_fn(chunk) for chunk in flat_X.split(chunk_size)])
    else:
        raw_costs_flat = objective_fn(flat_X)

    raw_costs = raw_costs_flat.reshape(batch, num_verts, num_probe)
    dtype = torch.float64 if raw_costs.dtype == torch.float64 else torch.float32
    cost_matrix = raw_costs.mean(dim=-1, dtype=dtype)

    return cost_matrix


def resolve_cost_scale(
    cost_matrix: torch.Tensor,
    scale_cost: Optional[Union[str, float]] = None,
    finite: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the divisor ``scale_cost`` selects, as a 0-d tensor.

    Data-dependent modes ('mean', 'max_cost') are not shift-invariant, so recenter
    the cost first; otherwise a constant offset changes the effective temperature.
    Given a ``finite`` mask, they read only the entries it marks, so the penalty that
    replaced a nonfinite cost does not set the divisor.
    """
    if scale_cost is None:
        return cost_matrix.new_ones(())

    if finite is not None and isinstance(scale_cost, str):
        cost_matrix = torch.where(finite, cost_matrix, 0)
    if scale_cost == "mean":
        mean = cost_matrix.abs().mean(dtype=torch.float64).to(cost_matrix.dtype)
        if finite is not None:
            count = finite.sum()
            mean = torch.where(count == finite.numel(), mean, mean * (finite.numel() / count.clamp(min=1)))
        return torch.clamp(mean, min=1e-10)
    elif scale_cost == "max_cost":
        return torch.clamp(cost_matrix.abs().max(), min=1e-10)
    elif isinstance(scale_cost, (int, float)):
        s = float(scale_cost)
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError(
                f"Numeric scale_cost must be finite and positive, got {scale_cost!r}. "
                "A negative divisor flips the cost sign and reverses the objective."
            )
        return cost_matrix.new_full((), s)
    else:
        raise ValueError(f"Unknown scale_cost: {scale_cost!r}. Expected 'mean', 'max_cost', or a float.")


def validate_cost_scale(scale_cost) -> None:
    """Raise ValueError unless ``scale_cost`` is None, 'mean', 'max_cost', or a finite positive number."""
    resolve_cost_scale(torch.ones((), dtype=torch.float64, device="cpu"), scale_cost)


def scale_cost_matrix(
    cost_matrix: torch.Tensor,
    scale_cost: Optional[Union[str, float]] = None,
) -> torch.Tensor:
    """Apply cost scaling to a cost matrix.

    Recenter before calling with a data-dependent mode; see ``resolve_cost_scale``.
    """
    if scale_cost is None:
        return cost_matrix
    return cost_matrix / resolve_cost_scale(cost_matrix, scale_cost)
