"""Cost matrix computation for Sinkhorn Step."""

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

    if chunk_size is not None and chunk_size > 0:
        flat_X = X_probe.reshape(-1, dim)
        N = flat_X.shape[0]

        results = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = flat_X[start:end]
            results.append(objective_fn(chunk))

        raw_costs_flat = torch.cat(results, dim=0)
    else:
        flat_X = X_probe.reshape(-1, dim)
        raw_costs_flat = objective_fn(flat_X)

    raw_costs = raw_costs_flat.reshape(batch, num_verts, num_probe)
    cost_matrix = raw_costs.mean(dim=-1)

    return cost_matrix


def resolve_cost_scale(
    cost_matrix: torch.Tensor,
    scale_cost: Optional[Union[str, float]] = None,
) -> torch.Tensor:
    """Return the divisor ``scale_cost`` selects, as a 0-d tensor.

    Data-dependent modes ('mean', 'max_cost') are not shift-invariant, so recenter
    the cost first; otherwise a constant offset changes the effective temperature.
    """
    if scale_cost is None:
        return cost_matrix.new_ones(())

    if scale_cost == "mean":
        return torch.clamp(cost_matrix.abs().mean(), min=1e-10)
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
