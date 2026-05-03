"""Cost matrix computation for Sinkhorn Step.

Wraps an objective function into a cost matrix by evaluating it at
probe points and averaging over the probe dimension.
"""
from typing import Callable, Optional, Union

import torch


def compute_cost_matrix(
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    X_probe: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute the cost matrix by evaluating objective at probe points.

    Evaluates the objective function on all probe points and averages
    over the probe dimension to get a (batch, num_vertices) cost matrix.

    Args:
        objective_fn: Function mapping (..., dim) -> (...) costs.
        X_probe: Probe array of shape (batch, num_vertices, num_probe, dim).
        chunk_size: If set, evaluate in chunks to limit memory.

    Returns:
        Cost matrix of shape (batch, num_vertices).
    """
    batch, num_verts, num_probe, dim = X_probe.shape

    if chunk_size is not None and chunk_size > 0:
        # Chunked evaluation for memory efficiency
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
    cost_matrix = raw_costs.mean(dim=-1)  # (batch, num_vertices)

    return cost_matrix


def scale_cost_matrix(
    cost_matrix: torch.Tensor,
    scale_cost: Optional[Union[str, float]] = None,
) -> torch.Tensor:
    """Apply cost scaling to a cost matrix.

    Args:
        cost_matrix: Raw cost matrix.
        scale_cost: Scaling strategy ('mean', 'max_cost', or float).

    Returns:
        Scaled cost matrix.
    """
    if scale_cost is None:
        return cost_matrix

    if scale_cost == 'mean':
        s = torch.clamp(cost_matrix.abs().mean(), min=1e-10)
        return cost_matrix / s
    elif scale_cost == 'max_cost':
        s = torch.clamp(cost_matrix.abs().max(), min=1e-10)
        return cost_matrix / s
    elif isinstance(scale_cost, (int, float)):
        return cost_matrix / float(scale_cost)
    else:
        raise ValueError(
            f"Unknown scale_cost: {scale_cost!r}. Expected 'mean', 'max_cost', or a float."
        )
