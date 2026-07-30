"""Subspace classes do not share a coordinate-to-weight normalization.

``step_radius`` is measured in subspace coordinates, so the physical step it produces
depends on which class is in use. These gains are pinned because a change to either
normalization silently retunes every run that switches class.
"""

import pytest
import torch
import torch.nn as nn

from polystep.hybrid_subspace import HybridSubspace
from polystep.factored_subspace import FactoredSubspace
from polystep.subspace import LinearSubspace
from polystep.transform import ParamLayout


def _weight_norm_for_unit_coords(subspace, projections, model, seed=0):
    """||dW||_F produced by a unit-norm coordinate vector."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(subspace.subspace_dim, generator=g)
    coords = coords / coords.norm()
    base = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    args = (projections, base, coords) if projections is not None else (base, coords)
    sd = subspace.apply_perturbation(*args)
    return torch.cat([v.reshape(-1) for v in sd.values()]).norm().item()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_linear_amplifies_where_hybrid_and_factored_are_unit_gain():
    """A shared step_radius is a ~4.4x different weight step on this model.

    The amplification is sqrt(total_params / subspace_dim), so a smaller model would
    shrink the very gap this test exists to measure.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    layout = ParamLayout.from_module(model)

    linear = LinearSubspace.from_layout(layout, rank=8)
    hybrid = HybridSubspace.from_layout(layout, rank=8)
    hybrid_proj = hybrid.init_projections(torch.device("cpu"), torch.float32)

    linear_gain = _weight_norm_for_unit_coords(linear, None, model)
    hybrid_gain = _weight_norm_for_unit_coords(hybrid, hybrid_proj, model)

    # LinearSubspace scales its Gaussian by 1/sqrt(num_coords) per layer, so the map
    # amplifies by sqrt(num_params / num_coords). Hybrid orthonormalizes instead.
    expected = (layout.total_params / linear.subspace_dim) ** 0.5
    assert linear_gain == pytest.approx(expected, rel=0.15), (linear_gain, expected)
    assert hybrid_gain == pytest.approx(1.0, rel=0.05), hybrid_gain
    assert linear_gain / hybrid_gain > 2.0, "the class-dependent step size documented on step_radius is gone"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_factored_preserves_the_coordinate_norm():
    """B has orthonormal rows, so ||A @ B||_F == ||A||_F."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    layout = ParamLayout.from_module(model)
    factored = FactoredSubspace.from_layout(layout, rank=4)
    proj = factored.init_projections(torch.device("cpu"), torch.float32)
    gain = _weight_norm_for_unit_coords(factored, proj, model)
    assert gain == pytest.approx(1.0, rel=0.05), gain
