"""Pin the coordinate-to-weight gain of each subspace class.

``step_radius`` is measured in subspace coordinates, so the physical step depends on
the class in use; a change to either normalization silently retunes every run.
"""

import pytest
import torch
import torch.nn as nn

from polystep.hybrid_subspace import HybridSubspace
from polystep.factored_subspace import FactoredSubspace
from polystep.transform import ParamLayout


def _weight_norm_for_unit_coords(subspace, projections, model, seed=0):
    """||dW||_F produced by a unit-norm coordinate vector."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(subspace.subspace_dim, generator=g)
    coords = coords / coords.norm()
    base = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    sd = subspace.apply_perturbation(projections, base, coords)
    return torch.cat([v.reshape(-1) for v in sd.values()]).norm().item()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hybrid_is_unit_gain_on_its_dense_blocks():
    """step_radius is in coordinates, so a gain other than 1 silently retunes every run."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    layout = ParamLayout.from_module(model)

    hybrid = HybridSubspace.from_layout(layout, rank=8)
    hybrid_proj = hybrid.init_projections(torch.device("cpu"), torch.float32)

    assert _weight_norm_for_unit_coords(hybrid, hybrid_proj, model) == pytest.approx(1.0, rel=0.05)


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
