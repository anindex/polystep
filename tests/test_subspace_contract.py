"""Behaviour every subspace class must share, parametrized over all of them.

Four properties follow from reconstructing a state_dict from a coordinate vector:
a zero coordinate changes nothing, batched reconstruction agrees with the
single-vector one, absorb folds the perturbation into the base and zeros the
coordinates, and absorb does not move the represented point.
"""

import pytest
import torch
import torch.nn as nn

from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cma_subspace import CMAAdaptiveSubspace
from polystep.factored_subspace import FactoredSubspace
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(12, 10), nn.ReLU(), nn.Linear(10, 6))


# Each entry returns (subspace, projection); every remaining class takes one.
BUILDERS = {
    "adaptive": lambda lay: _with_single(AdaptiveSubspace.from_layout(lay, rank=3)),
    "cma": lambda lay: _with_single(
        CMAAdaptiveSubspace.from_adaptive_subspace(AdaptiveSubspace.from_layout(lay, rank=3))
    ),
    "hybrid": lambda lay: _with_dict(HybridSubspace.from_layout(lay, rank=3)),
    "factored": lambda lay: _with_dict(FactoredSubspace.from_layout(lay, rank=3, seed=0)),
}


def _with_single(sub):
    return sub, sub.init_projection(generator=torch.Generator().manual_seed(0))


def _with_dict(sub):
    return sub, sub.init_projections(torch.device("cpu"), torch.float32)


def _apply(sub, proj, base_sd, coords):
    return sub.apply_perturbation(proj, base_sd, coords)


def _batch(sub, proj, base_sd, coords_batch):
    return sub.reconstruct_batch(proj, base_sd, coords_batch)


def _absorb(sub, proj, base_sd, coords):
    return sub.absorb(proj, base_sd, coords)


@pytest.fixture(params=sorted(BUILDERS))
def case(request):
    model = _model()
    sub, proj = BUILDERS[request.param](ParamLayout.from_module(model))
    return sub, proj, model.state_dict()


def _coords(sub, scale=0.01):
    torch.manual_seed(1)
    return torch.randn(sub.subspace_dim) * scale


def test_a_zero_coordinate_leaves_the_base_unchanged(case):
    sub, proj, base_sd = case
    result = _apply(sub, proj, base_sd, torch.zeros(sub.subspace_dim))
    for key, value in result.items():
        torch.testing.assert_close(value, base_sd[key], msg=lambda m, k=key: f"{k}: {m}")


def test_a_nonzero_coordinate_moves_at_least_one_parameter(case):
    """Guards the zero test above: it would also pass if the class ignored coords."""
    sub, proj, base_sd = case
    result = _apply(sub, proj, base_sd, _coords(sub))
    assert any(not torch.allclose(v, base_sd[k]) for k, v in result.items())


def test_the_batched_reconstruction_agrees_with_the_single_one(case):
    sub, proj, base_sd = case
    torch.manual_seed(2)
    batch = torch.randn(4, sub.subspace_dim) * 0.01

    batched = _batch(sub, proj, base_sd, batch)
    for i in range(batch.shape[0]):
        single = _apply(sub, proj, base_sd, batch[i])
        for key, value in single.items():
            torch.testing.assert_close(
                batched[key][i], value, atol=1e-5, rtol=1e-5, msg=lambda m, k=key, i=i: f"row {i}, {k}: {m}"
            )


def test_absorb_zeros_the_coordinates_and_folds_them_into_the_base(case):
    sub, proj, base_sd = case
    coords = _coords(sub)
    expected = _apply(sub, proj, base_sd, coords)

    new_base, zeroed = _absorb(sub, proj, base_sd, coords)

    assert zeroed.shape == coords.shape
    assert torch.all(zeroed == 0)
    for key, value in expected.items():
        torch.testing.assert_close(new_base[key], value, atol=1e-6, rtol=1e-6, msg=lambda m, k=key: f"{k}: {m}")


def test_absorb_does_not_move_the_represented_point(case):
    """base + P @ coords must be the same point before and after the fold."""
    sub, proj, base_sd = case
    coords = _coords(sub)
    before = _apply(sub, proj, base_sd, coords)

    new_base, zeroed = _absorb(sub, proj, base_sd, coords)
    after = _apply(sub, proj, new_base, zeroed)

    for key, value in before.items():
        torch.testing.assert_close(after[key], value, atol=1e-6, rtol=1e-6, msg=lambda m, k=key: f"{k}: {m}")
