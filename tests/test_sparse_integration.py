"""Integration tests for sparse projection in PolyStepOptimizer."""

import math

import pytest
import torch
import torch.nn as nn

from polystep.optimizer import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.projection import SparseRandomProjection


@pytest.fixture
def small_model():
    """Small model below the auto-sparse threshold."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )


@pytest.fixture
def medium_model():
    """Model above the 10K tiny threshold but below the 1M auto-sparse threshold."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(100, 100),  # 10100 params
        nn.ReLU(),
        nn.Linear(100, 10),  # 1010 params
    )


@pytest.fixture
def adaptive_subspace(small_model):
    """AdaptiveSubspace configured for small model."""
    return AdaptiveSubspace.auto_from_params(small_model, max_rank=16)


@pytest.fixture
def medium_subspace(medium_model):
    """AdaptiveSubspace configured for medium model."""
    return AdaptiveSubspace.auto_from_params(medium_model, max_rank=32)


def _mlp(*dims):
    torch.manual_seed(42)
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@pytest.mark.parametrize(
    "dims, requested, expected, why",
    [
        ((100, 100, 10), "sparse", "sparse", "explicit sparse above the 10K tiny threshold"),
        ((10, 5, 2), "dense", "dense", "explicit dense"),
        ((1000, 1000, 1000), "auto", "sparse", "auto goes sparse above 1M params"),
        ((10, 5, 2), "auto", "dense", "auto stays dense below 1M params"),
        ((10, 10, 2), "sparse", "dense", "below 10K params, sparse request falls back"),
        ((1000, 1001), "dense", "dense", "explicit dense wins over the auto-sparse threshold"),
    ],
)
def test_projection_type_selection(dims, requested, expected, why):
    """One table for every branch of the dense/sparse selection rule."""
    model = _mlp(*dims)
    subspace = AdaptiveSubspace.auto_from_params(model, min_rank=8, max_rank=8)
    opt = PolyStepOptimizer(model, subspace=subspace, projection_type=requested, seed=0, compile=False)

    assert opt.projection_type == expected, why
    is_sparse = isinstance(opt.state.projection, SparseRandomProjection)
    assert is_sparse == (expected == "sparse"), f"{why}: got {type(opt.state.projection).__name__}"


def test_projection_type_invalid_raises(small_model, adaptive_subspace):
    with pytest.raises(ValueError, match="Invalid projection_type"):
        PolyStepOptimizer(small_model, subspace=adaptive_subspace, projection_type="invalid", compile=False)


def test_sparse_projection_step_descends(medium_model, medium_subspace, regression_closure):
    """Sparse projection must descend on a parameter-dependent closure."""
    opt = PolyStepOptimizer(
        medium_model,
        subspace=medium_subspace,
        projection_type="sparse",
        seed=42,
        step_radius=0.5,
        compile=False,
    )
    closure = regression_closure(medium_model)

    initial_proj_seed = opt.state.projection.seed
    costs = [opt.step(closure) for _ in range(4)]

    assert all(math.isfinite(c) for c in costs), f"non-finite OT cost: {costs}"
    assert closure.true_loss() < closure.initial_loss, (
        f"sparse projection did not descend: {closure.initial_loss:.5f} -> {closure.true_loss():.5f}"
    )
    assert opt.state.projection.seed != initial_proj_seed, "projection seed should change after rotation"
    assert isinstance(opt.state.projection, SparseRandomProjection)


def test_sparse_projection_absorb_works(medium_model, medium_subspace, regression_closure):
    """Absorb with sparse projection builds a fresh SparseRandomProjection."""
    sub = AdaptiveSubspace(
        full_dim=medium_subspace.full_dim,
        subspace_dim=medium_subspace.subspace_dim,
        absorb_mode="periodic",
        absorb_interval=2,
        _entry_specs=medium_subspace._entry_specs,
    )

    opt = PolyStepOptimizer(
        medium_model,
        subspace=sub,
        projection_type="sparse",
        seed=42,
        compile=False,
    )

    closure = regression_closure(medium_model)
    initial_seed = opt.state.projection.seed
    initial_base = {k: v.clone() for k, v in opt.state.base_params.items()}

    for _ in range(3):
        opt.step(closure)

    assert opt.state.absorb_count >= 1, f"expected at least 1 absorb, got {opt.state.absorb_count}"
    assert any(not torch.equal(initial_base[k], opt.state.base_params[k]) for k in initial_base), (
        "absorb fired but the base weights never took on the perturbation"
    )
    assert isinstance(opt.state.projection, SparseRandomProjection)
    assert opt.state.projection.seed != initial_seed, "absorb should build a fresh projection"


def test_sparse_projection_dimensions(medium_model, medium_subspace):
    """Sparse projection reports the subspace's full_dim and subspace_dim."""
    opt = PolyStepOptimizer(
        medium_model,
        subspace=medium_subspace,
        projection_type="sparse",
        seed=42,
        compile=False,
    )

    sparse_proj = opt.state.projection
    assert sparse_proj.full_dim == medium_subspace.full_dim, (
        f"full_dim mismatch: {sparse_proj.full_dim} vs {medium_subspace.full_dim}"
    )
    assert sparse_proj.subspace_dim == medium_subspace.subspace_dim, (
        f"subspace_dim mismatch: {sparse_proj.subspace_dim} vs {medium_subspace.subspace_dim}"
    )
