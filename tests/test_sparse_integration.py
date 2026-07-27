"""Integration tests for sparse projection in PolyStepOptimizer.

Covers the projection_type parameter, sparse projection creation, step execution,
rotation, absorb, and dtype compatibility.
"""

import math

import pytest
import torch
import torch.nn as nn

from polystep.optimizer import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.projection import SparseRandomProjection


@pytest.fixture
def small_model():
    """Small model for quick testing (below auto-sparse threshold).

    Note: This model has ~67 params which is below the 10K sparse threshold.
    Use medium_model for sparse projection tests.
    """
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )


@pytest.fixture
def medium_model():
    """Medium model above tiny threshold but below auto-sparse threshold.

    Has ~11K params which is above 10K tiny threshold but below 1M auto-sparse.
    Suitable for testing explicit sparse projection.
    """
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


class TestProjectionTypeSparse:
    def test_projection_type_sparse_creates_sparse(self, medium_model, medium_subspace):
        """projection_type='sparse' creates SparseRandomProjection.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        opt = PolyStepOptimizer(
            medium_model,
            subspace=medium_subspace,
            projection_type="sparse",
            seed=0,
            compile=False,
        )
        assert isinstance(opt.state.projection, SparseRandomProjection), (
            f"Expected SparseRandomProjection, got {type(opt.state.projection).__name__}"
        )
        assert opt.projection_type == "sparse"


class TestProjectionTypeDense:
    def test_projection_type_dense_creates_dense(self, small_model, adaptive_subspace):
        """projection_type='dense' creates torch.Tensor (not SparseRandomProjection)."""
        opt = PolyStepOptimizer(
            small_model,
            subspace=adaptive_subspace,
            projection_type="dense",
            seed=0,
            compile=False,
        )
        assert isinstance(opt.state.projection, torch.Tensor), (
            f"Expected torch.Tensor, got {type(opt.state.projection).__name__}"
        )
        assert not isinstance(opt.state.projection, SparseRandomProjection)
        assert opt.projection_type == "dense"


class TestProjectionTypeInvalid:
    def test_projection_type_invalid_raises(self, small_model, adaptive_subspace):
        """Invalid projection_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid projection_type"):
            PolyStepOptimizer(
                small_model,
                subspace=adaptive_subspace,
                projection_type="invalid",
                compile=False,
            )


class TestSparseProjectionStep:
    def test_sparse_projection_step_descends(self, medium_model, medium_subspace, regression_closure):
        """Sparse projection actually optimizes, not merely runs.

        The closure has to depend on the parameters. With ``torch.rand`` the cost matrix,
        the plan and the barycentric step are all driven by noise, so nothing about the
        sparse pipeline is under test.
        """
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


class TestSparseProjectionAbsorb:
    def test_sparse_projection_absorb_works(self, medium_model, medium_subspace, regression_closure):
        """Absorb with sparse projection creates new SparseRandomProjection.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        # Configure for aggressive absorb
        sub = AdaptiveSubspace(
            full_dim=medium_subspace.full_dim,
            subspace_dim=medium_subspace.subspace_dim,
            absorb_mode="periodic",
            absorb_interval=2,  # Absorb every 2 steps
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


class TestSparseProjectionDimensions:
    def test_sparse_projection_dimensions(self, medium_model, medium_subspace):
        """Sparse projection has correct full_dim and subspace_dim.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
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


class TestAutoSelectsSparseForLargeModel:
    def test_auto_selects_sparse_for_large_model(self):
        """Auto-selection chooses sparse for models > 1M params."""
        torch.manual_seed(42)
        # Create model with ~2M params (above CPU threshold of 1M)
        large_model = nn.Sequential(
            nn.Linear(1000, 1000),  # 1M params
            nn.ReLU(),
            nn.Linear(1000, 1000),  # 1M params
        )
        num_params = sum(p.numel() for p in large_model.parameters())
        assert num_params > 1_000_000, f"Model should have >1M params, has {num_params}"

        subspace = AdaptiveSubspace.auto_from_params(large_model, max_rank=64)
        opt = PolyStepOptimizer(
            large_model,
            subspace=subspace,
            projection_type="auto",
            compile=False,
        )

        assert opt.projection_type == "sparse", f"Expected 'sparse' for large model, got '{opt.projection_type}'"
        assert isinstance(opt.state.projection, SparseRandomProjection)


class TestAutoSelectsDenseForSmallModel:
    def test_auto_selects_dense_for_small_model(self, small_model, adaptive_subspace):
        """Auto-selection chooses dense for models < 1M params."""
        num_params = sum(p.numel() for p in small_model.parameters())
        assert num_params < 1_000_000, f"Model should have <1M params, has {num_params}"

        opt = PolyStepOptimizer(
            small_model,
            subspace=adaptive_subspace,
            projection_type="auto",
            compile=False,
        )

        assert opt.projection_type == "dense", f"Expected 'dense' for small model, got '{opt.projection_type}'"
        assert isinstance(opt.state.projection, torch.Tensor)
        assert not isinstance(opt.state.projection, SparseRandomProjection)


class TestTinyModelFallbackToDense:
    def test_tiny_model_fallback_to_dense(self):
        """Tiny models (<10K params) fall back to dense even if sparse requested."""
        torch.manual_seed(42)
        # Very small model with < 10K params
        tiny_model = nn.Sequential(
            nn.Linear(10, 10),  # 110 params
            nn.ReLU(),
            nn.Linear(10, 2),  # 22 params
        )
        num_params = sum(p.numel() for p in tiny_model.parameters())
        assert num_params < 10_000, f"Model should have <10K params, has {num_params}"

        subspace = AdaptiveSubspace.auto_from_params(tiny_model, max_rank=8)
        opt = PolyStepOptimizer(
            tiny_model,
            subspace=subspace,
            projection_type="sparse",  # Explicitly request sparse
            compile=False,
        )

        # Should fall back to dense because model is too small
        assert opt.projection_type == "dense", f"Expected 'dense' fallback for tiny model, got '{opt.projection_type}'"
        assert isinstance(opt.state.projection, torch.Tensor)
        assert not isinstance(opt.state.projection, SparseRandomProjection)


class TestExplicitProjectionTypeOverride:
    def test_explicit_dense_creates_dense(self):
        """Explicit 'dense' creates dense tensor even for large model."""
        torch.manual_seed(42)
        # Large model that would auto-select sparse
        large_model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
        )
        num_params = sum(p.numel() for p in large_model.parameters())
        assert num_params > 1_000_000

        subspace = AdaptiveSubspace.auto_from_params(large_model, max_rank=64)
        opt = PolyStepOptimizer(
            large_model,
            subspace=subspace,
            projection_type="dense",  # Explicit dense
            compile=False,
        )

        assert opt.projection_type == "dense", f"Expected 'dense' with explicit request, got '{opt.projection_type}'"
        assert isinstance(opt.state.projection, torch.Tensor)
        assert not isinstance(opt.state.projection, SparseRandomProjection)
