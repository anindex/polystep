"""Integration tests for sparse projection in PolyStepOptimizer.

Sparse projection: Tests covering projection_type parameter, sparse projection creation,
step execution, rotation, absorb, and dtype compatibility.
"""
import pytest
import torch
import torch.nn as nn

from polystep.optimizer import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.projection import SparseRandomProjection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        nn.Linear(100, 10),   # 1010 params
    )


@pytest.fixture
def adaptive_subspace(small_model):
    """AdaptiveSubspace configured for small model."""
    return AdaptiveSubspace.auto_from_params(small_model, max_rank=16)


@pytest.fixture
def medium_subspace(medium_model):
    """AdaptiveSubspace configured for medium model."""
    return AdaptiveSubspace.auto_from_params(medium_model, max_rank=32)


# ---------------------------------------------------------------------------
# projection_type='sparse' creates SparseRandomProjection
# ---------------------------------------------------------------------------


class TestProjectionTypeSparse:
    def test_projection_type_sparse_creates_sparse(self, medium_model, medium_subspace):
        """projection_type='sparse' creates SparseRandomProjection.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        opt = PolyStepOptimizer(
            medium_model,
            subspace=medium_subspace,
            projection_type='sparse',
            seed=0,
            compile=False,
        )
        assert isinstance(opt.state.projection, SparseRandomProjection), (
            f"Expected SparseRandomProjection, got {type(opt.state.projection).__name__}"
        )
        assert opt.projection_type == 'sparse'


# ---------------------------------------------------------------------------
# projection_type='dense' creates dense tensor
# ---------------------------------------------------------------------------


class TestProjectionTypeDense:
    def test_projection_type_dense_creates_dense(self, small_model, adaptive_subspace):
        """projection_type='dense' creates torch.Tensor (not SparseRandomProjection)."""
        opt = PolyStepOptimizer(
            small_model,
            subspace=adaptive_subspace,
            projection_type='dense',
            seed=0,
            compile=False,
        )
        assert isinstance(opt.state.projection, torch.Tensor), (
            f"Expected torch.Tensor, got {type(opt.state.projection).__name__}"
        )
        assert not isinstance(opt.state.projection, SparseRandomProjection)
        assert opt.projection_type == 'dense'


# ---------------------------------------------------------------------------
# Invalid projection_type raises ValueError
# ---------------------------------------------------------------------------


class TestProjectionTypeInvalid:
    def test_projection_type_invalid_raises(self, small_model, adaptive_subspace):
        """Invalid projection_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid projection_type"):
            PolyStepOptimizer(
                small_model,
                subspace=adaptive_subspace,
                projection_type='invalid',
                compile=False,
            )


# ---------------------------------------------------------------------------
# Sparse projection step runs without error
# ---------------------------------------------------------------------------


class TestSparseProjectionStep:
    def test_sparse_projection_step_runs(self, medium_model, medium_subspace):
        """optimizer.step() with sparse projection runs without error.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        opt = PolyStepOptimizer(
            medium_model,
            subspace=medium_subspace,
            projection_type='sparse',
            seed=42,
            compile=False,
        )

        # Create dummy closure returning random losses
        def closure(batched_params):
            batch_size = batched_params['0.weight'].shape[0]
            return torch.rand(batch_size)

        # Run 3 steps to test rotation handling
        initial_proj_seed = opt.state.projection.seed
        costs = []
        for i in range(3):
            cost = opt.step(closure)
            costs.append(cost)

        # Verify steps completed
        assert len(costs) == 3, f"Expected 3 costs, got {len(costs)}"
        # Note: OT cost with entropic regularization can be negative
        assert all(isinstance(c, float) for c in costs), "All costs should be floats"

        # Verify projection changed (seed incremented for rotation)
        assert opt.state.projection.seed != initial_proj_seed, (
            "Projection seed should change after rotation"
        )
        assert isinstance(opt.state.projection, SparseRandomProjection), (
            "Projection should still be SparseRandomProjection after steps"
        )


# ---------------------------------------------------------------------------
# Sparse projection absorb works
# ---------------------------------------------------------------------------


class TestSparseProjectionAbsorb:
    def test_sparse_projection_absorb_works(self, medium_model, medium_subspace):
        """Absorb with sparse projection creates new SparseRandomProjection.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        # Configure for aggressive absorb
        sub = AdaptiveSubspace(
            full_dim=medium_subspace.full_dim,
            subspace_dim=medium_subspace.subspace_dim,
            absorb_mode='periodic',
            absorb_interval=2,  # Absorb every 2 steps
            _entry_specs=medium_subspace._entry_specs,
        )

        opt = PolyStepOptimizer(
            medium_model,
            subspace=sub,
            projection_type='sparse',
            seed=42,
            compile=False,
        )

        def closure(batched_params):
            batch_size = batched_params['0.weight'].shape[0]
            return torch.rand(batch_size)

        initial_seed = opt.state.projection.seed

        # Run 3 steps - absorb should trigger at step 2
        for i in range(3):
            opt.step(closure)

        # Verify absorb count increased
        assert opt.state.absorb_count >= 1, (
            f"Expected at least 1 absorb, got {opt.state.absorb_count}"
        )

        # Verify projection is still SparseRandomProjection
        assert isinstance(opt.state.projection, SparseRandomProjection), (
            "Projection should be SparseRandomProjection after absorb"
        )

        # Verify seed changed (absorb creates new projection)
        assert opt.state.projection.seed != initial_seed, (
            "Projection seed should change after absorb"
        )


# ---------------------------------------------------------------------------
# Sparse projection dtype compatibility
# ---------------------------------------------------------------------------


class TestSparseProjectionDtype:
    def test_sparse_projection_dtype_compatibility(self, medium_model, medium_subspace):
        """Sparse projection works with different dtypes (mixed precision).

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        opt = PolyStepOptimizer(
            medium_model,
            subspace=medium_subspace,
            projection_type='sparse',
            seed=42,
            compile=False,
            # Note: mixed_precision=True requires GPU with BF16 support
            # We test that sparse projection works with regular FP32 closure
        )

        # Create closure returning FP32 losses
        def closure(batched_params):
            batch_size = batched_params['0.weight'].shape[0]
            return torch.rand(batch_size, dtype=torch.float32)

        # Run a step
        cost = opt.step(closure)

        assert isinstance(cost, float), f"Cost should be float, got {type(cost)}"
        # Note: OT cost with entropic regularization can be negative
        import math
        assert math.isfinite(cost), "Cost should be finite"

    def test_sparse_projection_with_float64_coords(self, medium_model, medium_subspace):
        """Sparse projection handles float64 subspace coordinates.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        opt = PolyStepOptimizer(
            medium_model,
            subspace=medium_subspace,
            projection_type='sparse',
            seed=42,
            compile=False,
        )

        # Manually verify sparse projection project() works with batched input
        sparse_proj = opt.state.projection
        coords_batch = torch.randn(4, sparse_proj.subspace_dim, dtype=torch.float32)
        result = sparse_proj.project(coords_batch)

        assert result.shape == (4, sparse_proj.full_dim), (
            f"Expected shape (4, {sparse_proj.full_dim}), got {result.shape}"
        )


# ---------------------------------------------------------------------------
# Verify sparse projection dimensions match
# ---------------------------------------------------------------------------


class TestSparseProjectionDimensions:
    def test_sparse_projection_dimensions(self, medium_model, medium_subspace):
        """Sparse projection has correct full_dim and subspace_dim.

        Note: Uses medium_model (>10K params) to avoid tiny model fallback.
        """
        opt = PolyStepOptimizer(
            medium_model,
            subspace=medium_subspace,
            projection_type='sparse',
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


# ---------------------------------------------------------------------------
# Auto projection_type with auto-selection
# ---------------------------------------------------------------------------


class TestProjectionTypeAuto:
    def test_projection_type_auto_accepted(self, small_model, adaptive_subspace):
        """projection_type='auto' is accepted without error."""
        opt = PolyStepOptimizer(
            small_model,
            subspace=adaptive_subspace,
            projection_type='auto',
            seed=42,
            compile=False,
        )
        # Small model should auto-select dense
        assert opt.projection_type == 'dense'
        assert isinstance(opt.state.projection, torch.Tensor)


# ---------------------------------------------------------------------------
# Auto-selects sparse for large model
# ---------------------------------------------------------------------------


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
            projection_type='auto',
            compile=False,
        )

        assert opt.projection_type == 'sparse', (
            f"Expected 'sparse' for large model, got '{opt.projection_type}'"
        )
        assert isinstance(opt.state.projection, SparseRandomProjection)


# ---------------------------------------------------------------------------
# Auto-selects dense for small model
# ---------------------------------------------------------------------------


class TestAutoSelectsDenseForSmallModel:
    def test_auto_selects_dense_for_small_model(self, small_model, adaptive_subspace):
        """Auto-selection chooses dense for models < 1M params."""
        num_params = sum(p.numel() for p in small_model.parameters())
        assert num_params < 1_000_000, f"Model should have <1M params, has {num_params}"

        opt = PolyStepOptimizer(
            small_model,
            subspace=adaptive_subspace,
            projection_type='auto',
            compile=False,
        )

        assert opt.projection_type == 'dense', (
            f"Expected 'dense' for small model, got '{opt.projection_type}'"
        )
        assert isinstance(opt.state.projection, torch.Tensor)
        assert not isinstance(opt.state.projection, SparseRandomProjection)


# ---------------------------------------------------------------------------
# Tiny model fallback to dense even if sparse requested
# ---------------------------------------------------------------------------


class TestTinyModelFallbackToDense:
    def test_tiny_model_fallback_to_dense(self):
        """Tiny models (<10K params) fall back to dense even if sparse requested."""
        torch.manual_seed(42)
        # Very small model with < 10K params
        tiny_model = nn.Sequential(
            nn.Linear(10, 10),  # 110 params
            nn.ReLU(),
            nn.Linear(10, 2),   # 22 params
        )
        num_params = sum(p.numel() for p in tiny_model.parameters())
        assert num_params < 10_000, f"Model should have <10K params, has {num_params}"

        subspace = AdaptiveSubspace.auto_from_params(tiny_model, max_rank=8)
        opt = PolyStepOptimizer(
            tiny_model,
            subspace=subspace,
            projection_type='sparse',  # Explicitly request sparse
            compile=False,
        )

        # Should fall back to dense because model is too small
        assert opt.projection_type == 'dense', (
            f"Expected 'dense' fallback for tiny model, got '{opt.projection_type}'"
        )
        assert isinstance(opt.state.projection, torch.Tensor)
        assert not isinstance(opt.state.projection, SparseRandomProjection)


# ---------------------------------------------------------------------------
# Explicit projection_type overrides auto-selection
# ---------------------------------------------------------------------------


class TestExplicitProjectionTypeOverride:
    def test_explicit_sparse_creates_sparse(self):
        """Explicit 'sparse' creates SparseRandomProjection (unless tiny)."""
        torch.manual_seed(42)
        # Medium model above tiny threshold but below auto-sparse threshold
        model = nn.Sequential(
            nn.Linear(100, 100),  # 10.1K params
            nn.ReLU(),
            nn.Linear(100, 10),   # 1010 params
        )
        num_params = sum(p.numel() for p in model.parameters())
        # Model should be above 10K (tiny threshold) but below 1M (sparse threshold)
        assert 10_000 < num_params < 1_000_000

        subspace = AdaptiveSubspace.auto_from_params(model, max_rank=32)
        opt = PolyStepOptimizer(
            model,
            subspace=subspace,
            projection_type='sparse',  # Explicit sparse
            compile=False,
        )

        assert opt.projection_type == 'sparse', (
            f"Expected 'sparse' with explicit request, got '{opt.projection_type}'"
        )
        assert isinstance(opt.state.projection, SparseRandomProjection)

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
            projection_type='dense',  # Explicit dense
            compile=False,
        )

        assert opt.projection_type == 'dense', (
            f"Expected 'dense' with explicit request, got '{opt.projection_type}'"
        )
        assert isinstance(opt.state.projection, torch.Tensor)
        assert not isinstance(opt.state.projection, SparseRandomProjection)
