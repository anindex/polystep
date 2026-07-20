"""Unit tests for SparseRandomProjection.

Tests cover:
- Core functionality (shapes, transpose, determinism)
- Memory efficiency (O(nnz) not O(n*k))
- JLT distance preservation property
- Device compatibility (CPU and CUDA)
- Statistical properties (unit variance, extreme-compression warning)
"""

import warnings

import pytest
import torch

from polystep.projection import SparseRandomProjection


class TestCoreProjection:
    """Tests for basic projection functionality."""

    @pytest.mark.parametrize(
        "input_shape, expected_shape",
        [
            ((64,), (10000,)),
            ((16, 64), (16, 10000)),
        ],
    )
    def test_project_shape_1d(self, input_shape, expected_shape):
        """Projection produces correct output shape."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)
        coords = torch.randn(*input_shape)
        full = proj.project(coords)
        assert full.shape == expected_shape
        assert full.dtype == coords.dtype

    @pytest.mark.parametrize(
        "input_shape, expected_shape",
        [
            ((10000,), (64,)),
            ((16, 10000), (16, 64)),
        ],
    )
    def test_project_transpose_shape_1d(self, input_shape, expected_shape):
        """Transpose projection (full -> subspace) produces correct shape."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)
        full = torch.randn(*input_shape)
        coords = proj.project_transpose(full)
        assert coords.shape == expected_shape
        assert coords.dtype == full.dtype

    def test_deterministic_with_seed(self):
        """Same seed produces identical projections."""
        proj1 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=123)
        proj2 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=123)

        coords = torch.randn(64)
        full1 = proj1.project(coords)
        full2 = proj2.project(coords)

        assert torch.allclose(full1, full2)

    def test_different_seeds_differ(self):
        """Different seeds produce different projections."""
        proj1 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=123)
        proj2 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=456)

        coords = torch.randn(64)
        full1 = proj1.project(coords)
        full2 = proj2.project(coords)

        # Should not be equal (extremely unlikely for random projections)
        assert not torch.allclose(full1, full2)


class TestMemoryEfficiency:
    """Tests for memory efficiency properties."""

    def test_memory_estimate(self):
        """Memory estimate is calculated correctly."""
        proj = SparseRandomProjection(full_dim=100_000, subspace_dim=256, seed=42)

        # Expected: nnz_per_col = int(0.01 * 100000) = 1000 per Li, Hastie, Church
        # Actually: density = 1/sqrt(100000) ~ 0.00316, nnz_per_col = 316
        expected_nnz_per_col = max(1, int(1.0 / (100_000**0.5) * 100_000))
        total_nnz = expected_nnz_per_col * 256

        # Memory: indices (2 * nnz * 8) + values (nnz * 4)
        expected_bytes = 2 * total_nnz * 8 + total_nnz * 4

        assert proj.memory_bytes == expected_bytes
        assert proj._nnz_per_col == expected_nnz_per_col

    def test_memory_vs_dense(self):
        """Sparse projection uses much less memory than dense equivalent."""
        full_dim = 100_000
        subspace_dim = 256
        proj = SparseRandomProjection(full_dim, subspace_dim, seed=42)

        # Dense would be full_dim * subspace_dim * 4 bytes (float32)
        dense_bytes = full_dim * subspace_dim * 4
        sparse_bytes = proj.memory_bytes

        # At default density 1/sqrt(100K) ~ 0.316%, should be >50x smaller
        # (Memory overhead from int64 indices reduces ratio vs float32-only dense)
        ratio = dense_bytes / sparse_bytes
        assert ratio > 50, f"Expected >50x reduction, got {ratio:.1f}x"

    @pytest.mark.filterwarnings("ignore:SparseRandomProjection.*below the empirical floor:UserWarning")
    def test_large_scale_memory(self):
        """Memory stays bounded for large parameter counts."""
        # Simulate 100M params with rank-256
        proj = SparseRandomProjection(full_dim=100_000_000, subspace_dim=256, seed=42)

        # Dense: 100M * 256 * 4 = 100GB
        dense_gb = 100_000_000 * 256 * 4 / 1e9

        # Sparse should be << 1GB
        sparse_gb = proj.memory_bytes / 1e9

        assert sparse_gb < 1.0, f"Expected <1 GB, got {sparse_gb:.3f} GB"
        assert dense_gb > 90, f"Dense baseline should be ~100GB, got {dense_gb:.1f}GB"

    def test_custom_density(self):
        """Custom density is respected."""
        # Use 1% density explicitly
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, density=0.01, seed=42)

        expected_nnz_per_col = max(1, int(0.01 * 10000))  # 100
        assert proj._nnz_per_col == expected_nnz_per_col
        assert proj.nnz == expected_nnz_per_col * 64


class TestJLTProperty:
    """Tests for Johnson-Lindenstrauss distance preservation."""

    def test_distance_preservation(self):
        """Sparse projection approximately preserves distances."""
        # JLT: distances preserved within (1 +/- eps) factor
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=256, seed=42)

        # Create random vectors in subspace
        torch.manual_seed(123)
        x1 = torch.randn(256)
        x2 = torch.randn(256)

        # Distance in subspace
        d_sub = torch.norm(x1 - x2).item()

        # Distance after projection to full space
        d_full = torch.norm(proj.project(x1) - proj.project(x2)).item()

        # Should be approximately equal
        # JLT allows multiplicative distortion; sparse JLT has similar bounds
        ratio = d_full / d_sub
        assert 0.5 < ratio < 2.0, f"Distance ratio {ratio} outside [0.5, 2.0]"

    def test_multiple_distance_preservation(self):
        """Distance preservation holds across multiple pairs."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=128, seed=42)

        torch.manual_seed(999)
        ratios = []
        for _ in range(20):
            x1 = torch.randn(128)
            x2 = torch.randn(128)

            d_sub = torch.norm(x1 - x2).item()
            d_full = torch.norm(proj.project(x1) - proj.project(x2)).item()

            if d_sub > 1e-6:  # Avoid division by zero
                ratios.append(d_full / d_sub)

        # Most ratios should be reasonably close to 1
        mean_ratio = sum(ratios) / len(ratios)
        assert 0.7 < mean_ratio < 1.5, f"Mean distance ratio {mean_ratio} too far from 1"

    def test_zero_vector_maps_to_zero(self):
        """Zero vector maps to zero (linearity check)."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        zero = torch.zeros(64)
        result = proj.project(zero)

        assert torch.allclose(result, torch.zeros(10000))

    def test_linearity(self):
        """Projection is linear: P(a*x) = a*P(x)."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        x = torch.randn(64)
        scale = 3.14

        # P(scale * x) should equal scale * P(x)
        result1 = proj.project(scale * x)
        result2 = scale * proj.project(x)

        assert torch.allclose(result1, result2, rtol=1e-5)


class TestDtypeHandling:
    """Tests for dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_float32(self, dtype):
        """Works with float32 and float64 inputs."""
        proj = SparseRandomProjection(full_dim=1000, subspace_dim=32, seed=42)
        coords = torch.randn(32, dtype=dtype)
        full = proj.project(coords)
        assert full.dtype == dtype


class TestCUDA:
    """Tests for CUDA compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "input_shape, expected_shape",
        [
            ((64,), (10000,)),
            ((8, 64), (8, 10000)),
        ],
    )
    def test_cuda_projection(self, input_shape, expected_shape):
        """Projection works on GPU."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        coords = torch.randn(*input_shape, device="cuda")
        full = proj.project(coords)

        assert full.device.type == "cuda"
        assert full.shape == expected_shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_transpose(self):
        """Transpose projection works on GPU."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        full = torch.randn(10000, device="cuda")
        coords = proj.project_transpose(full)

        assert coords.device.type == "cuda"
        assert coords.shape == (64,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_determinism(self):
        """Same seed produces same results on GPU."""
        proj1 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=777)
        proj2 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=777)

        coords = torch.randn(64, device="cuda")
        full1 = proj1.project(coords)
        full2 = proj2.project(coords)

        assert torch.allclose(full1, full2)


class TestRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Repr contains useful info."""
        proj = SparseRandomProjection(full_dim=100000, subspace_dim=256, seed=42)
        r = repr(proj)

        assert "SparseRandomProjection" in r
        assert "full_dim=100000" in r
        assert "subspace_dim=256" in r
        assert "density=" in r
        assert "memory=" in r


class TestEdgeCases:
    """Tests for edge cases."""

    def test_min_nnz_per_col(self):
        """At least 1 nonzero per column even at very low density."""
        # Very small full_dim with very low density
        proj = SparseRandomProjection(full_dim=10, subspace_dim=5, density=0.001, seed=42)

        # Should have at least 1 nonzero per column
        assert proj._nnz_per_col >= 1

        # Should still project correctly
        coords = torch.randn(5)
        full = proj.project(coords)
        assert full.shape == (10,)


class TestStatisticalProperties:
    """Variance and warning behavior for the sparse JL projection."""

    @pytest.mark.filterwarnings("ignore:Sparse invariant checks:UserWarning")
    def test_project_transpose_has_unit_variance(self):
        """``project_transpose`` computes ``P^T @ full`` with ``P`` having
        ``nnz_per_col`` Rademacher entries scaled by
        ``1/sqrt(nnz_per_col)``. For ``full ~ N(0, I)`` the projected
        coordinate variance is ~1.
        """
        full_dim = 10000
        subspace_dim = 256
        proj = SparseRandomProjection(
            full_dim=full_dim,
            subspace_dim=subspace_dim,
            seed=0,
        )

        gen = torch.Generator(device="cpu").manual_seed(1)
        x = torch.randn(full_dim, generator=gen)
        y = proj.project_transpose(x)
        assert y.shape == (subspace_dim,)
        sample_var = y.var().item()
        assert 0.5 < sample_var < 2.0, f"projected coordinate variance off: got {sample_var:.3f}, expected ~1.0"

    def test_warns_at_extreme_compression(self):
        """Subspace ratio below 1e-5 triggers a UserWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SparseRandomProjection(
                full_dim=10_000_000,
                subspace_dim=64,
                seed=0,
            )

        msgs = [str(w.message).lower() for w in caught]
        assert any("compression" in m or "below the empirical floor" in m for m in msgs), (
            f"expected extreme-compression warning; got {msgs}"
        )
