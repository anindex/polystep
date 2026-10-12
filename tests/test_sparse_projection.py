"""Unit tests for SparseRandomProjection: shapes, memory, JLT, determinism."""

import warnings

import pytest
import torch

from polystep.projection import SparseRandomProjection


class TestCoreProjection:
    @pytest.mark.parametrize(
        "input_shape, expected_shape",
        [
            ((64,), (10000,)),
            ((16, 64), (16, 10000)),
        ],
    )
    def test_project_shape_1d(self, input_shape, expected_shape):
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)
        coords = torch.randn(*input_shape)
        full = proj.project(coords)
        assert full.shape == expected_shape
        assert full.dtype == coords.dtype

    @pytest.mark.parametrize("seed2, same", [(123, True), (456, False)])
    def test_seed_alone_fixes_projection(self, seed2, same):
        coords = torch.randn(64)
        a = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=123).project(coords)
        b = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=seed2).project(coords)
        assert torch.allclose(a, b) is same


class TestMemoryEfficiency:
    def test_memory_estimate(self):
        """Density is 1/sqrt(full_dim) per Li, Hastie, Church; values are pinned
        so the test does not move with the formula.
        """
        proj = SparseRandomProjection(full_dim=100_000, subspace_dim=256, seed=42)

        assert proj._nnz_per_col == 316  # int(100_000 / sqrt(100_000))
        # An int64 row and fp32 value per drawn entry, then the CSR's int32 column and
        # fp32 value per entry and its int32 row pointers.
        assert proj.memory_bytes == 316 * 256 * 20 + 4 * 100_001

    @pytest.mark.filterwarnings("ignore:SparseRandomProjection.*below the empirical floor:UserWarning")
    def test_large_scale_memory(self):
        """Memory stays bounded for large parameter counts."""
        # Simulate 100M params with rank-256
        proj = SparseRandomProjection(full_dim=100_000_000, subspace_dim=256, seed=42)

        sparse_gb = proj.memory_bytes / 1e9

        assert sparse_gb < 1.0, f"Expected <1 GB, got {sparse_gb:.3f} GB"

    def test_custom_density(self):
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, density=0.01, seed=42)

        expected_nnz_per_col = max(1, int(0.01 * 10000))
        assert proj._nnz_per_col == expected_nnz_per_col
        assert proj.nnz == expected_nnz_per_col * 64


@pytest.mark.parametrize(
    "full_dim, subspace_dim, touched_rows_only", [(100_000, 16, True), (100_000, 140, False), (1000, 64, False)]
)
def test_projection_matches_coalesced_csr(full_dim, subspace_dim, touched_rows_only):
    """Both cached layouts match the int64 CSR; the size counts what is held."""
    proj = SparseRandomProjection(full_dim=full_dim, subspace_dim=subspace_dim, seed=0)
    coords = torch.randn(5, subspace_dim, generator=torch.Generator().manual_seed(0))
    full = proj.project(coords)

    rows, vals = proj.columns(torch.arange(subspace_dim), torch.device("cpu"), torch.float32)
    cols = torch.arange(subspace_dim).repeat_interleave(rows.shape[1])
    reference = torch.sparse_coo_tensor(
        torch.stack([rows.reshape(-1), cols]), vals.reshape(-1), (full_dim, subspace_dim)
    )
    expected = (reference.coalesce().to_sparse_csr() @ coords.T.contiguous()).T

    assert torch.equal(full, expected)
    assert (proj._csr_rows is not None) is touched_rows_only
    assert proj._csr_matrix.col_indices().dtype == torch.int32
    held = [v for v in vars(proj).values() if isinstance(v, torch.Tensor)]
    parts = [t for h in held for t in ((h.crow_indices(), h.col_indices(), h.values()) if h.is_sparse_csr else (h,))]
    assert proj.memory_bytes == sum(t.untyped_storage().nbytes() for t in parts)


class TestJLTProperty:
    def test_distance_preservation(self):
        """Sparse projection approximately preserves distances."""
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=256, seed=42)

        torch.manual_seed(123)
        x1 = torch.randn(256)
        x2 = torch.randn(256)

        d_sub = torch.norm(x1 - x2).item()

        d_full = torch.norm(proj.project(x1) - proj.project(x2)).item()

        # JLT allows multiplicative distortion
        ratio = d_full / d_sub
        assert 0.5 < ratio < 2.0, f"Distance ratio {ratio} outside [0.5, 2.0]"

    def test_multiple_distance_preservation(self):
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

        mean_ratio = sum(ratios) / len(ratios)
        assert 0.7 < mean_ratio < 1.5, f"Mean distance ratio {mean_ratio} too far from 1"

    def test_zero_vector_maps_to_zero(self):
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        zero = torch.zeros(64)
        result = proj.project(zero)

        assert torch.allclose(result, torch.zeros(10000))

    def test_linearity(self):
        """Projection is linear: ``P(a x) == a P(x)``.

        Compared in the norm, not per element: the two forms reassociate the fp32
        sparse matmul, so near-cancellation entries have large relative error.
        """
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        x = torch.randn(64, generator=torch.Generator().manual_seed(0))
        scale = 3.14

        scaled_then_projected = proj.project(scale * x)
        projected_then_scaled = scale * proj.project(x)

        error = torch.linalg.vector_norm(scaled_then_projected - projected_then_scaled)
        assert error <= 1e-6 * torch.linalg.vector_norm(projected_then_scaled), f"linearity broken: {error}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_preserves_input_dtype(dtype):
    """The projection must come back at the coordinate dtype, not the buffer's."""
    proj = SparseRandomProjection(full_dim=1000, subspace_dim=32, seed=42)
    coords = torch.randn(32, dtype=dtype)
    full = proj.project(coords)
    assert full.dtype == dtype


class TestCUDA:
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "input_shape, expected_shape",
        [
            ((64,), (10000,)),
            ((8, 64), (8, 10000)),
        ],
    )
    def test_cuda_projection(self, input_shape, expected_shape):
        proj = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=42)

        coords = torch.randn(*input_shape, device="cuda")
        full = proj.project(coords)

        assert full.device.type == "cuda"
        assert full.shape == expected_shape

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_determinism(self):
        proj1 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=777)
        proj2 = SparseRandomProjection(full_dim=10000, subspace_dim=64, seed=777)

        coords = torch.randn(64, device="cuda")
        full1 = proj1.project(coords)
        full2 = proj2.project(coords)

        assert torch.allclose(full1, full2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_csr_build_holds_only_counted_bytes(self):
        """Building the CSR keeps nothing of the COO it sorted, and its peak stays near the CSR."""
        proj = SparseRandomProjection(full_dim=1_000_000, subspace_dim=1024, seed=0)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["requested_bytes.all.current"]
        torch.cuda.reset_peak_memory_stats()
        proj._get_csr_matrix(torch.device("cuda"), torch.float32)
        torch.cuda.synchronize()

        stats = torch.cuda.memory_stats()
        held = stats["requested_bytes.all.current"] - before
        peak = stats["requested_bytes.all.peak"] - before
        assert held - proj.memory_bytes < 2**12
        assert peak < 64 * proj.nnz


def test_repr():
    proj = SparseRandomProjection(full_dim=100000, subspace_dim=256, seed=42)
    r = repr(proj)

    assert "SparseRandomProjection" in r
    assert "full_dim=100000" in r
    assert "subspace_dim=256" in r
    assert "density=" in r
    assert "memory=" in r


def test_min_nnz_per_col():
    """At least 1 nonzero per column even at very low density."""
    proj = SparseRandomProjection(full_dim=10, subspace_dim=5, density=0.001, seed=42)

    assert proj._nnz_per_col >= 1

    coords = torch.randn(5)
    full = proj.project(coords)
    assert full.shape == (10,)


def test_warns_at_extreme_compression():
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


def test_sparse_projection_ignores_default_device():
    """The index/sign factories draw from a CPU generator, so without ``device="cpu"``
    ``torch.set_default_device`` would send them to the default device and the
    generator would mismatch.
    """
    target = torch.device("cuda" if torch.cuda.is_available() else "meta")
    torch.set_default_device(target)
    try:
        projection = SparseRandomProjection(full_dim=64, subspace_dim=8, seed=0)
        projection._init_sparse_matrix(torch.device("cpu"), torch.float32)
    finally:
        torch.set_default_device("cpu")

    assert projection._rows.shape == (8, projection._nnz_per_col)


def test_sparse_columns_dtype_change_invalidates_csr():
    projection = SparseRandomProjection(16, 8, density=0.5, seed=2)
    projection.project(torch.ones(8))  # populate the float32 CSR cache
    projection.columns(torch.arange(8), torch.device("cpu"), torch.float64)
    actual = projection.project(torch.ones(8, dtype=torch.float64))
    expected = SparseRandomProjection(16, 8, density=0.5, seed=2).project(torch.ones(8, dtype=torch.float64))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "kwargs", [{"full_dim": 0}, {"subspace_dim": -1}, {"density": 0}, {"density": float("nan")}, {"density": 2}]
)
def test_sparse_projection_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SparseRandomProjection(**({"full_dim": 16, "subspace_dim": 8} | kwargs))
