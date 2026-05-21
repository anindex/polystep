"""Regression tests for polytope rotations and sparse projection.

- ``get_random_rotation_matrices`` is Haar-distributed: the
  Mezzadri sign correction gives ``E[R_ij] ~= 0`` and ``Var[R_ij]
  ~= 1/d`` with ``det = +1``.
- ``dp = 2`` analytic path samples uniformly on ``SO(2)``.
- ``SparseRandomProjection``: per-coordinate variance ~ 1, extreme
  compression ratio (< 1e-5) emits a UserWarning, projection matrix
  is fixed across forward calls within a session.
"""
from __future__ import annotations

import math
import warnings

import pytest
import torch

from polystep import (
    SparseRandomProjection,
    get_random_rotation_matrices,
    get_orthoplex_vertices,
)


# ---------------------------------------------------------------------------
# Mezzadri sign-corrected QR Haar test
# ---------------------------------------------------------------------------


def test_mezzadri_rotation_is_haar_distributed():
    """Mezzadri 2007: QR with sign correction produces Haar-distributed
    O(d). This is further restricted to SO(d) by flipping the first
    column when det = -1 (geometry.py:192-196). Test both moments:
    E[R_ij] -> 0 and Cov[R] -> I/d for diagonal entries.
    """
    d = 8
    n = 8000
    g = torch.Generator(device="cpu").manual_seed(0)
    R = get_random_rotation_matrices(batch=n, dim=d, generator=g)
    assert R.shape == (n, d, d)

    # All rotations are exactly orthogonal: R R^T = I
    gram = torch.einsum("nij,nkj->nik", R, R)
    eye = torch.eye(d).expand(n, d, d)
    max_orthogonality_err = (gram - eye).abs().max().item()
    assert max_orthogonality_err < 1e-5

    # All rotations have det = +1 (SO(d), not just O(d))
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(n), atol=1e-4)

    # Empirical E[R_ij] -> 0 (Haar moment 1)
    mean = R.mean(dim=0)  # (d, d)
    # Std of mean over n samples is ~ sqrt(1/d) / sqrt(n).
    # 6-sigma upper bound at n=20000, d=8 ~ 0.005.
    assert mean.abs().max().item() < 0.02, (
        f"E[R_ij] not centered: max |mean| = {mean.abs().max().item():.4f}"
    )

    # Diagonal-of-cov per element: Var[R_ij] ~ 1/d
    var = (R ** 2).mean(dim=0)
    expected = torch.full_like(var, 1.0 / d)
    rel_err = ((var - expected).abs() / expected).max().item()
    assert rel_err < 0.05, (
        f"Var[R_ij] differs from 1/d by {rel_err*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# dp=2 analytic SO(2) rotation
# ---------------------------------------------------------------------------


def test_dp2_rotations_are_proper_so2():
    """dp=2 takes the analytic path (geometry.py:159-161) which samples
    angle ~ U(0, 2*pi). All 2x2 rotations must be in SO(2): det = +1
    and exactly orthogonal."""
    d = 2
    n = 5000
    g = torch.Generator(device="cpu").manual_seed(0)
    R = get_random_rotation_matrices(batch=n, dim=d, generator=g)

    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(n), atol=1e-5), (
        f"dp=2 rotations should have det=+1; got dets in "
        f"[{dets.min().item():.3e}, {dets.max().item():.3e}]"
    )

    gram = torch.einsum("nij,nkj->nik", R, R)
    assert torch.allclose(gram, torch.eye(2).expand(n, 2, 2), atol=1e-5)


def test_orthoplex_vertices_for_dp2():
    """dp=2 orthoplex has 4 vertices: +/-e_i. Verify shape and that
    the 4 vertices are pairwise orthogonal in the +/- pairs."""
    verts = get_orthoplex_vertices(2, radius=1.0)
    assert verts.shape == (4, 2)
    expected = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
    ])
    assert torch.allclose(verts, expected)


# ---------------------------------------------------------------------------
# Sparse JL: variance, warning, fixed seed within session
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Sparse invariant checks:UserWarning")
def test_sparse_jl_unit_variance_per_column():
    """SparseRandomProjection.project_transpose computes coords = P^T @ full
    with P (full_dim, subspace_dim). Each column of P has nnz_per_col
    Rademacher entries scaled by 1/sqrt(nnz_per_col), so for an input
    full ~ N(0, I), the projected coordinate has variance ~1.
    """
    full_dim = 10000
    subspace_dim = 256
    proj = SparseRandomProjection(full_dim=full_dim, subspace_dim=subspace_dim, seed=0)

    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(full_dim, generator=g)
    y = proj.project_transpose(x)
    assert y.shape == (subspace_dim,)
    sample_var = y.var().item()
    assert 0.5 < sample_var < 2.0, (
        f"projected coordinate variance off: got {sample_var:.3f}, expected ~1.0"
    )


def test_sparse_jl_warns_at_extreme_compression():
    """Subspace ratio below 1e-5 triggers a UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SparseRandomProjection(full_dim=10_000_000, subspace_dim=64, seed=0)

    msgs = [str(w.message).lower() for w in caught]
    assert any(
        "compression" in m or "below the empirical floor" in m
        for m in msgs
    ), f"expected extreme-compression warning; got {msgs}"


def test_sparse_jl_projection_is_fixed_within_session():
    """Two consecutive projections of the same input must return the
    same output: the projection matrix is cached at first use, not
    re-sampled per call."""
    proj = SparseRandomProjection(full_dim=2048, subspace_dim=64, seed=0)
    coords = torch.randn(64)
    y1 = proj.project(coords)
    y2 = proj.project(coords)
    assert torch.equal(y1, y2)
