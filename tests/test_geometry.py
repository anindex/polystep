"""Unit tests for geometry module: polytopes, rotations, and probe points."""

import math

import torch
import pytest

from polystep.geometry import (
    get_orthoplex_vertices,
    get_simplex_vertices,
    get_cube_vertices,
    get_random_rotation_matrices,
    get_rotation_matrix_2d,
    get_probe_points,
    get_sampled_polytope_vertices,
    POLYTOPE_NUM_VERTICES_MAP,
)


# ---------------------------------------------------------------------------
# Polytope tests
# ---------------------------------------------------------------------------


class TestPolytopes:
    """Tests for polytope vertex generators."""

    @pytest.mark.parametrize("dim", [2, 3, 5, 10])
    def test_orthoplex_vertex_count(self, dim):
        """Orthoplex generates exactly 2*dim vertices."""
        verts = get_orthoplex_vertices(dim)
        assert verts.shape == (2 * dim, dim), f"Expected ({2 * dim}, {dim}), got {verts.shape}"

    @pytest.mark.parametrize("dim", [2, 3, 5, 10])
    def test_simplex_vertex_count(self, dim):
        """Simplex generates exactly dim+1 vertices."""
        verts = get_simplex_vertices(dim)
        assert verts.shape == (dim + 1, dim), f"Expected ({dim + 1}, {dim}), got {verts.shape}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_cube_vertex_count(self, dim):
        """Cube generates exactly 2^dim vertices."""
        verts = get_cube_vertices(dim)
        assert verts.shape == (2**dim, dim), f"Expected ({2**dim}, {dim}), got {verts.shape}"

    def test_orthoplex_centered_at_origin(self):
        """Orthoplex vertices are centered at the origin."""
        dim = 5
        verts = get_orthoplex_vertices(dim, radius=1.0)
        mean = verts.mean(dim=0)
        assert torch.allclose(mean, torch.zeros(dim), atol=1e-6), f"Mean: {mean}"

    def test_simplex_equidistant(self):
        """Simplex vertices are pairwise equidistant."""
        dim = 4
        verts = get_simplex_vertices(dim, radius=1.0)

        # Compute all pairwise distances
        n_verts = verts.shape[0]
        dists = []
        for i in range(n_verts):
            for j in range(i + 1, n_verts):
                d = torch.norm(verts[i] - verts[j])
                dists.append(d.item())

        dists_t = torch.tensor(dists)
        # All distances should be approximately equal
        assert torch.allclose(dists_t, dists_t[0] * torch.ones_like(dists_t), atol=1e-5), (
            f"Distance range: [{dists_t.min():.6f}, {dists_t.max():.6f}]"
        )

    def test_vertex_count_map(self):
        """POLYTOPE_NUM_VERTICES_MAP returns correct counts."""
        dim = 5
        assert POLYTOPE_NUM_VERTICES_MAP["orthoplex"](dim) == 2 * dim
        assert POLYTOPE_NUM_VERTICES_MAP["simplex"](dim) == dim + 1
        assert POLYTOPE_NUM_VERTICES_MAP["cube"](dim) == 2**dim


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------


class TestRotations:
    """Tests for rotation matrix generation."""

    @pytest.mark.parametrize("dim", [3, 5, 8])
    def test_rotation_matrix_is_in_so_d(self, dim):
        """Random rotation matrices are orthogonal (``R R^T = I``) and
        have determinant ``+1`` (SO(d), not O(d))."""
        gen = torch.Generator().manual_seed(42)

        R = get_random_rotation_matrices(batch=4, dim=dim, generator=gen)
        eye = torch.eye(dim).unsqueeze(0).expand(4, -1, -1)
        RtR = torch.bmm(R.transpose(-1, -2), R)
        assert torch.allclose(RtR, eye, atol=1e-5), f"Max orthogonality error: {(RtR - eye).abs().max():.8f}"

        dets = torch.det(R)
        assert torch.allclose(dets, torch.ones(4), atol=1e-4), f"Determinants: {dets.tolist()}"

    def test_rotation_is_haar_distributed(self):
        """Mezzadri 2007 sign-corrected QR produces Haar-distributed
        ``O(d)``, restricted to ``SO(d)`` by the det fix-up in
        :func:`get_random_rotation_matrices`. Verify both moments:
        ``E[R_ij] -> 0`` and ``Var[R_ij] -> 1/d``.
        """
        d = 8
        n = 8000
        gen = torch.Generator(device="cpu").manual_seed(0)
        R = get_random_rotation_matrices(batch=n, dim=d, generator=gen)
        assert R.shape == (n, d, d)

        # Empirical E[R_ij] -> 0 (Haar first moment). Std of mean over
        # n samples is ~ sqrt(1/d) / sqrt(n); at n=8000, d=8 a 6-sigma
        # upper bound is below 0.02.
        mean = R.mean(dim=0)
        assert mean.abs().max().item() < 0.02, f"E[R_ij] not centered: max |mean| = {mean.abs().max().item():.4f}"

        # Per-entry Var[R_ij] -> 1/d (Haar second moment).
        var = (R**2).mean(dim=0)
        expected = torch.full_like(var, 1.0 / d)
        rel_err = ((var - expected).abs() / expected).max().item()
        assert rel_err < 0.05, f"Var[R_ij] differs from 1/d by {rel_err * 100:.1f}%"

    def test_rotation_2d_uses_analytical(self):
        """Dim=2 rotation uses the analytical SO(2) path and produces
        valid orthogonal matrices with ``det = +1``."""
        gen = torch.Generator().manual_seed(42)
        R = get_random_rotation_matrices(batch=3, dim=2, generator=gen)
        assert R.shape == (3, 2, 2)

        eye = torch.eye(2).unsqueeze(0).expand(3, -1, -1)
        RtR = torch.bmm(R.transpose(-1, -2), R)
        assert torch.allclose(RtR, eye, atol=1e-6)
        assert torch.allclose(torch.det(R), torch.ones(3), atol=1e-6)

    def test_rotation_deterministic_with_generator(self):
        """Same generator seed produces identical rotation matrices."""
        gen1 = torch.Generator().manual_seed(123)
        R1 = get_random_rotation_matrices(batch=5, dim=4, generator=gen1)

        gen2 = torch.Generator().manual_seed(123)
        R2 = get_random_rotation_matrices(batch=5, dim=4, generator=gen2)
        assert torch.equal(R1, R2), "Same seed should produce identical rotations"


# ---------------------------------------------------------------------------
# Probe tests
# ---------------------------------------------------------------------------


class TestProbes:
    """Tests for probe point generation."""

    def test_probe_points_shape(self):
        """Probe points have correct output shape (batch, num_points, num_probe, dim)."""
        batch, num_points, dim, num_probe = 3, 6, 4, 5

        origin = torch.randn(batch, dim)
        directions = torch.randn(batch, num_points, dim)
        scales = torch.linspace(0, 1, num_probe)

        probes = get_probe_points(origin, directions, scales, probe_radius=2.0)

        assert probes.shape == (batch, num_points, num_probe, dim), (
            f"Expected ({batch}, {num_points}, {num_probe}, {dim}), got {probes.shape}"
        )

    def test_probe_points_at_zero_scale(self):
        """Probe points at scale=0 should be at the origin."""
        batch, num_points, dim = 2, 4, 3

        origin = torch.randn(batch, dim)
        directions = torch.randn(batch, num_points, dim)
        scales = torch.tensor([0.0, 0.5, 1.0])

        probes = get_probe_points(origin, directions, scales, probe_radius=2.0)

        # At scale=0 (index 0), probes should equal origin
        for b in range(batch):
            for p in range(num_points):
                assert torch.allclose(probes[b, p, 0], origin[b], atol=1e-6), (
                    f"Probe at scale=0 differs from origin at batch={b}, point={p}"
                )

    def test_sampled_polytope_vertices_shapes(self):
        """get_sampled_polytope_vertices returns correct shapes."""
        dim = 4
        batch = 3
        num_probe = 5

        origin = torch.randn(batch, dim)
        probes = torch.linspace(0, 1, num_probe)
        polytope_verts = get_orthoplex_vertices(dim)  # (2*dim, dim)

        gen = torch.Generator()
        gen.manual_seed(42)

        step_pts, probe_pts, rot_verts = get_sampled_polytope_vertices(
            origin,
            probes,
            polytope_verts,
            step_radius=1.0,
            probe_radius=2.0,
            generator=gen,
        )

        n_verts = 2 * dim
        assert step_pts.shape == (batch, n_verts, dim), (
            f"step_pts: expected ({batch}, {n_verts}, {dim}), got {step_pts.shape}"
        )
        assert probe_pts.shape == (batch, n_verts, num_probe, dim), (
            f"probe_pts: expected ({batch}, {n_verts}, {num_probe}, {dim}), got {probe_pts.shape}"
        )
        assert rot_verts.shape == (batch, n_verts, dim), (
            f"rot_verts: expected ({batch}, {n_verts}, {dim}), got {rot_verts.shape}"
        )

    def test_sampled_polytope_deterministic_with_generator(self):
        """Same generator seed produces identical sampled vertices."""
        dim = 3
        batch = 2
        origin = torch.randn(batch, dim)
        probes = torch.linspace(0, 1, 4)
        polytope_verts = get_orthoplex_vertices(dim)

        gen1 = torch.Generator()
        gen1.manual_seed(99)
        s1, p1, r1 = get_sampled_polytope_vertices(
            origin,
            probes,
            polytope_verts,
            generator=gen1,
        )

        gen2 = torch.Generator()
        gen2.manual_seed(99)
        s2, p2, r2 = get_sampled_polytope_vertices(
            origin,
            probes,
            polytope_verts,
            generator=gen2,
        )

        assert torch.equal(s1, s2), "Step points differ with same seed"
        assert torch.equal(p1, p2), "Probe points differ with same seed"
        assert torch.equal(r1, r2), "Rotated vertices differ with same seed"
