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
        origin = torch.zeros(dim)
        verts = get_orthoplex_vertices(origin)
        assert verts.shape == (2 * dim, dim), \
            f"Expected ({2*dim}, {dim}), got {verts.shape}"

    @pytest.mark.parametrize("dim", [2, 3, 5, 10])
    def test_simplex_vertex_count(self, dim):
        """Simplex generates exactly dim+1 vertices."""
        origin = torch.zeros(dim)
        verts = get_simplex_vertices(origin)
        assert verts.shape == (dim + 1, dim), \
            f"Expected ({dim+1}, {dim}), got {verts.shape}"

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_cube_vertex_count(self, dim):
        """Cube generates exactly 2^dim vertices."""
        origin = torch.zeros(dim)
        verts = get_cube_vertices(origin)
        assert verts.shape == (2 ** dim, dim), \
            f"Expected ({2**dim}, {dim}), got {verts.shape}"

    def test_orthoplex_centered_at_origin(self):
        """Orthoplex centered at origin has mean close to zero."""
        dim = 5
        origin = torch.zeros(dim)
        verts = get_orthoplex_vertices(origin, radius=1.0)
        mean = verts.mean(dim=0)
        assert torch.allclose(mean, torch.zeros(dim), atol=1e-6), \
            f"Mean: {mean}"

    def test_orthoplex_offset_by_origin(self):
        """Orthoplex offset by non-zero origin has mean close to that origin."""
        origin = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        verts = get_orthoplex_vertices(origin, radius=1.0)
        mean = verts.mean(dim=0)
        assert torch.allclose(mean, origin, atol=1e-6), \
            f"Mean: {mean}, Expected: {origin}"

    def test_simplex_equidistant(self):
        """Simplex vertices are pairwise equidistant."""
        dim = 4
        origin = torch.zeros(dim)
        verts = get_simplex_vertices(origin, radius=1.0)

        # Compute all pairwise distances
        n_verts = verts.shape[0]
        dists = []
        for i in range(n_verts):
            for j in range(i + 1, n_verts):
                d = torch.norm(verts[i] - verts[j])
                dists.append(d.item())

        dists_t = torch.tensor(dists)
        # All distances should be approximately equal
        assert torch.allclose(dists_t, dists_t[0] * torch.ones_like(dists_t), atol=1e-5), \
            f"Distance range: [{dists_t.min():.6f}, {dists_t.max():.6f}]"

    def test_vertex_count_map(self):
        """POLYTOPE_NUM_VERTICES_MAP returns correct counts."""
        dim = 5
        assert POLYTOPE_NUM_VERTICES_MAP['orthoplex'](dim) == 2 * dim
        assert POLYTOPE_NUM_VERTICES_MAP['simplex'](dim) == dim + 1
        assert POLYTOPE_NUM_VERTICES_MAP['cube'](dim) == 2 ** dim


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------


class TestRotations:
    """Tests for rotation matrix generation."""

    @pytest.mark.parametrize("dim", [3, 5, 8])
    def test_rotation_matrix_orthogonal(self, dim):
        """Random rotation matrices are orthogonal: R^T R = I."""
        torch.manual_seed(42)
        gen = torch.Generator()
        gen.manual_seed(42)

        R = get_random_rotation_matrices(batch=4, dim=dim, generator=gen)
        eye = torch.eye(dim).unsqueeze(0).expand(4, -1, -1)
        RtR = torch.bmm(R.transpose(-1, -2), R)

        assert torch.allclose(RtR, eye, atol=1e-5), \
            f"Max orthogonality error: {(RtR - eye).abs().max():.8f}"

    @pytest.mark.parametrize("dim", [3, 5, 8])
    def test_rotation_matrix_det_positive(self, dim):
        """Random rotation matrices have determinant +1."""
        torch.manual_seed(42)
        gen = torch.Generator()
        gen.manual_seed(42)

        R = get_random_rotation_matrices(batch=4, dim=dim, generator=gen)
        dets = torch.det(R)

        assert torch.allclose(dets, torch.ones(4), atol=1e-4), \
            f"Determinants: {dets.tolist()}"

    def test_rotation_2d_uses_analytical(self):
        """Dim=2 rotation uses analytical formula and produces valid result."""
        gen = torch.Generator()
        gen.manual_seed(42)

        R = get_random_rotation_matrices(batch=3, dim=2, generator=gen)
        assert R.shape == (3, 2, 2)

        # Check orthogonality
        eye = torch.eye(2).unsqueeze(0).expand(3, -1, -1)
        RtR = torch.bmm(R.transpose(-1, -2), R)
        assert torch.allclose(RtR, eye, atol=1e-6)

        # Check det = +1
        dets = torch.det(R)
        assert torch.allclose(dets, torch.ones(3), atol=1e-6)

    def test_rotation_deterministic_with_generator(self):
        """Same generator seed produces identical rotation matrices."""
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        R1 = get_random_rotation_matrices(batch=5, dim=4, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(123)
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

        assert probes.shape == (batch, num_points, num_probe, dim), \
            f"Expected ({batch}, {num_points}, {num_probe}, {dim}), got {probes.shape}"

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
                assert torch.allclose(probes[b, p, 0], origin[b], atol=1e-6), \
                    f"Probe at scale=0 differs from origin at batch={b}, point={p}"

    def test_sampled_polytope_vertices_shapes(self):
        """get_sampled_polytope_vertices returns correct shapes."""
        dim = 4
        batch = 3
        num_probe = 5

        origin = torch.randn(batch, dim)
        probes = torch.linspace(0, 1, num_probe)
        polytope_verts = get_orthoplex_vertices(torch.zeros(dim))  # (2*dim, dim)

        gen = torch.Generator()
        gen.manual_seed(42)

        step_pts, probe_pts, rot_verts = get_sampled_polytope_vertices(
            origin, probes, polytope_verts,
            step_radius=1.0, probe_radius=2.0, generator=gen,
        )

        n_verts = 2 * dim
        assert step_pts.shape == (batch, n_verts, dim), \
            f"step_pts: expected ({batch}, {n_verts}, {dim}), got {step_pts.shape}"
        assert probe_pts.shape == (batch, n_verts, num_probe, dim), \
            f"probe_pts: expected ({batch}, {n_verts}, {num_probe}, {dim}), got {probe_pts.shape}"
        assert rot_verts.shape == (batch, n_verts, dim), \
            f"rot_verts: expected ({batch}, {n_verts}, {dim}), got {rot_verts.shape}"

    def test_sampled_polytope_deterministic_with_generator(self):
        """Same generator seed produces identical sampled vertices."""
        dim = 3
        batch = 2
        origin = torch.randn(batch, dim)
        probes = torch.linspace(0, 1, 4)
        polytope_verts = get_orthoplex_vertices(torch.zeros(dim))

        gen1 = torch.Generator()
        gen1.manual_seed(99)
        s1, p1, r1 = get_sampled_polytope_vertices(
            origin, probes, polytope_verts, generator=gen1,
        )

        gen2 = torch.Generator()
        gen2.manual_seed(99)
        s2, p2, r2 = get_sampled_polytope_vertices(
            origin, probes, polytope_verts, generator=gen2,
        )

        assert torch.equal(s1, s2), "Step points differ with same seed"
        assert torch.equal(p1, p2), "Probe points differ with same seed"
        assert torch.equal(r1, r2), "Rotated vertices differ with same seed"
