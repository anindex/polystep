"""Polytope templates and random rotations."""

import math

import torch
import pytest

from polystep.geometry import (
    get_orthoplex_vertices,
    get_simplex_vertices,
    get_cube_vertices,
    get_random_rotation_matrices,
    _QR_MAX_BATCH,
    get_rotation_matrix_2d,
)


class TestPolytopes:
    @pytest.mark.parametrize("dim", [2, 3, 4])
    @pytest.mark.parametrize(
        "gen, name",
        [(get_orthoplex_vertices, "orthoplex"), (get_simplex_vertices, "simplex"), (get_cube_vertices, "cube")],
    )
    def test_vertex_count(self, gen, name, dim):
        """The count is the per-step forward budget: P * V * K evaluations."""
        expected = {"orthoplex": 2 * dim, "simplex": dim + 1, "cube": 2**dim}[name]
        verts = gen(dim)
        assert verts.shape == (expected, dim)

    @pytest.mark.parametrize("dim", [2, 3, 5])
    @pytest.mark.parametrize("gen", [get_orthoplex_vertices, get_simplex_vertices, get_cube_vertices])
    def test_every_vertex_sits_at_the_requested_radius(self, gen, dim):
        """step_radius is a distance, so the templates must be unit-norm before scaling.

        A template whose vertices are not all at radius 1 makes the realized step depend
        on which vertex the transport picks, and rescales it silently on any polytope
        change.
        """
        verts = gen(dim, radius=2.5)
        norms = verts.norm(dim=-1)
        torch.testing.assert_close(norms, torch.full_like(norms, 2.5), rtol=1e-5, atol=1e-6)

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


class TestRotations:
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

    @pytest.mark.parametrize("dim", [3, 5, 8])
    def test_rotation_matches_reference_qr_sampler(self, dim):
        """The Householder subgroup sampler is distributionally identical to
        Mezzadri sign-corrected QR, which it replaced for speed.

        Moments alone would pass on samplers that are merely orthogonal, so
        compare full distributions: a two-sample KS statistic on ``tr(R)`` and on
        individual entries, against an old-vs-old control drawn from the
        reference sampler itself. A biased sampler separates from the reference
        while the control does not.
        """
        n = 20000

        def reference(batch, d, generator):
            Z = torch.randn(batch, d, d, generator=generator)
            Q, R = torch.linalg.qr(Z)
            diag = torch.diagonal(R, dim1=-2, dim2=-1)
            Q = Q * torch.where(diag == 0, torch.ones_like(diag), torch.sign(diag)).unsqueeze(-2)
            Q[:, :, 0] = Q[:, :, 0] * torch.where(torch.det(Q) < 0, -1.0, 1.0).unsqueeze(-1)
            return Q

        def ks(a, b):
            a = a.sort().values
            b = b.sort().values
            grid = torch.cat([a, b]).sort().values
            fa = torch.searchsorted(a, grid, right=True).float() / len(a)
            fb = torch.searchsorted(b, grid, right=True).float() / len(b)
            return (fa - fb).abs().max().item()

        actual = get_random_rotation_matrices(n, dim, generator=torch.Generator().manual_seed(1))
        expected = reference(n, dim, torch.Generator().manual_seed(2))
        control = reference(n, dim, torch.Generator().manual_seed(3))

        # KS 99.9% critical value for two samples of size n.
        crit = 1.95 * (2.0 / n) ** 0.5
        for name, stat in (
            ("trace", lambda Q: Q.diagonal(dim1=-2, dim2=-1).sum(-1)),
            ("R[0,0]", lambda Q: Q[:, 0, 0]),
            ("R[2,1]", lambda Q: Q[:, 2, 1]),
        ):
            d_new = ks(stat(actual), stat(expected))
            d_ctl = ks(stat(control), stat(expected))
            assert d_new < crit, f"{name}: KS={d_new:.5f} exceeds {crit:.5f} (control {d_ctl:.5f})"

    def test_rotation_is_haar_distributed(self):
        """The Householder subgroup algorithm produces Haar-distributed ``O(d)``,
        restricted to ``SO(d)`` by fixing the sign of the last coordinate.
        Verify both moments: ``E[R_ij] -> 0`` and ``Var[R_ij] -> 1/d``.
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


@pytest.mark.parametrize(
    "theta, expected",
    [
        (0.0, [[1.0, 0.0], [0.0, 1.0]]),
        (math.pi / 2, [[0.0, -1.0], [1.0, 0.0]]),
        (math.pi, [[-1.0, 0.0], [0.0, -1.0]]),
    ],
)
def test_rotation_matrix_2d_against_known_angles(theta, expected):
    """A swapped sign gives a reflection, which every orthonormality check accepts."""
    R = get_rotation_matrix_2d(torch.tensor(theta))
    torch.testing.assert_close(R, torch.tensor(expected), atol=1e-6, rtol=0)
    # Counter-clockwise: e_x must land on +e_y at a quarter turn, not -e_y.
    assert torch.det(R) == pytest.approx(1.0, abs=1e-6)


def test_small_batch_takes_qr_and_stays_haar_on_so():
    """At or below ``_QR_MAX_BATCH`` one batched QR replaces the reflection loop.

    The substitute has to be the same distribution, not merely orthogonal: assert it
    reproduces the Mezzadri sampler this file already uses as ground truth, and that it
    lands on ``SO(d)`` rather than ``O(d)``.
    """
    cap = _QR_MAX_BATCH[False]

    def reference(batch, d, generator):
        Z = torch.randn(batch, d, d, generator=generator)
        Q, R = torch.linalg.qr(Z)
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Q = Q * torch.where(diag == 0, torch.ones_like(diag), torch.sign(diag)).unsqueeze(-2)
        Q[:, :, 0] = Q[:, :, 0] * torch.where(torch.det(Q) < 0, -1.0, 1.0).unsqueeze(-1)
        return Q

    for dim in (3, 4, 8, 16):
        got = get_random_rotation_matrices(cap, dim, generator=torch.Generator().manual_seed(7))
        want = reference(cap, dim, torch.Generator().manual_seed(7))
        torch.testing.assert_close(got, want, atol=0, rtol=0)

        eye = torch.eye(dim).expand(cap, dim, dim)
        torch.testing.assert_close(got @ got.transpose(-1, -2), eye, atol=1e-5, rtol=0)
        # det = +1, not -1: a reflection passes every orthonormality check. The tolerance
        # is fp32's, not fp64's; .double() promotes the rounding rather than removing it.
        torch.testing.assert_close(torch.det(got.double()), torch.ones(cap, dtype=torch.float64), atol=1e-5, rtol=0)

    # Just past the cap the reflection loop takes over and must still be SO(d).
    over = get_random_rotation_matrices(cap + 1, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(torch.det(over.double()), torch.ones(cap + 1, dtype=torch.float64), atol=1e-5, rtol=0)
