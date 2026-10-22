"""apply_biased_rotation must land the vertex nearest the bias on it, stay a
proper rotation, and work in bf16.
"""

import pytest
import torch

from polystep.geometry import (
    POLYTOPE_MAP,
    apply_biased_rotation,
    get_random_rotation_matrices,
    get_simplex_vertices,
)


def _vertex_reach(frames, verts, bias):
    return torch.einsum("pji,vi,pj->pv", frames, verts, bias / bias.norm(dim=-1, keepdim=True))


def test_biased_frame_lands_vertex_on_bias():
    gen = torch.Generator().manual_seed(0)
    verts = get_simplex_vertices(2, dtype=torch.float64)
    R = get_random_rotation_matrices(600, 2, dtype=torch.float64, generator=gen)
    b = torch.randn(600, 2, dtype=torch.float64, generator=gen)
    reach = _vertex_reach(apply_biased_rotation(R, b, verts), verts, b)
    assert (reach.amax(dim=1) - 1).abs().max() < 1e-12
    assert torch.bincount(reach.argmax(dim=1), minlength=3).min() >= 150


@pytest.mark.parametrize("polytope", sorted(POLYTOPE_MAP))
def test_stays_proper_rotation(polytope):
    torch.manual_seed(1)
    R = get_random_rotation_matrices(6, 4)
    b = torch.randn(6, 4)
    b = b / b.norm(dim=1, keepdim=True)
    out = apply_biased_rotation(R, b, POLYTOPE_MAP[polytope](4))
    eye = torch.eye(4).expand(6, 4, 4)
    assert torch.allclose(torch.einsum("bij,bik->bjk", out, out), eye, atol=1e-5)
    assert torch.allclose(torch.det(out), torch.ones(6), atol=1e-5)


def test_bf16_cpu_returns_proper_rotation():
    R = torch.eye(3, dtype=torch.bfloat16)[None]
    b = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16)
    out = apply_biased_rotation(R, b, get_simplex_vertices(3))
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    # Orthonormal with det +1, or the frame it defines is not a rotation.
    m = out.float()[0]
    assert torch.allclose(m @ m.T, torch.eye(3), atol=5e-2)
    assert torch.det(m) > 0


def test_bf16_applies_bias():
    """No silent fallback to the unbiased rotation."""
    for dim in (2, 4, 8, 16):
        torch.manual_seed(dim)
        R = get_random_rotation_matrices(64, dim, dtype=torch.float32)
        b = torch.randn(64, dim)
        verts = get_simplex_vertices(dim)
        out = apply_biased_rotation(R.bfloat16(), b.bfloat16(), verts)
        assert out.dtype == torch.bfloat16
        rate = (_vertex_reach(out.float(), verts, b).amax(dim=1) > 0.99).float().mean().item()
        assert rate >= 0.9, f"dim={dim}: bf16 landed only {rate:.3f}"


def test_nonfinite_bias_row_keeps_own_frame():
    torch.manual_seed(2)
    R = get_random_rotation_matrices(4, 3)
    b = torch.randn(4, 3)
    b = b / b.norm(dim=1, keepdim=True)
    b[1, 0] = float("nan")
    b[2, 2] = float("inf")
    verts = get_simplex_vertices(3)
    out = apply_biased_rotation(R, b, verts)
    assert torch.equal(out[1:3], R[1:3])
    assert torch.allclose(_vertex_reach(out[[0, 3]], verts, b[[0, 3]]).amax(dim=1), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("pdim", [2, 3, 8])
def test_vanishing_bias_keeps_frame_orthonormal(pdim):
    """A vanishing descent direction must leave the frame orthonormal.

    Normalizing by ``norm.clamp(min=1e-10)`` would aim a vertex at rounding
    noise, so such a particle keeps its unbiased rotation instead.
    """
    P = 32
    gen = torch.Generator().manual_seed(7)
    rot_mats = get_random_rotation_matrices(P, pdim, device="cpu", dtype=torch.float32, generator=gen)

    # Half the particles have collapsed, half still carry a usable heading.
    bias_dir = torch.randn(P, pdim, generator=gen)
    bias_dir[::2] *= 1e-12

    biased = apply_biased_rotation(rot_mats, bias_dir, get_simplex_vertices(pdim))

    gram = biased.transpose(-1, -2) @ biased
    assert torch.allclose(gram, torch.eye(pdim).expand(P, -1, -1), atol=1e-4)
    assert torch.allclose(torch.det(biased), torch.ones(P), atol=1e-3)
    # Collapsed particles fall back to exactly the rotation they came in with.
    assert torch.equal(biased[::2], rot_mats[::2])


def test_biased_rotation_1d_returns_identity():
    """SO(1) = {[[1]]}; a 1D biased rotation cannot represent a sign flip (that
    is a reflection), so it must return identity, not flip the aligned axis back."""
    out = apply_biased_rotation(torch.ones(1, 1, 1), torch.tensor([[-1.0]]), get_simplex_vertices(1))
    assert torch.allclose(out, torch.ones(1, 1, 1)), out
    assert torch.det(out[0]).item() > 0
