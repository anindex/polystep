"""apply_biased_rotation must point the first search axis along the requested
bias, stay a proper rotation, and work in bf16. The Householder reflection lands
the first axis on the bias only up to sign, so the columns need a sign fix.
"""

import pytest
import torch

from polystep.geometry import apply_biased_rotation, get_random_rotation_matrices


def test_first_axis_aligns_with_bias():
    torch.manual_seed(0)
    R = get_random_rotation_matrices(8, 3)
    b = torch.randn(8, 3)
    b = b / b.norm(dim=1, keepdim=True)
    out = apply_biased_rotation(R, b)
    # Column 0 must equal the requested bias, not its negation.
    assert torch.allclose(out[:, :, 0], b, atol=1e-5)


def test_sign_ambiguous_bias_not_flipped():
    # eye rotation with a negative-x bias is the QR sign-flip trigger.
    out = apply_biased_rotation(torch.eye(2)[None], torch.tensor([[-1.0, 0.0]]))
    assert (out[0, :, 0] * torch.tensor([-1.0, 0.0])).sum() > 0


def test_stays_proper_rotation():
    torch.manual_seed(1)
    R = get_random_rotation_matrices(6, 4)
    b = torch.randn(6, 4)
    b = b / b.norm(dim=1, keepdim=True)
    out = apply_biased_rotation(R, b)
    eye = torch.eye(4).expand(6, 4, 4)
    assert torch.allclose(torch.einsum("bij,bik->bjk", out, out), eye, atol=1e-5)
    assert torch.allclose(torch.det(out), torch.ones(6), atol=1e-5)


def test_bf16_cpu_returns_proper_rotation():
    R = torch.eye(3, dtype=torch.bfloat16)[None]
    b = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16)
    out = apply_biased_rotation(R, b)
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
        b = b / b.norm(dim=1, keepdim=True)

        def kept(rot, bias):
            out = apply_biased_rotation(rot, bias)
            return ((out[:, :, 0].float() * b).sum(dim=1) > 0.99).float().mean().item()

        rate32 = kept(R, b)
        rate16 = kept(R.bfloat16(), b.bfloat16())
        assert apply_biased_rotation(R.bfloat16(), b.bfloat16()).dtype == torch.bfloat16
        assert rate32 > 0.9, f"dim={dim}: fp32 baseline only {rate32:.3f}"
        assert rate16 >= rate32 - 0.02, f"dim={dim}: bf16 kept {rate16:.3f} vs fp32 {rate32:.3f}"


def test_axis_aligned_bias_is_applied():
    """A bias along an existing axis must land in column 0, not be dropped."""
    for dim in (2, 3, 8):
        for k in range(dim):
            for sign in (1.0, -1.0):
                bias = torch.zeros(1, dim)
                bias[0, k] = sign
                Q = apply_biased_rotation(torch.eye(dim).unsqueeze(0), bias)[0]
                assert torch.equal(Q[:, 0], bias[0]), f"dim={dim} k={k} sign={sign}"
                assert torch.allclose(Q.T @ Q, torch.eye(dim), atol=1e-6)
                assert torch.allclose(torch.det(Q), torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize("dim", [2, 8, 32])
def test_near_aligned_bias_keeps_column_zero_fp32(dim):
    gen = torch.Generator().manual_seed(dim)
    R = get_random_rotation_matrices(512, dim, generator=gen)
    sign = torch.where(torch.rand(512, 1, generator=gen) < 0.5, -1.0, 1.0)
    b = sign * R[:, :, 0] + 1e-6 * torch.randn(512, dim, generator=gen)
    out = apply_biased_rotation(R, b)
    b = b / b.norm(dim=1, keepdim=True)
    assert (out[:, :, 0] - b).abs().max() < 1e-5
    assert (out.mT @ out - torch.eye(dim)).abs().max() < 1e-5
    assert (torch.det(out) - 1).abs().max() < 1e-5


def test_complement_axes_stay_haar():
    R = get_random_rotation_matrices(20000, 3, generator=torch.Generator().manual_seed(0))
    out = apply_biased_rotation(R, torch.tensor([[1.0, 0.0, 0.0]]).expand(20000, 3))
    x = out[:, 1:, 1].double()
    assert x.mean(dim=0).abs().max() < 0.02
    assert (x.T @ x / len(x) - torch.eye(2, dtype=torch.float64) / 2).abs().max() < 0.02


def test_nonfinite_bias_row_keeps_own_frame():
    torch.manual_seed(2)
    R = get_random_rotation_matrices(4, 3)
    b = torch.randn(4, 3)
    b = b / b.norm(dim=1, keepdim=True)
    b[1, 0] = float("nan")
    b[2, 2] = float("inf")
    out = apply_biased_rotation(R, b)
    assert torch.equal(out[1:3], R[1:3])
    assert torch.allclose(out[[0, 3], :, 0], b[[0, 3]], atol=1e-5)


@pytest.mark.parametrize("pdim", [2, 3, 8])
def test_vanishing_bias_keeps_frame_orthonormal(pdim):
    """A vanishing descent direction must leave the frame orthonormal.

    Normalizing by ``norm.clamp(min=1e-10)`` would aim the first axis at rounding
    noise, so such a particle keeps its unbiased rotation instead.
    """
    P = 32
    gen = torch.Generator().manual_seed(7)
    rot_mats = get_random_rotation_matrices(P, pdim, device="cpu", dtype=torch.float32, generator=gen)

    # Half the particles have collapsed, half still carry a usable heading.
    bias_dir = torch.randn(P, pdim, generator=gen)
    bias_dir[::2] *= 1e-12

    biased = apply_biased_rotation(rot_mats, bias_dir)

    gram = biased.transpose(-1, -2) @ biased
    assert torch.allclose(gram, torch.eye(pdim).expand(P, -1, -1), atol=1e-4)
    assert torch.allclose(torch.det(biased), torch.ones(P), atol=1e-3)
    # Collapsed particles fall back to exactly the rotation they came in with.
    assert torch.equal(biased[::2], rot_mats[::2])


def test_biased_rotation_1d_returns_identity():
    """SO(1) = {[[1]]}; a 1D biased rotation cannot represent a sign flip (that
    is a reflection), so it must return identity, not flip the aligned axis back."""
    out = apply_biased_rotation(torch.ones(1, 1, 1), torch.tensor([[-1.0]]))
    assert torch.allclose(out, torch.ones(1, 1, 1)), out
    assert torch.det(out[0]).item() > 0
