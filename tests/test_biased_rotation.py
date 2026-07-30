"""apply_biased_rotation must point the first search axis along the requested
bias, stay a proper rotation, and work in bf16.

QR fixes a column only up to sign, so without a correction the biased axis could
come back negated and steer the search toward ascent instead of descent.
"""

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


def test_bfloat16_cpu_still_returns_a_proper_rotation():
    R = torch.eye(3, dtype=torch.bfloat16)[None]
    b = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16)
    out = apply_biased_rotation(R, b)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    # Orthonormal with det +1, or the frame it defines is not a rotation.
    m = out.float()[0]
    assert torch.allclose(m @ m.T, torch.eye(3), atol=5e-2)
    assert torch.det(m) > 0


def test_bfloat16_actually_applies_the_bias():
    """bf16 must keep the bias, not silently fall back to the unbiased rotation.

    Gram-Schmidt in bf16 leaves the Gram matrix ~5.7e-3 off identity, so a fixed
    1e-3 orthonormality tolerance rejected every particle at dim >= 4 and
    biased_rotation became a no-op under mixed_precision.
    """
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


def test_collapsed_column_falls_back_to_an_orthonormal_frame():
    """A bias parallel to an existing axis must not yield duplicate columns.

    Gram-Schmidt annihilates the axis the bias replaced, and restoring the original
    column leaves it non-orthogonal to the bias. R=I, bias=e_k produced a finite
    singular frame (two identical columns, det 0) that the finiteness guard let past,
    so two polytope vertices coincided.
    """
    for dim in (2, 3, 8):
        for k in range(dim):
            bias = torch.zeros(1, dim)
            bias[0, k] = 1.0
            Q = apply_biased_rotation(torch.eye(dim).unsqueeze(0), bias)[0]
            gram = Q.T @ Q
            assert torch.allclose(gram, torch.eye(dim), atol=1e-4), f"dim={dim} k={k}:\n{gram}"
            assert torch.det(Q) > 0.9, f"dim={dim} k={k}: det={torch.det(Q)}"


if __name__ == "__main__":
    test_first_axis_aligns_with_bias()
    test_sign_ambiguous_bias_not_flipped()
    test_stays_proper_rotation()
    test_bfloat16_cpu_still_returns_a_proper_rotation()
    test_collapsed_column_falls_back_to_an_orthonormal_frame()
    print("ok")
