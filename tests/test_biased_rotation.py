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


def test_bfloat16_cpu_no_crash():
    R = torch.eye(3, dtype=torch.bfloat16)[None]
    b = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.bfloat16)
    out = apply_biased_rotation(R, b)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


if __name__ == "__main__":
    test_first_axis_aligns_with_bias()
    test_sign_ambiguous_bias_not_flipped()
    test_stays_proper_rotation()
    test_bfloat16_cpu_no_crash()
    print("ok")
