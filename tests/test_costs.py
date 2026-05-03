"""Unit tests for cost matrix computation and scaling."""
import torch
from polystep.costs import compute_cost_matrix, scale_cost_matrix


def test_compute_cost_matrix_shape():
    """Output shape should be (batch, num_vertices)."""
    batch, verts, probes, dim = 3, 5, 2, 4
    X = torch.randn(batch, verts, probes, dim)
    C = compute_cost_matrix(lambda x: x.sum(dim=-1), X)
    assert C.shape == (batch, verts)


def test_compute_cost_matrix_averages_probes():
    """Cost should be the mean over probe dimension."""
    batch, verts, probes, dim = 2, 3, 4, 2
    X = torch.ones(batch, verts, probes, dim)
    # Identity sum: each point sums to dim=2, mean over probes = 2
    C = compute_cost_matrix(lambda x: x.sum(dim=-1), X)
    assert torch.allclose(C, torch.full((batch, verts), 2.0))


def test_compute_cost_matrix_chunked_matches_unchunked():
    """Chunked evaluation should produce same result as unchunked."""
    batch, verts, probes, dim = 3, 5, 2, 4
    X = torch.randn(batch, verts, probes, dim)
    fn = lambda x: x.norm(dim=-1)
    C_full = compute_cost_matrix(fn, X)
    C_chunked = compute_cost_matrix(fn, X, chunk_size=4)
    assert torch.allclose(C_full, C_chunked, atol=1e-6)


def test_scale_cost_matrix_none():
    """None scaling should return unchanged matrix."""
    C = torch.randn(3, 5)
    assert torch.equal(scale_cost_matrix(C, None), C)


def test_scale_cost_matrix_mean():
    """Mean scaling divides by the mean."""
    C = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    scaled = scale_cost_matrix(C, 'mean')
    expected = C / C.mean()
    assert torch.allclose(scaled, expected)


def test_scale_cost_matrix_max():
    """Max scaling divides by the max."""
    C = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    scaled = scale_cost_matrix(C, 'max_cost')
    expected = C / 8.0
    assert torch.allclose(scaled, expected)


def test_scale_cost_matrix_float():
    """Float scaling divides by that float."""
    C = torch.tensor([[2.0, 4.0]])
    scaled = scale_cost_matrix(C, 5.0)
    assert torch.allclose(scaled, C / 5.0)


def test_scale_cost_matrix_zero_safe():
    """Scaling should not produce inf on zero-mean matrix."""
    C = torch.zeros(3, 5)
    scaled = scale_cost_matrix(C, 'mean')
    assert torch.isfinite(scaled).all()
