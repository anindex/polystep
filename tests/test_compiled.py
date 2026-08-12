"""Tests for compilation infrastructure: fallback, numerical equivalence."""

import warnings

import pytest
import torch
import torch._dynamo

from polystep._compiled import (
    CompiledFunctions,
    _barycentric_projection,
    _compute_probe_points,
    _fused_softmax_project,
    _rotate_and_translate,
    _sinkhorn_iteration,
)


def test_compile_false_stores_eager_functions():
    """compile=False stores raw eager functions."""
    cf = CompiledFunctions(compile=False)
    assert cf.sinkhorn_iter is _sinkhorn_iteration
    assert cf.rotate_and_translate is _rotate_and_translate
    assert cf.barycentric_projection is _barycentric_projection
    assert cf.compute_probe_points is _compute_probe_points


def test_per_function_fallback_independence(monkeypatch):
    """One function's compile failure does not block others."""
    original_compile = torch.compile

    def selective_compile(fn, *, fullgraph=True, mode="reduce-overhead", **kw):
        if getattr(fn, "__name__", "") == "_sinkhorn_iteration":
            raise RuntimeError("Simulated compile failure for sinkhorn_iteration")
        return original_compile(fn, fullgraph=fullgraph, mode=mode, **kw)

    monkeypatch.setattr(torch, "compile", selective_compile)

    # CompiledFunctions only attempts compilation when CUDA appears available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cf = CompiledFunctions(compile=True)

    assert cf.sinkhorn_iter is _sinkhorn_iteration, "sinkhorn_iter should be the original eager function after fallback"

    assert cf.rotate_and_translate is not _rotate_and_translate
    assert cf.barycentric_projection is not _barycentric_projection

    fail_warnings = [x for x in w if "sinkhorn_iteration" in str(x.message)]
    assert len(fail_warnings) >= 1, "Expected warning about sinkhorn_iteration failure"


def _make_sinkhorn_args(device):
    """Inputs for _sinkhorn_iteration. Small: the assertion is about the traced graph,
    not the arithmetic, and a large kernel only lengthens the trace."""
    n, m = 40, 30
    f = torch.randn(n, device=device)
    g = torch.randn(m, device=device)
    log_K = torch.randn(n, m, device=device)
    log_a = torch.log(torch.ones(n, device=device) / n)
    log_b = torch.log(torch.ones(m, device=device) / m)
    eps = 0.1
    return (f, g, log_K, log_a, log_b, eps)


def _make_rotate_args(device):
    """Create input tensors for _rotate_and_translate on the given device."""
    B, d, V = 10, 20, 40
    raw = torch.randn(B, d, d, device=device)
    Q, _ = torch.linalg.qr(raw)
    rot_mats = Q
    polytope_verts = torch.randn(V, d, device=device)
    origin = torch.randn(B, d, device=device)
    step_radius = 0.5
    return (rot_mats, polytope_verts, origin, step_radius)


def _make_barycentric_args(device):
    """Create input tensors for _barycentric_projection on the given device."""
    B, V, d = 10, 40, 20
    transport = torch.softmax(torch.randn(B, V, device=device), dim=-1)
    X_vertices = torch.randn(B, V, d, device=device)
    return (transport, X_vertices)


def _make_probe_args(device):
    """Create input tensors for _compute_probe_points on the given device."""
    B, num_points, d = 10, 40, 20
    origin = torch.randn(B, d, device=device)
    directions = torch.randn(B, num_points, d, device=device)
    scales = torch.linspace(0.2, 0.8, 3, device=device)
    probe_radius = 1.0
    return (origin, directions, scales, probe_radius)


def _make_fused_softmax_args(device, P=10, V=8, dim=4, seed=42):
    """Create input tensors for _fused_softmax_project on the given device."""
    torch.manual_seed(seed)
    cost_matrix = torch.rand(P, V, device=device) + 0.01
    epsilon = 0.1
    a = torch.ones(P, device=device) / P
    polytope_verts = torch.randn(V, dim, device=device)
    raw = torch.randn(P, dim, dim, device=device)
    Q, _ = torch.linalg.qr(raw)
    rot_mats = Q
    step_radius = 0.5
    X = torch.randn(P, dim, device=device)
    return cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X


class TestFusedSoftmaxProjectEquivalence:
    """The fused kernel must equal the two-step softmax solve plus barycentric projection."""

    def test_fused_matches_two_step_path(self):
        cost, eps, a, verts, rot, step_r, X = _make_fused_softmax_args(torch.device("cpu"))
        X_fused, transport = _fused_softmax_project(cost, eps, a, verts, rot, step_r, X, scale_cost_mean=False)

        # Two-step reference: per-row softmax weighting, then project onto the
        # rotated, translated vertices.
        W = torch.softmax(-(cost - cost.amin(dim=-1, keepdim=True)) / eps, dim=-1)
        X_vertices, _ = _rotate_and_translate(rot, verts, X, step_r)
        X_ref = _barycentric_projection(W * a.unsqueeze(-1), X_vertices)

        torch.testing.assert_close(X_fused, X_ref, atol=1e-5, rtol=1e-5)
        # The transport rows must carry the source marginal.
        torch.testing.assert_close(transport.sum(dim=-1), a, atol=1e-6, rtol=1e-6)

    def test_no_scaling_leaves_the_cost_untouched(self):
        cost, eps, a, verts, rot, step_r, X = _make_fused_softmax_args(torch.device("cpu"))
        _, t_raw = _fused_softmax_project(cost, eps, a, verts, rot, step_r, X, scale_cost_mean=False)
        _, t_scaled = _fused_softmax_project(cost, eps, a, verts, rot, step_r, X, scale_cost_mean=True)
        # Dividing the cost by its mean changes the effective temperature, so the two
        # plans must differ; identical output would mean the flag does nothing.
        assert not torch.allclose(t_raw, t_scaled, atol=1e-6)


@pytest.mark.parametrize(
    "fn,make_args",
    [
        (_sinkhorn_iteration, _make_sinkhorn_args),
        (_rotate_and_translate, _make_rotate_args),
        (_barycentric_projection, _make_barycentric_args),
        (_compute_probe_points, _make_probe_args),
        (_fused_softmax_project, _make_fused_softmax_args),
    ],
    ids=[
        "sinkhorn_iteration",
        "rotate_and_translate",
        "barycentric_projection",
        "compute_probe_points",
        "fused_softmax_project",
    ],
)
def test_compiles_without_graph_breaks(fn, make_args):
    """Every hot-path kernel must compile as a single graph.

    ``fullgraph=True`` raises on any graph break, so compiling and running is the
    check. This runs on CPU, where a break is a break regardless of device."""
    args = make_args(torch.device("cpu"))
    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    got = compiled(*args)
    want = fn(*args)
    if isinstance(want, tuple):
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
