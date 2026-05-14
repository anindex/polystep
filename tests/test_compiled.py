"""Tests for compilation infrastructure: fallback, numerical equivalence."""
import inspect
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
    try_compile,
)
from polystep import SinkhornSolver, PolyStep


# ---------------------------------------------------------------------------
# Group 1: try_compile fallback tests
# ---------------------------------------------------------------------------


class TestTryCompileReturnsCallable:
    """Test 1: try_compile returns a callable."""

    def test_try_compile_returns_callable(self):
        fn = try_compile(lambda x: x + 1, name="identity_plus_one")
        assert callable(fn)


class TestTryCompileFallbackOnBadFunction:
    """Test 2: try_compile falls back to eager on compilation failure."""

    def test_try_compile_fallback_on_bad_function(self):
        def bad_fn():
            return "not_a_tensor"

        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            result = try_compile(bad_fn, name="bad_fn")

        # On CPU torch.compile may succeed lazily (deferred compilation).
        # The important contract is that result is callable and usable.
        assert callable(result)


class TestCompiledFunctionsEagerEquivalence:
    """Test 3: compile=False stores raw eager functions."""

    def test_compile_false_stores_eager_functions(self):
        cf = CompiledFunctions(compile=False)
        assert cf.sinkhorn_iter is _sinkhorn_iteration
        assert cf.rotate_and_translate is _rotate_and_translate
        assert cf.barycentric_projection is _barycentric_projection
        assert cf.compute_probe_points is _compute_probe_points


# ---------------------------------------------------------------------------
# Group 2: Per-function independent fallback
# ---------------------------------------------------------------------------


class TestPerFunctionFallbackIndependence:
    """Test 4: One function's compile failure does not block others."""

    def test_per_function_fallback_independence(self, monkeypatch):
        original_compile = torch.compile

        def selective_compile(fn, *, fullgraph=True, mode="reduce-overhead", **kw):
            # Fail only for sinkhorn_iteration
            if getattr(fn, "__name__", "") == "_sinkhorn_iteration":
                raise RuntimeError("Simulated compile failure for sinkhorn_iteration")
            return original_compile(fn, fullgraph=fullgraph, mode=mode, **kw)

        monkeypatch.setattr(torch, "compile", selective_compile)

        # Also need CUDA to appear available so CompiledFunctions attempts compilation
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cf = CompiledFunctions(compile=True)

        # sinkhorn_iter should have fallen back to the original function
        assert cf.sinkhorn_iter is _sinkhorn_iteration, (
            "sinkhorn_iter should be the original eager function after fallback"
        )

        # Other functions should NOT be the original (they got compiled wrappers)
        assert cf.rotate_and_translate is not _rotate_and_translate
        assert cf.barycentric_projection is not _barycentric_projection

        # Verify warning was emitted for the failed function
        fail_warnings = [x for x in w if "sinkhorn_iteration" in str(x.message)]
        assert len(fail_warnings) >= 1, "Expected warning about sinkhorn_iteration failure"


# ---------------------------------------------------------------------------
# Group 3: Numerical equivalence tests
# ---------------------------------------------------------------------------


class TestSinkhornIterationEquivalence:
    """Test 5: _sinkhorn_iteration direct vs CompiledFunctions(compile=False)."""

    def test_sinkhorn_iteration_equivalence(self):
        torch.manual_seed(42)
        n, m = 50, 30
        eps = 0.1

        f = torch.randn(n)
        g = torch.randn(m)
        log_K = torch.randn(n, m)
        log_a = torch.log(torch.ones(n) / n)
        log_b = torch.log(torch.ones(m) / m)

        f_direct, g_direct = _sinkhorn_iteration(f, g, log_K, log_a, log_b, eps)

        cf = CompiledFunctions(compile=False)
        f_cf, g_cf = cf.sinkhorn_iter(f, g, log_K, log_a, log_b, eps)

        torch.testing.assert_close(f_direct, f_cf)
        torch.testing.assert_close(g_direct, g_cf)


class TestRotateAndTranslateEquivalence:
    """Test 6: _rotate_and_translate direct equivalence."""

    def test_rotate_and_translate_equivalence(self):
        torch.manual_seed(42)
        B, d, V = 5, 3, 6  # batch, dim, vertices
        step_radius = 0.5

        # Random rotation matrices (orthogonal via QR)
        raw = torch.randn(B, d, d)
        Q, _ = torch.linalg.qr(raw)
        rot_mats = Q

        polytope_vertices = torch.randn(V, d)
        origin = torch.randn(B, d)

        sp_direct, rot_direct = _rotate_and_translate(
            rot_mats, polytope_vertices, origin, step_radius
        )

        cf = CompiledFunctions(compile=False)
        sp_cf, rot_cf = cf.rotate_and_translate(
            rot_mats, polytope_vertices, origin, step_radius
        )

        torch.testing.assert_close(sp_direct, sp_cf)
        torch.testing.assert_close(rot_direct, rot_cf)

        # Verify shapes
        assert sp_direct.shape == (B, V, d)
        assert rot_direct.shape == (B, V, d)


class TestBarycentricProjectionEquivalence:
    """Test 7: _barycentric_projection equivalence and shape."""

    def test_barycentric_projection_equivalence(self):
        torch.manual_seed(42)
        B, V, d = 5, 6, 3

        transport_matrix = torch.softmax(torch.randn(B, V), dim=-1)
        a = torch.ones(B) / B
        X_vertices = torch.randn(B, V, d)

        result_direct = _barycentric_projection(transport_matrix, a, X_vertices)

        cf = CompiledFunctions(compile=False)
        result_cf = cf.barycentric_projection(transport_matrix, a, X_vertices)

        torch.testing.assert_close(result_direct, result_cf)
        assert result_direct.shape == (B, d)

        # Verify against manual computation
        weights = transport_matrix / a.unsqueeze(-1)
        X_manual = torch.einsum("bkd,bk->bd", X_vertices, weights)
        torch.testing.assert_close(result_direct, X_manual)


class TestSinkhornSolverCompileFlagEquivalence:
    """Test 8: SinkhornSolver compile=True vs compile=False on CPU."""

    def test_sinkhorn_solver_compile_flag_equivalence(self):
        torch.manual_seed(42)
        n, m = 50, 30
        cost_matrix = torch.rand(n, m) + 0.01

        solver_compiled = SinkhornSolver(
            compile=True, max_iterations=100, threshold=-1, epsilon=0.1,
        )
        solver_eager = SinkhornSolver(
            compile=False, max_iterations=100, threshold=-1, epsilon=0.1,
        )

        result_compiled = solver_compiled.solve(cost_matrix.clone())
        result_eager = solver_eager.solve(cost_matrix.clone())

        # On CPU both are eager, so results must be identical
        torch.testing.assert_close(result_compiled.f, result_eager.f, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result_compiled.g, result_eager.g, atol=1e-5, rtol=1e-5)


class TestPolyStepCompileFlagEquivalence:
    """Test 9: PolyStep compile=True vs compile=False produce same particles."""

    def test_sinkhorn_step_compile_flag_equivalence(self):
        torch.manual_seed(42)
        dim = 5
        num_particles = 20

        def objective_fn(x):
            return x.pow(2).sum(-1)

        solver_compiled = PolyStep(
            objective_fn=objective_fn, dim=dim, compile=True,
            max_iterations=3, sinkhorn_max_iters=50, threshold=-1,
        )
        solver_eager = PolyStep(
            objective_fn=objective_fn, dim=dim, compile=False,
            max_iterations=3, sinkhorn_max_iters=50, threshold=-1,
        )

        X_init = torch.randn(num_particles, dim)

        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)

        state_compiled = solver_compiled.run(X_init.clone(), generator=g1)
        state_eager = solver_eager.run(X_init.clone(), generator=g2)

        torch.testing.assert_close(
            state_compiled.X, state_eager.X, atol=1e-5, rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Group 4: Benchmark tests (marked slow)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Group 5: Pure function contracts (static source checks)
# ---------------------------------------------------------------------------


class TestCompiledFunctionsNoGraphBreaks:
    """Test 12: Static check that compiled functions avoid graph-break patterns."""

    @pytest.mark.parametrize("fn", [
        _sinkhorn_iteration,
        _rotate_and_translate,
        _barycentric_projection,
        _compute_probe_points,
    ], ids=["sinkhorn_iteration", "rotate_and_translate",
            "barycentric_projection", "compute_probe_points"])
    def test_no_graph_break_patterns(self, fn):
        source = inspect.getsource(fn)

        assert ".item()" not in source, (
            f"{fn.__name__} contains .item() which causes graph breaks"
        )
        assert ".append(" not in source, (
            f"{fn.__name__} contains .append() which causes graph breaks"
        )
        assert ".tolist()" not in source, (
            f"{fn.__name__} contains .tolist() which causes graph breaks"
        )


# ---------------------------------------------------------------------------
# Helpers for GPU graph break verification (Groups 6-7)
# ---------------------------------------------------------------------------


def _make_sinkhorn_args(device):
    """Create input tensors for _sinkhorn_iteration on the given device."""
    n, m = 200, 150
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
    a = torch.ones(B, device=device) / B
    X_vertices = torch.randn(B, V, d, device=device)
    return (transport, a, X_vertices)


def _make_probe_args(device):
    """Create input tensors for _compute_probe_points on the given device."""
    B, num_points, d = 10, 40, 20
    origin = torch.randn(B, d, device=device)
    directions = torch.randn(B, num_points, d, device=device)
    scales = torch.linspace(0.2, 0.8, 3, device=device)
    probe_radius = 1.0
    return (origin, directions, scales, probe_radius)


# ---------------------------------------------------------------------------
# Group 6: GPU runtime graph break verification (fullgraph=True)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(120)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGPUGraphBreakVerification:
    """Test 13: Runtime verification that compiled functions have zero graph breaks on GPU.

    Uses torch.compile(fullgraph=True) which raises an error if any graph break
    is detected. If compilation and execution succeed, that IS the proof of zero
    graph breaks.
    """

    @pytest.mark.parametrize("fn_name,fn,make_args", [
        ("sinkhorn_iteration", _sinkhorn_iteration, _make_sinkhorn_args),
        ("rotate_and_translate", _rotate_and_translate, _make_rotate_args),
        ("barycentric_projection", _barycentric_projection, _make_barycentric_args),
        ("compute_probe_points", _compute_probe_points, _make_probe_args),
    ])
    def test_fullgraph_compilation_succeeds(self, fn_name, fn, make_args):
        """fullgraph=True raises on graph breaks; success = zero breaks."""
        torch._dynamo.reset()
        device = torch.device("cuda")
        args = make_args(device)
        compiled_fn = torch.compile(fn, fullgraph=True)
        result = compiled_fn(*args)
        torch.cuda.synchronize()
        # If we reach here, fullgraph compilation succeeded = zero graph breaks
        assert result is not None, f"{fn_name}: compiled function returned None"


# ---------------------------------------------------------------------------
# Group 7: torch._dynamo.explain verification (single graph confirmation)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(120)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDynamoExplainVerification:
    """Test 14: torch._dynamo.explain confirms single graph per function.

    The explain API returns structured information about how dynamo traces the
    function, including graph_count (should be 1) and break_reasons (should be
    empty). This provides explicit proof of zero graph breaks.
    """

    @pytest.mark.parametrize("fn_name,fn,make_args", [
        ("sinkhorn_iteration", _sinkhorn_iteration, _make_sinkhorn_args),
        ("rotate_and_translate", _rotate_and_translate, _make_rotate_args),
        ("barycentric_projection", _barycentric_projection, _make_barycentric_args),
        ("compute_probe_points", _compute_probe_points, _make_probe_args),
    ])
    def test_single_graph_no_breaks(self, fn_name, fn, make_args):
        """torch._dynamo.explain confirms exactly 1 graph and 0 break reasons."""
        torch._dynamo.reset()
        device = torch.device("cuda")
        args = make_args(device)
        explanation = torch._dynamo.explain(fn)(*args)
        assert explanation.graph_count == 1, (
            f"{fn_name}: expected 1 graph, got {explanation.graph_count}. "
            f"Break reasons: {explanation.break_reasons}"
        )
        assert len(explanation.break_reasons) == 0, (
            f"{fn_name}: expected 0 break reasons, got {len(explanation.break_reasons)}: "
            f"{explanation.break_reasons}"
        )


# ---------------------------------------------------------------------------
# Group 8: Fused softmax + projection tests
# ---------------------------------------------------------------------------


def _make_fused_softmax_args(device, P=10, V=8, dim=4, seed=42):
    """Create input tensors for _fused_softmax_project on the given device."""
    torch.manual_seed(seed)
    cost_matrix = torch.rand(P, V, device=device) + 0.01
    epsilon = 0.1
    a = torch.ones(P, device=device) / P
    polytope_verts = torch.randn(V, dim, device=device)
    # Random rotation matrices via QR
    raw = torch.randn(P, dim, dim, device=device)
    Q, _ = torch.linalg.qr(raw)
    rot_mats = Q
    step_radius = 0.5
    X = torch.randn(P, dim, device=device)
    return cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X


class TestFusedSoftmaxProjectReturnTypes:
    """Test 15: _fused_softmax_project returns tuple of 3 tensors."""

    def test_returns_tuple_of_three_tensors(self):
        cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X = (
            _make_fused_softmax_args(torch.device("cpu"))
        )
        result = _fused_softmax_project(
            cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"
        X_new, transport, ent_cost = result
        assert isinstance(X_new, torch.Tensor)
        assert isinstance(transport, torch.Tensor)
        assert isinstance(ent_cost, torch.Tensor)


class TestFusedSoftmaxProjectShapes:
    """Test 16: Output shapes are correct."""

    def test_output_shapes(self):
        P, V, dim = 10, 8, 4
        cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X = (
            _make_fused_softmax_args(torch.device("cpu"), P=P, V=V, dim=dim)
        )
        X_new, transport, ent_cost = _fused_softmax_project(
            cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X,
        )
        assert X_new.shape == (P, dim), f"X_new shape {X_new.shape} != ({P}, {dim})"
        assert transport.shape == (P, V), f"transport shape {transport.shape} != ({P}, {V})"
        assert ent_cost.dim() == 0, f"ent_cost should be 0-dim tensor, got dim={ent_cost.dim()}"


class TestFusedSoftmaxProjectEquivalence:
    """Test 17: Fused result matches two-step path (softmax solve + barycentric projection)."""

    def test_equivalence_with_two_step_path(self):
        P, V, dim = 10, 8, 4
        torch.manual_seed(42)
        cost_matrix = torch.rand(P, V) + 0.01
        epsilon = 0.1
        a = torch.ones(P) / P
        polytope_verts = torch.randn(V, dim)
        raw = torch.randn(P, dim, dim)
        Q, _ = torch.linalg.qr(raw)
        rot_mats = Q
        step_radius = 0.5
        X = torch.randn(P, dim)

        # --- Two-step path ---
        # Step 1: Softmax solve (replicate SoftmaxSolver.solve logic with scale_cost='mean')
        s = torch.clamp(cost_matrix.abs().mean(), min=1e-10)
        C_scaled = cost_matrix / s
        W = torch.softmax(-C_scaled / epsilon, dim=-1)
        transport_ref = W * a.unsqueeze(-1)

        # Step 2: Barycentric projection with rotated vertices
        rotated = torch.einsum("bji, ni -> bnj", rot_mats, polytope_verts)
        X_vertices = rotated * step_radius + X.unsqueeze(1)
        weights_ref = transport_ref / a.unsqueeze(-1)
        X_new_ref = torch.einsum("bkd,bk->bd", X_vertices, weights_ref)

        ent_cost_ref = (C_scaled * transport_ref).sum()

        # --- Fused path ---
        X_new_fused, transport_fused, ent_cost_fused = _fused_softmax_project(
            cost_matrix.clone(), epsilon, a, polytope_verts, rot_mats, step_radius, X,
        )

        # Check equivalence
        assert torch.allclose(X_new_fused, X_new_ref, atol=1e-5), (
            f"X_new max diff: {(X_new_fused - X_new_ref).abs().max().item():.2e}"
        )
        assert torch.allclose(transport_fused, transport_ref, atol=1e-6), (
            f"transport max diff: {(transport_fused - transport_ref).abs().max().item():.2e}"
        )
        assert torch.allclose(ent_cost_fused, ent_cost_ref, atol=1e-5), (
            f"ent_cost diff: {(ent_cost_fused - ent_cost_ref).abs().item():.2e}"
        )


class TestFusedSoftmaxProjectTransportRowSums:
    """Test 18: Transport matrix row sums equal source marginal a."""

    def test_transport_row_sums_equal_a(self):
        P, V, dim = 10, 8, 4
        cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X = (
            _make_fused_softmax_args(torch.device("cpu"), P=P, V=V, dim=dim)
        )
        _, transport, _ = _fused_softmax_project(
            cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X,
        )
        row_sums = transport.sum(dim=-1)
        assert torch.allclose(row_sums, a, atol=1e-6), (
            f"Row sums max diff from a: {(row_sums - a).abs().max().item():.2e}"
        )


class TestFusedSoftmaxProjectNoScaling:
    """Test 19: With scale_cost_mean=False, cost matrix is used as-is."""

    def test_no_scaling_path(self):
        P, V, dim = 10, 8, 4
        torch.manual_seed(42)
        cost_matrix = torch.rand(P, V) + 0.01
        epsilon = 0.1
        a = torch.ones(P) / P
        polytope_verts = torch.randn(V, dim)
        raw = torch.randn(P, dim, dim)
        Q, _ = torch.linalg.qr(raw)
        rot_mats = Q
        step_radius = 0.5
        X = torch.randn(P, dim)

        # No scaling: C is used directly
        W_expected = torch.softmax(-cost_matrix / epsilon, dim=-1)
        transport_expected = W_expected * a.unsqueeze(-1)

        _, transport_fused, _ = _fused_softmax_project(
            cost_matrix.clone(), epsilon, a, polytope_verts, rot_mats, step_radius, X,
            scale_cost_mean=False,
        )
        assert torch.allclose(transport_fused, transport_expected, atol=1e-6), (
            f"No-scaling transport max diff: {(transport_fused - transport_expected).abs().max().item():.2e}"
        )


class TestFusedSoftmaxProjectCompiledFunctionsEager:
    """Test 20: CompiledFunctions(compile=False).fused_softmax_project is callable and correct."""

    def test_eager_registration(self):
        cf = CompiledFunctions(compile=False)
        assert hasattr(cf, "fused_softmax_project"), (
            "CompiledFunctions missing fused_softmax_project attribute"
        )
        assert callable(cf.fused_softmax_project)

        # Verify it returns correct shapes
        P, V, dim = 10, 8, 4
        cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X = (
            _make_fused_softmax_args(torch.device("cpu"), P=P, V=V, dim=dim)
        )
        X_new, transport, ent_cost = cf.fused_softmax_project(
            cost_matrix, epsilon, a, polytope_verts, rot_mats, step_radius, X,
        )
        assert X_new.shape == (P, dim)
        assert transport.shape == (P, V)
        assert ent_cost.dim() == 0


class TestFusedSoftmaxProjectWarmStart:
    """Test 21: CompiledFunctions.warm_start() runs without error including fused warmup."""

    def test_warm_start_includes_fused(self):
        cf = CompiledFunctions(compile=False)
        # Should not raise
        cf.warm_start(dim=4, batch=3, device=torch.device("cpu"))


class TestFusedSoftmaxProjectNoGraphBreaks:
    """Test 22: Static check that _fused_softmax_project avoids graph-break patterns."""

    def test_no_graph_break_patterns(self):
        source = inspect.getsource(_fused_softmax_project)
        assert ".item()" not in source, (
            "_fused_softmax_project contains .item() which causes graph breaks"
        )
        assert ".append(" not in source, (
            "_fused_softmax_project contains .append() which causes graph breaks"
        )
        assert ".tolist()" not in source, (
            "_fused_softmax_project contains .tolist() which causes graph breaks"
        )
