"""Mixed precision: bf16 geometry with an fp32 OT solve.

Covers the dtype contract at construction, the step itself, and the barycentric and
fused-softmax kernels that have to normalize across the two.
"""

import math

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cma_subspace import CMAAdaptiveSubspace
from polystep._compiled import _barycentric_projection, _fused_softmax_project
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


class TestMixedPrecisionProperties:
    @pytest.mark.parametrize(
        "kwargs,expected_mixed_precision,expected_dtype",
        [
            ({}, False, torch.float32),
            ({"mixed_precision": True}, True, torch.bfloat16),
        ],
    )
    def test_mixed_precision_default_false(self, kwargs, expected_mixed_precision, expected_dtype):
        """mixed_precision property and model_dtype follow the mixed_precision flag."""
        model = nn.Linear(10, 5)
        opt = PolyStepOptimizer(model, compile=False, **kwargs)
        assert opt.mixed_precision is expected_mixed_precision
        # On CPU, BF16 is always supported
        assert opt.model_dtype == expected_dtype

    @pytest.mark.parametrize(
        "mixed_precision,expected_dtype",
        [
            (True, torch.bfloat16),
            (False, torch.float32),
        ],
    )
    def test_model_cast_to_bfloat16(self, mixed_precision, expected_dtype):
        """Model parameters are cast to BF16 only when mixed precision enabled."""
        model = nn.Linear(10, 5)
        PolyStepOptimizer(model, mixed_precision=mixed_precision, compile=False)
        assert next(model.parameters()).dtype == expected_dtype


class TestMixedPrecisionStep:
    def test_step_with_mixed_precision(self):
        """Optimizer step works with mixed precision enabled."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        opt = PolyStepOptimizer(
            model,
            mixed_precision=True,
            epsilon=0.1,
            step_radius=0.5,
            compile=False,
        )

        # Verify model is BF16
        assert next(model.parameters()).dtype == torch.bfloat16

        def closure(batched_params):
            from torch.func import functional_call, vmap

            model.eval()
            x = torch.randn(4, 10)

            def forward(params):
                # Candidate params are BF16 under mixed_precision (the real
                # NNCostEvaluator casts inputs to the param dtype); mirror that.
                xc = x.to(next(iter(params.values())).dtype)
                return functional_call(model, params, (xc,)).mean()

            losses = vmap(forward)(batched_params)
            model.train()
            return losses

        # Should not raise
        cost = opt.step(closure)
        assert isinstance(cost, float)
        assert not torch.isnan(torch.tensor(cost))

    def test_step_costs_are_fp32(self):
        """Cost matrix in Sinkhorn solver is FP32 even with mixed precision."""
        model = nn.Linear(10, 5)
        opt = PolyStepOptimizer(
            model,
            mixed_precision=True,
            epsilon=0.1,
            compile=False,
        )

        # Observe what the solver returns, not a value the test computes itself. The
        # previous version called sanitize_cost inside the patch and asserted on that,
        # so it passed even if the solver iterated entirely in BF16.
        captured = {}
        original_solve = opt.solver.solve

        def patched_solve(cost_matrix, **kwargs):
            result = original_solve(cost_matrix, **kwargs)
            captured["entry"] = cost_matrix.dtype
            captured["plan"] = result.matrix.dtype
            captured["duals"] = None if result.f is None else result.f.dtype
            return result

        opt.solver.solve = patched_solve

        def closure(batched_params):
            from torch.func import functional_call, vmap

            model.eval()
            x = torch.randn(4, 10)

            def forward(params):
                xc = x.to(next(iter(params.values())).dtype)
                return functional_call(model, params, (xc,)).mean()

            losses = vmap(forward)(batched_params)
            model.train()
            return losses

        opt.step(closure)
        assert captured, "solver was never called"
        assert captured["plan"] == torch.float32, (
            f"Sinkhorn returned a {captured['plan']} plan; BF16's 7 mantissa bits collapse "
            f"the log-domain row-max trick (entry dtype was {captured['entry']})"
        )
        if captured["duals"] is not None:
            assert captured["duals"] == torch.float32, f"duals came back {captured['duals']}"


def test_projection_dtype_matches_model():
    """Projection matrix dtype matches model dtype for memory savings."""
    # Without mixed precision
    model1 = nn.Linear(100, 50)
    subspace1 = AdaptiveSubspace(full_dim=100 * 50 + 50, subspace_dim=32)
    opt1 = PolyStepOptimizer(model1, subspace=subspace1, mixed_precision=False, compile=False)
    assert opt1.state.projection.dtype == torch.float32

    # With mixed precision
    model2 = nn.Linear(100, 50)
    subspace2 = AdaptiveSubspace(full_dim=100 * 50 + 50, subspace_dim=32)
    opt2 = PolyStepOptimizer(model2, subspace=subspace2, mixed_precision=True, compile=False)
    assert opt2.state.projection.dtype == torch.bfloat16


class TestProjectionDtype:
    @pytest.mark.parametrize(
        "dtype,expected_dtype",
        [
            (None, torch.float32),
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.bfloat16),
        ],
    )
    def test_init_projection_default_fp32(self, dtype, expected_dtype):
        """init_projection defaults to FP32 and honors an explicit dtype."""
        subspace = AdaptiveSubspace(full_dim=100, subspace_dim=16)
        projection = subspace.init_projection(dtype=dtype)
        assert projection.dtype == expected_dtype

    def test_cma_subspace_projection_dtype(self):
        """CMAAdaptiveSubspace passes dtype through."""
        base = AdaptiveSubspace(full_dim=100, subspace_dim=16)
        cma = CMAAdaptiveSubspace(base)

        projection_fp32 = cma.init_projection(dtype=torch.float32)
        assert projection_fp32.dtype == torch.float32

        projection_bf16 = cma.init_projection(dtype=torch.bfloat16)
        assert projection_bf16.dtype == torch.bfloat16

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_displacement_rotation_runs_in_half_precision(self, dtype):
        """CPU LAPACK has no half QR or SVD, so the decomposition must run in fp32.

        The guards covered bf16 only, so fp16 raised
        ``"linalg_svd_cpu" not implemented for 'Half'`` on the first rotation with
        history, which the default rotation_mode reaches on step 1.
        """
        model = nn.Sequential(nn.Linear(16, 12), nn.Linear(12, 4))
        layout = ParamLayout.from_module(model)
        full_dim = sum(p.numel() for p in model.parameters())
        history = torch.randn(5, full_dim, dtype=dtype)

        adaptive = AdaptiveSubspace(full_dim=full_dim, subspace_dim=8, rotation_mode="displacement")
        rotated = adaptive.rotate(
            adaptive.init_projection(dtype=dtype),
            step=1,
            total_steps=10,
            displacement_history=history,
            history_is_full=True,
        )
        assert rotated.dtype == dtype and torch.isfinite(rotated).all()

        hybrid = HybridSubspace.from_layout(
            layout, rank=4, rotation_mode="displacement", rotation_interval=1, svd_ratio_init=0.5
        )
        projections = hybrid.rotate_all(
            hybrid.init_projections(torch.device("cpu"), dtype), step=1, total_steps=10, displacement_history=history
        )
        for P in projections.values():
            assert P.dtype == dtype and torch.isfinite(P).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_bf16_support():
    """GPU BF16 support based on compute capability."""
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(device)

    model = nn.Linear(10, 5).to(device)
    opt = PolyStepOptimizer(model, mixed_precision=True, compile=False)

    if cap[0] >= 7:
        # Volta+ supports BF16
        assert opt.model_dtype == torch.bfloat16
    else:
        # Pre-Volta falls back to FP32
        assert opt.model_dtype == torch.float32


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_rotation_on_cuda(dtype):
    """CUDA has no half-precision geqrf or lu_factor, so QR must run in FP32."""
    from polystep.geometry import get_random_rotation_matrices

    R = get_random_rotation_matrices(4, 5, device=torch.device("cuda"), dtype=dtype)
    assert R.dtype == dtype and R.device.type == "cuda"
    eye = R.float().transpose(-2, -1) @ R.float()
    assert torch.allclose(eye, torch.eye(5, device=R.device).expand_as(eye), atol=1e-2)
    assert (torch.det(R.float()) > 0).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_step_with_wide_polytope_on_cuda():
    """particle_dim > 2 takes the QR branch, which has no half-precision kernel."""
    from torch.func import functional_call, vmap

    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4)).cuda()
    opt = PolyStepOptimizer(model, mixed_precision=True, particle_dim=4, compile=False)
    x = torch.randn(16, 8, device="cuda", dtype=opt.model_dtype)

    def closure(batched_params):
        return vmap(lambda p: functional_call(model, p, (x,)).mean())(batched_params)

    cost = opt.step(closure)
    assert not torch.isnan(torch.tensor(cost))


def test_no_nans_in_normal_training():
    """Normal training with mixed precision produces no NaNs."""
    model = nn.Sequential(
        nn.Linear(8, 6),
        nn.ReLU(),
        nn.Linear(6, 4),
    )
    opt = PolyStepOptimizer(
        model,
        mixed_precision=True,
        epsilon=0.1,
        step_radius=0.3,
        compile=False,
    )

    # Drawn once, outside the closure. Redrawing per call gave every candidate in a
    # step a different objective, so the cost matrix ranked noise rather than
    # vertices and the step direction was meaningless.
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(8, 8, generator=gen)
    target = torch.randn(8, 4, generator=gen)

    def closure(batched_params):
        from torch.func import functional_call, vmap

        model.eval()

        def forward(params):
            xc = x.to(next(iter(params.values())).dtype)
            out = functional_call(model, params, (xc,))
            return ((out - target.to(out.dtype)) ** 2).mean()

        losses = vmap(forward)(batched_params)
        model.train()
        return losses

    costs = []
    for _ in range(3):
        cost = opt.step(closure)
        assert not torch.isnan(torch.tensor(cost)), "OT cost should not be NaN"
        costs.append(cost)
    # A stationary objective must actually be descended, not merely survived.
    assert min(costs) < costs[0], f"no progress on a fixed objective: {costs}"


def test_barycentric_projection_keeps_the_vertex_dtype():
    """fp32 transport weights against bf16 vertices return finite bf16."""
    b, V, d = 3, 6, 3
    transport = torch.rand(b, V)  # fp32
    X_vertices = torch.randn(b, V, d, dtype=torch.bfloat16)
    out = _barycentric_projection(transport, X_vertices)
    assert out.dtype == torch.bfloat16
    assert out.shape == (b, d)
    assert torch.isfinite(out.float()).all()


def test_fused_softmax_project_keeps_the_geometry_dtype():
    """fp32 cost against bf16 geometry returns finite bf16."""
    b, V, d = 3, 6, 3
    C = torch.randn(b, V)  # fp32 cost
    pv = torch.randn(V, d, dtype=torch.bfloat16)
    rot = torch.eye(d, dtype=torch.bfloat16).expand(b, d, d).contiguous()
    X = torch.randn(b, d, dtype=torch.bfloat16)
    a = torch.full((b,), 1.0 / b)
    X_new, transport = _fused_softmax_project(C, 0.1, a, pv, rot, 1.0, X, scale_cost_mean=False)
    assert X_new.dtype == torch.bfloat16
    assert torch.isfinite(X_new.float()).all()


def test_barycentric_softmax_path_unchanged():
    """When rows already sum to ``a`` (softmax path), /rowsum == /a exactly."""
    b, V, d = 4, 5, 3
    a = torch.full((b,), 1.0 / b)
    transport = torch.softmax(-torch.randn(b, V), dim=-1) * a.unsqueeze(-1)
    X_vertices = torch.randn(b, V, d)
    new = _barycentric_projection(transport, X_vertices)
    old = torch.einsum("bkd,bk->bd", X_vertices, transport / a.unsqueeze(-1))
    assert torch.allclose(new, old, atol=1e-6)


def test_barycentric_translation_invariant_unconverged_plan():
    """An unconverged plan (rows not summing to a) must give a translation-invariant step."""
    b, V, d = 3, 6, 3
    plan = torch.rand(b, V)  # rows deliberately do NOT sum to a
    origin = torch.randn(b, d)
    shift = torch.tensor([5.0, -2.0, 1.0])
    X_o = torch.randn(b, V, d) + origin.unsqueeze(1)
    X_s = X_o + shift  # shift every vertex by the same constant
    step_o = _barycentric_projection(plan, X_o)
    step_s = _barycentric_projection(plan, X_s)
    # Shifting all vertices by c must shift the barycenter by exactly c.
    assert torch.allclose(step_s - step_o, shift.expand(b, d), atol=1e-5)


def _run_mixed_precision_steps(solver: str, n_steps: int = 3, model=None) -> float:
    model = model if model is not None else nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4)
    opt = PolyStepOptimizer(
        model,
        subspace=sub,
        solver=solver,
        epsilon=0.5,
        step_radius=0.5,
        probe_radius=1.0,
        mixed_precision=True,
    )
    x = torch.randn(32, 16)
    y = torch.randint(0, 3, (32,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    loss = float("nan")
    for _ in range(n_steps):
        loss = float(opt.step(lambda s: ev.evaluate(s, x, y)))
    return loss


@pytest.mark.parametrize("solver", ["softmax", "sinkhorn"])
def test_optimizer_mixed_precision_step(solver):
    """End-to-end mixed_precision=True must run on CPU with HybridSubspace.

    ``loss == loss`` rules out NaN and nothing else, so an inf or a frozen optimizer
    passed it. The parameters have to move and the loss has to stay in range.
    """
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 3))
    before = [p.detach().clone() for p in model.parameters()]
    loss = _run_mixed_precision_steps(solver, model=model)

    assert math.isfinite(loss), loss
    # Cross-entropy over 3 classes starts near ln(3); a step that broke the geometry
    # lands far outside that scale without ever producing a NaN.
    assert 0.0 < loss < 10.0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), "the model never moved"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")


def test_mixed_dtype_model_runs_in_subspace_mode():
    """Coordinates carry the layout's dominant dtype while each projection carries
    its own parameter's, so an FP64 minority parameter mixed dtypes in the matmul
    and raised. The minority parameter must also keep its dtype.
    """
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.subspace import LinearSubspace

    class MixedDtype(nn.Module):
        def __init__(self):
            super().__init__()
            self.big = nn.Parameter(torch.randn(20, 20))
            self.small = nn.Parameter(torch.randn(4, 4, dtype=torch.float64))

        def forward(self, x):
            return x @ self.big

    for build in (LinearSubspace.from_layout, HybridSubspace.from_layout):
        torch.manual_seed(0)
        model = MixedDtype()
        subspace = build(ParamLayout.from_module(model), rank=2)
        opt = PolyStepOptimizer(model, subspace=subspace, compile=False, seed=0)
        inputs, targets = torch.randn(8, 20), torch.randn(8, 20)
        loss_fn = nn.MSELoss()

        def closure(batched):
            from torch.func import functional_call, vmap

            return vmap(lambda p: loss_fn(functional_call(model, p, (inputs,)), targets))(batched)

        opt.step(closure)

        assert model.small.dtype == torch.float64
        assert model.big.dtype == torch.float32


def test_mixed_dtype_model_computing_in_both_dtypes():
    """A model whose forward actually runs in two dtypes, not just holding an unused one.

    Coordinates carry the layout's dominant dtype, so each per-entry projection has to be
    built at its own parameter's or the reconstruction matmul mixes Double and Float. The
    site-aware evaluator must also stop casting inputs to the perturbed layer's dtype: the
    input reaches the fp32 layer first.
    """
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.subspace import LinearSubspace, LowRankSubspace
    from polystep.cost_nn import NNCostEvaluator

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4).double()

        def forward(self, x):
            return self.b(self.a(x).double())

    builders = [
        lambda lo: HybridSubspace.from_layout(lo, rank=2),
        lambda lo: HybridSubspace.from_layout(
            lo, rank=2, rotation_interval=1, rotation_mode="displacement", absorb_mode="periodic", absorb_interval=2
        ),
        lambda lo: LinearSubspace.from_layout(lo, rank=2),
        lambda lo: LowRankSubspace.from_layout(lo, rank=2),
    ]
    for build in builders:
        torch.manual_seed(0)
        model = Mixed()
        opt = PolyStepOptimizer(model, subspace=build(ParamLayout.from_module(model)), seed=0)
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        inputs, targets = torch.randn(8, 8), torch.randint(0, 4, (8,))
        for i in range(4):
            opt.register_evaluator(evaluator, inputs, targets)
            opt.step(lambda p: evaluator.evaluate(p, inputs, targets), objective_token=i)

        sd = model.state_dict()
        assert sd["a.weight"].dtype is torch.float32
        assert sd["b.weight"].dtype is torch.float64, "the minority dtype was rewritten"
        assert model(inputs).dtype is torch.float64


def test_global_vector_modes_reject_a_mixed_dtype_model():
    """Full space and AdaptiveSubspace hold every parameter in one vector at one dtype.

    A minority-dtype weight would be optimized at the majority's precision, so this has to
    raise rather than downgrade it silently.
    """
    from polystep.adaptive_subspace import AdaptiveSubspace

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4).double()

        def forward(self, x):
            return self.b(self.a(x).double())

    for kwargs in ({}, {"subspace": AdaptiveSubspace.from_layout(ParamLayout.from_module(Mixed()), rank=8)}):
        with pytest.raises(ValueError, match="mixes parameter dtypes"):
            PolyStepOptimizer(Mixed(), seed=0, **kwargs)
