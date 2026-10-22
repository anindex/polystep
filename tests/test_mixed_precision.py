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
from polystep._compiled import _barycentric_projection, _fused_softmax_project
from polystep.cost_nn import NNCostEvaluator, SiteVmapEvaluator
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
    def test_step_costs_are_fp32(self):
        """Cost matrix in Sinkhorn solver is FP32 even with mixed precision."""
        model = nn.Linear(10, 5)
        opt = PolyStepOptimizer(
            model,
            mixed_precision=True,
            epsilon=0.1,
            compile=False,
        )

        # Observe what the solver returns, not a value the test computes itself:
        # asserting on a locally sanitized copy would pass even if the solver
        # iterated entirely in BF16.
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


@pytest.mark.parametrize("register", [True, False])
def test_evaluator_before_mixed_precision_steps(register):
    """An evaluator built before the BF16 cast scores with the cast buffers."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(144, 3))
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, mixed_precision=True, seed=0, compile=False)
    x = torch.randn(16, 1, 8, 8, dtype=torch.bfloat16)
    y = torch.randint(0, 3, (16,))
    if register:
        opt.register_evaluator(evaluator, x, y)
    loss = opt.step(lambda p: evaluator.evaluate(p, x, y))
    assert torch.isfinite(torch.as_tensor(loss))


@pytest.mark.parametrize(
    "path",
    [
        "batched",
        "vmap",
        "loop",
        "inplace",
        "site",
        pytest.param(
            "compiled",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="compile_forward needs CUDA"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_bf16_candidate_losses_reduce_in_fp32(path, reduction):
    """A BF16 mean near 2.37 moves in steps of 2^-6, so candidates whose FP32 cross
    entropy differs by about 1e-3 would tie."""
    gen = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 10)).bfloat16()
    with torch.no_grad():
        model[0].weight.copy_(0.1 * torch.randn(10, 16, generator=gen))
        model[0].bias.zero_()
    inputs, targets = torch.randn(256, 16, generator=gen), torch.zeros(256, dtype=torch.long)
    bias = torch.zeros(2, 10, dtype=torch.bfloat16)
    bias[1, 0] = 2.0**-10
    weight = model[0].weight.detach()
    reference = [nn.functional.cross_entropy(inputs @ weight.float().t() + b.float(), targets) for b in bias]
    assert 5e-4 < reference[0] - reference[1] < 2e-3

    device = "cuda" if path == "compiled" else "cpu"
    evaluator = NNCostEvaluator(
        model.to(device), nn.CrossEntropyLoss(reduction=reduction), use_inplace=path in ("inplace", "compiled")
    )
    if path in ("vmap", "loop"):
        evaluator._batched_linear = None
        evaluator._vmap_failed = path == "loop"
    if path == "site":
        layout = ParamLayout.from_module(model)
        entry = next(e for e in layout.entries if e.key == "0.bias")
        site = SiteVmapEvaluator.try_build(evaluator, layout)
        base_sd = {k: v.detach() for k, v in model.state_dict().items()}
        losses = site.evaluate(base_sd, entry, torch.tensor([[0]]), bias[:, :1].reshape(1, 2, 1), inputs, targets)
    else:
        stacked = {"0.weight": weight.expand(2, *weight.shape).clone(), "0.bias": bias}
        stacked = {k: v.to(device) for k, v in stacked.items()}
        losses = evaluator.evaluate(stacked, inputs.to(device), targets.to(device))
    assert losses[1] < losses[0]
    assert evaluator._compile_forward_verified is (path == "compiled")


def test_bf16_batched_cross_entropy_matches_vmap():
    """Zero logits put both candidates on one BF16 cross entropy; FP32 logits keep
    the 2e-3 bias gap, and the batched path scores them as vmap does."""
    model = nn.Sequential(nn.Linear(1, 2)).bfloat16()
    nn.init.zeros_(model[0].weight)
    nn.init.zeros_(model[0].bias)
    inputs, targets = torch.zeros(4, 1), torch.zeros(4, dtype=torch.long)
    weight = model[0].weight.detach()
    stacked = {
        "0.weight": weight.expand(2, *weight.shape).clone(),
        "0.bias": torch.tensor([[0.0, 0.0], [0.002, 0.0]], dtype=torch.bfloat16),
    }
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert evaluator._batched_linear is not None
    batched = evaluator.evaluate(stacked, inputs, targets)
    evaluator._batched_linear = None
    torch.testing.assert_close(batched, evaluator.evaluate(stacked, inputs, targets), rtol=0, atol=1e-7)
    assert batched[1] < batched[0]


def test_bf16_weighted_cross_entropy_scores_candidates():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3)).bfloat16()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0, 1.0], dtype=torch.bfloat16))
    inputs, targets = torch.randn(8, 4, dtype=torch.bfloat16), torch.randint(0, 3, (8,))
    weight = model[0].weight.detach()
    bias = torch.randn(2, 3, dtype=torch.bfloat16)
    evaluator = NNCostEvaluator(model, loss_fn)
    evaluator._batched_linear = None
    losses = evaluator.evaluate({"0.weight": weight.expand(2, *weight.shape).clone(), "0.bias": bias}, inputs, targets)
    expected = torch.stack([loss_fn(nn.functional.linear(inputs, weight, b), targets) for b in bias])
    torch.testing.assert_close(losses, expected)


@pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.L1Loss()])
def test_bf16_batched_regression_reduces_in_fp32(loss_fn):
    """The batched path averages BF16 outputs in FP32, so each candidate matches its
    FP32 loss instead of the nearest BF16 value, which is up to 2^-7 away here."""
    gen = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 1)).bfloat16()
    with torch.no_grad():
        model[0].weight.copy_(0.1 * torch.randn(1, 16, generator=gen))
        model[0].bias.zero_()
    inputs, targets = torch.randn(256, 16, generator=gen), torch.full((256, 1), 1.5, dtype=torch.bfloat16)
    bias = torch.tensor([[0.0], [2.0**-8]], dtype=torch.bfloat16)
    weight = model[0].weight.detach()
    outputs = [nn.functional.linear(inputs.bfloat16(), weight, b).float() for b in bias]
    reference = torch.stack([loss_fn(out, targets.float()) for out in outputs])

    evaluator = NNCostEvaluator(model, loss_fn)
    assert evaluator._batched_linear is not None
    stacked = {"0.weight": weight.expand(2, *weight.shape).clone(), "0.bias": bias}
    torch.testing.assert_close(evaluator.evaluate(stacked, inputs, targets), reference, rtol=0, atol=1e-4)


def test_projection_dtype_matches_model():
    """Projection matrix dtype matches model dtype for memory savings."""
    model1 = nn.Linear(100, 50)
    subspace1 = AdaptiveSubspace(full_dim=100 * 50 + 50, subspace_dim=32)
    opt1 = PolyStepOptimizer(model1, subspace=subspace1, mixed_precision=False, compile=False)
    assert opt1.state.projection.dtype == torch.float32

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

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_displacement_rotation_runs_in_half_precision(self, dtype):
        """CPU LAPACK has no half QR or SVD, so the decomposition must run in fp32
        for both bf16 and fp16 inputs."""
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

    def test_fp16_history_overflow_keeps_displacement_rotation(self):
        """Only nonfinite entries trigger the random fallback."""
        model = nn.Sequential(nn.Linear(16, 12), nn.Linear(12, 4))
        layout = ParamLayout.from_module(model)
        full_dim = sum(p.numel() for p in model.parameters())
        cpu = torch.device("cpu")

        adaptive = AdaptiveSubspace(full_dim=full_dim, subspace_dim=8, svd_ratio_init=0.5, svd_ratio_final=0.5)
        projection = adaptive.init_projection(dtype=torch.float16)
        history = torch.randn(5, full_dim, generator=torch.Generator().manual_seed(0)).mul(30).half()
        history[0, 0] = 300
        assert torch.isinf(history.square()).any()
        rotated = adaptive.rotate(
            projection, 1, 10, history, generator=torch.Generator().manual_seed(1), history_is_full=True
        )
        expected = adaptive._rotate_displacement(
            projection, history, 0.5, cpu, torch.float16, torch.Generator().manual_seed(1), True
        )
        assert torch.equal(rotated, expected)

        hybrid = HybridSubspace.from_layout(
            layout, rank=4, rotation_mode="displacement", rotation_interval=1, svd_ratio_init=0.5
        )
        projections = hybrid.init_projections(cpu, torch.float16)
        history = torch.randn(5, hybrid.subspace_dim, generator=torch.Generator().manual_seed(2)).mul(30).half()
        history[0, 0] = 300
        assert torch.isinf(history.square()).any()
        rotated = hybrid.rotate_all(projections, step=1, total_steps=10, displacement_history=history)
        expected = hybrid._rotate_all_displacement(
            projections, history, hybrid.get_svd_ratio(1, 10), cpu, torch.float16, 1
        )
        assert rotated.keys() == expected.keys()
        assert all(torch.equal(rotated[k], expected[k]) for k in expected)


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
def test_mixed_precision_wide_polytope_on_cuda():
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

    # Drawn once outside the closure: redrawing per call would give every candidate
    # in a step a different objective and the cost matrix would rank noise.
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


def test_barycentric_projection_keeps_vertex_dtype():
    """fp32 transport weights against bf16 vertices return finite bf16."""
    b, V, d = 3, 6, 3
    transport = torch.rand(b, V)  # fp32
    X_vertices = torch.randn(b, V, d, dtype=torch.bfloat16)
    out = _barycentric_projection(transport, X_vertices)
    assert out.dtype == torch.bfloat16
    assert out.shape == (b, d)
    assert torch.isfinite(out.float()).all()


def test_fused_softmax_project_keeps_geometry_dtype():
    """fp32 cost against bf16 geometry returns finite bf16."""
    b, V, d = 3, 6, 3
    C = torch.randn(b, V)  # fp32 cost
    pv = torch.randn(V, d, dtype=torch.bfloat16)
    rot = torch.eye(d, dtype=torch.bfloat16).expand(b, d, d).contiguous()
    X = torch.randn(b, d, dtype=torch.bfloat16)
    X_new, transport = _fused_softmax_project(C, 0.1, pv, rot, 1.0, X)
    assert X_new.dtype == torch.bfloat16
    assert torch.isfinite(X_new.float()).all()


def test_barycentric_normalized_plan_divides_by_marginal():
    """When rows already sum to ``a`` (softmax path), /rowsum == /a exactly."""
    b, V, d = 4, 5, 3
    a = torch.full((b,), 1.0 / b)
    transport = torch.softmax(-torch.randn(b, V), dim=-1) * a.unsqueeze(-1)
    X_vertices = torch.randn(b, V, d)
    got = _barycentric_projection(transport, X_vertices)
    reference = torch.einsum("bkd,bk->bd", X_vertices, transport / a.unsqueeze(-1))
    assert torch.allclose(got, reference, atol=1e-6)


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
    """End-to-end mixed_precision=True must run on CPU with HybridSubspace: the
    parameters have to move and the loss has to stay in range, not merely be non-NaN."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 3))
    before = [p.detach().clone() for p in model.parameters()]
    loss = _run_mixed_precision_steps(solver, model=model)

    assert math.isfinite(loss), loss
    # Cross-entropy over 3 classes starts near ln(3); a step that broke the geometry
    # lands far outside that scale without ever producing a NaN.
    assert 0.0 < loss < 10.0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), "the model never moved"


def test_subspace_keeps_minority_param_dtype():
    """Coordinates carry the layout's dominant dtype while each projection carries its
    own parameter's, so a minority FP64 parameter must not mix dtypes in the
    reconstruction matmul or lose its dtype."""

    class MixedDtype(nn.Module):
        def __init__(self):
            super().__init__()
            self.big = nn.Parameter(torch.randn(20, 20))
            self.small = nn.Parameter(torch.randn(4, 4, dtype=torch.float64))

        def forward(self, x):
            return x @ self.big

    torch.manual_seed(0)
    model = MixedDtype()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    opt = PolyStepOptimizer(model, subspace=subspace, compile=False, seed=0)
    inputs, targets = torch.randn(8, 20), torch.randn(8, 20)
    loss_fn = nn.MSELoss()

    def closure(batched):
        from torch.func import functional_call, vmap

        return vmap(lambda p: loss_fn(functional_call(model, p, (inputs,)), targets))(batched)

    opt.step(closure)

    assert model.small.dtype == torch.float64
    assert model.big.dtype == torch.float32


def test_mixed_precision_refreshes_entry_dtypes():
    """A HybridSubspace built on a mixed-dtype model before the BF16 cast builds every projection at BF16."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3).double())
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    opt = PolyStepOptimizer(model, subspace=subspace, mixed_precision=True, seed=0)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(16, 4), torch.randint(0, 3, (16,))

    loss = opt.step(lambda p: evaluator.evaluate(p, inputs, targets))

    assert {p.dtype for p in opt.state.hybrid_projections.values()} == {torch.bfloat16}
    assert math.isfinite(float(loss))


@pytest.mark.parametrize("minority", [torch.float64, torch.bfloat16])
def test_two_dtype_forward_keeps_projection_dtypes(minority):
    """A model whose forward runs in two dtypes, not just holding an unused one. Each
    per-entry projection must be built at its own parameter's dtype or the
    reconstruction matmul mixes dtypes, and the evaluator must not cast inputs to the
    perturbed layer's dtype. A resync mid-run must keep each projection at its
    parameter's dtype, and a sparse projection must not promote a narrower one."""

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4).to(minority)

        def forward(self, x):
            return self.b(self.a(x).to(minority))

    builders = [
        lambda lo: HybridSubspace.from_layout(lo, rank=2),
        lambda lo: HybridSubspace.from_layout(
            lo, rank=2, rotation_interval=1, rotation_mode="displacement", absorb_mode="periodic", absorb_interval=2
        ),
        lambda lo: HybridSubspace.from_layout(
            lo, rank=2, sparse_threshold_bytes=0, absorb_mode="periodic", absorb_interval=2
        ),
    ]
    for build in builders:
        torch.manual_seed(0)
        model = Mixed()
        opt = PolyStepOptimizer(model, subspace=build(ParamLayout.from_module(model)), seed=0)
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        inputs, targets = torch.randn(8, 8), torch.randint(0, 4, (8,))
        for i in range(4):
            if i == 2:
                opt.resync_from_model()
            opt.register_evaluator(evaluator, inputs, targets)
            opt.step(lambda p: evaluator.evaluate(p, inputs, targets), objective_token=i)

        sd = model.state_dict()
        assert sd["a.weight"].dtype is torch.float32
        assert sd["b.weight"].dtype is minority, "the minority dtype was rewritten"
        assert opt.state.base_params["b.weight"].dtype is minority
        assert model(inputs).dtype is minority


@pytest.mark.parametrize("fused", [False, True])
def test_inplace_half_weights_round_like_batch(fused):
    """The in-place path scores the fallback candidates of a step whose site candidates
    read ``reconstruct_batch`` weights, so a BF16 weight must round the product before
    adding the base, as that does, not in one fused addmm."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).bfloat16()
    if not fused:
        model[1].weight.data = model[1].weight.data.t().contiguous().t()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    projections = subspace.init_projections(torch.device("cpu"), torch.bfloat16)
    if fused:
        subspace.build_fused_projection(projections)
        assert subspace._fused_P is not None
    base_sd = {k: v.detach().clone() for k, v in model.named_parameters()}
    coords = torch.randn(subspace.subspace_dim, generator=torch.Generator().manual_seed(1)).bfloat16() * 3

    subspace.apply_perturbation_inplace(projections, model, base_sd, coords)
    expected = subspace.reconstruct_batch(projections, base_sd, coords.unsqueeze(0))
    for key, param in model.named_parameters():
        torch.testing.assert_close(param.data, expected[key][0], rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_synced_half_weights_match_scored(fused):
    """The step scores ``reconstruct_batch`` weights, which round a half-precision
    product before adding the base, so the weights it writes back must round the same
    way, through the fused matmul when there is one."""
    torch.manual_seed(0)
    if fused:
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 4)).half()
    else:
        model = nn.Sequential(nn.Linear(4, 3).bfloat16(), nn.Linear(3, 1))
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    opt = PolyStepOptimizer(model, subspace=subspace, step_radius=3.0, seed=0)
    subspace = opt.subspace
    assert (subspace._fused_P is not None) is fused
    opt.step(lambda p: sum(v.float().flatten(1).sin().sum(1) for v in p.values()))

    coords = opt.state.X.reshape(-1)[: subspace.subspace_dim]
    assert coords.abs().min() > 0
    expected = subspace.reconstruct_batch(opt.state.hybrid_projections, opt.state.base_params, coords.unsqueeze(0))
    for key, param in model.named_parameters():
        assert torch.equal(param.data, expected[key][0]), key


def test_fused_reconstruction_keeps_sparse_minority_dtype():
    """Two FP32 blocks fuse, so the sparse BF16 layer is rebuilt next to the fused
    matmul and must not be promoted to the coordinates' dtype."""
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 64))
    model[2].bfloat16()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2, sparse_threshold_bytes=10_000)
    projections = subspace.init_projections(torch.device("cpu"), torch.float32)
    subspace.build_fused_projection(projections)
    assert subspace._fused_P is not None
    assert [spec.entry_key for spec, _ in subspace._fused_sparse_specs] == ["2.weight"]

    base_sd = model.state_dict()
    batch = subspace.reconstruct_batch(projections, base_sd, torch.randn(3, subspace.subspace_dim))
    assert {k: v.dtype for k, v in batch.items()} == {k: v.dtype for k, v in base_sd.items()}


def test_bf16_skips_factored_path_until_cast():
    """The low-rank path adds each candidate's change after the matmul, so BF16 weights
    would be ranked on changes they round away."""
    from polystep import FactoredSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    opt = PolyStepOptimizer(model, subspace=subspace, mixed_precision=True, seed=0)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(8, 6), torch.randint(0, 3, (8,))
    opt.register_evaluator(evaluator, inputs, targets)
    assert opt._factored_evaluator is None
    assert opt._subspace_delta_evaluator is None

    model.float()
    opt.resync_from_model()
    opt.register_evaluator(evaluator, inputs, targets)
    assert opt._factored_evaluator is not None
    assert opt._subspace_delta_evaluator is not None
    assert opt.state.X.dtype is torch.bfloat16
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert model[0].weight.dtype is torch.float32


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_resync_after_cast_rebuilds_fused_projection(dtype):
    """Same-dtype dense blocks share one fused projection, so a resync after a cast has
    to rebuild it at the new dtype before the next step reconstructs through it."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    opt = PolyStepOptimizer(model, subspace=subspace, seed=0)
    subspace = opt.subspace
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(16, 8), torch.randint(0, 4, (16,))
    for i in range(4):
        if i == 2:
            model.to(dtype)
            evaluator.reset_vmap()
            opt.resync_from_model()
        opt.register_evaluator(evaluator, inputs, targets)
        opt.step(lambda p: evaluator.evaluate(p, inputs, targets), objective_token=i)
    assert subspace._fused_P.dtype is dtype
    assert model(inputs.to(dtype)).dtype is dtype


@pytest.mark.parametrize("rotate", [False, True])
def test_projections_after_resync_follow_cast(rotate):
    """A rank transition or a rotation after the cast draws each projection at its
    parameter's new dtype, not at the coordinates' or the construction-time one."""
    from polystep import RankSchedule

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4).bfloat16()

        def forward(self, x):
            return self.b(self.a(x.to(self.a.weight.dtype)).to(self.b.weight.dtype))

    torch.manual_seed(0)
    model = Mixed()
    layout = ParamLayout.from_module(model)
    if rotate:
        subspace = HybridSubspace.from_layout(layout, rank=1, rotation_interval=1, rotation_mode="random")
        schedule = None
    else:
        subspace, schedule = HybridSubspace.from_layout(layout, rank=1), RankSchedule([(0, 1), (3, 2)])
    opt = PolyStepOptimizer(model, subspace=subspace, rank_schedule=schedule, seed=0)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(16, 8), torch.randint(0, 4, (16,))
    for i in range(5):
        if i == 2:
            model.double()
            evaluator.reset_vmap()
            opt.resync_from_model()
        opt.register_evaluator(evaluator, inputs, targets)
        opt.step(lambda p: evaluator.evaluate(p, inputs, targets), objective_token=i)
    assert opt._applied_rank == (None if rotate else 2)
    dtypes = {p.dtype for p in opt.state.hybrid_projections.values() if isinstance(p, torch.Tensor)}
    assert dtypes == {torch.float64}


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_resync_casts_global_projection_and_coords(dtype):
    """One global projection multiplies the coordinates straight into the parameters, so
    both have to follow the model's dtype."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    subspace = AdaptiveSubspace.from_layout(ParamLayout.from_module(model), rank=16)
    opt = PolyStepOptimizer(model, subspace=subspace, use_momentum=True, seed=0)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(16, 8), torch.randint(0, 4, (16,))
    for i in range(4):
        if i == 2:
            model.to(dtype)
            evaluator.reset_vmap()
            opt.resync_from_model()
        opt.register_evaluator(evaluator, inputs, targets)
        opt.step(lambda p: evaluator.evaluate(p, inputs, targets), objective_token=i)
    assert opt.state.projection.dtype is dtype
    assert opt.state.X.dtype is dtype
    assert opt.state.velocity.dtype is dtype
    assert model[0].weight.dtype is dtype


@pytest.mark.parametrize("register", [True, False])
def test_resync_after_dtype_cast_steps(register):
    """Full space follows a cast to float64 and back to float32."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    opt = PolyStepOptimizer(model, use_momentum=True, seed=0)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(16, 8), torch.randint(0, 4, (16,))
    losses = []
    for i, dtype in enumerate([None, None, torch.float64, None, torch.float32, None]):
        if dtype is not None:
            model.to(dtype)
            evaluator.reset_vmap()
            opt.resync_from_model()
            assert opt.state.X.dtype is dtype
        if register:
            opt.register_evaluator(evaluator, inputs, targets)
        losses.append(float(opt.step(lambda p: evaluator.evaluate(p, inputs, targets), objective_token=i)))
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
    assert opt.state.velocity.dtype is torch.float32


def test_global_vector_modes_reject_mixed_dtype():
    """Full space and AdaptiveSubspace hold every parameter in one vector at one dtype.

    A minority-dtype weight would be optimized at the majority's precision, so this has to
    raise rather than downgrade it silently.
    """

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


def _biased_fallback_step(dtype):
    torch.manual_seed(0)
    model = nn.Linear(6, 3).to(dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(3.0)
    X, Y = torch.randn(64, 6), torch.randn(64, 3)

    def closure(bp):
        pred = torch.einsum("cij,nj->cni", bp["weight"].float(), X) + bp["bias"].float()[:, None, :]
        return ((pred - Y) ** 2).mean(dim=(1, 2))

    opt = PolyStepOptimizer(
        model, particle_dim=3, biased_rotation=True, polytope_type="simplex", epsilon=0.3, step_radius=0.05, seed=0
    )
    before = nn.utils.parameters_to_vector(model.parameters()).float()
    with pytest.warns(UserWarning, match="OT barycenter"):
        opt.step(closure)
    step = nn.utils.parameters_to_vector(model.parameters()).float() - before
    return opt._prev_descent_direction, step.view_as(opt._prev_descent_direction)


def test_biased_fallback_follows_barycentre_step():
    direction, step = _biased_fallback_step(torch.float32)
    torch.testing.assert_close(direction, step)


def test_biased_fallback_keeps_sub_bf16_steps():
    """Near 3 a bf16 parameter rounds some 0.05 steps to zero; the bias direction must not."""
    direction, step = _biased_fallback_step(torch.bfloat16)
    assert direction.dtype == torch.float32
    assert (step.abs().amax(-1) == 0).any()
    assert (direction.abs().amax(-1) > 0).all()


def test_blockwise_biased_rotation_runs_under_bfloat16():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    opt = PolyStepOptimizer(model, seed=0, mixed_precision=True, block_strategy="per_layer", biased_rotation=True)

    def closure(params):
        v = next(iter(params.values()))
        return v.reshape(v.shape[0], -1).sum(1).float()

    losses = [opt.step(closure) for _ in range(3)]

    assert all(math.isfinite(loss) for loss in losses)
    # Reaching step 3 with finite particles is the contract, not just the absence of an exception.
    assert opt.state.iteration_count == 3
    assert torch.isfinite(opt.state.X).all()
