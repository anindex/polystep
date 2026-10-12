"""Integration tests for PolyStepOptimizer step, momentum, and adaptive radius."""

import copy
import warnings

import math

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from polystep.solver import PolyStep
from polystep import CosineEpsilon, PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def _make_model():
    """Small MLP for fast testing."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _make_closure(model):
    """Create a batched closure using NNCostEvaluator."""
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 1)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return closure


@pytest.fixture
def model():
    torch.manual_seed(42)
    return _make_model()


@pytest.fixture
def closure(model):
    return _make_closure(model)


@pytest.fixture
def optimizer(model):
    return PolyStepOptimizer(
        model,
        max_iterations=50,
        epsilon=0.1,
        sinkhorn_max_iters=100,
        compile=False,
        seed=42,
    )


class TestClosureInterface:
    def test_each_step_moves_model_records_one_cost(self, model, optimizer, closure):
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        assert optimizer.state.iteration_count == 0

        for i in range(5):
            optimizer.step(closure)
            assert optimizer.state.iteration_count == i + 1
            assert len(optimizer.state.costs) == i + 1

        updated = model.state_dict()
        assert any(not torch.equal(initial[k], updated[k]) for k in initial), (
            "Model parameters should change after step"
        )

    @pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer"])
    def test_scalar_closure_raises_instead_of_broadcasting(self, block_strategy):
        """Broadcasting would score every candidate alike and make the step 0."""
        torch.manual_seed(0)
        opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, block_strategy=block_strategy)
        with pytest.raises(ValueError, match="one loss per candidate"):
            opt.step(lambda bp: torch.tensor(3.0))


class TestMomentum:
    def test_momentum_initializes_velocity(self, model):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
        )
        assert opt.state.velocity is not None
        assert torch.all(opt.state.velocity == 0)

    def test_momentum_updates_velocity(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
        )
        opt.step(closure)
        assert opt.state.velocity is not None
        assert torch.any(opt.state.velocity != 0)

    def test_momentum_smooths_trajectory(self):
        """Momentum must accumulate velocity and change the trajectory."""
        torch.manual_seed(42)
        model_no_mom = _make_model()
        model_mom = copy.deepcopy(model_no_mom)
        closure_no_mom = _make_closure(model_no_mom)
        closure_mom = _make_closure(model_mom)

        opt_no_mom = PolyStepOptimizer(
            model_no_mom,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt_mom = PolyStepOptimizer(
            model_mom,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
            momentum_init=0.5,
            momentum_final=0.95,
        )

        n_steps = 10
        for _ in range(n_steps):
            opt_no_mom.step(closure_no_mom)
            opt_mom.step(closure_mom)

        assert len(opt_no_mom.state.displacement_sqnorms) == n_steps
        assert len(opt_mom.state.displacement_sqnorms) == n_steps

        # Both runs are seeded identically, so any trajectory difference is due
        # to momentum.
        assert opt_mom.state.velocity is not None
        assert opt_mom.state.velocity.abs().sum() > 0
        traj_no_mom = torch.tensor(opt_no_mom.state.displacement_sqnorms)
        traj_mom = torch.tensor(opt_mom.state.displacement_sqnorms)
        assert not torch.allclose(traj_no_mom, traj_mom)


class TestAdaptiveRadius:
    def test_adaptive_disabled_by_default(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt.step(closure)
        assert opt.state.radius_multiplier == 1.0

    def test_radius_stays_in_bounds(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_adaptive_radius=True,
            radius_min=0.5,
            radius_max=3.0,
        )
        seen = []
        for _ in range(20):
            opt.step(closure)
            seen.append(opt.state.radius_multiplier)
        assert 0.5 <= opt.state.radius_multiplier <= 3.0
        # A controller pinned at 1.0 satisfies the bounds without ever adapting.
        assert any(m != 1.0 for m in seen), "the radius multiplier never moved"


class TestIntegration:
    def test_momentum_and_adaptive_together(self, model, closure):
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            use_momentum=True,
            use_adaptive_radius=True,
        )
        for _ in range(10):
            opt.step(closure)
        assert opt.state.velocity is not None
        assert torch.any(opt.state.velocity != 0)
        assert 0.5 <= opt.state.radius_multiplier <= 3.0
        assert opt.state.iteration_count == 10


class TestParticleDim:
    @pytest.mark.parametrize(
        "polytope, dim, verts", [("orthoplex", 4, 8), ("orthoplex", 8, 16), ("simplex", 4, 5), ("simplex", 8, 9)]
    )
    def test_particle_dim_4(self, polytope, dim, verts):
        """particle_dim=D gives D-dim particles and the polytope's own vertex count.

        The simplex has D+1 vertices, the orthoplex 2*D.
        """
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            particle_dim=dim,
            polytope_type=polytope,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        assert opt.layout.particle_dim == dim
        assert opt._particle_dim == dim
        assert opt._polytope_vertices.shape[0] == verts
        assert opt._polytope_vertices.shape[1] == dim

    def test_particle_dim_step(self):
        """particle_dim=4 optimizer can run step() without error and updates model."""
        torch.manual_seed(42)
        model = _make_model()
        initial_params = {k: v.clone() for k, v in model.state_dict().items()}

        opt = PolyStepOptimizer(
            model,
            particle_dim=4,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        closure = _make_closure(model)
        loss = opt.step(closure)

        assert math.isfinite(loss)
        updated_params = model.state_dict()
        any_changed = any(not torch.equal(initial_params[k], updated_params[k]) for k in initial_params)
        assert any_changed, "Model parameters should change after step with particle_dim=4"

    def test_particle_dim_invalid(self):
        """ValueError for particle_dim < 2."""
        torch.manual_seed(42)
        model = _make_model()
        with pytest.raises(ValueError, match="particle_dim must be >= 2"):
            PolyStepOptimizer(
                model,
                particle_dim=1,
                max_iterations=50,
                epsilon=0.1,
                compile=False,
            )

    def test_particle_dim_cube_warning(self):
        """UserWarning when particle_dim=8 and polytope_type='cube'."""
        torch.manual_seed(42)
        model = _make_model()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PolyStepOptimizer(
                model,
                particle_dim=8,
                polytope_type="cube",
                max_iterations=50,
                epsilon=0.1,
                sinkhorn_max_iters=100,
                compile=False,
                seed=42,
            )
            cube_warnings = [x for x in w if "cube" in str(x.message).lower()]
            assert len(cube_warnings) >= 1, f"Expected cube warning, got: {[str(x.message) for x in w]}"


class TestDualMomentum:
    def test_beta_zero_matches_standard(self):
        """dual_momentum_beta=0.0 produces identical cost to default optimizer over 3 steps."""
        torch.manual_seed(42)
        model_default = _make_model()
        model_beta0 = copy.deepcopy(model_default)

        evaluator_default = NNCostEvaluator(model_default, loss_fn=nn.MSELoss())
        evaluator_beta0 = NNCostEvaluator(model_beta0, loss_fn=nn.MSELoss())
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 1)

        def closure_default(bp):
            return evaluator_default.evaluate(bp, inputs, targets)

        def closure_beta0(bp):
            return evaluator_beta0.evaluate(bp, inputs, targets)

        opt_default = PolyStepOptimizer(
            model_default,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
        )
        opt_beta0 = PolyStepOptimizer(
            model_beta0,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            dual_momentum_beta=0.0,
        )

        for _ in range(3):
            cost_default = opt_default.step(closure_default)
            cost_beta0 = opt_beta0.step(closure_beta0)
            assert cost_default == pytest.approx(cost_beta0, rel=1e-6), (
                f"beta=0 should match default: {cost_beta0} vs {cost_default}"
            )

    def test_extrapolation_applied(self):
        """After 2+ steps, state.prev_prev_f is not None (history is being tracked)."""
        torch.manual_seed(42)
        model = _make_model()
        closure = _make_closure(model)
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            dual_momentum_beta=0.3,
        )
        opt.step(closure)
        opt.step(closure)
        assert opt.state.prev_prev_f is not None, "After 2 steps, prev_prev_f should be set"
        assert opt.state.prev_prev_g is not None, "After 2 steps, prev_prev_g should be set"

    def test_extrapolation_clamped(self):
        """The extrapolated warm start handed to the solver stays inside the clamp.

        Checked on the value passed to solve(): sinkhorn clamps its output again
        internally, so the output cannot show whether this clamp ran.
        """
        torch.manual_seed(42)
        model = _make_model()
        closure = _make_closure(model)
        opt = PolyStepOptimizer(
            model,
            max_iterations=50,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            compile=False,
            seed=42,
            dual_momentum_beta=0.3,
        )
        seen = []
        real_solve = opt.solver.solve

        def spy(*args, **kwargs):
            seen.append((kwargs.get("init_f"), kwargs.get("init_g")))
            return real_solve(*args, **kwargs)

        opt.solver.solve = spy
        for _ in range(5):
            opt.step(closure)

        max_abs = 80.0 * max(opt.state.epsilon, 0.01)
        warm = [f for f, _ in seen if f is not None]
        assert warm, "no warm start was ever passed, so the clamp was never reached"
        for f in warm:
            assert f.abs().max().item() <= max_abs + 1e-6, f"warm-start dual exceeds max_abs={max_abs}"


class TestSinkhornParamWiring:
    """Verify anderson_depth and adaptive_omega wire through
    from PolyStepOptimizer constructor to the internal SinkhornSolver."""

    def test_custom_sinkhorn_params_wired(self):
        """PolyStepOptimizer(anderson_depth=5, adaptive_omega=True) creates solver
        with those exact values."""
        torch.manual_seed(42)
        model = _make_model()
        opt = PolyStepOptimizer(
            model,
            compile=False,
            seed=42,
            anderson_depth=5,
            adaptive_omega=True,
        )
        assert opt.solver.anderson_depth == 5
        assert opt.solver.adaptive_omega is True


class TestProgressiveEpsilon:
    def test_progressive_epsilon_adjusts_from_solver_feedback(self):
        from polystep import ProgressiveEpsilon

        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        optimizer = PolyStepOptimizer(
            model,
            epsilon=ProgressiveEpsilon(init=1.0, target=0.1, max_epsilon=5.0),
            step_radius=0.5,
            num_probe=2,
            compile=False,
            seed=42,
        )
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8,))
        loss_fn = nn.CrossEntropyLoss()

        def closure(batched_params):
            evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
            return evaluator.evaluate(batched_params, inputs, targets)

        eps_values = []
        for _ in range(5):
            optimizer.step(closure)
            eps_values.append(optimizer._progressive_epsilon.at())

        assert len(set(round(e, 6) for e in eps_values)) > 1, f"Epsilon did not change across steps: {eps_values}"
        assert all(0 < e < 100 for e in eps_values)


def test_both_entry_points_default_num_probe_one():
    """K=1 is the reported and optimal setting; the two APIs must not disagree."""

    def objective(x):
        return (x**2).sum(-1)

    assert PolyStepOptimizer(nn.Linear(4, 2, bias=False), epsilon=0.5).num_probe == 1
    assert PolyStep.create(objective, dim=4).num_probe == 1


class _FakeLIF(nn.Module):
    """Stand-in for an snnTorch Leaky neuron, without the snntorch dependency."""

    def forward(self, x):
        return torch.relu(x)


class _SNNStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.lif1 = _FakeLIF()  # the name pattern the guard matches on
        self.fc2 = nn.Linear(8, 2)


@pytest.mark.parametrize("build, expect_warning", [(_SNNStub, True), (_make_model, False)])
def test_cosine_step_radius_warns_only_for_snn(build, expect_warning):
    """A cosine step_radius schedule collapses hard-LIF SNN accuracy, so it warns only there."""
    cosine = CosineEpsilon(init=5.0, target=1.0, decay=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PolyStepOptimizer(build(), epsilon=0.5, step_radius=cosine)

    hits = [
        m
        for m in (str(w.message).lower() for w in caught)
        if any(k in m for k in ("snn", "leaky", "lif", "spik")) and ("step_radius" in m or "cosine" in m)
    ]
    assert bool(hits) is expect_warning, hits


def test_replaced_layer_receives_next_update():
    torch.manual_seed(0)
    model = _make_model()
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs, targets = torch.randn(16, 4), torch.randn(16, 1)
    opt = PolyStepOptimizer(model, epsilon=0.1, sinkhorn_max_iters=100, seed=42)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    opt.register_evaluator(evaluator, inputs, targets)
    opt.step(closure)

    orphan = model[0]
    orphan_weight = orphan.weight.detach().clone()
    model[0] = nn.Linear(4, 8)
    replacement_weight = model[0].weight.detach().clone()
    evaluator.reset_vmap()
    opt.register_evaluator(evaluator, inputs, targets)
    opt.resync_from_model()
    opt.step(closure)

    assert not torch.equal(model[0].weight, replacement_weight)
    assert torch.equal(orphan.weight, orphan_weight)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_resync_rejects_device_move():
    """resync_from_model() re-reads the weights; it does not move the optimizer state."""
    model = _make_model()
    opt = PolyStepOptimizer(model, seed=0)
    model.cuda()
    with pytest.raises(ValueError, match="moved from cpu to cuda"):
        opt.resync_from_model()


@pytest.mark.parametrize("mode", ["mean", "max_cost"])
@pytest.mark.parametrize(
    ("solver", "block_strategy"),
    [("sinkhorn", "monolithic"), ("softmax", "monolithic"), ("sinkhorn", "per_layer"), ("softmax", "per_layer")],
)
def test_nan_candidate_excluded_from_step_divisor(monkeypatch, solver, block_strategy, mode):
    """Finite losses span under 0.01, so the divisor must too.

    Handing every solver, and the fused softmax, the penalty in place of the NaN
    would set the divisor near 0.5 under 'mean' and 3.3 under 'max_cost'.
    """
    import polystep.solvers._shared as shared

    divisors = []
    resolve = shared.resolve_cost_scale

    def spy(*args):
        divisors.append(resolve(*args))
        return divisors[-1]

    monkeypatch.setattr(shared, "resolve_cost_scale", spy)
    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    opt = PolyStepOptimizer(
        model, solver=solver, scale_cost=mode, use_momentum=False, block_strategy=block_strategy, seed=0
    )

    def closure(params):
        w = params["weight"].reshape(-1, 4)
        loss = 2.30 + 0.01 * torch.sigmoid(w.sum(-1))
        loss[w[:, 0].argmax()] = float("nan")
        return loss

    opt.step(closure)

    assert divisors
    assert max(d.item() for d in divisors) < 0.01


@pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer"])
def test_softmax_avoids_nan_probe_vertex(block_strategy):
    """Vertex 0 scores -10 on one probe and NaN on the other; every other probe scores 10.

    Averaging the per-probe penalty into it would give vertex 0 a cost of (-10 + 21) / 2,
    the lowest of the four, and step the fused softmax toward an infeasible vertex.
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 1, bias=False)
    opt = PolyStepOptimizer(
        model, solver="softmax", polytope_type="orthoplex", num_probe=2, block_strategy=block_strategy, seed=0
    )
    start = model.weight.detach().reshape(-1).clone()
    toward = []

    def closure(params):
        w = params["weight"].reshape(-1, 2)
        toward.append(w[0] - start)
        loss = torch.full((w.shape[0],), 10.0)
        loss[0], loss[1] = -10.0, float("nan")
        return loss

    opt.step(closure)

    assert len(toward) == 1
    assert torch.dot(model.weight.detach().reshape(-1) - start, toward[0]) < 0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_overflowing_probe_mean_gets_penalty(sign):
    """Vertex 0 scores +-[3e38, 3e38, -3e38], finite on every probe, and the rest score 1.

    The FP32 mean overflows to +-inf, which the step must not record as its loss.
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 1, bias=False)
    opt = PolyStepOptimizer(model, polytope_type="orthoplex", num_probe=3, seed=0)

    def closure(params):
        loss = torch.ones(params["weight"].reshape(-1, 2).shape[0])
        loss[:3] = sign * torch.tensor([3e38, 3e38, -3e38])
        return loss

    assert opt.step(closure) == pytest.approx(1.5)


def test_single_particle_moves():
    """Balanced OT with one row forces a uniform plan, which would leave the iterate in place."""
    torch.manual_seed(0)
    model = nn.Linear(1, 1, bias=False)
    inputs, targets = torch.randn(8, 1), torch.randn(8, 1)
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    start = model.weight.detach().clone()

    with pytest.warns(UserWarning, match="num_particles=1"):
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=20, seed=1)
    assert opt.state.X.shape[0] == 1

    for _ in range(5):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets))
    assert not torch.equal(start, model.weight.detach())


def test_explicit_sinkhorn_with_one_particle_warns():
    torch.manual_seed(0)
    model = nn.Linear(1, 1, bias=False)
    with pytest.warns(UserWarning, match="uniform transport plan"):
        PolyStepOptimizer(model, solver="sinkhorn", epsilon=0.1, max_iterations=20)


class TestNaNRevertRestoresState:
    """A NaN step must clear the derived state, not merely leave it finite.

    A partial revert keeps running and stays finite, then poisons later steps through
    momentum or the transport EMA. A NaN loss does not reach this path
    (``sanitize_cost`` substitutes a penalty); a poisoned X or velocity does.
    """

    @staticmethod
    def _build(**kwargs):
        model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        opt = PolyStepOptimizer(model, epsilon=0.1, seed=0, compile=False, **kwargs)
        x = torch.randn(8, 8, generator=torch.Generator().manual_seed(0))

        def closure(batched_params):
            n = next(iter(batched_params.values())).shape[0]
            return torch.stack(
                [
                    functional_call(model, {k: v[i] for k, v in batched_params.items()}, (x,)).pow(2).mean()
                    for i in range(n)
                ]
            )

        return opt, closure

    def test_poisoned_velocity_reverts_x_to_pre_step(self):
        """Pre-step X is finite here, so the revert restores it rather than the origin."""
        opt, closure = self._build(use_momentum=True, momentum_init=0.9, momentum_final=0.9)
        opt.step(closure)
        before = opt.state.X.clone()

        opt.state.velocity = torch.full_like(opt.state.velocity, float("inf"))
        opt.step(closure)

        torch.testing.assert_close(opt.state.X, before, rtol=0, atol=0)
        assert torch.count_nonzero(opt.state.velocity) == 0, "velocity survived the revert"
        assert opt.state.displacement_sqnorms[-1] == 0.0, "reported a move that did not happen"

    def test_poisoned_x_reverts_to_origin(self):
        """With no finite point to return to, the revert falls back to the origin."""
        opt, closure = self._build(use_momentum=True)
        opt.step(closure)

        opt.state.X = torch.full_like(opt.state.X, float("nan"))
        opt.step(closure)

        assert torch.count_nonzero(opt.state.X) == 0
        assert torch.count_nonzero(opt.state.velocity) == 0

    def test_amortized_transport_direction_is_dropped(self):
        opt, closure = self._build(amortize_steps=3)
        opt.step(closure)
        assert opt._transport_direction_ema is not None

        opt.state.X = torch.full_like(opt.state.X, float("nan"))
        opt.step(closure)
        assert opt._transport_direction_ema is None


def test_float64_tiny_cost_gaps_move_particles():
    torch.manual_seed(0)
    model = nn.Linear(4, 2).double()
    opt = PolyStepOptimizer(model, seed=0, step_radius=0.5, probe_radius=0.5)

    def tiny_differences(params):
        v = next(iter(params.values()))
        return 1.0 + v.reshape(v.shape[0], -1)[:, 0] * 1e-11

    start = opt.state.X.clone()
    for _ in range(3):
        opt.step(tiny_differences)

    assert not torch.equal(opt.state.X, start)


@pytest.mark.parametrize("chunk_size", [3, 7, None])
def test_chunking_keeps_infeasible_vertex_ranking(chunk_size):
    """Chunking splits the candidate loop; it must not change a single cost.

    ``sanitize_cost`` maps a hard-constraint ``+inf`` to ``2 * max|finite| + 1`` of
    whatever matrix it is handed. Per chunk, that penalty is relative to the costs
    sharing the chunk, so an infeasible vertex among cheap candidates scores below a
    legitimate expensive one elsewhere and the plan transports mass onto it.
    """

    torch.manual_seed(0)
    reference = nn.Linear(4, 2)
    # A fixed threshold, not a batch statistic: infeasibility must be a function of the
    # candidate's own parameters so the same candidates are masked however the loop is
    # split. The 1000x scale gives the finite costs a wide enough spread that a
    # chunk-local penalty lands below them.
    threshold = float(sum(p.detach().pow(2).sum() for p in reference.parameters()) * 1000.0)

    def closure(batched):
        cost = sum(v.flatten(1).pow(2).sum(dim=1) for v in batched.values()) * 1000.0
        return torch.where(cost > threshold, torch.full_like(cost, float("inf")), cost)

    model = copy.deepcopy(reference)
    opt = PolyStepOptimizer(model, compile=False, chunk_size=chunk_size, seed=0)
    loss = opt.step(closure)

    baseline_model = copy.deepcopy(reference)
    baseline = PolyStepOptimizer(baseline_model, compile=False, chunk_size=None, seed=0)
    expected = baseline.step(closure)

    assert loss == pytest.approx(expected)
    assert torch.allclose(model.weight, baseline_model.weight)


def test_all_nonfinite_cost_matrix_reported():
    """Sanitize maps it to a constant, so the step must not report that as a loss."""
    torch.manual_seed(0)
    opt = PolyStepOptimizer(nn.Linear(4, 2), seed=0, use_adaptive_radius=True)

    def nan_closure(bp):
        n = next(iter(bp.values())).shape[0]
        return torch.full((n,), float("nan"))

    with pytest.warns(RuntimeWarning, match="carries no ranking information"):
        opt.step(nan_closure)
    assert opt.state.costs[-1] == float("inf")
    assert opt.state.linear_convergence[-1] is False
    assert opt.state.radius_multiplier == 1.0


def test_blockwise_reports_all_nonfinite_sweep():
    """The sanitize penalty must not be reported as a finite loss with convergence."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    opt = PolyStepOptimizer(model, block_strategy="per_layer", seed=0)

    def nan_closure(bp):
        n = next(iter(bp.values())).shape[0]
        return torch.full((n,), float("nan"))

    with pytest.warns(RuntimeWarning, match="carries no ranking information"):
        opt.step(nan_closure)
    assert opt.state.costs[-1] == float("inf")
    assert opt.state.linear_convergence[-1] is False


@pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer", "grouped"])
def test_step_without_feasible_vertex_warns(block_strategy):
    """Every vertex has one finite probe, which must not read as feasible."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    opt = PolyStepOptimizer(model, block_strategy=block_strategy, num_probe=3, seed=0, use_adaptive_radius=True)

    def closure(bp):
        losses = torch.full((next(iter(bp.values())).shape[0],), float("nan"))
        losses[0::3] = 0.0
        return losses

    with pytest.warns(RuntimeWarning, match="carries no ranking information"):
        assert opt.step(closure) == float("inf")
    assert opt.state.costs[-1] == float("inf")
    assert opt.state.linear_convergence[-1] is False
    assert opt.state.radius_multiplier == 1.0


@pytest.mark.parametrize("kind", [nn.Embedding, nn.EmbeddingBag])
def test_registering_keeps_max_norm_embedding(kind):
    model = kind(2, 4, max_norm=1.0)
    with torch.no_grad():
        model.weight.copy_(torch.arange(2.0, 10.0).reshape(2, 4))
    before = model.weight.detach().clone()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    opt.register_evaluator(ev, torch.tensor([[0], [1]]), torch.zeros(2, 4))
    assert opt._act_ratios is not None
    assert torch.equal(model.weight, before)


class _Noisy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        self.calls += 1
        return x + 0 * torch.randn_like(x)


def test_registering_keeps_train_mode_stats_rng():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), _Noisy(), nn.ReLU(), nn.Linear(8, 2))
    opt = PolyStepOptimizer(model, compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss())
    model.train()
    x, y = torch.randn(32, 4) + 3, torch.randn(32, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    opt.register_evaluator(ev, x, y)
    assert opt._act_ratios is not None
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
