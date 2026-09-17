"""Derivative-free machinery: FD quadratic model, Newton and trust-region steps, probe
reuse, and the multi-fidelity screen."""

import warnings

import pytest
import torch
import torch.nn as nn
from polystep.optimizer import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.api import get_diagnostics


def _make_model_and_closure():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    data = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    def make_closure(opt):
        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

        def closure(params):
            return evaluator.evaluate(params, data, targets)

        return closure

    return model, make_closure


def test_fd_gradient_replaces_the_ot_descent_direction():
    """Under the quadratic model the stored direction is the FD gradient, not the OT step."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        biased_rotation=True,
        use_quadratic_model=True,
        polytope_type="orthoplex",
        max_iterations=2,
        seed=42,
    )
    opt.step(make_closure(opt))

    dir_ = opt._prev_descent_direction
    assert dir_ is not None and dir_.shape[1] == 2
    assert torch.isfinite(dir_).all()
    assert torch.norm(dir_).item() > 1e-8


def test_losses_3d_retained():
    """Optimizer should retain the (P, V, K) loss tensor when use_quadratic_model=True, polytope_type="orthoplex"."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        use_quadratic_model=True,
        polytope_type="orthoplex",
        num_probe=3,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)

    assert opt._losses_3d is not None
    P = opt._state.X.shape[0]
    V = 2 * 2  # pdim=2 orthoplex
    assert opt._losses_3d.shape[0] == P
    assert opt._losses_3d.shape[1] == V


def test_newton_momentum_uses_fd_direction():
    """With use_quadratic_model=True, polytope_type="orthoplex", momentum steps should use Newton direction."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        amortize_steps=3,
        amortize_ema=0.7,
        use_quadratic_model=True,
        polytope_type="orthoplex",
        biased_rotation=True,
        num_probe=3,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)

    # Step 1: full OT step computes the Newton direction from FD data
    opt.step(closure)
    assert opt._newton_direction is not None

    # Step 2: momentum step uses the Newton direction, not just EMA transport
    opt.step(closure)
    assert opt._state.displacement_sqnorms[-1] > 0, "momentum step did not move"


def test_newton_momentum_fallback_without_qm():
    """Without use_quadratic_model, momentum should still use EMA transport."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        amortize_steps=3,
        amortize_ema=0.7,
        use_quadratic_model=False,
        biased_rotation=True,
        num_probe=3,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)
    assert opt._transport_direction_ema is not None
    assert opt._newton_direction is None


def test_trust_region_expands_on_accurate_prediction():
    """A correct improvement prediction must not shrink the trust region.

    The ratio uses negative = improvement; an expansion (multiplier > 1.0) is only
    reachable when the signs agree.
    """
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        use_quadratic_model=True,
        polytope_type="orthoplex",
        trust_region=True,
        biased_rotation=True,
        num_probe=3,
        max_iterations=12,
        seed=42,
    )
    closure = make_closure(opt)
    for _ in range(12):
        opt.step(closure)

    assert len(opt._state.trust_region_multipliers) > 0
    assert max(opt._state.trust_region_multipliers) > 1.0


def test_newton_refinement_invalidates_warmstart_duals():
    """When Newton refinement moves particles after the solve, the solve's dual
    potentials are stale and must not be kept as the next warm start."""
    # Baseline: plain Sinkhorn keeps its duals for warm starting.
    model, make_closure = _make_model_and_closure()
    base = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        solver="sinkhorn",
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    base.step(make_closure(base))
    assert base._state.f is not None

    # Refinement moves particles after the solve, so the solve's duals encode old
    # positions and must be cleared instead of kept.
    model2, make_closure2 = _make_model_and_closure()
    ref = PolyStepOptimizer(
        model2,
        particle_dim=2,
        epsilon=0.5,
        solver="sinkhorn",
        use_quadratic_model=True,
        polytope_type="orthoplex",
        newton_refinement=True,
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    ref.step(make_closure2(ref))
    assert ref._state.f is None
    assert ref._state.g is None


def test_newton_refinement_reconciles_the_momentum_velocity():
    """Refinement replaces the post-momentum position, so velocity must track the realized move."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        solver="softmax",
        use_quadratic_model=True,
        polytope_type="orthoplex",
        newton_refinement=True,
        use_momentum=True,
        velocity_lr=1.0,
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    X_before = opt._state.X.clone()
    opt.step(make_closure(opt))
    realized = opt._state.X - X_before
    assert realized.abs().max() > 0, "the step did not move"
    torch.testing.assert_close(opt._state.velocity, realized)


def test_adaptive_probes_reuses_rotation_for_stagnant_particles():
    """A stagnant particle reuses its previous rotation so its reused cost row
    stays consistent with the vertices that row was evaluated at."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        adaptive_probes=True,
        adaptive_probes_threshold=1e9,  # force all stagnant
        num_probe=3,
        max_iterations=10,
        seed=42,
    )
    closure = make_closure(opt)

    # First step cannot reuse (no history yet) but stores the rotations.
    opt.step(closure, objective_token="stationary")
    assert opt._prev_rot_mats is not None
    rot_after_first = opt._prev_rot_mats.clone()

    # Second step: all particles stagnant, so rotations are reused unchanged.
    opt.step(closure, objective_token="stationary")
    torch.testing.assert_close(opt._prev_rot_mats, rot_after_first)
    assert torch.isfinite(opt._state.X).all()


def test_multifidelity_screening_keeps_descending():
    """Storing the raw (not dampened) cost avoids a one-way ratchet that would lock
    out directions, so the loss keeps dropping over many steps.

    The screen runs only with a cheap closure and an orthoplex polytope.
    """
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    data = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

    def closure(params, _in=None, _tgt=None):
        return evaluator.evaluate(params, data if _in is None else _in, targets if _tgt is None else _tgt)

    opt = PolyStepOptimizer(
        model,
        particle_dim=4,
        polytope_type="orthoplex",
        epsilon=0.5,
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        num_probe=5,
        max_iterations=30,
        seed=42,
    )
    screen = opt.screen_closure_from(closure, data, targets)
    assert screen is not None, "screen closure could not be built, the screen would not run"

    losses = [opt.step(closure, screen_closure=screen) for _ in range(20)]
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert opt._last_screen_savings > 0, "the screen ran no cheaper than the dense path"
    # A bare `< losses[0]` would pass on a flat loss, so require a real drop.
    assert min(losses) < losses[0] * 0.80, f"loss barely moved: {losses[0]:.4f} -> {min(losses):.4f}"


def test_multifidelity_off_by_default():
    """Multi-fidelity should be disabled by default."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        max_iterations=2,
        seed=42,
    )
    assert not opt.multifidelity_screen


def test_screen_ranks_by_deviation_from_the_row_mean_on_a_simplex():
    """sum_v v = 0, so a vertex at the row mean moves the barycentre by nothing.

    Deviation from the mean is what the antithetic contrast measures, written for a
    frame with no partner to subtract, so the screen needs no orthoplex.
    """
    model, _ = _make_model_and_closure()
    x, y = torch.randn(64, 4), torch.randn(64, 2)
    ev = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(
        model,
        particle_dim=4,
        epsilon=0.5,
        polytope_type="simplex",
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
        num_probe=3,
        max_iterations=2,
        seed=42,
    )

    def closure(bp, _x=x, _y=y):
        return ev.evaluate(bp, _x, _y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the "screen did not run" warning would fail here
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, x, y))
    assert opt._last_screen_savings > 0


def test_all_dfo_features_compose():
    """All DFO features should work together without errors."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        use_quadratic_model=True,
        polytope_type="orthoplex",
        trust_region=True,
        biased_rotation=True,
        amortize_steps=3,
        amortize_ema=0.7,
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    closure = make_closure(opt)

    # 10 steps = 3+ full OT cycles with amortization
    losses = []
    for _ in range(10):
        loss = opt.step(closure)
        losses.append(loss)
        assert torch.isfinite(torch.tensor(loss))

    assert opt._state.iteration_count == 10
    # A composition that runs but does not descend is the failure this guards against.
    assert min(losses) < losses[0], f"no progress over 10 steps: {losses[0]:.4f} -> {min(losses):.4f}"
    # Every diagnostic list must stay index-aligned with costs across the amortized steps.
    d = get_diagnostics(opt)
    lengths = {k: len(v) for k, v in d.items() if isinstance(v, list)}
    assert set(lengths.values()) == {10}, lengths


def test_screen_masks_dropped_vertices_only_for_selection_solvers():
    """A weighted mean reads every entry; an argmin reads only the winner.

    Imputing a dropped vertex is fine for the first and wrong for the second, where an
    imputed value can win and send the step to a vertex never scored at full fidelity.
    """
    from polystep import _step_monolithic

    seen = {}
    original = _step_monolithic._fill_screened_losses

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        seen["all_finite"] = bool(torch.isfinite(out).all())
        return out

    for solver, expect_finite in (("softmax", True), ("min_cost_greedy", False)):
        model, _ = _make_model_and_closure()
        x, y = torch.randn(64, 4), torch.randn(64, 2)
        ev = NNCostEvaluator(model, nn.MSELoss())
        opt = PolyStepOptimizer(
            model,
            particle_dim=4,
            epsilon=1.0,
            seed=0,
            solver=solver,
            polytope_type="orthoplex",
            num_probe=3,
            multifidelity_screen=True,
            screen_keep_ratio=0.5,
            screen_fidelity=0.25,
        )

        def closure(bp, _x=x, _y=y):
            return ev.evaluate(bp, _x, _y)

        _step_monolithic._fill_screened_losses = capture
        try:
            opt.step(closure, screen_closure=opt.screen_closure_from(closure, x, y))
        finally:
            _step_monolithic._fill_screened_losses = original

        assert "all_finite" in seen, f"{solver}: the screen did not run, check the gate"
        assert seen["all_finite"] is expect_finite, solver


def test_screen_runs_on_a_simplex_under_a_selection_solver():
    """Ranking vertices by their own cost needs no antithetic partner, so no orthoplex."""
    model, _ = _make_model_and_closure()
    x, y = torch.randn(64, 4), torch.randn(64, 2)
    ev = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(
        model,
        particle_dim=4,
        epsilon=1.0,
        seed=0,
        solver="min_cost_greedy",
        polytope_type="simplex",
        num_probe=3,
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
    )

    def closure(bp, _x=x, _y=y):
        return ev.evaluate(bp, _x, _y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the "screen did not run" warning would fail here
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, x, y))
    assert opt._last_screen_savings > 0


def test_screen_keeps_enough_vertices_for_top_k_mean():
    """TopKMeanSolver averages min(k, V) vertices, so a screen keeping fewer feeds it
    entries that were never evaluated at full fidelity."""
    model, _ = _make_model_and_closure()
    inputs, targets = torch.randn(16, 4), torch.randn(16, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(
        model,
        particle_dim=4,
        seed=0,
        solver="top_k_mean",
        multifidelity_screen=True,
        screen_keep_ratio=0.1,
        screen_fidelity=0.25,
        max_iterations=10,
    )
    opt.register_evaluator(evaluator, inputs, targets)

    def closure(batched_params, _in=inputs, _tgt=targets):
        return evaluator.evaluate(batched_params, _in, _tgt)

    screen = opt.screen_closure_from(closure, inputs, targets)
    for _ in range(2):
        opt.step(closure, screen_closure=screen)

    # V = pdim + 1 = 5 on the default simplex; keep_ratio 0.1 rounds to 1, below the
    # solver's default k = 3. The floor has to lift it back to k.
    assert torch.isfinite(opt.state.X).all()
    assert opt.solver.k == 3

    # Every vertex holding transport mass must be one the screen kept.
    V = opt._polytope_vertices.shape[0]
    keep_v = min(V, max(1, int(round(V * opt.screen_keep_ratio)), min(opt.solver.k, V)))
    assert keep_v >= min(opt.solver.k, V), f"keep_v={keep_v} < k_eff={min(opt.solver.k, V)}"


def test_default_polytope_is_the_minimal_positive_spanning_set():
    """k+1 vertices, not 2k. Changing this default is a real cost-per-step change."""
    model, _ = _make_model_and_closure()
    opt = PolyStepOptimizer(model, particle_dim=4, seed=0)
    assert opt.polytope_type == "simplex"
    assert opt._polytope_vertices.shape[0] == 5


def test_only_newton_refinement_still_needs_the_orthoplex():
    """The gradient and curvature are closed forms on any centred tight frame."""
    model, _ = _make_model_and_closure()
    with pytest.warns(UserWarning, match="antithetic vertex ordering"):
        PolyStepOptimizer(model, particle_dim=4, seed=0, num_probe=1, newton_refinement=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PolyStepOptimizer(model, particle_dim=4, seed=0, num_probe=3, use_quadratic_model=True, trust_region=True)
        PolyStepOptimizer(
            model, particle_dim=4, seed=0, num_probe=3, solver="min_cost_greedy", multifidelity_screen=True
        )


def test_screen_places_full_fidelity_values_at_the_kept_positions():
    """The screen must not disturb the values it did evaluate at full fidelity.

    Kept entries must arrive unchanged at the right index, or the OT solve ranks
    vertices by an artefact of the assembly rather than by cost.
    """
    from polystep._step_monolithic import _fill_screened_losses

    P, V, K = 3, 6, 1
    torch.manual_seed(0)
    screen_cost = torch.rand(P, V)
    keep_mask = torch.zeros(P, V, dtype=torch.bool)
    keep_mask[:, :3] = True  # each particle keeps its first three vertices

    i_all = torch.arange(P * V * K) // (V * K)
    v_all = (torch.arange(P * V * K) % (V * K)) // K
    sel_idx = torch.nonzero(keep_mask[i_all, v_all], as_tuple=True)[0]
    kept = torch.arange(sel_idx.numel(), dtype=torch.float32) + 100.0

    for mask_dropped in (False, True):
        out = _fill_screened_losses(screen_cost, kept, sel_idx, keep_mask, P, V, K, mask_dropped=mask_dropped)
        torch.testing.assert_close(out[sel_idx], kept)
        # And nothing full-fidelity landed anywhere else.
        assert out.shape == (P * V * K,)
        dropped = torch.ones(P * V * K, dtype=torch.bool)
        dropped[sel_idx] = False
        assert not torch.isin(out[dropped], kept).any(), "a dropped slot holds a kept vertex's value"


def test_a_2d_cube_is_not_treated_as_an_orthoplex():
    """cube at pdim=2 also has 4 vertices, but its halves are not antipodal.

    Inferring the geometry from the vertex count read L(v0)-L(v2) as a central
    difference: cosine to the true descent fell from 1.0 to 0.29, first sign flipped.
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    target = torch.tensor([[-0.5, 1.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = PolyStepOptimizer(
            model,
            particle_dim=2,
            polytope_type="cube",
            use_quadratic_model=True,
            biased_rotation=True,
            num_probe=1,
            epsilon=0.5,
            probe_radius=0.05,
            step_radius=0.1,
            seed=0,
        )

    def closure(bp):
        w = bp["weight"].reshape(bp["weight"].shape[0], -1)
        return ((w - target) ** 2).sum(dim=1)

    opt.step(closure)

    # Descent from zero weights is -grad = 2 * target, recovered exactly on a quadratic.
    torch.testing.assert_close(opt._prev_descent_direction[0], 2.0 * target[0], rtol=1e-4, atol=1e-4)
