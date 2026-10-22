"""Derivative-free machinery: FD quadratic model, Newton and trust-region steps, and the
multi-fidelity screen."""

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

    def make_closure():
        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

        def closure(params):
            return evaluator.evaluate(params, data, targets)

        return closure

    return model, make_closure


def test_fd_gradient_replaces_ot_direction():
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
    opt.step(make_closure())

    dir_ = opt._prev_descent_direction
    assert dir_ is not None and dir_.shape[1] == 2
    assert torch.isfinite(dir_).all()
    assert torch.norm(dir_).item() > 1e-8


def test_losses_3d_retained():
    """Keeps the (P, V, K) tensor under the orthoplex quadratic model."""
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
    closure = make_closure()
    opt.step(closure)

    assert opt._losses_3d is not None
    P = opt._state.X.shape[0]
    V = 2 * 2  # pdim=2 orthoplex
    assert opt._losses_3d.shape[0] == P
    assert opt._losses_3d.shape[1] == V


def test_newton_momentum_uses_fd_direction():
    """Orthoplex quadratic model: momentum steps take the Newton direction."""
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
    closure = make_closure()

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
    closure = make_closure()
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
    closure = make_closure()
    for _ in range(12):
        opt.step(closure)

    assert len(opt._state.trust_region_multipliers) > 0
    assert max(opt._state.trust_region_multipliers) > 1.0


def test_newton_refinement_keeps_warmstart_duals():
    """Every step moves particles off the solved positions, refined or not, so the
    solve's duals stay as the next warm start."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
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
    opt.step(make_closure())
    assert opt._state.f is not None
    assert opt._state.g is not None


def test_newton_refinement_keeps_momentum():
    """Momentum integrates the refined target, so the refinement still moves it."""
    final = []
    for momentum, alpha in ((False, 0.0), (False, 0.3), (True, 0.0), (True, 0.3)):
        model, make_closure = _make_model_and_closure()
        opt = PolyStepOptimizer(
            model,
            particle_dim=2,
            epsilon=0.5,
            solver="softmax",
            polytope_type="orthoplex",
            newton_refinement=True,
            newton_refinement_alpha=alpha,
            use_momentum=momentum,
            num_probe=3,
            max_iterations=3,
            seed=42,
        )
        closure = make_closure()
        for _ in range(3):
            opt.step(closure)
        final.append(opt._state.X.clone())
    assert not torch.allclose(final[1], final[0])
    assert not torch.allclose(final[3], final[1])
    assert not torch.allclose(final[3], final[2])


def _bowl(batched):
    return sum(v.flatten(1).pow(2).sum(dim=1) for v in batched.values())


def _bowl_probe_counts(**kwargs):
    torch.manual_seed(0)
    opt = PolyStepOptimizer(
        nn.Linear(4, 2),
        polytope_type="orthoplex",
        newton_refinement=True,
        num_probe=3,
        adaptive_probe_warmup=0,
        seed=0,
        **kwargs,
    )
    counts = []
    for _ in range(10):
        opt.step(_bowl)
        counts.append(opt._losses_3d.shape[-1])
    return opt, counts


def test_newton_refinement_default_keeps_probes():
    opt, counts = _bowl_probe_counts()
    assert opt.adaptive_num_probe is False
    assert counts == [3] * 10


def test_adaptive_probes_newton_refinement_warns():
    with pytest.warns(UserWarning, match="refinement pauses"):
        _, counts = _bowl_probe_counts(adaptive_num_probe=True)
    assert counts[-1] == 1


@pytest.mark.parametrize("kwargs", [{"block_strategy": "per_layer"}, {"num_probe": 1}])
def test_inactive_newton_refinement_skips_pause_warning(kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PolyStepOptimizer(
            nn.Linear(4, 2),
            polytope_type="orthoplex",
            newton_refinement=True,
            adaptive_num_probe=True,
            **kwargs,
        )
    assert not any("refinement pauses" in str(w.message) for w in caught)


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
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        max_iterations=2,
        seed=42,
    )
    assert not opt.multifidelity_screen


def test_simplex_screen_ranks_by_mean_deviation():
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
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        use_quadratic_model=True,
        polytope_type="orthoplex",
        biased_rotation=True,
        amortize_steps=3,
        amortize_ema=0.7,
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    closure = make_closure()

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


def test_screen_masks_dropped_only_for_selection():
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


def test_selection_screen_runs_on_simplex():
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


@pytest.mark.parametrize("solver", ["min_cost_greedy", "top_k_mean", None])
def test_screened_cost_averages_scored_losses(monkeypatch, solver):
    """The reported cost averages the losses the solve saw, never a dropped vertex's +inf mask."""
    from polystep import _step_monolithic

    scored = []
    fill = _step_monolithic._fill_screened_losses

    def spy(*args, **kwargs):
        losses = fill(*args, **kwargs)
        scored.append(losses[torch.isfinite(losses)].mean().item())
        return losses

    monkeypatch.setattr(_step_monolithic, "_fill_screened_losses", spy)
    model, _ = _make_model_and_closure()
    x, y = torch.randn(64, 4), torch.randn(64, 2)
    ev = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(
        model,
        particle_dim=5,
        epsilon=1.0,
        seed=0,
        solver=solver,
        polytope_type="simplex",
        num_probe=1,
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
    )

    def closure(bp, _x=x, _y=y):
        return ev.evaluate(bp, _x, _y)

    cost = opt.step(closure, screen_closure=opt.screen_closure_from(closure, x, y))
    assert scored == [pytest.approx(cost, rel=1e-6)]


def test_screen_keeps_top_k_vertices(monkeypatch):
    """TopKMeanSolver averages min(k, V) vertices, so a screen keeping fewer feeds it
    entries that were never evaluated at full fidelity."""
    from polystep import _step_monolithic

    kept_per_row = []
    fill = _step_monolithic._fill_screened_losses

    def spy(screen_cost, kept, sel_idx, keep_mask, *args, **kwargs):
        kept_per_row.append(keep_mask.sum(-1))
        return fill(screen_cost, kept, sel_idx, keep_mask, *args, **kwargs)

    monkeypatch.setattr(_step_monolithic, "_fill_screened_losses", spy)
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
    closure = opt.register_evaluator(evaluator, inputs, targets)
    screen = opt.screen_closure_from(closure, inputs, targets)
    for _ in range(2):
        opt.step(closure, screen_closure=screen)

    # V = pdim + 1 = 5 on the default simplex; keep_ratio 0.1 rounds to 1, below the
    # solver's default k = 3. The floor has to lift it back to k.
    assert torch.isfinite(opt.state.X).all()
    assert opt.solver.k == 3

    V = opt._polytope_vertices.shape[0]
    assert kept_per_row, "the screen did not run"
    assert all(bool((rows >= min(opt.solver.k, V)).all()) for rows in kept_per_row)


def test_default_polytope_is_minimal_spanning_set():
    """k+1 vertices, not 2k. Changing this default is a real cost-per-step change."""
    model, _ = _make_model_and_closure()
    opt = PolyStepOptimizer(model, particle_dim=4, seed=0)
    assert opt.polytope_type == "simplex"
    assert opt._polytope_vertices.shape[0] == 5


def test_only_newton_refinement_needs_orthoplex():
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


def test_screen_places_kept_values_in_place():
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


def test_screen_calibrates_on_finite_kept_costs():
    """A nonfinite kept cost must not reach the offset given to the dropped vertices."""
    from polystep._step_monolithic import _fill_screened_losses

    P, V, K = 3, 4, 2
    screen_cost = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(P, 1)
    keep_mask = torch.zeros(P, V, dtype=torch.bool)
    keep_mask[:, :2] = True
    sel_idx = torch.nonzero(keep_mask.repeat_interleave(K, dim=1).reshape(-1), as_tuple=True)[0]
    inf, nan = float("inf"), float("nan")
    kept = torch.tensor([10.0, 12.0, 20.0, 22.0, 10.0, 12.0, inf, 22.0, nan, inf, -inf, nan])

    out = _fill_screened_losses(screen_cost, kept, sel_idx, keep_mask, P, V, K).reshape(P, V, K)
    finite_gaps = (torch.tensor([9.0, 11.0, 18.0, 20.0]).mean(), torch.tensor([9.0, 11.0, 20.0]).mean(), 0.0)
    for p, offset in enumerate(finite_gaps):
        torch.testing.assert_close(out[p, 2:], (screen_cost[p, 2:] + offset).unsqueeze(-1).expand(2, K))


def test_screen_calibrates_on_finite_screen_costs():
    """A nonfinite screen cost stays out of the offset, and a dropped vertex keeps it."""
    from polystep._step_monolithic import _fill_screened_losses

    nan = float("nan")
    screen_cost = torch.tensor([[nan, 2.0, 3.0, nan]])
    keep_mask = torch.tensor([[True, True, False, False]])

    out = _fill_screened_losses(screen_cost, torch.tensor([10.0, 12.0]), torch.tensor([0, 1]), keep_mask, 1, 4, 1)

    assert out[2].item() == 13.0
    assert out[3].isnan()


def test_site_vmap_screens_on_screen_slice():
    """A Conv2d model scores full-space candidates on the site-vmap path, which must
    screen on the same slice as the screen closure, not on the whole batch."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(32, 3))
    x, y = torch.randn(64, 1, 6, 6), torch.randint(0, 3, (64,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(
        model,
        seed=0,
        polytope_type="orthoplex",
        num_probe=1,
        chunk_size=4,
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
    )
    opt.register_evaluator(ev, x, y)
    site = opt._site_vmap_evaluator
    assert site is not None and opt._sparse_delta_evaluator is None
    rows = set()
    evaluate = site.evaluate

    def spy(*args):
        rows.add(args[4].shape[0])
        return evaluate(*args)

    site.evaluate = spy

    def closure(bp, _x=x, _y=y):
        return ev.evaluate(bp, _x, _y)

    opt.step(closure, screen_closure=opt.screen_closure_from(closure, x, y))
    assert rows == {16, 64}
    assert opt._last_screen_savings > 0


@pytest.mark.parametrize(
    "hybrid, drop, batch, chunk",
    [
        (False, (), 64, 240),
        (True, (), 64, 240),
        (True, ("_subspace_delta_evaluator",), 64, 240),
        (True, ("_subspace_delta_evaluator",), 1500, None),
    ],
)
def test_screened_candidates_skip_materialization(hybrid, drop, batch, chunk):
    """Above the size gate both screen stages score on the delta and site paths, never the closure."""
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    def run(off):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4)).double()
        x, y = torch.randn(batch, 32, dtype=torch.float64), torch.randint(0, 4, (batch,))
        ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
        if hybrid:
            sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4)
            kwargs = dict(subspace=sub, subspace_particle_dim=4)
        else:
            kwargs = dict(particle_dim=4)
        opt = PolyStepOptimizer(
            model,
            seed=0,
            polytope_type="orthoplex",
            multifidelity_screen=True,
            screen_keep_ratio=0.75,
            num_probe=3,
            chunk_size=chunk,
            **kwargs,
        )
        opt.register_evaluator(ev, x, y)
        for name in off:
            setattr(opt, name, None)
        rows = [0]

        def closure(bp, _x=x, _y=y):
            rows[0] += next(iter(bp.values())).shape[0]
            return ev.evaluate(bp, _x, _y)

        opt.step(closure, screen_closure=opt.screen_closure_from(closure, x, y))
        assert opt._last_screen_savings > 0
        return rows[0], opt.state.X

    rows, X = run(drop)
    rows_mat, X_mat = run(("_sparse_delta_evaluator", "_site_vmap_evaluator", "_subspace_delta_evaluator"))
    assert rows == 0 < rows_mat
    torch.testing.assert_close(X, X_mat, rtol=0, atol=1e-12)


def test_2d_cube_not_treated_as_orthoplex():
    """cube at pdim=2 also has 4 vertices, but its halves are not antipodal.

    Inferring the geometry from the vertex count would read L(v0)-L(v2) as a central
    difference and point the step away from descent.
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


def test_newton_refinement_warns_default_num_probe():
    """The finite-difference model needs two probe scales; num_probe defaults to 1."""
    with pytest.warns(UserWarning, match="newton_refinement needs num_probe"):
        PolyStepOptimizer(nn.Linear(8, 4), epsilon=0.1, max_iterations=5, newton_refinement=True)


@pytest.mark.parametrize("mode", ["mean", "max_cost"])
def test_nan_screen_loss_excluded_from_divisor(monkeypatch, mode):
    """A sanitized screen loss would pass the penalty off as a cost.

    It would set the divisor, and through the calibration offset drag the dropped
    vertices far below the scored ones.
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
        model,
        solver="sinkhorn",
        polytope_type="orthoplex",
        scale_cost=mode,
        use_momentum=False,
        seed=0,
        multifidelity_screen=True,
        screen_keep_ratio=0.25,
        screen_fidelity=0.25,
    )

    def closure(params):
        return 2.30 + 0.01 * torch.sigmoid(params["weight"].reshape(-1, 4).sum(-1))

    def screen(params):
        loss = closure(params)
        loss[params["weight"].reshape(-1, 4)[:, 0].argmax()] = float("nan")
        return loss

    opt.step(closure, screen_closure=screen)

    assert divisors
    assert max(d.item() for d in divisors) < 0.01


def test_screen_dropped_vertices_keep_clean_trajectory():
    """A selection solver's screen drops vertices as +inf, though every loss is finite.

    The pinned weight is the trajectory with screen-dropped vertices penalized per probe.
    """
    torch.manual_seed(3)
    model = nn.Linear(6, 1, bias=False)
    target = torch.linspace(-1, 1, 6)
    opt = PolyStepOptimizer(
        model,
        solver="min_cost_greedy",
        polytope_type="orthoplex",
        num_probe=3,
        seed=3,
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
    )

    def closure(params):
        w = params["weight"].reshape(-1, 6)
        return ((w - target) ** 2).sum(-1) + 0.5 * torch.sin(9 * w).sum(-1)

    def screen(params):
        w = params["weight"].reshape(-1, 6)
        return ((w - target) ** 2).sum(-1) + 0.2 * torch.cos(3 * w).sum(-1)

    for _ in range(60):
        opt.step(closure, screen_closure=screen)

    assert model.weight[0, 2].item() == pytest.approx(-0.42342189, abs=1e-7)


def test_multifidelity_screen_skips_forwards():
    """The screen has to run before the full-fidelity sweep, or it saves nothing."""
    torch.manual_seed(0)
    inputs, targets = torch.randn(64, 12), torch.randn(64, 3)
    seen = {}

    def build(screen):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        seen[screen] = 0

        def closure(batched, _in=inputs, _tgt=targets, _key=screen):
            seen[_key] += next(iter(batched.values())).shape[0] * _in.shape[0]
            return evaluator.evaluate(batched, _in, _tgt)

        opt = PolyStepOptimizer(
            model,
            epsilon=0.1,
            max_iterations=20,
            seed=3,
            multifidelity_screen=screen,
            polytope_type="orthoplex",
            screen_keep_ratio=0.5,
            screen_fidelity=0.25,
        )
        return closure, opt

    for screen in (False, True):
        closure, opt = build(screen)
        screen_closure = opt.screen_closure_from(closure, inputs, targets)
        assert (screen_closure is not None) == screen
        for _ in range(5):
            opt.step(closure, screen_closure=screen_closure)

    # Budget is screen_fidelity + screen_keep_ratio of the unscreened cost.
    assert seen[True] < 0.85 * seen[False], (seen[True], seen[False])


def test_multifidelity_screen_warns_without_screen_closure():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs, targets = torch.randn(16, 12), torch.randn(16, 3)
    opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=20, multifidelity_screen=True, polytope_type="orthoplex")
    with pytest.warns(UserWarning, match="no forward evaluations are saved"):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets))


def _screen_step(num_probe, keep_ratio, screen_fidelity=0.25):
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = PolyStepOptimizer(
        model,
        seed=0,
        particle_dim=4,
        multifidelity_screen=True,
        polytope_type="orthoplex",
        screen_keep_ratio=keep_ratio,
        screen_fidelity=screen_fidelity,
        num_probe=num_probe,
    )
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    inputs, targets = torch.randn(16, 8), torch.randn(16, 4)

    def closure(batched, _in=inputs, _tgt=targets):
        return evaluator.evaluate(batched, _in, _tgt)

    return opt, closure, inputs, targets


def test_multifidelity_screen_refuses_costly_config():
    """screen_fidelity/num_probe + keep_ratio >= 1: the screen would buy work, not save it."""
    opt, closure, inputs, targets = _screen_step(num_probe=1, keep_ratio=0.9, screen_fidelity=0.5)
    with pytest.warns(UserWarning, match="costs more work than it saves"):
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    assert opt._last_screen_savings == 0.0


@pytest.mark.parametrize(
    "solver, keep_ratio, screen_fidelity, batch, register",
    [
        (None, 0.55, 0.4, 16, True),
        ("top_k_mean", 0.5, 0.25, 16, True),
        (None, 0.5, 0.1, 2, True),
        (None, 0.5, 0.1, 2, False),
    ],
)
def test_screen_realized_cost_over_sweep_refused(solver, keep_ratio, screen_fidelity, batch, register):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    inputs, targets = torch.randn(batch, 4), torch.randn(batch, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(
        model,
        seed=0,
        solver=solver,
        multifidelity_screen=True,
        screen_keep_ratio=keep_ratio,
        screen_fidelity=screen_fidelity,
    )
    if register:
        opt.register_evaluator(evaluator, inputs, targets)

    def closure(batched, _in=inputs, _tgt=targets):
        return evaluator.evaluate(batched, _in, _tgt)

    with pytest.warns(UserWarning, match="costs more work than it saves"):
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    assert opt._last_screen_savings == 0.0


def test_refused_screen_reports_zero_savings():
    model = nn.Linear(2, 1)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(model, seed=0, solver="softmax", multifidelity_screen=True, screen_fidelity=0.1)
    savings = []
    for batch in (16, 2):
        inputs, targets = torch.ones(batch, 2), torch.zeros(batch, 1)
        opt.register_evaluator(evaluator, inputs, targets)

        def closure(batched, _in=inputs, _tgt=targets):
            return evaluator.evaluate(batched, _in, _tgt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
        savings.append(opt._last_screen_savings)
    assert savings[0] > 0.0
    assert savings[1] == 0.0


@pytest.mark.parametrize("screen", [False, True])
@pytest.mark.parametrize(
    "model, inputs, targets",
    [(nn.Linear(4, 2), torch.randn(0, 4), torch.randn(0, 2)), (nn.PReLU(), torch.tensor(-2.0), torch.tensor(1.0))],
)
def test_empty_or_scalar_registered_batch_steps(model, inputs, targets, screen):
    evaluator = NNCostEvaluator(model, nn.MSELoss())

    def closure(batched):
        return evaluator.evaluate(batched, inputs, targets)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = PolyStepOptimizer(model, seed=0, multifidelity_screen=screen)
        opt.register_evaluator(evaluator, inputs, targets)
        opt.step(closure, screen_closure=closure)


@pytest.mark.parametrize("num_probe", [1, 5])
def test_multifidelity_screen_saves_work_when_cheap(num_probe):
    """A cheap stage one pays for itself even at a single probe scale."""
    opt, closure, inputs, targets = _screen_step(num_probe=num_probe, keep_ratio=0.5)
    opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    assert opt._last_screen_savings > 0.0


def test_screen_ranks_directions_per_particle():
    """The kept set is per particle, not one index list shared by all of them.

    Each particle carries its own rotation, so vertex index j points somewhere
    different in every row. Averaging the contrast down the rows would make the topk
    a draw between exchangeable values and spend the full-fidelity budget on
    directions picked at random.
    """
    from polystep import _step_monolithic

    torch.manual_seed(0)
    inputs, targets = torch.randn(64, 12), torch.randn(64, 3)
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

    def closure(batched, _in=inputs, _tgt=targets):
        return evaluator.evaluate(batched, _in, _tgt)

    seen = {}
    original = _step_monolithic._fill_screened_losses

    def capture(screen_cost, kept, sel_idx, keep_mask, *args, **kwargs):
        seen["mask"] = keep_mask.clone()
        seen["screen"] = screen_cost.clone()
        return original(screen_cost, kept, sel_idx, keep_mask, *args, **kwargs)

    opt = PolyStepOptimizer(
        model,
        epsilon=0.1,
        max_iterations=20,
        seed=3,
        multifidelity_screen=True,
        polytope_type="orthoplex",
        screen_keep_ratio=0.5,
        screen_fidelity=0.25,
    )
    _step_monolithic._fill_screened_losses = capture
    try:
        opt.step(closure, screen_closure=opt.screen_closure_from(closure, inputs, targets))
    finally:
        _step_monolithic._fill_screened_losses = original

    mask, screen = seen["mask"], seen["screen"]
    assert mask.dim() == 2, "the kept set is per particle, not one shared index list"
    assert not bool((mask == mask[0]).all()), "every particle kept the same directions"

    # Each row must keep its own highest-contrast pairs.
    pdim = mask.shape[1] // 2
    contrast = (screen[:, :pdim] - screen[:, pdim:]).abs()
    kept = mask[:, :pdim]
    assert bool(
        (
            contrast.masked_fill(~kept, float("inf")).amin(dim=1)
            >= contrast.masked_fill(kept, -float("inf")).amax(dim=1)
        ).all()
    )
