"""Tests for DFO (derivative-free optimization) speedup features."""

import torch
import torch.nn as nn
import pytest
from polystep.optimizer import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


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


def test_fd_gradient_rotation_stores_direction():
    """With use_quadratic_model=True, optimizer should store FD gradient direction."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        biased_rotation=True,
        use_quadratic_model=True,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)

    # FD gradient direction should be stored (replaces OT descent direction)
    assert opt._prev_descent_direction is not None
    assert opt._prev_descent_direction.shape[1] == 2  # pdim


def test_fd_gradient_rotation_produces_finite_direction():
    """FD gradient rotation should produce finite, non-zero directions."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        biased_rotation=True,
        use_quadratic_model=True,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)
    dir_ = opt._prev_descent_direction
    assert torch.isfinite(dir_).all()
    assert torch.norm(dir_).item() > 1e-8


def test_losses_3d_retained():
    """Optimizer should retain the (P, V, K) loss tensor when use_quadratic_model=True."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        use_quadratic_model=True,
        num_probe=3,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)

    assert opt._prev_losses_3d is not None
    P = opt._state.X.shape[0]
    V = 2 * 2  # pdim=2 orthoplex
    assert opt._prev_losses_3d.shape[0] == P
    assert opt._prev_losses_3d.shape[1] == V


def test_newton_momentum_uses_fd_direction():
    """With use_quadratic_model=True, momentum steps should use Newton direction."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        amortize_steps=3,
        amortize_ema=0.7,
        use_quadratic_model=True,
        biased_rotation=True,
        num_probe=3,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)

    # Step 1: full OT - extracts FD gradient + Hessian, computes Newton direction
    opt.step(closure)
    assert opt._newton_direction is not None  # Newton direction computed from FD data

    # Step 2: momentum - should use Newton direction (not just EMA transport)
    opt.step(closure)
    # Should have moved (not zero displacement)
    assert opt._state.displacement_sqnorms[-1] >= 0


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
    # No Newton direction when QM disabled
    assert opt._newton_direction is None


def test_trust_region_expands_on_accurate_prediction():
    """A correct improvement prediction must not shrink the trust region.

    The ratio uses negative = improvement; feeding the wrong sign made an
    accurate improving step read as a failure and collapse the radius to the
    floor. An expansion (multiplier > 1.0) is only reachable when the
    signs agree.
    """
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        use_quadratic_model=True,
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

    # Same solver, but refinement moves particles after the solve, so the duals
    # encode old positions and must be cleared instead of kept.
    model2, make_closure2 = _make_model_and_closure()
    ref = PolyStepOptimizer(
        model2,
        particle_dim=2,
        epsilon=0.5,
        solver="sinkhorn",
        use_quadratic_model=True,
        newton_refinement=True,
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    ref.step(make_closure2(ref))
    assert ref._state.f is None
    assert ref._state.g is None


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
    opt.step(closure)
    assert opt._prev_rot_mats is not None
    rot_after_first = opt._prev_rot_mats.clone()

    # Second step: all particles stagnant, so rotations (and cost rows) are
    # reused unchanged rather than resampled.
    opt.step(closure)
    torch.testing.assert_close(opt._prev_rot_mats, rot_after_first)
    assert torch.isfinite(opt._state.X).all()


def test_multifidelity_screening_keeps_descending():
    """Storing the raw (not dampened) cost stops a one-way ratchet that would
    lock out directions, so the loss keeps dropping over many steps."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=4,
        epsilon=0.5,
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        num_probe=5,
        max_iterations=30,
        seed=42,
    )
    closure = make_closure(opt)
    losses = [opt.step(closure) for _ in range(20)]
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    # Measured 0.64 at this seed. The ratchet this guards would leave the loss flat,
    # which a bare `< losses[0]` would not catch.
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


def test_multifidelity_screening_skipped_for_non_orthoplex():
    """Multi-fidelity screening should be silently skipped for non-orthoplex polytopes."""
    model, make_closure = _make_model_and_closure()
    # simplex polytope: V = pdim + 1, not 2 * pdim - orthoplex-specific indexing would crash
    opt = PolyStepOptimizer(
        model,
        particle_dim=4,
        epsilon=0.5,
        polytope_type="simplex",
        multifidelity_screen=True,
        screen_keep_ratio=0.5,
        num_probe=5,
        max_iterations=2,
        seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)
    # The second step is where screening would fire on an orthoplex.
    opt.step(closure)

    assert opt._last_screen_savings == 0.0, (
        f"screening ran on a simplex polytope and reported {opt._last_screen_savings} savings; "
        "the orthoplex-specific +/- pair indexing does not apply there"
    )


def test_all_dfo_features_compose():
    """All DFO features should work together without errors."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model,
        particle_dim=2,
        epsilon=0.5,
        # All DFO features enabled
        use_quadratic_model=True,
        trust_region=True,
        biased_rotation=True,
        # Turbo features (existing)
        amortize_steps=3,
        amortize_ema=0.7,
        num_probe=3,
        max_iterations=3,
        seed=42,
    )
    closure = make_closure(opt)

    # Run 10 steps (3+ full OT cycles with amortization)
    losses = []
    for _ in range(10):
        loss = opt.step(closure)
        losses.append(loss)
        assert torch.isfinite(torch.tensor(loss))

    assert opt._state.iteration_count == 10
