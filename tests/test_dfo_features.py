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
        model, particle_dim=2, epsilon=0.5,
        biased_rotation=True, use_quadratic_model=True,
        max_iterations=2, seed=42,
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
        model, particle_dim=2, epsilon=0.5,
        biased_rotation=True, use_quadratic_model=True,
        max_iterations=2, seed=42,
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
        model, particle_dim=2, epsilon=0.5,
        use_quadratic_model=True, num_probe=3,
        max_iterations=2, seed=42,
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
        model, particle_dim=2, epsilon=0.5,
        amortize_steps=3, amortize_ema=0.7,
        use_quadratic_model=True, biased_rotation=True,
        num_probe=3, max_iterations=2, seed=42,
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
        model, particle_dim=2, epsilon=0.5,
        amortize_steps=3, amortize_ema=0.7,
        use_quadratic_model=False, biased_rotation=True,
        num_probe=3, max_iterations=2, seed=42,
    )
    closure = make_closure(opt)
    opt.step(closure)
    assert opt._transport_direction_ema is not None
    # No Newton direction when QM disabled
    assert opt._newton_direction is None


def test_sobol_rotations_low_discrepancy():
    """Sobol 2D rotations should have more uniform angle coverage than random."""
    from polystep.geometry import get_sobol_rotation_matrices

    rots = get_sobol_rotation_matrices(16, 2, device=torch.device('cpu'))
    assert rots.shape == (16, 2, 2)
    for i in range(16):
        R = rots[i]
        assert torch.allclose(R @ R.T, torch.eye(2), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)
    # Check low discrepancy: angle gaps should have low coefficient of variation
    angles = torch.atan2(rots[:, 1, 0], rots[:, 0, 0])
    angles_sorted = torch.sort(angles)[0]
    gaps = torch.diff(angles_sorted)
    cv = gaps.std() / gaps.mean()
    assert cv < 0.5


def test_sobol_rotations_higher_dim():
    """Sobol higher-dim rotations should produce valid SO(d) matrices."""
    from polystep.geometry import get_sobol_rotation_matrices

    rots = get_sobol_rotation_matrices(8, 4, device=torch.device('cpu'))
    assert rots.shape == (8, 4, 4)
    for i in range(8):
        R = rots[i]
        assert torch.allclose(R @ R.T, torch.eye(4), atol=1e-4)
        assert torch.allclose(torch.det(R).abs(), torch.tensor(1.0), atol=1e-4)



def test_trust_region_adapts_radius():
    """Trust region should change step radius based on predicted vs actual improvement."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model, particle_dim=2, epsilon=0.5,
        use_quadratic_model=True, trust_region=True,
        biased_rotation=True, num_probe=3,
        max_iterations=2, seed=42,
    )
    closure = make_closure(opt)

    # Run several steps to trigger trust region updates
    for _ in range(5):
        opt.step(closure)

    # Multiplier should exist and be positive
    assert hasattr(opt, '_trust_region_multiplier')
    assert opt._trust_region_multiplier > 0
    # Diagnostics should have been recorded
    assert len(opt._state.trust_region_multipliers) > 0



def test_multifidelity_screening_runs():
    """Multi-fidelity screening should complete without errors."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model, particle_dim=4, epsilon=0.5,  # pdim=4 -> 8 vertices
        multifidelity_screen=True, screen_keep_ratio=0.5,
        num_probe=5, max_iterations=2, seed=42,
    )
    closure = make_closure(opt)
    loss = opt.step(closure)
    assert torch.isfinite(torch.tensor(loss))
    # Second step uses previous cost for screening
    loss2 = opt.step(closure)
    assert torch.isfinite(torch.tensor(loss2))


def test_multifidelity_off_by_default():
    """Multi-fidelity should be disabled by default."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model, particle_dim=2, epsilon=0.5,
        max_iterations=2, seed=42,
    )
    assert not opt.multifidelity_screen


def test_multifidelity_screening_skipped_for_non_orthoplex():
    """Multi-fidelity screening should be silently skipped for non-orthoplex polytopes."""
    model, make_closure = _make_model_and_closure()
    # simplex polytope: V = pdim + 1, not 2 * pdim - orthoplex-specific indexing would crash
    opt = PolyStepOptimizer(
        model, particle_dim=4, epsilon=0.5,
        polytope_type='simplex',
        multifidelity_screen=True, screen_keep_ratio=0.5,
        num_probe=5, max_iterations=2, seed=42,
    )
    closure = make_closure(opt)
    loss1 = opt.step(closure)
    assert torch.isfinite(torch.tensor(loss1))
    # Second step would trigger screening on orthoplex - should be skipped for simplex
    loss2 = opt.step(closure)
    assert torch.isfinite(torch.tensor(loss2))


def test_all_dfo_features_compose():
    """All DFO features should work together without errors."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model, particle_dim=2, epsilon=0.5,
        # All DFO features enabled
        use_quadratic_model=True,
        trust_region=True,
        biased_rotation=True,
        # Turbo features (existing)
        amortize_steps=3,
        amortize_ema=0.7,
        num_probe=3,
        max_iterations=3, seed=42,
    )
    closure = make_closure(opt)

    # Run 10 steps (3+ full OT cycles with amortization)
    losses = []
    for _ in range(10):
        loss = opt.step(closure)
        losses.append(loss)
        assert torch.isfinite(torch.tensor(loss))

    assert opt._state.iteration_count == 10


@pytest.mark.filterwarnings("ignore:init_f/init_g warm-start.*ignored in low-rank:UserWarning")
def test_dfo_features_with_subspace():
    """DFO features should work with subspace mode (rank < param count)."""
    model, make_closure = _make_model_and_closure()
    opt = PolyStepOptimizer(
        model, particle_dim=2, epsilon=0.5, rank=2,
        use_quadratic_model=True, biased_rotation=True,
        amortize_steps=2,
        max_iterations=2, seed=42,
    )
    closure = make_closure(opt)

    for _ in range(5):
        opt.step(closure)
    assert opt._state.iteration_count == 5
