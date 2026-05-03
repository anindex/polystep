"""Tests for momentum and adaptive radius dynamics functions."""

import math

import pytest
import torch

from polystep.dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
    update_adaptive_radius,
)


# ---------------------------------------------------------------------------
# TestMomentumCoefficient
# ---------------------------------------------------------------------------


class TestMomentumCoefficient:
    """Tests for compute_momentum_coefficient linear warmup."""

    def test_warmup_start(self):
        """At iteration 0, beta equals momentum_init."""
        beta = compute_momentum_coefficient(0, max_iterations=100)
        assert beta == pytest.approx(0.5)

    def test_warmup_end(self):
        """At final iteration, beta equals momentum_final."""
        beta = compute_momentum_coefficient(99, max_iterations=100)
        assert beta == pytest.approx(0.95)

    def test_warmup_midpoint(self):
        """At midpoint, beta is halfway between init and final."""
        mid = 49  # (100-1)//2
        beta = compute_momentum_coefficient(mid, max_iterations=100)
        expected = 0.5 + (mid / 99) * (0.95 - 0.5)
        assert beta == pytest.approx(expected)

    def test_beyond_max(self):
        """Iteration beyond max_iterations caps at momentum_final."""
        beta = compute_momentum_coefficient(200, max_iterations=100)
        assert beta == pytest.approx(0.95)

    def test_single_iteration(self):
        """Edge case: max_iterations=1 clamps progress correctly."""
        # max(1, 1-1) = max(1, 0) = 1, progress = min(1.0, 0/1) = 0.0
        beta = compute_momentum_coefficient(0, max_iterations=1)
        assert beta == pytest.approx(0.5)

    def test_custom_range(self):
        """Custom init/final values interpolate correctly."""
        beta = compute_momentum_coefficient(
            9, max_iterations=10, momentum_init=0.0, momentum_final=1.0
        )
        assert beta == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestApplyMomentum
# ---------------------------------------------------------------------------


class TestApplyMomentum:
    """Tests for apply_momentum velocity update."""

    def test_zero_velocity(self):
        """Starting from zero velocity, X_new = X_old + velocity_lr * displacement."""
        X_old = torch.tensor([[1.0, 2.0]])
        X_bary = torch.tensor([[3.0, 4.0]])
        velocity = torch.zeros_like(X_old)

        X_new, v_new = apply_momentum(X_old, X_bary, velocity, beta=0.9)

        displacement = X_bary - X_old  # [[2, 2]]
        expected_v = displacement  # beta * 0 + displacement
        expected_X = X_old + expected_v
        assert torch.allclose(X_new, expected_X)
        assert torch.allclose(v_new, expected_v)

    def test_accumulation(self):
        """Two successive calls accumulate velocity."""
        X0 = torch.tensor([[0.0, 0.0]])
        X1_bary = torch.tensor([[1.0, 0.0]])
        v0 = torch.zeros_like(X0)

        X1, v1 = apply_momentum(X0, X1_bary, v0, beta=0.5)
        # v1 = 0.5*0 + (1-0) = [1, 0]
        assert torch.allclose(v1, torch.tensor([[1.0, 0.0]]))

        X2_bary = torch.tensor([[2.0, 0.0]])
        X2, v2 = apply_momentum(X1, X2_bary, v1, beta=0.5)
        # displacement = X2_bary - X1 = [2, 0] - [1, 0] = [1, 0]
        # v2 = 0.5 * [1, 0] + [1, 0] = [1.5, 0]
        assert torch.allclose(v2, torch.tensor([[1.5, 0.0]]))

    def test_velocity_lr_scaling(self):
        """velocity_lr=0.5 halves the effective movement."""
        X_old = torch.tensor([[0.0, 0.0]])
        X_bary = torch.tensor([[2.0, 0.0]])
        velocity = torch.zeros_like(X_old)

        X_full, _ = apply_momentum(X_old, X_bary, velocity, beta=0.0, velocity_lr=1.0)
        X_half, _ = apply_momentum(X_old, X_bary, velocity, beta=0.0, velocity_lr=0.5)

        # X_full = [0,0] + 1.0 * [2,0] = [2,0]
        # X_half = [0,0] + 0.5 * [2,0] = [1,0]
        assert torch.allclose(X_full, torch.tensor([[2.0, 0.0]]))
        assert torch.allclose(X_half, torch.tensor([[1.0, 0.0]]))

    def test_shape_preservation(self):
        """Output shapes match input shapes."""
        N, D = 10, 5
        X_old = torch.randn(N, D)
        X_bary = torch.randn(N, D)
        velocity = torch.randn(N, D)

        X_new, v_new = apply_momentum(X_old, X_bary, velocity, beta=0.9)
        assert X_new.shape == (N, D)
        assert v_new.shape == (N, D)


# ---------------------------------------------------------------------------
# TestAdaptiveRadius
# ---------------------------------------------------------------------------


class TestAdaptiveRadius:
    """Tests for update_adaptive_radius stagnation and radius adaptation."""

    def test_stagnation_detection(self):
        """Small relative change increments stagnation_count."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=1.0,
            prev_loss=1.0 + 1e-6,  # tiny change
            stagnation_count=0,
            radius_multiplier=1.0,
        )
        assert sc == 1  # incremented

    def test_stagnation_reset_on_change(self):
        """Large relative change resets stagnation_count to 0."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=0.5,
            prev_loss=1.0,  # 50% change
            stagnation_count=5,
            radius_multiplier=1.0,
        )
        assert sc == 0

    def test_boost_after_patience(self):
        """After patience iterations, radius_multiplier increases."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=1.0,
            prev_loss=1.0 + 1e-6,
            stagnation_count=9,  # will become 10 == patience
            radius_multiplier=1.0,
            stagnation_patience=10,
            radius_increase=1.5,
        )
        assert rm == pytest.approx(1.5)

    def test_boost_resets_count(self):
        """After boost, stagnation_count resets to 0."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=1.0,
            prev_loss=1.0 + 1e-6,
            stagnation_count=9,
            radius_multiplier=1.0,
            stagnation_patience=10,
        )
        assert sc == 0

    def test_decay_on_improvement(self):
        """When current_loss < prev_loss and not stagnating, radius decays."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=0.5,
            prev_loss=1.0,
            stagnation_count=0,
            radius_multiplier=1.0,
            radius_decrease=0.9,
        )
        assert rm == pytest.approx(0.9)

    def test_radius_upper_bound(self):
        """radius_multiplier cannot exceed radius_max."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=1.0,
            prev_loss=1.0 + 1e-6,
            stagnation_count=9,
            radius_multiplier=2.5,
            stagnation_patience=10,
            radius_increase=1.5,
            radius_max=3.0,
        )
        # 2.5 * 1.5 = 3.75, clamped to 3.0
        assert rm == pytest.approx(3.0)

    def test_radius_lower_bound(self):
        """radius_multiplier cannot go below radius_min."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=0.5,
            prev_loss=1.0,
            stagnation_count=0,
            radius_multiplier=0.55,
            radius_decrease=0.9,
            radius_min=0.5,
        )
        # 0.55 * 0.9 = 0.495, clamped to 0.5
        assert rm == pytest.approx(0.5)

    def test_inf_prev_loss(self):
        """First iteration with prev_loss=inf detects improvement."""
        rm, sc, pl = update_adaptive_radius(
            current_loss=1.0,
            prev_loss=float("inf"),
            stagnation_count=0,
            radius_multiplier=1.0,
            radius_decrease=0.9,
        )
        # With inf prev_loss (first step), skip adaptation entirely -
        # no history to compare against, so radius stays unchanged
        assert sc == 0
        assert rm == pytest.approx(1.0)

    def test_returns_current_loss(self):
        """Third return value is current_loss for storing as prev_loss."""
        _, _, pl = update_adaptive_radius(
            current_loss=42.0,
            prev_loss=100.0,
            stagnation_count=0,
            radius_multiplier=1.0,
        )
        assert pl == 42.0
