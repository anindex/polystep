"""Unit tests for CMA-ES pure functions in polystep.cma.

Tests verify that the CMA-ES formulas are correctly implemented and
produce expected behavior for hyperparameter computation, evolution
path updates, covariance updates, and OT weight computation.
"""

import math

import pytest
import torch

from polystep.cma import (
    compute_cma_hyperparameters,
    compute_heaviside_sigma,
    compute_ot_bias_directions,
    update_covariance_diagonal,
    update_evolution_path_c,
    update_evolution_path_sigma,
    update_step_size_csa,
)


# ---------------------------------------------------------------------------
# Test: compute_cma_hyperparameters
# ---------------------------------------------------------------------------


class TestComputeCMAHyperparameters:
    """Tests for CMA hyperparameter computation following Hansen's formulas."""

    def test_returns_all_expected_keys(self):
        """Hyperparameter dict contains all required keys."""
        params = compute_cma_hyperparameters(n=100, mu_eff=2.0)
        expected_keys = {"c_sigma", "c_c", "c_1", "c_mu", "d_sigma", "expected_norm"}
        assert set(params.keys()) == expected_keys

    def test_hyperparams_positive(self):
        """All hyperparameters should be positive."""
        params = compute_cma_hyperparameters(n=50, mu_eff=2.0)
        for key, value in params.items():
            assert value > 0, f"{key} should be positive, got {value}"

    def test_learning_rates_bounded(self):
        """c_1 + c_mu should be <= 1 (covariance stability)."""
        for n in [10, 50, 100, 500, 1000]:
            for mu_eff in [1.0, 2.0, 5.0, 10.0]:
                params = compute_cma_hyperparameters(n=n, mu_eff=mu_eff)
                assert params["c_1"] + params["c_mu"] <= 1.0 + 1e-9, (
                    f"c_1 + c_mu > 1 for n={n}, mu_eff={mu_eff}: "
                    f"{params['c_1']} + {params['c_mu']} = {params['c_1'] + params['c_mu']}"
                )

    def test_c_sigma_formula_variant(self):
        """c_sigma uses the +3 variant for faster adaptation."""
        n, mu_eff = 100, 2.0
        params = compute_cma_hyperparameters(n=n, mu_eff=mu_eff)
        expected = (mu_eff + 2) / (n + mu_eff + 3)  # +3 variant
        assert params["c_sigma"] == pytest.approx(expected, rel=1e-6)

    def test_expected_norm_formula(self):
        """expected_norm follows E[||N(0,I)||] ~ sqrt(n) * correction."""
        n = 100
        params = compute_cma_hyperparameters(n=n, mu_eff=2.0)
        # E[||N(0,I)||] ~ sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
        expected = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        assert params["expected_norm"] == pytest.approx(expected, rel=1e-6)

    def test_small_dimension(self):
        """Small dimensions (n=2) should work without error."""
        params = compute_cma_hyperparameters(n=2, mu_eff=1.0)
        assert params["c_sigma"] > 0
        assert params["c_c"] > 0
        assert params["expected_norm"] > 0

    def test_large_dimension(self):
        """Large dimensions (n=10000) should work."""
        params = compute_cma_hyperparameters(n=10000, mu_eff=50.0)
        # c_sigma should be small for large n
        assert 0 < params["c_sigma"] < 0.1
        # expected_norm should be close to sqrt(n)
        assert params["expected_norm"] == pytest.approx(100, rel=0.1)


# ---------------------------------------------------------------------------
# Test: update_evolution_path_sigma
# ---------------------------------------------------------------------------


class TestEvolutionPathSigma:
    """Tests for step-size evolution path update (p_sigma)."""

    def test_output_shape_matches_input(self):
        """Output p_sigma has same shape as input."""
        n = 64
        p_sigma = torch.zeros(n)
        displacement = torch.randn(n)
        C_diag = torch.ones(n)

        p_sigma_new = update_evolution_path_sigma(p_sigma, displacement, C_diag, c_sigma=0.1, mu_eff=2.0)
        assert p_sigma_new.shape == p_sigma.shape

    def test_zero_displacement_accumulates_decay(self):
        """Zero displacement decays existing p_sigma by (1 - c_sigma)."""
        n = 32
        p_sigma = torch.ones(n) * 0.5
        displacement = torch.zeros(n)
        C_diag = torch.ones(n)
        c_sigma = 0.1

        p_sigma_new = update_evolution_path_sigma(p_sigma, displacement, C_diag, c_sigma=c_sigma, mu_eff=2.0)
        expected = (1 - c_sigma) * p_sigma
        assert torch.allclose(p_sigma_new, expected, atol=1e-6)

    def test_evolution_path_sigma_tiny_covariance(self):
        """Evolution path should not produce NaN/Inf with very small C_diag."""
        n = 10
        p_sigma = torch.zeros(n)
        displacement = torch.randn(n)
        C_diag = torch.full((n,), 1e-15)  # Extremely small
        result = update_evolution_path_sigma(p_sigma, displacement, C_diag, c_sigma=0.3, mu_eff=3.0)
        assert torch.isfinite(result).all(), f"Non-finite result: {result}"

    def test_covariance_scaling_applied(self):
        """C_diag^(-1/2) scaling is applied to displacement."""
        n = 16
        p_sigma = torch.zeros(n)
        displacement = torch.ones(n)
        C_diag = torch.ones(n) * 4.0  # sqrt(4) = 2, so C^{-1/2} = 0.5
        c_sigma = 0.1
        mu_eff = 2.0

        p_sigma_new = update_evolution_path_sigma(p_sigma, displacement, C_diag, c_sigma=c_sigma, mu_eff=mu_eff)

        # Manual: sqrt_factor * (1/2) * 1.0 = sqrt_factor * 0.5
        sqrt_factor = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
        expected_component = sqrt_factor * 0.5
        assert p_sigma_new[0].item() == pytest.approx(expected_component, rel=1e-5)

    def test_covariance_scaled_step_whitens_to_z(self):
        """Feeding y = sqrt(C) * z recovers the isotropic z scale via the internal
        C^{-1/2}. This is the coordinate convention the monolithic driver relies on."""
        n = 5
        C_diag = torch.tensor([4.0, 1.0, 9.0, 0.25, 1.0])
        z = torch.tensor([1.0, -2.0, 0.5, 3.0, 0.0])
        y = torch.sqrt(C_diag) * z
        p0 = torch.zeros(n)
        p_from_y = update_evolution_path_sigma(p0, y, C_diag, c_sigma=0.3, mu_eff=3.0)
        p_from_z = update_evolution_path_sigma(p0, z, torch.ones(n), c_sigma=0.3, mu_eff=3.0)
        assert torch.allclose(p_from_y, p_from_z, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: update_evolution_path_c
# ---------------------------------------------------------------------------


class TestEvolutionPathC:
    """Tests for covariance evolution path update (p_c)."""

    def test_output_shape_matches_input(self):
        """Output p_c has same shape as input."""
        n = 64
        p_c = torch.zeros(n)
        displacement = torch.randn(n)

        p_c_new = update_evolution_path_c(p_c, displacement, h_sigma=True, c_c=0.1, mu_eff=2.0)
        assert p_c_new.shape == p_c.shape

    def test_h_sigma_false_disables_accumulation(self):
        """When h_sigma=False, displacement is not added (only decay)."""
        n = 32
        p_c = torch.ones(n) * 0.5
        displacement = torch.ones(n) * 10.0  # Large, should be ignored
        c_c = 0.1

        p_c_new = update_evolution_path_c(p_c, displacement, h_sigma=False, c_c=c_c, mu_eff=2.0)
        expected = (1 - c_c) * p_c
        assert torch.allclose(p_c_new, expected, atol=1e-6)

    def test_h_sigma_true_accumulates_displacement(self):
        """When h_sigma=True, displacement contributes to p_c."""
        n = 32
        p_c = torch.zeros(n)
        displacement = torch.ones(n) * 0.1
        c_c = 0.1
        mu_eff = 2.0

        p_c_new = update_evolution_path_c(p_c, displacement, h_sigma=True, c_c=c_c, mu_eff=mu_eff)
        sqrt_factor = math.sqrt(c_c * (2 - c_c) * mu_eff)
        expected_component = sqrt_factor * 0.1
        assert p_c_new[0].item() == pytest.approx(expected_component, rel=1e-5)


# ---------------------------------------------------------------------------
# Test: compute_heaviside_sigma
# ---------------------------------------------------------------------------


class TestHeavisideSigma:
    """Tests for Heaviside stall detection function."""

    def test_healthy_p_sigma_returns_true(self):
        """Normal p_sigma norm returns h_sigma=True (healthy)."""
        n = 100
        expected_norm = math.sqrt(n)
        c_sigma = 0.1

        # p_sigma_norm slightly below threshold -> healthy
        p_sigma_norm = expected_norm * 0.5
        h_sigma = compute_heaviside_sigma(p_sigma_norm, expected_norm, n, c_sigma, generation=10)
        assert h_sigma is True

    def test_stalled_p_sigma_returns_false(self):
        """Very large p_sigma norm returns h_sigma=False (stalled)."""
        n = 100
        expected_norm = math.sqrt(n)
        c_sigma = 0.1

        # p_sigma_norm much larger than threshold -> stalled
        p_sigma_norm = expected_norm * 10.0
        h_sigma = compute_heaviside_sigma(p_sigma_norm, expected_norm, n, c_sigma, generation=100)
        assert h_sigma is False

    def test_threshold_uses_generation_plus_one(self):
        """The cumulation correction uses generation + 1 (caller passes a 0-based count)."""
        n = 100
        expected_norm = math.sqrt(n)
        c_sigma = 0.3
        generation = 3
        gen = generation + 1
        threshold = (1.4 + 2 / (n + 1)) * expected_norm * math.sqrt(1 - (1 - c_sigma) ** (2 * gen))
        assert compute_heaviside_sigma(threshold * 0.99, expected_norm, n, c_sigma, generation) is True
        assert compute_heaviside_sigma(threshold * 1.01, expected_norm, n, c_sigma, generation) is False
        # The former max(1, generation) form used a smaller threshold and would misclassify here.
        wrong = (1.4 + 2 / (n + 1)) * expected_norm * math.sqrt(1 - (1 - c_sigma) ** (2 * generation))
        assert wrong < threshold


# ---------------------------------------------------------------------------
# Test: update_step_size_csa
# ---------------------------------------------------------------------------


class TestCSAStepSize:
    """Tests for CSA step-size update."""

    def test_p_sigma_at_expected_norm_no_change(self):
        """When ||p_sigma|| == E[||N(0,I)||], sigma stays roughly same."""
        n = 100
        expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        p_sigma = torch.randn(n)
        p_sigma = p_sigma / p_sigma.norm() * expected_norm

        sigma = 1.0
        sigma_new = update_step_size_csa(sigma, p_sigma, c_sigma=0.1, d_sigma=1.1, n=n)
        # Should be close to 1.0 (exponent ~ 0)
        assert sigma_new == pytest.approx(1.0, rel=0.1)

    def test_p_sigma_larger_than_expected_increases_sigma(self):
        """When ||p_sigma|| > E[||N(0,I)||], sigma increases."""
        n = 100
        expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        p_sigma = torch.randn(n)
        p_sigma = p_sigma / p_sigma.norm() * (expected_norm * 2.0)  # 2x expected

        sigma = 1.0
        sigma_new = update_step_size_csa(sigma, p_sigma, c_sigma=0.1, d_sigma=1.1, n=n)
        assert sigma_new > sigma

    def test_p_sigma_smaller_than_expected_decreases_sigma(self):
        """When ||p_sigma|| < E[||N(0,I)||], sigma decreases."""
        n = 100
        expected_norm = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        p_sigma = torch.randn(n)
        p_sigma = p_sigma / p_sigma.norm() * (expected_norm * 0.5)  # 0.5x expected

        sigma = 1.0
        sigma_new = update_step_size_csa(sigma, p_sigma, c_sigma=0.1, d_sigma=1.1, n=n)
        assert sigma_new < sigma

    def test_clamping_lower_bound(self):
        """Sigma is clamped to minimum (1e-6)."""
        n = 10
        p_sigma = torch.zeros(n)  # Norm = 0, should decrease sigma a lot
        sigma = 1e-5  # Already small

        sigma_new = update_step_size_csa(sigma, p_sigma, c_sigma=0.5, d_sigma=1.0, n=n)
        assert sigma_new >= 1e-6

    def test_clamping_upper_bound(self):
        """Sigma is clamped to maximum (100.0)."""
        n = 10
        p_sigma = torch.randn(n) * 100  # Very large norm
        sigma = 50.0

        sigma_new = update_step_size_csa(sigma, p_sigma, c_sigma=0.5, d_sigma=1.0, n=n)
        assert sigma_new <= 100.0


# ---------------------------------------------------------------------------
# Test: update_covariance_diagonal
# ---------------------------------------------------------------------------


class TestCovarianceUpdate:
    """Tests for diagonal covariance update."""

    def test_output_shape_matches_input(self):
        """Output C_diag has same shape as input."""
        n = 64
        C_diag = torch.ones(n)
        p_c = torch.randn(n)
        rank_mu = torch.rand(n)

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=0.1, c_mu=0.1, h_sigma=True, c_c=0.1)
        assert C_new.shape == C_diag.shape

    def test_cov_bounds_enforced(self):
        """Covariance is clamped to [1e-6, 1e6]."""
        n = 16
        C_diag = torch.ones(n) * 1e-10
        p_c = torch.zeros(n)
        rank_mu = torch.zeros(n)

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=0.1, c_mu=0.1, h_sigma=True, c_c=0.1)
        assert (C_new >= 1e-6).all()
        assert (C_new <= 1e6).all()

    def test_rank_one_term_from_p_c(self):
        """Rank-one update term c_1 * p_c^2 is present."""
        n = 16
        C_diag = torch.ones(n)
        p_c = torch.ones(n) * 2.0  # p_c^2 = 4
        rank_mu = torch.zeros(n)
        c_1 = 0.2
        c_mu = 0.0
        c_c = 0.1

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=c_1, c_mu=c_mu, h_sigma=True, c_c=c_c)
        expected = 0.8 * 1.0 + c_1 * 4.0
        assert C_new[0].item() == pytest.approx(expected, rel=1e-5)

    def test_rank_mu_term_applied(self):
        """Rank-mu update term c_mu * rank_mu is present."""
        n = 16
        C_diag = torch.ones(n)
        p_c = torch.zeros(n)
        rank_mu = torch.full((n,), 4.0)
        c_1 = 0.0
        c_mu = 0.2
        c_c = 0.1

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=c_1, c_mu=c_mu, h_sigma=True, c_c=c_c)
        expected = 0.8 * 1.0 + c_mu * 4.0
        assert C_new[0].item() == pytest.approx(expected, rel=1e-5)

    def test_rank_mu_not_rescaled_by_covariance(self):
        """rank_mu is added raw, with no sqrt(C) (Mahalanobis) rescaling."""
        n = 8
        C_diag = torch.ones(n) * 4.0
        p_c = torch.zeros(n)
        rank_mu = torch.full((n,), 4.0)
        c_1 = 0.0
        c_mu = 0.2
        c_c = 0.1

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=c_1, c_mu=c_mu, h_sigma=True, c_c=c_c)
        expected = 0.8 * 4.0 + c_mu * 4.0
        assert C_new[0].item() == pytest.approx(expected, rel=1e-5)

    def test_h_sigma_false_dampens_update(self):
        """When h_sigma=False, the missing rank-one mass is added back to the
        old-C coefficient (canonical additive sep-CMA make-up)."""
        n = 16
        C_diag = torch.ones(n) * 2.0
        p_c = torch.zeros(n)
        rank_mu = torch.zeros(n)
        c_1 = 0.1
        c_mu = 0.1
        c_c = 0.1

        old_coeff = (1 - c_1 - c_mu) + c_1 * c_c * (2 - c_c)
        expected = old_coeff * 2.0

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=c_1, c_mu=c_mu, h_sigma=False, c_c=c_c)
        assert C_new[0].item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Test: compute_ot_bias_directions
# ---------------------------------------------------------------------------


class TestOTBiasDirections:
    """Tests for OT-bias direction extraction."""

    def test_output_shape(self):
        """Output has shape (top_k_actual, particle_dim)."""
        P, V, pdim = 10, 4, 8
        transport_matrix = torch.rand(P, V)
        X_vertices = torch.randn(P, V, pdim)
        X_current = torch.randn(P, pdim)
        top_k = 5

        dirs = compute_ot_bias_directions(transport_matrix, X_vertices, X_current, top_k)
        assert dirs.shape[0] <= top_k
        assert dirs.shape[1] == pdim

    def test_directions_normalized(self):
        """Each direction should have unit norm."""
        P, V, pdim = 10, 4, 8
        transport_matrix = torch.rand(P, V)
        X_vertices = torch.randn(P, V, pdim)
        X_current = torch.randn(P, pdim)
        top_k = 5

        dirs = compute_ot_bias_directions(transport_matrix, X_vertices, X_current, top_k)
        norms = torch.norm(dirs, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fewer_particles_than_top_k(self):
        """If P < top_k, returns P directions."""
        P, V, pdim = 3, 4, 8
        transport_matrix = torch.rand(P, V)
        X_vertices = torch.randn(P, V, pdim)
        X_current = torch.randn(P, pdim)
        top_k = 10

        dirs = compute_ot_bias_directions(transport_matrix, X_vertices, X_current, top_k)
        assert dirs.shape[0] == P

    def test_direction_toward_high_transport_vertex(self):
        """Direction should point toward vertex that received most transport."""
        P, V, pdim = 1, 2, 4
        # Single particle, all transport to vertex 1
        transport_matrix = torch.tensor([[0.0, 1.0]])
        X_vertices = torch.zeros(P, V, pdim)
        X_vertices[0, 1, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Vertex 1 at (1,0,0,0)
        X_current = torch.zeros(P, pdim)  # Particle at origin
        top_k = 1

        dirs = compute_ot_bias_directions(transport_matrix, X_vertices, X_current, top_k)
        # Direction should be toward (1,0,0,0) from origin -> normalized = (1,0,0,0)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(dirs, expected, atol=1e-5)

    def test_handles_zero_norm_displacement(self):
        """Edge case: particle at centroid doesn't cause NaN."""
        P, V, pdim = 2, 2, 4
        transport_matrix = torch.ones(P, V) / (P * V)
        X_vertices = torch.zeros(P, V, pdim)
        # Set particle exactly at centroid
        X_current = torch.zeros(P, pdim)
        top_k = 2

        dirs = compute_ot_bias_directions(transport_matrix, X_vertices, X_current, top_k)
        assert not torch.isnan(dirs).any()
