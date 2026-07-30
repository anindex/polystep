"""Unit tests for the pure CMA-ES functions in polystep.cma.

Covers hyperparameter derivation, both evolution paths, the Heaviside gate, and the
sep-CMA diagonal covariance update, each against the Hansen formula it claims.
"""

import math

import pytest
import torch

from polystep.cma import (
    compute_cma_hyperparameters,
    compute_heaviside_sigma,
    update_covariance_diagonal,
    update_evolution_path_c,
    update_evolution_path_sigma,
)


def _chi_n(n: int) -> float:
    """E[||N(0,I)||] in n dimensions, exact, via log-gamma so it survives n = 1e4."""
    return math.exp(0.5 * math.log(2) + math.lgamma((n + 1) / 2) - math.lgamma(n / 2))


class TestComputeCMAHyperparameters:
    def test_returns_all_expected_keys(self):
        """Hyperparameter dict contains all required keys."""
        params = compute_cma_hyperparameters(n=100, mu_eff=2.0)
        assert set(params.keys()) == {"c_sigma", "c_c", "c_1", "c_mu"}

    def test_hyperparams_positive(self):
        """All hyperparameters should be positive."""
        params = compute_cma_hyperparameters(n=50, mu_eff=2.0)
        for key, value in params.items():
            assert value > 0, f"{key} should be positive, got {value}"

    @pytest.mark.parametrize("n, mu_eff", [(10, 3.0), (50, 2.0), (512, 8.0)])
    def test_rates_match_hansen_table_1(self, n, mu_eff):
        """c_c, c_1 and c_mu are the tutorial forms; only c_sigma deviates."""
        params = compute_cma_hyperparameters(n=n, mu_eff=mu_eff)
        assert params["c_c"] == pytest.approx((4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n))
        assert params["c_1"] == pytest.approx(2 / ((n + 1.3) ** 2 + mu_eff))
        assert params["c_mu"] == pytest.approx(2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

    def test_rank_mu_vanishes_at_unit_mu_eff(self):
        """One recombination point carries no second moment, so rank-mu must be exactly
        off. The optimizer runs at mu_eff=1 by default and skips building the term."""
        assert compute_cma_hyperparameters(n=64, mu_eff=1.0)["c_mu"] == 0.0

    def test_learning_rates_bounded(self):
        """c_1 + c_mu should be <= 1 (covariance stability)."""
        for n in [10, 50, 100, 500, 1000]:
            for mu_eff in [1.0, 2.0, 5.0, 10.0]:
                params = compute_cma_hyperparameters(n=n, mu_eff=mu_eff)
                assert params["c_1"] + params["c_mu"] <= 1.0 + 1e-9, (
                    f"c_1 + c_mu > 1 for n={n}, mu_eff={mu_eff}: "
                    f"{params['c_1']} + {params['c_mu']} = {params['c_1'] + params['c_mu']}"
                )

    def test_c_sigma_uses_the_faster_plus_three_variant(self):
        """Hansen gives both (mu_eff+2)/(n+mu_eff+3) and the slower +5 form. This picks
        the first, so c_sigma must come out strictly above the +5 value."""
        n, mu_eff = 100, 2.0
        c_sigma = compute_cma_hyperparameters(n=n, mu_eff=mu_eff)["c_sigma"]
        slower = (mu_eff + 2) / (n + mu_eff + 5)
        assert c_sigma > slower
        assert c_sigma == pytest.approx(slower * (n + mu_eff + 5) / (n + mu_eff + 3), rel=1e-9)

    def test_c_sigma_shrinks_with_dimension(self):
        """Cumulation must slow as the search space grows, or the path never averages."""
        c = [compute_cma_hyperparameters(n=n, mu_eff=2.0)["c_sigma"] for n in (2, 10, 100, 10000)]
        assert all(a > b for a, b in zip(c, c[1:])), c
        assert c[-1] < 0.1


class TestEvolutionPathSigma:
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


class TestEvolutionPathC:
    def test_h_sigma_false_disables_accumulation(self):
        """When h_sigma=False, displacement is not added (only decay)."""
        n = 32
        p_c = torch.ones(n) * 0.5
        displacement = torch.ones(n) * 10.0  # Large, should be ignored
        c_c = 0.1

        p_c_new = update_evolution_path_c(p_c, displacement, h_sigma=False, c_c=c_c, mu_eff=2.0)
        expected = (1 - c_c) * p_c
        assert torch.allclose(p_c_new, expected, atol=1e-6)

    def test_h_sigma_true_uses_the_same_normalization_as_the_sigma_path(self):
        """Both paths accumulate with sqrt(c*(2-c)*mu_eff), so at C=I they must agree.

        That factor is what keeps the path unit-variance under random selection; a change
        to one path and not the other decouples the step-size and covariance adaptation.
        """
        n, c, mu_eff = 32, 0.1, 2.0
        p0 = torch.zeros(n)
        displacement = torch.ones(n) * 0.1
        p_c = update_evolution_path_c(p0, displacement, h_sigma=True, c_c=c, mu_eff=mu_eff)
        p_sigma = update_evolution_path_sigma(p0, displacement, torch.ones(n), c_sigma=c, mu_eff=mu_eff)
        torch.testing.assert_close(p_c, p_sigma, rtol=1e-6, atol=1e-8)
        # sqrt(c (2-c) mu_eff) = 0.616 here, so a single step lands short of the raw
        # displacement; the path reaches unit variance only after accumulating.
        assert p_c[0].item() == pytest.approx(0.0616441, rel=1e-4)  # 0.616441 * 0.1


class TestHeavisideSigma:
    @pytest.mark.parametrize("norm_mult, expected", [(0.5, True), (10.0, False)])
    def test_healthy_p_sigma_returns_true(self, norm_mult, expected):
        """p_sigma norm below threshold is healthy (True); far above is stalled (False)."""
        n = 100
        expected_norm = math.sqrt(n)
        c_sigma = 0.1

        p_sigma_norm = expected_norm * norm_mult
        h_sigma = compute_heaviside_sigma(p_sigma_norm, expected_norm, n, c_sigma, generation=10)
        assert h_sigma is expected

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
        # generation, not generation+1, gives a smaller threshold that misclassifies here.
        wrong = (1.4 + 2 / (n + 1)) * expected_norm * math.sqrt(1 - (1 - c_sigma) ** (2 * generation))
        assert wrong < threshold


class TestCovarianceUpdate:
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

    @pytest.mark.parametrize("c_diag_val", [1.0, 4.0])
    def test_rank_mu_not_rescaled_by_covariance(self, c_diag_val):
        """rank_mu is added raw (c_mu * rank_mu), with no sqrt(C) Mahalanobis rescaling."""
        n = 16
        C_diag = torch.ones(n) * c_diag_val
        p_c = torch.zeros(n)
        rank_mu = torch.full((n,), 4.0)
        c_1 = 0.0
        c_mu = 0.2
        c_c = 0.1

        C_new = update_covariance_diagonal(C_diag, p_c, rank_mu, c_1=c_1, c_mu=c_mu, h_sigma=True, c_c=c_c)
        expected = 0.8 * c_diag_val + c_mu * 4.0
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


class TestFullGenerationAgainstHansenReference:
    """One whole sep-CMA generation at production hyperparameters.

    Every other test here zeroes a term, so the assembled
    ``old_coeff*C + c_1*rank_one + c_mu*rank_mu`` and ``trace_scale`` are never
    checked together against an independent reference.
    """

    N = 8
    MU_EFF = 3.0

    @staticmethod
    def _reference(n, mu_eff, C_diag, p_c, p_sigma, y, weights, generation, hp):
        """Hansen arXiv:1604.00772 eqs. (24), (26), (30), (37), (43), (45)."""
        import math as _m

        c_sigma, c_c = hp["c_sigma"], hp["c_c"]
        c_1, c_mu = hp["c_1"], hp["c_mu"]
        chi_n = _chi_n(n)

        # y is (mu, n) offspring steps in the covariance metric; <y>_w is the mean.
        y_w = sum(w * yk for w, yk in zip(weights, y))

        # (43) p_sigma, whitened by C^{-1/2} (diagonal, so an elementwise divide).
        disc = _m.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
        p_sigma_new = (1 - c_sigma) * p_sigma + disc * (y_w / C_diag.clamp(min=1e-6).sqrt())

        # (44) Heaviside: the path is damped over the first generations.
        thresh = (1.4 + 2.0 / (n + 1)) * chi_n * _m.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
        h_sigma = p_sigma_new.norm().item() < thresh

        # (45) p_c.
        p_c_new = (1 - c_c) * p_c
        if h_sigma:
            p_c_new = p_c_new + _m.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

        # (37)/(30) diagonal covariance, with the rank-one trace convention.
        delta_h = 0.0 if h_sigma else c_1 * c_c * (2 - c_c)
        rank_mu = sum(w * yk**2 for w, yk in zip(weights, y))
        C_new = (1 - c_1 - c_mu + delta_h) * C_diag + c_1 * float(n) * p_c_new**2 + c_mu * rank_mu
        return p_sigma_new, p_c_new, C_new, h_sigma

    def test_one_generation_matches_the_reference(self):
        n, mu_eff = self.N, self.MU_EFF
        hp = compute_cma_hyperparameters(n, mu_eff)
        gen = torch.Generator().manual_seed(11)

        # A non-isotropic covariance and a non-zero path, so every term is live.
        C_diag = torch.linspace(0.5, 2.0, n)
        p_c = torch.randn(n, generator=gen) * 0.3
        p_sigma = torch.randn(n, generator=gen) * 0.4
        weights = [0.5, 0.3, 0.2]
        y = [torch.randn(n, generator=gen) * 0.25 for _ in weights]
        y_w = sum(w * yk for w, yk in zip(weights, y))
        generation = 3

        p_sigma_ref, p_c_ref, C_ref, h_ref = self._reference(
            n, mu_eff, C_diag, p_c, p_sigma, y, weights, generation, hp
        )

        p_sigma_got = update_evolution_path_sigma(p_sigma, y_w, C_diag, hp["c_sigma"], mu_eff)
        norm = p_sigma_got.norm().item()
        h_got = compute_heaviside_sigma(norm, _chi_n(n), n, hp["c_sigma"], generation)
        p_c_got = update_evolution_path_c(p_c, y_w, h_got, hp["c_c"], mu_eff)
        rank_mu = sum(w * yk**2 for w, yk in zip(weights, y))
        C_got = update_covariance_diagonal(
            C_diag=C_diag,
            p_c=p_c_got,
            rank_mu=rank_mu,
            c_1=hp["c_1"],
            c_mu=hp["c_mu"],
            h_sigma=h_got,
            c_c=hp["c_c"],
            trace_scale=float(n),
        )

        assert h_got == h_ref
        torch.testing.assert_close(p_sigma_got, p_sigma_ref, rtol=0, atol=1e-6)
        torch.testing.assert_close(p_c_got, p_c_ref, rtol=0, atol=1e-6)
        torch.testing.assert_close(C_got, C_ref, rtol=0, atol=1e-6)

    def test_covariance_moves_when_every_term_is_live(self):
        """Guards the reference test: a no-op update would match a no-op reference."""
        n, mu_eff = self.N, self.MU_EFF
        hp = compute_cma_hyperparameters(n, mu_eff)
        C_diag = torch.linspace(0.5, 2.0, n)
        gen = torch.Generator().manual_seed(12)
        C_new = update_covariance_diagonal(
            C_diag=C_diag,
            p_c=torch.randn(n, generator=gen) * 0.3,
            rank_mu=torch.rand(n, generator=gen) * 0.1,
            c_1=hp["c_1"],
            c_mu=hp["c_mu"],
            h_sigma=True,
            c_c=hp["c_c"],
            trace_scale=float(n),
        )
        assert hp["c_1"] > 0
        assert not torch.allclose(C_new, C_diag)
