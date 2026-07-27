"""Integration tests for CMAAdaptiveSubspace and optimizer CMA integration.

Tests verify that the CMA-ES wrapper works correctly with AdaptiveSubspace,
that the optimizer properly integrates CMA features, and that OT-bias
rotation mode functions as expected.
"""

import math

import pytest
import torch
import torch.nn as nn

from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cma_subspace import CMAAdaptiveSubspace
from polystep.optimizer import PolyStepOptimizer
from polystep.solver import SolverState


@pytest.fixture
def simple_model():
    """Small MLP for testing: Linear(20,10) -> ReLU -> Linear(10,5)."""
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))


@pytest.fixture
def base_adaptive_subspace(simple_model):
    """AdaptiveSubspace from the simple model."""
    return AdaptiveSubspace.auto_from_params(simple_model)


@pytest.fixture
def cma_adaptive_subspace(simple_model):
    """CMAAdaptiveSubspace from the simple model."""
    return CMAAdaptiveSubspace.auto_from_params(simple_model)


class TestCMAAdaptiveSubspace:
    """Tests for CMAAdaptiveSubspace wrapper class."""

    def test_from_adaptive_subspace_factory(self, base_adaptive_subspace):
        """from_adaptive_subspace wraps base correctly."""
        cma_sub = CMAAdaptiveSubspace.from_adaptive_subspace(base_adaptive_subspace)

        assert cma_sub.base is base_adaptive_subspace
        assert cma_sub.full_dim == base_adaptive_subspace.full_dim
        assert cma_sub.subspace_dim == base_adaptive_subspace.subspace_dim

    def test_auto_from_params_factory(self, simple_model):
        """auto_from_params creates CMAAdaptiveSubspace directly."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)

        total_params = sum(p.numel() for p in simple_model.parameters())
        assert cma_sub.full_dim == total_params
        assert cma_sub.subspace_dim > 0
        assert cma_sub.subspace_dim <= cma_sub.full_dim

    def test_mu_eff_default_heuristic(self, base_adaptive_subspace):
        """mu_eff defaults to subspace_dim / 4."""
        cma_sub = CMAAdaptiveSubspace.from_adaptive_subspace(base_adaptive_subspace)
        expected_mu_eff = max(1.0, base_adaptive_subspace.subspace_dim / 4.0)
        assert cma_sub.mu_eff == pytest.approx(expected_mu_eff)

    def test_delegated_properties(self, base_adaptive_subspace):
        """Properties delegate to base AdaptiveSubspace."""
        cma_sub = CMAAdaptiveSubspace.from_adaptive_subspace(base_adaptive_subspace)

        assert cma_sub.full_dim == base_adaptive_subspace.full_dim
        assert cma_sub.subspace_dim == base_adaptive_subspace.subspace_dim
        assert cma_sub.compression_ratio == base_adaptive_subspace.compression_ratio
        assert cma_sub.rotation_mode == base_adaptive_subspace.rotation_mode

    def test_init_projection_delegated(self, cma_adaptive_subspace):
        """init_projection delegates to base and returns correct shape."""
        gen = torch.Generator().manual_seed(42)
        P = cma_adaptive_subspace.init_projection(generator=gen)

        assert P.shape == (cma_adaptive_subspace.full_dim, cma_adaptive_subspace.subspace_dim)
        # Check orthogonality
        PtP = P.T @ P
        eye = torch.eye(cma_adaptive_subspace.subspace_dim)
        assert torch.allclose(PtP, eye, atol=1e-4)

    def test_init_cma_state_shapes(self, cma_adaptive_subspace):
        """init_cma_state returns tensors with correct shapes."""
        cma_state = cma_adaptive_subspace.init_cma_state()

        sub_dim = cma_adaptive_subspace.subspace_dim
        assert cma_state["p_c"].shape == (sub_dim,)
        assert cma_state["p_sigma"].shape == (sub_dim,)
        assert cma_state["C_diag"].shape == (sub_dim,)

    def test_init_cma_state_initial_values(self, cma_adaptive_subspace):
        """init_cma_state returns correct initial values."""
        cma_state = cma_adaptive_subspace.init_cma_state()

        # p_c and p_sigma start at zero
        assert torch.all(cma_state["p_c"] == 0)
        assert torch.all(cma_state["p_sigma"] == 0)
        # C_diag starts at one (isotropic)
        assert torch.all(cma_state["C_diag"] == 1)

    def test_init_cma_state_device_dtype(self, cma_adaptive_subspace):
        """init_cma_state respects device and dtype arguments."""
        cma_state = cma_adaptive_subspace.init_cma_state(device="cpu", dtype=torch.float64)

        assert cma_state["p_c"].device.type == "cpu"
        assert cma_state["p_c"].dtype == torch.float64

    def test_apply_covariance_scaling(self, cma_adaptive_subspace):
        """apply_covariance_scaling scales projection columns by sqrt(C_diag)."""
        gen = torch.Generator().manual_seed(42)
        P = cma_adaptive_subspace.init_projection(generator=gen)

        # C_diag = 4 -> sqrt = 2, columns should be scaled by 2
        C_diag = torch.ones(cma_adaptive_subspace.subspace_dim) * 4.0
        P_scaled = cma_adaptive_subspace.apply_covariance_scaling(P, C_diag)

        # P_scaled = P * sqrt(C_diag) = P * 2
        expected = P * 2.0
        assert torch.allclose(P_scaled, expected, atol=1e-6)

    def test_covariance_scaling_clamps_to_the_configured_bounds(self, cma_adaptive_subspace):
        """C_diag outside [cov_min, cov_max] is clamped before the square root.

        Without the clamp a collapsed coordinate scales the projection to zero and the
        search direction disappears; a diverged one scales it past the trust region.
        Only in-range values were exercised before.
        """
        sub = cma_adaptive_subspace
        gen = torch.Generator().manual_seed(42)
        P = sub.init_projection(generator=gen)

        C_diag = torch.full((sub.subspace_dim,), 4.0)
        C_diag[0] = sub.cov_min * 1e-3  # below the floor
        C_diag[1] = sub.cov_max * 1e3  # above the ceiling

        P_scaled = sub.apply_covariance_scaling(P, C_diag)

        torch.testing.assert_close(P_scaled[:, 0], P[:, 0] * math.sqrt(sub.cov_min), rtol=1e-5, atol=0)
        torch.testing.assert_close(P_scaled[:, 1], P[:, 1] * math.sqrt(sub.cov_max), rtol=1e-5, atol=0)
        torch.testing.assert_close(P_scaled[:, 2], P[:, 2] * 2.0, rtol=1e-5, atol=0)


class TestOptimizerCMAIntegration:
    """Tests for optimizer integration with CMA features."""

    def test_cma_flags_default_false(self, simple_model):
        """CMA flags default to False for backward compatibility."""
        opt = PolyStepOptimizer(simple_model, compile=False)
        assert opt.use_covariance_adaptation is False
        assert opt.use_csa is False

    def test_cma_features_require_cma_subspace(self, simple_model):
        """CMA features require CMAAdaptiveSubspace, warn otherwise."""
        # Regular AdaptiveSubspace with CMA flags should warn
        base_sub = AdaptiveSubspace.auto_from_params(simple_model)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opt = PolyStepOptimizer(
                simple_model,
                subspace=base_sub,
                use_csa=True,
                compile=False,
            )
            # Should warn and disable CMA
            assert len(w) == 1
            assert "CMAAdaptiveSubspace" in str(w[0].message)

        assert opt.use_csa is False

    def test_optimizer_initializes_cma_state(self, simple_model):
        """Optimizer initializes CMA state when covariance adaptation is enabled."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=cma_sub,
            use_covariance_adaptation=True,
            compile=False,
        )

        state = opt.state
        assert state.p_c is not None
        assert state.p_sigma is not None
        assert state.C_diag is not None
        assert state.sigma == 1.0
        assert state.generation == 0
        assert state.use_csa is False

    def test_optimizer_cma_state_shapes(self, simple_model):
        """CMA state tensors have correct shapes."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=cma_sub,
            use_covariance_adaptation=True,
            compile=False,
        )

        state = opt.state
        sub_dim = cma_sub.subspace_dim
        assert state.p_c.shape == (sub_dim,)
        assert state.p_sigma.shape == (sub_dim,)
        assert state.C_diag.shape == (sub_dim,)

    def test_optimizer_stores_cma_params(self, simple_model):
        """Optimizer stores CMA hyperparameters for step function."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=cma_sub,
            use_covariance_adaptation=True,
            compile=False,
        )

        # CMA params should be stored
        assert opt._cma_params is not None
        assert "c_sigma" in opt._cma_params
        assert "c_c" in opt._cma_params
        assert "c_1" in opt._cma_params
        assert "c_mu" in opt._cma_params
        assert "d_sigma" in opt._cma_params
        assert "expected_norm" in opt._cma_params
        assert "mu_eff" in opt._cma_params

    def test_cma_step_updates_state(self, simple_model):
        """A step with CMA enabled updates evolution paths and generation."""
        torch.manual_seed(42)
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=cma_sub,
            use_covariance_adaptation=True,
            epsilon=0.5,
            max_iterations=10,
            compile=False,
        )

        # Create a simple closure
        inputs = torch.randn(8, 20)
        targets = torch.randn(8, 5)
        loss_fn = nn.MSELoss()

        def closure(batched_params):
            # Simplified: just compute a scalar loss per config
            N = list(batched_params.values())[0].shape[0]
            losses = []
            for i in range(N):
                config = {k: v[i] for k, v in batched_params.items()}
                simple_model.load_state_dict(config, strict=False)
                out = simple_model(inputs)
                loss = loss_fn(out, targets)
                losses.append(loss.item())
            return torch.tensor(losses)

        state_before = opt.state
        gen_before = state_before.generation

        # Run one step
        opt.step(closure)

        state_after = opt.state
        # Generation should increment
        assert state_after.generation == gen_before + 1
        # p_sigma may change (unless displacement is exactly zero)
        # Just verify no errors occurred

    def test_covariance_adaptation_rank_mu_updates_C_diag(self, simple_model):
        """Rank-mu builds C_diag from transport-weighted vertex variance, so the
        covariance adapts off the isotropic ones and stays bounded and finite."""
        torch.manual_seed(0)
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=cma_sub,
            use_covariance_adaptation=True,
            epsilon=0.5,
            max_iterations=10,
            compile=False,
        )
        inputs = torch.randn(8, 20)
        targets = torch.randn(8, 5)
        loss_fn = nn.MSELoss()

        def closure(batched_params):
            N = list(batched_params.values())[0].shape[0]
            losses = []
            for i in range(N):
                config = {k: v[i] for k, v in batched_params.items()}
                simple_model.load_state_dict(config, strict=False)
                losses.append(loss_fn(simple_model(inputs), targets).item())
            return torch.tensor(losses)

        C0 = opt.state.C_diag.clone()
        for _ in range(4):
            opt.step(closure)
        C = opt.state.C_diag
        assert torch.isfinite(C).all()
        assert (C >= opt._cma_params["cov_min"]).all()
        assert (C <= opt._cma_params["cov_max"]).all()
        assert not torch.allclose(C, C0)

    def test_csa_does_not_collapse_sigma(self, simple_model):
        """CSA calibrated to the running ||p_sigma|| keeps sigma alive; the old
        sqrt(n) normalizer drove it to the 1e-6 floor every generation."""
        torch.manual_seed(0)
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model,
            subspace=cma_sub,
            use_covariance_adaptation=True,
            epsilon=0.5,
            max_iterations=40,
            compile=False,
        )
        inputs = torch.randn(8, 20)
        targets = torch.randn(8, 5)
        loss_fn = nn.MSELoss()

        def closure(batched_params):
            N = list(batched_params.values())[0].shape[0]
            losses = []
            for i in range(N):
                config = {k: v[i] for k, v in batched_params.items()}
                simple_model.load_state_dict(config, strict=False)
                losses.append(loss_fn(simple_model(inputs), targets).item())
            return torch.tensor(losses)

        sigma0 = opt.state.sigma
        for _ in range(10):
            opt.step(closure)
        assert opt.state.sigma > 0.5 * sigma0

    def test_cma_disabled_for_blockwise(self, simple_model):
        """CMA sampling and updates are monolithic-only; a block strategy warns
        and disables the flags instead of silently no-op adapting."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        with pytest.warns(UserWarning, match="block_strategy='monolithic'"):
            opt = PolyStepOptimizer(
                simple_model,
                subspace=cma_sub,
                block_strategy="per_layer",
                use_covariance_adaptation=True,
                use_csa=True,
                compile=False,
            )
        assert opt.use_covariance_adaptation is False
        assert opt.use_csa is False

    def test_covariance_adaptation_scales_sampling_projection(self, simple_model):
        """With use_covariance_adaptation on, the coord->param projection is the
        base projection scaled by sqrt(C_diag), so the learned covariance shapes
        the search distribution."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model, subspace=cma_sub, use_covariance_adaptation=True, epsilon=0.5, compile=False
        )
        state = opt._state
        state.C_diag = torch.linspace(0.25, 4.0, state.projection.shape[1])
        opt._update_sampling_projection()
        expected = cma_sub.apply_covariance_scaling(state.projection, state.C_diag)
        assert torch.allclose(opt._sampling_projection, expected)
        assert not torch.allclose(opt._sampling_projection, state.projection)

    def test_no_covariance_uses_plain_projection(self, simple_model):
        """Without covariance adaptation the sampling projection is unscaled."""
        cma_sub = CMAAdaptiveSubspace.auto_from_params(simple_model)
        opt = PolyStepOptimizer(
            simple_model, subspace=cma_sub, use_covariance_adaptation=False, epsilon=0.5, compile=False
        )
        state = opt._state
        state.C_diag = torch.linspace(0.25, 4.0, state.projection.shape[1])
        opt._update_sampling_projection()
        assert opt._sampling_projection is state.projection


class TestSolverStateCMAFields:
    """Tests for SolverState CMA-related fields."""

    def test_cma_fields_default_none(self):
        """CMA fields default to None/default values."""
        X = torch.randn(10, 2)
        state = SolverState(X=X)

        assert state.p_c is None
        assert state.p_sigma is None
        assert state.C_diag is None
        assert state.sigma == 1.0
        assert state.generation == 0
        assert state.use_csa is False
