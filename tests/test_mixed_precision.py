"""Tests for mixed precision support in PolyStepOptimizer."""

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.cma_subspace import CMAAdaptiveSubspace


class TestMixedPrecisionProperties:
    """Test mixed precision properties and initialization."""

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
    """Test optimizer step with mixed precision."""

    def test_step_with_mixed_precision(self):
        """Optimizer step works with mixed precision enabled."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        opt = PolyStepOptimizer(
            model,
            mixed_precision=True,
            epsilon=0.1,
            step_radius=0.5,
            compile=False,
        )

        # Verify model is BF16
        assert next(model.parameters()).dtype == torch.bfloat16

        def closure(batched_params):
            from torch.func import functional_call, vmap

            model.eval()
            x = torch.randn(4, 10)  # Will be cast by matmul rules

            def forward(params):
                return functional_call(model, params, (x,)).mean()

            losses = vmap(forward)(batched_params)
            model.train()
            return losses

        # Should not raise
        cost = opt.step(closure)
        assert isinstance(cost, float)
        assert not torch.isnan(torch.tensor(cost))

    def test_step_costs_are_fp32(self):
        """Cost matrix in Sinkhorn solver is FP32 even with mixed precision."""
        model = nn.Linear(10, 5)
        opt = PolyStepOptimizer(
            model,
            mixed_precision=True,
            epsilon=0.1,
            compile=False,
        )

        captured_dtype = [None]

        # Monkey-patch to capture dtype
        original_solve = opt.solver.solve

        def patched_solve(cost_matrix, **kwargs):
            captured_dtype[0] = cost_matrix.dtype
            return original_solve(cost_matrix, **kwargs)

        opt.solver.solve = patched_solve

        def closure(batched_params):
            from torch.func import functional_call, vmap

            model.eval()
            x = torch.randn(4, 10)

            def forward(params):
                return functional_call(model, params, (x,)).mean()

            losses = vmap(forward)(batched_params)
            model.train()
            return losses

        opt.step(closure)
        assert captured_dtype[0] == torch.float32, "Cost matrix should be FP32"


class TestMixedPrecisionSubspace:
    """Test mixed precision with subspace modes."""

    def test_projection_dtype_matches_model(self):
        """Projection matrix dtype matches model dtype for memory savings."""
        # Without mixed precision
        model1 = nn.Linear(100, 50)
        subspace1 = AdaptiveSubspace(full_dim=100 * 50 + 50, subspace_dim=32)
        opt1 = PolyStepOptimizer(model1, subspace=subspace1, mixed_precision=False, compile=False)
        assert opt1.state.projection.dtype == torch.float32

        # With mixed precision
        model2 = nn.Linear(100, 50)
        subspace2 = AdaptiveSubspace(full_dim=100 * 50 + 50, subspace_dim=32)
        opt2 = PolyStepOptimizer(model2, subspace=subspace2, mixed_precision=True, compile=False)
        assert opt2.state.projection.dtype == torch.bfloat16


class TestProjectionDtype:
    """Test AdaptiveSubspace init_projection dtype parameter."""

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

    def test_cma_subspace_projection_dtype(self):
        """CMAAdaptiveSubspace passes dtype through."""
        base = AdaptiveSubspace(full_dim=100, subspace_dim=16)
        cma = CMAAdaptiveSubspace(base)

        projection_fp32 = cma.init_projection(dtype=torch.float32)
        assert projection_fp32.dtype == torch.float32

        projection_bf16 = cma.init_projection(dtype=torch.bfloat16)
        assert projection_bf16.dtype == torch.bfloat16


class TestBF16SupportDetection:
    """Test BF16 support detection logic."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_bf16_support(self):
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


class TestNaNHandling:
    """Test NaN handling with mixed precision."""

    def test_no_nans_in_normal_training(self):
        """Normal training with mixed precision produces no NaNs."""
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

        def closure(batched_params):
            from torch.func import functional_call, vmap

            model.eval()
            x = torch.randn(8, 8)
            target = torch.randn(8, 4)

            def forward(params):
                out = functional_call(model, params, (x,))
                return ((out - target) ** 2).mean()

            losses = vmap(forward)(batched_params)
            model.train()
            return losses

        for _ in range(3):
            cost = opt.step(closure)
            assert not torch.isnan(torch.tensor(cost)), "OT cost should not be NaN"
