"""Tests for LowRankSubspace and LinearSubspace compression and solver integration."""

import math

import pytest
import torch
import torch.nn as nn

from polystep.subspace import FactorSpec, LowRankSubspace, LinearSubspace, ProjectionSpec
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)  # (16, 3, 3, 3) = (16, 27)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = self.conv(x).mean(dim=[2, 3])
        return self.fc(x)


class BiasOnlyModel(nn.Module):
    """Model with only 1D parameters (biases)."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        return self.ln(x)


# ---------------------------------------------------------------------------
# Test: from_layout with MLP
# ---------------------------------------------------------------------------


class TestFromLayout:
    def test_from_layout_mlp(self):
        """from_layout creates correct specs for MLP with fixed rank."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)

        # Should have one spec per layout entry
        assert len(sub.specs) == len(layout.entries)
        assert sub.rank == 4
        assert sub.subspace_dim > 0
        assert sub.subspace_dim < layout.total_params  # compressed
        assert sub.compression_ratio < 1.0

        # Check 2D entries are low-rank, 1D are full
        for spec in sub.specs:
            if len(spec.original_shape) >= 2:
                assert spec.is_lowrank
                assert len(spec.b_shape) == 2
                assert len(spec.a_shape) == 2
            else:
                assert not spec.is_lowrank
                assert spec.b_shape == ()
                assert spec.a_shape == ()

        # Check flat offsets are contiguous and non-overlapping
        prev_end = 0
        for spec in sub.specs:
            assert spec.flat_start == prev_end
            assert spec.flat_end > spec.flat_start
            prev_end = spec.flat_end
        assert prev_end == sub.subspace_dim

    def test_from_layout_rank_clamping(self):
        """effective_rank is clamped to min(rank, d_in, d_out)."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        # Use rank=100, which exceeds some layer dimensions
        sub = LowRankSubspace.from_layout(layout, rank=100)

        for spec in sub.specs:
            if spec.is_lowrank:
                d_out = spec.original_shape[0]
                d_in = math.prod(spec.original_shape[1:])
                effective_rank = spec.b_shape[1]
                assert effective_rank <= min(100, d_in, d_out)


# ---------------------------------------------------------------------------
# Test: auto_from_layout
# ---------------------------------------------------------------------------


class TestAutoFromLayout:
    def test_auto_from_layout(self):
        """auto_from_layout selects per-layer ranks without user tuning."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.auto_from_layout(layout)

        assert sub.rank == 0  # auto mode marker
        assert sub.subspace_dim > 0
        assert sub.subspace_dim < layout.total_params
        assert sub.compression_ratio < 1.0

        # Ranks should be bounded by min(auto_rank, d_in, d_out)
        for spec in sub.specs:
            if spec.is_lowrank:
                r = spec.b_shape[1]
                d_out = spec.original_shape[0]
                d_in = math.prod(spec.original_shape[1:])
                # effective_rank = min(auto_rank, d_in, d_out)
                # auto_rank >= min_rank=4, but effective_rank capped by layer dims
                assert r <= min(64, d_in, d_out)
                assert r <= 64  # max_rank default


# ---------------------------------------------------------------------------
# Test: apply_perturbation
# ---------------------------------------------------------------------------


class TestApplyPerturbation:
    def test_apply_perturbation_zeros(self):
        """Zero perturbation returns base params unchanged."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)
        base_sd = model.state_dict()

        flat_sub = torch.zeros(sub.subspace_dim)
        result = sub.apply_perturbation(base_sd, flat_sub)

        for key in base_sd:
            if key in result:
                assert torch.allclose(result[key], base_sd[key], atol=1e-6), f"Key {key} differs with zero perturbation"

    def test_apply_perturbation_nonzero(self):
        """Nonzero perturbation changes parameters."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        flat_sub = torch.randn(sub.subspace_dim) * 0.01
        result = sub.apply_perturbation(base_sd, flat_sub)

        # At least some parameters should differ
        any_different = False
        for key in base_sd:
            if key in result:
                if not torch.allclose(result[key], base_sd[key]):
                    any_different = True
                    break
        assert any_different, "Nonzero perturbation should change some params"


# ---------------------------------------------------------------------------
# Test: reconstruct_batch
# ---------------------------------------------------------------------------


class TestReconstructBatch:
    def test_reconstruct_batch_consistency(self):
        """reconstruct_batch matches apply_perturbation for each row."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)
        base_sd = model.state_dict()

        N = 3
        torch.manual_seed(123)
        flat_batch = torch.randn(N, sub.subspace_dim) * 0.01
        batch_result = sub.reconstruct_batch(base_sd, flat_batch)

        for i in range(N):
            single_result = sub.apply_perturbation(base_sd, flat_batch[i])
            for key in single_result:
                assert torch.allclose(batch_result[key][i], single_result[key], atol=1e-5), (
                    f"Row {i}, key {key} mismatch between batch and single"
                )


# ---------------------------------------------------------------------------
# Test: absorb
# ---------------------------------------------------------------------------


class TestAbsorb:
    def test_absorb(self):
        """absorb folds perturbation into base and returns zeroed subspace."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        flat_sub = torch.randn(sub.subspace_dim) * 0.01

        # Apply perturbation to get expected result
        expected = sub.apply_perturbation(base_sd, flat_sub)

        # Absorb
        new_base, zeroed = sub.absorb(base_sd, flat_sub)

        # Zeroed subspace
        assert torch.all(zeroed == 0)
        assert zeroed.shape == flat_sub.shape

        # New base matches apply_perturbation result
        for key in expected:
            assert torch.allclose(new_base[key], expected[key], atol=1e-6)


# ---------------------------------------------------------------------------
# Test: conv handling
# ---------------------------------------------------------------------------


class TestConvHandling:
    def test_conv_kernel_decomposition(self):
        """Conv (C_out, C_in, H, W) is decomposed as (C_out, C_in*H*W)."""
        model = ConvModel()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)

        # Find the conv weight spec
        conv_spec = None
        for spec in sub.specs:
            if spec.entry_key == "conv.weight":
                conv_spec = spec
                break

        assert conv_spec is not None
        assert conv_spec.is_lowrank
        assert conv_spec.original_shape == (16, 3, 3, 3)
        assert conv_spec.b_shape[0] == 16  # d_out = C_out
        assert conv_spec.a_shape[1] == 27  # d_in = C_in * H * W

    def test_conv_reconstruction_shape(self):
        """Reconstructed conv weight has original (C_out, C_in, H, W) shape."""
        model = ConvModel()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)
        base_sd = model.state_dict()

        flat_sub = torch.zeros(sub.subspace_dim)
        result = sub.apply_perturbation(base_sd, flat_sub)

        assert result["conv.weight"].shape == (16, 3, 3, 3)


# ---------------------------------------------------------------------------
# Test: 1D params
# ---------------------------------------------------------------------------


class TestOneDimParams:
    def test_1d_params_full_perturbation(self):
        """1D parameters (biases, LayerNorm) use full perturbation, not B@A."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)

        for spec in sub.specs:
            if len(spec.original_shape) == 1:
                assert not spec.is_lowrank
                # flat chunk size equals numel
                assert spec.flat_end - spec.flat_start == spec.original_shape[0]


# ---------------------------------------------------------------------------
# Test: solver integration
# ---------------------------------------------------------------------------


class TestSolverIntegration:
    def test_subspace_solver_integration(self):
        """PolyStep with subspace runs end-to-end on a synthetic NN objective."""
        from polystep.solver import PolyStep, SolverState
        from polystep.cost_nn import NNCostEvaluator
        from polystep.transform import ParamLayout

        # Tiny model keeps subspace_dim small; the test checks the solver
        # contract (iteration count, costs, displacements, X shape), not accuracy.
        model = nn.Sequential(nn.Linear(4, 3))
        layout = ParamLayout.from_module(model)
        sub = LowRankSubspace.from_layout(layout, rank=4)

        # Create inputs and targets
        torch.manual_seed(0)
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 3, (8,)).long()
        loss_fn = nn.CrossEntropyLoss()

        # Create evaluator
        evaluator = NNCostEvaluator(model, loss_fn)

        # Base params
        base_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # In subspace mode, each particle IS a full subspace vector.
        # num_particles = number of candidate solutions (e.g. 5).
        # dim = subspace_dim (each particle row is a full subspace vector).
        num_particles = 5
        dim = sub.subspace_dim

        # A dummy objective_fn (not used in subspace mode)
        def dummy_obj(x):
            return torch.zeros(x.shape[0])

        solver = PolyStep(
            objective_fn=dummy_obj,
            dim=dim,
            epsilon=0.1,
            num_probe=2,
            max_iterations=2,
            min_iterations=1,
            sinkhorn_max_iters=50,
            compile=False,
            subspace=sub,
            nn_evaluator=evaluator,
            layout=layout,
            train_inputs=inputs,
            train_targets=targets,
        )

        # Init with small random subspace vectors (near base params)
        torch.manual_seed(99)
        X_init = torch.randn(num_particles, dim) * 0.001
        state = solver.init_state(X_init, base_params=base_sd)

        assert state.base_params is not None
        assert state.subspace is not None

        # Run one step
        gen = torch.Generator().manual_seed(42)
        state = solver.step(state, generator=gen)

        # State should be updated
        assert state.iteration_count == 1
        assert len(state.costs) == 1
        assert len(state.displacement_sqnorms) == 1
        # Particles should have moved (at least slightly)
        assert state.X is not None
        assert state.X.shape == (num_particles, dim)


# ===========================================================================
# LinearSubspace tests
# ===========================================================================


class TestLinearSubspaceFromLayout:
    def test_from_layout_linear(self):
        """from_layout creates correct specs and matches LowRankSubspace subspace_dim."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        lin_sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        lr_sub = LowRankSubspace.from_layout(layout, rank=4)

        # Same number of specs
        assert len(lin_sub.specs) == len(lr_sub.specs)
        # Same subspace_dim (drop-in compatible)
        assert lin_sub.subspace_dim == lr_sub.subspace_dim
        assert lin_sub.subspace_dim > 0
        assert lin_sub.subspace_dim < layout.total_params

        # Check 2D entries are projected, 1D are full
        for spec in lin_sub.specs:
            if len(spec.original_shape) >= 2:
                assert spec.is_projected
            else:
                assert not spec.is_projected

        # Check flat offsets are contiguous
        prev_end = 0
        for spec in lin_sub.specs:
            assert spec.flat_start == prev_end
            assert spec.flat_end > spec.flat_start
            prev_end = spec.flat_end
        assert prev_end == lin_sub.subspace_dim

    def test_auto_from_layout_linear(self):
        """auto_from_layout selects per-layer coords without user tuning."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        lin_sub = LinearSubspace.auto_from_layout(layout, seed=42)
        lr_sub = LowRankSubspace.auto_from_layout(layout)

        assert lin_sub.subspace_dim == lr_sub.subspace_dim
        assert lin_sub.subspace_dim > 0
        assert lin_sub.subspace_dim < layout.total_params


class TestLinearSubspaceApplyPerturbation:
    def test_apply_perturbation_zeros_linear(self):
        """Zero perturbation returns base params unchanged."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        base_sd = model.state_dict()

        flat_sub = torch.zeros(sub.subspace_dim)
        result = sub.apply_perturbation(base_sd, flat_sub)

        for key in base_sd:
            if key in result:
                assert torch.allclose(result[key], base_sd[key], atol=1e-6), f"Key {key} differs with zero perturbation"

    def test_apply_perturbation_nonzero_linear(self):
        """Nonzero perturbation changes parameters."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        flat_sub = torch.randn(sub.subspace_dim) * 0.01
        result = sub.apply_perturbation(base_sd, flat_sub)

        any_different = False
        for key in base_sd:
            if key in result:
                if not torch.allclose(result[key], base_sd[key]):
                    any_different = True
                    break
        assert any_different, "Nonzero perturbation should change some params"


class TestLinearSubspaceReconstructBatch:
    def test_reconstruct_batch_consistency_linear(self):
        """reconstruct_batch matches apply_perturbation for each row."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        base_sd = model.state_dict()

        N = 3
        torch.manual_seed(123)
        flat_batch = torch.randn(N, sub.subspace_dim) * 0.01
        batch_result = sub.reconstruct_batch(base_sd, flat_batch)

        for i in range(N):
            single_result = sub.apply_perturbation(base_sd, flat_batch[i])
            for key in single_result:
                assert torch.allclose(batch_result[key][i], single_result[key], atol=1e-5), (
                    f"Row {i}, key {key} mismatch between batch and single"
                )


class TestLinearSubspaceAbsorb:
    def test_absorb_linear(self):
        """absorb folds perturbation into base and returns zeroed subspace."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        base_sd = model.state_dict()

        torch.manual_seed(42)
        flat_sub = torch.randn(sub.subspace_dim) * 0.01

        expected = sub.apply_perturbation(base_sd, flat_sub)
        new_base, zeroed = sub.absorb(base_sd, flat_sub)

        assert torch.all(zeroed == 0)
        assert zeroed.shape == flat_sub.shape

        for key in expected:
            assert torch.allclose(new_base[key], expected[key], atol=1e-6)


class TestLinearSubspaceLinearity:
    def test_linearity(self):
        """CRITICAL: verify weight perturbation scales proportionally with coords.

        For linear subspace: apply(base, alpha * v) - base == alpha * (apply(base, v) - base).
        This is the property that LowRankSubspace (B@A, bilinear) violates.
        """
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        base_sd = model.state_dict()

        torch.manual_seed(99)
        v = torch.randn(sub.subspace_dim) * 0.1

        result_1x = sub.apply_perturbation(base_sd, v)
        result_2x = sub.apply_perturbation(base_sd, 2.0 * v)

        for key in base_sd:
            if key in result_1x:
                delta_1 = result_1x[key] - base_sd[key]
                delta_2 = result_2x[key] - base_sd[key]
                assert torch.allclose(delta_2, 2.0 * delta_1, atol=1e-5), (
                    f"Key {key}: delta(2v) != 2*delta(v) -- linearity violated"
                )

    def test_linearity_additivity(self):
        """Verify apply(base, u + v) - base == (apply(base, u) - base) + (apply(base, v) - base)."""
        model = SimpleMLP()
        layout = ParamLayout.from_module(model)
        sub = LinearSubspace.from_layout(layout, rank=4, seed=42)
        base_sd = model.state_dict()

        torch.manual_seed(77)
        u = torch.randn(sub.subspace_dim) * 0.1
        v = torch.randn(sub.subspace_dim) * 0.1

        result_u = sub.apply_perturbation(base_sd, u)
        result_v = sub.apply_perturbation(base_sd, v)
        result_uv = sub.apply_perturbation(base_sd, u + v)

        for key in base_sd:
            if key in result_u:
                delta_u = result_u[key] - base_sd[key]
                delta_v = result_v[key] - base_sd[key]
                delta_uv = result_uv[key] - base_sd[key]
                assert torch.allclose(delta_uv, delta_u + delta_v, atol=1e-5), (
                    f"Key {key}: delta(u+v) != delta(u)+delta(v) -- additivity violated"
                )
