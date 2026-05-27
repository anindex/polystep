"""Regression tests for specific bug fixes."""
import torch
import torch.nn as nn
import pytest

from polystep import ParamLayout, PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.geometry import get_random_rotation_matrices


# ---------------------------------------------------------------------------
# Biased rotation must preserve det(R) = +1
# ---------------------------------------------------------------------------

class TestBiasedRotationDet:
    """Biased rotation Gram-Schmidt must produce proper rotations, not reflections."""

    @pytest.mark.parametrize("pdim", [2, 3, 4, 8])
    def test_biased_rotation_preserves_det(self, pdim):
        """After replacing column 0 + Gram-Schmidt + det fix, det must be +1."""
        P = 50
        gen = torch.Generator().manual_seed(42)
        rot_mats = get_random_rotation_matrices(
            P, pdim, device="cpu", dtype=torch.float32, generator=gen,
        )

        # Simulate biased rotation (same as optimizer.py)
        bias_dir = torch.randn(P, pdim)
        bias_norms = torch.norm(bias_dir, dim=-1, keepdim=True).clamp(min=1e-10)
        bias_dir_norm = bias_dir / bias_norms
        rot_mats_orig = rot_mats.clone()
        rot_mats[:, :, 0] = bias_dir_norm

        for col in range(1, pdim):
            v = rot_mats[:, :, col].clone()
            for prev_col in range(col):
                proj = (v * rot_mats[:, :, prev_col]).sum(dim=-1, keepdim=True)
                v = v - proj * rot_mats[:, :, prev_col]
            raw_norm = torch.norm(v, dim=-1, keepdim=True)
            norms_v = raw_norm.clamp(min=1e-10)
            mask = (raw_norm > 1e-6).float()
            rot_mats[:, :, col] = (
                mask * (v / norms_v) + (1 - mask) * rot_mats_orig[:, :, col]
            )

        # THE FIX: det correction
        dets = torch.det(rot_mats)
        flip = (dets < 0).unsqueeze(-1)
        rot_mats[:, :, -1] = torch.where(
            flip, -rot_mats[:, :, -1], rot_mats[:, :, -1]
        )

        dets_final = torch.det(rot_mats)
        assert (dets_final > 0).all(), (
            f"Found {(dets_final < 0).sum()} reflections out of {P}"
        )
        assert torch.allclose(dets_final, torch.ones(P), atol=1e-3)


# ---------------------------------------------------------------------------
# Eval mode must be enforced during NNCostEvaluator.evaluate()
# ---------------------------------------------------------------------------

class TestEvalModeEnforced:
    """NNCostEvaluator must enforce eval mode even if user calls model.train()."""

    def test_eval_mode_enforced_during_evaluation(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)

        # User switches to train mode
        model.train()
        assert model.training

        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)
        N = 4
        flat_batch = flat.unsqueeze(0).repeat(N, 1, 1) + torch.randn(N, *flat.shape) * 0.01
        stacked = layout.batch_unflatten(flat_batch)

        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 2, (8,))

        rm_before = model.state_dict()["1.running_mean"].clone()
        evaluator.evaluate(stacked, inputs, targets)
        rm_after = model.state_dict()["1.running_mean"]

        assert torch.equal(rm_before, rm_after), (
            "BatchNorm running stats were mutated during evaluation!"
        )
        assert model.training, "Model should be restored to train mode after evaluate()"

    def test_eval_mode_restored_on_error(self):
        """If evaluation raises, model mode should still be restored."""
        model = nn.Linear(10, 2)

        def bad_loss(output, targets):
            raise ValueError("intentional")

        evaluator = NNCostEvaluator(model, bad_loss)
        # Simulate user switching to train mode AFTER evaluator creation
        model.train()
        assert model.training

        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)
        stacked = layout.batch_unflatten(flat.unsqueeze(0))

        with pytest.raises(ValueError, match="intentional"):
            evaluator.evaluate(stacked, torch.randn(1, 10), torch.zeros(1, dtype=torch.long))

        assert model.training, "Model mode should be restored even after error"


# ---------------------------------------------------------------------------
# Non-trainable buffers excluded from particle layout
# ---------------------------------------------------------------------------

class TestBuffersExcluded:
    """Only requires_grad=True params should be in the particle layout."""

    def test_batchnorm_buffers_excluded(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 5),
        )
        layout = ParamLayout.from_module(model)

        buffer_keys = {k for k, _ in model.named_buffers()}
        layout_keys = {e.key for e in layout.entries}
        for alias_tuple in layout.shared_groups:
            layout_keys.update(alias_tuple)

        overlap = buffer_keys & layout_keys
        assert len(overlap) == 0, f"Buffers should not be in layout: {overlap}"

    def test_shared_params_still_work(self):
        """Shared/tied params should still be detected even after buffer exclusion."""

        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 10)
                self.fc2.weight = self.fc1.weight

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = TiedModel()
        layout = ParamLayout.from_module(model)

        # fc2.weight should appear as shared alias of fc1.weight
        all_layout_keys = set()
        for e in layout.entries:
            all_layout_keys.add(e.key)
            all_layout_keys.update(e.shared_with)

        assert "fc1.weight" in all_layout_keys
        assert "fc2.weight" in all_layout_keys, (
            "Shared param fc2.weight missing from layout"
        )

        # Round-trip should preserve both
        flat = layout.flatten(model)
        recovered = layout.unflatten(flat)
        assert "fc1.weight" in recovered
        assert "fc2.weight" in recovered
        assert torch.equal(recovered["fc1.weight"], recovered["fc2.weight"])

    def test_load_state_dict_preserves_buffers(self):
        """After unflatten + load_state_dict(strict=False), buffers unchanged."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 5),
        )
        # Give BN non-trivial running stats
        model.train()
        model(torch.randn(8, 10))
        model.eval()

        rm_original = model.state_dict()["1.running_mean"].clone()

        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)
        flat_perturbed = flat + 0.1
        sd = layout.unflatten(flat_perturbed)

        model.load_state_dict(sd, strict=False)

        rm_after = model.state_dict()["1.running_mean"]
        assert torch.equal(rm_original, rm_after), (
            "BatchNorm running stats should be preserved by load_state_dict"
        )


# ---------------------------------------------------------------------------
# Turbo features in blockwise and subspace_blockwise modes
# ---------------------------------------------------------------------------

class TestBlockwiseTurboFeatures:
    """Turbo features (dual momentum, biased rotation, amortized EMA) must
    work in blockwise and subspace_blockwise step modes, not just monolithic."""

    def _make_simple_model(self):
        return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 2))

    def test_blockwise_transport_direction_ema_populated(self):
        """After a blockwise step with amortize_steps>1, _transport_direction_ema
        must be populated (not None) so momentum steps can fire."""
        model = self._make_simple_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            amortize_steps=2,
            amortize_ema=0.7,
            block_strategy="per_layer",
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        # First step: full OT (amortize counter=0 -> triggers full OT)
        optimizer.step(closure)
        ema = optimizer._transport_direction_ema
        assert ema is not None, (
            "After blockwise OT step, _transport_direction_ema should be populated "
            "for amortized momentum steps"
        )

    def test_blockwise_biased_rotation_descent_dirs_populated(self):
        """After a blockwise step with biased_rotation=True,
        _prev_block_descent_directions must be populated."""
        model = self._make_simple_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            biased_rotation=True,
            block_strategy="per_layer",
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        optimizer.step(closure)
        dirs = getattr(optimizer, '_prev_block_descent_directions', None)
        assert dirs is not None, (
            "After blockwise step with biased_rotation=True, "
            "_prev_block_descent_directions should be populated"
        )
        assert len(dirs) > 0, "Should have at least one block descent direction"

    def test_blockwise_dual_momentum_prev_duals_populated(self):
        """After 2 blockwise steps with dual_momentum_beta>0,
        _prev_prev_block_duals must be populated for extrapolation."""
        model = self._make_simple_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            dual_momentum_beta=0.3,
            block_strategy="per_layer",
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        # First step: no previous duals yet
        optimizer.step(closure)
        # Second step: prev_prev_block_duals should now be populated
        optimizer.step(closure)
        ppbd = getattr(optimizer._state, '_prev_prev_block_duals', None)
        assert ppbd is not None, (
            "After 2 blockwise steps with dual_momentum_beta>0, "
            "_prev_prev_block_duals should be populated"
        )
        assert len(ppbd) > 0
        # At least one block should have non-None duals
        has_duals = any(f is not None for f, g in ppbd)
        assert has_duals, "At least one block should have previous duals"


# ---------------------------------------------------------------------------
# Regression: no-amort path equivalence and fixed epsilon stability
# ---------------------------------------------------------------------------

class TestNoAmortAndFixedEpsilon:
    """Verify no-amort (amortize_steps=1) and fixed epsilon behavior."""

    def _make_model(self):
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    def test_fixed_epsilon_does_not_decay(self):
        """Float epsilon must remain constant across all iterations."""
        model = self._make_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=1.0,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
        )
        # Check epsilon at multiple iterations
        for i in range(100):
            eps = optimizer._get_epsilon(i)
            assert eps == 1.0, f"Fixed epsilon changed at iteration {i}: {eps}"

    def test_noamort_never_takes_momentum_step(self):
        """With amortize_steps=1, every step should be a full OT step."""
        model = self._make_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            amortize_steps=1,
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        for _ in range(5):
            optimizer.step(closure)

        # Transport direction EMA should never be populated
        assert optimizer._transport_direction_ema is None, (
            "amortize_steps=1 should never populate _transport_direction_ema"
        )

    def test_noamort_all_steps_are_ot_steps(self):
        """With amortize_steps=1, iteration_count should match total steps."""
        model = self._make_model()
        optimizer = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            num_probe=1,
            sinkhorn_max_iters=20,
            amortize_steps=1,
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)
        inputs = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4,))

        def closure(bp):
            return evaluator.evaluate(bp, inputs, targets)

        n_steps = 5
        for _ in range(n_steps):
            optimizer.step(closure)

        assert optimizer._state.iteration_count == n_steps, (
            f"Expected {n_steps} OT iterations, got {optimizer._state.iteration_count}"
        )
