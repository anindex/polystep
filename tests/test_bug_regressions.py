"""Regression tests for specific bug fixes."""

import torch
import torch.nn as nn
import pytest

from polystep import ParamLayout, PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.geometry import apply_biased_rotation, get_random_rotation_matrices


# ---------------------------------------------------------------------------
# Biased rotation must preserve det(R) = +1
# ---------------------------------------------------------------------------


class TestBiasedRotationDet:
    """Biased rotation Gram-Schmidt must produce proper rotations, not reflections."""

    @pytest.mark.parametrize("pdim", [2, 3, 4, 8])
    def test_biased_rotation_preserves_det(self, pdim):
        """The production apply_biased_rotation must return proper rotations
        (det = +1, orthonormal) after biasing the first axis."""
        P = 50
        gen = torch.Generator().manual_seed(42)
        rot_mats = get_random_rotation_matrices(P, pdim, device="cpu", dtype=torch.float32, generator=gen)

        bias_dir = torch.randn(P, pdim)
        bias_dir_norm = bias_dir / torch.norm(bias_dir, dim=-1, keepdim=True).clamp(min=1e-10)
        biased = apply_biased_rotation(rot_mats, bias_dir_norm)

        dets = torch.det(biased)
        assert (dets > 0).all(), f"Found {(dets < 0).sum()} reflections out of {P}"
        assert torch.allclose(dets, torch.ones(P), atol=1e-3)
        gram = biased.transpose(-1, -2) @ biased
        assert torch.allclose(gram, torch.eye(pdim).expand(P, -1, -1), atol=1e-4)


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

        assert torch.equal(rm_before, rm_after), "BatchNorm running stats were mutated during evaluation!"
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
        assert "fc2.weight" in all_layout_keys, "Shared param fc2.weight missing from layout"

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
        assert torch.equal(rm_original, rm_after), "BatchNorm running stats should be preserved by load_state_dict"


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
            "After blockwise OT step, _transport_direction_ema should be populated for amortized momentum steps"
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
        dirs = getattr(optimizer, "_prev_block_descent_directions", None)
        assert dirs is not None, (
            "After blockwise step with biased_rotation=True, _prev_block_descent_directions should be populated"
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
        ppbd = getattr(optimizer._state, "_prev_prev_block_duals", None)
        assert ppbd is not None, (
            "After 2 blockwise steps with dual_momentum_beta>0, _prev_prev_block_duals should be populated"
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

        n_steps = 5
        for _ in range(n_steps):
            optimizer.step(closure)

        # Transport direction EMA should never be populated
        assert optimizer._transport_direction_ema is None, (
            "amortize_steps=1 should never populate _transport_direction_ema"
        )
        assert optimizer._state.iteration_count == n_steps, (
            f"Expected {n_steps} OT iterations, got {optimizer._state.iteration_count}"
        )


# ---------------------------------------------------------------------------
# Edge-case hardening
# ---------------------------------------------------------------------------


def test_sinkhorn_rejects_empty_cost_matrix():
    """An empty (0-row or 0-col) cost matrix must raise a clear error rather
    than crash deep inside marginal alignment (1.0 / n)."""
    from polystep.solvers.sinkhorn import SinkhornSolver

    with pytest.raises(ValueError, match="empty cost matrix"):
        SinkhornSolver(epsilon=0.5).solve(torch.zeros(0, 3))


def test_cosine_epsilon_stays_in_range_past_schedule():
    """Warm-restart cosine epsilon must stay within [target, init] even far
    beyond the schedule (a maxed-out restart loop cannot push cos past pi)."""
    from polystep.epsilon import CosineEpsilon

    ce = CosineEpsilon(init=1.0, target=0.01, total_steps=50, restart_mult=1.0001)
    for i in range(0, 100_000, 2500):
        v = ce.at(i)
        assert 0.01 - 1e-9 <= v <= 1.0 + 1e-9


def test_fd_gradient_requires_orthoplex_vertices():
    """FD gradient slices [:pdim]/[pdim:], so a non-orthoplex vertex count
    (V != 2*pdim) must fail loudly, not silently return wrong gradients."""
    from polystep.quadratic_model import extract_fd_gradient

    losses = torch.randn(4, 5, 2)  # V=5, not 2*pdim
    with pytest.raises(AssertionError, match="orthoplex"):
        extract_fd_gradient(losses, torch.ones(2), probe_radius=0.1, pdim=3)


# ---------------------------------------------------------------------------
# Large-cost-offset stability: the entropic-OT plan is invariant to a constant
# shift of C, so solvers recenter the cost (min -> 0) to keep FP32 precision
# when |C| is much larger than eps. See recenter_cost().
# ---------------------------------------------------------------------------


class TestLargeCostOffsetStability:
    def test_sinkhorn_matrix_correct_under_large_offset(self):
        """exp((f+g-C)/eps) must not lose the plan to FP32 cancellation. A
        constant cost gives the product plan a (x) b regardless of the offset."""
        from polystep.solvers import SinkhornSolver

        C = torch.full((2, 2), 1_000_000.0)
        res = SinkhornSolver(epsilon=1.0, max_iterations=200, threshold=1e-9).solve(C)
        P = res.matrix
        assert torch.isfinite(P).all()
        # Uniform marginals a=b=[0.5,0.5] and constant C -> every entry 0.25.
        assert torch.allclose(P, torch.full((2, 2), 0.25), atol=1e-4), P
        assert torch.allclose(P.sum(dim=1), torch.full((2,), 0.5), atol=1e-4)
        assert torch.allclose(P.sum(dim=0), torch.full((2,), 0.5), atol=1e-4)

    def test_klsoftmax_lam0_matches_softmax_under_large_offset(self):
        """KLSoftmax(lam=0) is exactly SoftmaxSolver, even at |C|~1e6."""
        from polystep.solvers import KLSoftmaxSolver, SoftmaxSolver

        C = torch.full((1, 3), 1_000_000.0)
        kl = KLSoftmaxSolver(epsilon=1.0, lam=0.0).solve(C).matrix
        sm = SoftmaxSolver(epsilon=1.0).solve(C).matrix
        assert torch.isfinite(kl).all()
        assert torch.allclose(kl, sm, atol=1e-6), (kl, sm)
        assert torch.allclose(kl, torch.full((1, 3), 1.0 / 3.0), atol=1e-5)
        assert torch.allclose(kl.sum(dim=1), torch.ones(1), atol=1e-5)

    @pytest.mark.parametrize("solver_name", ["softmax", "tempered"])
    def test_softmax_no_nan_at_extreme_epsilon(self, solver_name):
        """A finite cost with a huge negative entry must not NaN the softmax;
        the limiting weights put all mass on the minimum-cost vertex."""
        from polystep.solvers import SoftmaxSolver, TemperedSoftmaxSolver

        C = torch.tensor([[-1e20, 0.0]], dtype=torch.float32)
        if solver_name == "softmax":
            W = SoftmaxSolver(epsilon=1e-20).solve(C).matrix
        else:
            W = TemperedSoftmaxSolver(tau=1e-20).solve(C).matrix
        assert torch.isfinite(W).all(), W
        # a=[1.0]; all mass on the min-cost (col 0).
        assert torch.allclose(W, torch.tensor([[1.0, 0.0]]), atol=1e-5), W


class TestSinkhornMarginalBalance:
    def test_warns_on_unequal_total_mass(self):
        """Balanced OT with sum(a) != sum(b) is infeasible; warn instead of
        silently reporting converged."""
        from polystep.solvers import SinkhornSolver

        C = torch.zeros(2, 1)
        with pytest.warns(UserWarning, match="unequal total mass"):
            SinkhornSolver(epsilon=0.5, max_iterations=50).solve(C, a=torch.tensor([1.0, 1.0]), b=torch.tensor([1.0]))

    def test_no_warning_on_uniform_marginals(self):
        """The hot path (a=b=None uniform) must never trip the balance warning."""
        import warnings

        from polystep.solvers import SinkhornSolver

        C = torch.rand(4, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> test failure
            SinkhornSolver(epsilon=0.5, max_iterations=50).solve(C)


def test_biased_rotation_1d_returns_identity():
    """SO(1) = {[[1]]}; a 1D biased rotation cannot represent a sign flip (that
    is a reflection), so it must return identity, not flip the aligned axis back."""
    from polystep.geometry import apply_biased_rotation

    out = apply_biased_rotation(torch.ones(1, 1, 1), torch.tensor([[-1.0]]))
    assert torch.allclose(out, torch.ones(1, 1, 1)), out
    assert torch.det(out[0]).item() > 0
