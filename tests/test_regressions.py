"""Correctness invariants across geometry, solvers, epsilon schedules, and the
optimizer state machine."""

import torch
import torch.nn as nn
import pytest

from torch.func import functional_call

from polystep import ParamLayout, PolyStepOptimizer
from polystep.cma_subspace import CMAAdaptiveSubspace
from polystep.cost_nn import NNCostEvaluator
from polystep.geometry import apply_biased_rotation, get_random_rotation_matrices


@pytest.mark.parametrize("pdim", [2, 3, 8])
def test_vanishing_bias_direction_keeps_the_frame_orthonormal(pdim):
    """A vanishing descent direction must leave the frame orthonormal.

    Normalizing by ``norm.clamp(min=1e-10)`` turns a 1e-12 direction into a column
    of norm 1e-2, and Gram-Schmidt against a scaled axis skews the frame. A
    particle with no usable heading keeps its unbiased rotation instead.
    """
    P = 32
    gen = torch.Generator().manual_seed(7)
    rot_mats = get_random_rotation_matrices(P, pdim, device="cpu", dtype=torch.float32, generator=gen)

    # Half the particles have collapsed, half still carry a usable heading.
    bias_dir = torch.randn(P, pdim, generator=gen)
    bias_dir[::2] *= 1e-12

    biased = apply_biased_rotation(rot_mats, bias_dir)

    gram = biased.transpose(-1, -2) @ biased
    assert torch.allclose(gram, torch.eye(pdim).expand(P, -1, -1), atol=1e-4)
    assert torch.allclose(torch.det(biased), torch.ones(P), atol=1e-3)
    # Collapsed particles fall back to exactly the rotation they came in with.
    assert torch.equal(biased[::2], rot_mats[::2])


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


class TestBlockwiseTurboFeatures:
    """Dual momentum, biased rotation and the amortized EMA must
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
        assert ema is not None, "the amortized momentum step has nothing to coast along"
        # A populated-but-zero EMA makes every cheap step a no-op, which "is not None"
        # cannot see. The next step must then move the parameters without a full OT.
        assert ema.shape == optimizer.state.X.shape
        assert ema.abs().max() > 0, "the EMA is all zeros, so a momentum step would not move anything"

        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step(closure)
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters())), (
            "the amortized step did not move the model"
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
        assert dirs is not None, "the next step has no heading to bias its rotation toward"
        assert len(dirs) == len(optimizer._blocks), "one descent direction per block"
        # A list of zero vectors carries no heading, and apply_biased_rotation routes
        # those particles to the unbiased fallback.
        assert any(d is not None and torch.linalg.vector_norm(d) > 0 for d in dirs), (
            "every block descent direction is zero, so the bias is inert"
        )

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
    # Raises ValueError (not AssertionError) so the check survives `python -O`.
    with pytest.raises(ValueError, match="orthoplex"):
        extract_fd_gradient(losses, torch.ones(2), probe_radius=0.1, pdim=3)


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
        kl_result = KLSoftmaxSolver(epsilon=1.0, lam=0.0).solve(C)
        kl = kl_result.matrix
        sm = SoftmaxSolver(epsilon=1.0).solve(C).matrix
        assert torch.isfinite(kl).all()
        assert torch.allclose(kl, sm, atol=1e-6), (kl, sm)
        assert torch.allclose(kl, torch.full((1, 3), 1.0 / 3.0), atol=1e-5)
        assert torch.allclose(kl.sum(dim=1), torch.ones(1), atol=1e-5)
        # Recentering shifts C before the solve, so the reported cost has to come
        # back in the raw frame: <C_raw, P> = 1e6 at unit mass.
        assert kl_result.cost == pytest.approx(1_000_000.0, rel=1e-6)

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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = SinkhornSolver(epsilon=0.5, max_iterations=50).solve(C)

        assert caught == [], [str(w.message) for w in caught]
        torch.testing.assert_close(result.matrix.sum(dim=1), torch.full((4,), 0.25))


def test_biased_rotation_1d_returns_identity():
    """SO(1) = {[[1]]}; a 1D biased rotation cannot represent a sign flip (that
    is a reflection), so it must return identity, not flip the aligned axis back."""
    from polystep.geometry import apply_biased_rotation

    out = apply_biased_rotation(torch.ones(1, 1, 1), torch.tensor([[-1.0]]))
    assert torch.allclose(out, torch.ones(1, 1, 1)), out
    assert torch.det(out[0]).item() > 0


def _mlp_and_closure(seed, in_dim=20, out_dim=4, samples=64):
    """Build an MLP plus a batched-params closure over fixed random data."""
    gen = torch.Generator().manual_seed(1000 + seed)
    inputs = torch.randn(samples, in_dim, generator=gen)
    targets = torch.randn(samples, out_dim, generator=gen)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, out_dim))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    def true_loss():
        with torch.no_grad():
            return nn.functional.mse_loss(model(inputs), targets).item()

    return model, closure, true_loss


class TestRotationPreservesPoint:
    """The represented point is ``base + P @ coords``.

    Replacing ``P`` while coords are non-zero moves the weights with no evaluation
    behind it, so every rotation must fold coords into base first.
    """

    def test_coords_are_zero_after_rotation(self):
        """Rotation re-anchors the origin, so coords must be exactly zero."""
        from polystep.adaptive_subspace import AdaptiveSubspace

        model, closure, _ = _mlp_and_closure(0)
        subspace = AdaptiveSubspace.auto_from_params(model)
        opt = PolyStepOptimizer(model, subspace=subspace, epsilon=0.1, max_iterations=10)

        prev_projection = None
        for _ in range(5):
            opt.step(closure)
            coords = opt.state.X.reshape(-1)[: subspace.subspace_dim]
            assert torch.count_nonzero(coords) == 0
            if prev_projection is not None:
                teleport = torch.linalg.vector_norm((opt.state.projection - prev_projection) @ coords)
                assert teleport.item() == 0.0
            prev_projection = opt.state.projection.clone()

    def test_loss_reduction_floor(self):
        """Multi-seed floor. Keeping coords across rotation scored 6.1% here."""
        from polystep.adaptive_subspace import AdaptiveSubspace

        reductions = []
        for seed in range(6):
            model, closure, true_loss = _mlp_and_closure(seed)
            opt = PolyStepOptimizer(
                model,
                subspace=AdaptiveSubspace.auto_from_params(model),
                epsilon=0.1,
                max_iterations=60,
            )
            before = true_loss()
            for _ in range(60):
                opt.step(closure)
            reductions.append((before - true_loss()) / before * 100.0)

        reductions.sort()
        median = 0.5 * (reductions[2] + reductions[3])
        assert median > 8.0, reductions


class TestCMAWrapperIsWiredIn:
    """An ``isinstance(AdaptiveSubspace)`` gate skips the wrapper entirely."""

    def test_direct_construction_fills_hyperparameters(self):
        """The bare constructor must not leave the ``0.0`` sentinels in place."""
        from polystep.adaptive_subspace import AdaptiveSubspace
        from polystep.cma_subspace import CMAAdaptiveSubspace

        model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 4))
        base = AdaptiveSubspace.auto_from_params(model)
        direct = CMAAdaptiveSubspace(base)
        factory = CMAAdaptiveSubspace.from_adaptive_subspace(base)

        assert direct.c_mu > 0.0
        for name in ("c_c", "c_sigma", "c_1", "c_mu", "mu_eff"):
            assert getattr(direct, name) == pytest.approx(getattr(factory, name))

    @pytest.mark.parametrize("mu_eff", [1.0, 7.0])
    def test_explicit_mu_eff_reaches_the_evolution_path(self, mu_eff):
        """The rates honoured it while the path updates hardcoded 1.0.

        From ``p_sigma = 0`` and ``C_diag = 1``, one step leaves
        ``||p_sigma|| = sqrt(c_sigma (2 - c_sigma) mu_eff)`` because the step feeds the
        path a unit-norm direction, so the factor is readable straight off the norm.
        """
        import math

        from polystep.adaptive_subspace import AdaptiveSubspace
        from polystep.cma_subspace import CMAAdaptiveSubspace

        model, closure, _ = _mlp_and_closure(0)
        base = AdaptiveSubspace.auto_from_params(model)
        sub = CMAAdaptiveSubspace(base, mu_eff=mu_eff)
        assert sub.mu_eff == mu_eff

        opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.1, max_iterations=5, use_covariance_adaptation=True)
        opt.step(closure)

        c_sigma = opt._cma_params["c_sigma"]
        expected = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
        assert opt.state.p_sigma.norm().item() == pytest.approx(expected, rel=1e-5)

    def test_absorb_and_rotation_run_for_the_wrapper(self):
        """Periodic absorb must fire and the projection must change."""
        from polystep.adaptive_subspace import AdaptiveSubspace
        from polystep.cma_subspace import CMAAdaptiveSubspace

        model, closure, _ = _mlp_and_closure(0)
        base = AdaptiveSubspace.auto_from_params(model)
        base.absorb_mode = "periodic"
        base.absorb_interval = 1
        opt = PolyStepOptimizer(
            model,
            subspace=CMAAdaptiveSubspace.from_adaptive_subspace(base),
            epsilon=0.1,
            max_iterations=20,
        )
        first_projection = opt.state.projection.clone()
        for _ in range(5):
            opt.step(closure)

        assert opt.state.absorb_count == 5
        assert not torch.equal(first_projection, opt.state.projection)

    def test_wrapper_delegates_absorb_settings(self):
        """The constructor warning reads these through ``getattr`` on the subspace."""
        from polystep.adaptive_subspace import AdaptiveSubspace
        from polystep.cma_subspace import CMAAdaptiveSubspace

        model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 4))
        base = AdaptiveSubspace.auto_from_params(model)
        wrapper = CMAAdaptiveSubspace.from_adaptive_subspace(base)
        assert wrapper.absorb_mode == base.absorb_mode
        assert wrapper.absorb_patience == base.absorb_patience
        assert wrapper.absorb_interval == base.absorb_interval
        assert wrapper.displacement_history_size == base.displacement_history_size


def test_load_state_dict_drops_stale_reuse_cache():
    """Reuse rows describe the pre-load point, so they must not survive a load."""
    model, closure, _ = _mlp_and_closure(0)
    opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10, adaptive_probes=True)
    saved = opt.state_dict()
    for _ in range(3):
        opt.step(closure)

    assert opt._prev_cost_matrix is not None

    opt.load_state_dict(saved)
    assert opt._prev_cost_matrix is None
    assert opt._prev_rot_mats is None
    assert opt._losses_3d is None
    assert opt._prev_k_eff is None
    assert opt._prev_step_r is None
    assert opt._prev_probe_r is None
    assert opt._newton_direction is None
    assert opt._transport_direction_ema is None


def test_fully_frozen_model_raises_clearly():
    """A model with no trainable parameters must say so, not fail deep in the layout."""
    model = nn.Sequential(nn.Linear(4, 4))
    for param in model.parameters():
        param.requires_grad_(False)

    with pytest.raises(ValueError, match="requires_grad"):
        PolyStepOptimizer(model, epsilon=0.1, max_iterations=5)


def test_rank_transition_carries_hybrid_config():
    """Listing config fields by hand dropped the absorb_* fields."""
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.optimizer import RankSchedule

    model, closure, _ = _mlp_and_closure(0)
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout,
        rank=4,
        absorb_mode="periodic",
        absorb_interval=3,
        absorb_patience=7,
        svd_ratio_final=0.25,
        sparse_threshold_bytes=123_456_789,
        absorb_aligned_active=True,
    )
    opt = PolyStepOptimizer(
        model,
        subspace=subspace,
        epsilon=0.1,
        max_iterations=10,
        rank_schedule=RankSchedule(stages=[(0, 4), (1, 8)]),
    )
    for _ in range(3):
        opt.step(closure)

    assert opt.subspace is not subspace, "rank transition did not run"
    assert opt.subspace.absorb_mode == "periodic"
    assert opt.subspace.absorb_interval == 3
    assert opt.subspace.absorb_patience == 7
    assert opt.subspace.svd_ratio_final == 0.25
    assert opt.subspace.sparse_threshold_bytes == 123_456_789
    assert opt.subspace.absorb_aligned_active is True


def test_rank_transition_resets_momentum_and_cma_state():
    """Stale velocity either raises a shape error or broadcasts one row over all particles."""
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.optimizer import RankSchedule

    model, closure, _ = _mlp_and_closure(0)
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    opt = PolyStepOptimizer(
        model,
        subspace=subspace,
        epsilon=0.1,
        max_iterations=10,
        use_momentum=True,
        rank_schedule=RankSchedule(stages=[(0, 2), (2, 16)]),
    )
    for _ in range(4):
        opt.step(closure)

    assert opt.state.velocity.shape == opt.state.X.shape


def test_newton_refinement_warns_at_default_num_probe():
    """The finite-difference model needs two probe scales; num_probe defaults to 1."""
    with pytest.warns(UserWarning, match="newton_refinement needs num_probe"):
        PolyStepOptimizer(nn.Linear(8, 4), epsilon=0.1, max_iterations=5, newton_refinement=True)


def test_covariance_adaptation_preserves_trace():
    """Without trace-scaling and the mean-1 renormalization, an uninformative step
    shrinks every entry a little each generation and C_diag walks down to its floor.
    """
    from polystep.adaptive_subspace import AdaptiveSubspace
    from polystep.cma_subspace import CMAAdaptiveSubspace

    model, closure, _ = _mlp_and_closure(2)
    base = AdaptiveSubspace.auto_from_params(model)
    opt = PolyStepOptimizer(
        model,
        subspace=CMAAdaptiveSubspace.from_adaptive_subspace(base),
        epsilon=0.1,
        max_iterations=80,
        use_covariance_adaptation=True,
    )
    for _ in range(40):
        opt.step(closure)

    assert opt.state.C_diag.mean().item() == pytest.approx(1.0, abs=1e-4)
    assert opt.state.C_diag.min().item() > 0.1
    # Clamping runs after the mean-1 rescale, so the declared bounds actually hold.
    assert opt.state.C_diag.min().item() >= opt._cma_params["cov_min"]
    assert opt.state.C_diag.max().item() <= opt._cma_params["cov_max"]


def test_mu_eff_matches_the_unit_innovation_convention():
    """The learning rates must be derived at the same mu_eff the paths run at.

    The step feeds the evolution paths a unit-norm direction, which is mu_eff = 1.
    Deriving c_sigma from the vertex count instead gave ~2/3, a 1.5-step path memory
    that no update ever used.
    """
    from polystep.adaptive_subspace import AdaptiveSubspace
    from polystep.cma_subspace import CMAAdaptiveSubspace

    model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 4))
    base = AdaptiveSubspace.auto_from_params(model)
    opt = PolyStepOptimizer(
        model,
        subspace=CMAAdaptiveSubspace.from_adaptive_subspace(base),
        epsilon=0.1,
        max_iterations=5,
        use_covariance_adaptation=True,
    )
    assert opt._cma_params["mu_eff"] == 1.0
    n = opt.subspace.subspace_dim
    assert opt._cma_params["c_sigma"] == pytest.approx(3.0 / (n + 4))
    # Path memory must span many steps, not collapse to ~1.
    assert 1.0 / opt._cma_params["c_sigma"] > 10.0


@pytest.mark.gpu
def test_cma_state_is_fp32_under_mixed_precision():
    """bf16 rounds 1 + c_1*x back to 1, so the covariance would never move."""
    from polystep.adaptive_subspace import AdaptiveSubspace
    from polystep.cma_subspace import CMAAdaptiveSubspace

    if not torch.cuda.is_available():
        pytest.skip("mixed_precision requires CUDA")
    model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
    base = AdaptiveSubspace.auto_from_params(model)
    opt = PolyStepOptimizer(
        model,
        subspace=CMAAdaptiveSubspace.from_adaptive_subspace(base),
        epsilon=0.1,
        max_iterations=5,
        use_covariance_adaptation=True,
        mixed_precision=True,
    )
    assert opt.state.C_diag.dtype == torch.float32


def test_newton_step_survives_a_flat_coordinate():
    """Dividing by hessian_reg gives that coordinate a 1e4x step.

    The norm clip is a single global rescale, so without a per-coordinate cap the flat
    coordinate dominates the norm and shrinks every other component by the same factor.
    """
    from polystep.quadratic_model import compute_newton_step

    gradient = torch.full((1, 10), 0.1)
    hessian = torch.ones(1, 10)
    hessian[0, 9] = 0.0  # flat direction

    delta = compute_newton_step(gradient, hessian, max_step_norm=1.0, hessian_reg=1e-4)

    good = delta[0, :9].abs()
    assert (good > 0.01).all(), good
    assert delta[0, 9].abs() <= 1.0 + 1e-6
    assert (delta <= 0).all(), "Newton step must not ascend"


def test_infinite_cost_does_not_flatten_the_plan():
    """An absolute 1e6 penalty enters the 'mean' reduction and flattens every finite
    vertex to the same weight; the penalty must scale with the finite costs.

    Suppression strength depends on epsilon, which the substitution cannot see, so the
    contract is ordering plus preserved contrast, not an absolute weight.
    """
    from polystep.solvers import SoftmaxSolver

    cost = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reference = SoftmaxSolver(epsilon=0.5).solve(cost, scale_cost="mean").matrix

    masked = torch.tensor([[1.0, 2.0, 3.0, float("inf")]])
    plan = SoftmaxSolver(epsilon=0.5).solve(masked, scale_cost="mean").matrix

    # The masked vertex ranks below every finite one.
    assert plan[0, 3] < plan[0, :3].min()
    # The finite vertices keep their ordering and stay far from uniform.
    assert plan[0, 0] > plan[0, 1] > plan[0, 2]
    assert plan[0, 0] / plan[0, 2] > 1.5
    assert reference[0, 0] > reference[0, 1] > reference[0, 2]


def test_all_infinite_cost_gives_uniform_plan_without_nan():
    from polystep.solvers import SoftmaxSolver

    plan = SoftmaxSolver(epsilon=0.5).solve(torch.full((2, 4), float("inf"))).matrix
    assert torch.isfinite(plan).all()
    assert torch.allclose(plan, torch.full_like(plan, 0.125))


def test_entropic_plan_is_invariant_to_a_constant_cost_shift():
    """Scaling before recentering tied the effective temperature to the absolute
    loss level, so adding a constant to every cost changed the plan."""
    from polystep.solvers import SinkhornSolver, SoftmaxSolver

    cost = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0]])
    for solver in (SinkhornSolver(epsilon=0.5, max_iterations=500), SoftmaxSolver(epsilon=0.5)):
        for mode in ("mean", "max_cost"):
            base = solver.solve(cost, scale_cost=mode).matrix
            shifted = solver.solve(cost + 1000.0, scale_cost=mode).matrix
            assert torch.allclose(base, shifted, atol=1e-6), (mode, (base - shifted).abs().max())


def test_fused_softmax_survives_large_negative_costs():
    """The fused kernel divided before recentering, so -C/epsilon overflowed to NaN."""
    from polystep._compiled import CompiledFunctions

    cost = torch.tensor([[-1e6, -1e6 + 1.0]])
    verts = torch.tensor([[1.0], [-1.0]])
    rot = torch.eye(1).unsqueeze(0)
    X = torch.zeros(1, 1)
    X_new, transport = CompiledFunctions(compile=False).fused_softmax_project(
        cost, 1e-3, torch.ones(1), verts, rot, 0.1, X, scale_cost_mean=False
    )
    assert torch.isfinite(X_new).all() and torch.isfinite(transport).all()
    # The cheaper vertex still wins.
    assert transport[0, 0] > transport[0, 1]


def test_anderson_acceleration_actually_accelerates():
    """The update subtracted dX instead of dX+dR, so acceleration was worse than none."""
    from polystep.solvers import SinkhornSolver

    torch.manual_seed(0)
    cost = torch.rand(20, 30) * 5.0
    kwargs = dict(epsilon=0.05, max_iterations=400, threshold=1e-7, check_every=1)
    plain = SinkhornSolver(anderson_depth=0, **kwargs).solve(cost)
    accelerated = SinkhornSolver(anderson_depth=3, **kwargs).solve(cost)

    assert plain.converged and accelerated.converged
    assert accelerated.n_iters < plain.n_iters


def test_single_particle_does_not_freeze():
    """Balanced OT with one row forces a uniform plan, so the iterate never moved."""
    torch.manual_seed(0)
    model = nn.Linear(1, 1, bias=False)
    inputs, targets = torch.randn(8, 1), torch.randn(8, 1)
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    start = model.weight.detach().clone()

    with pytest.warns(UserWarning, match="num_particles=1"):
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=20, seed=1)
    assert opt.state.X.shape[0] == 1

    for _ in range(5):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets))
    assert not torch.equal(start, model.weight.detach())


def test_explicit_sinkhorn_with_one_particle_warns():
    torch.manual_seed(0)
    model = nn.Linear(1, 1, bias=False)
    with pytest.warns(UserWarning, match="uniform transport plan"):
        PolyStepOptimizer(model, solver="sinkhorn", epsilon=0.1, max_iterations=20)


def test_kl_softmax_rejects_a_misshapen_warm_start():
    """A (1, m) init_g broadcast through the updates and produced a 3-D plan."""
    from polystep.solvers.kl_softmax import KLSoftmaxSolver

    cost = torch.rand(2, 2)
    solver = KLSoftmaxSolver(epsilon=0.1, lam=1.0, max_iterations=20)
    with pytest.warns(UserWarning):
        result = solver.solve(cost, init_g=torch.zeros(1, 2))
    assert result.matrix.shape == cost.shape


def test_sinkhorn_warns_when_only_one_marginal_is_unbalanced():
    """a=[1,1] against the default unit-mass b is infeasible and must warn."""
    from polystep.solvers import SinkhornSolver

    cost = torch.rand(2, 3)
    with pytest.warns(UserWarning, match="unequal total mass"):
        SinkhornSolver(epsilon=0.5, max_iterations=50).solve(cost, a=torch.ones(2))


def test_multifidelity_screen_actually_skips_forwards():
    """The screen has to run before the full-fidelity sweep, or it saves nothing."""
    torch.manual_seed(0)
    inputs, targets = torch.randn(64, 12), torch.randn(64, 3)
    seen = {}

    def build(screen):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        seen[screen] = 0

        def closure(batched, _in=inputs, _tgt=targets, _key=screen):
            seen[_key] += next(iter(batched.values())).shape[0] * _in.shape[0]
            return evaluator.evaluate(batched, _in, _tgt)

        opt = PolyStepOptimizer(
            model,
            epsilon=0.1,
            max_iterations=20,
            seed=3,
            multifidelity_screen=screen,
            polytope_type="orthoplex",
            screen_keep_ratio=0.5,
            screen_fidelity=0.25,
        )
        return closure, opt

    for screen in (False, True):
        closure, opt = build(screen)
        screen_closure = opt.screen_closure_from(closure, inputs, targets)
        assert (screen_closure is not None) == screen
        for _ in range(5):
            opt.step(closure, screen_closure=screen_closure)

    # Budget is screen_fidelity + screen_keep_ratio of the unscreened cost.
    assert seen[True] < 0.85 * seen[False], (seen[True], seen[False])


def test_multifidelity_screen_warns_when_it_cannot_run():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
    inputs, targets = torch.randn(16, 12), torch.randn(16, 3)
    opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=20, multifidelity_screen=True, polytope_type="orthoplex")
    with pytest.warns(UserWarning, match="no forward evaluations are saved"):
        opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets))


def test_batched_linear_rejects_a_flatten_after_a_linear():
    """The bmm path pre-flattens the input, which is only valid before the first Linear."""
    from polystep.cost_nn import BatchedLinearEvaluator

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Flatten(), nn.Linear(6, 2))
    assert BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss()) is None

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    layout = ParamLayout.from_module(model)
    flat = layout.flatten(model).reshape(-1)
    stacked = layout.batch_unflatten(torch.stack([flat, flat * 1.01]))
    losses = evaluator.evaluate(stacked, torch.randn(5, 2, 3), torch.randint(0, 2, (5,)))
    assert losses.shape == (2,) and torch.isfinite(losses).all()


class TestNaNRevertRestoresState:
    """A NaN step must clear the derived state, not merely leave it finite.

    A partial revert keeps running and stays finite, then poisons later steps through
    momentum, the transport EMA, or sqrt(C_diag). A NaN loss does not reach this path
    (``sanitize_cost`` substitutes a penalty); a poisoned X or velocity does.
    """

    @staticmethod
    def _build(**kwargs):
        model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        opt = PolyStepOptimizer(model, epsilon=0.1, seed=0, compile=False, **kwargs)
        x = torch.randn(8, 8, generator=torch.Generator().manual_seed(0))

        def closure(batched_params):
            n = next(iter(batched_params.values())).shape[0]
            return torch.stack(
                [
                    functional_call(model, {k: v[i] for k, v in batched_params.items()}, (x,)).pow(2).mean()
                    for i in range(n)
                ]
            )

        return opt, closure

    def test_a_poisoned_velocity_reverts_x_to_its_pre_step_value(self):
        """Pre-step X is finite here, so the revert restores it rather than the origin."""
        opt, closure = self._build(use_momentum=True, momentum_init=0.9, momentum_final=0.9)
        opt.step(closure)
        before = opt.state.X.clone()

        opt.state.velocity = torch.full_like(opt.state.velocity, float("inf"))
        opt.step(closure)

        torch.testing.assert_close(opt.state.X, before, rtol=0, atol=0)
        assert torch.count_nonzero(opt.state.velocity) == 0, "velocity survived the revert"
        assert opt.state.displacement_sqnorms[-1] == 0.0, "reported a move that did not happen"

    def test_a_poisoned_x_reverts_to_the_origin(self):
        """With no finite point to return to, the revert falls back to the origin."""
        opt, closure = self._build(use_momentum=True)
        opt.step(closure)

        opt.state.X = torch.full_like(opt.state.X, float("nan"))
        opt.step(closure)

        assert torch.count_nonzero(opt.state.X) == 0
        assert torch.count_nonzero(opt.state.velocity) == 0

    def test_amortized_transport_direction_is_dropped(self):
        opt, closure = self._build(amortize_steps=3)
        opt.step(closure)
        assert opt._transport_direction_ema is not None

        opt.state.X = torch.full_like(opt.state.X, float("nan"))
        opt.step(closure)
        assert opt._transport_direction_ema is None

    def test_cma_state_returns_to_its_identity_not_merely_to_finite(self):
        from polystep.adaptive_subspace import AdaptiveSubspace

        model = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4))
        base = AdaptiveSubspace.auto_from_params(model)
        opt = PolyStepOptimizer(
            model,
            subspace=CMAAdaptiveSubspace.from_adaptive_subspace(base),
            epsilon=0.1,
            use_covariance_adaptation=True,
            seed=0,
            compile=False,
        )
        x = torch.randn(8, 16, generator=torch.Generator().manual_seed(0))

        def closure(batched_params):
            n = next(iter(batched_params.values())).shape[0]
            return torch.stack(
                [
                    functional_call(model, {k: v[i] for k, v in batched_params.items()}, (x,)).pow(2).mean()
                    for i in range(n)
                ]
            )

        for _ in range(2):
            opt.step(closure)
        assert not torch.allclose(opt.state.C_diag, torch.ones_like(opt.state.C_diag)), "covariance never adapted"

        opt.state.X = torch.full_like(opt.state.X, float("nan"))
        opt.step(closure)

        torch.testing.assert_close(opt.state.C_diag, torch.ones_like(opt.state.C_diag))
        assert torch.count_nonzero(opt.state.p_c) == 0
        assert torch.count_nonzero(opt.state.p_sigma) == 0
