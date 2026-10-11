"""Tests for optimizer checkpoint/resume and the restore_best training guard."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer, ProgressiveEpsilon, TrainConfig, train
from polystep.hybrid_subspace import HybridSubspace
from polystep.optimizer import RankSchedule
from polystep.transform import ParamLayout
from polystep.cost_nn import NNCostEvaluator


def _model_and_closure(seed=0, in_dim=12, out_dim=3, samples=32):
    gen = torch.Generator().manual_seed(500 + seed)
    inputs = torch.randn(samples, in_dim, generator=gen)
    targets = torch.randn(samples, out_dim, generator=gen)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, out_dim))
    evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    return model, closure


class TestStateDictRoundTrip:
    """state_dict must capture enough to continue a run bit-for-bit."""

    def test_round_trip_preserves_solver_state(self):
        model, closure = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=20, use_momentum=True)
        for _ in range(3):
            opt.step(closure)

        saved = copy.deepcopy(opt.state_dict())
        x_before = opt.state.X.clone()
        iteration_before = opt.state.iteration_count

        for _ in range(2):
            opt.step(closure)
        assert not torch.equal(x_before, opt.state.X)

        opt.load_state_dict(saved)
        assert torch.equal(x_before, opt.state.X)
        assert opt.state.iteration_count == iteration_before

    def test_format_key_present(self):
        model, _ = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10)
        assert "format" in opt.state_dict()

    def test_unknown_keys_are_ignored(self):
        model, closure = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10)
        opt.step(closure)
        sd = opt.state_dict()
        before = opt.state.X.clone()
        sd["solver_state"]["a_field_from_a_future_version"] = 1234
        opt.load_state_dict(sd)
        # Skipping the key must not cost the keys around it.
        torch.testing.assert_close(opt.state.X, before)
        assert opt.state.iteration_count == 1

    def test_resume_matches_an_uninterrupted_run(self):
        """N steps must equal N/2 steps, save, load into a fresh optimizer, N/2 more."""
        model_a, closure_a = _model_and_closure(seed=1)
        opt_a = PolyStepOptimizer(model_a, epsilon=0.1, max_iterations=20, seed=7, use_momentum=True)
        for _ in range(6):
            opt_a.step(closure_a)
        reference = [p.detach().clone() for p in model_a.parameters()]

        model_b, closure_b = _model_and_closure(seed=1)
        opt_b = PolyStepOptimizer(model_b, epsilon=0.1, max_iterations=20, seed=7, use_momentum=True)
        for _ in range(3):
            opt_b.step(closure_b)
        checkpoint = copy.deepcopy(opt_b.state_dict())
        weights = copy.deepcopy(model_b.state_dict())

        model_c, closure_c = _model_and_closure(seed=1)
        opt_c = PolyStepOptimizer(model_c, epsilon=0.1, max_iterations=20, seed=7, use_momentum=True)
        model_c.load_state_dict(weights)
        opt_c.load_state_dict(checkpoint)
        for _ in range(3):
            opt_c.step(closure_c)

        for expected, got in zip(reference, model_c.parameters()):
            assert torch.allclose(expected, got, atol=1e-6), (expected - got).abs().max()
        assert opt_c.candidate_evals == opt_a.candidate_evals

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"amortize_steps": 3, "use_momentum": True}, id="amortize"),
            pytest.param(
                # orthoplex, or the constructor disables both and this tests nothing.
                {
                    "trust_region": True,
                    "use_quadratic_model": True,
                    "num_probe": 2,
                    "polytope_type": "orthoplex",
                },
                id="trust_region",
            ),
            pytest.param({"hybrid": True}, id="hybrid_subspace"),
            pytest.param(
                {"hybrid": True, "block_strategy": "per_layer", "solver": "sinkhorn", "dual_momentum_beta": 0.5},
                id="hybrid_row_segments",
            ),
            pytest.param({"probe_radius_jitter": 0.2}, id="probe_jitter"),
            pytest.param({"adaptive": True}, id="adaptive_subspace"),
        ],
    )
    def test_resume_is_exact_for_stateful_features(self, kwargs):
        """Features whose state lives outside SolverState still resume bit-exactly."""
        kwargs = dict(kwargs)
        hybrid = kwargs.pop("hybrid", False)
        adaptive = kwargs.pop("adaptive", False)

        def build():
            model, closure = _model_and_closure(seed=2)
            subspace = None
            if hybrid:
                from polystep.hybrid_subspace import HybridSubspace
                from polystep.transform import ParamLayout

                subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4, seed=3)
            if adaptive:
                from polystep.adaptive_subspace import AdaptiveSubspace

                subspace = AdaptiveSubspace.auto_from_params(model)
            opt = PolyStepOptimizer(
                model,
                subspace=subspace,
                epsilon=0.1,
                max_iterations=20,
                seed=7,
                **kwargs,
            )
            return model, closure, opt

        model_a, closure_a, opt_a = build()
        for _ in range(6):
            opt_a.step(closure_a)
        reference = [p.detach().clone() for p in model_a.parameters()]

        model_b, closure_b, opt_b = build()
        for _ in range(3):
            opt_b.step(closure_b)
        checkpoint = copy.deepcopy(opt_b.state_dict())
        weights = copy.deepcopy(model_b.state_dict())

        model_c, closure_c, opt_c = build()
        model_c.load_state_dict(weights)
        opt_c.load_state_dict(checkpoint)
        for _ in range(3):
            opt_c.step(closure_c)

        for expected, got in zip(reference, model_c.parameters()):
            assert torch.allclose(expected, got, atol=1e-6), (expected - got).abs().max()

    def test_subspace_load_drops_block_state(self):
        """Subspace runs keep their duals in state.f and state.g, so per-block state from
        an older checkpoint must not survive the load."""
        model, _ = _model_and_closure()
        opt = PolyStepOptimizer(
            model, subspace=HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2), epsilon=0.1
        )
        sd = opt.state_dict()
        stale = [(torch.zeros(2), torch.zeros(3))]
        sd["solver_state"]["block_duals"] = stale
        sd["prev_prev_block_duals"] = stale
        sd["control"]["_prev_block_descent_directions"] = [torch.zeros(2)]
        opt.load_state_dict(sd)
        assert opt.state.block_duals is None
        assert opt.state._prev_prev_block_duals is None
        assert opt._prev_block_descent_directions is None

    def test_rejects_a_newer_format(self):
        model, _ = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10)
        sd = opt.state_dict()
        sd["format"] = sd["format"] + 1
        with pytest.raises(ValueError, match="Unsupported optimizer state_dict format"):
            opt.load_state_dict(sd)

    def test_rejects_an_older_format(self):
        model, _ = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10)
        sd = opt.state_dict()
        sd["format"] = 5
        with pytest.raises(ValueError, match="Unsupported optimizer state_dict format"):
            opt.load_state_dict(sd)

    def test_generator_state_restores(self):
        model, closure = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10, seed=11)
        opt.step(closure)
        saved = copy.deepcopy(opt.state_dict())
        expected = opt._generator.get_state().clone()

        opt.step(closure)
        opt.load_state_dict(saved)
        assert torch.equal(expected, opt._generator.get_state())

    def test_progressive_epsilon_restores(self):
        model, closure = _model_and_closure()
        opt = PolyStepOptimizer(
            model,
            epsilon=ProgressiveEpsilon(init=0.5, target=0.05, max_epsilon=2.5),
            solver="sinkhorn",
            max_iterations=20,
        )
        for _ in range(3):
            opt.step(closure)
        saved = copy.deepcopy(opt.state_dict())
        assert saved["progressive_epsilon"] is not None
        current = opt._progressive_epsilon._current
        smoothed = opt._progressive_epsilon._smoothed

        for _ in range(3):
            opt.step(closure)
        opt.load_state_dict(saved)

        assert opt._progressive_epsilon._current == current
        assert opt._progressive_epsilon._smoothed == smoothed

    def test_tensors_are_detached_cpu_copies(self):
        """The snapshot must not alias live state, or later steps mutate it."""
        model, closure = _model_and_closure()
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=10)
        opt.step(closure)
        saved = opt.state_dict()
        snapshot = saved["solver_state"]["X"].clone()

        for _ in range(2):
            opt.step(closure)

        assert torch.equal(snapshot, saved["solver_state"]["X"])


class TestRestoreBest:
    """restore_best defaults to True, so it decides what train() hands back."""

    def _loaders(self, samples=32, in_dim=12, out_dim=3, batch_size=16):
        gen = torch.Generator().manual_seed(99)
        inputs = torch.randn(samples, in_dim, generator=gen)
        targets = torch.randn(samples, out_dim, generator=gen)
        return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=False)

    def test_returns_weights_no_worse_than_the_final_step(self):
        loader = self._loaders(batch_size=32)
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=40)
        losses = []

        class Recorder:
            def on_step_end(self, metrics):
                losses.append(metrics["loss"])
                return False

            def on_epoch_end(self, metrics):
                pass

        train(
            model,
            loader,
            nn.MSELoss(),
            opt,
            TrainConfig(epochs=6, callbacks=[Recorder()], restore_best=True),
        )

        inputs, targets = next(iter(self._loaders(batch_size=32)))
        with torch.no_grad():
            final = nn.functional.mse_loss(model(inputs), targets).item()

        assert losses, "callback never fired"
        assert final <= min(losses) + 1e-4

    def test_disabled_keeps_the_last_step(self):
        """With restore_best off, a run that ends worse than its best stays worse."""
        loader = self._loaders()
        results = {}
        for flag in (True, False):
            torch.manual_seed(3)
            model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
            opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=40)
            train(
                model,
                loader,
                nn.MSELoss(),
                opt,
                TrainConfig(epochs=3, restore_best=flag),
            )
            inputs, targets = next(iter(self._loaders(batch_size=32)))
            with torch.no_grad():
                results[flag] = nn.functional.mse_loss(model(inputs), targets).item()

        assert results[True] <= results[False] + 1e-6

    def test_reverts_a_deliberate_late_divergence(self):
        """A callback that blows the weights up on the last step must be undone."""
        loader = self._loaders()
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3))
        opt = PolyStepOptimizer(model, epsilon=0.1, max_iterations=40)

        state = {"n": 0}

        class Saboteur:
            def on_step_end(self, metrics):
                state["n"] += 1
                if state["n"] == 4:
                    with torch.no_grad():
                        for p in model.parameters():
                            p.mul_(50.0)
                return False

            def on_epoch_end(self, metrics):
                pass

        train(
            model,
            loader,
            nn.MSELoss(),
            opt,
            TrainConfig(epochs=2, callbacks=[Saboteur()], restore_best=True),
        )

        inputs, targets = next(iter(self._loaders(batch_size=32)))
        with torch.no_grad():
            final = nn.functional.mse_loss(model(inputs), targets).item()
        assert final < 100.0, f"late divergence was not reverted: {final}"


class TestAbsorbAlignedActive:
    """absorb_aligned_active changes how projections are rebuilt at an absorb."""

    def _run(self, aligned):
        from polystep.hybrid_subspace import HybridSubspace
        from polystep.transform import ParamLayout

        model, closure = _model_and_closure(seed=4)
        subspace = HybridSubspace.from_layout(
            ParamLayout.from_module(model),
            rank=4,
            absorb_mode="periodic",
            absorb_interval=2,
            absorb_aligned_active=aligned,
        )
        opt = PolyStepOptimizer(model, subspace=subspace, epsilon=0.1, max_iterations=20)
        for _ in range(6):
            opt.step(closure)
        return opt

    def test_flag_defaults_off(self):
        from polystep.hybrid_subspace import HybridSubspace
        from polystep.transform import ParamLayout

        model, _ = _model_and_closure()
        subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4)
        assert subspace.absorb_aligned_active is False

    def test_both_settings_absorb_and_stay_finite(self):
        for aligned in (False, True):
            opt = self._run(aligned)
            assert opt.state.absorb_count > 0
            assert torch.isfinite(opt.state.X).all()

    def test_aligned_changes_the_projections(self):
        """Displacement-SVD rebuild must not reproduce the fresh-redraw basis."""
        plain = self._run(False)
        aligned = self._run(True)
        key = next(iter(plain.state.hybrid_projections))
        assert not torch.allclose(
            plain.state.hybrid_projections[key],
            aligned.state.hybrid_projections[key],
        )


def test_rank_schedule_stage_zero_applies_before_the_first_step():
    """``RankSchedule.at(0)`` must apply before the first step, not after the
    first OT solve advances ``iteration_count``.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 3))
    supplied = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=4)
    dim_at_supplied_rank = supplied.subspace_dim

    opt = PolyStepOptimizer(
        model,
        subspace=supplied,
        rank_schedule=RankSchedule(stages=[(0, 2), (100, 8)]),
        compile=False,
        seed=0,
    )

    assert opt._applied_rank == 2
    assert opt.subspace.subspace_dim != dim_at_supplied_rank


def test_resume_across_a_rank_transition():
    """A checkpoint taken after a rank transition must restore ``_applied_rank``,
    so wide coordinates do not land in a narrow subspace.
    """
    schedule = RankSchedule(stages=[(0, 1), (2, 2)])
    inputs, targets = torch.randn(16, 8), torch.randn(16, 3)
    loss_fn = nn.MSELoss()

    def build():
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 3))
        subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
        opt = PolyStepOptimizer(model, subspace=subspace, rank_schedule=schedule, compile=False, seed=0)
        return model, opt

    def closure_for(model):
        from torch.func import functional_call, vmap

        return lambda p: vmap(lambda q: loss_fn(functional_call(model, q, (inputs,)), targets))(p)

    saved_model, saved = build()
    for _ in range(4):
        saved.step(closure_for(saved_model))
    checkpoint = saved.state_dict()
    assert saved._applied_rank == 2
    expected = saved.step(closure_for(saved_model))

    resumed_model, resumed = build()
    resumed.load_state_dict(checkpoint)
    actual = resumed.step(closure_for(resumed_model))

    assert actual == pytest.approx(expected, abs=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("src,dst", [("cuda", "cpu"), ("cpu", "cuda")])
def test_seeded_checkpoint_moves_between_device_types(src, dst):
    inputs, targets = torch.randn(8, 4), torch.randint(0, 3, (8,))

    def build(device):
        torch.manual_seed(0)
        model = nn.Linear(4, 3).to(device)
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        x, y = inputs.to(device), targets.to(device)
        return model, PolyStepOptimizer(model, seed=0, solver="softmax"), lambda p: evaluator.evaluate(p, x, y)

    saved_model, saved, saved_closure = build(src)
    saved.step(saved_closure)
    checkpoint = saved.state_dict()

    model, resumed, closure = build(dst)
    model.load_state_dict(saved_model.state_dict())
    with pytest.warns(UserWarning, match="another device type"):
        resumed.load_state_dict(checkpoint)
    assert torch.isfinite(torch.as_tensor(resumed.step(closure)))
