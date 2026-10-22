"""Tests for the high-level training API: train(), TrainConfig, TrainCallback,
LoggingCallback, EarlyStoppingCallback, get_diagnostics."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import (
    HybridSubspace,
    NNCostEvaluator,
    ParamLayout,
    PolyStepOptimizer,
    train,
    TrainConfig,
    TrainCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    get_diagnostics,
)


def _make_model():
    """Small MLP for fast testing."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _make_dataloader(num_samples=32, batch_size=16):
    """Create a simple DataLoader with random data."""
    torch.manual_seed(42)
    X = torch.randn(num_samples, 4)
    y = torch.randn(num_samples, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_optimizer(model):
    """Create a PolyStepOptimizer with fast settings."""
    return PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=0.1,
        sinkhorn_max_iters=100,
    )


def _batch(n=6, d=4, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, c, (n,), generator=g)


@pytest.mark.parametrize("epochs", [0, -1])
def test_invalid_epochs(epochs):
    with pytest.raises(ValueError):
        TrainConfig(epochs=epochs)


def test_parameters_generator_raises_type_error():
    with pytest.raises(TypeError, match="nn.Module"):
        PolyStepOptimizer(nn.Linear(4, 2).parameters())


class TestTrain:
    def test_train_runs_correct_steps(self):
        """With epochs=2 and 2 batches/epoch, expect 4 on_step_end calls."""
        torch.manual_seed(42)
        model = _make_model()
        dl = _make_dataloader(num_samples=32, batch_size=16)  # 2 batches
        opt = _make_optimizer(model)

        class StepCounter(TrainCallback):
            def __init__(self):
                self.count = 0

            def on_step_end(self, metrics):
                self.count += 1
                return False

        counter = StepCounter()
        config = TrainConfig(epochs=2, callbacks=[counter])
        train(model, dl, nn.MSELoss(), opt, config)
        assert counter.count == 4

    def test_train_metrics_keys(self):
        """Metrics dict contains all 8 required keys."""
        torch.manual_seed(42)
        model = _make_model()
        dl = _make_dataloader()
        opt = _make_optimizer(model)

        class MetricsCapture(TrainCallback):
            def __init__(self):
                self.first_metrics = None

            def on_step_end(self, metrics):
                if self.first_metrics is None:
                    self.first_metrics = dict(metrics)
                return False

        capture = MetricsCapture()
        config = TrainConfig(epochs=1, callbacks=[capture])
        train(model, dl, nn.MSELoss(), opt, config)

        expected_keys = {
            "step",
            "epoch",
            "loss",
            "ot_cost",
            "displacement",
            "velocity_mag",
            "converged",
            "absorb_count",
        }
        assert set(capture.first_metrics.keys()) == expected_keys

    def test_train_early_stop(self):
        """Callback returning True after 2 calls stops training."""
        torch.manual_seed(42)
        model = _make_model()
        dl = _make_dataloader(num_samples=32, batch_size=16)  # 2 batches
        opt = _make_optimizer(model)

        class EarlyStop(TrainCallback):
            def __init__(self):
                self.count = 0

            def on_step_end(self, metrics):
                self.count += 1
                if self.count >= 2:
                    return True
                return False

        stopper = EarlyStop()
        config = TrainConfig(epochs=2, callbacks=[stopper])
        train(model, dl, nn.MSELoss(), opt, config)
        assert stopper.count == 2

    def test_train_epoch_end_called(self):
        """on_epoch_end is called once per epoch."""
        torch.manual_seed(42)
        model = _make_model()
        dl = _make_dataloader()
        opt = _make_optimizer(model)

        class EpochCounter(TrainCallback):
            def __init__(self):
                self.count = 0

            def on_epoch_end(self, metrics):
                self.count += 1

        counter = EpochCounter()
        config = TrainConfig(epochs=3, callbacks=[counter])
        train(model, dl, nn.MSELoss(), opt, config)
        assert counter.count == 3

    @pytest.mark.parametrize("reduction", ["mean", "none"])
    @pytest.mark.parametrize("precision", ["float64", "mixed_precision", "candidate_autocast"])
    def test_train_scores_incumbent_like_candidates(self, precision, reduction):
        """FP32 batches reach an FP64 or BF16 model through the evaluator's input cast,
        under candidate_autocast the incumbent is scored in BF16 like every candidate,
        and an unreduced loss is averaged."""
        torch.manual_seed(42)
        model = _make_model()
        if precision == "float64":
            model.double()
        opt = PolyStepOptimizer(
            model,
            seed=42,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            mixed_precision=precision == "mixed_precision",
            candidate_autocast=precision == "candidate_autocast",
        )
        dl = _make_dataloader(num_samples=16, batch_size=16)
        x, y = next(iter(dl))

        class Rescore(TrainCallback):
            def __init__(self):
                self.pairs = []

            def on_step_end(self, metrics):
                with torch.inference_mode(), torch.autocast("cpu", enabled=precision == "candidate_autocast"):
                    expected = nn.MSELoss()(model(x.to(next(model.parameters()).dtype)), y).item()
                self.pairs.append((metrics["loss"], expected))
                return False

        rescore = Rescore()
        train(model, dl, nn.MSELoss(reduction=reduction), opt, TrainConfig(epochs=3, callbacks=[rescore]))
        assert len(rescore.pairs) == 3
        assert all(loss == expected for loss, expected in rescore.pairs), rescore.pairs

    def test_train_releases_evaluator_when_callback_raises(self):
        """``release_evaluator()`` must also run on an exception, or the batch stays alive
        on the optimizer for the rest of its lifetime (device memory on CUDA).
        """

        class Boom(TrainCallback):
            def on_step_end(self, metrics):
                raise RuntimeError("boom")

        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        optimizer = PolyStepOptimizer(model)
        loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,))), batch_size=4)

        with pytest.raises(RuntimeError, match="boom"):
            train(model, loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=1, callbacks=[Boom()]))

        assert optimizer._cost_evaluator is None
        assert optimizer._fused_inputs is None
        assert optimizer._fused_targets is None
        assert optimizer._registration_closure is None

    def test_dict_and_tuple_batches_train_alike(self):
        class Masked(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x, mask):
                return self.lin(x) * mask

        x, y = _batch(n=16)
        mask = (torch.rand(16, 3, generator=torch.Generator().manual_seed(1)) > 0.3).float()
        rows = [{"x": x[i], "mask": mask[i], "labels": y[i]} for i in range(16)]
        weights = []
        for dataset in (rows, TensorDataset(x, mask, y)):
            torch.manual_seed(0)
            model = Masked()
            start = model.lin.weight.detach().clone()
            opt = PolyStepOptimizer(model, seed=0, multifidelity_screen=True)
            config = TrainConfig(epochs=1, restore_best=False)
            train(model, DataLoader(dataset, batch_size=8), nn.CrossEntropyLoss(), opt, config)
            weights.append(model.lin.weight.detach())
        assert not torch.equal(weights[0], start)
        assert torch.equal(*weights)

    def test_cost_subsample_keeps_scalar_inputs(self):
        class Scaled(nn.Linear):
            def forward(self, x, scale):
                return super().forward(x) * scale

        x, y = _batch(n=8)
        opt = PolyStepOptimizer(Scaled(4, 3), seed=0, cost_batch_size=4)
        opt.step_batch((x, torch.tensor(2.0)), y, loss_fn=nn.CrossEntropyLoss())
        sub_x, scale = opt._fused_inputs
        assert sub_x.shape[0] == opt._fused_targets.shape[0] == 4
        assert scale.dim() == 0 and scale.item() == 2.0

    def test_cost_subsample_scalar_first_input(self):
        class Scaled(nn.Linear):
            def forward(self, scale, x):
                return super().forward(x) * scale

        class Weighted(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(3))

            def forward(self, scale):
                return self.w * scale

        class ScaledEmbedding(nn.Embedding):
            def forward(self, scale, ids):
                return super().forward(ids) * scale

        x, y = _batch(n=8)
        for autocast in (False, True):
            opt = PolyStepOptimizer(Scaled(4, 3), seed=0, cost_batch_size=4, candidate_autocast=autocast)
            opt.step_batch({"scale": torch.tensor(2.0), "x": x}, y, loss_fn=nn.CrossEntropyLoss())
            assert opt._fused_inputs["x"].shape[0] == opt._fused_targets.shape[0] == 4
            assert opt._fused_inputs["scale"].dim() == 0
            opt = PolyStepOptimizer(Weighted(), seed=0, cost_batch_size=2, candidate_autocast=autocast)
            loss = opt.step_batch({"scale": torch.tensor(2.0)}, torch.zeros(3), loss_fn=nn.MSELoss())
            assert torch.isfinite(torch.tensor(loss)) and opt._fused_targets.shape == (3,)
        opt = PolyStepOptimizer(ScaledEmbedding(10, 3), seed=0, cost_batch_size=4)
        opt.step_batch({"scale": torch.tensor(2.0), "ids": torch.arange(8)}, torch.zeros(8, 3), loss_fn=nn.MSELoss())
        assert opt._fused_inputs["ids"].shape[0] == 4

    def test_train_rejects_foreign_model(self):
        """Candidates would score one model while steps were written into the other."""
        a, b = nn.Linear(4, 2), nn.Linear(4, 2)
        loader = DataLoader(TensorDataset(*_batch(c=2)), batch_size=3)
        with pytest.raises(ValueError, match="different model"):
            train(b, loader, nn.CrossEntropyLoss(), PolyStepOptimizer(a, seed=0), TrainConfig(epochs=1))

    def test_train_rejects_trust_region(self):
        """Its ratio would compare a prediction against the loss on the next batch."""
        model = nn.Linear(4, 2)
        loader = DataLoader(TensorDataset(*_batch(c=2)), batch_size=3)
        opt = PolyStepOptimizer(model, seed=0, trust_region=True, polytope_type="orthoplex")
        opt.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss()), *_batch(c=2))
        with pytest.raises(ValueError, match="trust_region is not supported by train"):
            train(model, loader, nn.CrossEntropyLoss(), opt, TrainConfig(epochs=1))
        assert opt._cost_evaluator is not None

    def test_train_restores_model_train_mode(self):
        """The evaluator switches to eval to freeze statistics; train() must switch back."""
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        model.train()
        loader = DataLoader(TensorDataset(*_batch(c=2)), batch_size=3)
        opt = PolyStepOptimizer(model, seed=0)
        loss_fn = nn.CrossEntropyLoss()
        config = TrainConfig(epochs=1, restore_best=False, callbacks=[TrainCallback()])
        train(model, loader, loss_fn, opt, config)
        running_mean = model[1].running_mean.clone()
        train(model, loader, loss_fn, opt, config)
        assert torch.equal(model[1].running_mean, running_mean)
        assert model.training

    def test_train_compiled_module_matches_plain(self):
        plain = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        inner = copy.deepcopy(plain)
        loader = DataLoader(TensorDataset(*_batch(n=12)), batch_size=6)
        for model in (plain, torch.compile(inner)):
            opt = PolyStepOptimizer(model, compile=False, seed=0)
            train(model, loader, nn.CrossEntropyLoss(), opt, TrainConfig(epochs=1, restore_best=True))
        for a, b in zip(plain.parameters(), inner.parameters()):
            assert torch.equal(a, b)

    @pytest.mark.parametrize("written", [{"calibrated": False}, {"calibrated": True, "steps": 1}])
    def test_train_restores_non_tensor_extra_state(self, written):
        class Calibrated(nn.Linear):
            def get_extra_state(self):
                return dict(self.meta)

            def set_extra_state(self, state):
                self.meta = dict(state)

        torch.manual_seed(0)
        model = nn.Sequential(Calibrated(5, 4), nn.Tanh(), nn.Linear(4, 2))
        model[0].meta = {"calibrated": True}
        layout = ParamLayout.from_module(model)
        opt = PolyStepOptimizer(model, subspace=HybridSubspace.from_layout(layout, rank=2), seed=0, max_iterations=2)
        loader = DataLoader(TensorDataset(*_batch(n=64, d=5, c=2)), batch_size=32)

        class Recalibrate(TrainCallback):
            def on_step_end(self, metrics):
                model[0].meta = dict(written)
                return True

        config = TrainConfig(epochs=1, callbacks=[Recalibrate()], restore_best=True)
        train(model, loader, nn.CrossEntropyLoss(), opt, config)
        assert layout.total_params == 34
        assert model[0].meta == {"calibrated": True}

    def test_noop_callback_keeps_trajectory(self):
        class Stateful(nn.Linear):
            def __init__(self):
                super().__init__(2, 1)
                self.register_buffer("calls", torch.zeros(()))

            def forward(self, x):
                self.calls.add_(1)
                return super().forward(x) + self.calls + 0.1 * torch.randn(())

        costs = []
        for config in (TrainConfig(epochs=1, restore_best=False), TrainConfig(epochs=1, callbacks=[TrainCallback()])):
            torch.manual_seed(0)
            model = Stateful()
            opt = PolyStepOptimizer(model, solver="softmax", seed=0, compile_forward=False)
            train(model, [(torch.ones(4, 2), torch.zeros(4, 1))] * 3, nn.MSELoss(), opt, config)
            costs.append(opt.state.costs)
        assert costs[0] == costs[1]

    @pytest.mark.parametrize("restore_best", [False, True])
    def test_callback_loss_tracks_current_batch(self, restore_best):
        torch.manual_seed(0)
        model = nn.Linear(2, 1)
        opt = PolyStepOptimizer(model, solver="softmax", seed=0)
        batches = [(torch.zeros(4, 2), torch.full((4, 1), v)) for v in (0.0, 10.0)]
        pairs = []

        class Rescore(TrainCallback):
            def on_step_end(self, metrics):
                x, y = batches[metrics["step"]]
                with torch.inference_mode():
                    pairs.append((metrics["loss"], nn.functional.mse_loss(model(x), y).item()))
                return False

        config = TrainConfig(epochs=1, restore_best=restore_best, callbacks=[Rescore()])
        train(model, batches, nn.MSELoss(), opt, config)
        assert len(pairs) == 2
        assert all(loss == pytest.approx(expected) for loss, expected in pairs), pairs


class TestLoggingCallback:
    def test_logs_at_interval(self, capsys):
        cb = LoggingCallback(log_every=2)
        base = {"loss": 0.5, "ot_cost": 1.0, "displacement": 0.001, "converged": True}
        for step in range(4):
            cb.on_step_end({**base, "step": step})
        captured = capsys.readouterr().out
        assert "[Step 0]" in captured
        assert "[Step 2]" in captured
        assert "[Step 1]" not in captured
        assert "[Step 3]" not in captured

    def test_small_displacement_is_visible(self, capsys):
        cb = LoggingCallback(log_every=1)
        cb.on_step_end({"step": 0, "loss": 0.5, "ot_cost": 1.0, "displacement": 6.5e-7, "converged": True})
        assert "disp_sq=6.500e-07" in capsys.readouterr().out

    def test_epoch_end_prints(self, capsys):
        cb = LoggingCallback()
        cb.on_epoch_end({"epoch": 1, "avg_loss": 0.5})
        captured = capsys.readouterr().out
        assert "Epoch 1" in captured


class TestEarlyStoppingCallback:
    def test_stops_after_patience(self):
        cb = EarlyStoppingCallback(patience=3, min_delta=0.01)
        for loss in [1.0, 0.9, 0.8]:
            assert cb.on_step_end({"loss": loss, "step": 0}) is False
        assert cb.on_step_end({"loss": 0.8, "step": 1}) is False  # counter=1
        assert cb.on_step_end({"loss": 0.8, "step": 2}) is False  # counter=2
        assert cb.on_step_end({"loss": 0.8, "step": 3}) is True  # counter=3 >= patience

    def test_resets_on_improvement(self):
        cb = EarlyStoppingCallback(patience=3)
        cb.on_step_end({"loss": 1.0, "step": 0})
        cb.on_step_end({"loss": 0.95, "step": 1})  # improvement
        cb.on_step_end({"loss": 0.95, "step": 2})  # stagnation counter=1
        cb.on_step_end({"loss": 0.95, "step": 3})  # counter=2
        cb.on_step_end({"loss": 0.8, "step": 4})  # improvement, counter resets
        assert cb.on_step_end({"loss": 0.8, "step": 5}) is False  # counter=1
        assert cb.on_step_end({"loss": 0.8, "step": 6}) is False  # counter=2 < patience=3

    def test_min_delta_threshold(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=0.1)
        cb.on_step_end({"loss": 1.0, "step": 0})
        # 0.95 is NOT 0.1 better than 1.0 -> stagnation
        assert cb.on_step_end({"loss": 0.95, "step": 1}) is False  # counter=1
        assert cb.on_step_end({"loss": 0.92, "step": 2}) is True  # counter=2 >= patience

    def test_patience_below_one_raises(self):
        with pytest.raises(ValueError, match="patience"):
            EarlyStoppingCallback(patience=0)


class TestGetDiagnostics:
    def _run_optimizer(self, model, steps=3, use_momentum=False):
        """Helper to create and run optimizer for a few steps."""
        dl = _make_dataloader()
        opt = PolyStepOptimizer(
            model,
            compile=False,
            seed=42,
            epsilon=0.1,
            sinkhorn_max_iters=100,
            use_momentum=use_momentum,
        )
        loss_fn = nn.MSELoss()
        from polystep.cost_nn import NNCostEvaluator

        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        batch = next(iter(dl))
        inputs, targets = batch
        for _ in range(steps):

            def closure(bp, _in=inputs, _tgt=targets):
                return evaluator.evaluate(bp, _in, _tgt)

            opt.step(closure)
        return opt

    def test_diagnostics_keys(self):
        torch.manual_seed(42)
        model = _make_model()
        opt = self._run_optimizer(model, steps=3)
        diag = get_diagnostics(opt)
        expected = {
            "costs",
            "ess",
            "rho",
            "evals",
            "displacement_sqnorms",
            "convergence",
            "velocity_magnitude",
            "iteration_count",
            "epsilon",
            "radius_multiplier",
            "absorb_count",
        }
        assert set(diag.keys()) == expected
        assert len(diag["costs"]) == diag["iteration_count"] == 3
        assert diag["velocity_magnitude"] is None

    def test_diagnostics_with_momentum(self):
        torch.manual_seed(42)
        model = _make_model()
        opt = self._run_optimizer(model, steps=2, use_momentum=True)
        diag = get_diagnostics(opt)
        # The reported magnitude is the velocity's norm, not just some non-negative float.
        assert diag["velocity_magnitude"] == pytest.approx(opt.state.velocity.norm().item(), rel=1e-6)
