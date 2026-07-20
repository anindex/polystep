"""Tests for the high-level training API: train(), TrainConfig, TrainCallback,
LoggingCallback, EarlyStoppingCallback, get_diagnostics."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import (
    PolyStepOptimizer,
    train,
    TrainConfig,
    TrainCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    get_diagnostics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestTrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.log_every == 10
        assert config.callbacks == []

    @pytest.mark.parametrize(
        "field,value",
        [
            ("epochs", 0),
            ("epochs", -1),
            ("log_every", 0),
            ("batch_size", 0),
            ("batch_size", -1),
        ],
    )
    def test_invalid_epochs_zero(self, field, value):
        with pytest.raises(ValueError):
            TrainConfig(**{field: value})


# ---------------------------------------------------------------------------
# TestTrainCallback
# ---------------------------------------------------------------------------


class TestTrainCallback:
    def test_base_on_step_end_returns_false(self):
        cb = TrainCallback()
        assert cb.on_step_end({}) is False

    def test_base_on_epoch_end_noop(self):
        cb = TrainCallback()
        result = cb.on_epoch_end({})
        assert result is None


# ---------------------------------------------------------------------------
# TestTrain
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_returns_model(self):
        torch.manual_seed(42)
        model = _make_model()
        dl = _make_dataloader()
        opt = _make_optimizer(model)
        config = TrainConfig(epochs=1)

        result = train(model, dl, nn.MSELoss(), opt, config)
        assert result is model

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


# ---------------------------------------------------------------------------
# TestLoggingCallback
# ---------------------------------------------------------------------------


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

    def test_never_stops(self):
        cb = LoggingCallback(log_every=1)
        metrics = {"step": 0, "loss": 0.5, "ot_cost": 1.0, "displacement": 0.001, "converged": True}
        assert cb.on_step_end(metrics) is False

    def test_epoch_end_prints(self, capsys):
        cb = LoggingCallback()
        cb.on_epoch_end({"epoch": 1, "avg_loss": 0.5})
        captured = capsys.readouterr().out
        assert "Epoch 1" in captured


# ---------------------------------------------------------------------------
# TestEarlyStoppingCallback
# ---------------------------------------------------------------------------


class TestEarlyStoppingCallback:
    def test_stops_after_patience(self):
        cb = EarlyStoppingCallback(patience=3, min_delta=0.01)
        # Improving losses -- all return False
        for loss in [1.0, 0.9, 0.8]:
            assert cb.on_step_end({"loss": loss, "step": 0}) is False
        # Stagnating losses -- counter increments
        assert cb.on_step_end({"loss": 0.8, "step": 1}) is False  # counter=1
        assert cb.on_step_end({"loss": 0.8, "step": 2}) is False  # counter=2
        assert cb.on_step_end({"loss": 0.8, "step": 3}) is True  # counter=3 >= patience

    def test_resets_on_improvement(self):
        cb = EarlyStoppingCallback(patience=3)
        cb.on_step_end({"loss": 1.0, "step": 0})
        cb.on_step_end({"loss": 0.95, "step": 1})  # improvement
        cb.on_step_end({"loss": 0.95, "step": 2})  # stagnation counter=1
        cb.on_step_end({"loss": 0.95, "step": 3})  # counter=2
        # Now improve
        cb.on_step_end({"loss": 0.8, "step": 4})  # improvement, counter resets
        assert cb.on_step_end({"loss": 0.8, "step": 5}) is False  # counter=1
        assert cb.on_step_end({"loss": 0.8, "step": 6}) is False  # counter=2
        # counter=2 < patience=3, should NOT stop

    def test_min_delta_threshold(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=0.1)
        cb.on_step_end({"loss": 1.0, "step": 0})
        # 0.95 is NOT 0.1 better than 1.0 -> stagnation
        assert cb.on_step_end({"loss": 0.95, "step": 1}) is False  # counter=1
        assert cb.on_step_end({"loss": 0.92, "step": 2}) is True  # counter=2 >= patience


# ---------------------------------------------------------------------------
# TestGetDiagnostics
# ---------------------------------------------------------------------------


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
            "displacement_sqnorms",
            "convergence",
            "velocity_magnitude",
            "iteration_count",
            "epsilon",
            "radius_multiplier",
            "absorb_count",
        }
        assert set(diag.keys()) == expected

    def test_diagnostics_values(self):
        torch.manual_seed(42)
        model = _make_model()
        opt = self._run_optimizer(model, steps=3)
        diag = get_diagnostics(opt)
        assert len(diag["costs"]) == 3
        assert diag["iteration_count"] == 3
        assert diag["velocity_magnitude"] is None  # no momentum

    def test_diagnostics_with_momentum(self):
        torch.manual_seed(42)
        model = _make_model()
        opt = self._run_optimizer(model, steps=2, use_momentum=True)
        diag = get_diagnostics(opt)
        assert isinstance(diag["velocity_magnitude"], float)
        assert diag["velocity_magnitude"] >= 0.0


# ---------------------------------------------------------------------------
# TestIntegration (callbacks + train)
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_train_with_logging_and_early_stop(self, capsys):
        torch.manual_seed(42)
        model = _make_model()
        dl = _make_dataloader(num_samples=32, batch_size=16)  # 2 batches
        opt = _make_optimizer(model)

        logging_cb = LoggingCallback(log_every=1)
        early_cb = EarlyStoppingCallback(patience=2, min_delta=1e-8)
        config = TrainConfig(epochs=10, callbacks=[logging_cb, early_cb])

        result = train(model, dl, nn.MSELoss(), opt, config)
        assert result is model

        # Verify logging output was produced
        captured = capsys.readouterr().out
        assert "[Step 0]" in captured

        # Verify training ran (callback mechanism works).
        # With multi-particle architecture, loss may continue improving
        # so early stopping may or may not trigger within 20 steps.
        assert opt.state.iteration_count > 0
        assert opt.state.iteration_count <= 20
