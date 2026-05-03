"""Integration tests for run_experiment and FunctionEvalCounter.

Tests use tiny models (nn.Linear(4, 2)) and tiny synthetic data (20 samples)
to keep execution fast. Results are saved to tmpdir to avoid polluting
paper/results/.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.runners.common import FunctionEvalCounter, run_experiment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_data():
    """Create tiny synthetic data: 20 samples, 4 features, 2 classes."""
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    y = torch.randint(0, 2, (20,))
    ds = TensorDataset(X, y)
    train_loader = DataLoader(ds, batch_size=20, shuffle=False)
    test_loader = DataLoader(ds, batch_size=20, shuffle=False)
    return train_loader, test_loader


@pytest.fixture
def tiny_model_fn():
    """Factory returning a fresh tiny model."""
    def _make():
        return nn.Linear(4, 2)
    return _make


# ---------------------------------------------------------------------------
# FunctionEvalCounter tests
# ---------------------------------------------------------------------------

class TestFunctionEvalCounter:
    def test_counts_forward_passes(self):
        """FunctionEvalCounter increments count by 1 per call (not per sample)."""
        counter = FunctionEvalCounter(nn.CrossEntropyLoss())
        outputs = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        for _ in range(5):
            counter(outputs, targets)
        assert counter.count == 5

    def test_reset(self):
        """FunctionEvalCounter.reset() sets count back to 0."""
        counter = FunctionEvalCounter(nn.CrossEntropyLoss())
        outputs = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        counter(outputs, targets)
        counter(outputs, targets)
        assert counter.count == 2
        counter.reset()
        assert counter.count == 0

    def test_passes_through_loss_values(self):
        """FunctionEvalCounter returns the same loss value as the wrapped loss."""
        loss_fn = nn.CrossEntropyLoss()
        counter = FunctionEvalCounter(nn.CrossEntropyLoss())

        torch.manual_seed(42)
        outputs = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        expected = loss_fn(outputs, targets)
        actual = counter(outputs, targets)
        assert torch.allclose(expected, actual)


# ---------------------------------------------------------------------------
# run_experiment tests
# ---------------------------------------------------------------------------

class TestRunExperiment:
    def test_adam_produces_json_with_required_keys(self, tiny_data, tiny_model_fn, tmp_path):
        """run_experiment with method='adam' produces JSON with all 6 required metric keys."""
        train_loader, test_loader = tiny_data
        result_paths = run_experiment(
            model_fn=tiny_model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            method='adam',
            benchmark='test_bench',
            seeds=[42],
            device='cpu',
            epochs=2,
            results_dir=str(tmp_path),
        )

        assert len(result_paths) == 1
        with open(result_paths[0]) as f:
            data = json.load(f)

        required_keys = {
            "final_accuracy", "best_accuracy", "wall_time_seconds",
            "peak_gpu_memory_mb", "function_evals", "total_steps",
        }
        assert required_keys.issubset(set(data["metrics"].keys()))

    def test_adam_json_has_positive_metrics(self, tiny_data, tiny_model_fn, tmp_path):
        """run_experiment JSON has function_evals > 0 and wall_time_seconds > 0."""
        train_loader, test_loader = tiny_data
        result_paths = run_experiment(
            model_fn=tiny_model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            method='adam',
            benchmark='test_bench',
            seeds=[42],
            device='cpu',
            epochs=2,
            results_dir=str(tmp_path),
        )

        with open(result_paths[0]) as f:
            data = json.load(f)

        assert data["metrics"]["function_evals"] > 0
        assert data["metrics"]["wall_time_seconds"] > 0

    def test_adam_filename_convention(self, tiny_data, tiny_model_fn, tmp_path):
        """run_experiment with seeds=[42] produces exactly 1 file named {benchmark}_adam_42.json."""
        train_loader, test_loader = tiny_data
        result_paths = run_experiment(
            model_fn=tiny_model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            method='adam',
            benchmark='mybench',
            seeds=[42],
            device='cpu',
            epochs=1,
            results_dir=str(tmp_path),
        )

        assert len(result_paths) == 1
        assert os.path.basename(result_paths[0]) == "mybench_adam_42.json"

    def test_custom_loss_fn(self, tiny_data, tiny_model_fn, tmp_path):
        """run_experiment with custom loss_fn passes it to the training loop."""
        train_loader, test_loader = tiny_data

        # Use MSELoss with float targets as a "custom" loss
        # We need a wrapper model that produces 1 output for MSE
        def model_fn():
            return nn.Linear(4, 2)

        # Custom loss that records it was called
        class TrackingLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False
                self._ce = nn.CrossEntropyLoss()
            def forward(self, outputs, targets):
                self.called = True
                return self._ce(outputs, targets)

        custom_loss = TrackingLoss()

        result_paths = run_experiment(
            model_fn=model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            method='adam',
            benchmark='custom_loss_test',
            seeds=[42],
            device='cpu',
            epochs=1,
            results_dir=str(tmp_path),
            loss_fn=custom_loss,
        )

        # The custom loss should have been used (called at least once)
        assert custom_loss.called
        assert len(result_paths) == 1

    def test_polystep_smoke(self, tiny_data, tiny_model_fn, tmp_path):
        """run_experiment with method='polystep' completes without error on tiny model."""
        train_loader, test_loader = tiny_data
        result_paths = run_experiment(
            model_fn=tiny_model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            method='polystep',
            benchmark='smoke_test',
            seeds=[42],
            device='cpu',
            epochs=1,
            results_dir=str(tmp_path),
        )

        assert len(result_paths) == 1
        with open(result_paths[0]) as f:
            data = json.load(f)
        assert data["metrics"]["wall_time_seconds"] > 0

    def test_multiple_seeds(self, tiny_data, tiny_model_fn, tmp_path):
        """run_experiment with multiple seeds produces one JSON per seed."""
        train_loader, test_loader = tiny_data
        result_paths = run_experiment(
            model_fn=tiny_model_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            method='adam',
            benchmark='multi_seed',
            seeds=[42, 123],
            device='cpu',
            epochs=1,
            results_dir=str(tmp_path),
        )

        assert len(result_paths) == 2
        basenames = sorted(os.path.basename(p) for p in result_paths)
        assert basenames == ["multi_seed_adam_123.json", "multi_seed_adam_42.json"]
