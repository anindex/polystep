"""run_timeseries must select on validation and score test exactly once.

ETTh1 is a regression benchmark, so ES/SPSA reach it via their ``eval_fn`` rather
than the classification baselines' ``val_loader``; these checks guard that plumbing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rt(require_experiments):
    sys.path.insert(0, str(REPO_ROOT))
    import experiments.runners.run_timeseries as module

    return module


def test_dispatch_gives_every_method_the_val_split_and_the_guard(rt, monkeypatch, tmp_path):
    """audit_no_leakage and the val split must reach every method, not only polystep."""
    series = np.zeros(300, dtype=np.float32)
    monkeypatch.setattr(rt, "load_etth1", lambda: (series, series + 1, series + 2, {}))

    seen = {}
    for method in rt.METHOD_RUNNERS:

        def recorder(seed, device, train, val, test, results_dir, _m=method, **kwargs):
            seen[_m] = (val, kwargs)

        monkeypatch.setitem(rt.METHOD_RUNNERS, method, recorder)
        rt.run_method(method, 42, "cpu", str(tmp_path), audit_no_leakage=True)

    assert set(seen) == set(rt.METHOD_RUNNERS), "a method was never dispatched"
    for method, (val, kwargs) in seen.items():
        assert val is not None and val[0] == 1.0, f"{method} was dispatched without the validation split"
        assert kwargs.get("audit_no_leakage") is True, f"{method} was dispatched without the leakage guard"


class _ConstantForecaster(nn.Module):
    """Predicts the same learnable constant for every horizon step."""

    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.c.expand(x.shape[0], 96)


def test_selector_picks_the_best_val_checkpoint_not_the_best_test_one(rt):
    """A checkpoint worse on val must not be reported, however good it looks on test.

    val is the all-zero series and test the all-ten series, so the constant that
    fits val best fits test worst.
    """
    val = np.zeros(200, dtype=np.float32)
    test = np.full(200, 10.0, dtype=np.float32)

    model = _ConstantForecaster()
    sel = rt.ValSelected(model, val, test, "cpu")
    for c in (1.0, 5.0, 0.5, 9.0):  # best on val is 0.5, best on test is 9.0
        with torch.no_grad():
            model.c.fill_(c)
        sel()

    assert sel.best_mse == pytest.approx(0.25), "selection did not follow validation MSE"
    metrics = sel.metrics(wall_time_seconds=0.0, peak_gpu_memory_mb=0.0, function_evals=0, total_steps=0)

    # Reported keys all carry test at the selected checkpoint, i.e. (10 - 0.5)**2.
    assert metrics["best_mse"] == metrics["final_mse"] == metrics["test_mse_at_selected"]
    assert metrics["test_mse_at_selected"] == pytest.approx(90.25)
    assert metrics["val_mse_at_selected"] == pytest.approx(0.25)
    # No accuracy is invented for a regression benchmark.
    assert all(metrics[k] != metrics[k] for k in ("final_accuracy", "best_accuracy", "test_accuracy_at_selected"))
