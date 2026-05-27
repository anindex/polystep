"""Shared test fixtures and pytest configuration for polystep tests."""

import os
import sysconfig

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep.cost_nn import NNCostEvaluator


def _python_include_dir() -> str:
    """Return the directory containing ``Python.h`` for the active interpreter.

    ``torch.compile``'s C++ backend needs ``Python.h``, which may not
    be in the standard ``/usr/include/pythonX.Y`` when using venvs or
    conda environments.
    """
    include_dir = sysconfig.get_path("include")
    if os.path.isfile(os.path.join(include_dir, "Python.h")):
        return include_dir
    candidate = os.path.join(
        sysconfig.get_config_var("prefix") or "",
        "include",
        f"python{sysconfig.get_python_version()}",
    )
    if os.path.isfile(os.path.join(candidate, "Python.h")):
        return candidate
    return include_dir


@pytest.fixture(scope="session", autouse=True)
def _cplus_include_path():
    """Prepend the Python include directory to ``CPLUS_INCLUDE_PATH``.

    Scoped to the session and unwound at teardown so the mutation does
    not leak into the parent shell or sibling pytest processes.
    """
    include_dir = _python_include_dir()
    previous = os.environ.get("CPLUS_INCLUDE_PATH")
    if include_dir not in (previous or ""):
        os.environ["CPLUS_INCLUDE_PATH"] = (
            f"{include_dir}:{previous}" if previous else include_dir
        )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CPLUS_INCLUDE_PATH", None)
        else:
            os.environ["CPLUS_INCLUDE_PATH"] = previous


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring CUDA GPU")


@pytest.fixture
def cost_grid():
    """Yield a (cost, eps) grid for solver overflow / stability stress tests.

    Cost ranges {1, 10, 100, 1000} crossed with eps {0.01, 0.1, 1, 10} give
    16 cells covering small-eps explosion and large-eps near-uniform regimes.
    """
    cost_ranges = (1.0, 10.0, 100.0, 1000.0)
    eps_values = (0.01, 0.1, 1.0, 10.0)
    return [(c, e) for c in cost_ranges for e in eps_values]


@pytest.fixture
def simple_mlp():
    """Small MLP for fast testing: Linear(4,8) -> ReLU -> Linear(8,2)."""
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


@pytest.fixture
def simple_dataloader():
    """DataLoader with 32 random samples, batch_size=16."""
    torch.manual_seed(42)
    X = torch.randn(32, 4)
    y = torch.randn(32, 2)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def make_closure():
    """Factory fixture that creates an NNCostEvaluator closure for a model.

    Usage::

        def test_example(simple_mlp, make_closure):
            closure = make_closure(simple_mlp)
            # closure(batched_params) -> losses
    """
    def _make_closure(model, loss_fn=None, num_samples=16, input_dim=4, output_dim=None):
        torch.manual_seed(42)
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        # Infer output dim from last linear layer
        if output_dim is None:
            for m in reversed(list(model.modules())):
                if isinstance(m, nn.Linear):
                    output_dim = m.out_features
                    break
            else:
                output_dim = 1
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
        inputs = torch.randn(num_samples, input_dim)
        targets = torch.randn(num_samples, output_dim)

        def closure(batched_params):
            return evaluator.evaluate(batched_params, inputs, targets)

        return closure

    return _make_closure
