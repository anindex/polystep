"""Shared test fixtures and pytest configuration for polystep tests."""

import gzip
import os
import struct as pystruct
import sysconfig
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
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
        os.environ["CPLUS_INCLUDE_PATH"] = f"{include_dir}:{previous}" if previous else include_dir
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CPLUS_INCLUDE_PATH", None)
        else:
            os.environ["CPLUS_INCLUDE_PATH"] = previous


_DEFAULT_TORCH_THREADS = torch.get_num_threads()
_FAST_TEST_THREADS = int(os.environ.get("POLYSTEP_TEST_THREADS", "1"))


def pytest_configure(config):
    """Register custom markers and pin torch to one intra-op thread.

    The fast suite's tensors are small enough that the intra-op pool's fork/join costs
    more than the arithmetic, and under ``-n auto`` it also stops each worker grabbing
    every core. ``POLYSTEP_TEST_THREADS`` overrides the count; ``slow`` tests get the
    machine default back from the ``_torch_threads`` fixture. Numbers in CONTRIBUTING.md.
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring CUDA GPU")
    torch.set_num_threads(_FAST_TEST_THREADS)


@pytest.fixture(autouse=True)
def _torch_threads(request):
    """Give ``slow`` tests the machine's default thread count, pin everything else.

    Switching the count back and forth does not leave the pool degraded.
    """
    if "slow" not in request.keywords:
        yield
        return
    torch.set_num_threads(_DEFAULT_TORCH_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(_FAST_TEST_THREADS)


@pytest.fixture
def require_experiments():
    """Skip the requesting test when the reproduction harness is absent.

    ``experiments/`` is the paper reproduction harness and is deliberately not part of
    the distribution, so tests that read or import it cannot run from an sdist.
    """
    import pathlib

    if not (pathlib.Path(__file__).resolve().parent.parent / "experiments" / "runners").is_dir():
        pytest.skip("experiments/runners not present (running outside the repo)")


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


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_DIR = "/tmp/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        _magic, num, rows, cols = pystruct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        _magic, _num = pystruct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


@pytest.fixture(scope="session")
def mnist_arrays():
    """Download MNIST once per session and return normalized ``(train_x, train_y, test_x, test_y)``.

    Skips only on a download failure. A gzip or IDX parse error is a real bug and is
    allowed to propagate; catching everything here used to disable a whole file silently.
    """
    os.makedirs(MNIST_DIR, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = os.path.join(MNIST_DIR, filename)
        if not os.path.exists(path):
            try:
                urlretrieve(MNIST_URL + filename, path)
            except (URLError, OSError) as exc:
                pytest.skip(f"MNIST download failed ({exc}); needs network access")

    mean, std = 0.1307, 0.3081
    train_x = (_read_idx_images(os.path.join(MNIST_DIR, MNIST_FILES["train_images"])) - mean) / std
    test_x = (_read_idx_images(os.path.join(MNIST_DIR, MNIST_FILES["test_images"])) - mean) / std
    train_y = _read_idx_labels(os.path.join(MNIST_DIR, MNIST_FILES["train_labels"]))
    test_y = _read_idx_labels(os.path.join(MNIST_DIR, MNIST_FILES["test_labels"]))
    return train_x, train_y, test_x, test_y


@pytest.fixture(scope="session")
def mnist_loaders(mnist_arrays):
    """Factory: ``mnist_loaders(n_train, n_test, batch_size, downsample=1)``.

    ``downsample`` average-pools the images (2 -> 14x14, 4 -> 7x7) to shrink the input
    dimension, which is what keeps the full-space OT problem tractable on CPU.
    """
    train_x, train_y, test_x, test_y = mnist_arrays

    def _build(n_train, n_test, batch_size, downsample=1):
        tr_x = torch.from_numpy(train_x[:n_train].copy())
        te_x = torch.from_numpy(test_x[:n_test].copy())
        if downsample > 1:
            tr_x = nn.functional.avg_pool2d(tr_x, downsample)
            te_x = nn.functional.avg_pool2d(te_x, downsample)
        train_ds = TensorDataset(tr_x, torch.from_numpy(train_y[:n_train].copy()))
        test_ds = TensorDataset(te_x, torch.from_numpy(test_y[:n_test].copy()))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        )

    return _build


@pytest.fixture
def regression_closure():
    """Factory for a closure whose cost depends on the parameters.

    Returns a callable with two extras attached: ``initial_loss`` (the single-model loss
    at construction) and ``true_loss()`` (the current single-model loss). Tests that
    assert descent need both; a closure returning ``torch.rand`` drives the cost matrix,
    the plan and the barycentric step with noise, so nothing downstream is under test.
    """

    def _build(model, num_samples=32, seed=0):
        gen = torch.Generator().manual_seed(seed)
        first = next(m for m in model.modules() if isinstance(m, nn.Linear))
        last = next(m for m in reversed(list(model.modules())) if isinstance(m, nn.Linear))
        inputs = torch.randn(num_samples, first.in_features, generator=gen)
        targets = torch.randn(num_samples, last.out_features, generator=gen)
        loss_fn = nn.MSELoss()
        evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

        def true_loss():
            with torch.no_grad():
                return loss_fn(model(inputs), targets).item()

        def closure(batched_params):
            return evaluator.evaluate(batched_params, inputs, targets)

        closure.true_loss = true_loss
        closure.initial_loss = true_loss()
        return closure

    return _build
