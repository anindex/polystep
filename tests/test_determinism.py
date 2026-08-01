"""Two identically seeded runs must produce the same loss trajectory.

Guards the run-to-run drift recorded in CHANGELOG.md: seeding alone left
cuDNN autotuning, TF32 and the DataLoader shuffle order free, so repeats
of the same seed diverged. ``set_seed`` now calls ``set_deterministic``,
and every loader carries a seeded generator.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _restore_global_determinism_flags():
    """``use_deterministic_algorithms`` is process-global; put it back.

    Without this, whichever xdist worker picks up this file leaves every
    later test in it running under different kernel selection rules.

    The thread count is restored by conftest's ``_torch_threads``, which every
    test routes through; these tests call ``set_seed`` and so widen the pool.
    """
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        yield
    finally:
        (
            deterministic,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        ) = saved
        torch.use_deterministic_algorithms(deterministic)


def _train_once(train_loader, seed=42, epochs=2, device="cpu"):
    """Train a tiny MLP with PolyStep and return the per-step losses."""
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.common import set_seed
    from polystep.cost_nn import NNCostEvaluator
    from polystep.optimizer import PolyStepOptimizer

    set_seed(seed)
    # Built after seeding: identical initial weights are half of the claim.
    model = nn.Sequential(nn.Flatten(), nn.Linear(49, 16), nn.ReLU(), nn.Linear(16, 10)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = PolyStepOptimizer(model, seed=seed, epsilon=0.5, step_radius=1.0, probe_radius=0.5, num_probe=2)
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    losses = []
    for _ in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            def closure(batched_params, _d=data, _t=targets):
                return evaluator.evaluate(batched_params, _d, _t)

            optimizer.step(closure)
            with torch.no_grad():
                losses.append(loss_fn(model(data), targets).item())
    return losses


def test_two_seeded_runs_have_identical_loss_trajectories(require_experiments, mnist_loaders):
    train_loader, _ = mnist_loaders(n_train=128, n_test=32, batch_size=32, downsample=4)

    first = _train_once(train_loader)
    second = _train_once(train_loader)

    assert len(first) == 8, "expected 4 batches x 2 epochs"
    assert first == second, f"seed 42 drifted between runs:\n  {first}\n  {second}"
    # A trajectory that never moves would match trivially.
    assert len(set(first)) > 1, "loss never changed; the comparison proves nothing"


def test_set_deterministic_pins_tf32_and_cudnn(require_experiments):
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.common import set_deterministic

    set_deterministic()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_cublas_workspace_is_configured_before_cuda(require_experiments):
    """cuBLAS reads CUBLAS_WORKSPACE_CONFIG when the context is created,
    so importing the runner utilities has to be enough to set it."""
    sys.path.insert(0, str(REPO_ROOT))
    import os

    import experiments.runners.common  # noqa: F401

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_loaders_carry_a_seeded_generator():
    from polystep.benchmarks.utils import seeded_loader_kwargs

    torch.manual_seed(7)
    a = seeded_loader_kwargs()
    torch.manual_seed(7)
    b = seeded_loader_kwargs()
    assert a["generator"].initial_seed() == b["generator"].initial_seed()
    assert a["worker_init_fn"] is not None

    torch.manual_seed(8)
    c = seeded_loader_kwargs()
    assert c["generator"].initial_seed() != a["generator"].initial_seed(), (
        "loader order must still vary with the experiment seed"
    )


@pytest.mark.gpu
def test_two_seeded_cuda_runs_have_identical_loss_trajectories(require_experiments, mnist_loaders):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    train_loader, _ = mnist_loaders(n_train=128, n_test=32, batch_size=32, downsample=4)
    assert _train_once(train_loader, device="cuda") == _train_once(train_loader, device="cuda")
