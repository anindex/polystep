"""Two identically seeded runs must produce the same loss trajectory.

Seeding alone leaves cuDNN autotuning, TF32 and the DataLoader shuffle order
free, so ``set_seed`` also pins deterministic kernels and every loader carries
a seeded generator.
"""

import pytest
import torch
import torch.nn as nn


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


@pytest.fixture
def train_loader():
    from torch.utils.data import DataLoader, TensorDataset
    from polystep.benchmarks.utils import seeded_loader_kwargs

    gen = torch.Generator().manual_seed(0)
    data = TensorDataset(torch.randn(128, 1, 7, 7, generator=gen), torch.randint(10, (128,), generator=gen))
    return DataLoader(data, batch_size=32, shuffle=True, **seeded_loader_kwargs(42))


def _train_once(train_loader, seed=42, epochs=2, device="cpu"):
    """Train a tiny MLP with PolyStep and return the per-step losses."""
    from experiments.runners.common import reseed_loaders, set_seed
    from polystep.cost_nn import NNCostEvaluator
    from polystep.optimizer import PolyStepOptimizer

    set_seed(seed)
    torch.set_num_threads(1)
    # A loader's own generator advances per epoch and ignores the global RNG, so
    # the second run must rewind it to see the same stream.
    reseed_loaders(seed, train_loader)
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


def test_seeded_runs_repeat_loss_trajectory(require_experiments, train_loader):

    first = _train_once(train_loader)
    second = _train_once(train_loader)

    assert len(first) == 8, "expected 4 batches x 2 epochs"
    assert first == second, f"seed 42 drifted between runs:\n  {first}\n  {second}"
    # A trajectory that never moves would match trivially.
    assert len(set(first)) > 1, "loss never changed; the comparison proves nothing"


def test_set_deterministic_pins_tf32_and_cudnn(require_experiments):
    from experiments.runners.common import set_deterministic

    set_deterministic()
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_runner_import_sets_cublas_workspace(require_experiments):
    """cuBLAS reads CUBLAS_WORKSPACE_CONFIG when the context is created,
    so importing the runner utilities has to be enough to set it."""
    import os

    import experiments.runners.common  # noqa: F401

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_loaders_carry_seeded_generator():
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
def test_seeded_cuda_runs_repeat_loss_trajectory(require_experiments, train_loader):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    assert _train_once(train_loader, device="cuda") == _train_once(train_loader, device="cuda")
