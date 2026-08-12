"""End-to-end MNIST training tests.

Real-dataset validation for every subspace mode. These are ``slow``-marked because they
download MNIST and train for multiple epochs; the fast, network-free learning check that
runs in CI is ``tests/test_end_to_end_learning.py``.

MNIST loaders live in ``conftest.py`` as the ``mnist_loaders`` factory fixture.

Model size: the optimizer reshapes parameters to ``(num_particles, particle_dim)``,
so each OT problem has ``2 * particle_dim`` vertices per particle and costs ``P * V * K``
model evaluations per step. Downsampling the images to 7x7 keeps that tractable on CPU.
"""

from __future__ import annotations

import pytest
from collections import OrderedDict

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer, TrainCallback, TrainConfig, train
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.epsilon import LinearEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


class SmallMNISTNet(nn.Sequential):
    """7x7 input, 16 hidden: 49*16+16 + 16*10+10 = 970 params.

    An ``nn.Sequential`` subclass, not a plain ``nn.Module``: every batched
    evaluator checks ``type(model).forward is nn.Sequential.forward`` before it will
    build a plan, so an identical hand-written ``forward`` silently opts the model out
    of the bmm and subspace-delta paths. The ``OrderedDict`` keeps the ``fc1``/``fc2``
    state_dict keys.
    """

    def __init__(self, input_dim: int = 49, hidden: int = 16):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(input_dim, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, 10)),
                ]
            )
        )


class SmallMLP(nn.Sequential):
    """Full 28x28 input, 64 hidden: 50890 params. Used for the subspace tests."""

    def __init__(self, hidden: int = 64):
        super().__init__(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(784, hidden)),
                    ("relu", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, 10)),
                ]
            )
        )


@torch.no_grad()
def _accuracy(model: nn.Module, loader, device=None) -> float:
    was_training = model.training
    model.eval()
    correct = total = 0
    for inputs, targets in loader:
        if device is not None:
            inputs, targets = inputs.to(device), targets.to(device)
        correct += (model(inputs).argmax(dim=-1) == targets).sum().item()
        total += targets.size(0)
    if was_training:
        model.train()
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def _mean_loss(model: nn.Module, loader) -> float:
    """Mean cross-entropy over a loader, for a fixed before/after descent baseline."""
    was_training = model.training
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = total = 0.0
    for inputs, targets in loader:
        total_loss += loss_fn(model(inputs), targets).item()
        total += targets.size(0)
    if was_training:
        model.train()
    return total_loss / total if total > 0 else float("inf")


class _EpochLoss(TrainCallback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, metrics: dict) -> None:
        self.losses.append(metrics["avg_loss"])


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_mnist_accuracy(mnist_loaders):
    """Full-space training on 2000 downsampled samples clears 50%, well above 10% chance."""
    torch.manual_seed(42)
    train_loader, test_loader = mnist_loaders(n_train=2000, n_test=1000, batch_size=32, downsample=4)

    model = SmallMNISTNet(input_dim=49, hidden=16)
    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=LinearEpsilon(init=0.1, target=0.01, decay=0.001),
        step_radius=3.0,
        probe_radius=6.0,
        num_probe=2,
        sinkhorn_max_iters=100,
        scale_cost="mean",
        chunk_size=512,
    )
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=10))

    accuracy = _accuracy(model, test_loader)
    assert accuracy > 0.50, f"expected > 50% on the 1000-sample test subset, got {accuracy * 100:.1f}%"


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("kind", ["adaptive", "hybrid"])
def test_subspace_trains(mnist_loaders, kind):
    """One training run per subspace mode, checking everything that run can show.

    Steps are taken, displacement is real, absorb fires and moves the base, the loss
    falls, and the mode clears an accuracy floor.
    """
    train_loader, test_loader = mnist_loaders(n_train=1000, n_test=500, batch_size=512)

    torch.manual_seed(42)
    # hidden=32 rather than 64: half the parameters converge faster per step in a fixed
    # subspace rank, so this reaches a higher accuracy in a third of the wall clock.
    # Do not shrink it further: at hidden=24 the adaptive floor fails at 26.2%.
    model = SmallMLP(hidden=32)
    layout = ParamLayout.from_module(model)
    if kind == "adaptive":
        subspace = AdaptiveSubspace.from_layout(
            layout, rank=128, rotation_mode="displacement", absorb_mode="periodic", absorb_interval=5
        )
    else:
        subspace = HybridSubspace.from_layout(
            layout, rank=4, rotation_interval=0, absorb_mode="periodic", absorb_interval=5
        )

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=42,
        epsilon=LinearEpsilon(init=1.0, target=0.1, decay=0.01),
        # Per mode: a single global projection wants a larger radius than a per-layer one.
        step_radius={"adaptive": 10.0, "hybrid": 4.5}[kind],
        probe_radius=2.0,
        # K=1 is the library default and evaluates 31,320 candidates here against 93,960
        # at K=3, for higher accuracy on two of the three modes.
        num_probe=1,
        sinkhorn_max_iters=50,
        subspace=subspace,
    )
    initial_base = {k: v.clone() for k, v in optimizer.state.base_params.items()}
    initial_loss = _mean_loss(model, train_loader)

    tracker = _EpochLoss()
    model = train(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        TrainConfig(epochs=4, callbacks=[tracker], restore_best=False),
    )

    assert optimizer.state.iteration_count > 0
    assert len(optimizer.state.costs) == optimizer.state.iteration_count

    finite_disps = [d for d in optimizer.state.displacement_sqnorms if d == d]
    assert sum(finite_disps) > 0, "every displacement was zero or NaN, so no step moved"

    # Both remaining classes rotate, so they steer their next basis with this history.
    assert optimizer.state.displacement_history_count > 0, "displacement history was never populated"
    assert optimizer.state.absorb_count > 0, (
        f"absorb never fired in {optimizer.state.iteration_count} steps with absorb_interval=5"
    )
    base = optimizer.state.base_params
    assert any(not torch.equal(initial_base[k], base[k]) for k in initial_base), (
        "absorb_count incremented but the base weights are unchanged"
    )

    # Against the untrained model, not against epoch 1: the first epoch's average is
    # taken over a model that is already improving, so it is a moving baseline.
    assert len(tracker.losses) >= 2
    final_loss = _mean_loss(model, train_loader)
    assert final_loss < initial_loss, (
        f"train loss did not fall: {initial_loss:.4f} -> {final_loss:.4f} (epochs: {tracker.losses})"
    )

    # 10-class MNIST, so 10% is chance. Measured: adaptive 30.0%, hybrid 60.2%.
    # Adaptive's single global projection covers less per step than the per-layer
    # basis, so it gets a lower floor. A floor at 2x chance would pass on a
    # near-dead run; these sit just under the measured values instead.
    floor = {"hybrid": 0.50, "adaptive": 0.25}[kind]
    accuracy = _accuracy(model, test_loader)
    assert accuracy >= floor, f"{kind} accuracy {accuracy * 100:.1f}% is below the {floor * 100:.0f}% floor"


@pytest.mark.gpu
def test_mnist_gpu_full_space(mnist_loaders):
    """Full-space training with compiled ops on CUDA clears 70%."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)
    train_loader, test_loader = mnist_loaders(n_train=2000, n_test=1000, batch_size=32, downsample=4)

    model = SmallMNISTNet(input_dim=49, hidden=16).to(device)
    optimizer = PolyStepOptimizer(
        model,
        compile=True,
        seed=42,
        epsilon=LinearEpsilon(init=0.1, target=0.01, decay=0.001),
        step_radius=3.0,
        probe_radius=6.0,
        num_probe=2,
        sinkhorn_max_iters=100,
        scale_cost="mean",
        chunk_size=512,
    )
    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=10))

    accuracy = _accuracy(model, test_loader, device=device)
    assert accuracy > 0.70, f"expected > 70% on 7x7 MNIST on GPU, got {accuracy * 100:.1f}%"


@pytest.mark.gpu
def test_mnist_gpu_subspace_particle_dim_8(mnist_loaders):
    """subspace_particle_dim=8 (16 orthoplex vertices) plus a periodic absorb on CUDA.

    Covers the GPU subspace pipeline: projection build, chunked probe evaluation, model
    sync, and absorb folding into the base. Accuracy is not asserted;
    ``test_mnist_gpu_full_space`` owns that.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)
    train_loader, _ = mnist_loaders(n_train=500, n_test=200, batch_size=64, downsample=4)

    model = SmallMNISTNet(input_dim=49, hidden=16).to(device)
    layout = ParamLayout.from_module(model)
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = PolyStepOptimizer(
        model,
        compile=True,
        seed=42,
        epsilon=0.1,
        step_radius=30.0,
        probe_radius=60.0,
        num_probe=1,
        sinkhorn_max_iters=100,
        subspace=HybridSubspace.from_layout(layout, rank=4, absorb_mode="periodic", absorb_interval=10),
        subspace_particle_dim=8,
        scale_cost="mean",
        chunk_size=256,
    )
    assert optimizer.state.X.shape[1] == 8, f"expected particle_dim=8, got {optimizer.state.X.shape[1]}"

    model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=2))

    displacements = optimizer.state.displacement_sqnorms
    assert displacements, "no steps were taken"
    assert any(d > 0 for d in displacements), "all displacements were zero, transport stayed uniform"

    current = model.state_dict()
    assert any(not torch.equal(initial_params[k], current[k]) for k in initial_params), (
        "model parameters did not change during training"
    )
    base = optimizer.state.base_params
    assert any(not torch.equal(initial_params[k], base[k]) for k in initial_params), (
        "base params unchanged, so the periodic absorb never triggered"
    )
