"""The optimizer actually learns: loss falls and accuracy beats chance.

Synthetic separable data, fixed seed, no network, no download, under a second per case.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer, TrainCallback, TrainConfig, train
from polystep.adaptive_subspace import AdaptiveSubspace
from polystep.factored_subspace import FactoredSubspace
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

IN_DIM, N_CLASSES, N_SAMPLES = 12, 3, 192


def _blobs():
    """Well-separated Gaussian blobs, one per class. Linearly separable by design."""
    gen = torch.Generator().manual_seed(0)
    centers = torch.eye(N_CLASSES, IN_DIM) * 6.0
    y = torch.arange(N_SAMPLES) % N_CLASSES
    X = centers[y] + torch.randn(N_SAMPLES, IN_DIM, generator=gen) * 0.5
    return X, y


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(IN_DIM, 16), nn.ReLU(), nn.Linear(16, N_CLASSES))


def _accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def _subspace(kind, model):
    if kind is None:
        return None
    layout = ParamLayout.from_module(model)
    if kind == "adaptive":
        return AdaptiveSubspace.auto_from_params(model, compression_target=0.5)
    if kind == "hybrid":
        return HybridSubspace.from_layout(layout, rank=4)
    return FactoredSubspace.from_layout(layout, rank=4)


@pytest.mark.parametrize("kind", [None, "adaptive", "hybrid", "factored"])
def test_optimizer_learns_a_separable_classification_task(kind):
    """Loss falls and accuracy beats chance, through every subspace path."""
    X, y = _blobs()
    model = _model()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=False)

    start_acc = _accuracy(model, X, y)
    opt = PolyStepOptimizer(
        model,
        subspace=_subspace(kind, model),
        epsilon=0.5,
        step_radius=0.5,
        num_probe=2,
        compile=False,
        seed=0,
    )
    losses = []
    train(
        model,
        loader,
        nn.CrossEntropyLoss(),
        opt,
        TrainConfig(epochs=12, callbacks=[_Recorder(losses)]),
    )

    assert losses, "training produced no loss records"
    assert min(losses) < losses[0] * 0.75, f"loss barely moved: {losses[0]:.4f} -> {min(losses):.4f}"

    final_acc = _accuracy(model, X, y)
    chance = 1.0 / N_CLASSES
    assert final_acc > chance + 0.2, f"accuracy {final_acc:.3f} is not meaningfully above chance {chance:.3f}"
    assert final_acc > start_acc, f"accuracy did not improve: {start_acc:.3f} -> {final_acc:.3f}"


def test_a_zero_step_radius_does_not_learn():
    """Guards the test above: with no movement allowed, its thresholds must not be met.

    The weights still drift by ~1e-7 at ``step_radius=0``: reconstructing
    ``base + P @ coords`` and pushing it back through ``load_state_dict`` is a
    round-trip through fp32, so the bound is round-off scale, not exact equality.
    """
    X, y = _blobs()
    model = _model()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=False)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses = []
    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=0.0, num_probe=2, compile=False, seed=0)
    train(model, loader, nn.CrossEntropyLoss(), opt, TrainConfig(epochs=12, callbacks=[_Recorder(losses)]))

    after = model.state_dict()
    drift = max((after[k] - before[k]).abs().max().item() for k in before)
    assert drift < 1e-5, f"step_radius=0 moved the weights by {drift:.3e}, beyond fp32 round-off"
    assert min(losses) >= losses[0] * 0.75, f"loss fell without any step being taken: {losses[0]} -> {min(losses)}"


class _Recorder(TrainCallback):
    """Records the per-step loss. Also makes ``train`` compute it, rather than the
    cheaper OT-cost proxy that ``restore_best`` uses."""

    def __init__(self, sink):
        self.sink = sink

    def on_step_end(self, metrics: dict) -> bool:
        self.sink.append(float(metrics["loss"]))
        return False


@pytest.mark.slow
def test_orthoplex_with_the_quadratic_model_beats_the_simplex_per_forward_pass():
    """Locks the docs/performance.md recommendation to a measurement.

    The orthoplex costs 2k vertices against the simplex's k+1, so it pays off only
    with the finite-difference machinery its antithetic pairing enables. The budget
    is candidate evaluations, not steps, or the orthoplex just gets more forwards.
    A thin margin would flip on any BLAS blocking difference, hence the 1.25 bar.
    """
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    budget = 120_000

    def run(**kwargs):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 3))
        g = torch.Generator().manual_seed(100)
        x = torch.randn(256, 20, generator=g)
        y = (x @ torch.randn(20, 3, generator=g)).argmax(dim=1)
        loss_fn = nn.CrossEntropyLoss()
        ev = NNCostEvaluator(model, loss_fn, ParamLayout.from_module(model))
        opt = PolyStepOptimizer(model, epsilon=0.1, step_radius=0.1, seed=0, compile=False, **kwargs)
        opt.register_evaluator(ev, x, y)

        with torch.no_grad():
            start = float(loss_fn(model(x), y))
        while sum(opt.state.evals) < budget:
            opt.step(lambda bp: ev.evaluate(bp, x, y))
        with torch.no_grad():
            return start - float(loss_fn(model(x), y))

    simplex = run(polytope_type="simplex")
    orthoplex_quad = run(polytope_type="orthoplex", use_quadratic_model=True, trust_region=True, num_probe=2)

    assert orthoplex_quad > simplex * 1.25, (
        f"orthoplex+quadratic model does not pay for its extra vertices: {orthoplex_quad:.4f} vs simplex {simplex:.4f}"
    )
