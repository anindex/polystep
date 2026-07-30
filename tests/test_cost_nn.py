"""Tests for NNCostEvaluator, ParamLayout.batch_unflatten(), chunked eval, and cost matrix."""

import warnings

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from polystep import PolyStepOptimizer
from polystep.transform import ParamLayout
from polystep.cost_nn import (
    BatchedLinearEvaluator,
    NNCostEvaluator,
    auto_detect_chunk_size,
)


def test_compile_vmap_tracks_changing_batch():
    """compile_vmap must evaluate each call against its own batch, not bake in
    the first call's inputs/targets into the cached compiled graph."""
    torch.manual_seed(0)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x)

    model = TinyNet()
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_vmap=True)
    ref = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_vmap=False)
    assert ev._batched_linear is None  # custom forward routes through vmap

    N = 2
    base = dict(model.named_parameters())
    stacked = {k: v.detach()[None].expand(N, *v.shape).contiguous() for k, v in base.items()}
    xa, ta = torch.randn(5, 4), torch.randint(0, 3, (5,))
    xb, tb = torch.randn(7, 4), torch.randint(0, 3, (7,))

    la = ev.evaluate(stacked, xa, ta)
    lb = ev.evaluate(stacked, xb, tb)
    torch.testing.assert_close(la, ref.evaluate(stacked, xa, ta), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lb, ref.evaluate(stacked, xb, tb), rtol=1e-4, atol=1e-4)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim=10, hidden=5, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SharedWeightsModel(nn.Module):
    """Model where fc2.weight is tied to fc1.weight."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc2.weight = self.fc1.weight

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MLPWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.bn(self.fc1(x))))


class VmapIncompatibleModel(nn.Module):
    """Model that calls .item() in forward: incompatible with vmap."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self._scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # .item() is not traceable by vmap
        s = self._scale.item()
        return self.fc(x) * s


def test_batch_unflatten_shape():
    """batch_unflatten returns correct shapes."""
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 8
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    stacked = layout.batch_unflatten(batch)

    sd = model.state_dict()
    for key in sd:
        assert key in stacked, f"Missing key {key}"
        expected_shape = (N, *sd[key].shape)
        assert stacked[key].shape == expected_shape, f"{key}: expected {expected_shape}, got {stacked[key].shape}"


def test_batch_unflatten_values():
    """batch_unflatten agrees with per-particle unflatten."""
    model = SimpleMLP()
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 4
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    stacked = layout.batch_unflatten(batch)

    for i in range(N):
        single = layout.unflatten(batch[i])
        for key in single:
            torch.testing.assert_close(
                stacked[key][i],
                single[key],
                msg=lambda m: f"Mismatch at particle {i}, key {key}: {m}",
            )


class TestBatchUnflattenSharedParams:
    """Shared params produce aliased tensors."""

    def test_batch_unflatten_shared_params(self):
        model = SharedWeightsModel()
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 3
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        stacked = layout.batch_unflatten(batch)

        # The tied weight appears once, under the canonical key. functional_call
        # rejects a dict naming both, and the in-place swap writes the single
        # Parameter that fc1 and fc2 share.
        assert "fc1.weight" in stacked
        assert "fc2.weight" not in stacked
        # unflatten still carries the alias: it feeds load_state_dict.
        assert "fc2.weight" in layout.unflatten(particle)

    def test_a_tied_weight_evaluates_on_both_paths(self):
        from polystep.cost_nn import NNCostEvaluator

        torch.manual_seed(0)
        model = SharedWeightsModel()
        layout = ParamLayout.from_module(model)
        inputs, targets = torch.randn(6, 10), torch.randn(6, 10)

        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        batch = layout.flatten(model).unsqueeze(0).expand(3, -1, -1).clone()
        batch[1] += 0.05
        batch[2] -= 0.05
        stacked = layout.batch_unflatten(batch)

        evaluator._use_inplace = True
        inplace = evaluator.evaluate(stacked, inputs, targets)
        evaluator._use_inplace = False
        vmapped = evaluator.evaluate(stacked, inputs, targets)

        torch.testing.assert_close(inplace, vmapped)
        # The tie must survive: an in-place swap that wrote one alias and not the
        # other would leave the two layers holding different weights.
        assert model.fc2.weight is model.fc1.weight


def test_evaluator_unsupervised():
    """Unsupervised loss (targets=None) works."""
    from polystep.cost_nn import NNCostEvaluator

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 8
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, loss_fn=lambda output: output.pow(2).mean())
    inputs = torch.randn(16, 4)
    losses = evaluator.evaluate(stacked, inputs, targets=None)

    assert losses.shape == (N,)
    assert losses.isfinite().all()


def test_evaluator_different_params_different_losses():
    """Different params produce different losses."""
    from polystep.cost_nn import NNCostEvaluator

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 10
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    batch += torch.randn_like(batch) * 1.0  # large perturbation
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    inputs = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32,))
    losses = evaluator.evaluate(stacked, inputs, targets)

    # Not all losses should be identical
    assert not torch.all(losses == losses[0]), "All losses identical despite different params"


def test_evaluator_batchnorm():
    """Evaluator handles BatchNorm (frozen buffers)."""
    from polystep.cost_nn import NNCostEvaluator

    model = MLPWithBatchNorm()
    # Run a forward pass in train mode to populate running stats
    model.train()
    with torch.no_grad():
        model(torch.randn(32, 4))

    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 8
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    batch += torch.randn_like(batch) * 0.01
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16,))
    losses = evaluator.evaluate(stacked, inputs, targets)

    assert losses.shape == (N,)
    assert losses.isfinite().all(), "Non-finite losses with BatchNorm"


def test_evaluator_fallback_warning():
    """Fallback to loop with warning for vmap-incompatible model."""
    from polystep.cost_nn import NNCostEvaluator

    model = VmapIncompatibleModel()
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 4
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        losses = evaluator.evaluate(stacked, inputs, targets)

    # Check warning was emitted
    fallback_warnings = [x for x in w if "Falling back" in str(x.message)]
    assert len(fallback_warnings) > 0, "Expected 'Falling back' warning"

    # Results should still be valid
    assert losses.shape == (N,)
    assert losses.isfinite().all()


def test_evaluator_no_grad():
    """Evaluation does not attach gradients to model params."""
    from polystep.cost_nn import NNCostEvaluator

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 5
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    evaluator.evaluate(stacked, inputs, targets)

    for name, param in model.named_parameters():
        assert param.grad is None, f"{name} has gradient attached"


class TestBatchedLinearRespectsCrossEntropyConfig:
    """Configured CrossEntropyLoss must not take the bmm fast path (it hardcodes
    mean reduction and no class weights)."""

    def test_vanilla_ce_uses_fast_path(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        assert evaluator._batched_linear is not None

    def test_use_inplace_wins_over_the_bmm_path(self):
        """``use_inplace`` is a memory contract and must outrank the bmm fast path.

        The bmm branch returned first, so an MLP that asked for O(1) activation memory
        silently got the O(N x activation) stack instead. Auto-detection was preempted
        the same way, which is the >500K-param GPU regime where bmm is what OOMs.
        """
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        layout = ParamLayout.from_module(model)
        x = torch.randn(5, 4)
        y = torch.randint(0, 3, (5,))
        batch = layout.flatten(model).unsqueeze(0).expand(3, -1, -1).clone()
        batch = batch + 0.01 * torch.randn_like(batch)
        params = layout.batch_unflatten(batch)

        reference = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False).evaluate(params, x, y)

        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
        assert evaluator._batched_linear is not None, "the bmm path must be available, or this proves nothing"
        evaluator._batched_linear.evaluate = lambda *a, **k: pytest.fail("bmm path preempted use_inplace")
        torch.testing.assert_close(evaluator.evaluate(params, x, y), reference)

    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.L1Loss()])
    def test_regression_losses_use_fast_path_and_match_vmap(self, loss_fn):
        """The bmm forward is loss-independent; only the reduction differs.

        Gating the fast path on CrossEntropyLoss sent every regression MLP through
        vmap for no reason.
        """
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        layout = ParamLayout.from_module(model)

        N = 6
        batch = layout.flatten(model).unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.05
        stacked = layout.batch_unflatten(batch)
        inputs = torch.randn(16, 4)
        targets = torch.randn(16, 3)

        evaluator = NNCostEvaluator(model, loss_fn)
        assert evaluator._batched_linear is not None
        fast = evaluator.evaluate(stacked, inputs, targets)

        expected = torch.stack(
            [
                loss_fn(torch.func.functional_call(model, {k: v[i] for k, v in stacked.items()}, (inputs,)), targets)
                for i in range(N)
            ]
        )
        assert torch.allclose(fast, expected, atol=1e-6)

    def test_non_mean_reduction_falls_back(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        assert NNCostEvaluator(model, nn.MSELoss(reduction="sum"))._batched_linear is None

    def test_label_smoothing_falls_back_and_matches(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 6
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.05
        stacked = layout.batch_unflatten(batch)

        inputs = torch.randn(16, 4)
        targets = torch.randint(0, 3, (16,))
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        evaluator = NNCostEvaluator(model, loss_fn)
        # Configured loss must fall back so the real (smoothed) loss is used.
        assert evaluator._batched_linear is None

        losses = evaluator.evaluate(stacked, inputs, targets)

        model.eval()
        buffers = dict(model.named_buffers())
        expected = torch.zeros(N)
        for i in range(N):
            sd = {k: v[i] for k, v in stacked.items()}
            with torch.no_grad():
                out = functional_call(model, {**sd, **buffers}, (inputs,))
                expected[i] = loss_fn(out, targets)

        torch.testing.assert_close(losses, expected, atol=1e-5, rtol=1e-5)


def test_chunk_size_produces_same_result():
    """Chunked vmap evaluation matches unchunked. Uses MSELoss so the bmm fast
    path is skipped and chunk_size actually drives the vmap path."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 50
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    batch += torch.randn_like(batch) * 0.01
    stacked = layout.batch_unflatten(batch)

    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 2)

    ev_full = NNCostEvaluator(model, nn.MSELoss(), chunk_size=None)
    ev_chunk = NNCostEvaluator(model, nn.MSELoss(), chunk_size=4)

    losses_full = ev_full.evaluate(stacked, inputs, targets)
    losses_chunk = ev_chunk.evaluate(stacked, inputs, targets)

    torch.testing.assert_close(losses_chunk, losses_full, atol=1e-5, rtol=1e-5)


def test_auto_detect_chunk_size_cpu():
    """auto_detect_chunk_size returns None for CPU model (even on GPU machine)."""
    model = nn.Linear(100, 50)  # CPU model
    result = auto_detect_chunk_size(model)
    assert result is None, f"Expected None for CPU model, got {result}"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_auto_detect_chunk_size_returns_positive():
    """auto_detect_chunk_size returns positive int for GPU model."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).cuda()
    result = auto_detect_chunk_size(model)
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result > 0, f"Expected positive, got {result}"


def test_evaluator_auto_chunk_size():
    """NNCostEvaluator with chunk_size='auto' works correctly."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 20
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    batch += torch.randn_like(batch) * 0.01
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size="auto")
    # CPU model should always resolve to None, even on GPU machines
    resolved = evaluator.chunk_size
    assert resolved is None, f"Expected None for CPU model, got {resolved}"

    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 2, (16,))
    losses = evaluator.evaluate(stacked, inputs, targets)

    assert losses.shape == (N,)
    assert losses.isfinite().all()


def test_auto_chunk_size_cached():
    """Auto chunk_size should be computed once and cached."""
    model = nn.Linear(4, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss(), chunk_size="auto")
    cs1 = evaluator.chunk_size
    cs2 = evaluator.chunk_size
    assert cs1 == cs2  # same value (None on CPU)
    # Verify internal cache attribute exists and sentinel was replaced
    assert hasattr(evaluator, "_chunk_size_cached")
    from polystep.cost_nn import _UNSET

    assert evaluator._chunk_size_cached is not _UNSET


def _make_probe_setup(in_dim=4, hidden=8, out_dim=2, P=10, V=4, K=3):
    """Helper: create model, layout, evaluator, and probe array."""
    model = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    layout = ParamLayout.from_module(model)
    p = layout.flatten(model)
    D = p.shape[0] * p.shape[1]  # flat size
    X_probe = p.reshape(1, 1, 1, D).expand(P, V, K, D).clone()
    X_probe += torch.randn(P, V, K, D) * 0.01
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs = torch.randn(16, in_dim)
    targets = torch.randint(0, out_dim, (16,))
    return model, layout, evaluator, X_probe, inputs, targets


def _chunk_bound_step(chunk_size):
    """One full-space step at ~11K params, where the default chunk's memory bound engages."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 10))
    x = torch.randn(16, 100)
    y = torch.randint(0, 10, (16,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(
        model,
        subspace=None,
        solver="softmax",
        epsilon=0.5,
        step_radius=0.1,
        probe_radius=0.5,
        chunk_size=chunk_size,
        seed=123,
    )
    return float(opt.step(lambda s: ev.evaluate(s, x, y)))


def test_fullspace_default_chunk_matches_explicit_chunk():
    """Bounded default chunk must produce the same result as an explicit chunk."""
    loss_default = _chunk_bound_step(None)
    loss_explicit = _chunk_bound_step(500)
    assert abs(loss_default - loss_explicit) < 1e-4


if __name__ == "__main__":
    test_fullspace_default_chunk_matches_explicit_chunk()
    print("ok")


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(nn.ReLU(inplace=True), nn.Linear(6, 4)),
        nn.Sequential(nn.Linear(6, 5), nn.ReLU(inplace=True), nn.Linear(5, 4)),
    ],
    ids=["leading", "middle"],
)
def test_inplace_activations_defer_to_vmap(model):
    """The batched paths share activations and expand the input to stride 0, so an
    in-place module would write through tensors it does not own."""
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss()) is None

    # The correct path still runs: falling back is not the same as failing.
    ev = NNCostEvaluator(model, nn.MSELoss())
    layout = ParamLayout.from_module(model)
    flat = layout.flatten(model)
    losses = ev.evaluate(
        layout.batch_unflatten(flat.reshape(1, -1).repeat(3, 1)),
        torch.randn(8, 6, generator=torch.Generator().manual_seed(0)),
        torch.randn(8, 4, generator=torch.Generator().manual_seed(1)),
    )
    assert losses.shape == (3,) and torch.isfinite(losses).all()
    # Same weights in every row, so the same loss in every row.
    assert losses.std() == pytest.approx(0.0, abs=1e-6)
