"""Tests for NNCostEvaluator, ParamLayout.batch_unflatten(), chunked eval, and cost matrix."""

import inspect
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils._pytree import register_pytree_node

from polystep import HybridSubspace, PolyStepOptimizer, transform
from polystep.transform import ParamLayout
from polystep.cost_nn import (
    BatchedLinearEvaluator,
    NNCostEvaluator,
    SiteVmapEvaluator,
    auto_detect_chunk_size,
    cast_inputs_memo,
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
            return self.fc(x) * 1.0

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


def test_registration_rejects_another_model_without_replacing_current_batch():
    model = nn.Linear(4, 2)
    opt = PolyStepOptimizer(model, solver="softmax")
    inputs, targets = torch.randn(3, 4), torch.randn(3, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    closure = opt.register_evaluator(evaluator, inputs, targets)
    other = NNCostEvaluator(nn.Linear(4, 2), nn.MSELoss())
    with pytest.raises(ValueError, match="different model"):
        opt.register_evaluator(other, inputs + 1, targets)
    assert opt._cost_evaluator is evaluator
    assert opt._registration_closure is closure
    opt.step(closure)


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

    def test_tied_weight_inplace_matches_vmap(self):
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


class MaskedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x, mask):
        return self.lin(x) * mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pack", [lambda x, m: (x, m), lambda x, m: {"mask": m, "x": x}], ids=["tuple", "dict"])
def test_multi_input_matches_forward_loss(pack, dtype):
    torch.manual_seed(0)
    model = MaskedLinear()
    x, mask, y = torch.randn(5, 4), (torch.rand(5, 3) > 0.5).float(), torch.randn(5, 3)
    params = {k: v.detach()[None].expand(2, *v.shape) for k, v in model.named_parameters()}
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    losses = evaluator.evaluate(params, pack(x.to(dtype), mask), y)
    assert not evaluator._vmap_failed
    with torch.no_grad():
        expected = nn.functional.mse_loss(model(x, mask), y)
    torch.testing.assert_close(losses, expected.expand(2))


class NestedMaskedLinear(MaskedLinear):
    def forward(self, batch):
        return super().forward(batch["x"], batch["mask"])


def test_nested_inputs_cache_distinguishes_structure():
    torch.manual_seed(0)
    model = NestedMaskedLinear()
    x, mask, y = torch.randn(5, 4), (torch.rand(5, 3) > 0.5).float(), torch.randn(5, 3)
    params = {k: v.detach().unsqueeze(0) for k, v in model.named_parameters()}
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    with torch.no_grad():
        expected = nn.functional.mse_loss(model({"x": x, "mask": mask}), y)
    for batch in (
        {"x": x, "mask": mask.double()},
        {"mask": mask, "x": x.double()},
        {"pad": x.double(), "x": x, "mask": mask.long()},
        {"pad": x.long(), "x": x.double(), "mask": mask},
    ):
        losses = evaluator.evaluate(params, (batch,), y)
        torch.testing.assert_close(losses, expected.unsqueeze(0), check_dtype=False)


class AuxBox:
    def __init__(self, aux):
        self.aux = aux


register_pytree_node(AuxBox, lambda b: ([b.aux], torch.arange(3)), lambda leaves, _: AuxBox(*leaves))


class AddAux(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x, box):
        return self.lin(x) + box.aux


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_registered_pytree_inputs_skip_key_paths(dtype):
    torch.manual_seed(0)
    model = AddAux()
    x, aux, y = torch.randn(5, 4), torch.randn(5, 3), torch.randn(5, 3)
    params = {k: v.detach().unsqueeze(0) for k, v in model.named_parameters()}
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    with torch.no_grad():
        expected = nn.functional.mse_loss(model(x, AuxBox(aux)), y)
    for _ in range(2):
        losses = evaluator.evaluate(params, (x.to(dtype), AuxBox(aux)), y)
        torch.testing.assert_close(losses, expected.unsqueeze(0))


def test_evaluator_different_params_different_losses():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    layout = ParamLayout.from_module(model)
    particle = layout.flatten(model)

    N = 10
    batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
    batch += torch.randn_like(batch) * 1.0
    stacked = layout.batch_unflatten(batch)

    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    inputs = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32,))
    losses = evaluator.evaluate(stacked, inputs, targets)

    assert not torch.all(losses == losses[0]), "All losses identical despite different params"


def test_evaluator_batchnorm():
    """Evaluator handles BatchNorm (frozen buffers)."""
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

    fallback_warnings = [x for x in w if "Falling back" in str(x.message)]
    assert len(fallback_warnings) > 0, "Expected 'Falling back' warning"

    assert losses.shape == (N,)
    assert losses.isfinite().all()


def test_gru_forward_falls_back_to_loop():
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(3, 4, batch_first=True)
            self.out = nn.Linear(4, 2)

        def forward(self, x):
            return self.out(self.rnn(x)[0][:, -1])

    model, loss_fn = Net(), nn.MSELoss()
    x, y = torch.randn(4, 5, 3), torch.randn(4, 2)
    stacked = {k: v.detach() + 0.1 * torch.randn(3, *v.shape) for k, v in model.named_parameters()}
    evaluator = NNCostEvaluator(model, loss_fn, use_inplace=False)
    with pytest.warns(UserWarning, match="Falling back"):
        losses = evaluator.evaluate(stacked, x, y)
    expected = [loss_fn(functional_call(model, {k: v[i] for k, v in stacked.items()}, (x,)), y) for i in range(3)]
    torch.testing.assert_close(losses, torch.stack(expected))
    assert evaluator._vmap_failed
    assert evaluator.evaluate({k: v[:0] for k, v in stacked.items()}, x, y).shape == (0,)
    assert evaluator._evaluate_loop({}, x, y).shape == (0,)


def test_vmap_oom_raises_without_fallback():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            h = self.fc(x)
            if torch._C._functorch.is_batchedtensor(h):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return h

    model = Net()
    stacked = {k: v.detach().expand(3, *v.shape) for k, v in model.named_parameters()}
    evaluator = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        evaluator.evaluate(stacked, torch.randn(5, 4), torch.randn(5, 2))
    assert not evaluator._vmap_failed


def test_evaluator_no_grad():
    """Evaluation does not attach gradients to model params."""
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
    """The bmm fast path must reproduce a configured mean CrossEntropyLoss and decline
    other reductions."""

    def test_vanilla_ce_uses_fast_path(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        assert evaluator._batched_linear is not None

    def test_use_inplace_wins_over_bmm_path(self):
        """``use_inplace`` is a memory contract and must outrank the bmm fast path."""
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

    @pytest.mark.parametrize("weight", [None, torch.rand(200) + 0.5], ids=["plain", "weight"])
    def test_byte_label_156_stays_valid(self, weight):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 200))
        layout = ParamLayout.from_module(model)
        params = layout.batch_unflatten(layout.flatten(model).unsqueeze(0).expand(2, -1, -1).clone())
        x, y = torch.randn(3, 4), torch.tensor([156, 3, 156], dtype=torch.uint8)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        evaluator = NNCostEvaluator(model, loss_fn)
        assert evaluator._batched_linear is not None
        with torch.no_grad():
            expected = loss_fn(model(x), y.long()).expand(2)
        torch.testing.assert_close(evaluator.evaluate(params, x, y), expected)

    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.L1Loss()])
    def test_regression_losses_fast_path_matches_vmap(self, loss_fn):
        """The bmm forward is loss-independent; only the reduction differs."""
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

    @pytest.mark.parametrize("target_shape", [(6,), (5, 6, 1)])
    @pytest.mark.parametrize("loss_fn", [nn.MSELoss(), nn.L1Loss()])
    def test_mse_target_broadcast_matches_loop(self, loss_fn, target_shape):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 1)).double()
        layout = ParamLayout.from_module(model)
        batch = layout.flatten(model).unsqueeze(0).expand(5, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.05
        stacked = layout.batch_unflatten(batch)
        inputs = torch.randn(6, 4, dtype=torch.float64)
        targets = torch.randn(target_shape, dtype=torch.float64)

        evaluator = NNCostEvaluator(model, loss_fn)
        assert evaluator._batched_linear is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fast = evaluator.evaluate(stacked, inputs, targets)
            expected = torch.stack(
                [
                    loss_fn(functional_call(model, {k: v[i] for k, v in stacked.items()}, (inputs,)), targets)
                    for i in range(5)
                ]
            )
        torch.testing.assert_close(fast, expected, rtol=1e-12, atol=1e-12)

    def test_non_mean_reduction_falls_back(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        assert NNCostEvaluator(model, nn.MSELoss(reduction="sum"))._batched_linear is None

    class_weight = torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64)

    @pytest.mark.parametrize(
        "config, target_dtype",
        [
            ({"label_smoothing": 0.1}, torch.long),
            ({"weight": class_weight}, torch.long),
            ({"weight": class_weight}, torch.uint8),
            ({"weight": class_weight, "label_smoothing": 0.2, "ignore_index": 1}, torch.long),
            ({"weight": class_weight, "label_smoothing": 0.2}, torch.float64),
        ],
        ids=["smoothing", "weight", "weight_uint8", "weight_smoothing_ignore", "probability_targets"],
    )
    def test_label_smoothing_matches_closure_loss(self, config, target_dtype):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).double()
        layout = ParamLayout.from_module(model)
        particle = layout.flatten(model)

        N = 6
        batch = particle.unsqueeze(0).expand(N, -1, -1).clone()
        batch += torch.randn_like(batch) * 0.05
        stacked = layout.batch_unflatten(batch)

        inputs = torch.randn(16, 4, dtype=torch.float64)
        targets = torch.randint(0, 3, (16,)).to(target_dtype)
        if target_dtype.is_floating_point:
            targets = torch.softmax(torch.randn(16, 3, dtype=torch.float64), dim=1)
        loss_fn = nn.CrossEntropyLoss(**config)

        evaluator = NNCostEvaluator(model, loss_fn)
        assert evaluator._batched_linear is not None

        losses = evaluator.evaluate(stacked, inputs, targets)

        model.eval()
        buffers = dict(model.named_buffers())
        expected = torch.zeros(N, dtype=torch.float64)
        for i in range(N):
            sd = {k: v[i] for k, v in stacked.items()}
            with torch.no_grad():
                out = functional_call(model, {**sd, **buffers}, (inputs,))
                expected[i] = loss_fn(out, targets)

        torch.testing.assert_close(losses, expected, atol=1e-12, rtol=0)


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
    model = nn.Linear(100, 50)
    result = auto_detect_chunk_size(model)
    assert result is None, f"Expected None for CPU model, got {result}"


def test_evaluator_auto_chunk_size():
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
    model = nn.Linear(4, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss(), chunk_size="auto")
    cs1 = evaluator.chunk_size
    cs2 = evaluator.chunk_size
    assert cs1 == cs2
    assert hasattr(evaluator, "_chunk_size_cached")
    from polystep.cost_nn import _UNSET

    assert evaluator._chunk_size_cached is not _UNSET


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
    loss_default = _chunk_bound_step(None)
    loss_explicit = _chunk_bound_step(500)
    assert abs(loss_default - loss_explicit) < 1e-4


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


def test_docstring_names_every_constructor_argument():
    params = [p for p in inspect.signature(NNCostEvaluator.__init__).parameters if p != "self"]
    assert [p for p in params if f"{p}:" not in NNCostEvaluator.__doc__] == []


class TestEvalModeEnforced:
    """NNCostEvaluator must enforce eval mode even if user calls model.train()."""

    def test_eval_mode_enforced_during_evaluation(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        loss_fn = nn.CrossEntropyLoss()
        evaluator = NNCostEvaluator(model, loss_fn)

        model.train()
        assert model.training

        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)
        N = 4
        flat_batch = flat.unsqueeze(0).repeat(N, 1, 1) + torch.randn(N, *flat.shape) * 0.01
        stacked = layout.batch_unflatten(flat_batch)

        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 2, (8,))

        rm_before = model.state_dict()["1.running_mean"].clone()
        evaluator.evaluate(stacked, inputs, targets)
        rm_after = model.state_dict()["1.running_mean"]

        assert torch.equal(rm_before, rm_after), "BatchNorm running stats were mutated during evaluation!"
        assert model.training, "Model should be restored to train mode after evaluate()"

    def test_eval_mode_restored_on_error(self):
        model = nn.Linear(10, 2)

        def bad_loss(output, targets):
            raise ValueError("intentional")

        evaluator = NNCostEvaluator(model, bad_loss)
        model.train()
        assert model.training

        layout = ParamLayout.from_module(model)
        flat = layout.flatten(model)
        stacked = layout.batch_unflatten(flat.unsqueeze(0))

        with pytest.raises(ValueError, match="intentional"):
            evaluator.evaluate(stacked, torch.randn(1, 10), torch.zeros(1, dtype=torch.long))

        assert model.training, "Model mode should be restored even after error"
        assert not evaluator._vmap_failed


def test_inplace_evaluator_rejects_unknown_parameter_keys():
    model = nn.Linear(4, 2)
    evaluator = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)
    stacked = {
        "weight": model.weight.detach().unsqueeze(0).repeat(2, 1, 1),
        "ghost": torch.zeros(2, 3),
    }
    with pytest.raises(ValueError, match="ghost"):
        evaluator.evaluate(stacked, torch.randn(5, 4), torch.randn(5, 2))


def test_dense_activation_batches_respect_chunk_size():
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    ev = NNCostEvaluator(model, nn.MSELoss(), chunk_size=3, use_inplace=False)
    params = {k: v.detach()[None] + torch.randn(8, *v.shape) * 0.1 for k, v in model.named_parameters()}
    x, y = torch.randn(5, 3), torch.randn(5, 2)
    with patch("torch.baddbmm", wraps=torch.baddbmm) as matmul:
        got = ev.evaluate(params, x, y)
    assert max(call.args[1].shape[0] for call in matmul.call_args_list) <= 3
    torch.testing.assert_close(got, ev._evaluate_loop(params, x, y))


@pytest.mark.parametrize("chunk", [0, -1, True, 2.5, "invalid"])
def test_evaluator_rejects_invalid_chunks(chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        NNCostEvaluator(nn.Linear(3, 2), nn.MSELoss(), chunk_size=chunk)


def test_reset_recomputes_automatic_chunk_size():
    ev = NNCostEvaluator(nn.Linear(3, 2), nn.MSELoss(), chunk_size="auto")
    with patch("polystep.cost_nn.auto_detect_chunk_size", side_effect=[8, 2]):
        assert ev.chunk_size == 8
        ev.reset_vmap()
        assert ev.chunk_size == 2


@pytest.mark.parametrize("compile_vmap", [False, True])
def test_loss_fn_swap_scores_new_loss(compile_vmap):
    torch.manual_seed(0)
    layers = [nn.Linear(3, 4), nn.LayerNorm(4)] if compile_vmap else [nn.Linear(3, 4)]
    model = nn.Sequential(*layers).double()
    x, y = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, 4, dtype=torch.float64)
    params = {k: v.detach().unsqueeze(0) for k, v in model.named_parameters()}
    evaluator = NNCostEvaluator(model, nn.MSELoss(), compile_vmap=compile_vmap)
    evaluator.evaluate(params, x, y)
    evaluator.loss_fn = nn.L1Loss()
    with torch.no_grad():
        expected = nn.L1Loss()(model(x), y).reshape(1)
    torch.testing.assert_close(evaluator.evaluate(params, x, y), expected)


@pytest.mark.parametrize("site", ["0.weight", "0.bias", "2.weight", "2.bias"])
def test_dense_shared_weights_match_materialized_candidates(site):
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    params = {k: v.detach()[None].expand(8, *v.shape) for k, v in model.named_parameters()}
    params[site] = params[site] + torch.randn_like(params[site]) * 0.1
    x, y = torch.randn(5, 3).double(), torch.randn(5, 2).double()
    expected = ev._evaluate_loop(params, x, y)
    torch.testing.assert_close(ev.evaluate(params, x, y), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ev.evaluate({k: v.clone() for k, v in params.items()}, x, y), expected)


@pytest.mark.parametrize("inference_tensor", [False, True])
def test_cast_cache_observes_input_mutation(inference_tensor):
    holder = SimpleNamespace()
    with torch.inference_mode(inference_tensor):
        inputs = torch.ones(2, dtype=torch.float64)
        first = cast_inputs_memo(holder, inputs, torch.float32)
        if not inference_tensor:
            assert cast_inputs_memo(holder, inputs, torch.float32) is first
        inputs.fill_(7)
        torch.testing.assert_close(cast_inputs_memo(holder, inputs, torch.float32), torch.full((2,), 7.0))


@pytest.mark.parametrize("shape", [(), (64, 1)])
def test_forward_accepting_fp32_skips_cast(shape):
    class Centred(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("t0", torch.tensor(5000.0).expand(shape))
            self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)).bfloat16()

        def forward(self, t):
            return self.net((t - self.t0).to(torch.bfloat16)).float()

    torch.manual_seed(0)
    model = Centred()
    t = 5000 + torch.linspace(0, 3, 64)[:, None]
    y = torch.sin(t - 5000)
    params = {k: v.detach().unsqueeze(0) for k, v in model.named_parameters()}
    with torch.no_grad():
        expected = nn.MSELoss()(model(t), y).item()
    got = NNCostEvaluator(model, nn.MSELoss()).evaluate(params, t, y).item()
    assert got == pytest.approx(expected, rel=1e-2)


class CountingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.register_buffer("calls", torch.zeros(()))
        self.register_buffer("table", torch.ones(8))

    def forward(self, x):
        self.calls += 1
        return self.fc(x) * self.calls


@pytest.mark.parametrize("looped", [False, True])
def test_candidate_sweep_leaves_buffers_unchanged(looped):
    torch.manual_seed(0)
    model = CountingLinear()
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    params = {k: v.detach().expand(6, *v.shape) for k, v in model.named_parameters()}
    ev = NNCostEvaluator(model, nn.MSELoss())
    ev._vmap_failed = looped
    whole = ev.evaluate(params, x, y)
    assert model.calls.item() == 0
    for start in range(0, 6, 2):
        part = ev.evaluate({k: v[start : start + 2] for k, v in params.items()}, x, y)
        assert torch.equal(part, whole[start : start + 2])
        assert model.calls.item() == 0


@pytest.mark.parametrize("path", ["dense", "subspace", "site"])
def test_inplace_sweep_restores_buffer_writes(path):
    torch.manual_seed(0)
    model = CountingLinear()
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)
    if path == "dense":
        params = {k: v.detach() + 0.1 * torch.randn(4, *v.shape) for k, v in model.named_parameters()}
        losses = ev.evaluate(params, x, y)
        torch.testing.assert_close(losses, NNCostEvaluator(model, nn.MSELoss()).evaluate(params, x, y))
    elif path == "subspace":
        sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
        projections = sub.init_projections(torch.device("cpu"), torch.float32)
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        losses = ev.evaluate_subspace_inplace(sub, projections, base, torch.zeros(4, sub.subspace_dim), x, y)
        assert torch.equal(losses, losses[:1].expand_as(losses))
    else:
        layout = ParamLayout.from_module(model, particle_dim=2)
        site = SiteVmapEvaluator.try_build(ev, layout)
        flat = layout.flatten(model).reshape(-1)
        entry = site.resolve_site(torch.tensor([0]), 2, span=(0, 2))
        losses = site.evaluate(
            layout.unflatten(flat), entry, torch.arange(2).view(1, 2), flat[:2].repeat(1, 4, 1), x, y
        )
        assert torch.equal(losses, losses[:1].expand_as(losses))
    assert model.calls.item() == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_inplace_verify_restores_buffers():
    torch.manual_seed(0)
    model = CountingLinear().cuda()
    x, y = torch.randn(8, 4, device="cuda"), torch.randn(8, 1, device="cuda")
    params = {k: v.detach() + 0.1 * torch.randn(4, *v.shape, device="cuda") for k, v in model.named_parameters()}
    expected = NNCostEvaluator(model, nn.MSELoss()).evaluate(params, x, y)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)
    torch.testing.assert_close(ev.evaluate(params, x, y), expected)
    assert ev._compile_forward_verified
    assert model.calls.item() == 0


class LastStepRNN(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.rnn = cell(4, 6, batch_first=True)
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        return self.fc(self.rnn(x)[0][:, -1])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("path", ["dense", "site"])
@pytest.mark.parametrize("cell", [nn.LSTM, nn.GRU])
def test_rnn_inplace_after_rebind_matches_loop(cell, path):
    torch.manual_seed(0)
    model = LastStepRNN(cell).cuda()
    x, y = torch.randn(8, 5, 4, device="cuda"), torch.randint(0, 3, (8,), device="cuda")
    layout = ParamLayout.from_module(model, particle_dim=2)
    flat = layout.flatten(model).reshape(-1)
    noise = 0.05 * torch.randn(4, flat.numel(), device="cuda")
    if path == "site":
        noise[:, 2:] = 0
    stacked = layout.batch_unflatten(flat + noise)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    with pytest.warns(UserWarning, match="vmap failed"):
        expected = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False).evaluate(stacked, x, y)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True, compile_forward=False)
    if path == "dense":
        losses = ev.evaluate(stacked, x, y)
    else:
        site = SiteVmapEvaluator.try_build(ev, layout)
        entry = site.resolve_site(torch.tensor([0]), 2, span=(0, 2))
        values = (flat[:2] + noise[:, :2]).unsqueeze(0)
        losses = site.evaluate(layout.unflatten(flat), entry, torch.arange(2, device="cuda").view(1, 2), values, x, y)
    torch.testing.assert_close(losses, expected)
    for k, v in model.named_parameters():
        assert torch.equal(v, before[k]), k


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cell", [nn.LSTM, nn.GRU])
@pytest.mark.parametrize("key", ["rnn.weight_ih_l0", "rnn.bias_ih_l0"])
def test_rnn_subspace_after_rebind_matches_swap(cell, key):
    torch.manual_seed(0)
    model = LastStepRNN(cell).cuda()
    x, y = torch.randn(8, 5, 4, device="cuda"), torch.randint(0, 3, (8,), device="cuda")
    layout = ParamLayout.from_module(model, particle_dim=2)
    sub = HybridSubspace.from_layout(layout, rank=3)
    projections = sub.init_projections(torch.device("cuda"), torch.float32)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True, compile_forward=False)
    site = SiteVmapEvaluator.try_build(ev, layout)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    coords = 0.1 * torch.randn(sub.subspace_dim, device="cuda")
    bary = sub.apply_perturbation(projections, base, coords)
    spec = next(s for s in sub.specs if s.entry_key == key)
    values = coords[spec.flat_start : spec.flat_start + 2] + 0.3 * torch.randn(1, 4, 2, device="cuda")
    full = coords.repeat(4, 1)
    full[:, spec.flat_start : spec.flat_start + 2] = values[0]
    expected = ev.evaluate_subspace_inplace(sub, projections, base, full, x, y)
    torch.func.functional_call(model, {k: v.detach().clone() for k, v in model.named_parameters()}, (x,))
    losses = site.evaluate_subspace(projections, base, coords, bary, spec, 0, values, x, y)
    torch.testing.assert_close(losses, expected)
    for k, v in bary.items():
        assert torch.equal(model.state_dict()[k], v), k


def test_sequential_sweep_clones_buffers_once(monkeypatch):
    model = CountingLinear()
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    params = {k: v.detach().expand(6, *v.shape) for k, v in model.named_parameters()}
    ev = NNCostEvaluator(model, nn.MSELoss())
    ev._vmap_failed = True
    ev.evaluate(params, x, y)
    clone, table_clones = torch.Tensor.clone, []

    def counting_clone(t, *args, **kwargs):
        table_clones.append(t is model.table)
        return clone(t, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)
    ev.evaluate(params, x, y)
    assert sum(table_clones) == 1
    assert model.calls.item() == 0


@pytest.mark.parametrize("looped", [False, True])
def test_late_buffer_write_restored(looped):
    class LateCounter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, x):
            if x.shape[0] == 2:
                self.counter.add_(1)
            return x * self.weight + self.counter

    model = LateCounter()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    ev._vmap_failed = looped
    params = {"weight": torch.ones(3, 1)}
    ev.evaluate(params, torch.ones(1, 1), torch.zeros(1, 1))
    assert ev.evaluate(params, torch.ones(2, 1), torch.zeros(2, 1)).tolist() == [4.0, 4.0, 4.0]
    assert model.counter.item() == 0


def test_later_candidate_write_restored():
    class WeightCounter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, x):
            if self.weight.item() > 1:
                self.counter.add_(1)
            return x * self.weight + self.counter

    model = WeightCounter()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    ev._vmap_failed = True
    losses = ev.evaluate({"weight": torch.tensor([[1.0], [2.0], [1.0]])}, torch.ones(1, 1), torch.zeros(1, 1))
    assert losses.tolist() == [1.0, 9.0, 1.0]
    assert model.counter.item() == 0


def test_vmap_fallback_restores_unversioned_write():
    class ItemEma(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("ema", torch.zeros(()))

        def forward(self, x):
            self.ema.data.add_(self.weight.item())
            return x * self.weight + self.ema

    model = ItemEma()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    with pytest.warns(UserWarning, match="vmap failed"):
        losses = ev.evaluate({"weight": torch.ones(3, 1)}, torch.ones(1, 1), torch.zeros(1, 1))
    assert losses.tolist() == [4.0, 4.0, 4.0]
    assert model.ema.item() == 0


class CachedScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("score", torch.zeros(()))

    def forward(self, x):
        self.score.copy_((x * self.weight).sum())
        return self.score


@pytest.mark.filterwarnings("ignore:vmap failed:RuntimeWarning")
@pytest.mark.parametrize("path", ["loop", "site", "site_inplace"])
def test_loss_aliasing_buffer_survives_restore(path):
    model = CachedScore()
    ev = NNCostEvaluator(model, lambda out: out, use_inplace=path == "site_inplace", compile_forward=False)
    weights = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    if path == "loop":
        ev._vmap_failed = True
        losses = ev.evaluate({"weight": weights}, torch.ones(2), None)
    else:
        layout = ParamLayout.from_module(model, particle_dim=2)
        site = SiteVmapEvaluator.try_build(ev, layout)
        base = layout.unflatten(layout.flatten(model))
        losses = site.evaluate(base, layout.entries[0], torch.tensor([[0, 1]]), weights[None], torch.ones(2))
    assert losses.tolist() == [2.0, 4.0]
    assert model.score.item() == 0


@pytest.mark.parametrize("candidates, chunk", [(4, None), (4, 1), (4, 2), (1, None)])
def test_buffer_backed_loss_survives_restore(candidates, chunk):
    class CachedScore(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("score", torch.zeros(()))

        def forward(self, x):
            self.score.add_(x.sum())
            return self.score

    model = CachedScore()
    ev = NNCostEvaluator(model, lambda out: out, use_inplace=False, compile_forward=False, chunk_size=chunk)
    layout = ParamLayout.from_module(model, particle_dim=2)
    site = SiteVmapEvaluator.try_build(ev, layout)
    weights = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])[:candidates]
    dense = ev.evaluate({"weight": weights}, torch.ones(2))
    local = site.evaluate(
        layout.unflatten(layout.flatten(model)), layout.entries[0], torch.tensor([[0, 1]]), weights[None], torch.ones(2)
    )
    assert dense.tolist() == local.tolist() == [2.0] * candidates
    assert not ev._vmap_failed
    assert model.score.item() == 0


@pytest.mark.parametrize("path", ["vmap", "loop", "inplace", "hybrid"])
@pytest.mark.parametrize("fails", [False, True])
def test_frozen_max_norm_embedding_is_restored(path, fails):
    embedding = nn.Embedding(4, 2, max_norm=1.0)
    embedding.weight.requires_grad_(False)
    with torch.no_grad():
        embedding.weight.fill_(3.0)
    model = nn.Sequential(embedding, nn.Flatten(), nn.Linear(4, 2))
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    inputs = torch.tensor([[0, 1], [2, 3]])

    def loss(out):
        if fails:
            raise ValueError("failed loss")
        return out.square().mean()

    ev = NNCostEvaluator(model, loss, use_inplace=path in ("inplace", "hybrid"), compile_forward=False)
    ev._vmap_failed = path == "loop"
    layout = ParamLayout.from_module(model)
    if path == "hybrid":
        sub = HybridSubspace.from_layout(layout, rank=1)
        projections = sub.init_projections(torch.device("cpu"), torch.float32)
        base = {k: v for k, v in before.items() if k != "0.weight"}

        def evaluate():
            return ev.evaluate_subspace_inplace(sub, projections, base, torch.zeros(2, sub.subspace_dim), inputs)
    else:
        params = {k: v.expand(2, *v.shape).clone() for k, v in before.items() if k != "0.weight"}

        def evaluate():
            return ev.evaluate(params, inputs)

    if fails:
        with pytest.raises(ValueError, match="failed loss"):
            evaluate()
    else:
        with torch.inference_mode():
            losses = evaluate()
        torch.testing.assert_close(losses[0], losses[1])
    for key, param in model.named_parameters():
        torch.testing.assert_close(param, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("looped", [False, True])
def test_registration_probe_keeps_running_stats(looped):
    class RunningNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("mean", torch.zeros(2))
            self.register_buffer("var", torch.ones(2))
            self.register_buffer("seen", torch.zeros(0))
            self.register_buffer("wide", self.mean.expand(3, 2))

        def forward(self, x):
            torch.ops.aten.resize_(self.seen, [2])
            return nn.functional.batch_norm(x, self.mean, self.var, weight=self.weight, training=True) + self.mean

    model = RunningNorm()
    ev = NNCostEvaluator(model, nn.MSELoss())
    ev._vmap_failed = looped
    opt = PolyStepOptimizer(model, solver="softmax", seed=1)
    x = torch.arange(16.0).reshape(8, 2)
    closure = opt.register_evaluator(ev, x, torch.zeros_like(x))
    assert model.mean.tolist() == [0.0, 0.0]
    assert model.seen.shape == (0,)
    losses = ev.evaluate({"weight": torch.ones(3, 2)}, x, torch.zeros_like(x))
    assert losses.tolist() == [losses[0].item()] * 3
    opt.step(closure)
    assert model.mean.tolist() == [0.0, 0.0]
    assert model.var.tolist() == [1.0, 1.0]
    assert model.seen.shape == (0,)


@pytest.mark.parametrize("looped", [False, True])
@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_sparse_buffer_model_steps(layout, looped):
    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("adj", (torch.eye(8) + torch.eye(8).roll(1, 1)).to_sparse(layout=layout))

        def forward(self, x):
            return torch.sparse.mm(self.adj, self.lin(x))

    torch.manual_seed(0)
    model = Graph()
    dense = model.adj.to_dense()
    ev = NNCostEvaluator(model, nn.MSELoss())
    ev._vmap_failed = looped
    opt = PolyStepOptimizer(model, solver="softmax", seed=1)
    opt.step(opt.register_evaluator(ev, torch.randn(8, 2), torch.randn(8, 2)))
    assert torch.equal(model.adj.to_dense(), dense)


@pytest.mark.parametrize("looped", [False, True])
def test_nan_expanded_buffer_registers(looped):
    class Padded(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("pad", torch.tensor(float("nan")).expand(4))

        def forward(self, x):
            return self.lin(x) + torch.nan_to_num(self.pad[:2])

    torch.manual_seed(0)
    model = Padded()
    ev = NNCostEvaluator(model, nn.MSELoss())
    ev._vmap_failed = looped
    opt = PolyStepOptimizer(model, solver="softmax", seed=1)
    opt.step(opt.register_evaluator(ev, torch.randn(8, 2), torch.randn(8, 2)))
    assert model.pad.isnan().all()


@pytest.mark.parametrize("looped", [False, True])
def test_inference_expanded_buffer_scores(looped):
    class Tabled(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 1)
            with torch.inference_mode():
                self.register_buffer("table", torch.arange(4.0).expand(4, 4))

        def forward(self, x):
            return self.fc(x @ self.table)

    torch.manual_seed(0)
    model = Tabled()
    x, y = torch.randn(8, 4), torch.randn(8, 1)
    params = {k: v.detach().expand(3, *v.shape) for k, v in model.named_parameters()}
    ev = NNCostEvaluator(model, nn.MSELoss())
    ev._vmap_failed = looped
    with torch.no_grad():
        expected = nn.MSELoss()(model(x), y)
    torch.testing.assert_close(ev.evaluate(params, x, y), expected.expand(3))


def test_evaluator_buffer_alias_raises():
    model = nn.Sequential(nn.Linear(2, 2))
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    opt = PolyStepOptimizer(model, subspace=subspace, solver="softmax", seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss())
    model.register_buffer("alias", model[0].weight.detach())

    for rebind in (lambda: NNCostEvaluator(model, nn.MSELoss()), ev.reset_vmap, opt.resync_from_model):
        with pytest.raises(ValueError, match="'alias' and trainable parameter '0.weight' share storage"):
            rebind()


def test_lazy_evaluator_defers_alias_check():
    torch.manual_seed(0)
    model = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(3))
    model.register_buffer("alias", torch.zeros(()))
    ev = NNCostEvaluator(model, nn.MSELoss())
    x, y = torch.randn(16, 5), torch.randn(16, 3)
    model(x)
    model.alias = model[0].weight.detach()
    params = {k: v.detach().expand(3, *v.shape) for k, v in model.named_parameters()}
    for _ in range(2):
        with pytest.raises(ValueError, match="'alias' and trainable parameter '0.weight' share storage"):
            ev.evaluate(params, x, y)
    model.alias = torch.zeros(())
    opt = PolyStepOptimizer(model, solver="softmax", seed=0)
    assert torch.isfinite(torch.as_tensor(opt.step(lambda s: ev.evaluate(s, x, y))))


def test_failed_alias_check_stays_pending():
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    model.register_buffer("alias", torch.zeros(()))
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    model.alias = model[0].weight.detach()

    with pytest.raises(ValueError, match="'alias' and trainable parameter '0.weight' share storage"):
        ev.reset_vmap()
    with pytest.raises(ValueError, match="'alias' and trainable parameter '0.weight' share storage"):
        ev.evaluate({"0.weight": torch.tensor([[[2.0]], [[3.0]]])}, torch.ones(1, 1), torch.zeros(1, 1))


def test_meta_layer_keeps_alias_check():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2, device="meta"))
    NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    model.register_buffer("alias", model[0].weight.detach())

    with pytest.raises(ValueError, match="'alias' and trainable parameter '0.weight' share storage"):
        NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)


def test_evaluator_reset_reuses_alias_check(monkeypatch):
    calls, shares_bytes = [], transform._shares_bytes
    monkeypatch.setattr(transform, "_shares_bytes", lambda a, b: calls.append(a) or shares_bytes(a, b))
    monkeypatch.setattr(transform, "_validated_signature", None)
    model, matrix = nn.Module(), torch.ones(2, 2)
    model.a, model.b = nn.Parameter(matrix[:, 0]), nn.Parameter(matrix[:, 1])

    ParamLayout.from_module(model)
    NNCostEvaluator(model, nn.MSELoss(), use_inplace=False).reset_vmap()
    assert len(calls) == 1


def test_vmap_shares_expanded_parameters():
    torch.manual_seed(0)

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3)
            self.fc = nn.Linear(8, 3)

        def forward(self, x):
            return self.fc(torch.relu(self.conv(x)).flatten(1))

    model = ConvNet().double()
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    params = {k: v.detach().expand(4, *v.shape) for k, v in model.named_parameters()}
    params["fc.bias"] = params["fc.bias"] + torch.randn(4, 3, dtype=torch.float64)
    batched = []
    model.conv.register_forward_hook(lambda m, a, out: batched.append(torch._C._functorch.is_batchedtensor(out)))
    x, y = torch.randn(5, 1, 4, 4, dtype=torch.float64), torch.randint(0, 3, (5,))

    shared = ev.evaluate(params, x, y)
    cloned = ev.evaluate({k: v.clone() for k, v in params.items()}, x, y)
    torch.testing.assert_close(shared, cloned, rtol=0, atol=1e-12)
    assert batched == [False, True]


def test_residual_straddler_keeps_dense_vmap():
    torch.manual_seed(0)

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(4, 4)
            self.short = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            h = self.body(x)
            h += self.short(x)
            return self.head(torch.tanh(h))

    model = ResidualMLP().double()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    params = {k: v.detach().expand(3, *v.shape) for k, v in model.named_parameters()}
    for k in ("short.weight", "short.bias"):
        params[k] = params[k] + 0.1 * torch.randn_like(params[k])
    x, y = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    expected = [
        nn.functional.mse_loss(functional_call(model, {k: v[i] for k, v in params.items()}, (x,)), y) for i in range(3)
    ]

    torch.testing.assert_close(ev.evaluate(params, x, y), torch.stack(expected))
    assert not ev._vmap_failed


class MaskedAttention(nn.Module):
    def __init__(self, lead, tail):
        super().__init__()
        self.lead, self.attn, self.tail = lead, nn.MultiheadAttention(8, 2, batch_first=True), tail

    def forward(self, x, mask):
        h = self.lead(x)
        return self.tail(self.attn(h, h, h, key_padding_mask=mask, need_weights=False)[0]).mean((1, 2))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "lead, tail, misaligned",
    [
        (lambda: nn.Sequential(nn.PReLU(), nn.PReLU()), lambda: nn.Linear(8, 2), (True, False)),
        (lambda: nn.Linear(8, 8, bias=False), lambda: nn.Sequential(nn.PReLU(), nn.PReLU()), (False, True)),
    ],
    ids=["start", "stride"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_masked_attention_unaligned_views_match_slowpath(lead, tail, misaligned, dtype):
    torch.manual_seed(0)
    model = MaskedAttention(lead(), tail()).to("cuda", dtype)
    x = torch.randn(6, 5, 8, device="cuda", dtype=dtype)
    inputs, y = (x, x[..., 0] < -1), torch.zeros(6, device="cuda", dtype=dtype)
    layout = ParamLayout.from_module(model)
    offset = next(e.offset for e in layout.entries if e.key == "attn.in_proj_bias")
    assert (offset % 4 != 0, layout.padded_size % 4 != 0) == misaligned
    flat = layout.flatten(model).reshape(-1)
    stacked = layout.batch_unflatten(torch.stack([flat, flat + 0.01, flat - 0.01]))
    ev = NNCostEvaluator(model, nn.MSELoss())
    fast = ev.evaluate(stacked, inputs, y)
    shared = layout.batch_unflatten(flat.expand(3, -1))
    torch.testing.assert_close(ev.evaluate(shared, inputs, y), fast[:1].expand(3), rtol=1e-4, atol=1e-5)
    ev._vmap_failed = True
    torch.testing.assert_close(ev.evaluate(stacked, inputs, y), fast, rtol=1e-4, atol=1e-5)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        slow = ev.evaluate(stacked, inputs, y)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    torch.testing.assert_close(fast, slow, rtol=1e-4, atol=1e-5)
    for strategy in ("monolithic", "per_layer"):
        cost = PolyStepOptimizer(model, seed=0, block_strategy=strategy).step_batch(inputs, y, loss_fn=nn.MSELoss())
        assert torch.isfinite(torch.tensor(cost))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_attention_free_stacks_skip_alignment_copy(monkeypatch):
    model = nn.Sequential(nn.PReLU(), nn.Linear(8, 3)).cuda()
    layout = ParamLayout.from_module(model)
    stacked = layout.batch_unflatten(layout.flatten(model).reshape(1, -1).repeat(3, 1))
    ev, seen = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False), {}
    monkeypatch.setattr(ev, "_evaluate_vmap", lambda params, *_: seen.update(params) or torch.zeros(3))
    ev.evaluate(stacked, torch.randn(4, 8, device="cuda"), torch.zeros(4, 3, device="cuda"))
    assert stacked["1.weight"].data_ptr() % 16
    assert all(seen[k] is v for k, v in stacked.items())
    base = layout.unflatten(layout.flatten(model))
    assert base["1.weight"].data_ptr() % 16
    assert SiteVmapEvaluator.try_build(ev, layout).aligned_base(base) is base
