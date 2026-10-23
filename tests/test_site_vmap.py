"""Parity and reach of the site-aware evaluators.

A candidate perturbs one contiguous run of the flat parameter vector, so it differs
from the base inside a single parameter tensor; the vmap evaluator batches only that
tensor and the in-place one writes only it, and each must agree with the path it
replaces.
"""

import warnings
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils._pytree import tree_leaves

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator, SiteInplaceEvaluator, SiteVmapEvaluator, SparseDeltaEvaluator
from polystep.layers import VmapSafeMultiHeadAttention
from polystep.transform import ParamLayout

PDIM = 2


class RepackedLinear(nn.Module):
    """A layer whose forward reads state ``functional_call`` cannot substitute:
    only an in-place write reaches the packed storage.
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        self.phase = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
        self._packed = self.phase.data

    def forward(self, x):
        return x @ torch.cos(self._packed).t()


class NoisyMLP(nn.Module):
    """Draws its own randomness per forward, like thermal phase variation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h + 0.01 * torch.randn_like(h))


class ItemScaledMLP(nn.Module):
    """Reads an activation back as a Python number, which vmap cannot batch, and counts calls."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls += 1
        h = torch.relu(self.fc1(x))
        return self.fc2(h) * self.calls / (1.0 + h.mean().item())


class ResidualMLP(nn.Module):
    """Adds a shortcut in place, as a ResNet block does, so a shortcut site writes into an unbatched activation, and counts calls."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(6, 5)
        self.short = nn.Linear(6, 5)
        self.head = nn.Linear(5, 3)
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls += 1
        h = self.body(x)
        h += self.short(x)
        return self.head(torch.tanh(h)) * self.calls


class LayerScale(nn.Module):
    """Scales its input in place by a learned vector, as timm's LayerScale does."""

    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((width,), 0.5))

    def forward(self, x):
        return x.mul_(self.gamma)


class LayerScaleMLP(nn.Module):
    """vmap refuses its in-place scale only when ``ls.gamma`` alone carries the candidate axis."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.ls = LayerScale(5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc2(torch.tanh(self.ls(self.fc1(x))))


def _small_batch(seed=1, d_in=6, classes=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(7, d_in, generator=gen), torch.randint(0, classes, (7,), generator=gen)


class ConvNet(nn.Module):
    """Outside the Sequential-of-Linear set every other fast path requires."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 4)

    def forward(self, x):
        x = torch.relu(self.norm(self.conv(x)))
        return self.fc(nn.functional.avg_pool2d(x, 2).flatten(1))


class GroupedConvNet(nn.Module):
    """Feeds a depthwise convolution, whose weight has fewer input channels than its input."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.depthwise = nn.Conv2d(4, 8, 3, groups=4, padding=1)
        self.fc = nn.Linear(8 * 4 * 4, 4)

    def forward(self, x):
        x = self.depthwise(torch.relu(self.conv(x)))
        return self.fc(nn.functional.avg_pool2d(torch.relu(x), 2).flatten(1))


def _batch(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(6, 1, 8, 8, generator=gen), torch.randint(0, 4, (6,), generator=gen)


def test_matches_plain_vmap_at_every_site():
    """Every entry the evaluator accepts must score exactly like batching everything."""
    torch.manual_seed(0)
    model = ConvNet()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    loss_fn = nn.CrossEntropyLoss()
    dense = NNCostEvaluator(model, loss_fn)
    evaluator = SiteVmapEvaluator.try_build(dense, layout)
    assert evaluator is not None, "a traceable model must be supported"

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    inputs, targets = _batch()

    checked = 0
    for start in range(0, layout.total_params - PDIM, PDIM):
        entry = evaluator.resolve_site(torch.tensor([start]), PDIM, span=(start, start + PDIM))
        if entry is None:
            continue  # straddles two entries, which is the documented fallback
        values = flat[start : start + PDIM] + torch.tensor([0.31, -0.44])
        got = evaluator.evaluate(
            base_sd,
            entry,
            torch.arange(start - entry.offset, start - entry.offset + PDIM).reshape(1, PDIM),
            values.reshape(1, 1, PDIM),
            inputs,
            targets,
        )
        candidate = flat.clone()
        candidate[start : start + PDIM] = values
        reference = dense.evaluate(layout.batch_unflatten(candidate.unsqueeze(0)), inputs, targets)
        torch.testing.assert_close(got.reshape(-1), reference.reshape(-1), rtol=1e-5, atol=1e-6)
        checked += 1

    assert checked > 0, "no site resolved, the test proved nothing"


class SkipMLP(nn.Module):
    """Adds a shortcut out of place, so no chain plan takes it, and ends on a 2-D weight."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 5)
        self.head = nn.Linear(5, 3, bias=False)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.head(h + torch.tanh(self.fc2(h)))


class TransposedReadNet(nn.Module):
    """Reads its weight through ``.t()``, not ``F.linear``."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        return x @ self.fc.weight.t() + self.fc.bias


class StackedVmapNet(nn.Module):
    """Maps ``torch.vmap`` over a stack of weights inside its own forward."""

    def __init__(self):
        super().__init__()
        self.stack = nn.Parameter(torch.randn(3, 5, 6) * 0.3)
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        return self.head(torch.vmap(lambda w: torch.tanh(x @ w.t()))(self.stack).mean(0))


class ClampedWeightNet(nn.Module):
    """Clamps its weight in place before ``F.linear`` reads it."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 5)
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        self.fc.weight.clamp_(-0.2, 0.2)
        return self.head(torch.tanh(self.fc(x)))


class RecurrentNet(nn.Module):
    """Reads one weight at every step, with a candidate-dependent input after the first."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(6, 5)
        self.rec = nn.Linear(5, 5, bias=False)
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        for _ in range(3):
            h = torch.tanh(self.inp(x) + self.rec(h))
        return self.head(h)


class ScriptedLinearNet(nn.Module):
    """Reads its first weight inside a TorchScript module, which no torch function mode sees."""

    def __init__(self):
        super().__init__()
        self.fc = torch.jit.script(nn.Linear(6, 5))
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        return self.head(torch.tanh(self.fc(x)))


class WeightProduct(nn.Module):
    """Multiplies its input by the transpose of a weight it gets as an argument."""

    def forward(self, h, w):
        return h @ w.t()


class ScriptedHelperNet(nn.Module):
    """Reads its Linear weight through the Linear and again inside a TorchScript module."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.helper = torch.jit.script(WeightProduct())
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        h = torch.tanh(self.fc(x))
        return self.head(h + self.helper(h, self.fc.weight))


def weight_product(h, w):
    return h @ w.t()


class ScriptedFunctionNet(nn.Module):
    """Reads its Linear weight through the Linear and again inside a scripted function attribute."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.fn = torch.jit.script(weight_product)
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        h = torch.tanh(self.fc(x))
        return self.head(h + self.fn(h, self.fc.weight))


scripted_weight_product = torch.jit.script(weight_product)


def numpy_weight_product(h, w):
    with torch._C.DisableTorchFunction():
        w = torch.from_numpy(w.numpy())
    return h @ w.t()


@torch.jit.script
def tail_weight_product(h, w):
    return h @ torch.cat([torch.zeros_like(w[:3]), w[3:]]).t()


def wide_batch_product(h, w):
    return numpy_weight_product(h, w) if h.shape[0] > 7 else torch.zeros_like(h)


class ClassScriptedNet(nn.Module):
    """Reads its Linear weight again inside a scripted function held as a class attribute."""

    fn = scripted_weight_product

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        h = torch.tanh(self.fc(x))
        return self.head(h + self.fn(h, self.fc.weight))


class HiddenReadNet(nn.Module):
    """Reads its Linear weight again through a function kept in a list, out of reach of any scan."""

    def __init__(self, read):
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.head = nn.Linear(6, 3)
        self.reads = [read]

    def forward(self, x):
        h = torch.tanh(self.fc(x))
        return self.head(h + self.reads[0](h, self.fc.weight))


class InfeasibleWeightNet(nn.Module):
    """Scores infinite once its zero Linear weight moves, read out of reach of interception."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 3)
        nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        with torch._C.DisableTorchFunction():
            penalty = torch.where(self.fc.weight.any(), torch.inf, 0.0)
        return self.fc(x) + penalty


class AutocastNet(nn.Module):
    """Runs its Linear layers under its own BF16 autocast."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 5)
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return self.head(torch.tanh(self.fc(x))).float()


class SineTailNet(nn.Module):
    """Makes a hundred elementwise torch calls after its only Linear."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x):
        h = self.fc(x)
        for _ in range(100):
            h = h.sin()
        return h


class EncoderNet(nn.Module):
    """Runs ``nn.TransformerEncoderLayer`` between two Linear layers."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(3, 4)
        self.block = nn.TransformerEncoderLayer(4, 2, dim_feedforward=4, dropout=0.0, batch_first=True)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.block(self.inp(x.unflatten(1, (2, 3)))).mean(1))


def _injected_and_materialized(
    monkeypatch, model, inputs, targets, dtype=torch.float64, injects=True, loss_fn=nn.CrossEntropyLoss()
):
    """Losses per candidate at every entry, scored as offsets and from built weights, and the evaluator."""
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    model, inputs = model.to(dtype), inputs.to(dtype)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, loss_fn), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    materialize, injected = site._vmap_over_site, []
    monkeypatch.setattr(
        site,
        "_vmap_over_site",
        lambda key, sd, c, *a: injected.append(isinstance(c, tuple)) or materialize(key, sd, c, *a),
    )
    gen = torch.Generator().manual_seed(1)
    got, want = [], []
    for entry in layout.entries:
        flat = base_sd[entry.key].reshape(-1)
        starts = torch.tensor([0, entry.numel - (1 if entry is layout.entries[-1] else PDIM)])
        local_idx = starts.unsqueeze(1) + torch.arange(PDIM)
        noise = torch.randn(2, 3, PDIM, generator=gen, dtype=dtype)
        values = flat[local_idx.clamp(max=entry.numel - 1)].unsqueeze(1) + noise
        got.append(site.evaluate(base_sd, entry, local_idx, values, inputs, targets))
        idx, kept = site._clamp_padding(entry, local_idx, values)
        weights = flat.expand(2, 3, -1).clone().scatter_(2, idx.unsqueeze(1).expand(-1, 3, -1), kept)
        want.append(materialize(entry.key, base_sd, weights.view(6, *entry.shape), inputs, targets))
    assert any(injected) == injects
    return torch.cat(got), torch.cat(want), site


@pytest.mark.parametrize(
    "model_cls, batch, uninjectable",
    [
        (SkipMLP, _small_batch, set()),
        (ConvNet, _batch, set()),
        (GroupedConvNet, _batch, {"depthwise.weight"}),
        (RecurrentNet, _small_batch, set()),
        (StackedVmapNet, _small_batch, {"head.weight", "stack"}),
        (ClampedWeightNet, _small_batch, {"fc.weight"}),
        (ItemScaledMLP, _small_batch, set()),
        (ResidualMLP, _small_batch, set()),
    ],
)
def test_site_injection_matches_materialized_linear(monkeypatch, model_cls, batch, uninjectable):
    torch.manual_seed(0)
    got, want, site = _injected_and_materialized(monkeypatch, model_cls(), *batch())
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert site._uninjectable_sites == uninjectable


@pytest.mark.parametrize("model_cls", [ScriptedLinearNet, ScriptedHelperNet, ScriptedFunctionNet, ClassScriptedNet])
def test_script_module_model_skips_injection(monkeypatch, model_cls):
    torch.manual_seed(0)
    got, want, _ = _injected_and_materialized(monkeypatch, model_cls(), *_small_batch(), injects=False)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


class IdentityEqMeta(type):
    def __eq__(cls, other):
        return cls is other


class IdentityEqLinear(nn.Linear, metaclass=IdentityEqMeta):
    """Has an unhashable class, since its metaclass defines equality without a hash."""


@pytest.mark.parametrize("model_fn, scripted", [(ScriptedLinearNet, True), (lambda: IdentityEqLinear(6, 3), False)])
def test_site_construction_defers_script_scan(model_fn, scripted):
    model = model_fn()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    assert "_has_script_module" not in vars(site)
    assert site._has_script_module is scripted
    assert vars(site)["_has_script_module"] is scripted


@pytest.mark.filterwarnings("ignore:vmap failed:RuntimeWarning")
@pytest.mark.parametrize(
    "model_fn, loss_fn",
    [
        (lambda: HiddenReadNet(scripted_weight_product), nn.CrossEntropyLoss()),
        (lambda: HiddenReadNet(numpy_weight_product), nn.CrossEntropyLoss()),
        (lambda: HiddenReadNet(tail_weight_product), nn.CrossEntropyLoss()),
        (InfeasibleWeightNet, lambda out, y: out.mean()),
    ],
)
@pytest.mark.parametrize("dtype, atol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_hidden_weight_read_disables_injection(monkeypatch, model_fn, loss_fn, dtype, atol):
    torch.manual_seed(0)
    got, want, site = _injected_and_materialized(monkeypatch, model_fn(), *_small_batch(), dtype=dtype, loss_fn=loss_fn)
    torch.testing.assert_close(got, want, rtol=0, atol=atol)
    assert site._uninjectable_sites == {"fc.weight"}


@pytest.mark.filterwarnings("ignore:vmap failed:RuntimeWarning")
@pytest.mark.parametrize(
    "read, calls",
    [(scripted_weight_product, [(0.0, 7), (1.0, 7)]), (wide_batch_product, [(1.0, 7), (1.0, 9)])],
)
def test_later_call_hidden_read_declines_injection(monkeypatch, read, calls):
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    torch.manual_seed(0)
    model = HiddenReadNet(read).double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1], [2, 3]])
    flat = base_sd[entry.key].reshape(-1)
    noise = torch.randn(2, 1, PDIM, dtype=torch.float64)
    for scale, rows in calls:
        values = flat[local_idx].unsqueeze(1).repeat(1, 3, 1)
        values[:, 2:] += scale * noise
        weights = flat.expand(2, 3, -1).clone().scatter_(2, local_idx.unsqueeze(1).expand(-1, 3, -1), values)
        x, y = torch.randn(rows, 6, dtype=torch.float64), torch.randint(0, 3, (rows,))
        got = site.evaluate(base_sd, entry, local_idx, values, x, y)
        want = site._vmap_over_site(entry.key, base_sd, weights.view(6, *entry.shape), x, y)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert site._uninjectable_sites == {"fc.weight"}


def test_forward_counts_cap_input_signatures(monkeypatch):
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    monkeypatch.setattr(cost_nn, "_MAX_INPUT_SIGNATURES", 2)
    torch.manual_seed(0)
    model = SkipMLP().double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1]])
    values = base_sd[entry.key].reshape(-1)[local_idx].unsqueeze(1) + torch.randn(1, 3, PDIM, dtype=torch.float64)
    for rows in (5, 6, 7):
        x, y = torch.randn(rows, 6, dtype=torch.float64), torch.randint(0, 3, (rows,))
        site.evaluate(base_sd, entry, local_idx, values, x, y)
        assert 0 < len(site._forward_calls) <= 2


@pytest.mark.filterwarnings("ignore:vmap failed:RuntimeWarning")
@pytest.mark.parametrize("unread_first", [False, True])
def test_partial_move_hidden_read_declines_injection(monkeypatch, unread_first):
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    torch.manual_seed(0)
    model = HiddenReadNet(tail_weight_product).double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[17, 18]])
    flat = base_sd[entry.key].reshape(-1)
    moves = torch.tensor([[[0.4, 0.0], [0.3, -0.5], [-0.2, 0.0]]], dtype=torch.float64)
    x, y = _small_batch()
    for move in [moves * torch.tensor([1.0, 0.0]), moves] if unread_first else [moves]:
        values = flat[local_idx].unsqueeze(1) + move
        weights = flat.expand(1, 3, -1).clone().scatter_(2, local_idx.unsqueeze(1).expand(-1, 3, -1), values)
        got = site.evaluate(base_sd, entry, local_idx, values, x.double(), y)
        want = site._vmap_over_site(entry.key, base_sd, weights.view(3, *entry.shape), x.double(), y)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert site._uninjectable_sites == {"fc.weight"}


def test_barrier_loss_keeps_injection(monkeypatch):
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    torch.manual_seed(0)
    model = SkipMLP().double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(
        NNCostEvaluator(
            model, lambda out, y: torch.where(out.abs().amax() > 10, torch.inf, nn.functional.cross_entropy(out, y))
        ),
        layout,
    )
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[-1], torch.tensor([[0, 1]])
    flat = base_sd[entry.key].reshape(-1)
    x, y = _small_batch()
    for scale in (100.0, 0.1):
        values = flat[local_idx].unsqueeze(1) + scale * torch.tensor([[[0.0, 0.0], [1.0, -1.0]]], dtype=torch.float64)
        weights = flat.expand(1, 2, -1).clone().scatter_(2, local_idx.unsqueeze(1).expand(-1, 2, -1), values)
        got = site.evaluate(base_sd, entry, local_idx, values, x.double(), y)
        want = site._vmap_over_site(entry.key, base_sd, weights.view(2, *entry.shape), x.double(), y)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
        assert got[1].isinf() == (scale > 1)
    assert site._matched_injections and not site._uninjectable_sites


def test_noisy_forward_keeps_injection(monkeypatch):
    from polystep import cost_nn

    torch.manual_seed(0)
    model = NoisyMLP().double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1]])
    values = base_sd[entry.key].reshape(-1)[local_idx].unsqueeze(1) + torch.tensor([[[0.0], [0.5]]]).double()
    inputs, targets = _small_batch()
    scored = []
    for macs in (cost_nn._MACS_PER_INTERCEPTED_CALL, 0):
        monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", macs)
        torch.manual_seed(1)
        scored.append(site.evaluate(base_sd, entry, local_idx, values, inputs.double(), targets))
    assert site._matched_injections and not site._uninjectable_sites
    torch.testing.assert_close(scored[1], scored[0], rtol=0, atol=1e-12)


def test_integer_loss_keeps_injection(monkeypatch):
    torch.manual_seed(0)
    got, want, site = _injected_and_materialized(
        monkeypatch, SkipMLP(), *_small_batch(), loss_fn=lambda out, y: (out.argmax(1) == y).sum()
    )
    assert got.dtype == torch.int64 and torch.equal(got, want)
    assert site._uninjectable_sites == set()


def test_nonlinear_weight_read_falls_back(monkeypatch):
    torch.manual_seed(0)
    got, want, site = _injected_and_materialized(monkeypatch, TransposedReadNet(), *_small_batch())
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert site._uninjectable_sites == {"fc.weight"}


def test_internal_autocast_site_skips_injection(monkeypatch):
    torch.manual_seed(0)
    got, want, site = _injected_and_materialized(monkeypatch, AutocastNet(), *_small_batch(), dtype=torch.float32)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert site._uninjectable_sites == {"fc.weight", "head.weight"}


def test_long_forward_stops_injecting_small_site():
    from polystep import cost_nn

    torch.manual_seed(0)
    model = SineTailNet()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.MSELoss()), layout)
    rows = cost_nn._MACS_PER_INTERCEPTED_CALL * (cost_nn._INTERCEPT_SETUP_CALLS + 10) // (6 * 512) + 1
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1], [2, 3]])
    values = base_sd[entry.key].reshape(-1)[local_idx].unsqueeze(1) + torch.randn(2, 3, PDIM)
    materialize, injected = site._vmap_over_site, []
    site._vmap_over_site = lambda key, sd, c, *a: injected.append(isinstance(c, tuple)) or materialize(key, sd, c, *a)
    for r in (rows // 2, rows, rows):
        site.evaluate(base_sd, entry, local_idx, values, torch.randn(r, 32), torch.randn(r, 16))
    assert injected == [False, True, False, False, False]


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_injection_estimate_respects_chunk_size(chunk_size):
    from polystep import cost_nn

    torch.manual_seed(0)
    model = SkipMLP().double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=chunk_size), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1], [2, 3]])
    rows = cost_nn._MACS_PER_INTERCEPTED_CALL * cost_nn._INTERCEPT_SETUP_CALLS // (6 * entry.numel) + 1
    values = base_sd[entry.key].reshape(-1)[local_idx].unsqueeze(1) + torch.randn(2, 3, PDIM, dtype=torch.float64)
    inputs, targets = torch.randn(rows, 6, dtype=torch.float64), torch.randint(0, 3, (rows,))
    site.evaluate(base_sd, entry, local_idx, values, inputs, targets)
    assert bool(site._matched_injections) == (chunk_size is None)


def test_tuple_inputs_keep_site_injection(monkeypatch):
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    torch.manual_seed(0)
    model = SkipMLP().double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1], [2, 3]])
    values = base_sd[entry.key].reshape(-1)[local_idx].unsqueeze(1) + torch.randn(2, 3, PDIM, dtype=torch.float64)
    inputs, targets = _small_batch()
    got = site.evaluate(base_sd, entry, local_idx, values, (inputs.double(),), targets)
    assert site._matched_injections and not site._uninjectable_sites
    torch.testing.assert_close(got, site.evaluate(base_sd, entry, local_idx, values, inputs.double(), targets))


def test_zero_offset_conv_chunk_scores_without_raising(monkeypatch):
    from polystep import cost_nn

    monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
    torch.manual_seed(0)
    model = ConvNet().double()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    entry, local_idx = layout.entries[0], torch.tensor([[0, 1]])
    values = base_sd[entry.key].reshape(-1)[local_idx].unsqueeze(1).expand(1, 3, PDIM).clone()
    materialize, injected = site._vmap_over_site, []
    site._vmap_over_site = lambda key, sd, c, *a: injected.append(isinstance(c, tuple)) or materialize(key, sd, c, *a)
    inputs, targets = _batch()
    got = site.evaluate(base_sd, entry, local_idx, values, inputs.double(), targets)
    assert got.isfinite().all() and torch.equal(got, got[:1].expand_as(got))
    assert injected == [True, False, True]


def test_scalar_input_site_gate_skips_crash():
    torch.manual_seed(0)
    model = nn.Embedding(4, 3)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    flat = layout.flatten(model).reshape(-1)
    values = flat[3:5] + torch.tensor([[0.3, -0.2], [-0.4, 0.1]])
    inputs, targets = torch.tensor(1), torch.zeros(3)
    got = site.evaluate(
        layout.unflatten(flat), layout.entries[0], torch.tensor([[3, 4]]), values[None], inputs, targets
    )
    candidates = flat.expand(2, -1).clone()
    candidates[:, 3:5] = values
    torch.testing.assert_close(got, evaluator.evaluate(layout.batch_unflatten(candidates), inputs, targets))


def test_sites_feeding_attention_skip_injection(monkeypatch):
    torch.manual_seed(0)
    got, want, site = _injected_and_materialized(monkeypatch, EncoderNet(), *_small_batch())
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)
    assert site._uninjectable_sites == {
        "inp.weight",
        "block.self_attn.in_proj_weight",
        "block.self_attn.out_proj.weight",
    }


def test_conv_step_skips_closure_keeps_losses():
    """The path has to reach a model no other fast path accepts, and not change it."""

    def build():
        torch.manual_seed(0)
        model = ConvNet()
        inputs, targets = _batch()
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        calls = []

        def closure(batched_params):
            calls.append(1)
            return evaluator.evaluate(batched_params, inputs, targets)

        torch.manual_seed(7)
        opt = PolyStepOptimizer(model, epsilon=0.3, step_radius=0.05, seed=3, particle_dim=PDIM)
        return opt, evaluator, inputs, targets, closure, calls

    opt_dense, _, _, _, dense_closure, dense_calls = build()
    dense = [opt_dense.step(dense_closure) for _ in range(2)]

    opt_fast, evaluator, inputs, targets, fast_closure, fast_calls = build()
    opt_fast.register_evaluator(evaluator, inputs, targets)
    assert opt_fast._sparse_delta_evaluator is None, "a conv model must be outside the sparse-delta set"
    assert opt_fast._site_vmap_evaluator is not None, "the site path must be available, or this proves nothing"
    fast = [opt_fast.step(fast_closure) for _ in range(2)]

    assert dense_calls, "the dense run must actually call the closure"
    assert not fast_calls, "every chunk should have resolved a site"
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=1e-5, atol=1e-6)


def test_site_vmap_honors_evaluator_chunk_size(monkeypatch):
    from polystep import cost_nn

    chunks = []
    real_vmap = cost_nn.vmap

    def spy(fn, *args, **kwargs):
        if "_vmap_over_site" in fn.__qualname__:
            chunks.append(kwargs.get("chunk_size"))
        return real_vmap(fn, *args, **kwargs)

    monkeypatch.setattr(cost_nn, "vmap", spy)
    torch.manual_seed(0)
    model = ConvNet()
    inputs, targets = _batch()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), chunk_size=4, use_inplace=False)
    opt = PolyStepOptimizer(model, seed=0, particle_dim=PDIM)
    opt.register_evaluator(evaluator, inputs, targets)
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert chunks and set(chunks) == {4}


class TiedHead(nn.Module):
    """Reads its embedding table again as the output projection."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(24, 24)
        self.out = nn.Linear(24, 24)
        self.out.weight = self.emb.weight

    def forward(self, x):
        return self.out(torch.tanh(self.emb(x)))


@pytest.mark.parametrize("path", ["injected", "stacked", "hybrid"])
@pytest.mark.parametrize("tie", ["attribute", "module"])
def test_tied_site_vmap_matches_dense_costs(monkeypatch, tie, path):
    """Every alias of a tied site reads the candidate, with or without the alias keys in the base."""
    from polystep import HybridSubspace, cost_nn

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    if tie == "attribute":
        model, inputs = TiedHead().double(), torch.randint(0, 24, (7,), generator=gen)
    else:
        shared = nn.Linear(24, 24)
        model = nn.Sequential(shared, nn.Tanh(), shared).double()
        inputs = torch.randn(7, 24, generator=gen, dtype=torch.float64)
    targets = torch.randint(0, 24, (7,), generator=gen)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False), layout)
    assert layout.shared_groups and site is not None
    aliases = {a: e.key for e in layout.entries for a in e.shared_with}
    flat = layout.flatten(model).reshape(-1)
    base_sd = layout.unflatten(flat)
    if path == "hybrid":
        subspace = HybridSubspace.from_layout(layout, rank=3)
        projections = subspace.init_projections(torch.device("cpu"), torch.float64)
        vec = 0.1 * torch.randn(subspace.subspace_dim, generator=gen, dtype=torch.float64)
        to_sd = partial(subspace.apply_perturbation, projections, base_sd)
    else:
        vec, to_sd = flat, layout.unflatten
    candidates = vec.expand(2, -1).clone()
    candidates[:, :PDIM] += 0.3 * torch.randn(2, PDIM, generator=gen, dtype=torch.float64)
    values = candidates[:, :PDIM].reshape(1, 2, PDIM)
    if path == "hybrid":
        got = site.evaluate_subspace(
            projections, base_sd, vec, to_sd(vec), subspace.specs[0], 0, values, inputs, targets
        )
    else:
        if path == "stacked":
            site._uninjectable_sites.add(layout.entries[0].key)
        else:
            monkeypatch.setattr(cost_nn, "_MACS_PER_INTERCEPTED_CALL", 0)
        entry = site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM))
        got = site.evaluate(base_sd, entry, torch.arange(PDIM).reshape(1, PDIM), values, inputs, targets)
        if path == "injected":
            assert site._uninjectable_sites == ({entry.key} if tie == "attribute" else set())
            assert bool(site._matched_injections) == (tie == "module")

    expected = []
    for candidate in candidates:
        sd = to_sd(candidate)
        sd = {**sd, **{a: sd[k] for a, k in aliases.items()}}
        output = torch.func.functional_call(model, sd, (inputs,), tie_weights=False)
        expected.append(nn.functional.cross_entropy(output, targets))
    torch.testing.assert_close(got, torch.stack(expected), rtol=0, atol=1e-12)


@pytest.mark.parametrize("forced", [True, False])
def test_inplace_site_accepts_tied_weights(forced):
    """Writing the shared storage moves every alias, so in place a tied weight scores
    exactly like the full swap, whether in-place was forced or picked for memory;
    a per-sample evaluator is declined.
    """
    torch.manual_seed(0)
    shared = nn.Linear(6, 6)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True if forced else None)
    evaluator._use_inplace = True
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    assert isinstance(site, SiteInplaceEvaluator)
    per_sample = NNCostEvaluator(model, nn.MSELoss(reduction="none"), use_inplace=True, per_sample=True)
    assert SiteVmapEvaluator.try_build(per_sample, layout) is None

    flat = layout.flatten(model).reshape(-1)
    inputs, targets = torch.randn(7, 6), torch.randn(7, 6)
    values = flat[0:PDIM] + torch.tensor([[0.3, -0.2], [-0.4, 0.1]])
    got = site.evaluate(
        layout.unflatten(flat),
        site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
        torch.arange(PDIM).reshape(1, PDIM),
        values.reshape(1, 2, PDIM),
        inputs,
        targets,
    )
    candidates = flat.expand(2, -1).clone()
    candidates[:, 0:PDIM] = values
    assert torch.equal(got, evaluator.evaluate(layout.batch_unflatten(candidates), inputs, targets))


def test_sparse_delta_matches_site_on_mlp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    loss_fn = nn.CrossEntropyLoss()
    sparse = SparseDeltaEvaluator.try_build(model, loss_fn, layout)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, loss_fn), layout)
    assert sparse is not None and site is not None

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    gen = torch.Generator().manual_seed(1)
    inputs = torch.randn(7, 6, generator=gen)
    targets = torch.randint(0, 3, (7,), generator=gen)

    entry = site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM))
    values = flat[0:PDIM] + torch.tensor([0.2, -0.3])
    local = torch.arange(PDIM).reshape(1, PDIM)
    from_sparse = sparse.evaluate(base_sd, entry.key, local, values.reshape(1, 1, PDIM), inputs, targets)
    from_site = site.evaluate(base_sd, entry, local, values.reshape(1, 1, PDIM), inputs, targets)
    torch.testing.assert_close(from_sparse.reshape(-1), from_site.reshape(-1), rtol=1e-5, atol=1e-6)


def test_conv_subspace_step_matches_materializing_path():
    """Per-layer blocks map to one parameter, so the site argument holds in
    coordinate space; the offset is relative to the barycentre the weights carry.
    """
    from polystep import HybridSubspace

    def build(register):
        torch.manual_seed(0)
        model = ConvNet()
        inputs, targets = _batch()
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        torch.manual_seed(7)
        opt = PolyStepOptimizer(
            model,
            epsilon=0.5,
            step_radius=1.0,
            seed=3,
            subspace=HybridSubspace.auto_from_layout(ParamLayout.from_module(model)),
        )
        if register:
            opt.register_evaluator(evaluator, inputs, targets)
        return opt, [opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets)) for _ in range(4)]

    _, dense = build(register=False)
    opt_fast, fast = build(register=True)
    assert opt_fast._subspace_delta_evaluator is None, "a conv model must be outside the delta set"
    assert opt_fast._site_vmap_evaluator is not None, "the site path must be available, or this proves nothing"
    # Later steps are what catch a double-counted barycentre; step 0 starts at zero.
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=1e-5, atol=1e-5)


def test_stateless_paths_score_repacked_model_blind():
    """On a repacked model the stateless paths score every candidate blind: one
    loss for all candidates, and no error. The in-place site path writes the packed
    storage, so it tells them apart.
    """
    torch.manual_seed(0)
    model = nn.Sequential(RepackedLinear(6, 5), nn.ReLU(), RepackedLinear(5, 3))
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    assert site is not None, "the model must reach the site path, or this proves nothing"

    flat = layout.flatten(model).reshape(-1)
    entry = site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM))
    values = torch.stack([flat[0:PDIM], flat[0:PDIM] + 5.0])  # wildly different candidates
    blind = site.evaluate(
        {k: v.detach() for k, v in model.state_dict().items()},
        entry,
        torch.arange(PDIM).reshape(1, PDIM),
        values.reshape(1, 2, PDIM),
        inputs,
        targets,
    )
    torch.testing.assert_close(blind[0], blind[1])

    inplace = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), layout)
    seeing = inplace.evaluate(
        layout.unflatten(flat),
        entry,
        torch.arange(PDIM).reshape(1, PDIM),
        values.reshape(1, 2, PDIM),
        inputs,
        targets,
    )
    assert seeing[0] != seeing[1]


@pytest.mark.parametrize(
    "subspace, block_strategy", [(False, "monolithic"), (True, "monolithic"), (False, "per_layer")]
)
def test_forced_inplace_keeps_only_inplace_site(subspace, block_strategy):
    """``use_inplace=True`` is a correctness contract: the stateless paths score
    through ``functional_call`` and must step aside, in full space and subspace,
    leaving only the site path that writes the real weights. It is built once, and
    a sweep, monolithic or block-wise, lets go of the incumbent it wrote.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    inputs, targets = _small_batch()
    kwargs = (
        {"subspace": HybridSubspace.auto_from_layout(ParamLayout.from_module(model))}
        if subspace
        else {"particle_dim": PDIM, "block_strategy": block_strategy}
    )

    free = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, **kwargs)
    free.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss()), inputs, targets)
    armed = (free._subspace_delta_evaluator or free._site_vmap_evaluator) if subspace else free._sparse_delta_evaluator
    assert armed is not None, "a stateless path must arm here, or this proves nothing"

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, **kwargs)
    forced = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    opt.register_evaluator(forced, inputs, targets)
    site = opt._site_vmap_evaluator
    assert isinstance(site, SiteInplaceEvaluator)
    assert opt._sparse_delta_evaluator is None
    assert opt._subspace_delta_evaluator is None
    assert opt._factored_evaluator is None
    opt.register_evaluator(forced, inputs, targets)
    assert opt._site_vmap_evaluator is site
    opt.step(lambda bp: forced.evaluate(bp, inputs, targets))
    assert site._primed is None


@pytest.mark.parametrize("kind, width", [(nn.Embedding, 12), (nn.EmbeddingBag, 4)])
def test_inplace_max_norm_embedding_matches_default(kind, width):
    """A ``max_norm`` lookup renormalizes rows inside the forward; that write must not
    carry from one candidate into the next.
    """
    gen = torch.Generator().manual_seed(1)
    inputs, targets = torch.randint(0, 6, (16, 3), generator=gen), torch.randint(0, 3, (16,), generator=gen)
    weights = []
    for inplace in (True, None):
        torch.manual_seed(0)
        embedding = kind(6, 4, max_norm=1.0)
        with torch.no_grad():
            embedding.weight.mul_(3.0 / embedding.weight.norm(dim=1, keepdim=True))
        model = nn.Sequential(embedding, nn.Flatten(), nn.Linear(width, 3))
        opt = PolyStepOptimizer(model, solver="softmax", seed=0, epsilon=0.3)
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=inplace, compile_forward=False)
        for _ in range(3):
            opt.register_evaluator(evaluator, inputs, targets)
            opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
        weights.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
    torch.testing.assert_close(weights[0], weights[1], atol=1e-5, rtol=0)


def test_inplace_raise_restores_max_norm_rows():
    """A step that raises mid-sweep must leave every renormalized row as it was."""
    model = nn.Embedding(2, 4, max_norm=1.0)
    with torch.no_grad():
        model.weight.copy_(torch.arange(2.0, 10.0).reshape(2, 4))
    before = model.weight.detach().clone()
    calls = []

    def loss(output):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("failed loss")
        return output.square().mean()

    opt = PolyStepOptimizer(model, solver="softmax", particle_dim=2, seed=0)
    evaluator = NNCostEvaluator(model, loss, use_inplace=True, compile_forward=False)
    inputs = torch.tensor([0, 1])
    opt.register_evaluator(evaluator, inputs)
    with pytest.raises(RuntimeError, match="failed loss"):
        opt.step(lambda p: evaluator.evaluate(p, inputs))
    assert torch.equal(model.weight, before)


def test_site_max_norm_keeps_incumbent():
    torch.manual_seed(0)
    embedding = nn.Embedding(6, 4, max_norm=1.0)
    with torch.no_grad():
        embedding.weight.mul_(3.0 / embedding.weight.norm(dim=1, keepdim=True))
    model = nn.Sequential(embedding, nn.Flatten(), nn.Linear(12, 3))
    gen = torch.Generator().manual_seed(1)
    inputs, targets = torch.randint(0, 6, (16, 3), generator=gen), torch.randint(0, 3, (16,), generator=gen)
    opt = PolyStepOptimizer(model, solver="softmax", seed=0, step_radius=0.0)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_forward=False)
    before = opt.state.X.clone()
    opt.register_evaluator(evaluator, inputs, targets)
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert torch.equal(opt.state.X, before)


def test_unforced_evaluator_swap_rearms_fast_paths():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    loss_fn = nn.CrossEntropyLoss()  # one instance, so the cache key would not move
    inputs, targets = _small_batch()

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, particle_dim=PDIM)
    opt.register_evaluator(NNCostEvaluator(model, loss_fn, use_inplace=True), inputs, targets)
    assert opt._sparse_delta_evaluator is None
    opt.register_evaluator(NNCostEvaluator(model, loss_fn), inputs, targets)
    assert opt._sparse_delta_evaluator is not None or opt._site_vmap_evaluator is not None


def test_forced_inplace_disarms_factored_low_rank():
    """The factored low-rank path is stateless too."""
    from polystep import FactoredSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    inputs, targets = _small_batch()
    subspace = FactoredSubspace.from_layout(ParamLayout.from_module(model), rank=2)

    free = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, subspace=subspace)
    free.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss()), inputs, targets)
    assert free._factored_evaluator is not None, "the low-rank path must arm, or this proves nothing"

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=1.0, seed=3, subspace=subspace)
    with pytest.warns(UserWarning, match="low-rank"):
        opt.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), inputs, targets)
    assert opt._factored_evaluator is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inplace_site_matches_full_swap(dtype):
    """Writing only a candidate's entries scores exactly what swapping in all of its
    weights scores, and the model ends holding the incumbent.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3)).to(dtype)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    flat = layout.flatten(model).reshape(-1)
    base_sd = layout.unflatten(flat.clone())
    inputs, targets = _small_batch()

    gen = torch.Generator().manual_seed(2)
    checked = 0
    for start in range(0, layout.total_params - 2 * PDIM, 2 * PDIM):
        entry = site.resolve_site(torch.tensor([start]), 2 * PDIM, span=(start, start + 2 * PDIM))
        if entry is None:
            continue
        values = (flat[start : start + 2 * PDIM].view(2, 1, PDIM) + 0.3 * torch.randn(2, 3, PDIM, generator=gen)).to(
            dtype
        )
        local = torch.arange(start, start + 2 * PDIM).view(2, PDIM) - entry.offset
        got = site.evaluate(base_sd, entry, local, values, inputs, targets)
        candidates = flat.expand(6, -1).clone()
        for g in range(2):
            candidates[3 * g : 3 * g + 3, start + g * PDIM : start + (g + 1) * PDIM] = values[g]
        assert torch.equal(got, evaluator.evaluate(layout.batch_unflatten(candidates), inputs, targets))
        checked += 1
    assert checked > 0, "no site resolved, the test proved nothing"
    torch.testing.assert_close(layout.flatten(model).reshape(-1), flat, rtol=0, atol=0)


def test_inplace_site_keeps_float64_resolution():
    """FP32 loss buffers collapse candidates that differ below FP32 resolution."""
    model = nn.Sequential(nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
    layout = ParamLayout.from_module(model, particle_dim=1)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.MSELoss(), use_inplace=True), layout)
    bias = torch.tensor([1.0, 1.0 + 1e-9, 1.0 + 2e-9], dtype=torch.float64)

    losses = site.evaluate(
        layout.unflatten(layout.flatten(model).reshape(-1)),
        site.resolve_site(torch.tensor([2]), 1, span=(2, 3)),
        torch.zeros(1, 1, dtype=torch.long),
        bias.view(1, 3, 1),
        torch.ones(1, 2, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
    )
    assert losses.dtype == torch.float64
    assert len(torch.unique(losses)) == 3


@pytest.mark.parametrize("use_inplace", [True, False])
def test_site_loss_aliasing_buffer_survives_restore(use_inplace):
    class CachedScore(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("score", torch.zeros(()))

        def forward(self, x):
            self.score.copy_((x * self.weight).sum())
            return self.score

    model = CachedScore()
    layout = ParamLayout.from_module(model, particle_dim=2)
    evaluator = NNCostEvaluator(model, lambda out: out, use_inplace=use_inplace, compile_forward=False)
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        losses = site.evaluate(
            layout.unflatten(layout.flatten(model)),
            layout.entries[0],
            torch.tensor([[0, 1]]),
            torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]),
            torch.ones(2),
        )
    assert losses.tolist() == [2.0, 4.0]
    assert model.score.item() == 0


@pytest.mark.parametrize("projection", ["dense", "sparse", "unprojected"])
def test_inplace_subspace_site_matches_full_swap(projection):
    """Rewriting one spec's parameter from the incumbent, at a column past the spec's
    first, matches reconstructing every weight from the base. Dropout stays off while
    scoring, and the model ends holding the incumbent in train mode.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 4))
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=3, sparse_threshold_bytes=0 if projection == "sparse" else 10**9)
    projections = subspace.init_projections(torch.device("cpu"), torch.float32)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    site = SiteVmapEvaluator.try_build(evaluator, layout)
    model.train()
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gen = torch.Generator().manual_seed(1)
    coords = 0.1 * torch.randn(subspace.subspace_dim, generator=gen)
    bary_sd = subspace.apply_perturbation(projections, base_sd, coords)
    inputs, targets = torch.randn(5, 24, generator=gen), torch.randint(0, 4, (5,), generator=gen)

    if projection == "unprojected":
        spec = next(s for s in subspace.specs if not s.is_projected)
    else:
        spec = next(
            s
            for s in subspace.specs
            if s.is_projected and isinstance(projections[s.entry_key], torch.Tensor) is (projection == "dense")
        )
    col = 2 * PDIM
    start = spec.flat_start + col
    assert site.resolve_spec(subspace, start, start + 2 * PDIM, subspace.subspace_dim) is spec
    values = coords[start : start + 2 * PDIM].view(2, 1, PDIM) + 0.3 * torch.randn(2, 3, PDIM, generator=gen)
    got = site.evaluate_subspace(projections, base_sd, coords, bary_sd, spec, col, values, inputs, targets)
    for key, weight in bary_sd.items():
        torch.testing.assert_close(model.state_dict()[key], weight, rtol=0, atol=0)
    assert model.training

    full = coords.expand(6, -1).clone()
    for g in range(2):
        full[3 * g : 3 * g + 3, start + g * PDIM : start + (g + 1) * PDIM] = values[g]
    with torch.no_grad():
        expected = evaluator.evaluate_subspace_inplace(subspace, projections, base_sd, full, inputs, targets)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_inplace_site_declines_unwritable_sites():
    """A non-contiguous parameter has no flat view to write into, and an FP16/BF16
    subspace weight must be rebuilt from the base, so those sites take the dense path
    while the others stay in place.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    model[0].weight.data = model[0].weight.data.t().contiguous().t()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), layout)
    assert site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)) is None
    later = layout.entries[2].offset
    assert site.resolve_site(torch.tensor([later]), PDIM, span=(later, later + PDIM)) is not None
    subspace = HybridSubspace.from_layout(layout, rank=1)
    bias = subspace.specs[1].flat_start
    assert site.resolve_spec(subspace, 0, 1, subspace.subspace_dim) is None
    assert site.resolve_spec(subspace, bias, bias + 1, subspace.subspace_dim) is not None

    for dtype, accepted in ((torch.float32, True), (torch.bfloat16, False)):
        model = nn.Sequential(nn.Linear(6, 5)).to(dtype)
        layout = ParamLayout.from_module(model)
        subspace = HybridSubspace.from_layout(layout, rank=1)
        site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True), layout)
        assert (site.resolve_spec(subspace, 0, 1, subspace.subspace_dim) is not None) is accepted


@pytest.mark.parametrize("back", [torch.bfloat16, torch.float32])
def test_half_inplace_step_builds_no_incumbent(monkeypatch, back):
    """The in-place site declines every FP16/BF16 spec, so a step whose specs are all
    BF16 must not build an incumbent weight set that nothing reads, whatever dtype the
    buffers carry. An FP32 spec is still written in place and reads the incumbent."""
    from polystep import HybridSubspace

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = nn.Sequential(nn.Linear(6, 5), nn.BatchNorm1d(5), nn.Tanh()).bfloat16()
            self.back = nn.Linear(5, 3).to(back)

        def forward(self, x):
            return self.back(self.front(x.bfloat16()).to(back))

    torch.manual_seed(0)
    model = Model()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    opt = PolyStepOptimizer(model, seed=0, subspace=subspace)
    subspace = opt.subspace
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
    inputs, targets = _small_batch()
    with pytest.warns(UserWarning, match="use_inplace=True asks"):
        opt.register_evaluator(evaluator, inputs, targets)
    assert isinstance(opt._site_vmap_evaluator, SiteInplaceEvaluator)
    calls = []
    build = subspace.reconstruct_batch
    monkeypatch.setattr(subspace, "reconstruct_batch", lambda *a: calls.append(a) or build(*a))
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert bool(calls) == (back is torch.float32)


def test_unbatchable_forward_falls_back_to_loop():
    """An ``.item()`` on a batched activation makes vmap fail; the site path must
    fall back to a sequential loop, not crash.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = ItemScaledMLP()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()

    dense = NNCostEvaluator(model, nn.CrossEntropyLoss())
    site = SiteVmapEvaluator.try_build(dense, layout)
    args = (
        {k: v.detach() for k, v in model.state_dict().items()},
        site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
        torch.arange(PDIM).reshape(1, PDIM),
        layout.flatten(model).reshape(-1)[0:PDIM].expand(1, 2, PDIM),
        inputs,
        targets,
    )

    with pytest.warns(RuntimeWarning, match="site-aware"):
        losses = site.evaluate(*args)
    assert losses.shape == (2,) and torch.isfinite(losses).all()
    assert losses[0] == losses[1] and model.calls.item() == 0

    # Retired, so the step stops offering it chunks rather than warning on every one.
    assert not dense._vmap_failed
    assert site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)) is None
    later = layout.entries[2].offset
    assert site.resolve_site(torch.tensor([later]), PDIM, span=(later, later + PDIM)) is not None
    subspace = HybridSubspace.from_layout(layout, rank=1)
    kept = subspace.specs[2]
    assert site.resolve_spec(subspace, 0, 1, subspace.subspace_dim) is None
    assert site.resolve_spec(subspace, kept.flat_start, kept.flat_start + 1, subspace.subspace_dim) is kept


def test_site_forward_bug_raises_without_retiring():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 2)

        def forward(self, x):
            return self.fc(x) + torch.ones(7)

    torch.manual_seed(0)
    model = Net()
    inputs, targets = _small_batch()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, seed=3, particle_dim=PDIM)
    opt.register_evaluator(evaluator, inputs, targets)
    with pytest.raises(RuntimeError, match="must match"):
        opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    site = opt._site_vmap_evaluator
    assert not site._failed_sites and not site._owner._vmap_failed


def _float64_step_losses(model_cls, register):
    """Two float64 steps' losses, with the evaluator registered or closure-only, and the optimizer."""
    torch.manual_seed(0)
    model = model_cls().double()
    inputs, targets = _small_batch()
    inputs = inputs.double()
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, seed=3, particle_dim=PDIM)
    if register:
        opt.register_evaluator(evaluator, inputs, targets)
    return [opt.step(lambda p: evaluator.evaluate(p, inputs, targets)) for _ in range(2)], opt


def test_inplace_residual_site_matches_dense():
    with pytest.warns(RuntimeWarning, match="site-aware"):
        fast, opt = _float64_step_losses(ResidualMLP, register=True)
    assert not opt._site_vmap_evaluator._owner._vmap_failed
    torch.testing.assert_close(fast, _float64_step_losses(ResidualMLP, register=False)[0], rtol=0, atol=1e-12)


def test_site_failure_keeps_dense_vmap():
    with pytest.warns(RuntimeWarning, match="'ls.gamma'"):
        fast, opt = _float64_step_losses(LayerScaleMLP, register=True)
    site = opt._site_vmap_evaluator
    assert site._failed_sites == {"ls.gamma"} and not site._owner._vmap_failed
    torch.testing.assert_close(fast, _float64_step_losses(LayerScaleMLP, register=False)[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("compile_vmap", [False, True])
def test_noisy_forward_stays_batched(compile_vmap):
    """vmap refuses a per-forward noise draw under randomness='error'; both paths
    retry with a draw per candidate and stay batched, compiled or not, so identical
    candidates score differently.
    """
    torch.manual_seed(0)
    model = NoisyMLP()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()
    flat = layout.flatten(model).reshape(-1)

    dense = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_vmap=compile_vmap)
    site = SiteVmapEvaluator.try_build(dense, layout)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="vmap failed|compile_vmap")
        from_site = site.evaluate(
            {k: v.detach() for k, v in model.state_dict().items()},
            site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
            torch.arange(PDIM).reshape(1, PDIM),
            flat[0:PDIM].expand(1, 2, PDIM),
            inputs,
            targets,
        )
        from_dense = dense.evaluate(layout.batch_unflatten(flat.expand(2, -1)), inputs, targets)

    assert not dense._vmap_failed and not dense._compile_failed and not site._compile_failed_sites
    assert site._randomness == dense._randomness == "different"
    assert from_site[0] != from_site[1]
    assert from_dense[0] != from_dense[1]


class CountingNoisyMLP(NoisyMLP):
    """Advances a buffer before its noise draw, so a refused forward has written it."""

    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, x):
        self.calls.add_(1)
        return super().forward(x) * self.calls


def test_noisy_retry_rewinds_refused_forward_buffers():
    torch.manual_seed(0)
    model = CountingNoisyMLP()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    inputs, targets = _small_batch()
    flat = layout.flatten(model).reshape(-1)
    params = layout.batch_unflatten(torch.stack([flat, flat + 0.5]))

    def direct(p):
        out = torch.func.functional_call(model, {**p, "calls": model.calls}, (inputs,))
        return nn.functional.cross_entropy(out, targets)

    torch.manual_seed(1)
    expected = torch.func.vmap(direct, randomness="different")(params)
    model.calls.zero_()

    dense = NNCostEvaluator(model, nn.CrossEntropyLoss())
    site = SiteVmapEvaluator.try_build(dense, layout)
    torch.manual_seed(1)
    torch.testing.assert_close(dense.evaluate(params, inputs, targets), expected)
    assert model.calls.item() == 0

    site.evaluate(
        {k: v.detach() for k, v in model.state_dict().items() if k != "calls"},
        site.resolve_site(torch.tensor([0]), PDIM, span=(0, PDIM)),
        torch.arange(PDIM).reshape(1, PDIM),
        flat[0:PDIM].expand(1, 2, PDIM),
        inputs,
        targets,
    )
    assert model.calls.item() == 0


def _dense_reference(projections, bary_sd, spec, col_start, dcoords):
    """The full-width formulation the block-diagonal bmm replaces."""
    n_groups, n_cand, pdim = dcoords.shape
    n = n_groups * n_cand
    wide = dcoords.new_zeros(n, spec.flat_end - spec.flat_start)
    cols = torch.arange(col_start, col_start + n_groups * pdim)
    cols = cols.reshape(n_groups, 1, pdim).expand(n_groups, n_cand, pdim).reshape(n, pdim)
    wide.scatter_(1, cols, dcoords.reshape(n, pdim))
    base = bary_sd[spec.entry_key]
    P = projections[spec.entry_key] if spec.is_projected else None
    if P is None:
        delta = wide
    elif isinstance(P, torch.Tensor):
        delta = wide.to(P.dtype) @ P.t()
    else:
        delta = P.project(wide.to(P.dtype))
    return (base.reshape(1, -1) + delta.to(base.dtype)).reshape(-1, *base.shape)


@pytest.mark.parametrize(("dtype", "sparse_dtype"), [(torch.float32, torch.float64), (torch.bfloat16, torch.float32)])
@pytest.mark.parametrize("projection_type", ["dense", "sparse", "unprojected"])
def test_block_diagonal_correction_matches_dense(projection_type, dtype, sparse_dtype):
    """Only pdim columns per group are nonzero, so building the site from those
    columns alone, past the spec's first column and around a nonzero incumbent, must
    match the full-width ``dcoords @ P.t()``: a bmm for a dense projection, a
    scatter-add for a sparse one or an unprojected spec. A BF16 base is rebuilt from
    ``base_sd`` and the full coordinates while the reference offsets the rounded
    incumbent, so the two round differently and match at BF16 tolerance. Neither
    path redraws a sparse projection's cache, including one held at FP64 under FP32
    coordinates.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, 4)).to(dtype)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    subspace = HybridSubspace.from_layout(
        layout, rank=3, sparse_threshold_bytes=0 if projection_type == "sparse" else 10**9
    )
    projections = subspace.init_projections(torch.device("cpu"), dtype)

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    assert site is not None
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if projection_type == "unprojected":
        spec = next(s for s in subspace.specs if not s.is_projected)
    else:
        spec = next(
            s
            for s in subspace.specs
            if s.is_projected and isinstance(projections[s.entry_key], torch.Tensor) is (projection_type == "dense")
        )

    col = 2 * PDIM
    n_groups = (spec.flat_end - spec.flat_start - col) // PDIM
    gen = torch.Generator().manual_seed(1)
    coords = 0.1 * torch.randn(subspace.subspace_dim, generator=gen)
    dcoords = torch.randn(n_groups, 3, PDIM, generator=gen)
    values = dcoords + coords.narrow(0, spec.flat_start + col, n_groups * PDIM).view(n_groups, 1, PDIM)
    inputs = torch.randn(5, 24, generator=gen)
    targets = torch.randint(0, 4, (5,), generator=gen)

    bary_sd = subspace.apply_perturbation(projections, base_sd, coords)
    P = projections.get(spec.entry_key)
    if projection_type == "sparse":
        P.project(torch.zeros(spec.num_coords, dtype=sparse_dtype))
    expected = site._vmap_over_site(
        spec.entry_key,
        bary_sd,
        _dense_reference(projections, bary_sd, spec, col, dcoords),
        inputs,
        targets,
    )
    held = [P._rows, P._values, P._csr_matrix] if projection_type == "sparse" else []
    got = site.evaluate_subspace(projections, base_sd, coords, bary_sd, spec, col, values, inputs, targets)
    if held:
        assert all(a is b for a, b in zip(held, [P._rows, P._values, P._csr_matrix]))
    tol = 1e-5 if dtype is torch.float32 else 1.6e-2
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)


def test_half_site_holds_reconstruct_batch_weights(monkeypatch):
    """A BF16 bias of 1 at coordinate 0.003 stores 1, so the 0.002 step to a candidate
    at 0.005 vanishes when added to the stored incumbent, while the base plus 0.005
    rounds up to 1.0078125. The site must hold what ``reconstruct_batch`` builds from
    the base and the candidate's full coordinates.
    """
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8)).bfloat16()
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=1)
    projections = subspace.init_projections(torch.device("cpu"), torch.bfloat16)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_sd["0.bias"].fill_(1.0)
    coords = torch.full((subspace.subspace_dim,), 0.003, dtype=torch.bfloat16)
    bary_sd = subspace.apply_perturbation(projections, base_sd, coords)
    assert torch.equal(bary_sd["0.bias"], torch.ones(8, dtype=torch.bfloat16))

    site = SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss()), layout)
    scored = {}
    monkeypatch.setattr(site, "_vmap_over_site", lambda key, sd, weights, *_: scored.__setitem__(key, weights))
    pdim, steps = 4, torch.tensor([0.005, 0.0035], dtype=torch.bfloat16)
    for spec in subspace.specs:
        n_groups = spec.num_coords // pdim
        values = steps.view(1, -1, 1).expand(n_groups, -1, pdim)
        site.evaluate_subspace(projections, base_sd, coords, bary_sd, spec, 0, values, None)
        full = coords.repeat(n_groups, len(steps), 1)
        for g in range(n_groups):
            start = spec.flat_start + g * pdim
            full[g, :, start : start + pdim] = values[g]
        expected = subspace.reconstruct_batch(projections, base_sd, full.flatten(0, 1))[spec.entry_key]
        assert torch.equal(scored[spec.entry_key], expected)
    assert scored["0.bias"].unique().tolist() == [1.0, 1.0078125]


def test_half_subspace_step_matches_materializing_path():
    """The shared BF16 weights must round as the candidates do: a fused ``addmm``
    rounds once where the candidates round twice, so the two paths would score the
    same candidate differently and could swap a ranking.
    """
    from polystep import HybridSubspace

    def build(register):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Tanh(), nn.Linear(4, 4, bias=False)).bfloat16()
        gen = torch.Generator().manual_seed(2)
        inputs, targets = torch.randn(32, 4, generator=gen), torch.randint(0, 4, (32,), generator=gen)
        evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
        evaluator._batched_linear = None
        subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
        opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=0.1, seed=3, subspace=subspace)
        if register:
            opt.register_evaluator(evaluator, inputs, targets)
        return opt, [opt.step(lambda bp: evaluator.evaluate(bp, inputs, targets)) for _ in range(6)]

    _, dense = build(register=False)
    opt_fast, fast = build(register=True)
    assert opt_fast._site_vmap_evaluator is not None, "the site path must be available, or this proves nothing"
    assert fast == dense


def test_release_evaluator_drops_site_cast_batch():
    """The site path casts through its owner evaluator and keeps it across a release,
    so the release has to drop the owner's cast copy of the batch as well."""
    from polystep import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3)).bfloat16()
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=1)
    opt = PolyStepOptimizer(model, seed=0, subspace=subspace)
    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = _small_batch()
    opt.register_evaluator(evaluator, inputs, targets)
    opt.step(lambda p: evaluator.evaluate(p, inputs, targets))
    assert opt._site_vmap_evaluator._owner is evaluator
    assert evaluator._cast_cache is not None
    opt.release_evaluator()
    assert evaluator._cast_cache is None


def test_monolithic_site_step_after_release_runs():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Flatten(), nn.Linear(72, 3))
    opt = PolyStepOptimizer(model, compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    x, y = torch.randn(4, 1, 8, 8), torch.randint(0, 3, (4,))
    opt.register_evaluator(ev, x, y)
    opt.step(lambda p: ev.evaluate(p, x, y))
    opt.release_evaluator()
    assert opt._site_vmap_evaluator is not None
    assert torch.isfinite(torch.as_tensor(opt.step(lambda p: ev.evaluate(p, x, y))))


def test_compiled_site_tracks_base_and_batch(monkeypatch):
    # Dynamo's eager backend checks graph guards without requiring a C++ toolchain.
    torch.compiler.reset()
    compile_fn = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: compile_fn(fn, backend="eager", fullgraph=True))
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    owner = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_vmap=True)
    site_ev = SiteVmapEvaluator.try_build(owner, ParamLayout.from_module(model))
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    key = "2.weight"
    for batch in (5, 3):
        x, y = torch.randn(batch, 3), torch.randint(0, 2, (batch,))
        base["0.weight"].add_(0.2)
        candidates = base[key].unsqueeze(0) + torch.randn(3, *base[key].shape) * 0.1
        got = site_ev._vmap_over_site(key, base, candidates, x, y)
        expected = torch.stack(
            [
                nn.functional.cross_entropy(torch.func.functional_call(model, {**base, key: p}, (x,)), y)
                for p in candidates
            ]
        )
        torch.testing.assert_close(got, expected)
    assert set(site_ev._compiled_sites) == {key}
    assert not site_ev._compile_failed_sites
    torch.compiler.reset()


def test_mixed_dtype_site_skips_scalar_cast():
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones((), dtype=torch.float64))
            self.weight = nn.Parameter(torch.randn(2, 3))

        def forward(self, x):
            return (x @ self.weight.t()).double() * self.scale

    model = Mixed()
    ev = NNCostEvaluator(model, nn.MSELoss())
    site_ev = SiteVmapEvaluator.try_build(ev, ParamLayout.from_module(model))
    base = dict(model.named_parameters())
    site = base["weight"][None].expand(2, -1, -1)
    x, y = torch.randn(4, 3), torch.randn(4, 2, dtype=torch.float64)
    got = site_ev._vmap_over_site("weight", base, site, x, y)
    torch.testing.assert_close(got, nn.functional.mse_loss(model(x), y).expand(2))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_grouped_site_casts_past_float32_buffer(dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2).to(dtype)
            self.register_buffer("extra", torch.zeros(()))

        def forward(self, x):
            return self.fc(x) + self.extra

    torch.manual_seed(0)
    model, loss_fn = Model(), nn.MSELoss()
    x, y = torch.randn(8, 2), torch.randn(8, 2)
    opt = PolyStepOptimizer(model, block_strategy="grouped", seed=0)
    opt.register_evaluator(NNCostEvaluator(model, loss_fn), x, y)
    with torch.no_grad():
        loss = opt.step(lambda *_: loss_fn(model(x.to(dtype)).float(), y))
    assert torch.isfinite(torch.as_tensor(loss)).all()
    assert model.fc.weight.dtype is dtype


def test_reregister_rebuilds_site_evaluator_and_owner():
    model = nn.Sequential(nn.Linear(3, 2))
    opt = PolyStepOptimizer(model)
    ev = NNCostEvaluator(model, nn.MSELoss())
    x, y = torch.randn(4, 3), torch.randn(4, 2)
    opt.register_evaluator(ev, x, y)
    old = opt._site_vmap_evaluator
    ev.reset_vmap()
    opt.register_evaluator(ev, x, y)
    assert opt._site_vmap_evaluator is not old
    replacement = NNCostEvaluator(model, ev.loss_fn, compile_vmap=True)
    opt.register_evaluator(replacement, x, y)
    assert opt._site_vmap_evaluator._owner is replacement


@pytest.mark.parametrize("block_strategy, screen", [("monolithic", False), ("monolithic", True), ("per_layer", False)])
def test_materialized_fallback_honours_chunk_size(block_strategy, screen):
    """The site path rounds its stride to whole groups; the dense fallback keeps the user's cap."""
    model = nn.Sequential(nn.Linear(2, 2))
    opt = PolyStepOptimizer(
        model,
        particle_dim=3,
        chunk_size=1,
        multifidelity_screen=screen,
        block_strategy=block_strategy,
        solver="softmax",
        compile=False,
        seed=0,
    )
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    x, y = torch.randn(4, 2), torch.randn(4, 2)
    opt.register_evaluator(ev, x, y)
    rows = []

    def closure(params):
        rows.append(next(iter(params.values())).shape[0])
        return ev.evaluate(params, x, y)

    opt.step(closure, screen_closure=closure if screen else None)
    assert rows and max(rows) == 1
    assert torch.isfinite(opt.state.X).all()


def _record_site_batches(opt):
    """Candidate count of every site-path forward, keyed by parameter name."""
    site = opt._site_vmap_evaluator
    rows = {}
    inner = site._vmap_over_site

    def spy(key, base_sd, candidates, inputs, targets):
        rows.setdefault(key, []).append(tree_leaves(candidates)[0].shape[0])
        return inner(key, base_sd, candidates, inputs, targets)

    site._vmap_over_site = spy
    return rows


@pytest.mark.parametrize("route", ["monolithic", "per_layer", "hybrid"])
def test_site_chunk_budgets_downstream_activations(route):
    from polystep.hybrid_subspace import HybridSubspace

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64, 8)
    )
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=3) if route == "hybrid" else None
    block_strategy = "per_layer" if route == "per_layer" else "monolithic"
    opt = PolyStepOptimizer(
        model, subspace=subspace, block_strategy=block_strategy, solver="softmax", compile=False, seed=0
    )
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(32, 1, 64, 64), torch.randint(0, 8, (32,))
    opt.register_evaluator(ev, x, y)
    head = 32 * 8 / x.numel()
    assert opt._act_ratios == {None: 4.0, "0.weight": 4.0, "0.bias": 4.0, "4.weight": head, "4.bias": head}
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())

    rows = _record_site_batches(opt)
    opt.step(lambda p: ev.evaluate(p, x, y))
    conv_chunk = max(rows["0.weight"])
    assert 2 * 4 * x.numel() * conv_chunk <= 1 << 26
    assert (4 * x.numel() * conv_chunk * x.element_size() <= 32 << 20) == (route == "hybrid")
    assert max(rows["4.weight"]) > conv_chunk


@pytest.mark.parametrize("dtype, cpu_chunk", [(torch.float32, 5), (torch.float64, 2)])
def test_local_chunk_shrinks_only_on_cpu(dtype, cpu_chunk):
    from polystep._step_core import local_chunk

    opt = PolyStepOptimizer(nn.Linear(4, 2).to(dtype), compile=False, seed=0)
    act = 1 << 18
    assert local_chunk(opt, act, 3 * act, 1 << 20, cpu=False) == 85
    assert local_chunk(opt, act, 3 * act, 1 << 20, cpu=True) == cpu_chunk
    assert local_chunk(opt, act, 3 * act, 16, cpu=True) == 16
    assert local_chunk(opt, act, 3 * act, (32 << 20) // (act * dtype.itemsize) + 1, cpu=True) == cpu_chunk


def test_site_chunk_capped_by_own_span():
    from polystep._step_core import site_chunks

    opt = PolyStepOptimizer(nn.Linear(4, 2), compile=False, seed=0)
    x = torch.zeros(1 << 16, 4)
    sites = [(0, "weight"), (8, "bias")]
    assert site_chunks(opt, x, sites, 1, 6, 600, shrink=True) == {0: 48, 48: 12}
    assert site_chunks(opt, x.to("meta"), sites, 1, 6, 600, shrink=True)[48] == 252
    assert site_chunks(opt, x, sites, 1, 6, 600)[48] == 252
    assert site_chunks(opt, x, sites, 3, 6, 600) == {0: 12, 18: 252}
    assert site_chunks(opt, None, sites, 1, 6, 600, shrink=True) == {0: 48, 48: 552}


def test_cpu_site_chunk_keeps_injection():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(4, 8, 5), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
    opt = PolyStepOptimizer(model, compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(32, 4, 14, 14), torch.randint(0, 4, (32,))
    opt.register_evaluator(ev, x, y)
    opt.step(lambda p: ev.evaluate(p, x, y))
    site = opt._site_vmap_evaluator
    calls = []
    inner = site._vmap_over_site
    site._vmap_over_site = lambda key, sd, c, *a: calls.append((key, isinstance(c, tuple))) or inner(key, sd, c, *a)
    opt.step(lambda p: ev.evaluate(p, x, y))
    assert ("0.weight", True) in calls


@pytest.mark.parametrize("block_strategy", ["monolithic", "per_layer"])
def test_explicit_chunk_size_caps_site_batches(block_strategy):
    """Six is a whole number of particle groups, so rounding keeps it."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64, 8)
    )
    opt = PolyStepOptimizer(model, chunk_size=6, block_strategy=block_strategy, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(64, 1, 64, 64), torch.randint(0, 8, (64,))
    opt.register_evaluator(ev, x, y)
    rows = _record_site_batches(opt)
    opt.step(lambda p: ev.evaluate(p, x, y))
    assert set(rows) == {"0.weight", "0.bias", "4.weight", "4.bias"}
    assert max(map(max, rows.values())) == 6


def test_unaligned_site_keeps_downstream_chunk():
    """The head starts at coordinate 27, inside a two-coordinate particle."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 3, 3, bias=False), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, 16, bias=False)
    )
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(64, 1, 64, 64), torch.randint(0, 16, (64,))
    opt.register_evaluator(ev, x, y)
    opt.step(lambda p: ev.evaluate(p, x, y))
    rows = _record_site_batches(opt)
    opt.step(lambda p: ev.evaluate(p, x, y))
    assert len(rows["3.weight"]) == 1


@pytest.mark.parametrize("hybrid", [False, True])
def test_delta_paths_skip_site_chunks(monkeypatch, hybrid):
    import polystep._step_monolithic as monolithic
    from polystep.hybrid_subspace import HybridSubspace

    plans = []
    spans = monolithic.chunk_spans

    def spy(n, chunk, bounds, chunks):
        plans.append(chunks)
        return spans(n, chunk, bounds, chunks)

    monkeypatch.setattr(monolithic, "chunk_spans", spy)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2) if hybrid else None
    opt = PolyStepOptimizer(model, subspace=subspace, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    x, y = torch.randn(32, 8), torch.randn(32, 4)
    opt.register_evaluator(ev, x, y)
    delta = opt._subspace_delta_evaluator if hybrid else opt._sparse_delta_evaluator
    assert delta is not None
    opt.step(lambda p: ev.evaluate(p, x, y))
    assert plans and all(c is None for c in plans)


def test_site_budget_unexecuted_module_uses_global():
    from polystep._step_core import activation_elements

    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return nn.functional.linear(self.conv(x).mean((2, 3)), self.head.weight, self.head.bias)

    model = Functional()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    x = torch.randn(8, 1, 16, 16)
    opt.register_evaluator(NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False), x, torch.zeros(8).long())
    assert "head.weight" not in opt._act_ratios
    assert activation_elements(opt, x, "head.weight") == activation_elements(opt, x) == 4 * x.numel()


def test_site_budget_reused_weight_uses_global():
    class Reuse(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix = nn.Conv2d(2, 4, 1, bias=False)
            self.early = nn.Conv2d(2, 2, 1, bias=False)
            self.late = nn.Conv2d(2, 2, 1, bias=False)

        def forward(self, x):
            x = nn.functional.conv2d(self.prefix(x)[:, :2], self.late.weight)
            return self.late(self.early(x).mean((2, 3), keepdim=True)).flatten(1)

    model = Reuse()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    x = torch.randn(4, 2, 8, 8)
    opt.register_evaluator(NNCostEvaluator(model, nn.MSELoss(), use_inplace=False), x, torch.zeros(4, 2))
    assert opt._act_ratios == {None: 2.0, "prefix.weight": 2.0, "early.weight": 1.0}


class InputNoise(nn.Module):
    def forward(self, x):
        return x + 0.01 * torch.randn_like(x)


def test_site_budget_random_forward_uses_global():
    model = nn.Sequential(InputNoise(), nn.Linear(4, 8), nn.Linear(8, 2))
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    x = torch.randn(8, 4)
    opt.register_evaluator(NNCostEvaluator(model, nn.MSELoss(), use_inplace=False), x, torch.zeros(8, 2))
    assert opt._act_ratios == {None: 2.0}


@pytest.mark.parametrize(
    "layer, ratios",
    [("relu", {None: 2.0, "0.weight": 2.0, "0.bias": 2.0, "2.weight": 0.5, "2.bias": 0.5}), ("linear", None)],
    ids=["relu", "linear"],
)
def test_site_budget_scripted_submodule_skips_hooks(layer, ratios):
    layers = [nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)]
    i = {"linear": 0, "relu": 1}[layer]
    layers[i] = torch.jit.script(layers[i])
    model = nn.Sequential(*layers)
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=False)
    x, y = torch.randn(16, 4), torch.zeros(16, 2)
    opt.register_evaluator(ev, x, y)
    assert opt._act_ratios == ratios
    assert not any(
        m._forward_hooks or m._forward_pre_hooks for m in layers if not isinstance(m, torch.jit.ScriptModule)
    )
    opt.step(lambda p: ev.evaluate(p, x, y))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("noise", [nn.Identity, InputNoise])
def test_site_chunk_peak_within_budget(noise):
    torch.manual_seed(0)
    model = nn.Sequential(
        noise(), nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128, 8)
    ).cuda()
    opt = PolyStepOptimizer(model, solver="softmax", compile=False, seed=0)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    x, y = torch.randn(64, 1, 64, 64, device="cuda"), torch.randint(0, 8, (64,), device="cuda")
    opt.register_evaluator(ev, x, y)
    site = opt._site_vmap_evaluator
    inner = site._vmap_over_site
    peaks = {}

    def spy(key, base_sd, candidates, inputs, targets):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        out = inner(key, base_sd, candidates, inputs, targets)
        peaks[key] = max(peaks.get(key, 0), torch.cuda.max_memory_allocated() - start)
        return out

    site._vmap_over_site = spy
    opt.step(lambda p: ev.evaluate(p, x, y))
    assert set(peaks) == {"1.weight", "1.bias", "5.weight", "5.bias"}
    assert max(peaks.values()) <= 1.05 * 4 * (1 << 26)


def test_cpu_stock_attention_skips_hint():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*VmapSafeMultiHeadAttention")
        NNCostEvaluator(nn.MultiheadAttention(8, 2, batch_first=True), nn.MSELoss(), use_inplace=False)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stock_attention_warns_vmapsafe_layer(monkeypatch):
    model = nn.MultiheadAttention(8, 2, batch_first=True).cuda()
    monkeypatch.setattr(NNCostEvaluator, "_batched_footprint_exceeds_free_memory", staticmethod(lambda m: True))
    for use_inplace in (False, None):
        with pytest.warns(UserWarning, match="VmapSafeMultiHeadAttention"):
            NNCostEvaluator(model, nn.MSELoss(), use_inplace=use_inplace)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*VmapSafeMultiHeadAttention")
        NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)


class TinyAttention(nn.Module):
    """Embeds tokens and attends once, so every site runs through attention."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.attn = VmapSafeMultiHeadAttention(8, 2, batch_first=True)
        self.head = nn.Linear(8, 3)
        self.register_buffer("pos", torch.randn(5, 8))

    def forward(self, x):
        h = self.embed(x) + self.pos
        return self.head((h + self.attn(h, h, h)).mean(1))


class HostScaledMLP(nn.Module):
    """Divides by an input statistic read back to the host, which graph capture refuses."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))) / (1.0 + x.abs().mean().item())


class HostCopiedMLP(HostScaledMLP):
    """Copies a CPU tensor in before any kernel, so the refused capture is empty."""

    def forward(self, x):
        return super().forward(x * torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0]).to(x.device))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_site_graph_matches_eager_losses():
    from polystep import HybridSubspace

    batches = [(torch.randn(32, 5, 4, device="cuda"), torch.randint(0, 3, (32,), device="cuda")) for _ in range(10)]
    runs = []
    for compile_forward in (False, None):
        torch.manual_seed(0)
        model = TinyAttention().cuda()
        subspace = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
        opt = PolyStepOptimizer(model, seed=0, subspace=subspace)
        ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_forward=compile_forward)
        losses = [opt.step(opt.register_evaluator(ev, x, y)) for x, y in batches]
        runs.append((losses, list(model.parameters()), opt._site_vmap_evaluator))
    (eager, eager_params, _), (graphed, graphed_params, site) = runs
    assert any(entry is not None for entry in site._graphs.values())
    assert eager == graphed
    assert all(map(torch.equal, eager_params, graphed_params))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("model_cls", [HostScaledMLP, HostCopiedMLP])
def test_site_graph_sync_forward_falls_back(model_cls):
    torch.manual_seed(0)
    model = model_cls().cuda()
    layout = ParamLayout.from_module(model)
    eager, graphed = (
        SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_forward=flag), layout)
        for flag in (False, None)
    )
    base_sd = {k: v.detach() for k, v in model.named_parameters()}
    x, y = (t.cuda() for t in _small_batch())
    keys = ["fc1.weight", "fc2.weight"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for key in keys:
            candidates = base_sd[key] + 0.1 * torch.randn(4, *base_sd[key].shape, device="cuda")
            for _ in range(3):
                got = graphed._vmap_over_site(key, base_sd, candidates, x, y)
                assert torch.equal(got, eager._vmap_over_site(key, base_sd, candidates, x, y))
    assert len([w for w in caught if "cuda graph" in str(w.message).lower()]) == 1
    assert graphed._graph_failed_sites == set(keys)
    assert not any(graphed._graphs.values())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tied_site_graph_matches_eager():
    torch.manual_seed(0)
    model = TiedHead().cuda()
    layout = ParamLayout.from_module(model)
    eager, graphed = (
        SiteVmapEvaluator.try_build(NNCostEvaluator(model, nn.CrossEntropyLoss(), compile_forward=flag), layout)
        for flag in (False, None)
    )
    base_sd = layout.unflatten(layout.flatten(model))
    x, y = torch.randint(0, 24, (7,), device="cuda"), torch.randint(0, 24, (7,), device="cuda")
    candidates = base_sd["out.bias"] + 0.1 * torch.randn(4, 24, device="cuda")
    for _ in range(3):
        got = graphed._vmap_over_site("out.bias", base_sd, candidates, x, y)
        assert torch.equal(got, eager._vmap_over_site("out.bias", base_sd, candidates, x, y))
    graphed._vmap_over_site("out.bias", base_sd, candidates.flip(0), x, y)
    assert torch.equal(got, eager._vmap_over_site("out.bias", base_sd, candidates, x, y))
    assert not graphed._graph_failed_sites and any(graphed._graphs.values())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_site_graph_target_dtype_change_recaptures():
    model = nn.Linear(2, 2, bias=False).cuda()
    nn.init.zeros_(model.weight)
    ev = NNCostEvaluator(model, nn.MSELoss())
    opt = PolyStepOptimizer(model, solver="softmax", seed=1)
    x = torch.zeros(3, 2, device="cuda")
    targets = (x.long(), x.long(), x + 0.5, x + 0.5)
    assert [opt.step(opt.register_evaluator(ev, x, y)) for y in targets] == [0.0, 0.0, 0.25, 0.25]
    assert sum(entry is not None for entry in opt._site_vmap_evaluator._graphs.values()) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_site_graph_tuple_targets_no_warning():
    model = nn.Linear(2, 2, bias=False).cuda()
    ev = NNCostEvaluator(model, lambda out, t: nn.functional.mse_loss(out, t[0]) + nn.functional.mse_loss(out, t[1]))
    opt = PolyStepOptimizer(model, solver="softmax", seed=1)
    x = torch.zeros(3, 2, device="cuda")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "vmap failed")
        for _ in range(3):
            opt.step(opt.register_evaluator(ev, x, (x, x + 0.5)))
