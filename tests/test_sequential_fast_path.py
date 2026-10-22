"""Models that are MLPs must not opt themselves out of the batched evaluators.

A custom forward takes the batched paths only when it traces to a plain chain;
these tests pin that plan, the warning for forwards that do not, and the shipped models.
"""

import warnings
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from polystep.benchmarks.rl.policies import DiscreteMLPPolicy
from polystep.benchmarks.utils import MNISTNet
from polystep import PolyStepOptimizer
from polystep.cost_nn import BatchedLinearEvaluator, NNCostEvaluator, SubspaceDeltaEvaluator, _SEQ_EQUIVALENT_WARNED
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


def _builds(model):
    return BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss(), "cross_entropy") is not None


@pytest.mark.parametrize(
    "factory,expected_keys",
    [
        (lambda: MNISTNet(hidden=8), ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]),
        (
            lambda: DiscreteMLPPolicy(4, 6, 2),
            ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"],
        ),
    ],
)
def test_shipped_mlps_fast_path_keeps_keys(factory, expected_keys):
    """The Sequential conversion must not move state_dict keys, or checkpoints break."""
    model = factory()
    assert _builds(model), f"{type(model).__name__} does not take the batched path"
    assert list(model.state_dict()) == expected_keys


def test_subspace_delta_accepts_shipped_mlp():
    """The subspace delta path needs the same Sequential check as the bmm path."""
    model = MNISTNet(hidden=8)
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=2)
    assert SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub) is not None


def _warns(model):
    _SEQ_EQUIVALENT_WARNED.discard(id(type(model)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss(), "cross_entropy")
    return any("built from Linear/activation layers only" in str(w.message) for w in caught)


def _plan_names(model):
    plan = BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss(), "cross_entropy")
    return None if plan is None else [name for name, _, _ in plan._layer_keys]


def test_untraceable_forward_over_children_warns():
    class Handwritten(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.relu, self.fc2 = nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3)

        def forward(self, x):
            if x.dim() > 2:
                x = x.flatten(1)
            return self.fc2(self.relu(self.fc1(x)))

    assert _warns(Handwritten())


def test_sequential_wrapper_keeps_dotted_keys():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 3))

        def forward(self, x):
            return self.net(x)

    assert not _warns(Wrapper())
    assert _plan_names(Wrapper()) == ["net.0", "net.1", "net.2"]


class _InlineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(12, 8), nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _hybrid_run(model):
    """Losses and final parameters of three registered hybrid steps."""
    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(16, 12, generator=g), torch.randint(0, 3, (16,), generator=g)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    sub = HybridSubspace.from_layout(ParamLayout.from_module(model), rank=2)
    opt = PolyStepOptimizer(model, subspace=sub, epsilon=0.3, step_radius=0.1, seed=3)
    opt.register_evaluator(ev, x, y)
    losses = [opt.step(lambda bp: ev.evaluate(bp, x, y)) for _ in range(3)]
    return opt, losses, [p.detach().clone() for p in model.parameters()]


def test_inline_relu_mlp_takes_delta_path():
    """The traced plan scores exactly what the nn.Sequential twin scores."""
    torch.manual_seed(0)
    inline = _InlineMLP()
    twin = nn.Sequential(OrderedDict([("fc1", nn.Linear(12, 8)), ("relu", nn.ReLU()), ("fc2", nn.Linear(8, 3))]))
    twin.load_state_dict(inline.state_dict())
    assert _plan_names(inline) == ["fc1", "relu", "fc2"]

    opt, losses, params = _hybrid_run(inline)
    _, twin_losses, twin_params = _hybrid_run(twin)
    assert opt._subspace_delta_evaluator is not None
    assert losses == twin_losses
    assert all(torch.equal(a, b) for a, b in zip(params, twin_params))


def test_training_branch_traces_eval_forward():
    """Candidates are scored in eval mode, so the plan must follow the eval branch."""

    class Branchy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(6, 8), nn.Linear(8, 3)

        def forward(self, x):
            h = self.fc1(x)
            return self.fc2(torch.relu(h) if self.training else torch.tanh(h))

    torch.manual_seed(0)
    model = Branchy().train()
    plan = BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss(), "cross_entropy")
    assert model.training
    inputs, targets = torch.randn(5, 6), torch.randint(0, 3, (5,))
    stacked = {k: v.detach()[None] for k, v in model.named_parameters()}
    expected = nn.functional.cross_entropy(model.eval()(inputs), targets)
    torch.testing.assert_close(plan.evaluate(stacked, inputs, targets), expected[None])


def _single_linear(forward):
    model = type("SingleLinear", (nn.Module,), {"forward": forward})()
    model.fc = nn.Linear(4, 4)
    return model


@pytest.mark.parametrize(
    "forward,hooked",
    [
        (lambda m, x: torch.relu(m.fc(x)), True),
        (lambda m, x: m.fc(torch.relu(m.fc(x))), False),
        (lambda m, x: nn.functional.relu(m.fc(x), True), False),
        (lambda m, x: m.fc(x) + x, False),
        (lambda m, x: (m.fc(x), torch.relu(x))[1], False),
        (lambda m, x: m.fc(x.flatten(0)), False),
        (lambda m, x: (m.fc(x), x), False),
        (lambda m, x: nn.functional.gelu(m.fc(x), approximate="tanh"), False),
        (lambda m, x: nn.functional.leaky_relu(m.fc(x), negative_slope=0.2), False),
    ],
    ids=[
        "root_hook",
        "repeated_linear",
        "inplace_relu",
        "residual_add",
        "skipped_input",
        "flatten_batch_dim",
        "tuple_output",
        "gelu_tanh",
        "leaky_slope",
    ],
)
def test_untraceable_forward_keeps_vmap_path(forward, hooked):
    """A wrong accept scores a different function, so each of these must decline."""
    model = _single_linear(forward)
    if hooked:
        assert _plan_names(model) == ["fc", "relu"]
        model.register_forward_hook(lambda m, i, o: o + 1.0)
    assert _plan_names(model) is None


def test_traced_plan_uses_inference_branch():
    """Candidates are scored under inference_mode, so the plan must follow that branch."""
    model = _single_linear(lambda m, x: torch.relu(m.fc(x)) if torch.is_grad_enabled() else torch.sigmoid(m.fc(x)))
    assert _plan_names(model) == ["fc", "sigmoid"]


def test_traced_autocast_chain_declines():
    def forward(m, x):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return torch.tanh(m.fc(x))

    assert _plan_names(_single_linear(forward)) is None


def test_declined_trace_keeps_global_rng():
    model = _single_linear(lambda m, x: m.fc(x + torch.randn(4)))
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    assert _plan_names(model) is None
    assert torch.equal(torch.rand(3), expected)


def test_child_hook_keeps_vmap_path():
    model = _single_linear(lambda m, x: m.fc(x))
    model.fc = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    assert _plan_names(model) == ["fc.0", "fc.1"]
    model.fc.register_forward_hook(lambda m, i, o: None)
    assert _plan_names(model) is None


def test_module_extra_arg_keeps_vmap_path():
    class Scale(nn.Module):
        polystep_elementwise = True

        def forward(self, x, factor=1.0):
            return x * factor

    plans = []
    for forward in (lambda m, x: m.act(m.fc(x)), lambda m, x: m.act(m.fc(x), 2.0)):
        model = _single_linear(forward)
        model.act = Scale()
        plans.append(_plan_names(model))
    assert plans == [["fc", "act"], None]


def test_unsupported_model_falls_back_silently():
    class HasConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv, self.relu, self.fc = nn.Conv2d(1, 2, 3), nn.ReLU(), nn.Linear(2, 3)

        def forward(self, x):
            return self.fc(self.relu(self.conv(x)).flatten(1))

    class ExtraParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.relu = nn.Linear(8, 3), nn.ReLU()
            self.scale = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.relu(self.fc(x)) * self.scale

    assert not _warns(HasConv())
    assert not _warns(ExtraParam())


def test_sequential_subclass_matches_handwritten_form():
    """The conversion is a speed change, not a model change."""
    torch.manual_seed(0)
    seq = MNISTNet(hidden=8)

    class Handwritten(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten, self.fc1, self.relu, self.fc2 = nn.Flatten(), nn.Linear(784, 8), nn.ReLU(), nn.Linear(8, 10)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(self.flatten(x))))

    hand = Handwritten()
    hand.load_state_dict(seq.state_dict())
    x = torch.randn(4, 1, 28, 28)
    torch.testing.assert_close(seq(x), hand(x))


def test_ordereddict_form_keeps_children_walkable():
    """try_build walks one level of nn.Sequential; a named child must still resolve."""
    model = nn.Sequential(OrderedDict([("a", nn.Linear(4, 4)), ("act", nn.ReLU()), ("b", nn.Linear(4, 2))]))
    assert _builds(model)


def test_subspace_inplace_matches_materializing_path():
    """``evaluate_subspace_inplace`` must match ``reconstruct_batch`` plus ``evaluate``.

    It must also leave the model weights where it found them.
    """
    from polystep.cost_nn import NNCostEvaluator

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(20, 12), nn.ReLU(), nn.Linear(12, 4))
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=3)
    projections = subspace.init_projections(torch.device("cpu"), torch.float32)

    evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())
    inputs, targets = torch.randn(9, 20), torch.randint(0, 4, (9,))
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    coords = torch.randn(6, subspace.subspace_dim, generator=torch.Generator().manual_seed(1)) * 0.05

    fused = evaluator.evaluate_subspace_inplace(subspace, projections, base, coords, inputs, targets)
    reference = evaluator.evaluate(subspace.reconstruct_batch(projections, base, coords), inputs, targets)

    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-6)
    # Distinct coordinates must give distinct losses, or the swap never took effect.
    assert len(set(fused.tolist())) == len(fused)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, base[key])


# Each nn.Sequential model is paired with a twin holding the equivalent hand-written
# forward, so a Sequential form that computes something else shows up as a forward or
# state_dict difference.
class _HandStaircase(nn.Module):
    def __init__(self, levels=5):
        super().__init__()
        from experiments.runners.nondiff_models import StaircaseActivation

        self.fc1, self.staircase = nn.Linear(784, 128), StaircaseActivation(levels)
        self.fc2, self.staircase2 = nn.Linear(128, 128), StaircaseActivation(levels)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc3(self.staircase2(self.fc2(self.staircase(self.fc1(x)))))


class _HandQuantized(nn.Module):
    def __init__(self):
        super().__init__()
        from experiments.runners.nondiff_models import QuantizedLinear

        self.fc1, self.quant = nn.Linear(784, 128), QuantizedLinear(128, 128)
        self.relu, self.fc2 = nn.ReLU(), nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc2(self.relu(self.quant(torch.relu(self.fc1(x)))))


def _hand_two_layer(layer_cls):
    class Hand(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.relu, self.fc2 = layer_cls(784, 128), nn.ReLU(), layer_cls(128, 10)

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            return self.fc2(self.relu(self.fc1(x)))

    return Hand


def _sequential_twins():
    """The model pairs, or one skipped param when the harness is absent.

    ``experiments/`` ships in the repo but not the distribution, so this skips from
    an sdist rather than silently dropping the tests.
    """
    try:
        from experiments.runners import nondiff_models as models
    except ImportError:
        return [pytest.param(None, None, None, None, marks=pytest.mark.skip(reason="experiments/ not present (sdist)"))]

    return [
        ("StaircaseNet", models.StaircaseNet, _HandStaircase, True),
        ("QuantizedMLP", models.QuantizedMLP, _HandQuantized, True),
        ("BinaryMNISTNet", models.BinaryMNISTNet, _hand_two_layer(models.BinaryLinear), True),
        ("TernaryMNISTNet", models.TernaryMNISTNet, _hand_two_layer(models.TernaryLinear), True),
        ("BinaryMNISTNetSTE", models.BinaryMNISTNetSTE, _hand_two_layer(models.BinaryLinearSTE), False),
        ("TernaryMNISTNetSTE", models.TernaryMNISTNetSTE, _hand_two_layer(models.TernaryLinearSTE), False),
    ]


_SEQUENTIAL_TWINS = _sequential_twins()


@pytest.mark.parametrize(
    "name,seq_cls,hand_cls,on_fast_path", _SEQUENTIAL_TWINS, ids=lambda v: getattr(v, "__name__", v)
)
def test_sequential_models_match_handwritten_twin(name, seq_cls, hand_cls, on_fast_path):
    """Same seed, same weights, same keys, same output. Only the dispatch differs.

    Parameter names seed the random projections and parameter order fixes the flat
    layout, so a rename or a reorder changes the trajectory even at identical weights.
    """
    torch.manual_seed(0)
    seq = seq_cls()
    torch.manual_seed(0)
    hand = hand_cls()
    assert list(seq.state_dict()) == list(hand.state_dict()), name
    for key, value in seq.state_dict().items():
        assert torch.equal(value, hand.state_dict()[key]), f"{name}.{key}"
    assert _builds(seq) is on_fast_path, name


@pytest.mark.parametrize(
    "name,seq_cls,hand_cls,on_fast_path", _SEQUENTIAL_TWINS, ids=lambda v: getattr(v, "__name__", v)
)
def test_sequential_models_forward_match_handwritten_twin(name, seq_cls, hand_cls, on_fast_path):
    """Forward equality at shared weights, which is what a dropped activation breaks."""
    torch.manual_seed(0)
    seq = seq_cls()
    hand = hand_cls()
    hand.load_state_dict(seq.state_dict())
    torch.manual_seed(1)
    for inputs in (torch.randn(4, 1, 28, 28), torch.randn(3, 784)):
        assert torch.equal(seq(inputs), hand(inputs)), name


def test_weight_transform_keeps_bmm_declines_subspace(require_experiments):
    """The subspace and factored paths keep a correction that is linear in the delta.

    A piecewise-constant weight transform breaks that linearity, and rebuilding the
    weight per candidate is exactly what those paths exist to avoid, so they decline.
    """
    from polystep.cost_nn import FactoredEvaluator
    from experiments.runners.nondiff_models import BinaryMNISTNet

    model = BinaryMNISTNet(input_dim=16, hidden=8, output=4)
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=2)
    assert _builds(model)
    assert SubspaceDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), sub) is None
    assert FactoredEvaluator.try_build(model, nn.CrossEntropyLoss()) is None


def test_ternary_threshold_keeps_layer_alive(require_experiments):
    """A threshold several sigma past the initialization zeroes every effective weight.

    The layer then returns its bias for any input and the benchmark measures nothing.
    """
    from experiments.runners.nondiff_models import TernaryLinear

    torch.manual_seed(0)
    layer = TernaryLinear(64, 32)
    effective = layer.polystep_weight_transform(layer.weight.detach())
    assert (effective != 0).float().mean() > 0.4
    with pytest.warns(UserWarning, match="zeroes every"):
        TernaryLinear(64, 32, threshold=5.0)


def _record_fc1_shapes(net):
    """Swap fc1 for a module recording the shape of every input it is called with."""
    shapes = []
    real_fc1 = net.fc1

    class _Recorder(nn.Module):
        def forward(self, inp):
            shapes.append(tuple(inp.shape))
            return real_fc1(inp)

    net.fc1 = _Recorder()
    return shapes


def test_snn_loops_fc1_on_temporal_input():
    """Temporal input keeps fc1 inside the timestep loop.

    Hoisting fc1 to one (T*B, F) GEMM changes BLAS blocking, and the LIF threshold
    turns that into a whole spike. Asserted on call shapes, not outputs, since
    bitwise GEMM differences are machine-dependent.
    """
    from polystep.benchmarks.utils import SpikingNet

    torch.manual_seed(0)
    steps, batch, features = 6, 8, 64
    net = SpikingNet(input_dim=features, hidden=32, output=10, num_steps=steps)

    shapes = _record_fc1_shapes(net)
    net(torch.rand(steps, batch, features))
    assert shapes == [(batch, features)] * steps


def test_snn_hoists_fc1_on_static_input():
    """Static input is constant in time, so fc1 runs once and broadcasts."""
    from polystep.benchmarks.utils import SpikingNet

    torch.manual_seed(0)
    steps, batch, features = 6, 8, 64
    net = SpikingNet(input_dim=features, hidden=32, output=10, num_steps=steps)

    shapes = _record_fc1_shapes(net)
    net(torch.rand(batch, features))
    assert shapes == [(batch, features)]


def test_spiking_net_defaults_to_builtin_lif(monkeypatch):
    from types import SimpleNamespace

    from polystep.benchmarks import utils

    monkeypatch.setattr(utils, "_HAS_SNNTORCH", True)
    monkeypatch.setattr(utils, "snn", SimpleNamespace(Leaky=lambda beta: nn.Identity()), raising=False)
    net = utils.SpikingNet(input_dim=4, hidden=3, output=2, num_steps=2)
    assert isinstance(net.lif1, utils.LIFNeuron)
    assert isinstance(net.lif2, utils.LIFNeuron)
    stub_net = utils.SpikingNet(input_dim=4, hidden=3, output=2, num_steps=2, use_snntorch=True)
    assert isinstance(stub_net.lif1, nn.Identity)


def test_spiking_net_snntorch_missing_raises(monkeypatch):
    from polystep.benchmarks import utils

    monkeypatch.setattr(utils, "_HAS_SNNTORCH", False)
    with pytest.raises(ImportError, match="needs snntorch"):
        utils.SpikingNet(input_dim=4, hidden=3, output=2, num_steps=2, use_snntorch=True)
