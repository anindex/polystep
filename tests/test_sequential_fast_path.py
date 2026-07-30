"""Models that are MLPs must not opt themselves out of the batched evaluators.

Every batched path requires ``type(model).forward is nn.Sequential.forward``. A
hand-written forward identical to the inherited one costs 2x on CUDA and 5x on CPU with
no other symptom, so the library warns on that shape. These tests pin the warning and
the shipped models that would otherwise trigger it.
"""

import warnings
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from polystep.benchmarks.rl.policies import DiscreteMLPPolicy
from polystep.benchmarks.utils import MNISTNet
from polystep.cost_nn import BatchedLinearEvaluator, SubspaceDeltaEvaluator, _SEQ_EQUIVALENT_WARNED
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
def test_shipped_mlps_take_the_fast_path_without_renaming_parameters(factory, expected_keys):
    """The Sequential conversion must not move state_dict keys, or checkpoints break."""
    model = factory()
    assert _builds(model), f"{type(model).__name__} does not take the batched path"
    assert list(model.state_dict()) == expected_keys


def test_subspace_delta_accepts_a_shipped_mlp():
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


def test_warns_on_a_handwritten_forward_over_module_children():
    class Handwritten(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.relu, self.fc2 = nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    assert _warns(Handwritten())


def test_warns_on_a_thin_wrapper_around_a_sequential():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 3))

        def forward(self, x):
            return self.net(x)

    assert _warns(Wrapper())


def test_silent_when_activations_are_applied_inline():
    """No activation child means the children are not the whole forward.

    Telling this model to subclass nn.Sequential would change what it computes, not
    just how fast it runs.
    """

    class InlineMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(6, 8), nn.Linear(8, 3)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    assert not _warns(InlineMLP())


def test_silent_on_an_unsupported_layer_and_on_extra_parameters():
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


def test_sequential_subclass_is_numerically_identical_to_the_handwritten_form():
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


def test_ordereddict_form_keeps_the_children_walkable():
    """try_build walks one level of nn.Sequential; a named child must still resolve."""
    model = nn.Sequential(OrderedDict([("a", nn.Linear(4, 4)), ("act", nn.ReLU()), ("b", nn.Linear(4, 2))]))
    assert _builds(model)


def test_subspace_inplace_matches_the_materializing_path():
    """``evaluate_subspace_inplace`` is ~95 lines with no test of its own.

    It trades the stacked ``(N, *param_shape)`` dict for one in-place weight swap per
    candidate, so its only contract is that it returns what ``reconstruct_batch`` plus
    a normal ``evaluate`` returns, and that it leaves the model where it found it.
    """
    import torch.nn as nn

    from polystep.cost_nn import NNCostEvaluator
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

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
# forward, so a conversion that changed the model rather than its speed shows up as a
# forward or state_dict difference.
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


def _converted_models():
    """The model pairs, or one skipped param when the harness is absent.

    ``experiments/`` is the paper reproduction harness and ships in the repo, not in the
    distribution, so this runs at collection time in a checkout and skips from an sdist.
    Returning an empty list instead would delete the tests with no signal.
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


_CONVERTED_MODELS = _converted_models()


@pytest.mark.parametrize(
    "name,new_cls,old_cls,on_fast_path", _CONVERTED_MODELS, ids=lambda v: getattr(v, "__name__", v)
)
def test_converted_models_are_bit_identical_to_their_handwritten_form(name, new_cls, old_cls, on_fast_path):
    """Same seed, same weights, same keys, same output. Only the dispatch changed.

    Parameter names seed the random projections and parameter order fixes the flat
    layout, so a rename or a reorder changes the trajectory even at identical weights.
    """
    torch.manual_seed(0)
    new = new_cls()
    torch.manual_seed(0)
    old = old_cls()
    assert list(new.state_dict()) == list(old.state_dict()), name
    for key, value in new.state_dict().items():
        assert torch.equal(value, old.state_dict()[key]), f"{name}.{key}"
    assert _builds(new) is on_fast_path, name


@pytest.mark.parametrize(
    "name,new_cls,old_cls,on_fast_path", _CONVERTED_MODELS, ids=lambda v: getattr(v, "__name__", v)
)
def test_converted_models_forward_bit_identically(name, new_cls, old_cls, on_fast_path):
    """Forward equality at shared weights, which is what a dropped activation breaks."""
    torch.manual_seed(0)
    new = new_cls()
    old = old_cls()
    old.load_state_dict(new.state_dict())
    torch.manual_seed(1)
    for inputs in (torch.randn(4, 1, 28, 28), torch.randn(3, 784)):
        assert torch.equal(new(inputs), old(inputs)), name


def test_a_weight_transforming_layer_reaches_the_bmm_path_but_not_the_subspace_one(require_experiments):
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


def test_ternary_threshold_leaves_the_layer_alive(require_experiments):
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


def test_spiking_net_applies_fc1_per_timestep_on_temporal_input():
    """Temporal input keeps fc1 inside the timestep loop.

    One (T*B, F) GEMM blocks differently from the T (B, F) GEMMs it would replace, and
    the LIF threshold turns that difference into a whole spike. Only the static branch,
    where every call sees the same tensor, may hoist fc1 out.

    Asserted on the call shapes, not on the outputs: whether the two GEMM shapes differ
    bitwise is a property of the BLAS blocking, so an output comparison passes or fails
    by machine.
    """
    from polystep.benchmarks.utils import SpikingNet

    torch.manual_seed(0)
    steps, batch, features = 6, 8, 64
    net = SpikingNet(input_dim=features, hidden=32, output=10, num_steps=steps)

    shapes = _record_fc1_shapes(net)
    net(torch.rand(steps, batch, features))
    assert shapes == [(batch, features)] * steps


def test_spiking_net_hoists_fc1_on_static_input():
    """Static input is constant in time, so fc1 runs once and broadcasts."""
    from polystep.benchmarks.utils import SpikingNet

    torch.manual_seed(0)
    steps, batch, features = 6, 8, 64
    net = SpikingNet(input_dim=features, hidden=32, output=10, num_steps=steps)

    shapes = _record_fc1_shapes(net)
    net(torch.rand(batch, features))
    assert shapes == [(batch, features)]
