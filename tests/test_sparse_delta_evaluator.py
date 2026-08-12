"""Parity and fallback contract for the full-space sparse-delta evaluator: it
must match the dense path and decline where the confinement argument fails.
"""

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import BatchedLinearEvaluator, NNCostEvaluator, SparseDeltaEvaluator
from polystep.transform import ParamLayout

PDIM = 2


def _mlp(dims, bias=True, act=nn.ReLU):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _dense_reference(model, layout, loss_fn, kind, flat, start, values, inputs, targets):
    """Loss of the same candidate through the materializing bmm path."""
    dense = BatchedLinearEvaluator.try_build(model, loss_fn, kind)
    cand = flat.clone()
    cand[start : start + PDIM] = values
    return dense.evaluate(layout.batch_unflatten(cand.unsqueeze(0)), inputs, targets)


ARCHITECTURES = [
    ("two_layer", [6, 5, 3], nn.CrossEntropyLoss(), "cross_entropy"),
    ("single_layer", [6, 3], nn.CrossEntropyLoss(), "cross_entropy"),
    ("deep", [6, 5, 4, 3], nn.MSELoss(), "mse"),
    ("narrow_hidden", [6, 2, 3], nn.L1Loss(), "l1"),
]


@pytest.mark.parametrize("name,dims,loss_fn,kind", ARCHITECTURES, ids=[a[0] for a in ARCHITECTURES])
def test_matches_the_dense_path_at_every_perturbation_site(name, dims, loss_fn, kind):
    """Every accepted candidate must match the dense path, across all site kinds."""
    torch.manual_seed(0)
    model = _mlp(dims)
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator.try_build(model, loss_fn, layout)
    assert evaluator is not None, "a plain MLP must be supported"

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    inputs = torch.randn(9, dims[0])
    targets = torch.randint(0, dims[-1], (9,)) if kind == "cross_entropy" else torch.randn(9, dims[-1])

    gen = torch.Generator().manual_seed(1)
    accepted = 0
    for row in range(flat.shape[0] // PDIM):
        start = row * PDIM
        entry = evaluator.resolve_site(torch.tensor([start]), PDIM)
        if entry is None:
            continue
        local = (torch.arange(start, start + PDIM) - entry.offset).unsqueeze(0)
        values = torch.randn(1, 1, PDIM, generator=gen)

        fast = evaluator.evaluate(base_sd, entry.key, local, values, inputs, targets)
        reference = _dense_reference(model, layout, loss_fn, kind, flat, start, values[0, 0], inputs, targets)
        assert fast.shape == reference.shape
        torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-5)
        accepted += 1

    assert accepted > 0, "the fast path declined every site"


def test_a_batch_of_candidates_matches_one_at_a_time():
    """The chunk is evaluated in one call, so N candidates must agree with N calls."""
    torch.manual_seed(0)
    model = _mlp([6, 5, 4, 3])
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator.try_build(model, nn.MSELoss(), layout)

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    inputs, targets = torch.randn(9, 6), torch.randn(9, 3)

    starts = torch.arange(0, 8, PDIM)
    entry = evaluator.resolve_site(starts, PDIM)
    assert entry is not None, "the first four rows should sit inside one weight"

    local = starts.unsqueeze(1) + torch.arange(PDIM) - entry.offset
    values = torch.randn(len(starts), 1, PDIM, generator=torch.Generator().manual_seed(2))

    batched = evaluator.evaluate(base_sd, entry.key, local, values, inputs, targets)
    one_at_a_time = torch.cat(
        [
            evaluator.evaluate(base_sd, entry.key, local[i : i + 1], values[i : i + 1], inputs, targets)
            for i in range(len(starts))
        ]
    )
    torch.testing.assert_close(batched, one_at_a_time, rtol=1e-5, atol=1e-5)


def test_grouping_candidates_by_particle_changes_nothing():
    """Grouping candidates by particle must return the same losses in the same order."""
    torch.manual_seed(0)
    model = _mlp([6, 5, 4, 3])
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator.try_build(model, nn.MSELoss(), layout)

    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    inputs, targets = torch.randn(9, 6), torch.randn(9, 3)

    groups, cand = 4, 3
    starts = torch.arange(0, groups * PDIM, PDIM)
    entry = evaluator.resolve_site(starts, PDIM)
    assert entry is not None

    local = starts.unsqueeze(1) + torch.arange(PDIM) - entry.offset
    values = torch.randn(groups, cand, PDIM, generator=torch.Generator().manual_seed(2))

    grouped = evaluator.evaluate(base_sd, entry.key, local, values, inputs, targets)
    flat = evaluator.evaluate(
        base_sd,
        entry.key,
        local.repeat_interleave(cand, dim=0),
        values.reshape(groups * cand, 1, PDIM),
        inputs,
        targets,
    )
    torch.testing.assert_close(grouped, flat, rtol=0, atol=0)


def test_declines_a_run_that_straddles_two_parameters():
    """A run straddling two parameters perturbs two layers at once and must be declined."""
    torch.manual_seed(0)
    # Linear(3, 3) gives a 9-element weight, so the flat run [8, 10) crosses into bias.
    model = _mlp([3, 3, 2])
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator.try_build(model, nn.MSELoss(), layout)

    weight = next(e for e in layout.entries if e.key == "0.weight")
    assert weight.numel % PDIM == 1, "this layout needs an odd-numel entry to straddle"
    straddling = weight.offset + weight.numel - 1
    assert evaluator.resolve_site(torch.tensor([straddling]), PDIM) is None


def test_declines_a_chunk_spanning_two_parameters():
    """A chunk covering rows from two different entries has no single site."""
    torch.manual_seed(0)
    model = _mlp([4, 4, 2])
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator.try_build(model, nn.MSELoss(), layout)

    weight = next(e for e in layout.entries if e.key == "0.weight")
    spanning = torch.tensor([weight.offset, weight.offset + weight.numel])
    assert evaluator.resolve_site(spanning, PDIM) is None


def test_declines_tied_weights():
    """One flat position drives two module paths, so a single-site correction is wrong."""

    class Tied(nn.Sequential):
        def __init__(self):
            super().__init__(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
            self[2].weight = self[0].weight

    model = Tied()
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    assert layout.shared_groups, "this model is meant to have tied weights"
    assert SparseDeltaEvaluator.try_build(model, nn.MSELoss(), layout) is None


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2)),
        nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)),
        nn.Sequential(nn.Linear(4, 4), nn.Softmax(dim=-1), nn.Linear(4, 2)),
        nn.Sequential(nn.Conv1d(1, 2, 3), nn.Flatten(), nn.Linear(4, 2)),
    ],
    ids=["layernorm", "batchnorm", "softmax", "conv"],
)
def test_declines_layers_that_mix_features(model):
    """Normalization, softmax and convolution spread a perturbation across every
    output, breaking delta confinement.
    """
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    assert SparseDeltaEvaluator.try_build(model, nn.MSELoss(), layout) is None


def test_declines_a_configured_loss():
    """A loss the batched reduction cannot reproduce must use the real loss_fn."""
    model = _mlp([4, 4, 2])
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    assert SparseDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(label_smoothing=0.1), layout) is None


def test_optimizer_step_matches_with_and_without_the_fast_path():
    """End to end: the fast path must not change the trajectory it accelerates."""
    from polystep import PolyStepOptimizer
    from polystep.cost_nn import NNCostEvaluator

    calls = []

    def run(use_fast):
        torch.manual_seed(3)
        model = _mlp([8, 6, 3])
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs, targets = torch.randn(12, 8), torch.randint(0, 3, (12,))
        # Small chunk_size keeps each candidate run inside one layout entry;
        # otherwise resolve_site rejects every chunk and the fast path never runs.
        opt = PolyStepOptimizer(model, compile=False, seed=7, chunk_size=4)
        opt.register_evaluator(evaluator, inputs, targets)
        if not use_fast:
            opt._sparse_delta_evaluator = None
        assert use_fast == (opt._sparse_delta_evaluator is not None)
        if use_fast:
            real = opt._sparse_delta_evaluator.evaluate

            def counted(*a, **k):
                calls.append(1)
                return real(*a, **k)

            opt._sparse_delta_evaluator.evaluate = counted
        losses = [opt.step(lambda p: evaluator.evaluate(p, inputs, targets)) for _ in range(4)]
        return losses, model.state_dict()

    fast_losses, fast_sd = run(True)
    assert calls, "the fast path never ran, so this compares the dense path to itself"
    dense_losses, dense_sd = run(False)
    assert fast_losses == pytest.approx(dense_losses, rel=1e-5)
    for key in fast_sd:
        torch.testing.assert_close(fast_sd[key], dense_sd[key], rtol=1e-5, atol=1e-6)


class _SignAct(nn.Module):
    """Elementwise step function: no gradient, no continuity, still exact here."""

    polystep_elementwise = True

    def forward(self, x):
        return torch.sign(x)


class _StatefulAct(nn.Module):
    """Declares the contract but carries a buffer, so the plan must refuse it."""

    polystep_elementwise = True

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.zeros(1))

    def forward(self, x):
        return x + self.shift


class _PerTensorAct(nn.Module):
    """Declares the contract and breaks it: every output depends on every input."""

    polystep_elementwise = True

    def forward(self, x):
        return x / (x.abs().amax() + 1e-8)


class _SignLinear(nn.Module):
    """Binary-weight Linear declaring the transform its forward applies."""

    polystep_weight_transform = staticmethod(torch.sign)
    polystep_bias_transform = None

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        return x @ torch.sign(self.weight).t() + self.bias


def _sweep_every_site(model):
    """Worst disagreement with the dense path over every perturbation site."""
    loss_fn, kind = nn.CrossEntropyLoss(), "cross_entropy"
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator.try_build(model, loss_fn, layout)
    assert evaluator is not None
    dense = BatchedLinearEvaluator.try_build(model, loss_fn, kind)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    inputs, targets = torch.randn(7, 6), torch.randint(0, 3, (7,))
    worst, sites = 0.0, 0
    for particle in range(layout.padded_size // PDIM):
        start = particle * PDIM
        site = evaluator.resolve_site(torch.tensor([start]), PDIM, (start, start + PDIM))
        if site is None:
            continue
        sites += 1
        values = flat[start : start + PDIM] + torch.tensor([0.37, -0.52])
        got = evaluator.evaluate(
            base_sd,
            site.key,
            torch.tensor([[start]]) + torch.arange(PDIM) - site.offset,
            values.reshape(1, 1, PDIM),
            inputs,
            targets,
        )
        candidate = flat.clone()
        candidate[start : start + PDIM] = values
        reference = dense.evaluate(layout.batch_unflatten(candidate.unsqueeze(0)), inputs, targets)
        worst = max(worst, (got.reshape(-1) - reference.reshape(-1)).abs().max().item())
    assert sites, "no site resolved, so nothing was compared"
    return worst


@pytest.mark.parametrize("activation", [nn.ReLU, nn.Sigmoid, _SignAct], ids=["relu", "sigmoid", "sign"])
def test_delta_algebra_is_exact_for_any_elementwise_activation(activation):
    """``module(a + d) - module(a)`` is an exact finite difference for any
    elementwise activation.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), activation(), nn.Linear(5, 3))
    assert _sweep_every_site(model) < 1e-5


@pytest.mark.parametrize("layer", [nn.Linear, _SignLinear], ids=["linear", "sign_weight"])
def test_delta_algebra_is_exact_for_a_declared_weight_transform(layer):
    """A weight-transforming layer moves its output by ``Q(w + d) - Q(w)``, which
    the evaluator must reproduce.
    """
    torch.manual_seed(0)
    model = nn.Sequential(layer(6, 5), nn.ReLU(), layer(5, 3))
    assert _sweep_every_site(model) < 1e-5


@pytest.mark.parametrize("activation", [_StatefulAct, _PerTensorAct], ids=["buffer", "per_tensor"])
def test_a_module_that_cannot_keep_the_elementwise_contract_is_declined(activation):
    """A module with a buffer or a whole-tensor reduction cannot keep the
    elementwise contract and must be declined.
    """
    model = nn.Sequential(nn.Linear(6, 5), activation(), nn.Linear(5, 3))
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    assert SparseDeltaEvaluator.try_build(model, nn.CrossEntropyLoss(), layout) is None


def test_the_per_tensor_control_really_does_disagree():
    """Guards the test above: it must decline something that would be wrong if accepted."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), _PerTensorAct(), nn.Linear(5, 3))
    plan = BatchedLinearEvaluator.try_build(model, nn.CrossEntropyLoss(), "cross_entropy")
    assert plan is None
    # Force the plan the gate refused to build, then show the algebra breaks on it.
    forced = [("0", "linear", model[0]), ("1", "activation", model[1]), ("2", "linear", model[2])]
    layout = ParamLayout.from_module(model, particle_dim=PDIM)
    evaluator = SparseDeltaEvaluator(model, nn.CrossEntropyLoss(), forced, layout, "cross_entropy")
    object.__setattr__(evaluator, "_layer_keys", forced)
    base_sd = {k: v.detach() for k, v in model.state_dict().items()}
    flat = layout.flatten(model).reshape(-1)
    inputs, targets = torch.randn(7, 6), torch.randint(0, 3, (7,))
    values = flat[0:PDIM] + torch.tensor([0.37, -0.52])
    got = evaluator.evaluate(
        base_sd, "0.weight", torch.arange(PDIM).reshape(1, PDIM), values.reshape(1, 1, PDIM), inputs, targets
    )
    candidate = flat.clone()
    candidate[0:PDIM] = values
    reference = torch.func.functional_call(
        model, {k: v[0] for k, v in layout.batch_unflatten(candidate.unsqueeze(0)).items()}, inputs
    )
    reference = nn.CrossEntropyLoss()(reference, targets)
    assert (got.reshape(-1) - reference).abs().max() > 1e-3


def test_blockwise_step_scores_through_the_delta_path():
    """A block-wise step must reach the delta path and agree with the dense one.

    Without it block-wise builds a full model configuration per candidate and calls
    the closure, leaving every delta evaluator unreachable.
    """
    from polystep import PolyStepOptimizer
    from polystep.cost_nn import NNCostEvaluator

    def build():
        torch.manual_seed(0)
        model = _mlp([8, 12, 4])
        gen = torch.Generator().manual_seed(0)
        inputs = torch.randn(16, 8, generator=gen)
        targets = torch.randn(16, 4, generator=gen)
        evaluator = NNCostEvaluator(model, loss_fn=nn.MSELoss())
        calls = []

        def closure(batched_params):
            calls.append(1)
            return evaluator.evaluate(batched_params, inputs, targets)

        torch.manual_seed(7)
        opt = PolyStepOptimizer(model, epsilon=0.3, step_radius=0.1, seed=3, block_strategy="per_layer")
        return opt, evaluator, inputs, targets, closure, calls

    opt_dense, _, _, _, dense_closure, dense_calls = build()
    dense = [opt_dense.step(dense_closure) for _ in range(4)]

    opt_fast, evaluator, inputs, targets, fast_closure, fast_calls = build()
    opt_fast.register_evaluator(evaluator, inputs, targets)
    assert opt_fast._sparse_delta_evaluator is not None, "the delta path must be available, or this proves nothing"
    fast = [opt_fast.step(fast_closure) for _ in range(4)]

    assert dense_calls, "the dense run must actually call the closure"
    assert not fast_calls, "every chunk should have been scored without materializing a configuration"
    torch.testing.assert_close(torch.tensor(fast), torch.tensor(dense), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("frozen", ["0.weight", "0.bias", "2.weight", "2.bias"])
def test_delta_evaluator_declines_when_a_linear_param_is_frozen(frozen):
    """``ParamLayout`` drops ``requires_grad=False`` parameters, but the delta path
    read every Linear weight straight out of that partial ``base_sd``: a frozen
    weight raised KeyError and a frozen bias silently evaluated a bias-free net.
    Both must fall back to the materializing path instead.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    dict(model.named_parameters())[frozen].requires_grad_(False)

    optimizer = PolyStepOptimizer(model, seed=0)
    evaluator = NNCostEvaluator(model, nn.MSELoss())
    inputs, targets = torch.randn(8, 4), torch.randn(8, 2)
    optimizer.register_evaluator(evaluator, inputs, targets)

    assert optimizer._sparse_delta_evaluator is None
    # The step must still run, through the materializing path.
    optimizer.step(lambda params: evaluator.evaluate(params, inputs, targets))


def test_registering_a_new_objective_rebuilds_the_fast_paths():
    """The fast-path evaluators were built once and keyed on nothing, so
    re-registering an evaluator with a different loss left them scoring the old
    objective while the closure scored the new one.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    optimizer = PolyStepOptimizer(model, seed=0)
    inputs, targets = torch.randn(8, 4), torch.randn(8, 2)

    optimizer.register_evaluator(NNCostEvaluator(model, nn.MSELoss()), inputs, targets)
    first = optimizer._sparse_delta_evaluator
    assert isinstance(first.loss_fn, nn.MSELoss)

    optimizer.register_evaluator(NNCostEvaluator(model, nn.L1Loss()), inputs, targets)
    second = optimizer._sparse_delta_evaluator

    assert second is not first
    assert isinstance(second.loss_fn, nn.L1Loss)
