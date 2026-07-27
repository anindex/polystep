"""Parity and fallback contract for the full-space sparse-delta evaluator.

A full-space candidate perturbs one contiguous run of the flat parameter vector, so
every layer but one holds the shared base weight. :class:`SparseDeltaEvaluator`
exploits that by carrying a ``(N, B, particle_dim)`` delta instead of materializing
per-candidate activations. It must agree with the dense path to floating-point
tolerance, and must decline the cases where the confinement argument does not hold.
"""

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import BatchedLinearEvaluator, SparseDeltaEvaluator
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
    """Every candidate the fast path accepts must match the dense path.

    Sweeps every particle row, so weight sites, bias sites, the last layer, and rows
    whose two scalars share an output unit are all covered.
    """
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
        values = torch.randn(1, PDIM, generator=gen)

        fast = evaluator.evaluate(base_sd, entry.key, local, values, inputs, targets)
        reference = _dense_reference(model, layout, loss_fn, kind, flat, start, values[0], inputs, targets)
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
    values = torch.randn(len(starts), PDIM, generator=torch.Generator().manual_seed(2))

    batched = evaluator.evaluate(base_sd, entry.key, local, values, inputs, targets)
    one_at_a_time = torch.cat(
        [
            evaluator.evaluate(base_sd, entry.key, local[i : i + 1], values[i : i + 1], inputs, targets)
            for i in range(len(starts))
        ]
    )
    torch.testing.assert_close(batched, one_at_a_time, rtol=1e-5, atol=1e-5)


def test_declines_a_run_that_straddles_two_parameters():
    """The layout packs entries back to back with no particle_dim alignment.

    A row spanning the end of one parameter and the start of the next perturbs two
    layers at once, which breaks the single-site assumption.
    """
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
    """Delta confinement holds only for elementwise ops.

    Normalization, softmax and convolution spread a two-column perturbation across
    every output, so the delta goes dense immediately and the shortcut is invalid.
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

    def run(use_fast):
        torch.manual_seed(3)
        model = _mlp([8, 6, 3])
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        inputs, targets = torch.randn(12, 8), torch.randint(0, 3, (12,))
        opt = PolyStepOptimizer(model, compile=False, seed=7)
        opt.register_evaluator(evaluator, inputs, targets)
        if not use_fast:
            opt._sparse_delta_evaluator = None
        assert use_fast == (opt._sparse_delta_evaluator is not None)
        losses = [opt.step(lambda p: evaluator.evaluate(p, inputs, targets)) for _ in range(4)]
        return losses, model.state_dict()

    fast_losses, fast_sd = run(True)
    dense_losses, dense_sd = run(False)
    assert fast_losses == pytest.approx(dense_losses, rel=1e-5)
    for key in fast_sd:
        torch.testing.assert_close(fast_sd[key], dense_sd[key], rtol=1e-5, atol=1e-6)
