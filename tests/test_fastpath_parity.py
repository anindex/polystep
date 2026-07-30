"""A fast evaluator path must score what the materializing path scores.

Each model here looks eligible on a shallow reading. The plan must either decline it
or reproduce the reference loss; a silently different number reorders the cost matrix.
"""

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from polystep.cost_nn import BatchedLinearEvaluator, NNCostEvaluator, _batched_loss_kind
from polystep.transform import ParamLayout


def _reference_losses(model, stacked, inputs, targets, loss_fn):
    """Loss per candidate through plain functional_call, one candidate at a time."""
    buffers = dict(model.named_buffers())
    n = next(iter(stacked.values())).shape[0]
    out = []
    for i in range(n):
        params = {k: v[i] for k, v in stacked.items()}
        y = functional_call(model, {**params, **buffers}, (inputs,))
        out.append(loss_fn(y, targets))
    return torch.stack(out)


def _stack(model, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        k: (v.unsqueeze(0) + 0.1 * torch.randn(n, *v.shape, generator=g)).clone()
        for k, v in model.named_parameters(remove_duplicate=False)
    }


def test_repeated_module_is_not_planned_as_one_layer():
    """Sequential(fc, ReLU, fc) applies fc twice; named_children reports it once."""
    fc = nn.Linear(4, 4)
    model = nn.Sequential(fc, nn.ReLU(), fc)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_tied_weights_across_two_linears_decline_the_bmm_plan():
    a, b = nn.Linear(4, 4), nn.Linear(4, 4)
    b.weight = a.weight
    model = nn.Sequential(a, nn.ReLU(), b)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_frozen_bias_is_still_applied():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    model[2].bias.requires_grad_(False)
    with torch.no_grad():
        model[2].bias.fill_(7.0)

    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    loss_fn = nn.MSELoss()
    stacked = {k: v for k, v in _stack(model, 5).items() if k != "2.bias"}

    fast = BatchedLinearEvaluator.try_build(model, loss_fn, "mse")
    assert fast is not None
    torch.testing.assert_close(
        fast.evaluate(stacked, inputs, targets),
        _reference_losses(model, stacked, inputs, targets, loss_fn),
    )


def test_forward_hook_declines_the_bmm_plan():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    model[2].register_forward_hook(lambda m, i, o: o + 5.0)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_root_forward_hook_declines_the_bmm_plan():
    model = nn.Sequential(nn.Linear(4, 2))
    model.register_forward_hook(lambda m, i, o: o * 3.0)
    assert BatchedLinearEvaluator.try_build(model, nn.MSELoss(), "mse") is None


def test_loss_subclass_that_overrides_forward_is_refused():
    class PlusMSE(nn.MSELoss):
        def forward(self, output, target):
            return super().forward(output, target) + 10.0

    assert _batched_loss_kind(PlusMSE()) is None
    # A subclass that only renames itself keeps the base reduction and stays eligible.
    assert _batched_loss_kind(type("Renamed", (nn.MSELoss,), {})()) == "mse"


def test_soft_label_cross_entropy_falls_back_to_vmap():
    model = nn.Sequential(nn.Linear(4, 3))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert ev._batched_linear is not None  # the plan itself is eligible

    inputs = torch.randn(6, 4)
    soft = torch.softmax(torch.randn(6, 3), dim=1)
    stacked = _stack(model, 4)
    torch.testing.assert_close(
        ev.evaluate(stacked, inputs, soft),
        _reference_losses(model, stacked, inputs, soft, nn.CrossEntropyLoss()),
    )


def test_inplace_path_keeps_float64_resolution():
    """FP32 loss buffers collapse candidates that differ below FP32 resolution."""
    model = nn.Sequential(nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
    ev = NNCostEvaluator(model, nn.MSELoss(), use_inplace=True)

    inputs = torch.ones(1, 2, dtype=torch.float64)
    targets = torch.zeros(1, 1, dtype=torch.float64)
    bias = torch.tensor([[1.0], [1.0 + 1e-9], [1.0 + 2e-9]], dtype=torch.float64)
    stacked = {"0.weight": model[0].weight.unsqueeze(0).expand(3, 1, 2).clone(), "0.bias": bias}

    losses = ev.evaluate(stacked, inputs, targets)
    assert losses.dtype == torch.float64
    assert len(torch.unique(losses)) == 3


def test_shifted_overlapping_views_are_rejected():
    """Two windows of one buffer are not independent parameters."""
    # Parameter() copies, so assign .data to keep both views on one storage.
    shared = torch.zeros(6)
    model = nn.Module()
    model.a = nn.Parameter(torch.zeros(4))
    model.b = nn.Parameter(torch.zeros(4))
    model.a.data = shared[0:4]
    model.b.data = shared[2:6]
    with pytest.raises(ValueError, match="share storage"):
        ParamLayout.from_module(model)


def test_disjoint_slices_of_one_buffer_stay_independent():
    shared = torch.zeros(6)
    model = nn.Module()
    model.a = nn.Parameter(torch.zeros(3))
    model.b = nn.Parameter(torch.zeros(3))
    model.a.data = shared[0:3]
    model.b.data = shared[3:6]
    layout = ParamLayout.from_module(model)
    assert layout.total_params == 6
    assert layout.shared_groups == ()


def test_reset_vmap_rebuilds_the_plan_after_a_layer_swap():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    ev = NNCostEvaluator(model, nn.MSELoss())
    inputs, targets = torch.randn(5, 3), torch.randn(5, 2)
    stacked = _stack(model, 4)

    model[1] = nn.Sigmoid()
    ev.reset_vmap()
    torch.testing.assert_close(
        ev.evaluate(stacked, inputs, targets),
        _reference_losses(model, stacked, inputs, targets, nn.MSELoss()),
    )


def test_reset_vmap_rebinds_buffers():
    model = nn.Sequential(nn.Linear(2, 1))
    model.register_buffer("scale", torch.zeros(()))
    ev = NNCostEvaluator(model, nn.MSELoss())
    model.scale = torch.full((), 4.0)
    ev.reset_vmap()
    assert ev._buffers["scale"].item() == 4.0
