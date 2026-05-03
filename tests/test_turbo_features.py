"""Regression tests for turbo-mode features and defaults.

- ``amortize_loss_gate`` (opt-in): when the amortized cost is much
  worse than the last fresh cost, revert to the pre-momentum state
  and fall back to a full OT step on the next iteration.
- ``num_probe`` defaults: ``PolyStepOptimizer`` and the low-level
  ``PolyStep`` both default to K=1 (the paper's optimal value for
  the softmax solver).
- SNN + CosineEpsilon-on-step_radius warning: discrete-spike models
  collapse from ~93% to 10-47% accuracy under a scheduled
  ``step_radius`` (per experiments/EXPERIMENT_INDEX.md).
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import (
    PolyStepOptimizer,
    PolyStep,
    CosineEpsilon,
)
from polystep.cost_nn import NNCostEvaluator


# ---------------------------------------------------------------------------
# 2F.5 K=1 vs K=3 default unification
# ---------------------------------------------------------------------------


def test_poly_step_optimizer_num_probe_default_is_1():
    """Headline runners use K=1; the optimizer default must match."""
    model = nn.Linear(4, 2, bias=False)
    opt = PolyStepOptimizer(model, epsilon=0.5)
    assert opt.num_probe == 1


def test_poly_step_low_level_num_probe_default_is_1():
    """The low-level PolyStep (synthetic objectives) used to default to
    K=5; the default is unified to K=1 to match
    PolyStepOptimizer and the paper's optimal value."""
    def dummy_obj(x):
        return (x ** 2).sum(-1)

    solver = PolyStep.create(dummy_obj, dim=4)
    assert solver.num_probe == 1, (
        f"PolyStep.num_probe default should match PolyStepOptimizer "
        f"(K=1 per paper); got {solver.num_probe}"
    )


# ---------------------------------------------------------------------------
# 2F.6 SNN cosine epsilon guard
# ---------------------------------------------------------------------------


class _FakeLIF(nn.Module):
    """Stand-in for a snnTorch.Leaky neuron without the snntorch dep."""

    def forward(self, x):
        return torch.relu(x)


class _SNNStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.lif1 = _FakeLIF()  # name pattern matches LIF/Leaky cells
        self.fc2 = nn.Linear(8, 2)


def test_snn_with_cosine_step_radius_warns():
    """Per experiments/EXPERIMENT_INDEX.md, scheduling step_radius on an SNN model
    collapses accuracy from ~93% to 10-47%. The optimizer must warn the
    caller when this combination is detected.
    """
    model = _SNNStub()
    cosine = CosineEpsilon(init=5.0, target=1.0, decay=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PolyStepOptimizer(model, epsilon=0.5, step_radius=cosine)

    msgs = [str(w.message).lower() for w in caught]
    assert any(
        ("snn" in m or "leaky" in m or "lif" in m or "spik" in m)
        and ("step_radius" in m or "cosine" in m)
        for m in msgs
    ), (
        "expected a warning about CosineEpsilon on step_radius for an "
        f"SNN-like model; got: {msgs}"
    )


def test_non_snn_with_cosine_step_radius_no_warn():
    """The guard must NOT fire on an MLP-only model."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    cosine = CosineEpsilon(init=5.0, target=1.0, decay=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PolyStepOptimizer(model, epsilon=0.5, step_radius=cosine)
    msgs = [str(w.message).lower() for w in caught]
    snn_warnings = [m for m in msgs if "snn" in m or "leaky" in m or "lif" in m]
    assert not snn_warnings, (
        f"guard fired on a non-SNN MLP: {snn_warnings}"
    )


# ---------------------------------------------------------------------------
# 2F.1 EMA loss-gated revert (opt-in)
# ---------------------------------------------------------------------------


def test_ema_loss_gate_default_off_preserves_existing_behavior():
    """The new amortize_loss_gate flag defaults to False so headline
    runs are bit-for-bit unchanged."""
    model = nn.Linear(4, 2, bias=False)
    opt = PolyStepOptimizer(model, epsilon=0.5, amortize_steps=3)
    assert opt.amortize_loss_gate is False
    assert opt.amortize_loss_gate_threshold == 1.5
    assert opt.amortize_loss_gate_floor == 0.1


def test_ema_loss_gate_negative_loss_uses_additive_tolerance():
    """For negative losses the multiplicative 1.5x threshold is
    nonsense (1.5 * -10 = -15, which is BETTER than -10). The
    implementation switches to an additive tolerance whenever
    |last_loss| < amortize_loss_gate_floor or the loss is negative.
    Smoke-test the helper directly.
    """
    model = nn.Linear(4, 2, bias=False)
    opt = PolyStepOptimizer(
        model, epsilon=0.5, amortize_steps=3,
        amortize_loss_gate=True,
        amortize_loss_gate_threshold=1.5,
        amortize_loss_gate_floor=0.1,
    )

    # Helper signature: returns True if amortized step should be reverted.
    # Negative loss case: amortized=-9, last=-10 -> WORSE (less negative)
    # but multiplicative test would say -9 < 1.5 * -10 = -15 (False, no revert).
    # Additive fallback says |amortized - last| > floor AND amortized > last (worse).
    assert opt._amortize_should_revert(amortized=-9.0, last=-10.0) is True

    # Positive loss case: amortized = 2.0 vs last = 1.0 -> 2.0 > 1.5 * 1.0
    assert opt._amortize_should_revert(amortized=2.0, last=1.0) is True

    # Tiny improvement should not revert.
    assert opt._amortize_should_revert(amortized=1.05, last=1.0) is False
