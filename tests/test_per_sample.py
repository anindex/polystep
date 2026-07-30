"""Averaging the (N, B) per-sample tensor must reproduce the (N,) path exactly, or
anything read off it describes a different estimator than the one PolyStep steps with."""

import pytest
import torch
import torch.nn as nn

from polystep.cost_nn import NNCostEvaluator


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 4))


def _stacked(model, n):
    """N perturbed copies of the model's parameters, stacked on a leading axis."""
    return {
        k: v.unsqueeze(0).repeat(n, *([1] * v.dim())) + 0.05 * torch.randn(n, *v.shape)
        for k, v in model.named_parameters()
    }


def test_per_sample_mean_matches_scalar_path():
    model = _model()
    x, y = torch.randn(32, 12), torch.randint(0, 4, (32,))
    params = _stacked(model, 5)

    scalar = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=False)
    per = NNCostEvaluator(model, nn.CrossEntropyLoss(reduction="none"), use_inplace=False, per_sample=True)

    ref = scalar.evaluate(params, x, y)
    tensor = per.evaluate(params, x, y)

    assert scalar.per_sample is False, "per_sample must default off, the shipped path returns (N,)"
    assert ref.shape == (5,)
    assert tensor.shape == (5, 32)
    torch.testing.assert_close(tensor.mean(dim=1), ref, rtol=1e-5, atol=1e-6)


def test_per_sample_rejects_inplace_path():
    model = _model()
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(reduction="none"), use_inplace=True, per_sample=True)
    with pytest.raises(NotImplementedError, match="in-place path"):
        ev.evaluate(_stacked(model, 2), torch.randn(8, 12), torch.randint(0, 4, (8,)))


def test_per_sample_rejects_a_reducing_loss():
    """A mean-reduced loss silently returns (N,), which is the shape per_sample exists
    to avoid."""
    with pytest.raises(ValueError, match="reduction"):
        NNCostEvaluator(_model(), nn.CrossEntropyLoss(), per_sample=True)


def test_per_sample_rejects_a_reducing_callable():
    """A plain callable carries no ``reduction``, so the constructor check passes it."""
    model = _model()
    ev = NNCostEvaluator(model, lambda out, tgt: nn.functional.cross_entropy(out, tgt), per_sample=True)
    with pytest.raises(ValueError, match="returned a scalar"):
        ev.evaluate(_stacked(model, 3), torch.randn(8, 12), torch.randint(0, 4, (8,)))


def test_per_sample_folds_trailing_target_dims():
    """An unreduced loss over non-scalar targets keeps their trailing dims; per_sample
    promises exactly one value per sample."""
    model = nn.Sequential(nn.Linear(4, 3))
    evaluator = NNCostEvaluator(model, nn.MSELoss(reduction="none"), per_sample=True)
    inputs, targets = torch.randn(7, 4), torch.randn(7, 3)
    losses = evaluator.evaluate(_stacked(model, 5), inputs, targets)
    assert losses.shape == (5, 7)
