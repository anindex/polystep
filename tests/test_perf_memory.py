"""Memory and performance regression tests.

- ``auto_detect_chunk_size`` returns None on CPU and shrinks the
  chunk under ``compile_overhead=True`` (1.5x extra safety margin).
- ``BatchedLinearEvaluator`` (the bmm fast path) agrees bit-for-bit
  with the ``vmap + functional_call`` reference path on an MLP, and
  is built only for ``CrossEntropyLoss + Linear/activation`` models.
- ``PolyStepOptimizer.step()`` leaves no ``.grad`` attribute on the
  model parameters (it is a gradient-free optimizer; a populated
  ``.grad`` would point to a leftover backward pass).
"""
from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn as nn

from polystep import (
    NNCostEvaluator,
    PolyStepOptimizer,
    auto_detect_chunk_size,
)
from polystep.cost_nn import BatchedLinearEvaluator
from polystep.transform import ParamLayout


# ---------------------------------------------------------------------------
# 4.1a/b chunk_size auto-detect
# ---------------------------------------------------------------------------


def test_auto_chunk_size_returns_none_on_cpu():
    """A CPU model has no memory budget; auto_detect_chunk_size must
    return None to disable chunking."""
    model = nn.Linear(8, 4)
    assert auto_detect_chunk_size(model) is None


@pytest.mark.gpu
def test_auto_chunk_size_compile_factor_shrinks_chunk():
    """compile_overhead=True should produce a smaller (or equal)
    chunk_size because the 1.5x extra safety factor reserves
    headroom for torch.compile peaks."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = nn.Linear(64, 32).cuda()
    eager = auto_detect_chunk_size(model)
    compiled = auto_detect_chunk_size(model, compile_overhead=True)
    assert compiled is not None and eager is not None
    assert compiled <= eager
    # Ratio should be ~1/1.5 of eager (within rounding).
    assert compiled >= int(eager / 1.6)


# ---------------------------------------------------------------------------
# 4.3a/b bmm fast path matches vmap+functional_call output
# ---------------------------------------------------------------------------


def test_bmm_path_matches_vmap_path_on_mlp():
    """For an MLP-only model with CrossEntropyLoss the bmm fast path
    is built. Its cost matrix must match the vmap+functional_call path
    bit-for-bit (within FP32 round-off tolerance)."""
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    loss_fn = nn.CrossEntropyLoss()

    bmm_eval = NNCostEvaluator(model, loss_fn=loss_fn)
    assert bmm_eval._batched_linear is not None, (
        "expected bmm fast path to be built for MLP + CrossEntropyLoss"
    )
    # Disable bmm path on a second evaluator to force the vmap route.
    vmap_eval = NNCostEvaluator(model, loss_fn=loss_fn)
    vmap_eval._batched_linear = None

    layout = ParamLayout.from_module(model)
    flat = layout.flatten(model)  # (rows, particle_dim)
    # Batch of 4 candidates: identity + 3 small perturbations.
    perturbations = torch.randn(4, *flat.shape) * 1e-3
    perturbations[0].zero_()
    candidates = flat.unsqueeze(0) + perturbations
    stacked = layout.batch_unflatten(candidates)

    inputs = torch.randn(8, 8)
    targets = torch.randint(0, 4, (8,))

    bmm_costs = bmm_eval.evaluate(stacked, inputs, targets)
    vmap_costs = vmap_eval.evaluate(stacked, inputs, targets)

    assert bmm_costs.shape == (4,)
    assert vmap_costs.shape == (4,)
    diff = (bmm_costs - vmap_costs).abs().max().item()
    assert diff < 1e-4, (
        f"bmm vs vmap cost matrix diverged by {diff:.3e}; "
        f"bmm={bmm_costs.tolist()}, vmap={vmap_costs.tolist()}"
    )


@pytest.mark.filterwarnings("ignore:Initializing zero-element tensors:UserWarning")
def test_bmm_evaluator_built_for_relu_mlp_only():
    """The bmm fast path requires CrossEntropyLoss + Linear/activation
    only. Models with conv / attention / LSTM should fall back to vmap."""
    # Linear + ReLU -> bmm path
    mlp = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    bmm_mlp = BatchedLinearEvaluator.try_build(mlp, nn.CrossEntropyLoss())
    assert bmm_mlp is not None

    # Linear + Conv -> NO bmm path (Conv not handled)
    convnet = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(0, 2))
    # Using Conv2d should disqualify it
    bmm_conv = BatchedLinearEvaluator.try_build(convnet, nn.CrossEntropyLoss())
    assert bmm_conv is None


# ---------------------------------------------------------------------------
# 4.4b no retained .grad attributes (gradient-free smell)
# ---------------------------------------------------------------------------


def test_optimizer_step_does_not_set_grad():
    """PolyStepOptimizer is gradient-free; after `step()` no parameter
    should have a populated `.grad` attribute. A populated `.grad`
    indicates a leftover backward pass somewhere."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    loss_fn = nn.MSELoss()
    evaluator = NNCostEvaluator(model, loss_fn=loss_fn)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    def closure(batched_params):
        return evaluator.evaluate(batched_params, inputs, targets)

    opt = PolyStepOptimizer(model, epsilon=0.5, step_radius=0.1)
    opt.step(closure)

    leftover = [
        name for name, p in model.named_parameters() if p.grad is not None
    ]
    assert not leftover, (
        f"Gradient-free optimizer left .grad attribute(s) on: {leftover}"
    )
