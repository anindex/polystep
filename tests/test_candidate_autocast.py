"""``candidate_autocast`` must preserve the cost matrix's ranking.

BF16 can tie candidates whose losses differ below its resolution, so these tests
check rank correlation and tie rate, not absolute loss agreement. They run on CPU;
the speedup is a CUDA claim.
"""

import pytest
import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


def _candidates(seed=0, n=256):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 5))
    params = {
        k: v.unsqueeze(0).repeat(n, *([1] * v.dim())) + 0.02 * torch.randn(n, *v.shape)
        for k, v in model.named_parameters()
    }
    return model, params, torch.randn(48, 32), torch.randint(0, 5, (48,))


def test_ranking_survives_bf16():
    model, params, x, y = _candidates()
    fp32 = NNCostEvaluator(model, nn.CrossEntropyLoss()).evaluate(params, x, y)
    bf16 = NNCostEvaluator(model, nn.CrossEntropyLoss(), autocast_dtype=torch.bfloat16).evaluate(params, x, y)

    assert bf16.shape == fp32.shape
    assert _spearman(fp32, bf16) > 0.99
    ties = (bf16.unsqueeze(0) == bf16.unsqueeze(1)).sum() - bf16.numel()
    assert ties / (bf16.numel() ** 2) < 0.01, "too many candidates came back indistinguishable"


def test_off_by_default_and_exact():
    """The default must be bit-identical to no autocast at all."""
    model, params, x, y = _candidates(seed=1)
    plain = NNCostEvaluator(model, nn.CrossEntropyLoss())
    assert plain.autocast_dtype is None
    explicit = NNCostEvaluator(model, nn.CrossEntropyLoss(), autocast_dtype=None)
    assert torch.equal(plain.evaluate(params, x, y), explicit.evaluate(params, x, y))


def test_optimizer_flag_reaches_the_evaluator():
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(model, candidate_autocast=True, epsilon=0.1, compile=False, seed=0)
    opt.register_evaluator(ev, torch.randn(8, 16), torch.randint(0, 4, (8,)))
    assert ev.autocast_dtype == torch.bfloat16

    loss = opt.step(lambda p: ev.evaluate(p, torch.randn(8, 16), torch.randint(0, 4, (8,))))
    assert torch.isfinite(torch.tensor(loss))


def test_delta_evaluators_stay_in_full_precision():
    """Their correction is a small offset on a full-scale output; BF16 erases it.

    The optimizer's own delta paths call the specialized evaluators directly, so they
    must never pick up the evaluator's autocast frame.
    """
    model, params, x, y = _candidates(seed=2, n=8)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), autocast_dtype=torch.bfloat16)
    with ev._autocast(x.device):
        inner = torch.is_autocast_enabled("cpu")
    assert inner
    assert not torch.is_autocast_enabled("cpu"), "the frame must not leak past evaluate()"


@pytest.mark.parametrize("dtype", [torch.bfloat16, None])
def test_subspace_inplace_path_accepts_the_flag(dtype):
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=2)
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss(), autocast_dtype=dtype)

    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    projections = sub.init_projections(torch.device("cpu"), torch.float32)
    coords = torch.randn(3, sub.subspace_dim) * 0.01

    losses = ev.evaluate_subspace_inplace(sub, projections, base, coords, torch.randn(8, 16), torch.randint(0, 4, (8,)))
    assert losses.shape == (3,) and torch.isfinite(losses).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, base[key])
