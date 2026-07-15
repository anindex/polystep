"""The fused block-diagonal projection must be rebuilt only when the projection
rotates, not on every step.

With the default rotation_interval=0 the HybridSubspace returns the same
projection dict each step, so the block_diag rebuild (a full dense reconstruction)
must not run after the one-time build at construction.
"""

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


def test_fused_projection_not_rebuilt_when_static():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4)
    opt = PolyStepOptimizer(
        model,
        subspace=sub,
        solver="softmax",
        epsilon=0.5,
        step_radius=0.3,
        probe_radius=1.0,
    )

    # Count rebuilds after construction (the init build has already happened).
    calls = {"n": 0}
    original = sub.build_fused_projection

    def counting(projections):
        calls["n"] += 1
        return original(projections)

    sub.build_fused_projection = counting

    x = torch.randn(24, 32)
    y = torch.randint(0, 4, (24,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    for _ in range(5):
        opt.step(lambda s: ev.evaluate(s, x, y))

    assert calls["n"] == 0, f"expected no rebuilds with static projections, got {calls['n']}"


if __name__ == "__main__":
    test_fused_projection_not_rebuilt_when_static()
    print("ok")
