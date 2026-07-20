"""Mixed-precision (bf16) stepping and barycentric normalization.

mixed_precision=True mixes fp32 OT weights with bf16 geometry; the barycentric and
fused-softmax matmuls, the HybridSubspace QR, and the cost evaluator must all
handle it. The barycentric projection also divides by the realized row sum, so an
unconverged Sinkhorn plan gives a translation-invariant step.
"""

import pytest
import torch
import torch.nn as nn

from polystep._compiled import _barycentric_projection, _fused_softmax_project
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.optimizer import PolyStepOptimizer
from polystep.transform import ParamLayout


def test_barycentric_projection_mixed_dtype_no_crash():
    """fp32 transport weights against bf16 vertices must not raise."""
    b, V, d = 3, 6, 3
    transport = torch.rand(b, V)  # fp32
    a = torch.full((b,), 1.0 / b)
    X_vertices = torch.randn(b, V, d, dtype=torch.bfloat16)
    out = _barycentric_projection(transport, a, X_vertices)
    assert out.dtype == torch.bfloat16
    assert out.shape == (b, d)
    assert torch.isfinite(out.float()).all()


def test_fused_softmax_project_mixed_dtype_no_crash():
    """fp32 cost against bf16 geometry must not raise."""
    b, V, d = 3, 6, 3
    C = torch.randn(b, V)  # fp32 cost
    a = torch.full((b,), 1.0 / b)
    pv = torch.randn(V, d, dtype=torch.bfloat16)
    rot = torch.eye(d, dtype=torch.bfloat16).expand(b, d, d).contiguous()
    X = torch.randn(b, d, dtype=torch.bfloat16)
    X_new, transport = _fused_softmax_project(C, 0.1, a, pv, rot, 1.0, X, scale_cost_mean=False)
    assert X_new.dtype == torch.bfloat16
    assert torch.isfinite(X_new.float()).all()


def test_barycentric_softmax_path_unchanged():
    """When rows already sum to ``a`` (softmax path), /rowsum == /a exactly."""
    b, V, d = 4, 5, 3
    a = torch.full((b,), 1.0 / b)
    transport = torch.softmax(-torch.randn(b, V), dim=-1) * a.unsqueeze(-1)
    X_vertices = torch.randn(b, V, d)
    new = _barycentric_projection(transport, a, X_vertices)
    old = torch.einsum("bkd,bk->bd", X_vertices, transport / a.unsqueeze(-1))
    assert torch.allclose(new, old, atol=1e-6)


def test_barycentric_translation_invariant_unconverged_plan():
    """An unconverged plan (rows not summing to a) must give a translation-invariant step."""
    b, V, d = 3, 6, 3
    a = torch.full((b,), 1.0 / b)
    plan = torch.rand(b, V)  # rows deliberately do NOT sum to a
    origin = torch.randn(b, d)
    shift = torch.tensor([5.0, -2.0, 1.0])
    X_o = torch.randn(b, V, d) + origin.unsqueeze(1)
    X_s = X_o + shift  # shift every vertex by the same constant
    step_o = _barycentric_projection(plan, a, X_o)
    step_s = _barycentric_projection(plan, a, X_s)
    # Shifting all vertices by c must shift the barycenter by exactly c.
    assert torch.allclose(step_s - step_o, shift.expand(b, d), atol=1e-5)


def _run_mixed_precision_steps(solver: str, n_steps: int = 3) -> float:
    model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
    layout = ParamLayout.from_module(model)
    sub = HybridSubspace.from_layout(layout, rank=4)
    opt = PolyStepOptimizer(
        model,
        subspace=sub,
        solver=solver,
        epsilon=0.5,
        step_radius=0.5,
        probe_radius=1.0,
        mixed_precision=True,
    )
    x = torch.randn(32, 16)
    y = torch.randint(0, 3, (32,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    loss = float("nan")
    for _ in range(n_steps):
        loss = float(opt.step(lambda s: ev.evaluate(s, x, y)))
    return loss


@pytest.mark.parametrize("solver", ["softmax", "sinkhorn"])
def test_optimizer_mixed_precision_step(solver):
    """End-to-end mixed_precision=True must run on CPU with HybridSubspace."""
    loss = _run_mixed_precision_steps(solver)
    assert loss == loss  # not NaN


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
