"""Numerical stability boundary tests.

- BF16 outer autocast must NOT bleed into the solver internals
  (the solvers wrap their hot loop in ``autocast(enabled=False)``).
- Wide eps x cost_range grid stays finite.
- A single ``+Inf`` cost entry must drive that transport entry to 0
  without NaN-propagating across the rest of the row.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from polystep import SinkhornSolver, SoftmaxSolver


# ---------------------------------------------------------------------------
# 3.1 BF16 / FP16 outer autocast must not collapse solver internals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_cls", [SoftmaxSolver, SinkhornSolver])
def test_solver_promotes_bf16_inputs_to_fp32_internally(solver_cls):
    """A BF16 cost matrix must be promoted to FP32 inside the solver
    so log-sum-exp / row-max-subtract retain the 23 mantissa bits
    needed for high cost spread."""
    P, V = 12, 24
    torch.manual_seed(0)
    C = torch.randn(P, V, dtype=torch.bfloat16) * 30.0
    solver = solver_cls(epsilon=0.1) if solver_cls is SoftmaxSolver else solver_cls(
        epsilon=0.1, max_iterations=200, threshold=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(C)
    # Result transport should be FP32 (promoted) and finite.
    assert result.matrix.dtype == torch.float32
    assert torch.isfinite(result.matrix).all()


@pytest.mark.parametrize("solver_cls", [SoftmaxSolver, SinkhornSolver])
def test_solver_disables_outer_bf16_autocast(solver_cls):
    """Run the solver inside an outer ``autocast(bfloat16)`` and verify
    the result is still FP32. Without the autocast-disable wrapper
    inside the solver, the outer autocast would silently downcast
    intermediates and the transport matrix would come back BF16."""
    if not hasattr(torch.amp, "autocast"):
        pytest.skip("torch.amp.autocast not available")

    P, V = 8, 16
    torch.manual_seed(0)
    C = torch.randn(P, V) * 5.0
    solver = solver_cls(epsilon=0.5) if solver_cls is SoftmaxSolver else solver_cls(
        epsilon=0.5, max_iterations=200, threshold=1e-4,
    )

    device_type = "cpu"  # autocast(bfloat16) on CPU is universally available
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.amp.autocast(device_type, dtype=torch.bfloat16):
            result = solver.solve(C)

    assert result.matrix.dtype == torch.float32, (
        f"outer autocast leaked into solver: result dtype = {result.matrix.dtype}"
    )


# ---------------------------------------------------------------------------
# 3.2a Overflow grid (replicates 2A.1 / 2B.1a but for SinkhornSolver)
# ---------------------------------------------------------------------------


def test_sinkhorn_no_nan_across_eps_cost_grid():
    """Across (eps, cost_range) in {(0.01,10), (0.1,100), (1,1000)},
    SinkhornSolver returns finite duals or warns and zeros them."""
    P, V = 12, 24
    torch.manual_seed(0)

    cells = [
        (0.01, 10.0), (0.1, 10.0), (1.0, 10.0),
        (0.01, 100.0), (0.1, 100.0), (1.0, 100.0),
        (0.1, 1000.0), (1.0, 1000.0),
    ]
    failures = []
    for eps, scale in cells:
        C = torch.randn(P, V) * scale
        solver = SinkhornSolver(
            epsilon=eps, max_iterations=300, threshold=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = solver.solve(C)
        if not (torch.isfinite(result.f).all() and torch.isfinite(result.g).all()):
            failures.append(f"eps={eps}, range={scale}: non-finite duals")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# 3.2b +Inf entry handling: row stays valid, masked entry transports to 0
# ---------------------------------------------------------------------------


def test_softmax_handles_single_inf_entry():
    """A +Inf cost entry models a hard constraint: that vertex must not
    be picked. Pre-fix softmax produced NaN because the whole row went
    to -Inf in `-C/eps`. Post-fix the +Inf is replaced with a large
    finite penalty so the masked entry still gets near-zero weight
    while the rest of the row stays valid."""
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V)
    C[0, 3] = float("inf")
    solver = SoftmaxSolver(epsilon=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(C)

    assert torch.isfinite(result.matrix).all(), (
        "softmax produced NaN/Inf despite +Inf cost entry"
    )
    masked_weight = result.matrix[0, 3].item()
    other_weights = result.matrix[0, [i for i in range(V) if i != 3]]
    assert masked_weight < other_weights.min().item() / 10, (
        f"masked entry weight {masked_weight} not driven near zero; "
        f"row = {result.matrix[0].tolist()}"
    )


def test_sinkhorn_handles_single_inf_entry():
    """SinkhornSolver already replaces +Inf entries (sinkhorn.py:213-228);
    this test pins that behavior."""
    P, V = 4, 8
    torch.manual_seed(0)
    C = torch.randn(P, V)
    C[0, 3] = float("inf")
    solver = SinkhornSolver(epsilon=0.5, max_iterations=100, threshold=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(C)
    assert torch.isfinite(result.matrix).all()
    assert torch.isfinite(result.f).all()
    assert torch.isfinite(result.g).all()
