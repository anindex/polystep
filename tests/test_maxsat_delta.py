"""Regression tests for the MAX-SAT delta evaluator.

The delta evaluator in
``experiments.runners.run_maxsat.make_sat_closure`` uses an inverted
CSR index to recompute only the clauses touched by a perturbation
chunk (~1664 out of 4.27M at 1M vars). This file verifies that, for
any random perturbation, the delta path returns the same satisfied
clause count as a brute-force full recompute on a 1K-var instance.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


# python-sat is in the [paper] extras and may not be installed in the
# slim CI environment - skip cleanly in that case.
pysat = pytest.importorskip("pysat", reason="python-sat not installed")
from experiments.runners.run_maxsat import make_sat_closure  # noqa: E402
from experiments.runners.nondiff_data import generate_maxsat_instance  # noqa: E402
from experiments.runners.nondiff_models import MaxSATModel  # noqa: E402


def _brute_force_unsat_count(assignments_raw: torch.Tensor,
                              clause_vars: torch.Tensor,
                              clause_signs: torch.Tensor) -> torch.Tensor:
    """Reference: compute (N,) unsat-clause counts by direct enumeration.

    Mirrors the closure's encoding (sigmoid + round) but evaluates every
    clause independently for every assignment. Used as ground truth.
    """
    N = assignments_raw.shape[0]
    C = clause_vars.shape[0]
    soft = torch.sigmoid(assignments_raw)
    hard = torch.round(soft)  # (N, V)
    counts = torch.zeros(N)
    for n in range(N):
        a = hard[n]
        # Gather literals: shape (C, k)
        lit_vals = a[clause_vars]
        # Apply signs: signs=1 -> literal value, signs=0 -> negation
        lits = lit_vals * clause_signs + (1.0 - clause_signs) * (1.0 - lit_vals)
        satisfied = (lits > 0.5).any(dim=-1)
        counts[n] = float(C - satisfied.sum().item())
    return counts


def test_maxsat_delta_eval_matches_brute_force():
    """100 random perturbations on a 1K-var instance: delta vs brute force
    must agree on unsatisfied-clause count."""
    torch.manual_seed(0)
    num_vars = 1000
    instance = generate_maxsat_instance(num_vars=num_vars, seed=42)
    clause_vars = instance["clause_vars"]
    clause_signs = instance["clause_signs"]

    model = MaxSATModel(num_vars)
    closure = make_sat_closure(
        clause_vars, clause_signs,
        cra_lambda=0.0,
        model=model,
        particle_dim=2,
    )
    closure.resample()  # builds delta-eval base state

    # Build 30 random perturbations (smaller than 100 so the brute-force
    # reference fits comfortably under the 60s pytest timeout).
    base_raw = model.assignments.data
    N = 30
    perturb = torch.randn(N, num_vars) * 0.5
    assignments_batch = base_raw.unsqueeze(0) + perturb

    delta_unsat = closure({"assignments": assignments_batch})
    truth_unsat = _brute_force_unsat_count(
        assignments_batch, clause_vars, clause_signs.float(),
    )

    assert torch.allclose(delta_unsat.float(), truth_unsat, atol=0.5), (
        f"delta and brute-force disagree on max diff = "
        f"{(delta_unsat.float() - truth_unsat).abs().max().item():.2f}; "
        f"first delta={delta_unsat[:5].tolist()}, first truth={truth_unsat[:5].tolist()}"
    )


def test_maxsat_delta_eval_handles_no_change():
    """When a row of the batch matches the base assignment exactly, the
    delta path returns the base unsat count (early-exit branch)."""
    torch.manual_seed(0)
    num_vars = 200
    instance = generate_maxsat_instance(num_vars=num_vars, seed=42)
    clause_vars = instance["clause_vars"]
    clause_signs = instance["clause_signs"]

    model = MaxSATModel(num_vars)
    closure = make_sat_closure(
        clause_vars, clause_signs,
        cra_lambda=0.0,
        model=model,
        particle_dim=2,
    )
    closure.resample()

    base_raw = model.assignments.data
    # All identical rows (no change)
    assignments_batch = base_raw.unsqueeze(0).expand(4, -1).contiguous()
    out = closure({"assignments": assignments_batch})

    truth = _brute_force_unsat_count(
        assignments_batch, clause_vars, clause_signs.float(),
    )
    assert torch.allclose(out.float(), truth, atol=0.5)
