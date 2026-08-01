"""Fairness in both directions.

- Contamination check: baseline implementations under
  ``experiments/baselines/`` and ``src/polystep/benchmarks/baselines.py``
  must NOT import any polystep acceleration helper. Otherwise
  the "fair" comparison silently runs PolyStep-style acceleration on
  the other side of the table.
- The other direction, which the contamination check does not cover: inside a
  fairness-mode table every gradient-free method must get the *same* candidate
  budget and the *same* search space. That is what
  ``experiments/runners/fairness.run_baseline`` is for, and the
  ``test_fairness_table_*`` tests assert it on a real table.
- ``experiments/baselines/sls_pysat.py`` (the PySAT replacement
  for the in-repo Python WalkSAT) runs and returns sensible numbers on
  a tiny 50-var random 3-SAT instance.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


_TURBO_TOKENS = (
    "apply_momentum",
    "amortize_steps",
    "amortize_ema",
    "biased_rotation",
    "anderson_depth",
    "adaptive_omega",
    "dual_momentum_beta",
    "data_dependent_init",
)

_BASELINE_DIRS = (REPO_ROOT / "experiments" / "baselines",)
_BASELINE_FILES = (REPO_ROOT / "src" / "polystep" / "benchmarks" / "baselines.py",)


def _baseline_python_files():
    files = list(_BASELINE_FILES)
    for d in _BASELINE_DIRS:
        if d.is_dir():
            for p in d.glob("*.py"):
                if p.name == "__init__.py":
                    continue
                # External baselines themselves (sls_pysat.py) do not count
                # as contamination - they only exist inside experiments/
                # baselines/ to provide alternatives.
                files.append(p)
    return files


def test_no_baseline_imports_polystep_turbo_features(require_experiments):
    """Baselines must not import polystep acceleration helpers; otherwise the
    "fair" comparison is silently using PolyStep acceleration on the
    other side of the table."""
    failures = []
    checked = []
    for path in _baseline_python_files():
        if not path.is_file():
            continue
        checked.append(path)
        src = path.read_text()
        # Look for `from polystep... import ... TOKEN` or `polystep.TOKEN`.
        # Ignore textual mentions inside docstrings - look for either
        # `import` lines or attribute access.
        for token in _TURBO_TOKENS:
            pattern = rf"(from\s+polystep[\w.]*\s+import[^\n]*\b{token}\b|polystep[\w.]*\.{token}\b)"
            if re.search(pattern, src):
                failures.append(f"{path.relative_to(REPO_ROOT)} imports/uses {token}")

    # Without this the scan passes vacuously when the baselines tree is missing or
    # renamed: an empty file list makes `not failures` trivially true.
    assert len(checked) >= 4, f"expected at least 4 baseline files to scan, found {len(checked)}"
    assert not failures, "Baseline contamination detected:\n" + "\n".join(failures)


def test_sls_pysat_baseline_runs_on_small_instance(require_experiments):
    """The PySAT replacement baseline must execute on a tiny 50-var
    instance and return a sat_ratio in [0, 1]."""
    pytest.importorskip("pysat", reason="python-sat not installed")
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.baselines.sls_pysat import run_sls_pysat
    from experiments.runners.nondiff_data import generate_maxsat_instance

    instance = generate_maxsat_instance(num_vars=50, seed=42)
    result = run_sls_pysat(
        instance=instance,
        wall_clock_seconds=2.0,
        seed=42,
        solver_name="g4",
    )
    assert 0.0 <= result["sat_ratio"] <= 1.0
    assert result["num_satisfied"] <= result["num_clauses"]
    # 50-var random 3-SAT at ratio 4.27 is satisfiable with high probability.
    assert result["sat_ratio"] > 0.85, (
        f"PySAT baseline sat_ratio = {result['sat_ratio']:.3f}, expected > 0.85 on a 50-var instance"
    )


# --- The other direction: a fairness table must actually be matched ---------------


@pytest.fixture
def fairness_table(require_experiments):
    """Run every gradient-free method on one tiny task, in fairness mode.

    Deliberately the real code path (``run_baseline``) rather than a mock: the
    assertions below are about what the runners record, so they have to run what
    the runners run.
    """
    pytest.importorskip("cma", reason="pycma drives the CMA-ES baseline")
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    import torch.nn as nn

    from experiments.runners.fairness import FAIR_METHODS, make_subspace, run_baseline
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    budget, rank, seed = 256, 4, 0
    torch.manual_seed(seed)
    X, Y = torch.randn(8, 6), torch.randint(0, 3, (8,))

    table = {}
    for method in FAIR_METHODS:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
        layout = ParamLayout.from_module(model)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        table[method] = run_baseline(
            method,
            model=model,
            layout=layout,
            loss_batch=lambda stacked: evaluator.evaluate(stacked, X, Y),
            budget=budget,
            val_fn=lambda m: -float(nn.functional.cross_entropy(m(X), Y)),
            test_fn=lambda m: -float(nn.functional.cross_entropy(m(X), Y)),
            mode="max",
            seed=seed,
            subspace=make_subspace(layout, rank=rank, seed=seed, method=method),
            subspace_rank=rank,
            probe_scale=0.5,
            log_points=8,
        )
    return budget, rank, table


def test_fairness_table_shares_one_eval_budget(fairness_table):
    """Same budget for everyone, and nobody exceeds it."""
    budget, _, table = fairness_table
    budgets = {m: r["hyperparameters"]["eval_budget"] for m, r in table.items()}
    assert set(budgets.values()) == {budget}, f"budgets differ across the table: {budgets}"
    for method, r in table.items():
        used = r["hyperparameters"]["evals_used"]
        assert used == r["metrics"]["function_evals"] <= budget, f"{method} used {used} of {budget}"


def test_fairness_table_shares_one_representation(fairness_table):
    """Same subspace class and rank, with EGGROLL's documented exception recorded."""
    _, rank, table = fairness_table
    ranks = {m: r["hyperparameters"]["subspace_rank"] for m, r in table.items()}
    assert set(ranks.values()) == {rank}, f"ranks differ across the table: {ranks}"

    classes = {m: r["hyperparameters"]["subspace_class"] for m, r in table.items()}
    # EGGROLL's rank-r A B^T needs matrix-structured coordinates, which HybridSubspace
    # does not have; FactoredSubspace is that parameterization. The exception is
    # allowed only because it is recorded in the JSON, which is what this asserts.
    assert classes.pop("eggroll") == "FactoredSubspace", classes
    assert set(classes.values()) == {"HybridSubspace"}, f"classes differ across the table: {classes}"


def test_fairness_table_records_a_plottable_trajectory(fairness_table):
    """Accuracy against CUMULATIVE candidate evaluations, the paper's figure."""
    budget, _, table = fairness_table
    for method, r in table.items():
        traj = r["step_logs"]
        assert len(traj) >= 2, f"{method} recorded {len(traj)} trajectory points"
        evals = [p["evals"] for p in traj]
        assert evals == sorted(evals), f"{method} trajectory is not monotone in evals"
        assert evals[-1] <= budget
        assert all("test_accuracy" in p and "val_accuracy" in p for p in traj)


def test_polystep_eval_budget_matches_what_polystep_spends(require_experiments):
    """The shared budget is derived from PolyStep, so it must equal what it spends."""
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    import torch.nn as nn

    from experiments.runners.fairness import make_subspace, polystep_eval_budget
    from polystep.optimizer import PolyStepOptimizer
    from polystep.transform import ParamLayout

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3))
    subspace = make_subspace(ParamLayout.from_module(model), rank=4, seed=0)
    opt = PolyStepOptimizer(model, subspace=subspace, solver="softmax", seed=0, num_probe=1)

    steps = 3
    spent = 0

    def closure(stacked):
        nonlocal spent
        n = next(iter(stacked.values())).shape[0]
        spent += n
        return torch.randn(n)

    for _ in range(steps):
        opt.step(closure)
    assert polystep_eval_budget(opt, steps) == spent


# --- Theory mode: the configuration Theorem 4.2 actually assumes -------------------


def test_theory_mode_selects_the_analysed_configuration(require_experiments):
    """Jitter on, schedules off, orthoplex, no acceleration -- and the tuned config
    is left alone, because the point is to report the gap between the two."""
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.fairness import THEORY_GAMMA, THEORY_JITTER, apply_theory_mode
    from polystep.epsilon import PowerDecay

    tuned = {
        "rank": 8,
        "epsilon_init": 10.0,
        "epsilon_target": 0.1,
        "step_radius_init": 5.0,
        "step_radius_target": 1.0,
        "probe_radius_init": 10.0,
        "probe_radius_target": 2.0,
        "amortize_steps": 3,
        "amortize_ema": 0.7,
        "use_momentum": True,
        "biased_rotation": True,
    }
    theory = apply_theory_mode(tuned)

    assert tuned["amortize_steps"] == 3, "apply_theory_mode must not mutate its input"
    assert theory["probe_radius_jitter"] == THEORY_JITTER
    assert theory["probe_radius_jitter_dist"] == "smooth"
    assert theory["polytope_type"] == "orthoplex"
    assert theory["biased_rotation"] is False
    assert theory["use_momentum"] is False
    assert theory["amortize_steps"] == 1 and theory["anderson_depth"] == 0
    # Flat epsilon, flat probe radius, decaying step radius r_0 (t+1)^-(1/2+gamma).
    assert theory["epsilon"] == 0.1 and "epsilon_init" not in theory
    assert theory["probe_radius"] == 2.0 and "probe_radius_init" not in theory
    r = theory["step_radius"]
    assert isinstance(r, PowerDecay) and r.gamma == THEORY_GAMMA
    assert r.at(0) == 5.0
    assert r.at(3) == pytest.approx(5.0 * 4 ** -(0.5 + THEORY_GAMMA))
