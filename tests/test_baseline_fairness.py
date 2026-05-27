"""Baseline-fairness regression tests.

- Contamination check: baseline implementations under
  ``experiments/baselines/`` and ``src/polystep/benchmarks/baselines.py``
  must NOT import any polystep turbo / acceleration helper. Otherwise
  the "fair" comparison silently runs PolyStep-style acceleration on
  the other side of the table.
- ``experiments/baselines/sls_pysat.py`` (the PySAT replacement
  for the in-repo Python WalkSAT) runs and returns sensible numbers on
  a tiny 50-var random 3-SAT instance.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Turbo-feature contamination check
# ---------------------------------------------------------------------------


_TURBO_TOKENS = (
    "apply_momentum",
    "update_adaptive_radius",
    "amortize_steps",
    "amortize_ema",
    "biased_rotation",
    "anderson_depth",
    "adaptive_omega",
    "dual_momentum_beta",
    "data_dependent_init",
)

_BASELINE_DIRS = (
    REPO_ROOT / "experiments" / "baselines",
)
_BASELINE_FILES = (
    REPO_ROOT / "src" / "polystep" / "benchmarks" / "baselines.py",
)


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


def test_no_baseline_imports_polystep_turbo_features():
    """Baselines must not import polystep turbo helpers; otherwise the
    "fair" comparison is silently using PolyStep acceleration on the
    other side of the table."""
    failures = []
    for path in _baseline_python_files():
        if not path.is_file():
            continue
        src = path.read_text()
        # Look for `from polystep... import ... TOKEN` or `polystep.TOKEN`.
        # Ignore textual mentions inside docstrings - look for either
        # `import` lines or attribute access.
        for token in _TURBO_TOKENS:
            pattern = rf"(from\s+polystep[\w.]*\s+import[^\n]*\b{token}\b|polystep[\w.]*\.{token}\b)"
            if re.search(pattern, src):
                failures.append(f"{path.relative_to(REPO_ROOT)} imports/uses {token}")

    assert not failures, "Baseline contamination detected:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# PySAT baseline smoke
# ---------------------------------------------------------------------------


def test_sls_pysat_baseline_runs_on_small_instance():
    """The PySAT replacement baseline must execute on a tiny 50-var
    instance and return a sat_ratio in [0, 1]."""
    pytest.importorskip("pysat", reason="python-sat not installed")
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.baselines.sls_pysat import run_sls_pysat
    from experiments.runners.nondiff_data import generate_maxsat_instance

    instance = generate_maxsat_instance(num_vars=50, seed=42)
    result = run_sls_pysat(
        instance=instance, wall_clock_seconds=2.0, seed=42, solver_name="g4",
    )
    assert 0.0 <= result["sat_ratio"] <= 1.0
    assert result["num_satisfied"] <= result["num_clauses"]
    # 50-var random 3-SAT at ratio 4.27 is satisfiable with high probability.
    assert result["sat_ratio"] > 0.85, (
        f"PySAT baseline sat_ratio = {result['sat_ratio']:.3f}, "
        f"expected > 0.85 on a 50-var instance"
    )
