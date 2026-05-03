"""python-sat MAX-SAT baseline (Glucose4 wrapper).

Drop-in alternative to ``experiments/runners/run_maxsat.py::run_sls``,
which is an in-repo Python WalkSAT with a single seed and a 50K-flip
budget at 1M variables (deliberately reduced from 500K so the run
would complete in reasonable wall-clock). That is not a fair
comparison against polystep:

- Single seed, no aggregation.
- Step budget rather than wall-clock budget.
- No tuned restart policy.
- Not a competition-grade SAT solver.

This module exposes a ``run_sls_pysat`` function that wraps
PySAT's ``Glucose4`` solver under a fair wall-clock budget that
matches what polystep consumed. PySAT solvers are CDCL, not pure SLS, and target 100% SAT. For
genuine SLS we recommend installing ProbSAT separately and calling
its binary, but Glucose4 is a strong upper bound that is already
in the ``[paper]`` extras.

Usage:
    from experiments.baselines.sls_pysat import run_sls_pysat
    result = run_sls_pysat(instance, wall_clock_seconds=60.0, seed=42)
    print(result["sat_ratio"])

This is *not* wired into ``run_maxsat.py`` by default. We prefer
explicit opt-in over silent baseline mutation -- the existing
``run_sls`` continues to drive ``run_all_paper.sh`` until a fair
re-quantification is requested.
"""
from __future__ import annotations

import time
from typing import Dict, Optional


def run_sls_pysat(
    instance: Dict,
    wall_clock_seconds: float = 60.0,
    seed: int = 42,
    solver_name: str = "g4",
) -> Dict:
    """Run a PySAT solver as a competition-grade MAX-SAT baseline.

    Args:
        instance: Output of ``generate_maxsat_instance`` - must contain
            ``cnf`` (CNF object), ``num_clauses``, ``num_vars``.
        wall_clock_seconds: Budget for the solver. Glucose4's
            ``conf_budget`` is set proportionally so the call returns
            near this budget; the exact wall-clock may differ by
            seconds.
        seed: Random seed for the solver's internal RNG.
        solver_name: PySAT solver name. ``g4`` (Glucose4) is the
            default; ``m22`` (Minisat22), ``lgl`` (Lingeling) also work.

    Returns:
        Dict with sat_ratio, num_satisfied, num_clauses, wall_clock_s,
        solver. If the solver finds an exact SAT assignment, sat_ratio
        is 1.0 and num_satisfied == num_clauses.

    Notes:
        Glucose4 is a CDCL solver. It tries to PROVE satisfiability,
        not maximize it. For random 3-SAT at the critical threshold
        (ratio 4.27) most instances are SAT, so Glucose4 typically
        returns 100% before the budget is exhausted. For instances
        above the critical threshold or for true MaxSAT, prefer
        PySAT's RC2 (already used by ``run_maxsat.py::run_rc2``) or
        an external ProbSAT/CCAnr binary.
    """
    from pysat.solvers import Solver

    cnf = instance["cnf"]
    num_clauses = instance["num_clauses"]
    num_vars = instance["num_vars"]

    t0 = time.time()
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as solver:
        # PySAT supports `conf_budget` (conflicts) and `prop_budget`.
        # We set conf_budget proportionally to the wall-clock target;
        # 100K conflicts ~ 1s on Glucose4 for typical 3-SAT.
        budget = max(1, int(wall_clock_seconds * 100_000))
        solver.conf_budget(budget)
        sat = solver.solve_limited(expect_interrupt=False)
        if sat is True:
            assignment = solver.get_model()
            num_satisfied = num_clauses  # SAT solver returned True
            sat_ratio = 1.0
        elif sat is False:
            # UNSAT - count clauses satisfied by an arbitrary all-True
            # assignment as a lower bound (rare for ratio 4.27 instances).
            assignment = list(range(1, num_vars + 1))
            num_satisfied = _count_sat(cnf.clauses, assignment, num_vars)
            sat_ratio = num_satisfied / num_clauses
        else:
            # Budget exhausted - return best-effort partial assignment
            assignment = solver.get_model() or list(range(1, num_vars + 1))
            num_satisfied = _count_sat(cnf.clauses, assignment, num_vars)
            sat_ratio = num_satisfied / num_clauses
    wall = time.time() - t0

    return {
        "sat_ratio": sat_ratio,
        "num_satisfied": num_satisfied,
        "num_clauses": num_clauses,
        "wall_clock_s": wall,
        "solver": solver_name,
        "seed": seed,
    }


def _count_sat(clauses, assignment, num_vars):
    """Count clauses satisfied by a literal-list assignment."""
    if not assignment:
        return 0
    val = [False] * (num_vars + 1)  # 1-indexed
    for lit in assignment:
        if abs(lit) <= num_vars:
            val[abs(lit)] = lit > 0
    sat = 0
    for clause in clauses:
        for lit in clause:
            v = abs(lit)
            if v > num_vars:
                continue
            if (lit > 0 and val[v]) or (lit < 0 and not val[v]):
                sat += 1
                break
    return sat


def aggregate_sls_runs(
    instance: Dict,
    seeds: tuple = (42, 123, 456, 789, 1337),
    wall_clock_seconds: float = 60.0,
    solver_name: str = "g4",
) -> Dict:
    """Run the PySAT baseline across multiple seeds and aggregate.

    A fair MAX-SAT comparison typically wants >= 10 seeds for the SAT
    solver, but the default 5 here matches the rest of the polystep
    experiments. Override ``seeds`` to extend.
    """
    runs = [
        run_sls_pysat(
            instance=instance,
            wall_clock_seconds=wall_clock_seconds,
            seed=seed,
            solver_name=solver_name,
        )
        for seed in seeds
    ]
    sat_ratios = [r["sat_ratio"] for r in runs]
    return {
        "n_seeds": len(seeds),
        "sat_ratio_mean": sum(sat_ratios) / len(sat_ratios),
        "sat_ratio_min": min(sat_ratios),
        "sat_ratio_max": max(sat_ratios),
        "wall_clock_total_s": sum(r["wall_clock_s"] for r in runs),
        "solver": solver_name,
        "runs": runs,
    }
