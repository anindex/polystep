"""Fairness checks in both directions.

- Contamination: baselines must not import polystep acceleration helpers.
- Budget matching: every gradient-free method in a fairness table must get the
  same candidate budget and the same search space.
- The PySAT SLS baseline must run and return sensible numbers on a tiny 3-SAT.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# experiments/ is the paper reproduction harness and is not part of the distributed
# package, so this module is unimportable when the tests run from an sdist.
if not (REPO_ROOT / "experiments" / "runners").is_dir():
    pytest.skip("experiments/runners not present (running outside the repo)", allow_module_level=True)


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

# src/polystep/baselines is the tree the fairness tables actually run; the >= 4
# guard below keeps the scan from passing vacuously on a missing tree.
_BASELINE_DIRS = (
    REPO_ROOT / "experiments" / "baselines",
    REPO_ROOT / "src" / "polystep" / "baselines",
)


def _baseline_python_files():
    files = []
    for d in _BASELINE_DIRS:
        if d.is_dir():
            for p in d.glob("*.py"):
                if p.name == "__init__.py":
                    continue
                # External baselines (sls_pysat.py) only provide alternatives.
                files.append(p)
    return files


def test_no_baseline_imports_polystep_turbo_features():
    """Baselines must not import polystep acceleration helpers (contamination check)."""
    failures = []
    checked = []
    for path in _baseline_python_files():
        if not path.is_file():
            continue
        checked.append(path)
        src = path.read_text()
        # Match `from polystep... import TOKEN` or `polystep.TOKEN`, not docstring mentions.
        for token in _TURBO_TOKENS:
            pattern = rf"(from\s+polystep[\w.]*\s+import[^\n]*\b{token}\b|polystep[\w.]*\.{token}\b)"
            if re.search(pattern, src):
                failures.append(f"{path.relative_to(REPO_ROOT)} imports/uses {token}")

    # Guard against a vacuous pass when the baselines tree is missing or renamed.
    assert len(checked) >= 4, f"expected at least 4 baseline files to scan, found {len(checked)}"
    assert not failures, "Baseline contamination detected:\n" + "\n".join(failures)


def test_sls_pysat_baseline_runs_on_small_instance():
    """The PySAT baseline must run on a tiny 50-var instance and return sat_ratio in [0, 1]."""
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


# --- Fairness tables: every method must actually be matched -------------------------


@pytest.fixture
def fairness_table():
    """Run every gradient-free method on one tiny task through the real ``run_baseline`` path."""
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
        assert all("val_accuracy" in p for p in traj)
        # Validation curve only: a per-point test score would leak test into the paper's figure.
        assert not any("test_accuracy" in p for p in traj), f"{method} logs test per point"
        # Each point records whether validation scored the iterate or the best candidate.
        assert all(p.get("scored_at") in {"iterate", "best_x"} for p in traj)


def test_polystep_eval_budget_matches_what_polystep_spends():
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


# --- Theory mode: the unaccelerated reference configuration ------------------------


def test_theory_mode_selects_the_analysed_configuration():
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
    # Assert radii in physical units: a bare ``probe_radius`` is an epsilon
    # multiplier, while a ``*_init`` pair is the physical radius itself.
    from polystep.epsilon import resolve_radius

    assert theory["epsilon"] == 0.1 and "epsilon_init" not in theory
    assert "probe_radius_init" not in theory and "probe_radius_target" not in theory
    assert resolve_radius(theory["probe_radius"], 0, theory["epsilon"]) == pytest.approx(2.0)
    assert theory["probe_radius_realized"] == pytest.approx(2.0)
    r = theory["step_radius"]
    assert isinstance(r, PowerDecay) and r.gamma == THEORY_GAMMA
    assert resolve_radius(r, 0, theory["epsilon"]) == pytest.approx(5.0)
    assert r.at(3) == pytest.approx(5.0 * 4 ** -(0.5 + THEORY_GAMMA))


def test_theory_mode_preserves_the_physical_radii_from_a_scalar_config():
    """A scalar radius is an epsilon multiplier and must transplant at the same physical radius."""
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.fairness import apply_theory_mode
    from polystep.epsilon import resolve_radius

    tuned = {"epsilon": 0.5, "step_radius": 1.0, "probe_radius": 1.0}
    theory = apply_theory_mode(tuned)
    eps = theory["epsilon"]
    assert resolve_radius(theory["step_radius"], 0, eps) == pytest.approx(0.5)
    assert resolve_radius(theory["probe_radius"], 0, eps) == pytest.approx(0.5)


def test_polystep_budget_charges_only_the_ot_steps():
    """``amortize_steps=a`` means PolyStep is charged for probes on 1 step in ``a`` (ceiled)."""
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.runners.fairness import polystep_eval_budget

    class _Fake:
        def __init__(self, a):
            self.amortize_steps = a
            self._state = type("S", (), {"X": torch.zeros(4, 2)})()
            self._polytope_vertices = torch.zeros(3, 2)
            self.num_probe = 1

    assert polystep_eval_budget(_Fake(1), 30) == 12 * 30
    assert polystep_eval_budget(_Fake(3), 30) == 12 * 10
    # ceil: 31 steps at a=3 is 11 OT steps, not 10.
    assert polystep_eval_budget(_Fake(3), 31) == 12 * 11


def test_probe_scale_is_per_coordinate():
    """PolyStep's radius is a displacement norm; a baseline's sigma is per-coordinate.

    ``dim`` divides it out so both probes have the same displacement norm.
    """
    import math

    from experiments.runners.fairness import probe_scale_of

    cfg = {"probe_radius": 2.0}
    dim = 1242
    assert probe_scale_of(cfg, dim=dim) == pytest.approx(2.0 / math.sqrt(dim))
    # A Gaussian at the returned sigma has the radius back as its norm.
    assert probe_scale_of(cfg, dim=dim) * math.sqrt(dim) == pytest.approx(2.0)
    # A scheduled radius transfers its floor, still per-coordinate.
    sched = {"probe_radius_init": 10.0, "probe_radius_target": 2.0}
    assert probe_scale_of(sched, dim=dim) == pytest.approx(2.0 / math.sqrt(dim))
    # ...and epsilon must NOT be applied to a schedule: the runner reads an
    # init/target pair as the physical radius.
    sched_eps = {"probe_radius_init": 10.0, "probe_radius_target": 2.0, "epsilon_target": 0.1}
    assert probe_scale_of(sched_eps, dim=dim) == pytest.approx(2.0 / math.sqrt(dim))
    # A *scalar* radius is a multiplier on epsilon, and that rule still holds.
    assert probe_scale_of({"probe_radius": 2.0, "epsilon": 0.5}, dim=dim) == pytest.approx(1.0 / math.sqrt(dim))
    # No dim means no division, for callers with no dimension to hand.
    assert probe_scale_of(cfg) == 2.0


def test_runners_pass_dim_to_probe_scale():
    """Every runner that builds a table must divide the radius by its search dimension.

    Parsed with ``ast`` rather than grepped, so spacing and intermediate variables
    cannot hide a call.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "experiments"
    paths = sorted(root.glob("runners/run_*.py")) + sorted(root.glob("scripts/*.py"))
    checked = 0
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "probe_scale_of":
                continue
            checked += 1
            assert any(kw.arg == "dim" for kw in node.keywords), (
                f"{path.name}:{node.lineno} calls probe_scale_of without dim=, which "
                "hands the baseline PolyStep's radius as a per-coordinate sigma and "
                "makes it probe sqrt(dim) times too far"
            )
    assert checked >= 4, f"expected to find the known call sites, found {checked}"


def test_reseed_loaders_makes_repeated_iteration_reproducible():
    """A shared loader must give every run the same minibatch stream.

    ``seeded_loader_kwargs`` seeds each loader's generator once, at construction, so
    the shuffle order stops depending on global RNG consumption. It keeps depending on
    how many times the loader has already been iterated, because the generator advances
    per epoch. A sweep that builds its loaders once and reuses them across cells
    therefore trains cell *i* on the *i*-th stream and ranks configurations partly by
    their position in the sweep.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.runners.common import reseed_loaders
    from polystep.benchmarks.utils import seeded_loader_kwargs

    data = TensorDataset(torch.arange(64).unsqueeze(1).float(), torch.zeros(64))
    loader = DataLoader(data, batch_size=8, shuffle=True, **seeded_loader_kwargs(42))

    def order():
        return torch.cat([x.flatten() for x, _ in loader]).tolist()

    first, second = order(), order()
    assert first != second, "expected the generator to advance between epochs; without that this test asserts nothing"

    reseed_loaders(42, loader)
    assert order() == first
    reseed_loaders(42, loader)
    assert order() == first


def test_reseed_loaders_tolerates_none_and_generatorless_loaders():
    """Runners pass val/test loaders that may be None or unshuffled."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.runners.common import reseed_loaders

    plain = DataLoader(TensorDataset(torch.zeros(4, 1), torch.zeros(4)), batch_size=2)
    reseed_loaders(7, None, plain)  # must not raise


def test_eggroll_factors_over_factored_coordinates_only():
    """EGGROLL's low-rank sampler needs ``shapes`` to say where the matrices are.

    ``FactoredSubspace``'s projected coordinates *are* a ``(d_out, rank)`` matrix;
    ``HybridSubspace``'s are a QR'd Gaussian projection with no matrix structure. Both
    mark projected specs with ``is_projected``, so reporting every projected block as a
    matrix would make EGGROLL factor Hybrid coordinates that do not factor, and
    reporting none of them made the arm labelled EGGROLL run dense Gaussian ES while
    still paying FactoredSubspace's dimension penalty.
    """
    import math

    import torch.nn as nn

    from polystep.baselines.core import Objective
    from polystep.baselines.methods import _lowrank_noise
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    layout = ParamLayout.from_module(model)
    base = {k: v.detach() for k, v in model.state_dict().items()}

    def loss_batch(sd):
        return torch.zeros(next(iter(sd.values())).shape[0])

    factored = FactoredSubspace.from_layout(layout, rank=4, seed=0)
    obj = Objective.from_subspace(factored, base, loss_batch, budget=100)
    assert obj.shapes[0] == (32, 4) and obj.shapes[2] == (10, 4)
    assert obj.shapes[1] == (32,) and obj.shapes[3] == (10,), "1D entries stay flat"

    hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)
    flat = Objective.from_subspace(hybrid, base, loss_batch, budget=100)
    assert all(len(s) == 1 for s in flat.shapes), "Hybrid coordinates do not factor"

    # And the sampler actually produces rank-<=1 blocks on the factored objective.
    gen = torch.Generator().manual_seed(0)
    E = _lowrank_noise(obj.shapes, 1, 4, obj.dim, gen, torch.device("cpu"), torch.float32)
    off = 0
    for shape in obj.shapes:
        n = math.prod(shape)
        if len(shape) >= 2:
            assert torch.linalg.matrix_rank(E[0, off : off + n].reshape(shape)).item() <= 1
        off += n


def test_wallclock_deadline_stops_every_method():
    """A deadline stops a run through ``remaining``, without touching any method."""

    from polystep.baselines.core import BudgetExhausted, Objective

    obj = Objective(lambda X: torch.zeros(X.shape[0]), 4, budget=10**9, deadline_s=0.2)
    assert obj.elapsed_s == 0.0, "clock starts on the first call, not at construction"
    # remaining and the call check the clock separately, so the deadline can fall
    # between them. Either exit is correct; the test is that one of them happens.
    try:
        while obj.remaining > 0:
            obj(torch.zeros(2, 4))
    except BudgetExhausted:
        pass
    assert obj.out_of_time and obj.evals > 0
    with pytest.raises(BudgetExhausted, match="deadline"):
        obj(torch.zeros(2, 4))


def test_matched_budget_records_the_axis_it_used():
    """Each of the three axes returns its own name, and wall-clock needs a PolyStep run."""
    import json as _json
    import tempfile

    from experiments.runners.fairness import budget_for_method, matched_budget

    assert budget_for_method("openai_es", 10_000, 100, showcase="snn")[1] == "evals"
    assert budget_for_method("spsa", 10_000, 100, showcase="snn")[1] == "steps"
    assert budget_for_method("openai_es", 10_000, 100, showcase="snn", match="wallclock")[1] == "wallclock"

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            matched_budget(
                "openai_es",
                showcase="argmax",
                seed=42,
                polystep_steps=100,
                eval_budget=10_000,
                results_dir=d,
                match="wallclock",
            )
        with open(os.path.join(d, "argmax_polystep_42.json"), "w") as fh:
            _json.dump({"metrics": {"wall_time_seconds": 1234.5}}, fh)
        budget, axis, deadline = matched_budget(
            "openai_es",
            showcase="argmax",
            seed=42,
            polystep_steps=100,
            eval_budget=10_000,
            results_dir=d,
            match="wallclock",
        )
        assert axis == "wallclock" and deadline == 1234.5 and budget > 10**9
