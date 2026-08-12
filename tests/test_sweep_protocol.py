"""Guards for the sweep protocol: equal round-two budgets, keyword-built schedules,
config keys the runners read, and no test-split reads during tuning.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# experiments/ is the paper reproduction harness and is not part of the distributed
# package, so this module is unimportable when the tests run from an sdist.
if not (REPO_ROOT / "experiments" / "runners").is_dir():
    pytest.skip("experiments/runners not present (running outside the repo)", allow_module_level=True)

RUNNER_DIR = REPO_ROOT / "experiments" / "runners"
SCRIPT_DIR = REPO_ROOT / "experiments" / "scripts"
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(scope="module")
def fairness():
    from experiments.runners import fairness as f

    return f


# --------------------------------------------------------------------------------
# refine_grid: equal round-two budgets, and integer axes that the method accepts
# --------------------------------------------------------------------------------


def test_round_two_gives_every_method_the_same_cell_count(fairness):
    """The paper quotes "two rounds of nine cells per method". It has to be true."""
    for method, grid in fairness.TUNING_GRID.items():
        for winner in grid:
            got = fairness.refine_grid(method, winner)
            assert len(got) == len(grid), (
                f"{method}: round two produced {len(got)} cells against round one's {len(grid)} for winner {winner}"
            )


def test_round_two_integer_axes_stay_even(fairness):
    """eggroll raises on an odd popsize, and the sweep swallows the exception, so
    round-two cells must keep popsize even."""
    from polystep.baselines.methods import eggroll  # noqa: F401  (import proves it exists)

    for winner in fairness.TUNING_GRID["eggroll"]:
        for point in fairness.refine_grid("eggroll", winner):
            assert point["popsize"] % 2 == 0 and point["popsize"] >= 2, (
                f"eggroll round two produced popsize={point['popsize']} from {winner}, "
                "which methods.eggroll rejects outright"
            )


def test_tuning_cost_quotes_the_cells_that_ran(fairness):
    """A cell that raises is dropped by the sweep, so the count must be passed in."""
    nominal = fairness.tuning_cost("eggroll", 1000, rounds=2)
    realized = fairness.tuning_cost("eggroll", 1000, rounds=2, configs=14)
    assert nominal["configs"] == len(fairness.TUNING_GRID["eggroll"]) * 2
    assert realized["configs"] == 14
    assert realized["tuning_evals"] == 14 * 1000


# --------------------------------------------------------------------------------
# Schedules: CosineEpsilon declares target before init
# --------------------------------------------------------------------------------


def _cosine_calls(path: Path):
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "CosineEpsilon"
    ]


@pytest.mark.parametrize(
    "path",
    sorted(p for p in list(RUNNER_DIR.glob("*.py")) + list(SCRIPT_DIR.glob("*.py")) + list(EXAMPLES_DIR.glob("*.py"))),
    ids=lambda p: p.name,
)
def test_cosine_schedules_are_built_by_keyword(path):
    """Positional CosineEpsilon(init, target) runs the schedule backwards: the
    dataclass declares ``target`` first, so the positional form binds init to target."""
    for call in _cosine_calls(path):
        assert not call.args, (
            f"{path.name}:{call.lineno} builds CosineEpsilon positionally; "
            "the first field is `target`, so this runs the schedule backwards"
        )


def test_cosine_epsilon_first_field_is_target():
    """If this ever changes, the keyword rule above stops being the reason."""
    import dataclasses

    from polystep.epsilon import CosineEpsilon

    assert [f.name for f in dataclasses.fields(CosineEpsilon)][:2] == ["target", "init"]


# --------------------------------------------------------------------------------
# Configs a runner indexes must exist in the config it indexes
# --------------------------------------------------------------------------------


def test_maxsat_config_has_every_key_its_builder_reads():
    """get_polystep_config reads probe_radius_jitter, so POLYSTEP_CONFIG must define it."""
    from experiments.runners.run_maxsat import get_polystep_config

    cfg = get_polystep_config(100_000)
    assert "probe_radius_jitter" in cfg
    # Stated, not defaulted: this value decides whether condition (iv) of the
    # convergence theorem holds for this configuration.
    assert isinstance(cfg["probe_radius_jitter"], float)


# --------------------------------------------------------------------------------
# A sweep must not read its own picks, and must not reach test
# --------------------------------------------------------------------------------


def _calls_run_polystep_with_tuning(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "run_polystep":
            continue
        for kw in node.keywords:
            if kw.arg == "tuning" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                return True
        return False
    return False


def test_polystep_sweep_declares_itself_a_sweep():
    """Without ``tuning=True`` the runner applies its own selection file's multipliers
    a second time, so the recorded point stops describing the run that scored."""
    path = SCRIPT_DIR / "tune_polystep.py"
    assert _calls_run_polystep_with_tuning(path), "tune_polystep.py must call run_polystep(..., tuning=True)"


def test_tuning_mode_swaps_in_the_test_tripwire():
    """The baseline sweep is handed test_fn=nan; our own sweep must be as constrained."""
    src = (RUNNER_DIR / "run_elevation.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_polystep")
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "tuning" in args, "run_polystep must accept tuning="
    body = ast.get_source_segment(src, fn)
    assert "TestSplitTripwire()" in body, "tuning=True must swap the test loader"


def test_polystep_runner_reads_test_once():
    """Test is scored on the validation-selected checkpoint and nowhere else."""
    src = (RUNNER_DIR / "run_elevation.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_polystep")
    reads = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "evaluate_accuracy"
        and any(getattr(a, "id", None) == "test_loader" for a in n.args)
    ]
    assert len(reads) == 1, (
        f"run_polystep reads the test split {len(reads)} times; the protocol is once, "
        "on the validation-selected checkpoint"
    )


# --------------------------------------------------------------------------------
# Baselines are scored where the method says it is, not at a displaced candidate
# --------------------------------------------------------------------------------


def test_every_baseline_publishes_its_iterate():
    """obj.best_x is a sampled candidate; obj.iterate is the method's own estimate.

    For a population method the two differ by sigma*eps, whose norm is about one
    probe radius.
    """
    import torch

    from polystep.baselines.core import Objective
    from polystep.baselines.methods import METHODS

    for name, method in METHODS.items():
        if name == "cma_es":
            pytest.importorskip("cma")

        def loss(X, _n=name):
            return (X**2).sum(dim=-1)

        obj = Objective(loss, dim=4, budget=400)
        assert obj.iterate is None, "iterate starts unset"
        method(obj, x0=torch.zeros(4))
        assert obj.iterate is not None, f"{name} never published obj.iterate"
        assert obj.iterate.shape == (4,), f"{name} published a malformed iterate"


def test_run_baseline_records_which_point_validation_chose(fairness):
    """The choice between iterate and best_x has to be auditable, not conventional."""
    src = (RUNNER_DIR / "fairness.py").read_text()
    assert '"scored_at": label' in src, "each trajectory point must record its source"
