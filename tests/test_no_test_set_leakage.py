"""The main runners must expose --allow-test-leakage so test-set selection is
opt-in. A source scan avoids importing the runners' heavy optional deps.
"""

import ast
from pathlib import Path

import pytest

RUNNERS = ["run_mnist.py", "run_moe.py", "run_elevation.py", "run_timeseries.py"]
RUNNER_DIR = Path(__file__).resolve().parent.parent / "experiments" / "runners"


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_exposes_allow_test_leakage(runner, require_experiments):
    """The flag must be a live argparse option, not a string that appears in the file.

    A substring scan passes on a commented-out or docstring mention, which is exactly the
    state this guard exists to catch.
    """
    path = RUNNER_DIR / runner
    assert path.exists(), f"{runner} is missing; the leakage guard cannot be checked"

    source = path.read_text()
    tree = ast.parse(source)
    added = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--allow-test-leakage" in added, (
        f"{runner} does not register --allow-test-leakage with add_argument; "
        f"found options: {sorted(o for o in added if o.startswith('--'))}"
    )
