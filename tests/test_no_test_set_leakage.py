"""The main runners must expose --allow-test-leakage so test-set selection is
opt-in. A source scan avoids importing the runners' heavy optional deps.
"""

from pathlib import Path

import pytest

RUNNERS = ["run_mnist.py", "run_moe.py", "run_elevation.py", "run_timeseries.py"]
RUNNER_DIR = Path(__file__).resolve().parent.parent / "experiments" / "runners"


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_exposes_allow_test_leakage(runner):
    path = RUNNER_DIR / runner
    if not path.exists():
        pytest.skip(f"{runner} not present")
    assert "--allow-test-leakage" in path.read_text()
