"""Smoke tests for non-differentiable showcase elevation experiments.

Verifies that run_elevation.py can:
1. Import and expose 4 showcase configs and 5 method runners
2. Run each showcase in dry-run mode (1 epoch)
3. Produce valid JSON results with required metric keys

All tests use --dry-run mode (1 epoch/10 generations) for speed.
Uses a temp results dir to avoid polluting paper/results/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Test module imports
# ---------------------------------------------------------------------------


def test_showcase_configs_has_4_entries():
    """SHOWCASE_CONFIGS should have exactly 4 showcases."""
    from experiments.runners.run_elevation import SHOWCASE_CONFIGS
    assert len(SHOWCASE_CONFIGS) == 4
    assert set(SHOWCASE_CONFIGS.keys()) == {"snn", "int8", "argmax", "staircase"}


def test_method_runners_has_5_entries():
    """METHOD_RUNNERS should have exactly 5 methods."""
    from experiments.runners.run_elevation import METHOD_RUNNERS
    assert len(METHOD_RUNNERS) == 5
    assert set(METHOD_RUNNERS.keys()) == {"polystep", "adam", "cmaes", "openai_es", "spsa"}


# ---------------------------------------------------------------------------
# Dry-run smoke tests (one showcase at a time, adam method for speed)
# ---------------------------------------------------------------------------


def _run_dry(showcase, method="adam"):
    """Helper: run a single showcase/method/seed in dry-run mode, return result path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            "experiments/runners/run_elevation.py",
            "--showcases", showcase,
            "--methods", method,
            "--seeds", "42",
            "--device", "cpu",
            "--results-dir", tmpdir,
            "--dry-run",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"Dry run failed for {showcase}/{method}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        result_file = os.path.join(tmpdir, f"{showcase}_{method}_42.json")
        assert os.path.exists(result_file), f"Result file not found: {result_file}"

        with open(result_file) as f:
            data = json.load(f)
        return data


@pytest.mark.slow
def test_snn_dry_run():
    """SNN showcase dry-run with Adam should produce valid results."""
    data = _run_dry("snn")
    assert data["benchmark"] == "snn"
    assert data["method"] == "adam"
    assert data["seed"] == 42


def test_int8_dry_run():
    """Int8 showcase dry-run with Adam should produce valid results."""
    data = _run_dry("int8")
    assert data["benchmark"] == "int8"
    assert data["method"] == "adam"
    assert data["seed"] == 42


@pytest.mark.slow
def test_argmax_dry_run():
    """Argmax showcase dry-run with Adam should produce valid results."""
    data = _run_dry("argmax")
    assert data["benchmark"] == "argmax"
    assert data["method"] == "adam"
    assert data["seed"] == 42


@pytest.mark.slow
def test_staircase_dry_run():
    """Staircase showcase dry-run with Adam should produce valid results."""
    data = _run_dry("staircase")
    assert data["benchmark"] == "staircase"
    assert data["method"] == "adam"
    assert data["seed"] == 42


@pytest.mark.slow
def test_result_json_schema():
    """Dry-run result JSON should contain all required metric keys."""
    data = _run_dry("snn")
    required_keys = [
        "final_accuracy",
        "best_accuracy",
        "wall_time_seconds",
        "peak_gpu_memory_mb",
        "function_evals",
        "total_steps",
    ]
    for key in required_keys:
        assert key in data["metrics"], f"Missing required key: {key}"
        # All metric values should be numeric
        assert isinstance(data["metrics"][key], (int, float)), (
            f"Metric {key} should be numeric, got {type(data['metrics'][key])}"
        )
