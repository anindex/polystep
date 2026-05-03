"""Smoke tests for MAX-SAT experiment runner (experiments/runners/run_maxsat.py).

Tests use 20-variable instances (fast, completes in < 1 second each) to verify
all methods produce valid results. Uses tmp_path to avoid polluting paper/results/.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from experiments.runners.nondiff_data import generate_maxsat_instance
from experiments.runners.nondiff_models import MaxSATModel
from experiments.runners.run_maxsat import (
    CRA_LAMBDA,
    evaluate_sat_result,
    make_sat_closure,
    run_cmaes,
    run_openai_es,
    run_rc2,
    run_sls,
    run_polystep,
)


# Shared fixture: 20-var MAX-SAT instance for fast tests
@pytest.fixture(scope="module")
def instance():
    return generate_maxsat_instance(num_vars=20, seed=42)


# --- Closure and evaluation tests ---


@pytest.mark.timeout(10)
def test_make_sat_closure_shapes(instance):
    """make_sat_closure returns (N,) tensor for N=5 configs on 20-var instance."""
    closure = make_sat_closure(
        instance["clause_vars"],
        instance["clause_signs"],
        cra_lambda=0.005,
        cra_alpha=2,
    )
    # Simulate N=5 parameter configurations
    stacked_params = {"assignments": torch.randn(5, 20)}
    result = closure(stacked_params)
    assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"
    assert result.dtype == torch.float32 or result.dtype == torch.float64
    assert (result >= 0).all(), "All costs should be non-negative"


@pytest.mark.timeout(10)
def test_evaluate_sat_result_keys(instance):
    """evaluate_sat_result returns dict with sat_ratio, num_satisfied, num_clauses."""
    model = MaxSATModel(20)
    result = evaluate_sat_result(model, instance["clause_vars"], instance["clause_signs"])
    assert isinstance(result, dict)
    assert "sat_ratio" in result
    assert "num_satisfied" in result
    assert "num_clauses" in result
    assert 0.0 <= result["sat_ratio"] <= 1.0
    assert result["num_clauses"] == instance["num_clauses"]
    assert 0 <= result["num_satisfied"] <= result["num_clauses"]


# --- Method smoke tests ---


@pytest.mark.timeout(30)
def test_polystep_smoke(instance, tmp_path):
    """run_polystep on 20 vars for 5 steps produces JSON with sat_ratio > 0.0."""
    evals = run_polystep(
        num_vars=20,
        instance=instance,
        seed=42,
        device="cpu",
        steps=5,
        results_dir=str(tmp_path),
    )
    assert isinstance(evals, int) and evals > 0

    # Check result file
    result_file = os.path.join(str(tmp_path), "maxsat_20v_polystep_42.json")
    assert os.path.exists(result_file), f"Result file not found: {result_file}"
    with open(result_file) as f:
        data = json.load(f)
    assert data["metrics"]["final_accuracy"] > 0.0


@pytest.mark.timeout(30)
def test_cmaes_smoke(instance, tmp_path):
    """run_cmaes on 20 vars for 100 evals produces JSON with sat_ratio > 0.5."""
    run_cmaes(
        num_vars=20,
        instance=instance,
        seed=42,
        device="cpu",
        max_evals=100,
        results_dir=str(tmp_path),
    )
    result_file = os.path.join(str(tmp_path), "maxsat_20v_cmaes_42.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)
    assert data["metrics"]["final_accuracy"] > 0.5


@pytest.mark.timeout(30)
def test_openai_es_smoke(instance, tmp_path):
    """run_openai_es on 20 vars for 100 evals produces JSON with sat_ratio > 0.5."""
    run_openai_es(
        num_vars=20,
        instance=instance,
        seed=42,
        device="cpu",
        max_evals=100,
        results_dir=str(tmp_path),
    )
    result_file = os.path.join(str(tmp_path), "maxsat_20v_openai_es_42.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)
    assert data["metrics"]["final_accuracy"] > 0.5


@pytest.mark.timeout(30)
def test_rc2_smoke(instance, tmp_path):
    """run_rc2 on 20 vars returns sat_ratio >= 0.99."""
    run_rc2(
        num_vars=20,
        instance=instance,
        results_dir=str(tmp_path),
        timeout=10,
    )
    result_file = os.path.join(str(tmp_path), "maxsat_20v_rc2_0.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)
    assert data["metrics"]["final_accuracy"] >= 0.99


@pytest.mark.timeout(30)
def test_sls_smoke(instance, tmp_path):
    """run_sls on 20 vars returns sat_ratio > 0.8."""
    run_sls(
        num_vars=20,
        instance=instance,
        results_dir=str(tmp_path),
        max_flips=1000,
    )
    result_file = os.path.join(str(tmp_path), "maxsat_20v_sls_0.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)
    assert data["metrics"]["final_accuracy"] > 0.8


# --- Result format tests ---


@pytest.mark.timeout(10)
def test_result_json_schema(instance, tmp_path):
    """Saved JSON has required keys (benchmark, method, seed, metrics with all 6 required fields)."""
    # Run a quick method to get a JSON result
    run_sls(
        num_vars=20,
        instance=instance,
        results_dir=str(tmp_path),
        max_flips=100,
    )
    result_file = os.path.join(str(tmp_path), "maxsat_20v_sls_0.json")
    with open(result_file) as f:
        data = json.load(f)

    # Top-level keys
    for key in ["benchmark", "method", "seed", "timestamp", "environment",
                "hyperparameters", "metrics"]:
        assert key in data, f"Missing top-level key: {key}"

    # Required metrics keys
    for key in ["final_accuracy", "best_accuracy", "wall_time_seconds",
                "peak_gpu_memory_mb", "function_evals", "total_steps"]:
        assert key in data["metrics"], f"Missing metrics key: {key}"


@pytest.mark.timeout(10)
def test_cra_lambda_disabled():
    """CRA penalty is disabled by default (ablation shows no effect on polystep)."""
    assert CRA_LAMBDA == 0.0


# --- Large-scale smoke tests ---


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_polystep_1m_smoke(tmp_path):
    """run_polystep at 1M vars for 2 steps validates no crash and chunk_size=128 path.

    This test generates a small instance (1M vars, 4.27M clauses) and runs
    only 2 optimizer steps to verify the dynamic chunk_size logic activates
    and the forward pass completes without OOM. Marked @slow because instance
    generation alone takes ~10s on CPU.
    """
    instance = generate_maxsat_instance(num_vars=1000000, seed=42)
    assert instance["num_clauses"] == 4270000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evals = run_polystep(
        num_vars=1000000,
        instance=instance,
        seed=42,
        device=device,
        steps=2,
        results_dir=str(tmp_path),
    )
    assert isinstance(evals, int) and evals > 0

    result_file = os.path.join(str(tmp_path), "maxsat_1000000v_polystep_42.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)
    assert data["metrics"]["final_accuracy"] > 0.0
    assert data["metrics"]["total_steps"] == 2
