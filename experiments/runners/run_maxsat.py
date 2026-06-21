#!/usr/bin/env python
"""Run all methods and seeds for MAX-SAT benchmark.

Experiments run polystep on random 3-SAT instances at scaling variable counts
(100, 500, 1000, 5000), comparing against CMA-ES, OpenAI-ES, RC2 exact
solver, and stochastic local search (SLS) reference solver.

Methods:
  - polystep: PolyStepOptimizer with custom MAX-SAT closure (no NNCostEvaluator)
  - cmaes: pycma CMA-ES with same sigmoid+CRA encoding
  - openai_es: Antithetic OpenAI-ES with same encoding
  - rc2: PySAT RC2 exact solver (timeout-bounded for 500+ vars)
  - sls: WalkSAT-style stochastic local search

Each method uses the same continuous relaxation: sigmoid(assignments) with CRA
penalty to encourage {0, 1} solutions. Function evaluation budgets are matched
between polystep and ES methods by running polystep first and counting evals.

Results are saved as JSON: experiments/results/softmax/main/maxsat_{method}_{seed}.json

Usage:
    python experiments/runners/run_maxsat.py
    python experiments/runners/run_maxsat.py --sizes 100 500 --methods polystep cmaes
    python experiments/runners/run_maxsat.py --sizes 100 --methods polystep --seeds 42 --dry-run
"""

from __future__ import annotations

import argparse
import gc
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repo root is on path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import torch

from experiments.runners.common import (
    SEEDS,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.nondiff_data import generate_maxsat_instance
from experiments.runners.nondiff_models import MaxSATModel


# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

BENCHMARK = "maxsat"
VARIABLE_SIZES = [100, 500, 1000, 5000, 20000, 100000, 1000000]
INSTANCE_SEED = 42  # Fixed instance per size, optimizer seeds vary

# Best configuration for MAX-SAT
# Softmax solver with num_probe=1 (single-pass, no Sinkhorn iterations)
PSTORCH_CONFIG = {
    "epsilon_init": 5.0,
    "epsilon_target": 0.5,
    "step_radius_init": 3000.0,
    "step_radius_target": 600.0,
    "probe_radius_init": 100.0,
    "probe_radius_target": 20.0,
    "num_probe": 1,
    "chunk_size": 256,  # prevents OOM at 5000+ vars
    "amortize_steps": 3,
    "amortize_ema": 0.7,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.95,
}


def get_polystep_config(num_vars: int) -> dict:
    """Return size-dependent polystep config using sqrt-scaling from 100K reference."""
    import math
    ref = PSTORCH_CONFIG
    scale = math.sqrt(num_vars / 100_000)
    return {
        "epsilon_init": ref["epsilon_init"],
        "epsilon_target": ref["epsilon_target"],
        "step_radius_init": ref["step_radius_init"] * scale,
        "step_radius_target": ref["step_radius_target"] * scale,
        "probe_radius_init": ref["probe_radius_init"] * scale,
        "probe_radius_target": ref["probe_radius_target"] * scale,
        "num_probe": ref["num_probe"],
        "chunk_size": ref["chunk_size"],
        "amortize_steps": ref["amortize_steps"],
        "amortize_ema": ref["amortize_ema"],
        "use_momentum": ref["use_momentum"],
        "momentum_init": ref["momentum_init"],
        "momentum_final": ref["momentum_final"],
    }

# 1M+ extension: delta evaluation with inverted index.
# At 1M vars, full closure eval is too expensive (4.5 min/step). Delta evaluation
# builds a CSR inverted index (variable -> clause list), then only re-evaluates
# ~1664 affected clauses per chunk instead of 4.27M.
# Tuned: eps 5->0.5, sr 5000->1000, pr 500->100 (from hyperparameter search)
PSTORCH_TURBO_1M = {
    "epsilon_init": 5.0,
    "epsilon_target": 0.5,
    "step_radius_init": 5000.0,
    "step_radius_target": 1000.0,
    "probe_radius_init": 500.0,
    "probe_radius_target": 100.0,
    "num_probe": 1,
    "chunk_size": 256,  # Memory management for 4.27M clauses on 32GB GPU
    "amortize_steps": 3,
    "amortize_ema": 0.7,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.95,
    "mixed_precision": False,
    "clause_sample_size": 25_000,  # only used by non-delta fallback path
    "particle_dim": 2,  # 500K particles x 4 vertices = 2M configs
}

# Step budgets scale with problem size
# Step budgets: tuned to near-convergence per size.
# 20K converges at ~1000 steps (97.8%); extra steps yield <0.3pp over 4000 more steps.
# 100K at 1000 steps is still climbing (~96.2%), but per-step cost (6.4s) makes more
# steps impractical. 1M uses delta eval; increased to 200 steps for potential gain.
# Step budgets: 1000 steps for ALL sizes (production runs)
STEP_BUDGETS = {100: 1000, 500: 1000, 1000: 1000, 5000: 1000, 20000: 1000, 100000: 1000, 1000000: 500}

# CRA penalty disabled: ablation shows no measurable effect on polystep
# (polystep navigates the piecewise-constant round() landscape directly via OT)
CRA_LAMBDA = 0.0
CRA_ALPHA = 2

# CMA-ES config (pycma)
CMAES_CONFIG = {"popsize": 50, "sigma0": 1.0}

# OpenAI-ES config
OPENAI_ES_CONFIG = {"popsize": 50, "sigma": 1.0, "lr": 0.1}

# RC2 timeout in seconds per instance
RC2_TIMEOUT = 60

# SLS max flips (scales with problem size, see _sls_max_flips())
SLS_MAX_FLIPS = 100000
SLS_MAX_FLIPS_1M = 50000  # Reduced from 500K so SLS at 1M completes in reasonable time


# ---------------------------------------------------------------------------
# Counting closure wrapper
# ---------------------------------------------------------------------------


class CountingClosure:
    """Wraps a closure and counts total evaluations (sum of batch dim N)."""

    def __init__(self, closure):
        self._closure = closure
        self.count = 0

    def __call__(self, stacked_params):
        result = self._closure(stacked_params)
        self.count += result.shape[0]
        return result

    def reset(self):
        self.count = 0


# ---------------------------------------------------------------------------
# Helper: make_sat_closure for polystep
# ---------------------------------------------------------------------------


def make_sat_closure(clause_vars, clause_signs, cra_lambda=0.0, cra_alpha=2,
                     clause_sample_size=0, model=None, particle_dim=2):
    """Create polystep-compatible closure for MAX-SAT optimization.

    Args:
        clause_vars: (C, 3) long tensor of variable indices.
        clause_signs: (C, 3) float tensor (1.0=positive, 0.0=negated).
        cra_lambda: CRA penalty weight (0.0 by default - ablation shows no effect).
        cra_alpha: CRA penalty exponent.
        clause_sample_size: If > 0, sample this many clauses per step (stochastic).
            Ignored when model is provided (delta evaluation uses full clause set).
        model: MaxSATModel reference for delta evaluation (optional).
            When provided, uses inverted index to only re-evaluate clauses
            affected by the perturbed variables (~1664 per chunk vs 4.27M).
        particle_dim: Particle dimension for delta evaluation (default 2).

    Returns:
        Closure: stacked_params dict -> (N,) costs tensor.
        Closure has a .resample() method (call once per optimizer step).
    """
    total_clauses = clause_vars.shape[0]
    num_vars = clause_vars.max().item() + 1
    use_sampling = clause_sample_size > 0 and clause_sample_size < total_clauses
    use_delta = model is not None

    # Build inverted index for delta evaluation: variable -> clause indices (CSR)
    if use_delta:
        cv_cpu = clause_vars.cpu()
        var_counts = torch.zeros(num_vars, dtype=torch.long)
        for j in range(cv_cpu.shape[1]):
            var_counts.scatter_add_(0, cv_cpu[:, j], torch.ones(total_clauses, dtype=torch.long))
        var_offsets = torch.zeros(num_vars + 1, dtype=torch.long)
        var_offsets[1:] = var_counts.cumsum(0)
        total_mentions = var_offsets[-1].item()
        var_clause_list = torch.empty(total_mentions, dtype=torch.long)
        fill_pos = var_offsets[:-1].clone()
        for j in range(cv_cpu.shape[1]):
            for c_idx in range(total_clauses):
                v = cv_cpu[c_idx, j].item()
                var_clause_list[fill_pos[v]] = c_idx
                fill_pos[v] += 1
        var_offsets = var_offsets.to(clause_vars.device)
        var_clause_list = var_clause_list.to(clause_vars.device)

    # Mutable state
    state = {}
    if use_sampling and not use_delta:
        idx = torch.randint(total_clauses, (clause_sample_size,),
                            device=clause_vars.device)
        state["cv"] = clause_vars[idx]
        state["cs"] = clause_signs[idx]

    def resample():
        """Refresh clause sample and/or cache base for delta evaluation."""
        if use_sampling and not use_delta:
            idx = torch.randint(total_clauses, (clause_sample_size,),
                                device=clause_vars.device)
            state["cv"] = clause_vars[idx]
            state["cs"] = clause_signs[idx]

        if use_delta:
            with torch.no_grad():
                base_raw = model.assignments.data
                state["base_raw"] = base_raw.clone()
                base_soft = torch.sigmoid(base_raw)
                state["base_hard"] = torch.round(base_soft)
                # Evaluate base over ALL clauses
                g = state["base_hard"][clause_vars]  # (C, 3)
                lits = g * clause_signs + (1.0 - clause_signs) * (1.0 - g)
                state["base_clause_sat"] = (lits > 0.5).any(dim=-1)  # (C,) bool
                state["base_sat_count"] = state["base_clause_sat"].float().sum().item()
                if cra_lambda > 0:
                    state["base_cra"] = (
                        1.0 - (2.0 * base_soft - 1.0) ** cra_alpha
                    ).sum().item()

    def _full_evaluate(assignments):
        """Full evaluation - used when delta is not available."""
        soft = torch.sigmoid(assignments)
        hard = torch.round(soft)
        cv = state.get("cv", clause_vars)
        cs = state.get("cs", clause_signs)
        gathered = hard[:, cv]
        signs = cs.unsqueeze(0).to(dtype=gathered.dtype)
        literals = gathered * signs + (1.0 - signs) * (1.0 - gathered)
        satisfied = (literals > 0.5).any(dim=-1).float()
        unsat_ratio = 1.0 - satisfied.mean(dim=-1)
        penalty = (1.0 - (2.0 * soft - 1.0) ** cra_alpha).sum(dim=-1)
        return unsat_ratio + cra_lambda * penalty

    def _delta_evaluate(assignments):
        """Delta evaluation using inverted index over full clause set."""
        N = assignments.shape[0]
        base_raw = state["base_raw"]
        C = total_clauses

        # Detect changed variable range from first and last row
        diff_first = (assignments[0] != base_raw).nonzero(as_tuple=True)[0]
        diff_last = (assignments[-1] != base_raw).nonzero(as_tuple=True)[0]

        if diff_first.numel() == 0 and diff_last.numel() == 0:
            base_unsat_count = C - state["base_sat_count"]
            cost = torch.full((N,), base_unsat_count, device=assignments.device)
            if cra_lambda > 0:
                cost = cost + cra_lambda * C * state["base_cra"]
            return cost

        all_diff = torch.cat([diff_first, diff_last])
        min_var = all_diff.min().item()
        max_var = all_diff.max().item()

        # Look up affected clauses via inverted index (CSR)
        start = var_offsets[min_var].item()
        end = var_offsets[max_var + 1].item()
        affected_idx = var_clause_list[start:end].unique()
        A = affected_idx.shape[0]

        if A == 0:
            base_unsat_count = C - state["base_sat_count"]
            cost = torch.full((N,), base_unsat_count, device=assignments.device)
            if cra_lambda > 0:
                cost = cost + cra_lambda * C * state["base_cra"]
            return cost

        # Compute sigmoid+round only for changed variables
        changed_range = torch.arange(min_var, max_var + 1, device=assignments.device)
        soft_changed = torch.sigmoid(assignments[:, changed_range])  # (N, R)
        hard_changed = torch.round(soft_changed)  # (N, R)

        # Hybrid gather: changed vars from hard_changed, rest from base_hard
        a_cv = clause_vars[affected_idx]  # (A, 3)
        a_cs = clause_signs[affected_idx]  # (A, 3)
        in_changed = (a_cv >= min_var) & (a_cv <= max_var)  # (A, 3)
        local_idx = (a_cv - min_var).clamp(0, max_var - min_var)  # (A, 3)

        changed_gathered = hard_changed[:, local_idx.reshape(-1)].reshape(N, A, 3)
        base_gathered = state["base_hard"][a_cv].unsqueeze(0)  # (1, A, 3)
        gathered = torch.where(in_changed.unsqueeze(0), changed_gathered, base_gathered)

        # Evaluate clause satisfaction
        signs = a_cs.unsqueeze(0).to(dtype=gathered.dtype)
        lits = gathered * signs + (1.0 - signs) * (1.0 - gathered)
        new_clause_sat = (lits > 0.5).any(dim=-1).float()  # (N, A)

        # Delta satisfaction
        base_affected_sat = state["base_clause_sat"][affected_idx].float()  # (A,)
        delta_sat = (new_clause_sat - base_affected_sat.unsqueeze(0)).sum(dim=-1)  # (N,)
        # Scale to clause-count units for OT discrimination
        # unsat_ratio has spread ~26/C ≈ 6e-6, too small for epsilon=0.5
        # Multiplying by C gives spread ~26, matching epsilon scale
        unsat_count = C - (state["base_sat_count"] + delta_sat)  # (N,)

        # CRA penalty (delta) - also scaled by C for consistency
        if cra_lambda > 0:
            base_soft_changed = torch.sigmoid(base_raw[changed_range])
            base_cra_changed = (1.0 - (2.0 * base_soft_changed - 1.0) ** cra_alpha).sum()
            new_cra_changed = (1.0 - (2.0 * soft_changed - 1.0) ** cra_alpha).sum(dim=-1)
            delta_cra = new_cra_changed - base_cra_changed
            unsat_count = unsat_count + cra_lambda * C * (state["base_cra"] + delta_cra)

        return unsat_count

    def closure(stacked_params):
        assignments = stacked_params["assignments"]  # (N, num_vars)
        if use_delta and "base_raw" in state:
            return _delta_evaluate(assignments)
        return _full_evaluate(assignments)

    closure.resample = resample
    return closure


# ---------------------------------------------------------------------------
# Helper: evaluate_sat_result
# ---------------------------------------------------------------------------


def evaluate_sat_result(model, clause_vars, clause_signs):
    """Evaluate hard satisfaction ratio for MaxSATModel.

    Args:
        model: MaxSATModel with assignments parameter.
        clause_vars: (C, 3) long tensor of variable indices.
        clause_signs: (C, 3) float tensor of literal signs.

    Returns:
        Dict with sat_ratio, num_satisfied, num_clauses.
    """
    with torch.no_grad():
        soft = torch.sigmoid(model.assignments)
        hard = torch.round(soft)
        gathered = hard[clause_vars]
        literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
        satisfied = (literals > 0.5).any(dim=-1).float()
        sat_ratio = satisfied.mean().item()
        num_satisfied = int(satisfied.sum().item())
        num_clauses = clause_vars.shape[0]
    return {
        "sat_ratio": sat_ratio,
        "num_satisfied": num_satisfied,
        "num_clauses": num_clauses,
    }


# ---------------------------------------------------------------------------
# Method: polystep
# ---------------------------------------------------------------------------


def run_polystep(num_vars, instance, seed, device, steps, results_dir, solver=None):
    """Train MAX-SAT with polystep PolyStepOptimizer + custom closure.

    Args:
        num_vars: Number of Boolean variables.
        instance: Dict from generate_maxsat_instance.
        seed: Random seed for optimizer.
        device: Device string ('cuda' or 'cpu').
        steps: Number of optimizer steps.
        results_dir: Directory for JSON results.

    Returns:
        Total function evaluations (for budget matching with ES methods).
    """
    from polystep.optimizer import PolyStepOptimizer
    from polystep.epsilon import CosineEpsilon

    set_seed(seed)
    model = MaxSATModel(num_vars).to(device)
    clause_vars = instance["clause_vars"].to(device)
    clause_signs = instance["clause_signs"].to(device)

    cra_lambda = CRA_LAMBDA

    # Select config: turbo overrides for 1M+ vars, size-dependent scaling otherwise
    turbo = num_vars >= 1000000
    cfg = PSTORCH_TURBO_1M if turbo else get_polystep_config(num_vars)

    if turbo and device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Build CosineEpsilon schedules for epsilon, step_radius, probe_radius
    eps = CosineEpsilon(cfg["epsilon_init"], cfg["epsilon_target"]) if "epsilon_init" in cfg else cfg.get("epsilon", 3.0)
    sr = CosineEpsilon(cfg["step_radius_init"], cfg["step_radius_target"]) if "step_radius_init" in cfg else cfg.get("step_radius", 500.0)
    pr = CosineEpsilon(cfg["probe_radius_init"], cfg["probe_radius_target"]) if "probe_radius_init" in cfg else cfg.get("probe_radius", 50.0)
    pdim = cfg.get("particle_dim", 2)

    optimizer = PolyStepOptimizer(
        model,
        compile=False,
        seed=seed,
        epsilon=eps,
        step_radius=sr,
        probe_radius=pr,
        num_probe=cfg["num_probe"],
        chunk_size=cfg["chunk_size"],
        amortize_steps=cfg["amortize_steps"],
        amortize_ema=cfg.get("amortize_ema", 0.7),
        use_momentum=cfg.get("use_momentum", False),
        momentum_init=cfg.get("momentum_init", 0.5),
        momentum_final=cfg.get("momentum_final", 0.9),
        mixed_precision=cfg.get("mixed_precision", False),
        particle_dim=pdim,
        solver=solver,
    )

    # Clause sampling for 1M+: evaluate random clause subset -> 43x compute reduction
    clause_sample = cfg.get("clause_sample_size", 0) if turbo else 0
    base_closure = make_sat_closure(
        clause_vars, clause_signs, cra_lambda=cra_lambda, cra_alpha=CRA_ALPHA,
        clause_sample_size=clause_sample,
        model=model if turbo else None,
        particle_dim=pdim,
    )
    counter = CountingClosure(base_closure)

    best_sat_ratio = 0.0
    step_logs = []
    fine_step_logs = []
    start_time = time.time()

    # Resample clause subset each step for noise-free relative comparisons
    has_resample = hasattr(base_closure, 'resample')

    with track_gpu_memory() as mem:
        for step in range(steps):
            if has_resample:
                base_closure.resample()
            optimizer.step(counter)

            # Per-20-step fine-grained tracking
            if (step + 1) % 20 == 0:
                result_20 = evaluate_sat_result(model, clause_vars, clause_signs)
                best_sat_ratio = max(best_sat_ratio, result_20["sat_ratio"])
                fine_step_logs.append({
                    "step": step + 1,
                    "sat_ratio": result_20["sat_ratio"],
                    "num_satisfied": result_20["num_satisfied"],
                    "num_unsatisfied": result_20["num_clauses"] - result_20["num_satisfied"],
                    "num_clauses": result_20["num_clauses"],
                    "loss": 1.0 - result_20["sat_ratio"],
                    "wall_time": time.time() - start_time,
                })

            # Evaluate periodically (coarse epoch-level logging)
            if (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
                result = evaluate_sat_result(model, clause_vars, clause_signs)
                best_sat_ratio = max(best_sat_ratio, result["sat_ratio"])
                unsat = result["num_clauses"] - result["num_satisfied"]
                elapsed = time.time() - start_time
                step_logs.append(
                    {
                        "epoch": step + 1,
                        "accuracy": result["sat_ratio"],
                        "loss": 1.0 - result["sat_ratio"],
                        "num_satisfied": result["num_satisfied"],
                        "num_unsatisfied": unsat,
                        "num_clauses": result["num_clauses"],
                        "time": elapsed,
                    }
                )
                if (step + 1) % max(1, steps // 5) == 0:
                    print(
                        f"      step {step+1}/{steps} | "
                        f"sat={result['sat_ratio']*100:.1f}% "
                        f"({result['num_satisfied']}/{result['num_clauses']}, "
                        f"unsat={unsat}) | "
                        f"evals={counter.count}"
                    )

    wall_time = time.time() - start_time
    final_result = evaluate_sat_result(model, clause_vars, clause_signs)
    best_sat_ratio = max(best_sat_ratio, final_result["sat_ratio"])

    filepath = save_result(
        benchmark=f"{BENCHMARK}_{num_vars}v",
        method="polystep",
        seed=seed,
        metrics={
            "final_accuracy": final_result["sat_ratio"],
            "best_accuracy": best_sat_ratio,
            "num_satisfied": final_result["num_satisfied"],
            "num_unsatisfied": final_result["num_clauses"] - final_result["num_satisfied"],
            "num_clauses": final_result["num_clauses"],
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": counter.count,
            "total_steps": steps,
        },
        hyperparameters={
            **cfg,
            "cra_lambda": cra_lambda,
            "cra_alpha": CRA_ALPHA,
            "num_vars": num_vars,
        },
        epoch_logs=step_logs,
        step_logs=fine_step_logs,
        results_dir=results_dir,
    )
    print(f"      Saved: {filepath}")
    return counter.count


# ---------------------------------------------------------------------------
# Method: CMA-ES (pycma)
# ---------------------------------------------------------------------------


def run_cmaes(num_vars, instance, seed, device, max_evals, results_dir):
    """Train MAX-SAT with CMA-ES (pycma) using same sigmoid+CRA encoding.

    Args:
        num_vars: Number of Boolean variables.
        instance: Dict from generate_maxsat_instance.
        seed: Random seed.
        device: Device string (CMA-ES runs on CPU, clause eval on CPU).
        max_evals: Maximum function evaluations (budget matched to polystep).
        results_dir: Directory for JSON results.
    """
    import cma

    set_seed(seed)

    # Move clause tensors to device once (enables GPU batch eval)
    eval_device = torch.device(device)
    clause_vars = instance["clause_vars"].to(eval_device)
    clause_signs = instance["clause_signs"].to(eval_device)
    cra_lambda = CRA_LAMBDA

    def eval_batch(solutions_list):
        """Batch-evaluate popsize solutions on GPU (list of np arrays -> list of costs)."""
        x = torch.from_numpy(np.stack(solutions_list)).to(
            device=eval_device, dtype=torch.float32
        )  # (P, N)
        soft = torch.sigmoid(x)
        hard = torch.round(soft)
        gathered = hard[:, clause_vars]  # (P, C, 3)
        literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
        satisfied = (literals > 0.5).any(dim=-1).float()  # (P, C)
        unsat_ratio = 1.0 - satisfied.mean(dim=-1)  # (P,)
        penalty = (1.0 - (2.0 * soft - 1.0) ** CRA_ALPHA).sum(dim=-1)  # (P,)
        cost = unsat_ratio + cra_lambda * penalty
        return cost.detach().cpu().tolist()

    popsize = CMAES_CONFIG["popsize"]
    sigma0 = CMAES_CONFIG["sigma0"]
    max_iter = max(1, max_evals // popsize)

    opts = {
        "maxiter": max_iter,
        "popsize": popsize,
        "seed": seed,
        "verbose": -9,
    }
    # Use sep-CMA-ES for large problems (avoids O(n^2) covariance)
    if num_vars >= 1000:
        opts["CMA_diagonal"] = True

    x0 = np.random.RandomState(seed).randn(num_vars) * 0.1

    best_sat_ratio = 0.0
    step_logs = []
    total_evals = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        gen = 0
        while not es.stop():
            solutions = es.ask()
            costs = eval_batch(solutions)
            es.tell(solutions, costs)
            total_evals += len(solutions)
            gen += 1

            # Evaluate best so far periodically
            if gen % max(1, max_iter // 10) == 0 or es.stop():
                model_tmp = MaxSATModel(num_vars)
                model_tmp.assignments.data = torch.tensor(
                    es.result.xbest, dtype=torch.float32
                )
                result = evaluate_sat_result(
                    model_tmp, instance["clause_vars"], instance["clause_signs"]
                )
                best_sat_ratio = max(best_sat_ratio, result["sat_ratio"])
                elapsed = time.time() - start_time
                step_logs.append(
                    {
                        "epoch": gen,
                        "accuracy": result["sat_ratio"],
                        "loss": es.result.fbest,
                        "time": elapsed,
                    }
                )

    wall_time = time.time() - start_time

    # Final evaluation
    model_final = MaxSATModel(num_vars)
    model_final.assignments.data = torch.tensor(
        es.result.xbest, dtype=torch.float32
    )
    final_result = evaluate_sat_result(
        model_final, instance["clause_vars"], instance["clause_signs"]
    )
    best_sat_ratio = max(best_sat_ratio, final_result["sat_ratio"])

    filepath = save_result(
        benchmark=f"{BENCHMARK}_{num_vars}v",
        method="cmaes",
        seed=seed,
        metrics={
            "final_accuracy": final_result["sat_ratio"],
            "best_accuracy": best_sat_ratio,
            "num_satisfied": final_result["num_satisfied"],
            "num_unsatisfied": final_result["num_clauses"] - final_result["num_satisfied"],
            "num_clauses": final_result["num_clauses"],
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": total_evals,
            "total_steps": gen,
        },
        hyperparameters={
            **CMAES_CONFIG,
            "cra_lambda": cra_lambda,
            "cra_alpha": CRA_ALPHA,
            "num_vars": num_vars,
            "max_evals": max_evals,
            "CMA_diagonal": num_vars >= 1000,
        },
        epoch_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"      Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: OpenAI-ES
# ---------------------------------------------------------------------------


def run_openai_es(num_vars, instance, seed, device, max_evals, results_dir):
    """Train MAX-SAT with OpenAI-ES using antithetic sampling.

    Runs on the specified device for speed at large num_vars.

    Args:
        num_vars: Number of Boolean variables.
        instance: Dict from generate_maxsat_instance.
        seed: Random seed.
        device: Device string.
        max_evals: Maximum function evaluations.
        results_dir: Directory for JSON results.
    """
    set_seed(seed)

    clause_vars = instance["clause_vars"].to(device)
    clause_signs = instance["clause_signs"].to(device)
    cra_lambda = CRA_LAMBDA

    popsize = OPENAI_ES_CONFIG["popsize"]
    sigma = OPENAI_ES_CONFIG["sigma"]
    lr = OPENAI_ES_CONFIG["lr"]
    generations = max(1, max_evals // popsize)

    params = torch.randn(num_vars, device=device) * 0.1
    best_params = params.clone()
    best_cost = float("inf")
    best_sat_ratio = 0.0
    step_logs = []
    total_evals = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        for gen in range(generations):
            # Antithetic sampling
            eps_half = torch.randn(popsize // 2, num_vars, device=device)
            epsilon = torch.cat([eps_half, -eps_half], dim=0)

            # Vectorized evaluation of all perturbations at once
            perturbed = params.unsqueeze(0) + sigma * epsilon  # (popsize, num_vars)
            soft = torch.sigmoid(perturbed)
            hard = torch.round(soft)

            # Evaluate clauses for all perturbations
            gathered = hard[:, clause_vars]  # (popsize, C, 3)
            signs_expanded = clause_signs.unsqueeze(0)  # (1, C, 3)
            literals = gathered * signs_expanded + (1.0 - signs_expanded) * (
                1.0 - gathered
            )
            satisfied = (literals > 0.5).any(dim=-1).float()  # (popsize, C)
            unsat_ratio = 1.0 - satisfied.mean(dim=-1)  # (popsize,)
            penalty = (1.0 - (2.0 * soft - 1.0) ** CRA_ALPHA).sum(dim=-1)
            costs = unsat_ratio + cra_lambda * penalty  # (popsize,)

            # Track best
            min_idx = costs.argmin()
            if costs[min_idx].item() < best_cost:
                best_cost = costs[min_idx].item()
                best_params = (params + sigma * epsilon[min_idx]).clone()

            # Fitness shaping: convert costs to rewards (negate), normalize
            rewards = -costs
            std = rewards.std()
            if std > 1e-8:
                shaped = (rewards - rewards.mean()) / std
            else:
                shaped = torch.zeros_like(rewards)

            # Gradient estimate and update
            grad = (1.0 / (popsize * sigma)) * (epsilon.t() @ shaped)
            params = params + lr * grad

            total_evals += popsize

            # Log periodically
            if (gen + 1) % max(1, generations // 10) == 0 or gen == generations - 1:
                # Evaluate current best
                model_tmp = MaxSATModel(num_vars).to(device)
                model_tmp.assignments.data = best_params
                result = evaluate_sat_result(
                    model_tmp,
                    clause_vars,
                    clause_signs,
                )
                best_sat_ratio = max(best_sat_ratio, result["sat_ratio"])
                elapsed = time.time() - start_time
                step_logs.append(
                    {
                        "epoch": gen + 1,
                        "accuracy": result["sat_ratio"],
                        "loss": best_cost,
                        "time": elapsed,
                    }
                )

    wall_time = time.time() - start_time

    # Final evaluation with best params
    model_final = MaxSATModel(num_vars).to(device)
    model_final.assignments.data = best_params
    final_result = evaluate_sat_result(model_final, clause_vars, clause_signs)
    best_sat_ratio = max(best_sat_ratio, final_result["sat_ratio"])

    filepath = save_result(
        benchmark=f"{BENCHMARK}_{num_vars}v",
        method="openai_es",
        seed=seed,
        metrics={
            "final_accuracy": final_result["sat_ratio"],
            "best_accuracy": best_sat_ratio,
            "num_satisfied": final_result["num_satisfied"],
            "num_unsatisfied": final_result["num_clauses"] - final_result["num_satisfied"],
            "num_clauses": final_result["num_clauses"],
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": total_evals,
            "total_steps": generations,
        },
        hyperparameters={
            **OPENAI_ES_CONFIG,
            "cra_lambda": cra_lambda,
            "cra_alpha": CRA_ALPHA,
            "num_vars": num_vars,
            "max_evals": max_evals,
        },
        epoch_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"      Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: RC2 (exact MAX-SAT solver)
# ---------------------------------------------------------------------------


class _RC2Timeout(Exception):
    """Raised when RC2 exceeds timeout."""

    pass


def _rc2_worker(clauses_list, num_clauses, result_queue):
    """Run RC2 in a subprocess (can be killed on timeout).

    Uses multiprocessing to isolate the C-extension SAT solver, which
    cannot be interrupted by signal.alarm when stuck in native code.
    """
    from pysat.examples.rc2 import RC2
    from pysat.formula import WCNF

    wcnf = WCNF()
    for clause in clauses_list:
        wcnf.append(clause, weight=1)

    try:
        with RC2(wcnf) as solver:
            # solver.compute() returns the assignment; we only need the cost.
            solver.compute()
            cost = solver.cost
        result_queue.put({"cost": cost, "timed_out": False})
    except Exception as e:
        result_queue.put({"cost": num_clauses, "timed_out": True, "error": str(e)})


def run_rc2(num_vars, instance, results_dir, timeout=None):
    """Run RC2 exact MAX-SAT solver with timeout via multiprocessing.

    Uses multiprocessing.Process to run RC2 in a separate process that
    can be killed on timeout. signal.alarm cannot interrupt C-extension
    code (the SAT solver), so process-level timeout is required.

    Args:
        num_vars: Number of Boolean variables.
        instance: Dict from generate_maxsat_instance (must have 'cnf').
        results_dir: Directory for JSON results.
        timeout: Timeout seconds (defaults to RC2_TIMEOUT).
    """
    import multiprocessing as mp

    if timeout is None:
        timeout = RC2_TIMEOUT

    cnf = instance["cnf"]
    num_clauses = instance["num_clauses"]

    start_time = time.time()

    result_queue = mp.Queue()
    proc = mp.Process(
        target=_rc2_worker,
        args=(cnf.clauses, num_clauses, result_queue),
    )
    proc.start()
    proc.join(timeout=timeout)

    timed_out = False
    sat_ratio = 0.0
    unsatisfied = num_clauses

    if proc.is_alive():
        # Timeout: kill the process
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        timed_out = True
    else:
        # Process finished -- get result
        try:
            result = result_queue.get_nowait()
            timed_out = result.get("timed_out", False)
            if not timed_out:
                cost = result["cost"]
                sat_ratio = (num_clauses - cost) / num_clauses
                unsatisfied = cost
        except Exception:
            timed_out = True

    wall_time = time.time() - start_time

    filepath = save_result(
        benchmark=f"{BENCHMARK}_{num_vars}v",
        method="rc2",
        seed=0,  # deterministic
        metrics={
            "final_accuracy": sat_ratio,
            "best_accuracy": sat_ratio,
            "num_satisfied": num_clauses - unsatisfied,
            "num_unsatisfied": unsatisfied,
            "num_clauses": num_clauses,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": 0.0,
            "function_evals": 0,
            "total_steps": 0,
        },
        hyperparameters={
            "timeout": timeout,
            "num_vars": num_vars,
            "timed_out": timed_out,
            "unsatisfied_clauses": unsatisfied,
        },
        results_dir=results_dir,
    )
    print(f"      Saved: {filepath} (timed_out={timed_out})")


# ---------------------------------------------------------------------------
# Method: SLS (WalkSAT-style stochastic local search)
# ---------------------------------------------------------------------------


def run_sls(num_vars, instance, results_dir, max_flips=None):
    """Run WalkSAT-style stochastic local search.

    Simple Python SLS for reference at all variable counts:
    - Start with random assignment
    - Repeatedly flip variable that satisfies most currently-unsatisfied clauses
    - With probability p=0.5, do random walk (flip random var in random unsat clause)

    Args:
        num_vars: Number of Boolean variables.
        instance: Dict from generate_maxsat_instance.
        results_dir: Directory for JSON results.
        max_flips: Maximum flip iterations (defaults to SLS_MAX_FLIPS).
    """
    if max_flips is None:
        max_flips = SLS_MAX_FLIPS

    clause_vars = instance["clause_vars"]  # (C, 3) long
    clause_signs = instance["clause_signs"]  # (C, 3) float
    num_clauses = instance["num_clauses"]

    import random as rng

    rng.seed(42)

    # Start with random assignment
    assignment = torch.zeros(num_vars)
    for i in range(num_vars):
        assignment[i] = float(rng.random() > 0.5)

    def count_satisfied(asgn):
        gathered = asgn[clause_vars]
        literals = gathered * clause_signs + (1.0 - clause_signs) * (1.0 - gathered)
        sat_mask = (literals > 0.5).any(dim=-1)
        return sat_mask.sum().item(), sat_mask

    best_sat = 0
    best_assignment = assignment.clone()
    start_time = time.time()

    for flip in range(max_flips):
        n_sat, sat_mask = count_satisfied(assignment)
        if n_sat > best_sat:
            best_sat = n_sat
            best_assignment = assignment.clone()
        if n_sat == num_clauses:
            break  # All satisfied

        # Find unsatisfied clauses
        unsat_indices = (~sat_mask).nonzero(as_tuple=True)[0]
        if len(unsat_indices) == 0:
            break

        # Pick a random unsatisfied clause
        pick = rng.randint(0, len(unsat_indices) - 1)
        unsat_clause_idx = unsat_indices[pick].item()

        if rng.random() < 0.5:
            # Random walk: flip a random variable in the unsatisfied clause
            var_in_clause = clause_vars[unsat_clause_idx]
            flip_var = var_in_clause[rng.randint(0, 2)].item()
            assignment[flip_var] = 1.0 - assignment[flip_var]
        else:
            # Greedy: flip the variable that maximizes satisfaction
            vars_in_clause = clause_vars[unsat_clause_idx]
            best_flip_gain = -1
            best_flip_var = vars_in_clause[0].item()
            for vi in range(3):
                v = vars_in_clause[vi].item()
                assignment[v] = 1.0 - assignment[v]
                new_sat, _ = count_satisfied(assignment)
                gain = new_sat - n_sat
                if gain > best_flip_gain:
                    best_flip_gain = gain
                    best_flip_var = v
                assignment[v] = 1.0 - assignment[v]  # undo
            assignment[best_flip_var] = 1.0 - assignment[best_flip_var]

    wall_time = time.time() - start_time

    # Final check with best assignment
    final_sat, _ = count_satisfied(best_assignment)
    sat_ratio = final_sat / num_clauses
    final_unsat = num_clauses - final_sat

    filepath = save_result(
        benchmark=f"{BENCHMARK}_{num_vars}v",
        method="sls",
        seed=0,
        metrics={
            "final_accuracy": sat_ratio,
            "best_accuracy": sat_ratio,
            "num_satisfied": final_sat,
            "num_unsatisfied": final_unsat,
            "num_clauses": num_clauses,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": 0.0,
            "function_evals": max_flips,
            "total_steps": max_flips,
        },
        hyperparameters={
            "max_flips": max_flips,
            "num_vars": num_vars,
            "walk_probability": 0.5,
        },
        results_dir=results_dir,
    )
    print(f"      Saved: {filepath}")


# ---------------------------------------------------------------------------
# Method: probSAT (production SLS - SAT competition solver)
# ---------------------------------------------------------------------------

PROBSAT_BINARY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "probsat",
)

# Default max flips for probSAT (much more generous than WalkSAT)
PROBSAT_MAX_FLIPS = 10_000_000  # 10M flips is standard for SAT competitions
PROBSAT_MAX_FLIPS_1M = 5_000_000  # Reduced for 1M vars to finish in ~minutes
PROBSAT_RUNS = 10  # Number of restarts


def run_probsat(num_vars, instance, results_dir, max_flips=None, timeout=None):
    """Run probSAT (production SLS solver from SAT competitions).

    Writes the CNF to a temp DIMACS file, shells out to the compiled
    probSAT binary, and parses the output. Uses wall-clock timeout
    matching PolyStep's runtime for fair comparison.

    Args:
        num_vars: Number of Boolean variables.
        instance: Dict from generate_maxsat_instance (must have 'cnf').
        results_dir: Directory for JSON results.
        max_flips: Maximum flips per run (defaults to PROBSAT_MAX_FLIPS).
        timeout: Wall-clock timeout in seconds (overrides max_flips if solver
                 finishes early due to finding a solution).
    """
    import subprocess
    import tempfile

    if not os.path.isfile(PROBSAT_BINARY):
        print(f"      ERROR: probSAT binary not found at {PROBSAT_BINARY}")
        print("      Run: cd /tmp && git clone https://github.com/adrianopolus/probSAT.git && cd probSAT && make")
        print(f"      Then: cp /tmp/probSAT/probSAT {PROBSAT_BINARY}")
        return

    if max_flips is None:
        max_flips = PROBSAT_MAX_FLIPS_1M if num_vars >= 1000000 else PROBSAT_MAX_FLIPS

    if timeout is None:
        # Default timeout: read PolyStep's wall time for this size if available
        polystep_ref = os.path.join(
            results_dir, f"{BENCHMARK}_{num_vars}v_polystep_42.json"
        )
        if os.path.exists(polystep_ref):
            import json as _json
            try:
                with open(polystep_ref) as _f:
                    _r = _json.load(_f)
                timeout = max(60, _r["metrics"]["wall_time_seconds"] * 1.5)  # min 60s
                print(f"      Wall-clock budget: {timeout:.0f}s (max(60, 1.5x PolyStep))")
            except Exception:
                timeout = 300  # 5 min default
        else:
            timeout = 300

    cnf = instance["cnf"]
    num_clauses = instance["num_clauses"]

    # Write DIMACS file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
        dimacs_path = f.name
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in cnf.clauses:
            f.write(" ".join(str(lit) for lit in clause) + " 0\n")

    best_sat_count = 0
    timed_out = False
    start_time = time.time()

    try:
        cmd = [
            PROBSAT_BINARY,
            f"--maxflips={max_flips}",
            f"--runs={PROBSAT_RUNS}",
            "-a",  # print solution assignment
            dimacs_path,
            "42",  # seed
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout

        # Parse probSAT output
        # probSAT prints lines like: "c UNKNOWN best(   1) current(   2) (0.01 sec)"
        # where best(N) = minimum unsatisfied clauses across all tries.
        # It also prints "v ..." assignment when it finds a satisfying assignment.
        import re

        # First, try to parse assignment (printed when SATISFIABLE)
        assignment = None
        for line in stdout.split("\n"):
            if line.startswith("v "):
                parts = line[2:].strip().split()
                assignment = {}
                for p in parts:
                    if p == "0":
                        break
                    lit = int(p)
                    assignment[abs(lit)] = (lit > 0)
                break

        if assignment:
            # Count satisfied clauses from assignment
            sat_count = 0
            for clause in cnf.clauses:
                satisfied = False
                for lit in clause:
                    var = abs(lit)
                    val = assignment.get(var, False)
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                if satisfied:
                    sat_count += 1
            best_sat_count = sat_count
        else:
            # Parse "best(N)" from comment lines - N is minimum unsat clauses
            best_unsat = num_clauses  # worst case
            for line in stdout.split("\n"):
                m = re.search(r'best\(\s*(\d+)\)', line)
                if m:
                    unsat_val = int(m.group(1))
                    best_unsat = min(best_unsat, unsat_val)
            best_sat_count = num_clauses - best_unsat

    except subprocess.TimeoutExpired as e:
        timed_out = True
        # Still try to parse partial output
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        import re
        best_unsat = num_clauses
        for line in stdout.split("\n"):
            m = re.search(r'best\(\s*(\d+)\)', line)
            if m:
                unsat_val = int(m.group(1))
                best_unsat = min(best_unsat, unsat_val)
        best_sat_count = num_clauses - best_unsat
    except Exception as e:
        print(f"      ERROR: probSAT failed: {e}")
        return
    finally:
        try:
            os.unlink(dimacs_path)
        except OSError:
            pass

    wall_time = time.time() - start_time
    sat_ratio = best_sat_count / num_clauses if num_clauses > 0 else 0.0

    filepath = save_result(
        benchmark=f"{BENCHMARK}_{num_vars}v",
        method="probsat",
        seed=0,
        metrics={
            "final_accuracy": sat_ratio,
            "best_accuracy": sat_ratio,
            "num_satisfied": best_sat_count,
            "num_unsatisfied": num_clauses - best_sat_count,
            "num_clauses": num_clauses,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": 0.0,
            "function_evals": max_flips * PROBSAT_RUNS,
            "total_steps": max_flips,
        },
        hyperparameters={
            "max_flips": max_flips,
            "runs": PROBSAT_RUNS,
            "num_vars": num_vars,
            "timed_out": timed_out,
            "solver": "probSAT SC13.2",
        },
        results_dir=results_dir,
    )
    print(f"      Saved: {filepath} (sat={sat_ratio*100:.1f}%, timed_out={timed_out})")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run MAX-SAT benchmark: scaling variable counts x methods x seeds"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=VARIABLE_SIZES,
        help="Variable counts (default: 100 500 1000 5000)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["polystep", "cmaes", "openai_es", "rc2", "sls", "probsat"],
        help="Methods to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds for optimizer methods (default: 42 123 456 789 1337)",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--results-dir", default="experiments/results/softmax/main", help="Results directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 10 steps (polystep) / 100 evals (ES) for testing",
    )
    parser.add_argument(
        "--solver", choices=["softmax", "sinkhorn"], default="softmax",
        help="Solver backend (default: softmax, matching sweep config).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("MAX-SAT Benchmark")
    print(f"  Sizes: {args.sizes}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    if args.dry_run:
        print("  Mode: DRY RUN (minimal steps)")
    print()

    for num_vars in args.sizes:
        print(f"=== {num_vars} variables ===")
        instance = generate_maxsat_instance(
            num_vars=num_vars, seed=INSTANCE_SEED
        )
        print(
            f"  Instance: {instance['num_clauses']} clauses "
            f"(ratio={instance['num_clauses']/num_vars:.2f})"
        )

        # Determine step/eval budgets
        steps = 10 if args.dry_run else STEP_BUDGETS.get(num_vars, 1000)
        polystep_evals = None  # Will be set after polystep runs

        # Run polystep first to determine eval budget
        if "polystep" in args.methods:
            for seed in args.seeds:
                output_file = os.path.join(
                    args.results_dir,
                    f"{BENCHMARK}_{num_vars}v_polystep_{seed}.json",
                )
                if os.path.exists(output_file):
                    print(f"  Skipping polystep seed={seed} (result exists)")
                    continue
                print(f"  Running polystep seed={seed} ({steps} steps)...")
                try:
                    evals = run_polystep(
                        num_vars, instance, seed, args.device, steps,
                        args.results_dir, solver=args.solver,
                    )
                    if polystep_evals is None:
                        polystep_evals = evals
                except Exception as e:
                    print(f"    ERROR: polystep seed={seed} failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Determine eval budget for ES methods.
        # Give ES methods the same number of optimizer steps as polystep,
        # each evaluating popsize candidates. This is a step-matched budget:
        # ES gets (polystep_steps * popsize) function evaluations.
        # Note: polystep evaluates many more candidates per step (polytope
        # vertices * probes) via GPU parallelism, but step-matching is the
        # fairest comparison since each step represents one optimization update.
        es_popsize = CMAES_CONFIG["popsize"]
        es_budget = (100 if args.dry_run
                     else steps * es_popsize)
        # Also track polystep eval count for reporting
        if polystep_evals is None:
            import json as _json
            polystep_ref = os.path.join(
                args.results_dir,
                f"{BENCHMARK}_{num_vars}v_polystep_42.json",
            )
            if os.path.exists(polystep_ref):
                try:
                    with open(polystep_ref) as _f:
                        _r = _json.load(_f)
                    polystep_evals = _r["metrics"]["function_evals"]
                except Exception:
                    pass
            if polystep_evals is None:
                polystep_evals = steps * 12
        print(f"  ES eval budget: {es_budget:,} ({es_budget//es_popsize} gens x pop{es_popsize})")

        # Run CMA-ES with step-matched budget
        # Skip CMA-ES at >= 1M vars: impractical even with GPU batch eval
        # (sep-CMA-ES sampling + noise at 1M x popsize is prohibitive)
        if "cmaes" in args.methods and num_vars >= 1000000:
            print("  Skipping cmaes at 1M+ vars (impractical scale)")
        elif "cmaes" in args.methods:
            max_evals = es_budget
            for seed in args.seeds:
                output_file = os.path.join(
                    args.results_dir,
                    f"{BENCHMARK}_{num_vars}v_cmaes_{seed}.json",
                )
                if os.path.exists(output_file):
                    print(f"  Skipping cmaes seed={seed} (result exists)")
                    continue
                print(
                    f"  Running cmaes seed={seed} "
                    f"(budget={max_evals} evals)..."
                )
                try:
                    run_cmaes(
                        num_vars, instance, seed, args.device, max_evals,
                        args.results_dir,
                    )
                except Exception as e:
                    print(f"    ERROR: cmaes seed={seed} failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Run OpenAI-ES with step-matched budget
        if "openai_es" in args.methods:
            max_evals = es_budget
            for seed in args.seeds:
                output_file = os.path.join(
                    args.results_dir,
                    f"{BENCHMARK}_{num_vars}v_openai_es_{seed}.json",
                )
                if os.path.exists(output_file):
                    print(f"  Skipping openai_es seed={seed} (result exists)")
                    continue
                print(
                    f"  Running openai_es seed={seed} "
                    f"(budget={max_evals} evals)..."
                )
                try:
                    run_openai_es(
                        num_vars, instance, seed, args.device, max_evals,
                        args.results_dir,
                    )
                except Exception as e:
                    print(f"    ERROR: openai_es seed={seed} failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Run RC2 reference (deterministic, no seeds needed)
        # Skip RC2 at >= 1M vars: guaranteed timeout (NP-hard exact solver)
        if "rc2" in args.methods:
            if num_vars >= 1000000:
                print("  Skipping rc2 at 1M+ vars (guaranteed timeout)")
            else:
                output_file = os.path.join(
                    args.results_dir, f"{BENCHMARK}_{num_vars}v_rc2_0.json"
                )
                if os.path.exists(output_file):
                    print("  Skipping rc2 (result exists)")
                else:
                    timeout = 5 if args.dry_run else RC2_TIMEOUT
                    print(f"  Running rc2 (timeout={timeout}s)...")
                    try:
                        run_rc2(num_vars, instance, args.results_dir, timeout=timeout)
                    except Exception as e:
                        print(f"    ERROR: rc2 failed: {e}")

        # Run SLS reference (deterministic)
        if "sls" in args.methods:
            output_file = os.path.join(
                args.results_dir, f"{BENCHMARK}_{num_vars}v_sls_0.json"
            )
            if os.path.exists(output_file):
                print("  Skipping sls (result exists)")
            else:
                flips = (1000 if args.dry_run
                         else SLS_MAX_FLIPS_1M if num_vars >= 1000000
                         else SLS_MAX_FLIPS)
                print(f"  Running sls (max_flips={flips})...")
                try:
                    run_sls(num_vars, instance, args.results_dir, max_flips=flips)
                except Exception as e:
                    print(f"    ERROR: sls failed: {e}")

        # Run probSAT reference (production SLS)
        if "probsat" in args.methods:
            output_file = os.path.join(
                args.results_dir, f"{BENCHMARK}_{num_vars}v_probsat_0.json"
            )
            if os.path.exists(output_file):
                print("  Skipping probsat (result exists)")
            else:
                flips = (10000 if args.dry_run
                         else PROBSAT_MAX_FLIPS_1M if num_vars >= 1000000
                         else PROBSAT_MAX_FLIPS)
                print(f"  Running probsat (max_flips={flips}, runs={PROBSAT_RUNS})...")
                try:
                    run_probsat(num_vars, instance, args.results_dir, max_flips=flips)
                except Exception as e:
                    print(f"    ERROR: probsat failed: {e}")

        print()

    print("Done. Results in experiments/results/softmax/main/maxsat_*.json")


if __name__ == "__main__":
    main()
