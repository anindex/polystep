#!/bin/bash
# Master script - sweep-optimized production runs.
# 30 epochs (NN), 1000 steps (MAX-SAT), softmax solver, 5 seeds
# Run from repo root: bash experiments/runners/run_all_paper.sh
#
# Paper sections covered:
#   §5.2 Non-Differentiable Model Training  -> run_elevation.py (SNN, int8, argmax, staircase)
#                                            -> run_moe.py (Hard MoE)
#   §5.3 MAX-SAT Discrete Optimization      -> run_maxsat.py (100->1M variables, 1000 steps each)
#   §5.4 SNN Memory Scaling                 -> deep_snn_memory.py (O(1) vs O(T))
#   §5.5 MNIST Sanity Check                 -> run_mnist.py
#   §5.6 LSTM Time-Series Forecasting       -> run_timeseries.py (ETTh1)
#
#   - All experiments use softmax solver (including MAX-SAT full-space)
#
# Seeds: {42, 123, 456, 789, 1337} for all experiments
# Hardware: RTX 5090 32GB recommended
# Estimated GPU time: ~16-24 hours total (30 epochs + 1000-step MAX-SAT 1M)
#
# Ablation experiments are NOT included here; plan separately.

# NOTE: We do NOT use `set -e` so a segfault in one runner (e.g. transient CUDA
# crash) does not kill the chain. Each runner has its own skip-if-exists logic
# so re-running the chain is safe and resumes where it left off.
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

# Resolve Python: prefer the project venv so `python` is always defined,
# regardless of whether the venv was sourced before invoking this script.
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    PYTHON="python3"
fi
echo "Using Python: $PYTHON"

RESULTS_BASE="experiments/results"
RESULTS_DIR="$RESULTS_BASE/softmax/main"
DEVICE="cuda"
EPOCHS=30

echo "=============================================="
echo "  PolyStep - Sweep-Optimized Production Runs"
echo "  30 epochs (NN) · 1000 steps (MAX-SAT)"
echo "=============================================="
echo ""
echo "Results dir: $RESULTS_DIR"
echo "Device: $DEVICE"
echo "Seeds: 42 123 456 789 1337"
echo "Epochs: $EPOCHS"
echo ""

# ------------------------------------------------------------------
# 1. Non-Differentiable Model Training (§5.2)
#    SNN, int8, argmax, staircase - 4 showcases × 5 methods × 5 seeds
#    Configs: SNN=sm_sr2, INT8=rank8, Argmax=rank8, Staircase=sr64
# ------------------------------------------------------------------
echo ">>> [1/7] Non-Differentiable Showcases (elevation)..."
echo "    Showcases: snn, int8, argmax, staircase"
echo "    Methods: polystep, adam, cmaes, openai_es, spsa"
$PYTHON experiments/runners/run_elevation.py \
    --showcases snn int8 argmax staircase \
    --methods polystep adam cmaes openai_es spsa \
    --epochs-polystep $EPOCHS \
    --device $DEVICE \
    --results-dir $RESULTS_DIR
echo ""

# ------------------------------------------------------------------
# 2. Hard MoE (§5.2 continued)
#    Config: r4_sr12t4 (flat eps, scheduled sr, advanced features)
# ------------------------------------------------------------------
echo ">>> [2/7] Hard MoE..."
echo "    Methods: polystep, cmaes, openai_es, spsa"
$PYTHON experiments/runners/run_moe.py \
    --methods polystep cmaes openai_es spsa \
    --epochs $EPOCHS \
    --device $DEVICE \
    --results-dir $RESULTS_DIR
echo ""

# ------------------------------------------------------------------
# 3. MAX-SAT Discrete Optimization (§5.3)
#    100->1M variables, 1000 steps each, softmax solver (full-space)
#    Config: sr2000 (CosineEpsilon, momentum, amortize=3)
# ------------------------------------------------------------------
echo ">>> [3/7] MAX-SAT Scaling (100->1M, 1000 steps each)..."
$PYTHON experiments/runners/run_maxsat.py \
    --sizes 100 500 1000 5000 20000 100000 1000000 \
    --methods polystep cmaes openai_es sls \
    --device $DEVICE \
    --results-dir $RESULTS_DIR
echo ""

# ------------------------------------------------------------------
# 4. SNN Memory Scaling (§5.4)
#    polystep O(1) vs BPTT O(T) at varying timesteps
# ------------------------------------------------------------------
echo ">>> [4/7] SNN Memory Scaling..."
$PYTHON src/polystep/benchmarks/deep_snn_memory.py \
    --output-dir $RESULTS_DIR \
    --device $DEVICE 2>&1 || echo "  (memory scaling had warnings)"
echo ""

# ------------------------------------------------------------------
# 5. MNIST Sanity Check (§5.5)
#    5 methods × 5 seeds, softmax solver default
# ------------------------------------------------------------------
echo ">>> [5/7] MNIST Benchmark..."
$PYTHON experiments/runners/run_mnist.py \
    --methods polystep cmaes openai_es spsa adam \
    --device $DEVICE \
    --results-dir $RESULTS_DIR
echo ""

# ------------------------------------------------------------------
# 6. LSTM Time-Series Forecasting (§5.6)
#    ETTh1 dataset, 5 methods × 5 seeds, softmax solver default
# ------------------------------------------------------------------
echo ">>> [6/7] LSTM Time-Series (ETTh1)..."
$PYTHON experiments/runners/run_timeseries.py \
    --methods polystep cmaes openai_es spsa adam \
    --device $DEVICE \
    --results-dir $RESULTS_DIR
echo ""

echo "=============================================="
echo "  Complete! Results: $RESULTS_DIR"
echo "  Aggregate: python experiments/scripts/aggregate_results.py $RESULTS_DIR"
echo "=============================================="
