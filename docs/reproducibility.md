# Reproducibility Guide

This document describes how to reproduce all experiments and results in the PolyStep
paper.

## Environment Setup

### Requirements
- Python >= 3.11
- PyTorch >= 2.8 is the package floor; the committed results were produced on 2.10-2.12
- NVIDIA GPU with CUDA support (tested on RTX 5090, 32GB VRAM, CUDA 13.0)
- ~10GB disk space for results

### Installation

```bash
git clone https://github.com/anindex/polystep.git
cd polystep
pip install -e ".[experiments]"
```

The `[experiments]` extra installs numpy, pandas, python-sat, torchvision, cma, evotorch,
snntorch, datasets, and gymnasium.

### Datasets

MNIST is downloaded automatically from the Google Cloud Storage mirror as raw IDX
archives. The loaders in `polystep.benchmarks.utils` avoid torchvision so the core
reproduction path needs no extra vision dependency. SST-2 is downloaded via HuggingFace
`datasets` by `runners/run_gpt2_finetune.py`. No manual data setup is required.

## Running All Experiments

The master script runs the five main benchmarks in order. RL, GPT-2 and the ablations are
separate invocations, listed below.

```bash
cd experiments/runners
bash run_all_paper.sh              # ~16-24 GPU hours (RTX 5090)
```

## Individual Experiments

### Non-Differentiable Showcases (SNN, INT8, Argmax, Staircase)

```bash
python experiments/runners/run_elevation.py --showcases snn int8 argmax staircase --seeds 42 123 456 789 1337
```

### Hard MoE Routing

```bash
python experiments/runners/run_moe.py
```

### MNIST (Sanity Check)

```bash
python experiments/runners/run_mnist.py          # ~30 min
```

### MAX-SAT

```bash
python experiments/runners/run_maxsat.py         # Scales: 100 -> 1M variables
```

### Time Series (ETTh1)

```bash
python experiments/runners/run_timeseries.py
```

### RL Policy Search

```bash
python experiments/runners/run_rl.py --mode full --env cartpole
python experiments/runners/run_rl.py --mode full --env acrobot
```

### GPT-2 Fine-Tuning (Limitation Study)

```bash
python experiments/runners/run_gpt2_finetune.py
```

## Result Artifacts

Results are saved as JSON files under `experiments/results/softmax/`, laid out as `main/`
(SNN, INT8, argmax, staircase, MNIST, timeseries, MAX-SAT, MoE), `ablations/`,
`scalability/` and `rl/`. Each file contains:
- `benchmark`: which experiment produced it
- `method`: optimizer used (e.g. `polystep`, `adam`, `cmaes`)
- `seed`: random seed (42, 123, 456, 789, 1337)
- `metrics`: accuracy, loss, convergence history
- `epoch_logs` / `step_logs`: per-epoch and per-step traces
- `hyperparameters`: full configuration
- `environment`: hardware, PyTorch version
- `timestamp`: when the run finished

## View

`docs/figures/` and `examples/figures/` hold the rendered figures; the paper versions are
in the arXiv preprint (arXiv:2605.01928). To aggregate results from JSON:

```bash
python experiments/scripts/aggregate_results.py experiments/results/softmax/main/ --benchmark snn
```

## Expected numbers

Every experiment uses the 5 fixed seeds `{42, 123, 456, 789, 1337}` and reports mean +/-
std across them. The per-benchmark values are in
[`../experiments/EXPERIMENT_INDEX.md`](../experiments/EXPERIMENT_INDEX.md).
