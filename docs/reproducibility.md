# Reproducibility Guide

This document describes how to reproduce all experiments and results in the PolyStep paper.

## Environment Setup

### Requirements
- Python >= 3.11
- PyTorch >= 2.8
- NVIDIA GPU with CUDA support (tested on RTX 5090, 32GB VRAM, CUDA 13.0, PyTorch 2.12)
- ~10GB disk space for results

### Installation

```bash
git clone https://github.com/anindex/polystep.git
cd polystep
pip install -e ".[experiments]"
```

The `[experiments]` extra installs scipy, statsmodels, pandas, python-sat, torchvision, cma, and gymnasium.

### Datasets

MNIST is downloaded automatically via `torchvision.datasets`.
SST-2 is downloaded via HuggingFace `datasets`. No manual data setup is required.

## Running All Experiments

The master script runs all experiment phases sequentially:

```bash
cd experiments/runners
bash run_all_paper.sh
```

Estimated runtime: 8–10 GPU hours on RTX 5090.

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

Results are saved as JSON files in `experiments/results/softmax/`. Each file contains:
- `method`: optimizer used (e.g., `polystep`, `adam`, `cmaes`)
- `seed`: random seed (42, 123, 456, 789, 1337)
- `metrics`: accuracy, loss, convergence history
- `hyperparameters`: full configuration
- `environment`: hardware, PyTorch version

## View

Paper figures are pre-rendered in the paper repository (`paper/figures/`).
To aggregate CLI results from JSON:

```bash
python experiments/scripts/aggregate_results.py experiments/results/softmax/main/ --benchmark snn
```

## Seeds

All experiments use 5 fixed seeds: `{42, 123, 456, 789, 1337}`.
Results report mean ± standard deviation across seeds.

## Expected Results

| Benchmark | polystep (5-seed mean ± std) |
|-----------|------------------------------|
| MNIST | 96.0% ± 0.1% |
| SNN hard-LIF (MNIST) | 93.4% ± 0.2% |
| INT8 quantized (MNIST) | 97.1% ± 0.1% |
| Argmax attention (MNIST) | 86.8% ± 0.4% |
| Staircase (MNIST) | 93.2% ± 0.3% |
| Hard MoE (MNIST) | 90.7% ± 0.2% |
| MAX-SAT 100K vars | 98.0% ± 0.01% |
| ETTh1 timeseries | MSE 0.121 ± 0.004 |

Exact numbers may vary slightly due to hardware differences and PyTorch version.
