# Experiment Index

All runners default to the honest protocol (val-selected checkpoints). Seeds: `{42, 123, 456, 789, 1337}`.

## Installation

```bash
pip install -e ".[experiments]"
```

## Experiments

### Non-Differentiable Tasks (Primary)

| # | Task | Runner | Result (5-seed mean ± std) |
|---|------|--------|---------------------------|
| 1 | SNN hard-LIF | `run_elevation.py` | 93.4% ± 0.3 |
| 2 | INT8 quantized | `run_elevation.py` | 97.2% ± 0.1 |
| 3 | Argmax attention | `run_elevation.py` | 86.8% ± 0.4 |
| 4 | Staircase | `run_elevation.py` | 93.2% ± 0.3 |
| 5 | Hard MoE routing | `run_moe.py` | 90.7% ± 0.2 |
| 6 | MAX-SAT (100K vars) | `run_maxsat.py` | 98.0% sat ratio |
| 7 | MAX-SAT (1M vars) | `run_maxsat.py` | 92.6% sat ratio |

### Sanity Checks

| # | Task | Runner | Result |
|---|------|--------|--------|
| 8 | MNIST (101K MLP) | `run_mnist.py` | 96.0% ± 0.1 |
| 9 | ETTh1 timeseries | `run_timeseries.py` | MSE 0.121 ± 0.004 |
| 10 | GPT-2 SST-2 (head-only) | `run_gpt2_finetune.py` | 76.8% (limitation) |

### RL Policy Search

| # | Task | Runner |
|---|------|--------|
| 11 | CartPole / Acrobot (vanilla + hardened) | `run_rl.py` |

### Ablations

| # | Study | Runner |
|---|-------|--------|
| 13 | OT vs Softmax solver | `ablation_ot_vs_softmax.py` |
| 14 | Epsilon / radius / particles / subspace grid | `run_fill_ablation_grid.py` |
| 15 | MAX-SAT scaling (100–1M vars) | `run_maxsat_softmax_scaling.py` |

## Reproduce all

```bash
cd experiments/runners
bash run_all_paper.sh              # ~8–10 GPU hours (RTX 5090)
```

Results are saved to `experiments/results/`. Figures referenced in the paper
are pre-rendered in the paper repository (`paper/figures/`).

## Result layout

```
experiments/results/softmax/
  main/          SNN, INT8, argmax, staircase, MNIST, timeseries, MAX-SAT, MoE
  ablations/     Epsilon, radius, particles, compile, subspace, convergence, OT
  scalability/   Parameter scaling, sparse projection, memory
  rl/            RL policy search results (CartPole, Acrobot)
```

Each JSON file contains method, dataset, seed, config, and metrics (accuracy/loss/wallclock).

## Hardware

- NVIDIA RTX 5090, Python 3.11+, PyTorch 2.8+ (tested with 2.12+cu130 on Ubuntu Linux).
