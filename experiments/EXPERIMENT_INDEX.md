# Experiment Index

What each experiment measures, which runner produces it, and the full result tables.
The README carries a short version of the same tables; both are generated from
`results/revision/`, so neither can drift from the other:

```bash
python experiments/scripts/generate_paper_tables.py \
    --results-dir experiments/results/revision \
    --readme README.md --index experiments/EXPERIMENT_INDEX.md
```

All runners default to the honest protocol (validation-selected checkpoints) and 5
seeds `{42, 123, 456, 789, 1337}`. Results vary slightly with hardware and PyTorch
version. Setup and per-runner commands:
[`../docs/reproducibility.md`](../docs/reproducibility.md).

## Results

<!-- BENCH:START -->
### Non-differentiable tasks

Test accuracy %. A dash means the cell is not in this release.

| Task | PolyStep | OpenAI-ES | CMA-ES | EGGROLL | MeZO | SPSA | Random search | Adam (surrogate) | Non-diff op |
|---|---|---|---|---|---|---|---|---|---|
| SNN/LIF (MNIST) | 93.0 ± 0.2 | 79.6 ± 5.2 | 77.1 ± 13.3 | 55.3 ± 8.3 | 55.4 ± 13.3 | 53.9 ± 3.0 | 16.4 ± 4.4 | 86.3 ± 10.6 | `threshold()` |
| Int8 quantized | 97.0 ± 0.1 | 94.4 ± 0.2 | 91.6 ± 0.4 | 83.4 ± 1.2 | 80.7 ± 0.9 | 82.5 ± 0.5 | 29.2 ± 4.9 | 97.8 ± 0.1 | `round()` |
| Argmax attention | 86.6 ± 0.4 | 80.0 ± 0.4 | 79.5 ± 0.5 | 71.0 ± 0.4 | 71.6 ± 1.2 | 67.4 ± 1.3 | 42.8 ± 1.9 | 88.6 ± 0.1 | `argmax()` |
| Staircase activation | 94.3 ± 0.1 | 88.9 ± 0.5 | 89.2 ± 0.4 | 52.0 ± 1.5 | 61.1 ± 2.8 | 45.2 ± 4.2 | 18.1 ± 4.8 | 97.5 ± 0.1 | `floor()` |
| Hard MoE routing | 90.6 ± 0.2 | 82.5 ± 0.8 | 77.8 ± 1.5 | 52.5 ± 1.5 | 55.3 ± 3.0 | 25.4 ± 3.5 | 13.7 ± 1.9 | - | `argmax()` |

### MAX-SAT (% clauses satisfied)

| Variables | PolyStep | CMA-ES | OpenAI-ES | probSAT | RC2 |
|---|---|---|---|---|---|
| 100 | 98.0 ± 0.7 | 99.0 ± 0.3 | 99.4 ± 0.1 | 99.8 | 99.8 |
| 500 | 98.1 ± 0.2 | 98.4 ± 0.2 | 98.1 ± 0.2 | 100.0 | timeout |
| 1,000 | 98.1 ± 0.2 | 97.3 ± 0.1 | 96.9 ± 0.2 | 100.0 | timeout |
| 5,000 | 98.1 ± 0.1 | 95.2 ± 0.2 | 92.9 ± 0.2 | 99.9 | timeout |
| 20,000 | 98.2 ± 0.0 | 92.9 ± 0.2 | 90.5 ± 0.1 | 99.8 | timeout |
| 100,000 | 98.1 ± 0.0 | 90.7 ± 0.1 | 88.9 ± 0.0 | 99.6 | timeout |

### Differentiable sanity checks

| Task | PolyStep | Adam |
|---|---|---|
| MNIST (2-layer MLP) | 96.8 ± 0.1 | 97.7 ± 0.1 |
| ETTh1 (LSTM, MSE; lower is better) | 0.253 ± 0.023 | 0.247 ± 0.023 |
<!-- BENCH:END -->

## Experiments

### Non-differentiable tasks (primary)

| # | Task | Runner | Non-diff op |
|---|------|--------|-------------|
| 1 | SNN hard-LIF | `run_elevation.py` | `threshold()` |
| 2 | INT8 quantized | `run_elevation.py` | `round()` |
| 3 | Argmax attention | `run_elevation.py` | `argmax()` |
| 4 | Staircase | `run_elevation.py` | `floor()` |
| 5 | Hard MoE routing | `run_moe.py` | `argmax()` |
| 6 | MAX-SAT, 100 to 100K vars | `run_maxsat.py` | `round()` |

### Sanity checks

| # | Task | Runner | Note |
|---|------|--------|------|
| 7 | MNIST (101K MLP) | `run_mnist.py` | fully differentiable; Adam is the reference |
| 8 | ETTh1 timeseries | `run_timeseries.py` | reported as MSE, lower is better |
| 9 | GPT-2 SST-2 (head-only) | `run_gpt2_finetune.py` | see [`../LIMITATIONS.md`](../LIMITATIONS.md) |

### RL policy search

| Task | Runner |
|------|--------|
| CartPole / Acrobot (vanilla + hardened) | `run_rl.py` |

### Ablations

| Study | Runner |
|-------|--------|
| MAX-SAT scaling | `run_maxsat_softmax_scaling.py` |
| Optimizer variants (solver, subspace, block, schedule) | `variant_sweep.py` |
| Unaccelerated reference config (`--theory-mode`) | `run_revision.sh` |

## Reading the tables

- **Adam is not a gradient-free peer.** It backprops, on a smoothed surrogate where
  the task is not differentiable. It is there to bound the gap to a gradient method.
- **PolyStep's niche is hard non-differentiability.** It beats the gradient surrogate
  on the SNN's LIF threshold and loses on int8, argmax and staircase, where an
  accurate smooth surrogate exists. On the differentiable checks Adam is ahead.
- **MAX-SAT is a scaling claim, not a win.** PolyStep holds ~98% from 100 to 100,000
  variables while the ES baselines decay with problem size, but probSAT stays above
  99.5% throughout. RC2 is exact below ~500 variables and times out above.
- **Budgets are matched on optimizer steps.** Population methods are fixed at
  `FAIR_POPSIZE` so one generation costs one step. The wall-clock and
  evaluation-matched arms live in subdirectories and are reported separately.

## Result layout

Run outputs are not tracked in git. One JSON per `(benchmark, method, seed)` under
`results/revision/`, plus:

```text
experiments/results/revision/
  logs/         per-cell stdout
  theory/       --theory-mode cells
  evalmatched/  evaluation-matched arm
  wallclock/    wall-clock-matched arm
```

Per-file JSON schema: [`../docs/reproducibility.md`](../docs/reproducibility.md).

## Hardware

NVIDIA RTX 5090, Python 3.11+, PyTorch 2.8+ (tested with 2.13+cu130 on Ubuntu Linux).
