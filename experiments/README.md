# Experiments

Paper reproduction harness for PolyStep.

- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md): every experiment and the number it should
  produce.
- [`../docs/reproducibility.md`](../docs/reproducibility.md): environment setup and
  per-runner commands.

## What each runner makes non-differentiable

| Experiment | Runner | Non-diff op |
|-----------|--------|-------------|
| SNN hard-LIF | `runners/run_elevation.py` | threshold() |
| INT8 quantized | `runners/run_elevation.py` | round() |
| Argmax attention | `runners/run_elevation.py` | argmax() |
| Staircase | `runners/run_elevation.py` | floor() |
| Hard MoE | `runners/run_moe.py` | argmax() |
| MAX-SAT (100K-1M) | `runners/run_maxsat.py` | round() |
| MNIST | `runners/run_mnist.py` | - |
| ETTh1 timeseries | `runners/run_timeseries.py` | - |
| RL policy search | `runners/run_rl.py` | - |
| GPT-2 fine-tune | `runners/run_gpt2_finetune.py` | - |

## Layout

`runners/` experiment scripts, `baselines/` (Adam, OpenAI-ES, SPSA, and ProbSAT/SLS for
MAX-SAT; CMA-ES lives in `polystep.baselines`), `scripts/` aggregation and
microbenchmarks, `results/` result JSON.
