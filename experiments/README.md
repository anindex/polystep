# Experiments

Paper reproduction harness for PolyStep. All results use 5 seeds `{42, 123, 456, 789, 1337}`.

> Note: result JSON files and analysis scripts under this directory retain the
> legacy `pstorch` key as the method-name string (the project was renamed from
> `pstorch` to `polystep` for public GitHub release). The library, public API, and paper use `polystep`;
> the alias is preserved here only so cached result files remain readable
> without a full rerun.

## Quick start

```bash
pip install -e ".[experiments]"
bash experiments/runners/run_all_paper.sh             # ~8–10 GPU hours (RTX 5090)
python experiments/scripts/aggregate_results.py experiments/results/softmax/main/ --benchmark snn
```

## Experiment index

| Experiment | Runner | Non-diff op |
|-----------|--------|-------------|
| SNN hard-LIF | `runners/run_elevation.py` | threshold() |
| INT8 quantized | `runners/run_elevation.py` | round() |
| Argmax attention | `runners/run_elevation.py` | argmax() |
| Staircase | `runners/run_elevation.py` | floor() |
| Hard MoE | `runners/run_moe.py` | argmax() |
| MAX-SAT (100K–1M) | `runners/run_maxsat.py` | round() |
| MNIST | `runners/run_mnist.py` | - |
| ETTh1 timeseries | `runners/run_timeseries.py` | - |
| RL policy search | `runners/run_rl.py` | - |
| GPT-2 fine-tune | `runners/run_gpt2_finetune.py` | - |
| OT vs Softmax ablation | `runners/ablation_ot_vs_softmax.py` | - |
| Ablation grid | `runners/run_fill_ablation_grid.py` | - |

See [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) for detailed reproduction commands and result artifacts.

## Layout

```
experiments/
  runners/       Experiment scripts
  baselines/     CMA-ES, OpenAI-ES, SPSA, SLS/PySAT
  scripts/       Result aggregation utilities
  results/       Result JSON files (softmax/)
```

## Baselines

- **Adam** - gradient-based (sanity check)
- **CMA-ES** - covariance matrix adaptation
- **OpenAI-ES** - evolution strategies
- **SPSA** - simultaneous perturbation
- **probSAT** - domain-specialized SLS
