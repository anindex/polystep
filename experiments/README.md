# Experiments

- [Results](EXPERIMENT_INDEX.md)
- [Setup and reproduction](../docs/reproducibility.md)
- [Controlled hard-LIF comparison](CONTROLLED_EXPERIMENTS.md)
- [Forward benchmarks](../docs/performance.md#benchmarks)

## Runners

| Task | Runner |
|---|---|
| SNN, INT8, argmax attention, staircase | `runners/run_elevation.py` |
| Hard mixture of experts | `runners/run_moe.py` |
| MAX-SAT | `runners/run_maxsat.py` |
| MNIST | `runners/run_mnist.py` |
| ETTh1 | `runners/run_timeseries.py` |
| RL policy search | `runners/run_rl.py` |
| GPT-2 head tuning | `runners/run_gpt2_finetune.py` |
| Controlled update rules | `runners/run_controlled.py` |
| Candidate- and time-budget comparisons | `runners/run_practical.py` |
| Synthetic search methods | `scripts/bench_polytope.py` |

## Search benchmarks

The download-free suite covers MLPs, CNNs, attention, spiking, hard routing, and
quantization. It compares existing optimizer options with global-subspace, rank-two,
momentum-subspace, and same-batch acceptance experiments. All are opt-in.

```bash
PYTHONPATH=src:. python experiments/scripts/bench_polytope.py --seeds 0 1 2 3 4 --budget 4096 --wall
PYTHONPATH=src:. python experiments/scripts/bench_polytope.py --streaming --arms hybrid_deferred global_orthoplex same_batch_accept
```

The suite matches initialization, data, probe norm, and temperature. Validation
selects checkpoints; the test split is scored once afterward. Acceptance evaluations
count toward the candidate budget, and validation overhead counts toward wall time.
These small synthetic tasks do not establish performance on the paper workloads.

The experimental directions draw on [EGGROLL](https://arxiv.org/abs/2511.16652) and
[MpSub](https://arxiv.org/abs/2609.07666); they are adaptations, not reproductions.

`runners/` contains training code, `scripts/` contains aggregation and profiling,
and `results/` holds untracked output files.
