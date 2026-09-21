# Controlled hard-LIF comparison

Four weighting rules share the same hard-LIF MNIST model, orthoplex probes, dense QR
basis, and natural displacement update. Each rule uses 27 configurations, three
tuning seeds, and ten final seeds. Configuration and checkpoint selection use
validation data only.

| Rule | Test accuracy, mean ± sample SD |
|---|---:|
| Softmax | 74.487 ± 9.731% |
| Linear | 71.957 ± 10.943% |
| Exponential rank | 78.415 ± 10.096% |
| Greedy | 75.887 ± 9.267% |

All 40 final runs have finite results. Unadjusted paired sign-flip tests reject none
of the three softmax comparisons at 5%. These are part of a prespecified 16-contrast
family; incomplete contrasts remain explicit in aggregation.

## Reproduction

Install the experiment dependencies and download both MNIST partitions to
`data/mnist` with `torchvision.datasets.MNIST(..., download=True)`.

```bash
PYTHONPATH=src:. python -m experiments.runners.run_controlled campaign --compiled --chunk 512 \
  --arms orthoplex_softmax_natural orthoplex_greedy_natural \
  orthoplex_linear_natural orthoplex_rank_natural
PYTHONPATH=src:. python -m experiments.scripts.aggregate_controlled
```

Budgets are two million candidate evaluations per tuning seed and ten million per
final seed. The fixed split contains 52,976 update examples, 6,000 validation
examples, 10,000 test examples, and a reserved 1,024-image training holdout.
Compatible existing records are reused; use an empty result directory for a fresh run.

CPU checks require no dataset:

```bash
PYTHONPATH=src:. pytest tests/test_controlled.py tests/test_practical.py
```

The [paper](https://arxiv.org/abs/2605.01928) gives the full statistical
protocol. Aggregators retain individual seed outcomes and do not fill missing runs.
