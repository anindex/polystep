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
`data/mnist`:

```bash
python -c "from torchvision.datasets import MNIST; [MNIST('data/mnist', train=t, download=True) for t in (True, False)]"
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

## Practical controls

The practical controls compare PolyStep with OpenAI-ES, EGGROLL, SPSA, MeZO and random
search at equal candidate evaluations and at equal time on the hard-LIF MNIST and DVS
Gesture tasks, with full and subspace Adam added on the time axis. INT8 runs PolyStep,
MeZO and Adam on the time axis only. Each of 27 configurations is tuned on three seeds
at two million evaluations or 12 minutes, and the selected one runs on ten final seeds
at ten million evaluations or one hour.

The diagnostics and the practical campaign both wait, without output, until all 23
arms in `ARMS` (`run_controlled.py`) have a selected configuration and ten final
seeds, so the first command below runs every arm, reusing records of the four above.
The practical campaign also waits for the diagnostics to finish. It writes under
`experiments/results/controlled/practical` and reruns `aggregate_practical` after each
comparison. The DVS Gesture task reads frames that `controlled_data.py` builds from
the original `DvsGesture.tar.gz` archive:

```bash
PYTHONPATH=src:. python -m experiments.runners.run_controlled campaign --compiled --chunk 512 \
  --arms $(PYTHONPATH=src:. python -c "from experiments.runners.run_controlled import ARMS; print(*ARMS)")
PYTHONPATH=src:. python -m experiments.runners.controlled_data --archive DvsGesture.tar.gz
PYTHONPATH=src:. python -m experiments.scripts.controlled_diagnostics
PYTHONPATH=src:. python -m experiments.scripts.practical_campaign campaign
```

With the datasets and CUDA available, run the practical update checks with
`PYTHONPATH=src:. pytest tests/test_practical.py -m gpu`.
