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
of the three softmax comparisons at 5%. This does not establish equivalence. These
tests belong to a prespecified 16-contrast family; aggregation reports which
contrasts are incomplete. The table has not been rerun for v1.1.1.

## Reproduction

From the repository root, install the experiment and test dependencies, then download
both MNIST partitions to `data/mnist`:

```bash
pip install -e ".[experiments,dev]"
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
search on hard-LIF MNIST and DVS Gesture. They use equal candidate evaluations and
equal time as separate budgets. Full and subspace Adam join the time comparison.
INT8 runs PolyStep, MeZO and Adam on the time axis only. Each of 27 configurations
is tuned on three seeds at two million evaluations or 12 minutes. The selected one
runs on ten final seeds at ten million evaluations or one hour.

The diagnostics and practical campaign require CUDA. Both wait silently until all 23
arms in `ARMS` (`run_controlled.py`) have a selected configuration and ten final seeds.
The first command below completes those arms, reusing compatible records from the
four-arm comparison above. The practical campaign also waits for the diagnostics to
finish. It writes under `experiments/results/controlled/practical` and reruns
`aggregate_practical` after each comparison. The DVS Gesture task reads frames that
`controlled_data.py` builds from
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
