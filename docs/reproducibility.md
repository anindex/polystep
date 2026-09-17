# Reproducibility

Run commands from the repository root. Install Python 3.11+, a compatible PyTorch
build, and the experiment dependencies:

```bash
pip install -e ".[experiments]"
```

Record hardware, PyTorch version, seeds, and selected configurations with results.
MNIST and SST-2 runners download their datasets. The controlled-study runner expects
MNIST under `data/mnist`; see its [protocol](../experiments/CONTROLLED_EXPERIMENTS.md).

## Protocol

- Select configurations and checkpoints on validation data; score test data afterward.
- Use `--fair` for matched subspaces, candidate budgets, and minibatch streams.
  EGGROLL uses a factored representation to retain matrix perturbations.
- Tune every method at the same validation budget. Confirm shortlisted settings at
  the final training budget.
- Report the comparison axis: optimizer steps, candidate evaluations, or elapsed time.

The architecture tables compare optimizer steps. Current ES baselines average tied
ranks; the historical architecture runs used sorting-order ranks.
`--theory-mode` selects the reference configuration with jitter and without probe
reuse or amortization. Projection assumptions still need to be checked for each task.

## Architecture experiments

```bash
python experiments/scripts/tune_gallery.py
python experiments/scripts/tune_polystep.py
bash experiments/scripts/run_revision.sh
python experiments/scripts/check_grid_complete.py experiments/results/revision
python experiments/scripts/generate_paper_tables.py \
    --results-dir experiments/results/revision \
    --readme README.md --index experiments/EXPERIMENT_INDEX.md
```

Individual runners accept `--methods` and `--seeds`:

```bash
python experiments/runners/run_elevation.py --showcases snn int8 argmax staircase
python experiments/runners/run_moe.py
python experiments/runners/run_mnist.py
python experiments/runners/run_maxsat.py
python experiments/runners/run_timeseries.py
python experiments/runners/run_rl.py --mode full --env cartpole
python experiments/runners/run_gpt2_finetune.py
```

MAX-SAT's optional probSAT comparison requires a separately built executable at
`experiments/scripts/probsat`:

```bash
git clone https://github.com/adrianopolus/probSAT
make -C probSAT
cp probSAT/probSAT experiments/scripts/probsat
```

## Results

Results are untracked JSON files under `experiments/results/`. Architecture runs
store one file per benchmark, method, and seed under `revision/`. Fields include
metrics, trajectories, hyperparameters, environment, and evaluation protocol.
Standard aggregation rejects test-selected records. Rerunning a cell overwrites its
file; copy existing results before repeating a campaign.

```bash
python experiments/scripts/aggregate_results.py experiments/results/revision --benchmark snn
python experiments/scripts/aggregate_results.py experiments/results/revision --write summary
```

See [determinism](determinism.md), [controlled experiments](../experiments/CONTROLLED_EXPERIMENTS.md),
and [forward benchmarks](performance.md#benchmarks) for their specific checks.
