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

The architecture tables compare optimizer steps.
`--theory-mode` selects the reference configuration with jitter and without
amortization. Projection assumptions still need to be checked for each task.

## Architecture experiments

```bash
python experiments/scripts/tune_gallery.py
python experiments/scripts/tune_polystep.py
bash experiments/scripts/run_revision.sh
python experiments/scripts/check_grid_complete.py experiments/results/revision
python experiments/scripts/generate_tables.py \
    --results-dir experiments/results/revision \
    --readme README.md --index experiments/EXPERIMENT_INDEX.md
```

The table generator updates Markdown only and rejects test-selected results.

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

With the datasets and CUDA available, run the practical-study update checks with
`PYTHONPATH=src:. pytest tests/test_practical.py -m gpu`.

## Results recorded before 1.0

The tables in the README and the experiment index were recorded before 1.0. Fixes to
the harness and the library since then change the following results, so a rerun
will not reproduce them:

- Random search now scores the incumbent and the candidate on the same minibatch,
  two evaluations per step. The recorded column compared each candidate with the
  incumbent's loss on an earlier batch.
- SPSA on the evaluation- and wall-clock-matched axes took its gain offset `A` from
  the evaluation budget or the time cap. The runners now use a tenth of PolyStep's
  step count on every axis, so the step-matched rows are unchanged.
- The RL results predate four runner fixes. ES now averages the ranks of tied
  returns, `rl_env_steps` counts training steps only, success on Acrobot and
  MountainCar means reaching the goal, and every CartPole arm except PPO and DQN
  runs on the analytic simulator.
- Capped `HybridSubspace` runs used the old budget split, which kept full-width
  weight matrices out of `max_subspace_dim`. The RL configurations set a cap of 24,
  yet CartPole searched all 114 parameters and Acrobot 68 coordinates.
- For the same reason, the 0.12 hybrid timing row in
  [performance](performance.md#cpu-measurements) was labelled 512 coordinates but
  measured 1,542. That table has since been remeasured at 1,024.
- The SNN/LIF and hard-MoE PolyStep configurations use `biased_rotation` and
  periodic absorbs, and 1.0 changes both, so their seeded trajectories differ.
- The variant sweep lost its CMA arms (`cma_subspace`, `covariance`, and the
  `cma_x_adaptive` pair), the `tempered_softmax` solver arm, and the data-dependent
  Sinkhorn initialization arm, because 1.0 removes those features. The sweep's
  `block_x_repr`/`per_layer_hybrid` arm now runs the Sinkhorn solver instead of
  softmax, so that row changes too.
- The surviving `quadratic/trust_region`, `trust_x_quad/trust_quad_k3`,
  `quadratic/biased_rotation`, and `quadratic/newton` sweep arms change as well. The
  trust region now applies its new radius on the step that measured the ratio
  instead of one step later, `biased_rotation` turns each frame with a single
  Householder reflection, and Newton refinement runs before momentum and keeps the
  Sinkhorn warm start.
- The architecture runs ranked ES populations in sorting order; the ES baselines
  now average tied ranks.

Several other 1.0 changes move results in the last bit (see the
[changelog](../CHANGELOG.md)), so the remaining rows need not match bit for bit
either.
