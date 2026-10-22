# Reproducibility

Run commands from the repository root. Install Python 3.11+, a compatible PyTorch
build, and the experiment dependencies:

```bash
pip install -e ".[experiments,rl]"
```

`rl` adds stable-baselines3 for the PPO and DQN arms. `run_revision.sh` uses
`.venv/bin/python`; set `PY` to use another interpreter.

Record hardware, PyTorch version, seeds, and selected configurations with results.
The MNIST, ETTh1 and SST-2 runners download their datasets. The controlled-study
runner expects MNIST under `data/mnist`; see the
[controlled hard-LIF comparison](../experiments/CONTROLLED_EXPERIMENTS.md).

## Protocol

- Select configurations and checkpoints on validation data; score test data afterward.
- Use `--fair` for matched subspaces, candidate budgets, and minibatch streams.
  EGGROLL uses a factored representation to retain matrix perturbations.
- Tune every method at the same validation budget. Confirm shortlisted settings at
  the final training budget.
- Report the comparison axis: optimizer steps, candidate evaluations, or elapsed time.

Population methods (OpenAI-ES, CMA-ES, EGGROLL) get PolyStep's candidate-evaluation
budget. SPSA, MeZO and random search, which score at most two candidates per step, get
its step count. On argmax attention, hard MoE and ETTh1, the population methods instead
get 32 evaluations per PolyStep step, spent at their own population size, so their
step counts differ from PolyStep's. Adam trains for PolyStep's 30 epochs there, except
on ETTh1, where it trains for 50. On MAX-SAT, CMA-ES and OpenAI-ES get 50 evaluations
per PolyStep step. OpenAI-ES scores 50 candidates at each of PolyStep's steps; CMA-ES
doubles its population at each restart, so it takes fewer generations. probSAT and RC2
run once per instance size. Set `POLYSTEP_MATCH_AXIS=evals` or `wallclock` to force
one axis. Gradient-free baseline records from `run_elevation.py`, `run_moe.py`,
`run_mnist.py` and `run_timeseries.py` store the axis as `hyperparameters.match_axis`;
PolyStep, Adam, persistence, MAX-SAT and RL records carry none.

`--theory-mode` runs the unaccelerated reference configuration: jittered probe and
step radii, independent rotations, flat epsilon, a decaying step radius, orthoplex
probes, and no momentum, amortization or Anderson acceleration.

## Architecture experiments

```bash
python experiments/scripts/tune_gallery.py --showcases snn int8 argmax staircase mnist moe
for showcase in snn int8 argmax staircase mnist moe; do
    python experiments/scripts/tune_polystep.py --showcase "$showcase"
done
bash experiments/scripts/run_revision.sh
python experiments/scripts/check_grid_complete.py experiments/results/revision
python experiments/scripts/generate_tables.py \
    --results-dir experiments/results/revision \
    --readme README.md --index experiments/EXPERIMENT_INDEX.md
```

`run_elevation.py`, `run_mnist.py` and `run_moe.py` record the tuning settings they
applied under `hyperparameters.tuning_provenance`; the other runners do not. The table
generator updates Markdown only and rejects test-selected results.

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

They write to `experiments/results/revision` unless given `--results-dir`; the GPT-2
runner writes to `experiments/results`. `run_revision.sh` writes to
`experiments/results/revision` and its theory-mode runs to `theory/` there. It passes
`--fair` to every runner that accepts it, which excludes `run_maxsat.py` and
`run_rl.py`, and `--force` to every job except the RL ones.

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
Standard aggregation rejects test-selected records. The runners above skip a cell
whose result file exists; all but `run_gpt2_finetune.py` take `--force` to rerun it.
`run_revision.sh` refuses to start while `experiments/results/revision` holds
results; move them aside first.

```bash
python experiments/scripts/aggregate_results.py experiments/results/revision --benchmark snn
python experiments/scripts/aggregate_results.py experiments/results/revision --write summary
```

The `theory/`, `evalmatched/` and `wallclock/` subdirectories of
`experiments/results/revision` hold the separate protocols. `run_revision.sh` fills
`theory/`. For the other two, rerun the runners with `--fair` and
`--results-dir experiments/results/revision/evalmatched` under
`POLYSTEP_MATCH_AXIS=evals` (raise `POLYSTEP_FAIR_POPSIZE` there, so the population
methods keep their throughput), or with `.../wallclock` under
`POLYSTEP_MATCH_AXIS=wallclock`. The wall-clock runs read PolyStep's time for the same
cell from the parent directory. `run_elevation.py`, `run_moe.py`, and
`run_timeseries.py` read the variable; `run_maxsat.py` takes `--es-eval-budget`.

See [determinism](determinism.md), [controlled experiments](../experiments/CONTROLLED_EXPERIMENTS.md),
and [forward benchmarks](performance.md#benchmarks) for their specific checks.

## Changes since the published tables

The tables in the README and the experiment index, and the RL and variant-sweep
results, were recorded before 1.0. Fixes to the harness and the library since then
change the following results, so a rerun does not reproduce them:

- Random search scores the incumbent and the candidate on the same minibatch, two
  evaluations per step. The published column compared each candidate with the
  incumbent's loss on an earlier batch.
- The ES baselines average the ranks of tied returns and losses. The published RL and
  architecture runs ranked ties in sorting order.
- The RL results predate three more runner fixes: `rl_env_steps` counts training
  steps only, success on Acrobot means reaching the goal, and every CartPole arm
  except PPO and DQN runs on the analytic simulator.
- Capped `HybridSubspace` runs used the old budget split, which kept full-width
  weight matrices out of `max_subspace_dim`. The RL configurations set a cap of 24,
  yet CartPole searched all 114 parameters and Acrobot 68 coordinates.
- The SNN/LIF and hard-MoE PolyStep configurations use `biased_rotation` and
  periodic absorbs, and 1.0 changes both, so their seeded trajectories differ.
- The variant sweep (`experiments/runners/variant_sweep.py`) has lost its CMA arms
  (`cma_subspace`, `covariance`, and the `cma_x_adaptive` pair), the
  `tempered_softmax` solver arm, and the data-dependent Sinkhorn initialization arm,
  because 1.0 removes those features. Its `quadratic/multifidelity` arm is gone as
  well: it never ran the screen and repeated the baseline. The sweep's
  `block_x_repr`/`per_layer_hybrid` arm runs the Sinkhorn solver instead of softmax,
  so that row changes too.
- The surviving `quadratic/trust_region`, `trust_x_quad/trust_quad_k3`,
  `quadratic/biased_rotation`, and `quadratic/newton` sweep arms change as well. The
  trust region applies its new radius on the step that measured the ratio instead of
  one step later, `biased_rotation` turns each frame so the vertex nearest the
  previous descent direction points along it, and Newton refinement runs before
  momentum and keeps the Sinkhorn warm start.
- Sinkhorn stops when the relative marginal error falls below `threshold`, now 3e-5
  by default. Once the error stops falling, the test is floored at the precision the
  dtype can reach on the costs that carry mass. The old absolute 1e-6
  test loosened in proportion to the particle count and could not pass in FP32 at a
  sharp epsilon. Solves at a small epsilon, over few particles or over many particles
  with uneven costs now stop at a different check, as do runs with over-relaxation,
  adaptive omega, Anderson acceleration or `ProgressiveEpsilon`.
- Every `HybridSubspace` and `FactoredSubspace` run draws new bases. Their layer
  seeds now hash with CRC-32, because Adler-32 repeated bases across rotation steps
  such as 120 and 201 and across layers such as `120.weight` and `201.weight`. A
  subspace built without a `seed` also takes the optimizer's seed instead of 0. This
  changes examples 05, 06, 10 and 11, the paper runners and the fairness EGGROLL and
  Hybrid arms. `AdaptiveSubspace` and full-space runs are unchanged.

Several other 1.0 changes move results in the last bit (see the
[changelog](../CHANGELOG.md)), so the remaining rows need not match bit for bit
either.
