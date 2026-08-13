# Reproducibility

How to reproduce the numbers in `README.md` and in the paper. Everything below runs
from the repository root.

## Environment

- Python >= 3.11, PyTorch >= 2.8 (results produced on 2.13+cu130)
- NVIDIA GPU (tested on an RTX 5090, 32 GB, CUDA 13.0)
- ~10 GB of disk for results

```bash
git clone https://github.com/anindex/polystep.git
cd polystep
pip install -e ".[experiments]"
```

MNIST downloads itself as raw IDX archives; SST-2 comes from HuggingFace `datasets`
inside `run_gpt2_finetune.py`. No manual data setup.

MAX-SAT compares against probSAT, which is a third-party binary and is not shipped.
Build it and drop the executable at `experiments/scripts/probsat`:

```bash
git clone https://github.com/adrianopolus/probSAT && make -C probSAT
cp probSAT/probSAT experiments/scripts/probsat
```

`run_maxsat.py` skips the probSAT rows when the binary is absent.

## The protocol

Three properties make the comparison fair, and all three are on by default:

- **Honest evaluation.** Checkpoints are selected on validation and the test set is
  scored once. `tests/test_no_test_set_leakage.py` fails the build if a runner can
  reach the test split during selection. `--allow-test-leakage` exists only so that
  test can prove the guard fires.
- **Matched budgets (`--fair`).** Same subspace, same candidate-evaluation budget and
  the same minibatch stream across gradient-free methods. Population methods are
  fixed at `FAIR_POPSIZE` so one generation costs one optimizer step, which is the
  axis the headline tables match on.
- **Equal tuning.** Every method, PolyStep included, gets the same validation-only
  sweep over `TUNING_GRID` before its headline run.

`--theory-mode` additionally restricts PolyStep to the unaccelerated reference
configuration (mollifier jitter on, amortization and probe reuse off).

## Running the campaign

```bash
# 1. Tune every method on validation, at equal cost.
python experiments/scripts/tune_gallery.py          # baselines
python experiments/scripts/tune_polystep.py         # PolyStep

# 2. Run the grid. 3 workers x 8 threads fills a 24-core box; ~16-24 GPU hours.
bash experiments/scripts/run_revision.sh

# 3. Check nothing failed silently. The runners catch per-run exceptions and exit 0,
#    so a job that lost 39 of 40 cells still logs "done".
python experiments/scripts/check_grid_complete.py experiments/results/revision

# 4. Regenerate every table from the JSONs.
python experiments/scripts/generate_paper_tables.py \
    --results-dir experiments/results/revision \
    --readme README.md --index experiments/EXPERIMENT_INDEX.md
```

Re-running a cell overwrites its JSON. Archive `experiments/results/revision/` before
a re-run if you want to diff against it afterwards.

The tuning sweep shortlists at a reduced budget; confirm the pick on validation at the
budget actually reported (`tune_gallery.py --budget-frac 1.0`). A config selected at a
small fraction of the budget does not always transfer.

## Individual experiments

```bash
python experiments/runners/run_elevation.py --showcases snn int8 argmax staircase \
    --seeds 42 123 456 789 1337
python experiments/runners/run_moe.py
python experiments/runners/run_mnist.py
python experiments/runners/run_maxsat.py                  # 100 -> 100K variables
python experiments/runners/run_timeseries.py
python experiments/runners/run_rl.py --mode full --env cartpole
python experiments/runners/run_gpt2_finetune.py
```

Every runner takes `--methods` and `--seeds`, so a single cell is one invocation.

## Result artifacts

Run outputs are not tracked in git. One JSON per `(benchmark, method, seed)` under
`experiments/results/revision/`, each carrying:

| Field | Contents |
|---|---|
| `benchmark` | which experiment produced it |
| `method` | `polystep`, `adam`, `cma_es`, `openai_es`, `eggroll`, `mezo`, `spsa`, `random_search` |
| `seed` | one of 42, 123, 456, 789, 1337 |
| `metrics` | accuracy, loss, convergence history |
| `epoch_logs` / `step_logs` | per-epoch and per-step traces |
| `hyperparameters` | full configuration, including the selected tuning point |
| `environment` | hardware and PyTorch version |
| `timestamp` | UTC, when the run finished |
| `leaked` | set when a run selected on test; `load_single_result` refuses to read it |

`run_maxsat.py` writes `cmaes` where every other runner writes `cma_es`. The readers
normalise it; do not rename it in the runners, which would split the existing files.

To summarise without regenerating tables:

```bash
python experiments/scripts/aggregate_results.py experiments/results/revision --benchmark snn
python experiments/scripts/aggregate_results.py experiments/results/revision --write summary
```

## Determinism

See [`determinism.md`](determinism.md) for what is pinned, what is not, and why a
seeded run is only comparable to itself on the same machine and build.
