#!/usr/bin/env bash
# Full run of the benchmark grid under the honest protocol and matched budgets.
#
# Harness settings are measured, not guessed (docs/performance.md):
#   POLYSTEP_THREADS=8   fastest on this box; the torch default of nproc
#                        oversubscribes the pool against the main thread and
#                        OpenMP spin-wait takes over.
#   OMP_WAIT_POLICY      set to PASSIVE in common.py at import, before torch
#                        initializes its pool.
#   3 workers            the step is GPU-bound at low memory, so concurrency
#                        buys throughput; 3 x 8 threads fills the 24 cores.
#
# Every job runs --fair (same subspace, same candidate-evaluation budget, same
# minibatch stream across gradient-free methods) and the default honest protocol
# (val-selected checkpoint, test scored once).
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY="${PY:-.venv/bin/python}"
OUT=experiments/results/revision
LOG=experiments/results/revision/logs
mkdir -p "$OUT" "$LOG"

# Every runner skips a cell whose result file already exists, so re-running
# into a non-empty directory would silently measure only the missing cells and
# the tables would mix old and new runs.  Refuse, and say exactly what to do
# about it.
stale=$(find "$OUT" -maxdepth 2 -name '*.json' 2>/dev/null | wc -l)
if [ "$stale" -gt 0 ] && [ "${ALLOW_STALE:-0}" != "1" ]; then
  echo "refusing to launch: $OUT already holds $stale result file(s)." >&2
  echo "archive them first:" >&2
  echo "  mv $OUT $OUT.\$(date +%F-%H%M)" >&2
  echo "(or set ALLOW_STALE=1 to run anyway; every job below passes --force, so" >&2
  echo " existing cells will be overwritten rather than skipped)" >&2
  exit 1
fi

export POLYSTEP_THREADS="${POLYSTEP_THREADS:-8}"
export OMP_WAIT_POLICY=PASSIVE
# Required, not cosmetic: without it a job log stays at 0 bytes for the life of the
# run, so a hung job is indistinguishable from a slow one.
export PYTHONUNBUFFERED=1

WORKERS=3
SEEDS="42 123 456 789 1337"
GF="polystep openai_es spsa cma_es mezo random_search eggroll"

run() {   # run <tag> <args...>
  local tag=$1; shift
  echo "[$(date +%H:%M:%S)] start  $tag"
  if "$PY" "$@" > "$LOG/$tag.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] done   $tag"
  else
    echo "[$(date +%H:%M:%S)] FAILED $tag (see $LOG/$tag.log)"
    echo "$tag" >> "$LOG/failed"
  fi
}
export -f run
export PY LOG
rm -f "$LOG/failed"

JOBS=()
# Non-differentiable showcases: the headline table.
JOBS+=("nondiff_snn|experiments/runners/run_elevation.py --showcases snn --methods $GF adam --seeds $SEEDS --fair --force --results-dir $OUT")
JOBS+=("mnist|experiments/runners/run_mnist.py --methods $GF adam --seeds $SEEDS --fair --force --results-dir $OUT")
for sc in int8 argmax staircase; do
  JOBS+=("nondiff_$sc|experiments/runners/run_elevation.py --showcases $sc --methods $GF adam --seeds $SEEDS --fair --force --results-dir $OUT")
done
JOBS+=("moe|experiments/runners/run_moe.py --seeds $SEEDS --fair --force --results-dir $OUT")
JOBS+=("timeseries|experiments/runners/run_timeseries.py --seeds $SEEDS --fair --force --results-dir $OUT")
JOBS+=("maxsat|experiments/runners/run_maxsat.py --force --results-dir $OUT")
# RL: the 2 envs x 3 precisions grid at the standard seed set.
for env in cartpole acrobot; do
  for prec in float32 int8 binary; do
    JOBS+=("rl_${env}_${prec}|experiments/runners/run_rl.py --env $env --nondiff-mode $prec --methods polystep es ppo dqn --seeds $SEEDS --results-dir $OUT")
  done
done
# The unaccelerated reference configuration, reported separately.
JOBS+=("theory_snn|experiments/runners/run_elevation.py --showcases snn --methods polystep --seeds $SEEDS --theory-mode --fair --force --results-dir $OUT/theory")
JOBS+=("theory_mnist|experiments/runners/run_mnist.py --methods polystep --seeds $SEEDS --theory-mode --fair --force --results-dir $OUT/theory")
JOBS+=("theory_maxsat|experiments/runners/run_maxsat.py --theory-mode --force --results-dir $OUT/theory")

printf '%s\n' "${JOBS[@]}" | \
  xargs -P "$WORKERS" -I{} bash -c 'IFS="|" read -r tag cmd <<< "{}"; run "$tag" $cmd'

echo "[$(date +%H:%M:%S)] all jobs finished"

rc=0
if [ -s "$LOG/failed" ]; then
  echo "FAILED jobs:" >&2
  sed 's/^/  /' "$LOG/failed" >&2
  rc=1
fi

# The completeness gate: a job can exit 0 having produced nothing, so check the grid
# rather than the exit codes.  Counts only files directly in $OUT, so the theory-mode
# and tuning runs under $OUT/theory do not stand in for headline cells.
"$PY" experiments/scripts/check_grid_complete.py "$OUT" --require fair,tuned || rc=1

exit "$rc"
