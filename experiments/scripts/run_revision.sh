#!/usr/bin/env bash
# Full re-run for the paper revision, under the honest protocol and matched budgets.
#
# Harness settings are measured, not guessed (docs/performance.md plus a sweep on
# this box, RTX 5090 / 24 cores):
#   POLYSTEP_THREADS=8   45s/epoch on the SNN showcase, against 50s at 4 and 58s
#                        at 22. The torch default of nproc is the worst available:
#                        the pool and the main thread oversubscribe and OpenMP
#                        spin-wait takes over.
#   OMP_WAIT_POLICY      set to PASSIVE in common.py at import, before torch
#                        initializes its pool.
#   3 workers            the step is GPU-bound at ~91% utilization and 2.3GB of
#                        32GB, so concurrency buys throughput: 3 runs finish in
#                        101s against 135s sequential (1.34x). 3 x 8 threads also
#                        exactly fills the 24 cores.
#
# Every job runs --fair (same subspace, same candidate-evaluation budget, same
# minibatch stream across gradient-free methods) and the default honest protocol
# (val-selected checkpoint, test scored once).
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
OUT=experiments/results/revision
LOG=experiments/results/revision/logs
mkdir -p "$OUT" "$LOG"

export POLYSTEP_THREADS=8
export OMP_WAIT_POLICY=PASSIVE

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
  fi
}
export -f run
export PY LOG

JOBS=()
# Non-differentiable showcases: the headline table.
for sc in snn int8 argmax staircase; do
  JOBS+=("nondiff_$sc|experiments/runners/run_elevation.py --showcases $sc --methods $GF adam --seeds $SEEDS --fair --results-dir $OUT")
done
JOBS+=("mnist|experiments/runners/run_mnist.py --methods $GF adam --seeds $SEEDS --fair --results-dir $OUT")
JOBS+=("moe|experiments/runners/run_moe.py --seeds $SEEDS --fair --results-dir $OUT")
JOBS+=("timeseries|experiments/runners/run_timeseries.py --seeds $SEEDS --fair --results-dir $OUT")
JOBS+=("maxsat|experiments/runners/run_maxsat.py --results-dir $OUT")
# Theory mode: the configuration Theorem 4.2 actually analyses.
JOBS+=("theory_snn|experiments/runners/run_elevation.py --showcases snn --methods polystep --seeds $SEEDS --theory-mode --fair --results-dir $OUT/theory")
JOBS+=("theory_mnist|experiments/runners/run_mnist.py --methods polystep --seeds $SEEDS --theory-mode --fair --results-dir $OUT/theory")
JOBS+=("theory_maxsat|experiments/runners/run_maxsat.py --theory-mode --results-dir $OUT/theory")

printf '%s\n' "${JOBS[@]}" | \
  xargs -P "$WORKERS" -I{} bash -c 'IFS="|" read -r tag cmd <<< "{}"; run "$tag" $cmd'

echo "[$(date +%H:%M:%S)] all jobs finished"
