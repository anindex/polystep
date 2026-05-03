# BENCHMARKS.md

> **Note:** Dated snapshot below; numbers are not continuously revalidated against every commit.

Smoke-scale benchmark snapshot. Produced by benchmark scripts.

- Device: `cpu`
- Seeds: `[42, 123]`
- Generated: `2026-04-19 19:53:28`

These are smoke-scale benchmarks - their purpose is to confirm that
changes do not regress core optimizer behavior, NOT to reproduce
the headline numbers in [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md).
Full headline reproduction (5 seeds, full MNIST 3 epochs etc.) lives
under `experiments/runners/run_*.py` and is left as future GPU work.

## Results

### Rosenbrock d=100

- Seeds: 2
- elapsed_s: 23.6699 +/- 18.4199
- final_cost: 116.1597 +/- 0.0887

### Ackley d=50

- Seeds: 2
- elapsed_s: 8.0899 +/- 7.4575
- final_cost: -2.2370 +/- 0.0223

### MNIST-MLP smoke

- Seeds: 2
- best_loss: 2.2330 +/- 0.0026
- elapsed_s: 26.4851 +/- 0.3320
- final_loss: 2.2330 +/- 0.0026

### MAX-SAT 1K

- Seeds: 2
- elapsed_s: 7.5536 +/- 0.0218
- num_clauses: 4270.0000 +/- 0.0000
- num_satisfied: 3964.0000 +/- 26.8701
- sat_ratio: 0.9283 +/- 0.0063

## Notes

- Rosenbrock / Ackley convergence on small dimensions is the primary smoke test for the OT update path.
- The MNIST-MLP smoke uses random Gaussian inputs (no torchvision dependency); it measures step latency, not accuracy.
- MAX-SAT 1K satisfied fraction should exceed ~0.9 within 200 steps when python-sat is available.

For full headline reproduction, see [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) and `experiments/runners/run_all_paper.sh`.