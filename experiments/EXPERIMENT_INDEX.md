# Experiment results

Architecture results use seeds 42, 123, 456, 789, and 1337, with validation-selected
configurations and checkpoints. Tables compare optimizer steps. Evaluation- and
time-matched results are reported separately.

<!-- BENCH:START -->
### Non-differentiable tasks

Test accuracy %. A dash means the result is not in this release.

| Task | PolyStep | OpenAI-ES | CMA-ES | EGGROLL | MeZO | SPSA | Random search | Adam (surrogate) | Non-diff op |
|---|---|---|---|---|---|---|---|---|---|
| SNN/LIF (MNIST) | 93.0 ± 0.2 | 79.6 ± 5.2 | 77.1 ± 13.3 | 55.3 ± 8.3 | 55.4 ± 13.3 | 53.9 ± 3.0 | 16.4 ± 4.4 | 86.3 ± 10.6 | `threshold()` |
| Int8 quantized | 97.0 ± 0.1 | 94.4 ± 0.2 | 91.6 ± 0.4 | 83.4 ± 1.2 | 80.7 ± 0.9 | 82.5 ± 0.5 | 29.2 ± 4.9 | 97.8 ± 0.1 | `round()` |
| Argmax attention | 86.6 ± 0.4 | 80.0 ± 0.4 | 79.5 ± 0.5 | 71.0 ± 0.4 | 71.6 ± 1.2 | 67.4 ± 1.3 | 42.8 ± 1.9 | 88.6 ± 0.1 | `argmax()` |
| Staircase activation | 94.3 ± 0.1 | 88.9 ± 0.5 | 89.2 ± 0.4 | 52.0 ± 1.5 | 61.1 ± 2.8 | 45.2 ± 4.2 | 18.1 ± 4.8 | 97.5 ± 0.1 | `floor()` |
| Hard MoE routing | 90.6 ± 0.2 | 82.5 ± 0.8 | 77.8 ± 1.5 | 52.5 ± 1.5 | 55.3 ± 3.0 | 25.4 ± 3.5 | 13.7 ± 1.9 | - | `argmax()` |

### MAX-SAT (% clauses satisfied)

| Variables | PolyStep | CMA-ES | OpenAI-ES | probSAT | RC2 |
|---|---|---|---|---|---|
| 100 | 98.0 ± 0.7 | 99.0 ± 0.3 | 99.4 ± 0.1 | 99.8 | 99.8 |
| 500 | 98.1 ± 0.2 | 98.4 ± 0.2 | 98.1 ± 0.2 | 100.0 | timeout |
| 1,000 | 98.1 ± 0.2 | 97.3 ± 0.1 | 96.9 ± 0.2 | 100.0 | timeout |
| 5,000 | 98.1 ± 0.1 | 95.2 ± 0.2 | 92.9 ± 0.2 | 99.9 | timeout |
| 20,000 | 98.2 ± 0.0 | 92.9 ± 0.2 | 90.5 ± 0.1 | 99.8 | timeout |
| 100,000 | 98.1 ± 0.0 | 90.7 ± 0.1 | 88.9 ± 0.0 | 99.6 | timeout |

### Differentiable sanity checks

| Task | PolyStep | Adam |
|---|---|---|
| MNIST (2-layer MLP) | 96.8 ± 0.1 | 97.7 ± 0.1 |
| ETTh1 (LSTM, MSE; lower is better) | 0.253 ± 0.023 | 0.247 ± 0.023 |
<!-- BENCH:END -->

Adam uses surrogate gradients where necessary and is a gradient-based reference.
Specialized MAX-SAT solvers use different work units; their flip budgets do not equal
neural candidate budgets.

## Reproduction

See the [runner list](README.md#runners), [protocol](../docs/reproducibility.md), and
[controlled hard-LIF comparison](CONTROLLED_EXPERIMENTS.md).

Regenerate these tables and the README from result files:

```bash
python experiments/scripts/generate_tables.py \
    --results-dir experiments/results/revision \
    --readme README.md --index experiments/EXPERIMENT_INDEX.md
```

Architecture results live in `results/revision/`, with `theory/`, `evalmatched/`, and
`wallclock/` subdirectories for separate protocols. Outputs are not tracked in Git.
