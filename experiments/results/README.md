# Results

Pre-computed results for all paper experiments (5-seed, honest protocol).

## Layout

```
results/softmax/
├── main/          Primary experiments (SNN, INT8, argmax, staircase, MNIST, timeseries, MAX-SAT, MoE)
├── ablations/     Ablation studies (epsilon, radius, particles, compile, subspace, convergence, OT)
├── scalability/   Parameter scaling, sparse projection, memory
└── rl/            RL policy search (CartPole, Acrobot)
```

## Format

Each JSON file:
```json
{
  "method": "pstorch",
  "dataset": "mnist",
  "seed": 42,
  "config": { ... },
  "metrics": {
    "test_accuracy": 0.968,
    "train_loss_history": [...],
    "wall_time_seconds": 123.4
  }
}
```

The package was previously named `pstorch`. JSON files in this
directory record `"method": "pstorch"` for the main PolyStep results
to keep the on-disk layout stable.

## Regeneration

```bash
bash experiments/runners/run_all_paper.sh
```
