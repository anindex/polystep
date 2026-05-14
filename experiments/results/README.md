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
  "method": "polystep",
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

## Regeneration

```bash
bash experiments/runners/run_all_paper.sh
```
