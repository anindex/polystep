# API Overview

polystep provides two levels of API for gradient-free neural network training.

## High-Level API

### PolyStepOptimizer

The main entry point. Wraps any `nn.Module` for gradient-free training.

```python
from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = PolyStepOptimizer(model,
    epsilon=0.1,
    step_radius=0.15,
    polytope_type='orthoplex',
)

# The closure receives batched candidate parameters and returns one loss
# per candidate. NNCostEvaluator handles the vmap'd forward pass for you.
evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())

def closure(batched_params):
    return evaluator.evaluate(batched_params, x, y)

cost = optimizer.step(closure)
```

### train()

Complete training loop with automatic closure construction.

```python
from polystep import train, TrainConfig, LoggingCallback, EarlyStoppingCallback

config = TrainConfig(
    epochs=10,
    callbacks=[
        LoggingCallback(log_every=10),
        EarlyStoppingCallback(patience=5, min_delta=1e-4),
    ],
)
model = train(model, train_loader, nn.CrossEntropyLoss(), optimizer, config)
```

### Epsilon Schedulers

```python
from polystep import CosineEpsilon, LinearEpsilon

# Cosine decay (recommended) - more exploration mid-training
schedule = CosineEpsilon(init=1.0, decay=0.01, target=1e-3)

# Linear decay
schedule = LinearEpsilon(init=1.0, decay=0.01, target=1e-3)
```

## Subspace Compression

For large models, subspace projection reduces the OT problem dimension.

### HybridSubspace (recommended)

Per-layer projections with coordinated rotations.

```python
from polystep import HybridSubspace
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = HybridSubspace.from_layout(layout, rank=4,
    rotation_interval=0,   # disable rotation for best accuracy
    absorb_interval=20,    # fold perturbation into base weights
)
optimizer = PolyStepOptimizer(model, subspace=subspace,
    epsilon=CosineEpsilon(init=1.0, target=0.1, decay=0.01),
    step_radius=4.5,
)
```

### AdaptiveSubspace

Global rotating orthogonal projection. Fastest wall-clock time, lower accuracy.

```python
from polystep import AdaptiveSubspace
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = AdaptiveSubspace.from_layout(layout, rank=64)
```

### LinearSubspace

Fixed random projection baseline.

```python
from polystep import LinearSubspace
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = LinearSubspace.from_layout(layout, rank=8)
```

### SparseRandomProjection

For models with 1M+ parameters. Uses a sparse Johnson-Lindenstrauss transform
under the hood and is typically created automatically when the optimizer is
constructed with `projection_type='sparse'` or `'auto'`. The constructor
signature is:

```python
from polystep import SparseRandomProjection

proj = SparseRandomProjection(full_dim=10_000_000, subspace_dim=64, seed=0)
```

## VmapSafe Layers

Standard `nn.MultiheadAttention` and `nn.LSTM` fail under `torch.vmap`. Use these drop-in replacements:

```python
from polystep.layers import VmapSafeMultiHeadAttention, VmapSafeLSTM

attention = VmapSafeMultiHeadAttention(embed_dim=256, num_heads=4)
lstm = VmapSafeLSTM(input_size=128, hidden_size=256, num_layers=2)
```

## Low-Level API

### PolyStep

For synthetic objectives or custom optimization loops.

```python
from polystep import PolyStep

solver = PolyStep.create(objective_fn,
    epsilon=0.5,
    max_iterations=100,
    polytope_type='orthoplex',
)

# Full run
state = solver.run(X_init)

# Or step-by-step
state = solver.init_state(X_init)
for i in range(100):
    state = solver.step(state)
```

### SolverState

Mutable dataclass tracking optimization state:
- `X`: current particle positions
- `costs`: loss values at current positions
- `f`, `g`: dual potentials for warm-starting Sinkhorn
- `displacement_history`: for convergence detection

## Synthetic Objectives

Built-in functions for testing:

```python
from polystep import Ackley, Rosenbrock, Rastrigin, Levy, Sphere
```

## Block-Wise OT

Per-layer decomposition reduces memory for models with many parameters.

```python
optimizer = PolyStepOptimizer(model,
    block_strategy='per_layer',  # decompose OT per parameter group
)
```

## Configuration Summary

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epsilon` | 0.1 | Use `CosineEpsilon` for scheduled decay |
| `step_radius` | 1.0 | Multiplied by current epsilon |
| `probe_radius` | 2.0 | Multiplied by current epsilon |
| `num_probe` | 1 | K=1 is optimal (entropic regularization provides smoothing) |
| `polytope_type` | `'orthoplex'` | `'orthoplex'`, `'simplex'`, `'cube'` |
| `compile` | False | Enable for GPU acceleration |
| `chunk_size` | None | Set to 512 for memory control |
| `subspace` | None | `HybridSubspace` recommended for large models |
| `block_strategy` | `'monolithic'` | `'per_layer'` for memory efficiency |
