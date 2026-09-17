# API reference

## Optimizer

`PolyStepOptimizer` trains an `nn.Module` from candidate losses. The closure accepts
a dictionary of batched parameters and returns one loss per candidate:

```python
from torch import nn
from polystep import NNCostEvaluator, PolyStepOptimizer

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
optimizer = PolyStepOptimizer(model, epsilon=0.1, step_radius=0.15)
evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())

for x, y in loader:
    optimizer.register_evaluator(evaluator, x, y)
    optimizer.step(lambda p: evaluator.evaluate(p, x, y))
optimizer.release_evaluator()
```

The evaluator and closure must compute the same objective. After external parameter
updates, call `optimizer.resync_from_model()`. After replacing layers or buffers,
call `evaluator.reset_vmap()` and register it again.

## Training loop

`train()` builds the closure and registers each batch automatically:

```python
from polystep import train, TrainConfig, LoggingCallback, EarlyStoppingCallback

config = TrainConfig(
    epochs=10,
    callbacks=[LoggingCallback(log_every=10), EarlyStoppingCallback(patience=5)],
)
train(model, loader, nn.CrossEntropyLoss(), optimizer, config)
```

`restore_best=True` restores the weights with the lowest tracked minibatch loss.
Best-weight tracking and callbacks require an extra loss evaluation per step.
This is training-loss selection, not held-out validation selection.

## Subspaces

```python
from polystep import HybridSubspace, FactoredSubspace, AdaptiveSubspace
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=256)
optimizer = PolyStepOptimizer(model, subspace=subspace)
```

| Class | Representation |
|---|---|
| `HybridSubspace` | Per-parameter projections with an optional total dimension cap |
| `FactoredSubspace` | Matrix perturbations `dW = A @ B` with fixed `B` |
| `AdaptiveSubspace` | Global projection updated from displacement history |
| `CMAAdaptiveSubspace` | Adaptive subspace with diagonal covariance adaptation |

Factored evaluation avoids candidate weight construction for supported sequential
MLPs; other models use reconstructed candidates. Mixed-dtype models need a per-entry
subspace. See [performance](performance.md#subspaces-and-memory).

Sparse projections can also be constructed directly:

```python
from polystep import SparseRandomProjection

projection = SparseRandomProjection(full_dim=10_000, subspace_dim=64, seed=0)
```

### Rank schedules

`RankSchedule` changes the rank after absorbing the current perturbation. It requires
monolithic mode:

```python
from polystep.optimizer import RankSchedule

optimizer = PolyStepOptimizer(
    model,
    subspace=HybridSubspace.from_layout(layout, rank=2),
    rank_schedule=RankSchedule(stages=[(0, 2), (100, 4), (300, 8)]),
)
```

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `epsilon` | `0.1` | Entropic regularization and scalar-radius scaling |
| `step_radius` | `1.0` | Barycentric step radius, before momentum |
| `probe_radius` | `2.0` | Outer probe radius |
| `num_probe` | `1` | Probe count; probe `k` uses `k / (num_probe + 1)` of the radius |
| `polytope_type` | `'simplex'` | Also `'orthoplex'` or `'cube'` |
| `subspace` | `None` | A constructed subspace instance |
| `block_strategy` | `'monolithic'` | Also `'per_layer'` or `'grouped'` |
| `chunk_size` | `None` | Automatic candidate chunk estimate |
| `adaptive_probes` | Monolithic only | Reuses losses when state and objective are unchanged |
| `adaptive_num_probe` | Monolithic, multiple probes | Reduces probe count after decreasing costs |

Scalar radii are multiplied by epsilon. Scheduled radii supply their own scale:

```python
from polystep import CosineEpsilon, LinearEpsilon

schedule = CosineEpsilon(init=1.0, target=0.1, decay=0.01)
optimizer = PolyStepOptimizer(model, epsilon=schedule)
```

Solver choices are `sinkhorn` (full-space default), `softmax` (subspace default),
`kl_softmax`, `tempered_softmax`, `min_cost_greedy`, and `top_k_mean`. Sinkhorn
constrains both marginals; softmax constrains only the source marginal.

## Evaluation controls

| Option | Default | Scope |
|---|---|---|
| `compile` | `False` | CUDA geometry and solver kernels |
| `compile_evaluator` | `False` | Dense and site-local candidate vmap |
| `compile_forward` | `None` | Automatic compilation on the in-place forward path |
| `candidate_autocast` | `None` | Candidate forward precision |

Use `objective_token=0` for a stationary objective. Change the token whenever the
batch or loss changes. Probe reuse and deferred trust comparisons depend on it.

Multi-fidelity screening needs a cheaper closure:

```python
optimizer = PolyStepOptimizer(model, multifidelity_screen=True)
screen = optimizer.screen_closure_from(closure, inputs, targets)
optimizer.step(closure, screen_closure=screen)
```

Screening is disabled with quadratic-model options or when its estimated cost is
at least the full sweep. See [performance](performance.md#radius-models).

## Custom layers

Batched paths accept supported builtin layers and explicit custom contracts:

```python
class Staircase(nn.Module):
    polystep_elementwise = True

    def forward(self, x):
        return torch.floor(torch.sigmoid(x) * 5) / 5
```

Elementwise layers must have no parameters, buffers, or in-place writes, and must
return the same values for a slice as for the full tensor.

Linear-like layers can declare `polystep_weight_transform` and
`polystep_bias_transform` for elementwise parameter transforms. Their forward must
match `x @ Q(weight).T + Q(bias)`. Subspace-delta and factored paths do not support
these transforms and fall back to ordinary evaluation.

## Vmap-compatible layers

```python
from polystep.layers import VmapSafeMultiHeadAttention, VmapSafeLSTM

attention = VmapSafeMultiHeadAttention(embed_dim=256, num_heads=4)
lstm = VmapSafeLSTM(input_size=128, hidden_size=256, num_layers=2)
```

Use these when native layers lack vmap support. Both require batch-first inputs;
see [unsupported options](../LIMITATIONS.md#attention-and-recurrent-layers).

## Ask/tell

```python
from polystep import PolyStepES, minimize

es = PolyStepES(dim=20, epsilon=0.5, step_radius=0.5)
for _ in range(200):
    es.tell(objective(es.ask()))
best_x, best_loss = es.best_solution, es.best_fitness

es = minimize(objective, dim=20, steps=200)
```

`PolyStepES` defaults to `epsilon=0.5`, `step_radius=0.5`, and `scale_cost='mean'`.
It accepts neither `probe_radius` nor `num_probe`.

## Low-level solver

```python
from polystep import Ackley
from polystep.solver import PolyStep

solver = PolyStep.create(Ackley(dim=10), epsilon=0.5, max_iterations=100)
state = solver.run(X_init)
```

For manual steps, use `state = solver.init_state(X_init)` followed by
`state = solver.step(state)`. `SolverState` stores particles, costs, duals, and
convergence history. `get_diagnostics(optimizer)` reports evaluation counts and
transport statistics for the optimizer API.

## Checkpoints

Save model weights and optimizer state separately:

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint.pt")
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

Rebuild the optimizer with the original configuration before loading. Format 5
stores whole-model trust predictions. Older incumbent-based predictions are summed;
comparisons based on probe minima are discarded. Newer unsupported formats raise.
Reproducible continuation also requires the same objective and environment.
