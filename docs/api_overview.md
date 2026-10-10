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
updates, call `optimizer.resync_from_model()`. After replacing layers or buffers, or
casting the model to another dtype, call `evaluator.reset_vmap()` and register it
again. A swapped layer also needs `optimizer.resync_from_model()`, or the next step
overwrites the new layer with the old weights plus the update.

After a dtype cast, that resync moves the subspace projections to the new parameter
dtypes, and an `AdaptiveSubspace` moves its coordinates along with them.
`HybridSubspace`, `FactoredSubspace` and full space keep their coordinates in the dtype
they were built with, so build a new optimizer if those should follow the cast. No
optimizer follows the model to another device; the resync raises there.

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

`restore_best=True` restores the weights with the lowest tracked loss. Each step's
weights are scored on the batch that step trained on, with the same input cast,
autocast and loss reduction as the candidates, so the loss is in-sample and leans
optimistic. Best-weight tracking and callbacks require an extra loss evaluation per
step. This is training-loss selection, not held-out validation selection.

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

Factored evaluation avoids candidate weight construction for supported sequential
MLPs; other models use reconstructed candidates. Mixed-dtype models need a per-entry
subspace. See [performance](performance.md#subspaces-and-memory).

Sparse projections can also be constructed directly:

```python
from polystep import SparseRandomProjection

projection = SparseRandomProjection(full_dim=10_000, subspace_dim=64, seed=0)
```

### Rank schedules

`RankSchedule` changes the rank after absorbing the current perturbation. It needs a
`HybridSubspace`, and the constructor raises `ValueError` for any other subspace. It
works under any `block_strategy`:

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
| `block_strategy` | `'monolithic'` | Also `'per_layer'` or `'grouped'`; with a subspace, only splits Sinkhorn/KL solves into row segments |
| `chunk_size` | `None` | Automatic candidate chunk estimate |
| `adaptive_num_probe` | Multiple probes, except full-space blocks and orthoplex Newton refinement | Reduces probe count after decreasing costs |

Scalar radii are multiplied by epsilon. Scheduled radii supply their own scale:

```python
from polystep import CosineEpsilon, LinearEpsilon

schedule = CosineEpsilon(init=1.0, target=0.1, decay=0.01)
optimizer = PolyStepOptimizer(model, epsilon=schedule)
```

`ProgressiveEpsilon` adapts epsilon to how quickly Sinkhorn converges instead of
following the step count. It needs an iterative solver and cannot be combined with
`ent_epsilon`.

Solver choices are `sinkhorn` (full-space default), `softmax` (subspace default),
`kl_softmax`, `min_cost_greedy`, and `top_k_mean`. Sinkhorn constrains both
marginals; softmax constrains only the source marginal.

## Evaluation controls

| Option | Default | Scope |
|---|---|---|
| `compile` | `False` | CUDA geometry and solver kernels |
| `compile_evaluator` | `False` | Dense and site-local candidate vmap |
| `compile_forward` | `None` | Automatic compilation on the in-place forward path |
| `candidate_autocast` | `None` | Candidate forward precision |

Pass a new `objective_token` whenever the batch or loss changes. A change drops the
deferred trust-region comparison; `None` or a constant token means a stationary
objective.

Multi-fidelity screening needs a cheaper closure. `screen_closure_from` calls the
closure with the first `screen_fidelity` share of the batch as extra arguments, so
give the closure the batch as defaults and the same function serves both calls:

```python
def closure(params, x=inputs, y=targets):
    return evaluator.evaluate(params, x, y)

optimizer = PolyStepOptimizer(model, multifidelity_screen=True)
screen = optimizer.screen_closure_from(closure, inputs, targets)
optimizer.step(closure, screen_closure=screen)
```

A registered evaluator screens on the same leading slice of its batch.

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
It accepts neither `probe_radius` nor `num_probe`. With `seed` it draws rotations from
its own generator; without one it uses PyTorch's global RNG, so `torch.manual_seed`
reproduces the run.

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

Rebuild the optimizer with the original configuration before loading. This release
reads only format 6 and raises `ValueError` on any other format. To continue a run
saved by another release, build a fresh optimizer around the saved model weights.
The checkpoint also restores `candidate_evals`. Reproducible continuation requires
the same objective and environment.
