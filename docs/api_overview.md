# API Overview

Four entry points: `PolyStepOptimizer`, the `train()` loop, the low-level `PolyStep`
solver, and the `PolyStepES` ask/tell interface.

## High-Level API

### PolyStepOptimizer

The main entry point. Wraps any `nn.Module` for gradient-free training.

```python
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = PolyStepOptimizer(model,
    epsilon=0.1,
    step_radius=0.15,
)

# The closure receives batched candidate parameters and returns one loss
# per candidate. NNCostEvaluator handles the vmap'd forward pass for you.
evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())

def closure(batched_params):
    return evaluator.evaluate(batched_params, x, y)

cost = optimizer.step(closure)
```

### train()

Complete training loop with automatic closure construction. Pass any
``torch.utils.data.DataLoader`` (or compatible iterable of ``(inputs, targets)`` batches)
as ``dataloader``.

```python
import torch.nn as nn
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

### HybridSubspace

Per-layer projections with coordinated rotations. The default choice for most workloads.

```python
from polystep import HybridSubspace
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = HybridSubspace.from_layout(layout, rank=4,
    rotation_interval=0,   # disable rotation for best accuracy
    absorb_mode="periodic",  # absorb_interval only applies in periodic mode
    absorb_interval=20,      # fold perturbation into base weights
)
optimizer = PolyStepOptimizer(model, subspace=subspace,
    epsilon=CosineEpsilon(init=1.0, target=0.1, decay=0.01),
    step_radius=4.5,
)
```

### FactoredSubspace

Perturbs each 2D parameter by `dW = A @ B` with `B` fixed, so a candidate's weights are
never built. 11.2x faster per step than `HybridSubspace` at matched search dimension;
the identity and the accuracy trade are in
[`docs/performance.md`](performance.md#trading-accuracy-for-speed-factoredsubspace).

Needs a plain `nn.Sequential` of Linear and activation layers; anything else falls back
to the materializing path with a warning. Pass the evaluator via `register_evaluator` so
the optimizer can build the fused path.

```python
from polystep import FactoredSubspace, NNCostEvaluator
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = FactoredSubspace.from_layout(layout, rank=32)
evaluator = NNCostEvaluator(model, loss_fn)
optimizer = PolyStepOptimizer(model, subspace=subspace)
optimizer.register_evaluator(evaluator, inputs, targets)
```

### AdaptiveSubspace

Global rotating orthogonal projection. Fastest wall-clock time, lower accuracy.

```python
from polystep import AdaptiveSubspace
from polystep.transform import ParamLayout

layout = ParamLayout.from_module(model)
subspace = AdaptiveSubspace.from_layout(layout, rank=64)
```

### SparseRandomProjection

For models with 1M+ parameters. Uses a sparse Johnson-Lindenstrauss transform internally
and is typically created automatically when the optimizer is constructed with
`projection_type='sparse'` or `'auto'`. The constructor signature is:

```python
from polystep import SparseRandomProjection

proj = SparseRandomProjection(full_dim=10_000_000, subspace_dim=64, seed=0)
```

## VmapSafe Layers

Standard `nn.MultiheadAttention` and `nn.LSTM` fail under `torch.vmap`. Use these drop-in
replacements:

```python
from polystep.layers import VmapSafeMultiHeadAttention, VmapSafeLSTM

attention = VmapSafeMultiHeadAttention(embed_dim=256, num_heads=4)
lstm = VmapSafeLSTM(input_size=128, hidden_size=256, num_layers=2)
```

## Custom Layers on the Batched Paths

The batched evaluators accept a fixed set of `torch.nn` layers. A custom layer opts in by
declaring the one thing the evaluators cannot infer:

```python
class Staircase(nn.Module):
    polystep_elementwise = True           # module(x)[i] == module(x[i])

    def forward(self, x):
        return torch.floor(torch.sigmoid(x) * 5) / 5


class BinaryLinear(nn.Module):
    polystep_weight_transform = staticmethod(torch.sign)   # forward is x @ Q(w).t() + b
    polystep_bias_transform = None

    def forward(self, x):
        return x @ torch.sign(self.weight).t() + self.bias
```

`polystep_elementwise` requires no parameters, no buffers, no in-place writes, and the
same answer for a slice as for the whole tensor; `try_build` probes that and declines a
module that fails, so a softmax or layer norm falls back rather than returning a
plausible wrong loss. Continuity is not required. `polystep_weight_transform` requires
`Q` elementwise; `SubspaceDeltaEvaluator` and `FactoredEvaluator` decline such a layer
because their correction is linear in the coordinate delta by construction. See
[`docs/performance.md`](performance.md) for the measured effect.

## Low-Level API

### PolyStep

For synthetic objectives or custom optimization loops.

```python
from polystep.solver import PolyStep

solver = PolyStep.create(objective_fn,
    epsilon=0.5,
    max_iterations=100,
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
from polystep import Ackley, Rastrigin, Rosenbrock, Sphere
```

## Diagnostics

```python
from polystep import get_diagnostics

get_diagnostics(optimizer)  # per-step ess, rho and evals
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
| `step_radius` | 1.0 | Multiplied by the current epsilon unless itself a schedule |
| `probe_radius` | 2.0 | Multiplied by the current epsilon unless itself a schedule. Probe `k` sits at `k/(num_probe+1)` of it, so the default single probe is at half |
| `num_probe` | 1 | K=1 is optimal: entropic regularization already does the averaging that larger K buys |
| `polytope_type` | `'simplex'` | `'orthoplex'`, `'simplex'`, `'cube'`. The simplex is the minimum positive spanning set, so a step costs `k+1` evaluations instead of `2k`. Only `newton_refinement` needs `'orthoplex'` |
| `compile` | False | Compiles the geometry and solver kernels. Off by default because ablations showed no end-to-end gain against the JIT warm-up cost; also a no-op on CPU. For forward-pass compilation see `compile_evaluator` / `compile_forward` |
| `chunk_size` | None | Estimated from the tensors a step allocates; set it only to override |
| `adaptive_probes` | True (monolithic) | Reuses the cached cost matrix while the configuration has not moved. Needs `objective_token` |
| `adaptive_num_probe` | monolithic and `num_probe > 1` | Drops to K=1 once the last OT-step costs are strictly decreasing |
| `subspace` | None | Use `HybridSubspace` for large models |
| `block_strategy` | `'monolithic'` | `'per_layer'` for memory efficiency |

## Ask/Tell Interface

For objectives that are not PyTorch models, drive the search directly.

```python
from polystep import PolyStepES, minimize

es = PolyStepES(dim=20, epsilon=0.5, step_radius=0.5)
for _ in range(200):
    candidates = es.ask()          # (num_candidates, dim)
    es.tell(objective(candidates))  # 1-D tensor of losses
best_x, best_f = es.best_solution, es.best_fitness

# One-call form: returns the finished PolyStepES
es = minimize(objective, dim=20, steps=200)
best_x, best_f = es.best_solution, es.best_fitness
```

`PolyStepES` has its own defaults (`epsilon=0.5`, `step_radius=0.5`, `scale_cost="mean"`)
and takes neither `probe_radius` nor `num_probe`.

## Checkpoint and Resume

`state_dict()` serializes the resumable optimizer state: particle positions, warm-start
duals, momentum velocity, adaptive radius, subspace projection and CMA state, rolling
histories, step counters, the `ProgressiveEpsilon` internals, and the RNG state. Model
weights are **not** included; save them separately.

```python
torch.save({"model": model.state_dict(), "opt": optimizer.state_dict()}, "ckpt.pt")

# Resuming: rebuild the optimizer with the SAME configuration, then load both.
ckpt = torch.load("ckpt.pt")
model.load_state_dict(ckpt["model"])
optimizer = PolyStepOptimizer(model, ...)  # same arguments as the original run
optimizer.load_state_dict(ckpt["opt"])
```

Resume is bit-exact for every configuration.

The dict carries a `format` key, currently 4. Loading a newer checkpoint raises. A
format-1 checkpoint loads with the reuse caches dropped; a format-2 one warns and resets
the rotation-bias and quadratic-model steering state. Format 4 adds the probe-jitter
pool and the CMA sampling-projection cache, so resume is exact under
`probe_radius_jitter` and `use_covariance_adaptation` too.

## Cutting the Forward Budget

Each step costs `num_particles * num_vertices * num_probe` forward passes.

Register the evaluator per batch, as `train()` does; without it the fast evaluators
never run:

```python
evaluator = NNCostEvaluator(model, loss_fn)
for inputs, targets in loader:
    optimizer.register_evaluator(evaluator, inputs, targets)
    optimizer.step(lambda p: evaluator.evaluate(p, inputs, targets))
optimizer.release_evaluator()  # drops the reference to the last batch
```

Two options then cut the count itself:

- `adaptive_probes=True` (default under `block_strategy='monolithic'`) reuses the whole
  cached cost matrix while the configuration has not moved. It is all or nothing: a
  candidate is the parameter vector with one particle row replaced, so every row of the
  matrix depends on every particle's position. Reuse also requires an `objective_token`,
  since that is the only assertion that the objective is stationary; pass
  `objective_token=0` for a full-batch run. `train()` passes a per-batch token, so on a
  minibatch objective nothing is reused.
- `multifidelity_screen=True` ranks every direction on a `screen_fidelity` slice of the
  batch, then spends the full fidelity only on the top `screen_keep_ratio` directions
  (both signs of each on an orthoplex). Dropped vertices keep their cheap value plus a
  per-particle offset calibrated on the kept ones. See
  [`docs/performance.md`](performance.md) for the budget arithmetic and the measured
  wall-clock crossover.

It needs a cheap closure. `api.train` builds one; when driving `step()` yourself:

```python
opt = PolyStepOptimizer(model, multifidelity_screen=True, screen_keep_ratio=0.5,
                        screen_fidelity=0.25)
screen = opt.screen_closure_from(closure, inputs, targets)  # None when screening is off
opt.step(closure, screen_closure=screen)
```

Screening is skipped, with a one-time warning, while `use_quadratic_model`,
`newton_refinement` or `trust_region` is on, or when
`screen_fidelity/num_probe + screen_keep_ratio >= 1`.

## Restoring the Best Weights

`TrainConfig.restore_best` defaults to `True`. Gradient-free search updates the model in
place and the last step is not necessarily the best, so `train()` snapshots the weights
whenever the tracked loss reaches a new minimum and restores that snapshot before
returning. The tracked loss is the exact minibatch loss, one extra forward per step,
which `restore_best` and any callback both require.

## Forward-Pass Compilation

Separate from `compile` (which covers the geometry and solver kernels):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `compile_evaluator` | False | Compiles the batched `vmap` candidate evaluation |
| `compile_forward` | None | Compiles the single-candidate forward used by the in-place path, with CUDA graph replay on GPU. `None` enables it wherever that path runs; pass `False` to opt out |

See [performance.md](performance.md) for which backend is chosen when.

## Progressive Rank

`RankSchedule` grows the subspace rank at chosen steps, triggering an absorb plus
subspace reconstruction. Monolithic mode only.

```python
from polystep import PolyStepOptimizer
from polystep.optimizer import RankSchedule

optimizer = PolyStepOptimizer(model,
    subspace=HybridSubspace.from_layout(layout, rank=2),
    rank_schedule=RankSchedule(stages=[(0, 2), (100, 4), (300, 8)]),
)
```

## Solver Choices

`solver` accepts `'sinkhorn'`, `'softmax'` (default in subspace mode), `'kl_softmax'`,
`'tempered_softmax'`, `'min_cost_greedy'` and `'top_k_mean'`. `SinkhornSolver` enforces
both marginals; `SoftmaxSolver` enforces only the source marginal and is the cheaper
choice when the target constraint is not needed. `KLSoftmaxSolver` interpolates between
the two via a KL penalty on the target marginal.
