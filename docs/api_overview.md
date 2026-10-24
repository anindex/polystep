# API reference

Start with the [optimizer](#optimizer) and [training loop](#training-loop), then choose
a [subspace](#subspaces) and [configuration](#configuration). For custom models, see
[evaluation controls](#evaluation-controls) and [custom layers](#custom-layers).
[Ask/tell](#asktell), [checkpoints](#checkpoints) and the
[upgrade notes](#upgrading-from-10) cover the other workflows.

The Python snippets build on the imports, model and data in the first example.

## Optimizer

`PolyStepOptimizer` trains an `nn.Module` from candidate losses. Pass the module
itself; the optimizer reads its trainable parameters once, when it is built.
`step_batch` scores the candidates on one batch and moves the weights:

```python
import torch
from torch import nn
from polystep import PolyStepOptimizer

torch.set_num_threads(1)
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
loader = [(torch.randn(32, 16), torch.randint(0, 3, (32,))) for _ in range(8)]
optimizer = PolyStepOptimizer(model, epsilon=0.1, step_radius=0.15)
loss_fn = nn.CrossEntropyLoss()

for x, y in loader:
    optimizer.step_batch(x, y, loss_fn=loss_fn)
optimizer.release_evaluator()
```

`step_batch` creates an `NNCostEvaluator` for `loss_fn` and reuses it while you pass
the same loss object. Create the loss once; pass a new object when its settings
change. The [supported builtin losses](performance.md#candidate-evaluation) enable
the batched-matmul and delta paths. Other callables use the site-local or vmap path.

Each optimal-transport (OT) step scores the candidates, updates the model weights
and returns the mean candidate loss. With `cost_batch_size` set, it uses a random
subset of that many batch rows. Pass `targets=None` for a loss that reads only the
output. Under `amortize_steps`, calls between OT steps move the weights along the
last direction without scoring candidates and return the previous OT step's value.

The batch stays registered until the next call, `release_evaluator()` or the end of
`train()`. Call `release_evaluator()` once after a manual loop to free the batch and
cached evaluation resources.

`NNCostEvaluator` puts the model in eval mode when it is constructed and scores every
candidate in eval mode, so dropout stays off and batch-norm running statistics do not
update.

Forward-written buffers and embedding weights renormalized by `max_norm` are restored
after scoring, including between evaluator chunks and when evaluation raises.
Frozen embedding weights are protected too. Writes through `.data` and other
unversioned writes have [additional limits](../LIMITATIONS.md#evaluation).

At the default settings, a full-space step scores about 1.5 candidates per trainable
parameter, and the constructor warns when a step would score more than 100,000. Pass
a [subspace](#subspaces) for larger models.

### Inputs

`inputs` can be a tensor, a tuple or list of positional arguments, or a dict of
keyword arguments for the model's forward. `(ids, mask)` calls `model(ids, mask)`, and
`{"input_ids": ids, "attention_mask": mask}` calls
`model(input_ids=ids, attention_mask=mask)`. To pass a dict or tuple as one positional
argument, wrap it in a 1-tuple: `(batch,)`.

```python
class Masked(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 3)

    def forward(self, x, mask):
        return self.linear(x) * mask

masked = Masked()
masked_optimizer = PolyStepOptimizer(masked)
x, y = loader[0]
mask = torch.ones(32, 3)
masked_optimizer.step_batch((x, mask), y, loss_fn=loss_fn)
masked_optimizer.step_batch({"x": x, "mask": mask}, y, loss_fn=loss_fn)
masked_optimizer.release_evaluator()
```

The step splits the candidates into chunks, never the batch. `cost_batch_size`
subsampling indexes only the input and target tensors whose leading dimension matches
the first batched input's, and passes scalars and other tensors through unchanged.
Multi-input batches use vmap and the site-local path. The batched-matmul, delta and
factored paths require a single input tensor, as does `multifidelity_screen`.

### Custom closures

`step(closure)` calls `closure(batched_params)` with a dictionary of parameters that
hold one candidate per row and expects one loss per candidate; `step_batch` builds
this closure for you. To set evaluator options such as `chunk_size`, `use_inplace` or
`compile_vmap`, build the evaluator yourself, register every batch and step with the
closure `register_evaluator` returns:

```python
from polystep import NNCostEvaluator

evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss(), use_inplace=True)
for x, y in loader:
    optimizer.step(optimizer.register_evaluator(evaluator, x, y))
optimizer.release_evaluator()
```

Without a registration the delta, factored and site-local paths stay off, and every
candidate goes through the closure. The registered loss and any other closure you pass
must compute the same objective, regularization included. `step()` warns once if you
have not registered a batch since the previous step and pass a closure other than
the one returned by `register_evaluator`. The manual step API also supports
`trust_region`, through `objective_token`; see
[radius control](#quadratic-model-and-radius-control).
`step_batch` replaces any registration.

The evaluator must wrap the same module as the optimizer. Registering a different
model raises `ValueError` and leaves the previous registration intact.

### After changing the model

After writing weights into the model yourself, or swapping in a layer under the same
parameter names, call `optimizer.resync_from_model()`. Otherwise, the next step
overwrites the change with the old weights plus the update. Resync also rebuilds the
evaluator that `step_batch` caches.

After you replace layers, parameters or buffers, or cast the model to another dtype,
call `evaluator.reset_vmap()` on an evaluator you built yourself and register it
again. The evaluator also resets itself when a cast changes its buffers' dtypes.
Assigning `evaluator.loss_fn` rebuilds everything that depends on the loss. After
changing a loss's settings in place, such as `label_smoothing`, call `reset_vmap()`;
with `step_batch`, pass a new loss object instead.

The optimizer keeps the set of `requires_grad` parameters it was built with, and `step()`
raises `ValueError` if that set changes. To freeze, unfreeze, add or remove a parameter,
set `requires_grad` first and then build a new optimizer.

After a dtype cast, resync moves the subspace projections to the new parameter dtypes.
Full space and an `AdaptiveSubspace` also move their coordinates to the new dtype.
`HybridSubspace` and `FactoredSubspace` keep their original coordinate dtype.
If the coordinates must use the new dtype, build a new optimizer.
If the model moves to another device, build a new optimizer. Resync cannot change devices.

The optimizer, `NNCostEvaluator`, `ParamLayout` and `train()` unwrap a `torch.compile`
module and work on the module inside it, so parameter names carry no `_orig_mod.` prefix.
To compile candidate scoring, pass `compile_evaluator` or `compile_forward` instead.

### Coming from torch.optim

- Pass the module, not `model.parameters()`; anything else raises `TypeError`.
- There is no `zero_grad()` or `backward()`: a step scores candidates with forwards only.
- `epsilon` acts as the learning rate, since it scales the scalar step and probe radii.
  `LinearEpsilon` and `CosineEpsilon` replace an `lr_scheduler`, which does not accept
  this optimizer.
- There are no `param_groups`. Freeze parameters with `requires_grad_(False)` before
  building the optimizer.
- `max_iterations` only sets the length of the warm-up ramps; steps continue past it.
- `step(closure)` passes batched parameters and expects one loss per candidate;
  `step_batch` builds that closure for you.

## Training loop

`train()` puts the model in eval mode and calls `step_batch` for every batch the
loader yields:

```python
from polystep import train, TrainConfig, LoggingCallback, EarlyStoppingCallback

config = TrainConfig(
    epochs=10,
    callbacks=[LoggingCallback(log_every=10), EarlyStoppingCallback(patience=5)],
)
train(model, loader, nn.CrossEntropyLoss(), optimizer, config)
```

A tensor, or a one-element tuple, is the input alone. A longer tuple or list ends with
the target and passes everything before it to the model as positional arguments, so
`(x, y)` and `(ids, mask, y)` both work. A dict passes its entries to the model as
keyword arguments, except `labels`, which becomes the target; a dict without `labels`
trains unsupervised. Other batch types raise `TypeError`. When `train()` returns, the
model is back in the mode it had before.

`restore_best` (on by default) restores the weights with the lowest tracked loss.
After each step, `train()` scores the updated weights on the first training batch,
after `cost_batch_size` subsampling, so scores compare across steps. Callbacks receive
the loss of the updated weights on that step's training batch. Both losses use the
same input cast, autocast and loss reduction as the candidates. Checkpoint selection
uses training loss, which can be optimistic on unseen data. Best-weight tracking and
callbacks each require an extra loss evaluation per step. When the best weights are
the last ones, `train()` leaves the optimizer state alone, so calling `train()` again
continues the same run.

## Subspaces

```python
from polystep import HybridSubspace, FactoredSubspace, AdaptiveSubspace, ParamLayout

layout = ParamLayout.from_module(model)
subspace = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=256)
optimizer = PolyStepOptimizer(model, subspace=subspace)
```

| Class | Representation |
|---|---|
| `HybridSubspace` | Per-parameter projections with an optional total dimension cap |
| `FactoredSubspace` | Matrix perturbations `dW = A @ B`, with `B` redrawn every 5 OT steps by default |
| `AdaptiveSubspace` | Global projection updated from displacement history |

Factored evaluation avoids candidate weight construction for supported sequential
MLPs; other models use reconstructed candidates. Mixed-dtype models need a per-entry
subspace. See [performance](performance.md#subspaces-and-memory).

`AdaptiveSubspace` uses a dense projection unless the optimizer asks for a sparse
one. `projection_type='sparse'` switches it to sparse signed columns, and `'auto'`
picks sparse for models with at least 1,000,000 parameters on the CPU or 2,000,000
on CUDA. Models under 10,000 parameters keep the dense projection either way:

```python
optimizer = PolyStepOptimizer(
    model,
    subspace=AdaptiveSubspace.from_layout(layout, rank=128),
    projection_type="auto",
)
```

`projection_type` applies only to `AdaptiveSubspace`. `HybridSubspace` gives sparse
columns to any layer with `num_params * num_coords * 4 > sparse_threshold_bytes` (1 GB
by default). The estimate counts four bytes per entry whatever the parameter dtype, so
an FP64 layer's dense projection can reach twice the threshold.

### Rank schedules

`RankSchedule` changes the rank after absorbing the current perturbation. It needs a
`HybridSubspace`, and the constructor raises `ValueError` for any other subspace. It
works under any `block_strategy`:

```python
from polystep import RankSchedule

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
| `particle_dim` | `2` | Coordinates per particle in full space |
| `subspace_particle_dim` | `8` | Coordinates per particle with a subspace |
| `subspace` | `None` | A constructed subspace instance |
| `block_strategy` | `'monolithic'` | Also `'per_layer'` or `'grouped'`; with a subspace, only splits Sinkhorn/KL solves into row segments |
| `chunk_size` | `None` | Automatic candidate chunk estimate |
| `max_iterations` | `50` | Length of the momentum and svd_ratio warm-up ramps; steps continue past it |
| `adaptive_num_probe` | `None` (automatic) | Drops the probe count to 1 while costs keep falling |

By default, `adaptive_num_probe` is enabled when `num_probe > 1`, except with
full-space `per_layer`/`grouped` blocks or orthoplex `newton_refinement`. Set it
explicitly to keep or disable this behavior.

A step scores one candidate per particle, vertex and probe. With `d` coordinates per
particle the simplex has `d + 1` vertices, the orthoplex `2d` and the cube `2^d`; see
[performance](performance.md) for a worked count.

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

## Quadratic model and radius control

The quadratic model (`use_quadratic_model=True`) uses a centered unit-radius tight
frame with `V` vertices `v` in dimension `d`. Probe `k` of `K` places the vertices at
distance `s r` from the incumbent `X`, where `r` is the probe radius after epsilon
scaling and `s = k / (K + 1)`. With `L_v` the loss at vertex `v` on that probe and
`f(X)` the incumbent's loss:

```text
g       = d / (V s r) * sum_v (L_v - L_reference) v
tr(H)/d = 2 * (mean_v L_v - f(X)) / (s r)^2
```

Both estimates are averaged over the `K` probes. `L_reference` can be any loss shared
by every vertex, because `sum_v v = 0`. The generic path subtracts the first vertex's
loss so that a large constant objective does not round into a gradient. The orthoplex
path takes the central difference `L(+e_i) - L(-e_i)` for each coordinate instead.

The curvature identity is exact for quadratic objectives. Simplex gradient fits can
have an `O(r)` bias on anisotropic quadratics, which comes from the second-order term
`(s r)^2 / 2 * v^T H v`. Centrally symmetric templates (orthoplex, cube) remove it:
`v^T H v` is the same at `v` and `-v`, so its contributions cancel in pairs. Their
remaining gradient bias is `O(r^2)` for smooth objectives and zero on a quadratic.
Block models omit cross-block Hessian terms.

Denominators use the loss dtype's smallest normal value as a numerical floor.
Small radii still need resolvable loss differences; a fit cannot recover information
that the forward pass rounded away.

`trust_region` compares consecutive incumbents and sums block predictions. Minibatch
changes and momentum-only steps invalidate the deferred comparison, so `step_batch()`
and `train()` reject this controller. Use the manual step API with a stationary
objective. If the batch or loss changes, pass a new `objective_token` to
`optimizer.step(closure, objective_token=token)` to discard the pending comparison.
A constant token, including `None`, declares that the objective has not changed.

## Evaluation controls

| Option | Default | Scope |
|---|---|---|
| `compile` | `False` | CUDA geometry and solver kernels |
| `compile_evaluator` | `False` | Dense and site-local candidate vmap |
| `compile_forward` | `None` | CUDA graphs for the in-place forward and the CUDA site-local vmap |
| `candidate_autocast` | `False` | Candidate forward precision; disables incompatible fast paths |

`amortize_steps` skips evaluations between sampled updates. `multifidelity_screen`
scores candidates on a smaller batch, then re-scores the retained vertices on the
full batch. It keeps approximately `V * screen_keep_ratio` vertices per particle.
`top_k_mean` keeps at least `min(k, V)`; softmax, Sinkhorn and KL screening keep whole
antithetic pairs on the orthoplex. Compare either option at equal evaluation and time
budgets.

Screening needs a cheaper closure. `screen_closure_from` passes the first
`screen_fidelity` fraction of the batch as extra arguments to the closure. Give those
arguments defaults so the same closure can score both the full and reduced batch:

```python
inputs, targets = loader[0]

def closure(params, x=inputs, y=targets):
    return evaluator.evaluate(params, x, y)

optimizer = PolyStepOptimizer(model, multifidelity_screen=True)
screen = optimizer.screen_closure_from(closure, inputs, targets)
optimizer.step(closure, screen_closure=screen)
```

A registered evaluator screens on the same leading slice of its batch. Screening is
disabled with `use_quadratic_model`, `newton_refinement` or `trust_region`. Otherwise,
it runs only when `screened_batch_fraction / effective_num_probe + kept_vertex_fraction < 1`,
so the two sweeps need fewer sample forwards than a full sweep. Even then, two sweeps
can take longer on small models or GPU steps dominated by kernel launches.

Screened candidates use the same delta and site-local paths as the full sweep once
particles times vertices times model parameters reaches
5e5 on the CPU or 5e8 on CUDA. Below that, screened hybrid candidates build every
weight, and a screened full-space chunk uses a site-local path only when it lies
inside one parameter. The CPU threshold is the measured crossover. The CUDA one is
set conservatively to avoid regressions on the measured models, at the cost of some
speedups on small MLPs.

Small batches and polytopes make the requested fractions coarse. At the default
`particle_dim=2`, the simplex has 3 vertices, so `screen_keep_ratio=0.5` keeps 2.
A registered batch of 2 samples screens on 1 even at `screen_fidelity=0.1`.

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

objective = lambda x: (x - 1).square().sum(dim=1)
es = PolyStepES(dim=20, epsilon=0.5, step_radius=0.5)
for _ in range(200):
    es.tell(objective(es.ask()))
best_x, best_loss = es.best_solution, es.best_fitness

es = minimize(objective, dim=20, steps=200)
```

`PolyStepES` defaults to `epsilon=0.5`, `step_radius=0.5`, and `scale_cost='mean'`.
Its `step_radius` is a distance in solution space; unlike the optimizer's scalar
radii, it is not multiplied by epsilon. It accepts neither `probe_radius` nor
`num_probe`. With `seed` it draws rotations from its own generator; without one it
uses PyTorch's global RNG, so `torch.manual_seed` reproduces the run.
`tell()` averages the candidates as `ask()` returned them, since both share memory.
An in-place edit such as `candidates.clamp_(lower, upper)` therefore changes the
points `tell()` averages and the best solution it records.

## Low-level solver

```python
import torch
from polystep import Ackley
from polystep.solver import PolyStep

X_init = torch.randn(100, 10)
solver = PolyStep.create(Ackley(dim=10), epsilon=0.5, max_iterations=100)
state = solver.run(X_init)
```

For manual steps, use `state = solver.init_state(X_init)` followed by
`state = solver.step(state)`. `SolverState` stores particles, costs, duals, and
convergence history. `get_diagnostics(optimizer)` reports evaluation counts and
transport statistics for the optimizer API.

## Checkpoints

Save model weights and optimizer state in separate checkpoint entries:

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint.pt")
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

Rebuild the optimizer with the original configuration before loading.
`load_state_dict` reads only checkpoint format 6, written since 1.0, and raises
`ValueError` on any other format. For an older format, build a fresh optimizer around
the saved model weights.

The checkpoint also restores `candidate_evals`. Reproducible continuation requires
the same objective and environment. Checkpoints can move between CPU and CUDA.
When a seeded checkpoint crosses devices, loading keeps the new optimizer's
generator and warns because the two generator types cannot share state.

## Upgrading from 1.0

Version 1.1 adds errors and warnings and changes several defaults, so seeded runs can
differ from 1.0.
[Reproducibility](reproducibility.md#changes-since-the-published-tables) lists the
published results that change.

- A buffer or a separate Parameter that shares memory with a trainable parameter
  raises `ValueError` when the optimizer, an evaluator or a layout is built. Tie
  weights by assigning the Parameter itself, or register a clone of the buffer.
- `step()` raises `ValueError` when the set of `requires_grad` parameters changed after
  construction; build a new optimizer instead. Passing `model.parameters()` instead of
  the module raises `TypeError`.
- `FactoredSubspace` redraws `B` every 5 OT steps by default, which also clears
  momentum and the coasting and `biased_rotation` directions. Set
  `rotation_interval=0` for the 1.0 behavior.
- `HybridSubspace` and `FactoredSubspace` hash layer seeds with CRC-32 instead of
  Adler-32, and one built without a `seed` takes the optimizer's seed instead of 0, so
  both draw new bases. The optimizer keeps that seeded copy in `optimizer.subspace`.
- `SinkhornSolver` and `KLSoftmaxSolver` default `threshold` to 3e-5 instead of 1e-6.
  Sinkhorn compares it with the relative marginal error `max|row/a - 1|`, the KL solver
  with the dual step over epsilon. A solve whose error stalls at its dtype's precision
  stops early and reports `converged=False`.
- `train()` runs every batch through `step_batch` and accepts the batch formats listed
  under [training loop](#training-loop). `restore_best` picks weights by their loss on
  the first training batch.
- The optimizer, `NNCostEvaluator`, `ParamLayout` and `train()` unwrap `torch.compile`
  modules, so parameter names lose the `_orig_mod.` prefix. A 1.0 optimizer checkpoint
  of a compiled hybrid or factored run no longer loads; rebuild the optimizer from the
  saved model weights.
- New errors: momentum outside [0, 1), invalid radius settings, an unknown or nonfinite
  `scale_cost`, epsilon schedules with a non-positive target, SVD ratios outside
  [0, 1], `subspace_particle_dim` below 1, and `PolyStep.init_state` with one particle
  under `ProgressiveEpsilon`.
- New warnings: `step()` warns once when it may score a stale registered batch, the
  constructor warns when a full-space step would score more than 100,000 candidates,
  and a CUDA model with stock `nn.MultiheadAttention` gets a hint to use
  `VmapSafeMultiHeadAttention`. The every-step rotation warning covers every subspace,
  and `HybridSubspace` no longer warns for a `rotation_interval` above 1.
- `SpikingNet` uses the built-in LIF neuron by default; `use_snntorch=True` raises
  `ImportError` without snntorch.
- `SiteVmapEvaluator` and `SiteInplaceEvaluator` are built as
  `SiteVmapEvaluator(owner, layout)` and read the model, loss, buffers and
  `per_sample` from the `NNCostEvaluator` owner.
- Seeded runs also change with `AdaptiveSubspace`, displacement rotation or
  `biased_rotation`, under `amortize_steps` with periodic rotation or absorbs, with an
  even `num_probe` under `adaptive_num_probe`, with screening, and with
  `train(restore_best=True)` on loaders of more than one batch. Functional losses,
  traced chains, site-local and tied-weight scoring, and half-precision attention,
  losses and subspace folds move at rounding level.

## Upgrading from 0.13

Build a new optimizer from the saved model weights: checkpoint formats before 6 are
unsupported. Pass the module to `PolyStepOptimizer`; use `step_batch` for supervised
batches and `PolyStepES` for ask/tell objectives.

Use `AdaptiveSubspace` in place of the removed CMA subspace, `SoftmaxSolver` with
`ent_epsilon` in place of tempered softmax, and `ProgressiveEpsilon` in place of
`auto_epsilon`. The [1.0 release notes](../CHANGELOG.md#100---2026-10-13) record the
other removals. Current signatures and defaults are documented above.

Random streams and floating-point rounding can differ from earlier releases. Keep
the version, configuration, device, and thread count fixed for repeatable runs; see
[determinism](determinism.md).
