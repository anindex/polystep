# API reference

## Optimizer

`PolyStepOptimizer` trains an `nn.Module` from candidate losses. The closure accepts
a dictionary of batched parameters and returns one loss per candidate:

```python
import torch
from torch import nn
from polystep import NNCostEvaluator, PolyStepOptimizer

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
loader = [(torch.randn(32, 16), torch.randint(0, 3, (32,))) for _ in range(8)]
optimizer = PolyStepOptimizer(model, epsilon=0.1, step_radius=0.15)
evaluator = NNCostEvaluator(model, nn.CrossEntropyLoss())

for x, y in loader:
    optimizer.register_evaluator(evaluator, x, y)
    optimizer.step(lambda p: evaluator.evaluate(p, x, y))
optimizer.release_evaluator()
```

The evaluator and closure must compute the same objective, including regularization.
`NNCostEvaluator` puts the model in eval mode when it is constructed and scores every
candidate in eval mode, so dropout stays off and batch-norm running statistics do not
update.

The optimizer, `NNCostEvaluator`, `ParamLayout` and `train()` unwrap a `torch.compile`
module and work on the module inside it, so parameter names carry no `_orig_mod.` prefix.
To compile candidate scoring, pass `compile_evaluator` or `compile_forward` instead.

After external parameter updates, call `optimizer.resync_from_model()`.
After you replace layers, parameters, or buffers, call `evaluator.reset_vmap()`.
Do the same after a dtype cast. Then register the evaluator again.
For a layer swapped in under the same parameter names, also call `optimizer.resync_from_model()`.
Otherwise, the next step overwrites it with the old weights plus the update.
The optimizer keeps the set of `requires_grad` parameters it was built with, and `step()`
raises `ValueError` if that set changes. To freeze, unfreeze, add or remove a parameter,
set `requires_grad` first and then build a new optimizer.

After a dtype cast, resync moves the subspace projections to the new parameter dtypes.
Full space and an `AdaptiveSubspace` also move their coordinates to the new dtype.
`HybridSubspace` and `FactoredSubspace` keep their original coordinate dtype.
If the coordinates must use the new dtype, build a new optimizer.
If the model moves to another device, build a new optimizer. Resync cannot change devices.

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

`restore_best` (on by default) restores the weights with the lowest tracked loss.
After each step, `train()` scores the updated weights on the first training batch, after `cost_batch_size` subsampling, so scores compare across steps.
Callbacks receive the loss of the updated weights on that step's training batch.
Both losses use the same input cast, autocast and loss reduction as the candidates.
Checkpoint selection uses training loss, which can be optimistic on unseen data.
`cost_batch_size` subsampling indexes only the input and target tensors whose leading dimension matches the first input's batch size, and passes scalars and other tensors through unchanged.
Best-weight tracking and callbacks each require an extra loss evaluation per step.

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
| `adaptive_num_probe` | `None`: on when `num_probe > 1`, except full-space `per_layer`/`grouped` and orthoplex `newton_refinement` | Drops the probe count to 1 while costs keep falling |

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

The quadratic model (`use_quadratic_model=True`) reads a centred unit-radius tight
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
remaining bias comes from third-order terms, is `O(r^2)`, and is zero on a quadratic.
Block models omit cross-block Hessian terms.

`trust_region` compares consecutive incumbents and sums block predictions. Minibatch
changes and momentum-only steps invalidate the deferred comparison; `step_batch()` and
`train()` reject this controller. Use a stationary objective with the manual step API, and pass
`optimizer.step(closure, objective_token=token)` a new token whenever the batch or
loss changes; a change drops the pending trust-region comparison, and `None` or a
constant token means a stationary objective.

## Evaluation controls

| Option | Default | Scope |
|---|---|---|
| `compile` | `False` | CUDA geometry and solver kernels |
| `compile_evaluator` | `False` | Dense and site-local candidate vmap |
| `compile_forward` | `None` | CUDA graphs for the in-place forward and the CUDA site-local vmap |
| `candidate_autocast` | `False` | Candidate forward precision; disables incompatible fast paths |

`amortize_steps` skips evaluations between sampled updates. `multifidelity_screen`
scores candidates on a smaller batch and re-scores on the full batch only the
`round(V * screen_keep_ratio)` vertices it keeps per particle, at least the solver's
`k` for `top_k_mean` and whole antithetic pairs on the orthoplex. Measure the quality
of either at equal evaluation and time budgets.

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

A registered evaluator screens on the same leading slice of its batch. Screening runs
only when the screened share of the batch divided by `num_probe`, plus the share of
vertices kept per particle, is below 1, where it needs fewer forwards than the full
sweep, and not with `use_quadratic_model`, `newton_refinement` or `trust_region`. It
still runs two sweeps, so on small models and on launch-bound GPU steps it can take
longer than one unscreened sweep. Screened candidates use the same delta and site paths
as the full sweep once particles times vertices times model parameters reaches 5e5 on
the CPU or 5e8 on CUDA. Below that, screened hybrid candidates build every weight, and a
screened full-space chunk uses a site path only when it lies inside one parameter. The
CPU threshold is the measured crossover. The CUDA one is set conservatively so that no
measured model slows down, which forgoes wins on small MLPs. With few vertices the kept
share is coarse: at the default `particle_dim=2` the simplex has 3 vertices, so
`screen_keep_ratio=0.5` keeps 2 of them. The batch share rounds the same way: a
registered batch of 2 samples screens on 1 of them even at `screen_fidelity=0.1`.

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

objective = lambda x: x.square().sum(dim=1)
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
`ValueError` on any other format. To continue a run saved by another release, build
a fresh optimizer around the saved model weights.
The checkpoint also restores `candidate_evals`. Reproducible continuation requires
the same objective and environment. Checkpoints can move between CPU and CUDA.
When a seeded checkpoint crosses devices, loading keeps the new optimizer's
generator and warns because the two generator types cannot share state.

## Upgrading from 0.13

Version 1.0 removes unused APIs and changes the checkpoint format. Update callers
before loading a model into a new optimizer.

- `CMAAdaptiveSubspace`, `compute_cma_hyperparameters`, `use_covariance_adaptation` and
  the CMA fields of `SolverState` (`p_c`, `p_sigma`, `C_diag`, `generation`) are gone;
  use `AdaptiveSubspace`. `TemperedSoftmaxSolver`, `solver='tempered_softmax'`
  and `tempered_softmax_tau` are gone; use `solver='softmax'` with `ent_epsilon`.
- `auto_epsilon` and `auto_epsilon_config` are gone; pass a `ProgressiveEpsilon` as
  `epsilon`. To match the old schedule for a numeric `epsilon=e`, use
  `ProgressiveEpsilon(init=e, target=max(0.01, 0.1 * e), max_epsilon=5 * e)`.
  For a `LinearEpsilon`, reuse its `init` and `target` with `max_epsilon=5 * init`;
  for other schedules, use `ProgressiveEpsilon()`. Pass `auto_epsilon_config` keys
  as keyword arguments. An existing `ProgressiveEpsilon` works as before.
- `PolyStep` no longer trains networks: `subspace`, `nn_evaluator`, `train_inputs`,
  `train_targets` and `init_state(base_params=...)` are gone. Use `PolyStepOptimizer`.
- `polystep` no longer exports low-level helpers; import them from `subspace`
  (`ProjectionSpec`, which replaces `create_hybrid_blocks` and `LayerProjectionSpec`),
  `_compiled` (`CompiledFunctions`, `try_compile`), `dynamics` (`apply_momentum`,
  `compute_momentum_coefficient`, `update_radius_multiplier`, `update_stagnation`),
  `blockwise` (`BlockConfig`, `create_per_layer_blocks`, `create_grouped_blocks`),
  `cost_nn` (`auto_detect_chunk_size`), `transform` (`ParamEntry`, `create_generator`)
  and `costs` (`compute_cost_matrix`, `scale_cost_matrix`).
- `TrainConfig.batch_size` and `log_every` are gone; set them on the `DataLoader` and
  `LoggingCallback`, which now prints `disp_sq` in scientific notation, not `disp`.
- `SolverResult.cost` is gone; read `ent_reg_cost`, no longer a constructor argument.
  `KLSoftmaxSolver.last_marginal_violation` is read-only. `SolverState` and
  `CompiledFunctions.fused_softmax_project` drop the marginal `a`. The latter expects
  the caller to recenter and scale costs; its `scale_cost_mean` argument is gone.
  It returns `(X_new, W)` with rows of `W` summing to one; multiply by the marginal
  to get the old transport plan.
- Removed without replacement: `BenchmarkResult`, `data_dependent_init`, unused solver
  `compile` fields, the block-splitting helpers in `polystep.blockwise`, and
  `adaptive_probes` with its threshold. Candidates are scored on every OT step, and
  `objective_token` only gates a pending trust-region check.
- `get_simplex_vertices`, `get_orthoplex_vertices` and `get_cube_vertices` raise
  `TypeError` on unknown keyword arguments. `CartPoleRolloutResult` is gone;
  `CartPoleEvaluator.rollout_stacked_params` returns a `GymRolloutResult` with the same
  fields. `NonDiffMLPPolicy.mode` is gone; read it from the policy's `NonDiffActivation`
  layers. `SiteVmapEvaluator` takes a required keyword-only `owner`, and `PolyStep.step`
  leaves `state.last_solve_eps` at `None`.
- `experiments/baselines/openai_es.py` and `spsa.py` are gone; use `polystep.baselines`.
  `evaluate_accuracy` moved from `experiments.runners.common` to
  `polystep.benchmarks.utils` and reads the device from the model. With no replacement,
  `common.py` drops `FunctionEvalCounter`, `get_loss_fn`, `load_flat_params`,
  `set_flat_params` and the Tonic loaders; `nondiff_models.py` drops the soft MoE,
  permutation and binary-convolution models, `cra_penalty`, `evaluate_sat_loss` and
  `compute_expert_utilization`; `nondiff_data.py` drops `generate_sorting_data`; and
  `sls_pysat.py`, `bench_ask_tell.py`, `run_maxsat_softmax_scaling.py`,
  `bench_large_net_inplace.py` and `profile_controlled.py` are gone.
- The elevation, MNIST, MoE, timeseries, MAX-SAT and RL runners write to
  `experiments/results/revision` by default, not `results/softmax/main`
  (`results/softmax/rl` for RL). Baseline runs given neither `--fair` nor `--budget`
  record `match_axis` as `"unmatched"`. `run_gpt2_finetune.py` keeps only the
  hard-quantized head experiment and drops `--steps`, `--subspace-dim`, `--head-only`
  and `--measure-memory`. `bench_polytope.py` drops `--suite`, and the variant sweep
  drops its `quadratic/multifidelity` arm. The `[experiments]` extra no longer installs
  snntorch, and `[rl]` installs plain gymnasium without Box2D or swig.
- Checkpoints use format 6 and restore `candidate_evals`. Older ones, including the
  format 5 that 0.13 writes, raise `ValueError`; rebuild the optimizer from the model.
- New errors: `TypeError` for a subspace that is not an `AdaptiveSubspace`,
  `HybridSubspace` or `FactoredSubspace`; `ValueError` for `ProgressiveEpsilon` with
  `ent_epsilon`, `rank_schedule` without a `HybridSubspace`, `PowerDecay` with `gamma`
  outside `(0, 0.5]`, a negative `CosineEpsilon` decay, a blockwise closure that does
  not return one loss per candidate, and `resync_from_model()` on a moved model.
  `PolyStep.init_state` raises `ValueError` for one particle with `ProgressiveEpsilon`.
- New warnings cover two configurations: `AdaptiveSubspace` rotating every step
  with `amortize_steps > 1`, `biased_rotation` or `use_momentum`, because rotation
  clears their state; and `adaptive_num_probe=True` under orthoplex
  `newton_refinement`. The thread-oversubscription warning suggests
  `torch.set_num_threads(1)` or `python -m polystep.benchmarks.threads` instead of
  `ncpu - 8` threads.
- Call `resync_from_model()` after swapping a layer into the model, or steps restore the
  old weights. A closure that keeps a candidate tensor must clone it, since CPU
  `AdaptiveSubspace` steps now reuse one buffer, as full-space steps did.
- With `use_inplace=True` and a registered evaluator, full-space candidates that change
  one parameter are scored with the registered loss and batch, not the closure. Move
  any term the closure adds, such as a penalty, into that loss.
- Subspace `per_layer` and `grouped` runs with `num_probe > 1` now default
  `adaptive_num_probe` to on; pass `False` to keep a fixed probe count.

Random streams and floating-point rounding can differ from earlier releases. Keep
the version, configuration, device, and thread count fixed for repeatable runs; see
[determinism](determinism.md).
