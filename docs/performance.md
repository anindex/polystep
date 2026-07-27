# Per-step cost and the compile flags

PolyStep runs `P*V*K` forward passes per step (`P` particles x `V` polytope
vertices x `K` probe radii). Measure wall-clock, not forward count: on small
nets the cost is launch/dispatch overhead, not FLOPs.

## Particles are batched

`NNCostEvaluator` evaluates all `N = P*V*K` particles in one vectorized `vmap`
sweep (`torch.func.functional_call` over the candidate axis), so per-kernel
launch overhead is amortized across particles. This is the default path.

## Cutting the cost per candidate: `FactoredSubspace`

`HybridSubspace` builds each candidate's weights with a
`(N, coords) @ (coords, num_params)` matmul and hands the evaluator `N` full copies
of the model. On a 203K-param MLP at batch 256 that is 63% of the step:

| component | ms/step | share |
|---|---|---|
| `reconstruct_batch` | 355.6 | 62.7% |
| forward evaluation | 175.3 | 30.9% |
| everything else | 35.2 | 6.2% |

`FactoredSubspace` perturbs each 2D parameter by `dW = A @ B` with `B` fixed, so
`x (W + A B)^T = x W^T + (x B^T) A^T` and no candidate weight is ever built. Same
model and batch, matched search dimension:

| subspace | dim | ms/step | peak memory |
|---|---|---|---|
| `HybridSubspace(rank=8)` | 10714 | 626.6 | 19.6 GB |
| `FactoredSubspace(rank=32)` | 8558 | 56.1 | 7.9 GB |

11.2x per step, 8.9x per forward pass.

**It is a speed/accuracy trade, not a free win, and it is not the default.** The
per-step cost falls but so does per-step progress: `dW = A @ B` confines every
perturbation to the `r` input directions spanned by `B`, while `HybridSubspace`'s dense
projection spans arbitrary directions in the full weight space. On the MNIST example
(3 epochs, matched dimension, same schedules):

| subspace | dim | time | best test acc |
|---|---|---|---|
| `HybridSubspace(rank=8)` | 8538 | 38.1 s | **90.87%** |
| `FactoredSubspace(rank=64)` | 8430 | 10.2 s | 81.36% |
| `FactoredSubspace(rank=64, rotation_interval=5)` | 8430 | 11.8 s | 86.74% |
| `FactoredSubspace(rank=64, rotation_interval=1)` | 8430 | 30.5 s | 88.79% |
| `FactoredSubspace(rank=128, rotation_interval=1)` | 16622 | 55.1 s | 90.85% |

`rotation_interval > 0` redraws `B`, which recovers most of the accuracy by refreshing
the directions searched, at the cost of an absorb and a basis rebuild per rotation.
Reach for it when a step is dominated by weight materialization and you can accept a
few points, or when memory is the binding constraint.

Requires a plain `nn.Sequential` of Linear and activation layers; anything else falls
back to the materializing path with a warning. See `polystep.factored_subspace` and
arXiv:2511.16652.

## CPU thread count

The optimizer issues many small tensor ops per step: probe construction, the Sinkhorn
inner loop, the barycentric projection. Torch's intra-op pool loses badly at that size,
because the per-op fork/join costs more than the arithmetic. On a 24-core machine, 300
chained `(256, 256)` matmuls took **1.91 s on 24 threads and 0.070 s on one**, and the
whole test suite went from 504 s to 8 s under `torch.set_num_threads(1)`.

If your model is small enough that a forward pass does not saturate the cores, pin the
thread count:

```python
torch.set_num_threads(1)
```

The crossover is where a single candidate's forward pass becomes large enough to
parallelise on its own; measure rather than assume. When running several trials
concurrently, one thread each is almost always right, since the trials already fill the
machine.

## Basis construction

Every subspace builds its projection with a QR of a tall-thin CPU matrix. This is the
same effect at its worst: `qr` of a `(784, 64)` fp32 CPU tensor measured 1602 ms on 24
threads against 0.30 ms on one. `polystep.solvers._prelude.thin_qr` pins to a single
thread for the call and restores the previous count on the way out, which cut
`HybridSubspace` setup for the MNIST model from 27.0 s to 97.8 ms.

## Cutting the cost per candidate in full space: `SparseDeltaEvaluator`

A full-space candidate is the base parameter vector with exactly `particle_dim`
scalars replaced, so its output differs from the base in at most `particle_dim`
columns. `SparseDeltaEvaluator` propagates that delta through the elementwise layers
and only goes dense one `Linear` after the perturbed one, instead of running a dense
`bmm` over `N` stacked weight sets. It is built automatically for `nn.Sequential`
models of `Linear` and elementwise layers, and falls back per chunk when a chunk
straddles a parameter boundary, when weights are tied, or when a layer mixes features
(`LayerNorm`, `Softmax`, `Conv`).

It only runs when the optimizer holds the evaluator and its data. `api.train()`
registers both on every batch; direct `step()` callers must call
`register_evaluator(evaluator, inputs, targets)` themselves.

## Chunk sizing sets peak memory

`chunk` bounds how many candidates are evaluated at once. It is derived from the
tensors a step actually allocates per candidate: the config tensor, plus a full
weight set when a subspace runs without the fused or factored path, plus the vmap
forward's activations (batch rows x widest layer output). Budgeting on the config
tensor alone under-counted both of the other two.

Measured on a 120,010-param MLP (64-1600-10), batch 128, CPU, one step, peak RSS:

| mode | budget on configs only | budget on what is allocated |
|---|---|---|
| full space | 1661 MB, 43.6 s | 946 MB, 1.4 s |
| `HybridSubspace` | 10565 MB, 436.0 s | 1461 MB, 122.8 s |

Smaller chunks also raise sparse-delta coverage, since a short chunk is likelier to
sit inside one parameter entry: 1163 of 1166 chunks on this model, against 427 of 430
at the larger chunk. Set `chunk_size` explicitly to override the estimate.

## Cutting the candidate count

`multifidelity_screen=True` ranks all `V` directions on a `screen_fidelity` slice of
the batch, then spends the full fidelity only on the top `screen_keep_ratio`
directions. Sample-forwards (candidates x batch rows) drop to about
`screen_fidelity/num_probe + screen_keep_ratio` of the unscreened budget: 0.75x at
the defaults, 0.40x at `screen_keep_ratio=0.25, screen_fidelity=0.15`. The screen is
skipped, with a warning, whenever that figure is not below 1, since it would then buy
work rather than save it.

Wall-clock follows only when the per-sample cost dominates. The screen adds one
extra evaluation call per step, so on small batches that call costs more than the
rows it saves. Measured on CPU, 2-layer MLP, defaults `0.5 / 0.25`:

| batch | plain | screened | speedup |
|-------|-------|----------|---------|
| 64    | 325 ms  | 492 ms  | 0.66x |
| 1024  | 1458 ms | 1380 ms | 1.06x |
| 8192  | 6741 ms | 6015 ms | 1.12x |

Turn it on for large batches, large models, and GPU runs; leave it off for small
batches, where it is a slowdown. `optimizer._last_screen_savings` reports the
fraction of candidate evaluations skipped on the last step.

## Two opt-in compile flags (off by default)

- `compile_evaluator=True`: wraps the vmapped forward in
  `torch.compile(mode="default")` (Inductor fusion, no CUDA graphs). Roughly
  1.2-1.9x on top of vmap, largest on FLOP-heavy nets (CNN), smallest on
  recurrent SNNs that vmap already amortized. Best backend when it fits.
- `compile_forward=True`: for the sequential in-place path (auto-selected for
  >500K-param GPU models, or `use_inplace=True`, where a full vmap sweep would
  OOM), CUDA-graphs the forward+loss closure and replays it per candidate. It
  keeps the memory-forced path competitive with vmap; it does not beat vmap.
  The in-place path is chosen for memory, not speed.

Both are opt-in because compiling in the inner loop reorders floating-point
reductions (loss values change slightly, so the OT trajectory can shift), adds
first-call compile latency, and recompiles on shape/dtype changes. On
hard-threshold nets (SNN) the compiled loss can differ at the discretization
scale (~1.6e-3), shifting candidate selection. Enable for CUDA + shape-stable runs.

## Which path runs

| model                                 | path            | flag                     |
|---------------------------------------|-----------------|--------------------------|
| pure `nn.Sequential` MLP (vanilla CE) | `bmm` fast path | none (already fast)      |
| custom forward, fits in memory        | `vmap`          | `compile_evaluator=True` |
| large net / vmap OOMs                 | in-place swap   | `compile_forward=True`   |

Both flags propagate to a registered evaluator via `register_evaluator` and are
read by `api.train()`, so `PolyStepOptimizer(compile_evaluator=True)` works on
every path. Reproduce with `experiments/scripts/bench_forward_backends.py` and
`experiments/scripts/bench_large_net_inplace.py`.
