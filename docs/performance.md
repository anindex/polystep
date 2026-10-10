# Performance

A step usually scores `P * V * K` candidates: parameter blocks, vertices, and probe
radii. The simplex uses `d + 1` vertices; the orthoplex uses `2d`. The quadratic
model adds one shared incumbent evaluation when used by rotation or radius adaptation.

## Candidate evaluation

| Model | Path | Selection |
|---|---|---|
| Sequential MLP, full space | Sparse delta | Automatic after registration |
| Sequential MLP, hybrid subspace | Subspace delta | Automatic after registration |
| Sequential MLP, factored subspace | Subspace delta, low-rank correction as fallback | Automatic after registration |
| Convolution, attention, custom forward | Site-local vmap | Automatic when supported |
| Sequential MLP, dense candidates | Batched matrix multiplication | Automatic |
| Other models | vmap or sequential fallback | Automatic |
| Large candidates | In-place parameter swaps | `use_inplace=True` |

The delta and factored paths treat each sample as one feature vector, so they need
two-dimensional inputs or a leading `nn.Flatten`. A Sequential MLP applied per token
to `(batch, tokens, features)` inputs takes the site-local path instead in full-space
and hybrid runs; factored runs materialize its candidates.

`use_inplace=True` runs the model's own forward, one candidate at a time. Each sweep
writes the incumbent into the model once; a full-space or `HybridSubspace` candidate
that changes one parameter then rewrites only that parameter, and in full space only
its own entries. Tied weights share storage, so they move together. The registered
evaluator's loss scores these candidates. Candidates that span two parameters,
FP16/BF16 hybrid weights and non-contiguous parameters swap in every weight instead,
as do screened hybrid candidates and the hybrid incumbent.
An evaluator that picks the in-place path for memory takes the same route when the
model ties weights.

A forward that draws its own random numbers, such as injected noise, stays batched:
vmap gives each candidate its own draw, so identical candidates can score differently
and a seeded run's stream differs from scoring them one at a time. A forward vmap
cannot trace, such as one that calls `.item()` on an activation, falls back to a
sequential loop with a warning. Buffers the refused forward wrote are restored before
either the retry or the loop runs.

Register the evaluator before each batch; `train()` does this automatically:

```python
evaluator = NNCostEvaluator(model, loss_fn)
optimizer.register_evaluator(evaluator, inputs, targets)
optimizer.step(lambda p: evaluator.evaluate(p, inputs, targets))
```

The closure and evaluator must compute the same objective, including regularization.
Call `evaluator.reset_vmap()` after replacing layers, parameters, or buffers, then
register it again. Call `optimizer.resync_from_model()` after external weight changes,
including a swapped layer or parameter.

Delta paths reuse base activations within each candidate sweep. Reuse applies to
supported builtin layers and ends before the next batch or fidelity level.
Custom layers can declare the contracts in the [API reference](api_overview.md#custom-layers).

Dense MLP evaluation also shares activations until the first changed layer when
unchanged candidate parameters are expanded views. Shared weights use one linear
operation across candidates. `chunk_size` bounds activation batches on this path.

## Subspaces and memory

Use `HybridSubspace.from_layout(..., max_subspace_dim=...)` to bound candidate count;
rank alone does not bound the total dimension. The cap shrinks every weight matrix in
proportion, while biases and other 1D parameters keep their full width; a cap below
that width warns and leaves each matrix one coordinate. Smaller subspaces reduce cost
but restrict the search. Retune the radii when changing the representation.

`FactoredSubspace` uses `dW = A @ B`, with fixed `B`, so sequential linear layers can
compute `x @ W.T + (x @ B.T) @ A.T` without constructing candidate weights. A
candidate changes a few consecutive coordinates of one `A`, so the subspace-delta
path scores it from `x @ B.T` and moves only the output units those coordinates
belong to. Runs that span two parameters, screened candidates and the incumbent use
the full low-rank correction. The search is limited to the input directions spanned
by `B`; basis rotation changes that search space.

`AdaptiveSubspace` still builds every candidate's full weights, since its single
projection `P` spreads each coordinate over every parameter. On the CPU, with a dense
`P`, it builds them as the incumbent plus `P[:, cols] @ delta`, where `cols` are the
coordinates of the one particle the candidate moves, so each candidate reads
`subspace_particle_dim` columns of `P` instead of all of them. The padded last
particle, screened candidates and the incumbent use the full `base + P @ x`, as do
sparse projections and FP16/BF16 runs. When `P` is dense FP32 or FP64 and the step
has no screen, every chunk, the padded particle and the incumbent included, is written
into one buffer that the optimizer keeps between steps, because a fresh allocation of
that size costs the CPU a page fault per page. A closure that keeps a candidate
tensor after it returns therefore has to clone it. CUDA keeps the full product for every
candidate, since one dense GEMM measured faster there than the many thin batched
products the offsets need.

`chunk_size` limits candidates held at once. The automatic estimate counts the
candidate's parameters and its largest layer output, which one forward measures when
the evaluator is first registered and which then scales with the batch. Delta and
site paths build no full candidate, so they are budgeted on activations and the
perturbed parameter alone. Reduce `chunk_size` after an out-of-memory error. Steps
that score on a delta or site path, or on CPU `AdaptiveSubspace` offsets, keep
whole particles in each chunk, every vertex and probe of one particle, so
`chunk_size` drops to a multiple of the vertex count times `num_probe` and never
goes below one such group. A particle that straddles two parameters, or holds the
padding at the end, gets a chunk of its own and falls back to materialization, in
batches of at most `chunk_size`.
Per-layer blocks reduce the size of each OT solve, but not the forward count.
With a subspace, a block strategy only splits Sinkhorn and KL solves into at most
eight row segments; candidates are scored exactly as in the monolithic step.

## Compilation and precision

| Option | Default | Scope |
|---|---|---|
| `compile` | `False` | Geometry and solver kernels on CUDA |
| `compile_evaluator` | `False` | Dense and site-local vmap evaluation |
| `compile_forward` | `None` | Single-candidate in-place forward; automatic on that path |
| `candidate_autocast` | `None` | Candidate forward precision; disables incompatible fast paths |

Measure compilation separately from steady-state execution. Each site compiles
independently and falls back to eager execution on compilation failure. Profiler
output confirms whether a compiled region actually ran. PyTorch's recompilation
limit can cause later calls to run eagerly.

CPU attention uses native scaled-dot-product attention; CUDA uses explicit matmuls.
BF16/FP16 solver costs accumulate in FP32. An `nn.CrossEntropyLoss`, `nn.MSELoss`,
or `nn.L1Loss` left at its defaults apart from `reduction` also reduces
half-precision candidate outputs in FP32, whether they come from an FP16/BF16 model or from `candidate_autocast`.
Other losses keep their own dtype. Mixed-dtype models require per-entry subspaces;
see [limitations](../LIMITATIONS.md).

With FP16 or BF16 parameters the subspace-delta and factored paths are off. They add
each candidate's change after the matmul, so they would rank changes that the stored
weights round away. Hybrid candidates go to the site-local path instead, which
builds the changed weight and the shared ones from the base weights the way a
materialized candidate is built; factored candidates are materialized. The weights
swapped into the model under `use_inplace=True`, and the ones a step writes back
when it ends, round the same way.

## Radius models

For a centered unit-radius tight frame with `V` vertices in dimension `d`:

```text
g       = d / (V s r) * sum_v (L_v - L_reference) v
tr(H)/d = 2 * (mean_v L_v - f(X)) / (s r)^2
```

The curvature identity is exact for quadratic objectives. Simplex gradient fits can
have an `O(r)` bias on anisotropic quadratics; orthoplex central differences cancel
that term. Block models omit cross-block Hessian terms.

`trust_region` compares consecutive incumbents and sums block predictions. Pass a
new `objective_token` whenever the objective changes. Minibatch changes and
momentum-only steps invalidate the deferred comparison; `train()` rejects this
controller. Use a stationary objective with the manual step API.

`amortize_steps` skips evaluations between sampled updates. `multifidelity_screen`
uses a smaller batch to select candidates for full evaluation; it requires
`screen_fidelity / num_probe + screen_keep_ratio < 1` and cannot run with the
quadratic model. Measure quality at equal evaluation and time budgets.

## CPU measurements

PyTorch 2.11, one thread. Complete MLP steps, batch 32, median over five seeds and
seven timed steps after warmup:

| Hidden width | Search | Without prefix reuse | With reuse |
|---|---|---:|---:|
| 64 | Full space | 40.91 ms | 39.23 ms |
| 256 | Full space | 736.17 ms | 659.79 ms |
| 256 | 1,542 coordinates | 28.27 ms | 28.23 ms |

Attention kernel medians, 16 candidates and batch 16:

| Sequence length | Explicit attention | Native CPU SDPA |
|---|---:|---:|
| 6 | 0.452 ms | 0.289 ms |
| 64 | 7.267 ms | 1.841 ms |
| 128 | 37.051 ms | 6.303 ms |

These kernel timings do not measure complete optimizer steps. Test CPU thread
counts on the actual workload; small operations often run faster with one thread.
GPU results depend on device, shapes, and backend selection.

## Benchmarks

Run from the repository root, one timing job at a time:

```bash
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --profile
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --prefix --repeats 7 --warmup 2
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --attention --batch 16
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --compile --profile
PYTHONPATH=src:. python experiments/scripts/bench_polytope.py --seeds 0 1 2 3 4 --budget 4096 --wall
```

Use `--device cuda` for GPU measurements and `--output` to keep separate result files.
Forward benchmarks check loss parity and report cold time, median, IQR, and executed
operators. GPU peak allocation is measured; CPU allocation is left unreported.
