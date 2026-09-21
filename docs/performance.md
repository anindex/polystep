# Performance

A step usually scores `P * V * K` candidates: parameter blocks, vertices, and probe
radii. The simplex uses `d + 1` vertices; the orthoplex uses `2d`. The quadratic
model adds one shared incumbent evaluation when used by rotation or radius adaptation.

## Candidate evaluation

| Model | Path | Selection |
|---|---|---|
| Sequential MLP, full space | Sparse delta | Automatic after registration |
| Sequential MLP, hybrid subspace | Subspace delta | Automatic after registration |
| Sequential MLP, factored subspace | Factored correction | Automatic after registration |
| Convolution, attention, custom forward | Site-local vmap | Automatic when supported |
| Sequential MLP, dense candidates | Batched matrix multiplication | Automatic |
| Other models | vmap or sequential fallback | Automatic |
| Large candidates | In-place parameter swaps | `use_inplace=True` |

Register the evaluator before each batch; `train()` does this automatically:

```python
evaluator = NNCostEvaluator(model, loss_fn)
optimizer.register_evaluator(evaluator, inputs, targets)
optimizer.step(lambda p: evaluator.evaluate(p, inputs, targets))
```

The closure and evaluator must compute the same objective, including regularization.
Call `evaluator.reset_vmap()` after replacing layers, parameters, or buffers, then
register it again. Call `optimizer.resync_from_model()` after external weight changes.

Delta paths reuse base activations within each candidate sweep. Reuse applies to
supported builtin layers and ends before the next batch or fidelity level.
Custom layers can declare the contracts in the [API reference](api_overview.md#custom-layers).

Dense MLP evaluation also shares activations until the first changed layer when
unchanged candidate parameters are expanded views. Shared weights use one linear
operation across candidates. `chunk_size` bounds activation batches on this path.

## Subspaces and memory

Use `HybridSubspace.from_layout(..., max_subspace_dim=...)` to bound candidate count;
rank alone does not bound the total dimension. Smaller subspaces reduce cost but
restrict the search. Retune the radii when changing the representation.

`FactoredSubspace` uses `dW = A @ B`, with fixed `B`, so sequential linear layers can
compute `x @ W.T + (x @ B.T) @ A.T` without constructing candidate weights. Its
search is limited to the input directions spanned by `B`; basis rotation changes
that search space.

`chunk_size` limits candidates held at once. The automatic estimate includes
parameter and activation storage. Reduce it after an out-of-memory error.
Per-layer blocks reduce the size of each OT solve, but not the forward count.

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
BF16/FP16 solver costs accumulate in FP32. Mixed-dtype models require per-entry
subspaces; see [limitations](../LIMITATIONS.md).

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

`adaptive_probes` can reuse losses only when the whole configuration and objective
are unchanged. `amortize_steps` skips evaluations between sampled updates.
`multifidelity_screen` uses a smaller batch to select candidates for full evaluation;
it requires `screen_fidelity / num_probe + screen_keep_ratio < 1` and cannot run with
the quadratic model. Measure quality at equal evaluation and time budgets.

## CPU measurements

PyTorch 2.11, one thread. Complete MLP steps, batch 32, median over five seeds and
seven timed steps after warmup:

| Hidden width | Search | Without prefix reuse | With reuse |
|---|---|---:|---:|
| 64 | Full space | 40.91 ms | 39.23 ms |
| 256 | Full space | 736.17 ms | 659.79 ms |
| 256 | 512 coordinates | 28.27 ms | 28.23 ms |

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
