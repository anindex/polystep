# Performance

A step usually scores `P * V * K` candidates: particles, vertices, and probe radii.
Each particle holds `d` coordinates, `particle_dim` (default 2) in full space or
`subspace_particle_dim` (default 8) with a subspace. The simplex has `d + 1`
vertices, the orthoplex `2d`, and the cube `2^d`. The README quickstart searches 128
coordinates, so at the default `num_probe=1` a step scores 16 particles times the 9
simplex vertices, 144 candidates. The quadratic model adds one shared incumbent
evaluation when `biased_rotation` or `trust_region` uses it.

## Quick checklist

- Pick the thread count with `python -m polystep.benchmarks.threads`.
- Register the evaluator for each batch, or use `train()`.
- Cap `max_subspace_dim`.
- On a small CUDA model, try a `chunk_size` at or above the candidate count.
- Time compilation separately from steady-state steps.

## Candidate evaluation

| Model | Path | Selection |
|---|---|---|
| Sequential MLP, full space | Sparse delta | Automatic after registration |
| Sequential MLP, hybrid subspace | Subspace delta | Automatic after registration |
| Sequential MLP, factored subspace | Subspace delta, low-rank correction as fallback | Automatic after registration; on CUDA the low-rank correction when one chunk holds every candidate or chunks are small |
| Any model, `AdaptiveSubspace` on CPU | Incumbent plus the moved particle's projection columns | Automatic with a dense FP32 or FP64 projection |
| Convolution, attention, custom forward | Site-local vmap | Automatic when supported |
| Sequential MLP, dense candidates | Batched matrix multiplication | Automatic |
| Other models | vmap or sequential fallback | Automatic |
| Models too large for batched weight copies | In-place parameter swaps | Automatic on CUDA when eight weight copies would take more than half the free memory; `NNCostEvaluator(..., use_inplace=True)` forces it |

The delta and factored paths treat each sample as one feature vector, so they need
two-dimensional inputs or a leading `nn.Flatten`. A Sequential MLP applied per token
to `(batch, tokens, features)` inputs takes the site-local path instead in full-space
and hybrid runs; factored runs materialize its candidates.

`NNCostEvaluator(model, loss_fn, use_inplace=True)` runs the model's own forward,
one candidate at a time, and turns off the vmap, delta and factored paths. Each
sweep writes the incumbent into the model once; a full-space or `HybridSubspace`
candidate that changes one parameter then rewrites only that parameter, and in full
space only its own entries. Tied weights share storage, so they move together. The
registered evaluator's loss scores these candidates. Candidates that span two
parameters, FP16/BF16 hybrid weights and non-contiguous parameters swap in every
weight instead, as do screened hybrid candidates, the hybrid incumbent and models
with a `max_norm` embedding, whose forward rewrites the rows it looks up. When the
evaluator picks the in-place path on its own to save memory, it still uses the
site-local, delta and factored paths where they apply and swaps the remaining
candidates in one at a time. It rewrites a single parameter in place only when the
model ties weights.

A forward that draws its own random numbers, such as injected noise, stays batched.
vmap refuses the draw at first, so the evaluator retries once with
`randomness="different"`. Each candidate then gets its own draw: identical
candidates can score differently, and a seeded run's stream differs from scoring
them one at a time. A forward vmap cannot trace, such as one that calls `.item()` on
an activation, falls back to a sequential loop with a warning. Buffers the refused
forward wrote are restored before either the retry or the loop runs.

Register the evaluator before each batch; `train()` does this automatically:

```python
evaluator = NNCostEvaluator(model, loss_fn)
optimizer.register_evaluator(evaluator, inputs, targets)
optimizer.step(lambda p: evaluator.evaluate(p, inputs, targets))
```

The [API reference](api_overview.md#optimizer) covers the closure contract and what
to call after replacing layers, parameters or buffers.

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
the full low-rank correction, which computes `x @ W.T` and `x @ B.T` for the first
perturbed layer once per candidate sweep and reuses them across its chunks.
Screening and incumbent scoring are separate sweeps. The search is limited to the
input directions spanned by `B`; basis rotation changes that search space.

The delta path scores each parameter in its own pass and launches more kernels per
chunk than the low-rank correction, which a GPU pays for while its chunks are too
small to keep it busy. On CUDA it therefore runs only when one chunk of the default
budget cannot hold every candidate and `chunk_size`, or the candidate count when it is
unset, times a candidate's largest layer output reaches 2^24, about 16.8 million,
elements. Other CUDA steps use the low-rank correction, which needs more peak memory
per chunk. The threshold is where the delta path stops losing on shallow MLPs
(784-128-10 at batch 512 on an RTX 5090). On that GPU the delta steps this rule picks
at the default chunk size ran up to 5 times faster, though only a few percent faster
close to the threshold, with 4 to 54 times less peak memory. A batch of one on a
784-512-512-10 or 3072-1024-10 MLP stays below the threshold, and there the delta
path ran up to 1.35 times slower. Below the threshold the low-rank correction ran up
to 1.4 times faster on small or shallow MLPs, but wider, deeper and larger-batch ones
favor the delta path from smaller chunks on, so with a `chunk_size` set they may step
faster with a larger one.

`AdaptiveSubspace` still builds every candidate's full weights, since its single
projection `P` spreads each coordinate over every parameter. On the CPU, with a dense
`P`, it builds them as the incumbent plus `P[:, cols] @ delta`, where `cols` are the
coordinates of the one particle the candidate moves, so each candidate reads
`subspace_particle_dim` columns of `P` instead of all of them. The padded last
particle, screened candidates and the incumbent use the full `base + P @ x`, as do
sparse projections and FP16/BF16 runs. When `P` is dense FP32 or FP64 and the step
has no screen, every chunk, the padded particle and the incumbent included, is written
into one buffer that the optimizer keeps between steps, because a fresh allocation of
that size costs the CPU a page fault per page, so a closure must
[clone any candidate tensor it keeps](../LIMITATIONS.md#evaluation). CUDA keeps the
full product for every candidate, since one dense GEMM measured faster there than the
many thin batched products the offsets need.

`chunk_size` limits candidates held at once. The automatic estimate counts the
candidate's parameters and its largest layer output, which one forward measures when
the evaluator is first registered and which then scales with the batch. Delta and
site-local paths build no full candidate, so they are budgeted on activations and the
perturbed parameter alone. A site-local path batches only the layers from the
perturbed one on, so each site is budgeted on the module outputs from its own layer
on. A forward that draws random numbers is batched from the draw on, so its sites
keep the whole-model budget, as does a parameter the forward reads before calling its
layer. In example 10's LeNet-5 at batch 256, a chunk holds 27 candidates on the first
convolution, whose output fills the budget, 72 on the second and up to 729 on the
fully connected layers, so a step scores its 4,833 candidates in 36 forwards. Reduce
`chunk_size` after an out-of-memory error.

Steps that score on a delta or site-local path, or on CPU `AdaptiveSubspace` offsets,
keep whole particles in each chunk, every vertex and probe of one particle, so
`chunk_size` drops to a multiple of the vertex count times `num_probe` and never goes
below one such group. A particle that straddles two parameters gets a chunk of its
own and falls back to materialization, in batches of at most `chunk_size`. So does the
padded particle at the end of the CPU offsets, or of a per-layer or grouped block
other than the last. On the delta and site-local paths the padded last particle stays
with the particles before it, since its padding moves no parameter. Each chunk costs
one forward, so a small CUDA model, whose forwards are bound by kernel launches, can
step faster with a `chunk_size` at or above the candidate count than with the
automatic one. Per-layer blocks reduce the size of each OT solve, but not the forward
count. With a subspace, a block strategy only splits Sinkhorn and KL solves into at
most eight row segments; candidates are scored exactly as in the monolithic step.

## Compilation and precision

The [API reference](api_overview.md#evaluation-controls) lists the compilation and
autocast options. Measure compilation separately from steady-state execution. Each
site compiles independently and falls back to eager execution on compilation failure.
Profiler output confirms whether a compiled region actually ran. PyTorch's
recompilation limit can cause later calls to run eagerly.

CPU attention uses native scaled-dot-product attention; CUDA uses explicit matmuls.
BF16/FP16 solver costs accumulate in FP32. An `nn.CrossEntropyLoss`, `nn.MSELoss`, or
`nn.L1Loss` left at its defaults apart from `reduction` also reduces half-precision
candidate outputs in FP32, whether they come from an FP16/BF16 model or from
`candidate_autocast`. Other losses keep their own dtype. Mixed-dtype models require
per-entry subspaces; see [limitations](../LIMITATIONS.md).

With FP16 or BF16 parameters the subspace-delta and factored paths are off. They add
each candidate's change after the matmul, so they would rank changes that the stored
weights round away. Hybrid candidates go to the site-local path instead, which
builds the changed weight and the shared ones from the base weights the way a
materialized candidate is built; factored candidates are materialized. The weights
swapped into the model under `use_inplace=True`, and the ones a step writes back
when it ends, round the same way.

## CPU measurements

PyTorch 2.13 on one performance-core thread of an Intel Core Ultra 9 285K. Complete
steps of a 32-w-w-4 MLP at batch 32 with `solver="softmax"`, `chunk_size=72` and a
registered evaluator, the median over five seeds of the median of seven timed steps
after one cold step and two warmup steps. The subspace row is a `HybridSubspace` at
rank 4 capped at 1,024 coordinates:

| Hidden width `w` | Search | Without prefix reuse | With reuse |
|---|---|---:|---:|
| 64 | Full space | 29.15 ms | 27.51 ms |
| 256 | Full space | 411.24 ms | 339.87 ms |
| 256 | 1,024 coordinates | 58.39 ms | 57.80 ms |

With the automatic chunk size, the full-space steps with reuse take about 39 ms at
width 64 and 470 ms at width 256. The subspace row barely changes.

Attention kernel medians, 16 candidates and batch 16:

| Sequence length | Explicit attention | Native CPU SDPA |
|---|---:|---:|
| 6 | 0.371 ms | 0.274 ms |
| 64 | 6.921 ms | 1.841 ms |
| 128 | 34.485 ms | 5.437 ms |

These kernel timings do not measure complete optimizer steps. GPU results depend on
device, shapes, and backend selection.

## CPU threads

PyTorch starts one intra-op thread per core, and whether that helps depends on the
step. On the machine above, the README quickstart step takes about 1.1 ms on one
thread and about the same on 24, while the full-space 784-64-10 step that
`python -m polystep.benchmarks.threads` times drops from 227 ms on one thread to
77 ms on 16 and 79 ms on 24. A pool that covers every core also slows down when
other processes compete for the CPU, so the fast test suite and the examples default
to one thread; [CONTRIBUTING](../CONTRIBUTING.md#tests) and the
[examples](../examples/README.md) list the exceptions. The optimizer warns once when
the pool covers every core or all but one.

To choose a count, run `python -m polystep.benchmarks.threads`, which times that
step at several thread counts in separate processes and prints the fastest, or time
your own workload, then call `torch.set_num_threads` before building the optimizer.
`OMP_WAIT_POLICY=PASSIVE`, set before PyTorch starts, keeps idle threads from
spinning; the experiment runners set it.

## Benchmarks

Run from the repository root, one timing job at a time:

```bash
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --profile
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --prefix --repeats 7 --warmup 2
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --attention --batch 16
PYTHONPATH=src:. python experiments/scripts/bench_forward_backends.py --compile --profile
```

Use `--device cuda` for GPU measurements and `--output` to keep separate result files.
Forward benchmarks check loss parity and report cold time, median, IQR, and executed
operators. GPU peak allocation is measured; CPU allocation is left unreported. The
[search benchmarks](../experiments/README.md#search-benchmarks) measure search
quality rather than speed.
