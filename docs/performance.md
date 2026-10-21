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
| Linear MLP, full space | Sparse delta | Automatic after registration |
| Linear MLP, hybrid subspace | Subspace delta | Automatic after registration |
| Linear MLP, factored subspace | Subspace delta, low-rank correction as fallback | Automatic after registration; on CUDA the low-rank correction when one chunk holds every candidate or chunks are small |
| Any model, `AdaptiveSubspace` on CPU | Incumbent plus the moved particle's projection columns | Automatic with a dense FP32 or FP64 projection |
| Convolution, attention, other custom forwards | Site-local vmap | Automatic when supported |
| Linear MLP, dense candidates | Batched matrix multiplication | Automatic |
| Other models | vmap or sequential fallback | Automatic |
| Models too large for batched weight copies | In-place parameter swaps | Automatic on CUDA when eight weight copies would take more than half the free memory; `NNCostEvaluator(..., use_inplace=True)` forces it |

A Linear MLP is an `nn.Sequential`, or a module whose forward chains Linear layers,
the common activations and `x.flatten(1)`, one call feeding the next. Activations count
as modules or as `torch.relu`, `F.gelu` and similar calls with default arguments. Hooks,
buffers, residual connections and other operations keep a custom forward on the
site-local or vmap path. Tied weights, such as an output layer that reuses the
embedding table, keep the site-local path too: a candidate for the tied weight reaches
every module that reads it.

On the site-local path on CPU, a candidate that moves an FP32 or FP64 weight with two or
more dimensions and at least 512 elements does not get its own copy of that weight. Each
`F.linear` or ungrouped convolution that reads the weight with an input shared by all
candidates adds the candidate's output change to the shared output. Any other read,
including one whose input already depends on the candidate, as in the later steps of a
recurrent loop, uses the candidate's weight, built once per forward, and so does every
read after it. A forward that opens its own `torch.vmap` or other `torch.func` transform
gets a full copy per candidate. So does a weight that `nn.MultiheadAttention` reads, or
whose change reaches its input, as in `nn.TransformerEncoderLayer`: intercepting calls
turns off PyTorch's fused attention, and `torch.vmap` runs the unfused CPU attention one
candidate at a time. The added change rounds differently from a built weight,
so FP32 losses can differ in the last bits. On one CPU thread this made a step 3.5 times
faster for a residual 784-256-256-10 MLP at batch 64 and 3.7 times faster for a LeNet-5
at batch 32. CUDA keeps a full copy per candidate. There, intercepting each call of the
forward costs a few microseconds that a launch-bound step pays in full: on an RTX 5090
the residual MLP gained only 3%, while a LeNet-5 and a 20-step RNN with hidden size 64
took about 1.3 times as long per step. Biases, weights below that size, FP16/BF16
weights and `compile_vmap=True` keep a full copy too; for a small weight the copy costs
less than intercepting every call of the forward.

The Linear MLP paths need a builtin loss: `nn.CrossEntropyLoss`, `nn.MSELoss` or
`nn.L1Loss` with mean reduction, or `F.cross_entropy`, `F.mse_loss` or `F.l1_loss`.
Cross-entropy may carry class weights, `ignore_index` and label smoothing. Any other
loss, including a lambda that calls one of these, takes the site-local or vmap path.

The delta and factored paths treat each sample as one feature vector, so they need
two-dimensional inputs or a leading `nn.Flatten` or `x.flatten(1)`. A Linear MLP
applied per token to `(batch, tokens, features)` inputs takes the site-local path
instead in full-space and hybrid runs; factored runs materialize its candidates.

`NNCostEvaluator(model, loss_fn, use_inplace=True)` runs the model's own forward,
one candidate at a time, and turns off the vmap, delta and factored paths. Each
sweep writes the incumbent into the model once; a full-space or `HybridSubspace`
candidate that changes one parameter then rewrites only that parameter, and in full
space only its own entries. Tied weights share storage, so they move together. The
registered evaluator's loss scores these candidates. Candidates that span two
parameters, FP16/BF16 hybrid weights and non-contiguous parameters swap in every
weight instead, as do screened hybrid candidates of
[small steps](api_overview.md#evaluation-controls), the hybrid incumbent and models
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
an activation or runs a stock `nn.GRU`, falls back to a sequential loop with a
warning. When the error names no functorch cause, the evaluator first scores one
candidate unbatched, so a forward that fails on its own raises its own error. On the
site-local path the fallback is per parameter: a parameter whose site vmap refuses,
such as a shortcut added in place with `+=`, moves to the dense path, and every other
parameter stays site-local. Buffers the forward writes are restored after every
batched call and after every looped candidate, so the retry, the probe, the loop and
later calls all start from the same buffer values.

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
belong to. Runs that span two parameters, screened candidates of small steps and the
incumbent use the full low-rank correction, which computes `x @ W.T` and `x @ B.T`
for the first perturbed layer once per candidate sweep and reuses them across its
chunks. Screening and incumbent scoring are separate sweeps. The search is limited to
the input directions spanned by `B`; basis rotation changes that search space.

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
perturbed parameter alone. The full-space delta path on CUDA counts activations only,
since it never builds the parameter, so its chunks are larger and its peak memory
higher: 397 MiB instead of 163 MiB for 784-256-256-10 at batch 256 on an RTX 5090. A
candidate's loss can depend on the chunk shape, so on that model the larger chunks
change a few losses in the last float32 bit. The CPU keeps the parameter in the budget,
because smaller chunks run faster there. A site-local path batches only the layers from
the perturbed one on, so each site is budgeted on the module outputs from its own layer
on. A forward that draws random numbers is batched from the draw on, so its sites keep
the whole-model budget, as does a parameter the forward reads before calling its layer.
In example 10's LeNet-5 at batch 256, a chunk holds 27 candidates on the first
convolution, whose output fills the budget, 72 on the second and up to 729 on the fully
connected layers, so a step scores its 4,833 candidates in 36 forwards. Reduce
`chunk_size` after an out-of-memory error.

Steps that score on a delta or site-local path, or on CPU `AdaptiveSubspace` offsets,
keep whole particles in each chunk, every vertex and probe of one particle, so
`chunk_size` drops to a multiple of the vertex count times `num_probe` and never goes
below one such group. A particle that straddles two parameters gets a chunk of its
own. With `HybridSubspace`, FP32 or FP64 weights and no in-place evaluator, that chunk
builds only the parameters the particle moves, as the incumbent plus the touched
projection columns, and passes the rest as expanded views of the incumbent, so the vmap
forward runs the layers ahead of the first moved parameter once. Otherwise the chunk
falls back to materialization, in batches of at most `chunk_size`. So does the
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

On CUDA the site-local path replays its forward from CUDA graphs unless the evaluator
or the optimizer gets `compile_forward=False`. Each parameter, chunk shape and batch shape runs eagerly
once, is captured on its second call and replays after that, up to 64 graphs. This
helps launch-bound models: example 11 steps about 1.15x to 1.35x faster, while LeNet-5 in
example 10 is unchanged. The graphs share one memory pool sized to the largest chunk's
activations and one copy of the weights and buffers. Each graph also holds its own copy
of its candidate stack and batch, so memory grows with the number of graphs and can far
exceed the weights. The graphs added 32 MiB allocated and 78 MiB reserved on example 11
and about 120 MiB and 410 MiB on example 10. Widened to `d_model=256, d_ff=1024`,
example 11 held 266 MiB of candidate stacks across 21 graphs for 3 MiB of weights. Capture
freezes the Python-side state the forward reads, such as a Python counter, a flag or a
reassigned tensor attribute. A forward that draws random numbers stays eager, and one
that syncs with the host fails capture, warns once and keeps that parameter eager.
`release_evaluator()` frees the graphs, so call it once after training rather than
every epoch.

CPU attention uses native scaled-dot-product attention. CUDA uses it too without a mask,
or with `is_causal=True` and no padding mask, outside FP16; other CUDA attention uses
explicit matmuls.
BF16/FP16 solver costs accumulate in FP32. The builtin cross-entropy, MSE and L1
losses, as modules or `torch.nn.functional` calls, also reduce half-precision candidate
outputs in FP32, whether they come from an FP16/BF16 model or from
`candidate_autocast`. A cross-entropy with class weights does so only on the Linear
MLP paths. Other losses keep their own dtype. Mixed-dtype models require
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
