# Performance

A step usually scores `P * V * K` candidates, where `P` is the particle count, `V`
the vertices per particle and `K` the probe radii per vertex.
Each particle holds `d` coordinates, `particle_dim` (default 2) in full space or
`subspace_particle_dim` (default 8) with a subspace. The simplex has `d + 1`
vertices, the orthoplex `2d`, and the cube `2^d`. The README quickstart searches 128
coordinates, so at the default `num_probe=1` a step scores 16 particles times the 9
simplex vertices, 144 candidates. The quadratic model adds one shared incumbent
evaluation when `biased_rotation` or `trust_region` uses it.

## Quick checklist

- Pick the thread count with `python -m polystep.benchmarks.threads`.
- Use `step_batch()` or `train()`, or register the evaluator for every batch.
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
| Convolution nets on CPU with large activation maps | In-place parameter swaps | Opt-in with `NNCostEvaluator(model, loss_fn, use_inplace=True)`; time a few steps both ways |

Here, a Linear MLP means an `nn.Sequential` or a custom forward that chains Linear
layers, common activations and optional `x.flatten(1)` calls. Each call feeds the next.
Activations may be modules or functions such as `torch.relu` and `F.gelu` with default
arguments. Hooks, buffers, residual connections and other operations keep a custom
forward on the site-local or vmap path. Tied weights, such as an output layer that
reuses the embedding table, keep the site-local path too: a candidate for the tied
weight reaches every module that reads it.

The Linear MLP paths need a builtin loss: `nn.CrossEntropyLoss`, `nn.MSELoss` or
`nn.L1Loss` with mean reduction, or `F.cross_entropy`, `F.mse_loss` or `F.l1_loss`.
Cross-entropy may carry class weights, `ignore_index` and label smoothing. Any other
loss, including a lambda that calls one of these, takes the site-local or vmap path.

On CPU, vmap evaluates unweighted, unsmoothed mean cross-entropy on `(batch, classes)`
logits with `log_softmax` and indexed reduction. Ignored targets keep PyTorch's mean
semantics, including NaN when every target is ignored. Other shapes and loss settings
use the original loss callable.

The delta and factored paths treat each sample as one feature vector, so they need
two-dimensional inputs or a leading `nn.Flatten` or `x.flatten(1)`. A Linear MLP
applied per token to `(batch, tokens, features)` inputs takes the site-local path
instead in full-space and hybrid runs; factored runs materialize its candidates.

Delta paths reuse base activations within each candidate sweep. Reuse applies to
supported builtin layers and ends before the next batch or fidelity level. Custom
layers can declare the contracts in the [API reference](api_overview.md#custom-layers).

Dense MLP evaluation also shares activations until the first changed layer when
unchanged candidate parameters are expanded views. Shared weights use one linear
operation across candidates. `chunk_size` bounds activation batches on this path.

`step_batch()` and `train()` register each batch for these paths:

```python
optimizer.step_batch(inputs, targets, loss_fn=loss_fn)
```

With an evaluator you built yourself, step with the closure its registration returns:
`optimizer.step(optimizer.register_evaluator(evaluator, inputs, targets))`.

The [API reference](api_overview.md#optimizer) covers the closure contract and what
to call after replacing layers, parameters or buffers.

### Output-delta injection on CPU

In full space on the CPU, the site-local path can skip a candidate's copy of an FP32 or
FP64 weight with two or more dimensions. Each `F.linear` or ungrouped convolution that
reads the weight with an input shared by all candidates adds the candidate's output
change to the shared output instead. On one CPU thread this made a step 3.5 times
faster for a residual 784-256-256-10 MLP at batch 64 and 3.7 times faster for a
LeNet-5 at batch 32.

A weight keeps a full copy per candidate when:

- Another kind of read uses it, or a read's input already depends on the candidate,
  as in a recurrent loop. That read and all later reads use the candidate's weight,
  built once per forward.
- The forward opens its own `torch.vmap` or other `torch.func` transform.
- `nn.MultiheadAttention` reads it, or its change reaches that module's input, as in
  `nn.TransformerEncoderLayer`. Interception turns off PyTorch's fused attention,
  so vmap runs the CPU attention one candidate at a time.
- No such `F.linear` or convolution reads it, or the read runs under autocast in a
  lower precision. Examples include an embedding table and a weight read through `.t()`.
- The model holds a TorchScript or traced module or function, as a submodule or as an
  instance or class attribute.
- It is a bias or an FP16/BF16 weight, or `compile_vmap=True`.

A parity check tests for reads that escape interception, such as a module-level
scripted function, a scripted module kept in a list, a C++ extension or a read under
`torch._C.DisableTorchFunction`. For each input shape and dtype, the first chunk to
move a coordinate contributes that change to a combined candidate. The evaluator
scores this candidate and the incumbent both ways: with injected output changes and
with reconstructed weights, starting from the same random state.

The weight keeps the injection path if losses agree within 64 machine epsilons times
`max(abs(loss), 1)`, or within 1% of the combined candidate's loss change. A mismatch
disables injection for that weight for the rest of the run. A finite loss on only one
path counts as a mismatch. If both are nonfinite, the chunk uses reconstructed weights
and the check is retried later. Differences below the tolerance can go undetected;
FP32 losses can still differ in their last bits.

Intercepting a call costs about 3 microseconds on one CPU thread, roughly the cost of
150,000 multiply-adds in a batched matmul. Injection runs when
`chunk_candidates * batch_size * weight_elements >= 150000 * (forward_calls + 30)`.
Without `chunk_size`, the chunk includes all candidates; scalar inputs use batch size
1. The extra 30 calls account for setup.

The evaluator uses the call count from the previous forward on this path. On first
use, the parity check measures it and selects reconstructed weights if injection
would cost more. A resumed run therefore selects the same path as an uninterrupted
one. CUDA always reconstructs the weights. On an RTX 5090, injection improved the
residual MLP by only 3%. It made LeNet-5 and a 20-step RNN with hidden size 64 take
about 1.3 times as long per step.

### In-place evaluation

`NNCostEvaluator(model, loss_fn, use_inplace=True)` runs the model's own forward,
one candidate at a time, and turns off the vmap, delta and factored paths. Each
sweep writes the incumbent into the model once. A full-space or `HybridSubspace`
candidate that changes one parameter then rewrites only that parameter.
In full space, it rewrites only its own entries. Tied weights share storage, so they
move together. The registered evaluator's loss scores these candidates.

The evaluator swaps in every weight for candidates spanning two parameters,
FP16/BF16 hybrid weights, non-contiguous parameters, the hybrid incumbent and screened
hybrid candidates below the [fast-path threshold](api_overview.md#evaluation-controls).
Models with a `max_norm` embedding also swap every weight, since their forward
renormalizes the rows it reads.

When the evaluator selects in-place evaluation automatically to save memory, it
continues to use the site-local, delta and factored paths where supported. Remaining
candidates are swapped in one at a time. In this automatic mode, the single-parameter
rewrite applies only to tied-weight models.

On the CPU, batching candidates multiplies every activation map by the chunk size, so
a net that keeps full-resolution maps spends much of its step moving them through
memory. On one thread, a two-block ResNet with 16 channels on 16x16 inputs stepped 1.4
times faster with `use_inplace=True` at batch 32 and 128. The same option made LeNet-5
take 7.6 times as long and a pooled three-convolution CIFAR net 1.6 times as long.
On CUDA, the ResNet took 2.2 times as long in place.

### Random and untraceable forwards

A forward that draws its own random numbers, such as injected noise, stays batched.
vmap refuses the draw at first, so the evaluator retries once with
`randomness="different"`. Each candidate then gets its own draw: identical
candidates can score differently, and a seeded run's stream differs from scoring
them one at a time.

A forward that vmap cannot batch, such as one that calls `.item()` on an activation
or runs a stock `nn.GRU`, falls back to a sequential loop with a warning. If an error
does not identify a functorch limitation, the evaluator tries one candidate unbatched.
A failure there raises the model's own error. Site-local fallback applies per
parameter: if vmap rejects one site, for example because of an in-place `+=`
shortcut, that parameter moves to the dense path. Other parameters stay site-local.

Buffers written by the forward are restored after each batched chunk and looped
candidate. Embedding weights renormalized by `max_norm` are restored too, including
frozen weights. See [evaluation limits](../LIMITATIONS.md#evaluation) for writes that
bypass tensor version tracking.

## Subspaces and memory

Use `HybridSubspace.from_layout(..., max_subspace_dim=...)` to bound candidate count;
rank alone does not bound the total dimension. The cap shrinks every weight matrix in
proportion, while biases and other 1D parameters keep their full width; a cap below
that width warns and leaves each matrix one coordinate. Smaller subspaces reduce cost
but restrict the search. Retune the radii when changing the representation.
Random basis rotation is optional for `HybridSubspace`. Set `rotation_mode="random"`
with a positive `rotation_interval` to use it. On MNIST MLPs it measured between
no change and 4.8 points more test accuracy, at up to 1.7x the step time.

### FactoredSubspace

`FactoredSubspace` uses `dW = A @ B`, so sequential linear layers can
compute `x @ W.T + (x @ B.T) @ A.T` without constructing candidate weights. A
candidate changes a few consecutive coordinates of one `A`, so the subspace-delta
path scores it from `x @ B.T` and moves only the output units those coordinates
belong to. Runs that span two parameters, screened candidates of small steps and the
incumbent use the full low-rank correction. This computes `x @ W.T` and `x @ B.T`
for the first perturbed layer once per candidate sweep. It reuses them across the
chunks. Screening and incumbent scoring are separate sweeps.

The search is limited to the input directions spanned by the rows of `B`. A fixed `B`
keeps the whole run in one row space, so `FactoredSubspace` redraws `B` every
`rotation_interval` OT steps, 5 by default. On a 784-64-10 MNIST MLP at rank 8 and 300
steps, rotating every 5 steps raised test accuracy from 51% to 73% at equal time. A
rotation clears momentum, the `amortize_steps` coasting direction and the
`biased_rotation` direction, so at the default none of them lasts more than 5 OT
steps. Under amortization each rotation costs one extra full OT step: at
`amortize_steps=5` the same run gained 12 points for 19% more time. `B` is drawn on
the CPU so that a seed gives the same basis on every device. On CUDA the redraws added
about 1.5% to the mean step time of the same MLP. Pass `rotation_interval=0` to keep
`B` fixed.

The delta path scores each parameter in its own pass and launches more kernels per
chunk than the low-rank correction. On CUDA, that overhead outweighs the savings for
small chunks. The delta path runs only when both conditions hold:

- The default chunk budget cannot hold every candidate.
- `chunk_size` times a candidate's largest layer output reaches `2**24` elements
  (about 16.8 million). If `chunk_size` is unset, use the candidate count.

Other factored CUDA steps use the low-rank correction, which needs more peak memory
per chunk. `HybridSubspace` has no low-rank correction, so this rule does not apply.

The threshold is where the delta path stops losing on shallow MLPs (784-128-10 at
batch 512 on an RTX 5090). On that GPU, the rule selects delta steps that ran up to
5 times faster at the default chunk size. Close to the threshold, the gain was only
a few percent. Peak memory was 4 to 54 times lower. A batch of one on a 784-512-512-10 or
3072-1024-10 MLP stays below the threshold, and there the delta path ran up to 1.35
times slower. Below the threshold the low-rank correction ran up to 1.4 times faster
on small or shallow MLPs. Wider or deeper models and larger batches favor the delta
path from smaller chunks onward. If `chunk_size` is set, a larger value can make
these models step faster.

### AdaptiveSubspace

`AdaptiveSubspace` still builds every candidate's full weights, since its single
projection `P` spreads each coordinate over every parameter. On the CPU, with a dense
`P`, it builds them as the incumbent plus `P[:, cols] @ delta`.
Here, `cols` are the coordinates of the one particle the candidate moves.
Each candidate reads `subspace_particle_dim` columns of `P` instead of all of them.
The padded last particle, screened candidates and the incumbent use the full
`base + P @ x`, as do
sparse projections and FP16/BF16 runs.

When `P` is dense FP32 or FP64 and screening is off, the optimizer reuses one buffer
across chunks and steps, including the padded particle and incumbent. This avoids
page faults from fresh allocations. A closure must
[clone any candidate tensor it keeps](../LIMITATIONS.md#evaluation). CUDA computes the
full product for every candidate: one dense matrix multiplication measured faster
there than the many thin batched products needed for coordinate offsets.

### Chunk sizes

`chunk_size` limits candidates held at once. The automatic estimate counts the
candidate's parameters and its largest layer output, which one forward measures when
the evaluator is first registered and which then scales with the batch. Delta and
site-local paths build no full candidate, so they are budgeted on activations and the
perturbed parameter alone: 2^26 elements, counting two activations and the parameter.

On the CPU, subspace site-local and subspace-delta chunks use a 16 MiB budget when
one chunk activation would exceed 32 MiB. This is glibc's largest mmap threshold.
Above it, each allocation maps fresh pages that fault in zeroed. Example 11's
transformer keeps its chunks, whose activations stay under 32 MiB. Full-space delta and site-local
chunks keep the element budget on every device. The delta estimate counts more than
it allocates, and smaller chunks measured slower there. On the site-local path,
smaller chunks can fall below the per-call size the output-delta injection needs,
which turned it off for a LeNet-5 convolution and for an LSTM's recurrent weight.
Smaller chunks also free only small mapped blocks. Glibc raises its mmap threshold
to the largest block freed, so the later fully connected sites fault on every chunk.
A full-space LeNet-5 step took 17% to 49% longer at batch 32 and 128.

On CUDA the full-space delta path counts activations only, since it never builds the
parameter. This gives larger chunks and higher peak memory. For 784-256-256-10 at
batch 256 on an RTX 5090, peak memory was 397 MiB instead of 163 MiB. A candidate's
loss can depend on the chunk shape, so on that model the larger chunks change a few losses in the last
float32 bit. The CPU keeps the parameter in the budget, because there the larger
chunks stepped at about half the speed.

A site-local path batches only the layers from the perturbed one on, so each site is
budgeted on the module outputs from its own layer on. A forward that draws random
numbers is batched from the draw on, so its sites keep the whole-model budget, as does
a parameter the forward reads before calling its layer. In example 10's LeNet-5 at
batch 256 on CUDA, a chunk holds 27 candidates on the first convolution. It holds
72 on the second and up to 729 on the fully connected layers. A step scores its
4,833 candidates in 36 forwards. On the CPU a chunk holds 9 candidates on both
convolutions. The first two fully connected layers hold 36 and 45, and the last holds
up to 414. A step takes 197 forwards. Reduce `chunk_size` after an out-of-memory
error.

The delta, site-local and CPU `AdaptiveSubspace` offset paths keep each particle's
vertices and probes together. They round `chunk_size` down to a multiple of the
vertex count times `num_probe`, with at least one such group per chunk.
A particle that straddles two parameters gets its own chunk. With `HybridSubspace`,
FP32 or FP64 weights and no in-place evaluator, this chunk builds only the parameters
the particle moves. It adds the touched projection columns to the incumbent and
passes other parameters as expanded views. The vmap forward then runs the layers
ahead of the first moved parameter once. Otherwise the
chunk falls back to materialization, in batches of at most `chunk_size`. So does the
padded particle at the end of the CPU offsets, or of a per-layer or grouped block
other than the last. On the delta and site-local paths the padded last particle stays
with the particles before it, since its padding moves no parameter.

Each chunk costs one forward. Small CUDA models can be limited by kernel launches.
They can step faster with a `chunk_size` at or above the candidate count than with
the automatic one. Per-layer blocks reduce the size of each OT solve, but
not the forward count. With a subspace, a block strategy only splits Sinkhorn and KL
solves into at most eight row segments; candidates are scored exactly as in the
monolithic step.

## Compilation and precision

The [API reference](api_overview.md#evaluation-controls) lists the compilation and
autocast options. Measure compilation separately from steady-state execution. Each
site compiles independently and falls back to eager execution on compilation failure.
Profiler output confirms whether a compiled region actually ran. PyTorch's
recompilation limit can cause later calls to run eagerly.

On CUDA the site-local path replays its forward from CUDA graphs unless the evaluator
or the optimizer gets `compile_forward=False`. Each parameter, chunk shape and batch
shape runs eagerly once, is captured on its second call and replays after that, up to
64 graphs. This helps launch-bound models: on an RTX 5090 example 11 steps about 1.7
times faster, while LeNet-5 in example 10 is unchanged. The graphs share one memory
pool sized to the largest chunk's activations and one copy of the weights and buffers.
Each graph also holds its own copy of its candidate stack and batch, so memory grows
with the number of graphs and can far exceed the weights. The graphs added 34 MiB
allocated and 42 MiB reserved on example 11. They added 280 MiB allocated and 410 MiB
reserved on example 10, whose 12 graphs hold 240 MiB of candidate stacks. Widened to
`d_model=256, d_ff=1024`, example 11 held 266 MiB of candidate stacks across 21 graphs
for 3 MiB of weights.
Capture freezes the Python-side state the forward reads, such as a Python counter, a
flag or a reassigned tensor attribute. A forward that draws random numbers stays eager,
and one that syncs with the host fails capture, warns once and keeps that parameter
eager. `release_evaluator()` frees the graphs, so call it once after training rather
than every epoch.

CPU attention uses native scaled-dot-product attention. Outside vmap, CUDA uses it too
without a mask, or with `is_causal=True` and no padding mask, outside FP16. Other CUDA
attention, including every candidate batch scored under vmap, uses explicit matmuls,
because the fused CUDA kernel fails under vmap once candidates times batch reaches
65,536.

FP16 and BF16 solver costs accumulate in FP32. The builtin cross-entropy, MSE and L1
losses, as modules or `torch.nn.functional` calls, also reduce half-precision
candidate outputs in FP32, whether they come from an FP16/BF16 model or from
`candidate_autocast`. A cross-entropy with class weights does so only on the Linear
MLP paths. Other losses keep their own dtype. Mixed-dtype models require per-entry
subspaces; see [limitations](../LIMITATIONS.md).

With FP16 or BF16 parameters the subspace-delta and factored paths are off. They add
each candidate's change after the matmul, so they would rank changes that the stored
weights round away. Hybrid candidates use the site-local path. It builds the changed
and shared weights from the base weights, as it would for a materialized candidate.
Factored candidates are materialized. The weights
swapped into the model under `use_inplace=True`, and the ones a step writes back
when it ends, round the same way. A basis rotation, absorb or rank change folds the
subspace coordinates into the base weights. With FP16 or BF16 weights, the subspace
saves each weight's rounding error and adds it back at the next fold. This lets
updates smaller than the weight's precision accumulate.

The delta, site-local and vmap paths add each Linear's bias after the matmul.
The model's own Linear adds it inside the matmul and rounds once. With FP16 or BF16
weights, outputs can differ by about one ulp of the matmul result or the output,
whichever is larger. When the bias cancels most of that
result, the difference is large next to the output, and later layers can amplify it. A
hard threshold, such as a sign or a spike, flips when its pre-activation lies that
close to its cut. A smooth layer that scales up a cancelled output can also change
the loss. `use_inplace=True` scores with the model's own forward. On small sign,
ReLU and spiking models it took 11 to 118 times the step time and made no measurable
difference to training loss or accuracy.

## CPU measurements

These measurements use PyTorch 2.13 and one efficiency-core thread of an Intel Core
Ultra 9 285K. The workload is a complete step of a 32-w-w-4 MLP at batch 32, with
`solver="softmax"`, `chunk_size=72` and a registered evaluator. Each of five seeds
runs twice. Each run has one cold step, two warmup steps and seven timed steps; the
table reports the median of the ten per-run medians. The subspace row uses a
`HybridSubspace` at rank 4 capped at 1,024 coordinates:

| Hidden width `w` | Search | Without prefix reuse | With reuse |
|---|---|---:|---:|
| 64 | Full space | 34.50 ms | 31.69 ms |
| 256 | Full space | 493.15 ms | 368.27 ms |
| 256 | 1,024 coordinates | 57.14 ms | 56.36 ms |

With the automatic chunk size, the full-space steps with reuse take about 43 ms at
width 64 and 550 ms at width 256, and the subspace step about 60 ms. On an RTX 5090
the automatic chunk size brings the same three steps to 2.8 ms, 10.6 ms and 5.6 ms.
With `chunk_size=72`, the full-space steps take 40 ms and 426 ms, with one small
forward per 72 candidates.

Attention kernel medians over three runs, 16 candidates and batch 16:

| Sequence length | Explicit attention | Native CPU SDPA |
|---|---:|---:|
| 6 | 0.440 ms | 0.396 ms |
| 64 | 13.604 ms | 3.423 ms |
| 128 | 54.022 ms | 11.216 ms |

These kernel timings do not measure complete optimizer steps. GPU results depend on
device, shapes, and backend selection.

## INT8 policy evaluation

Version 1.1.1 removes host scalar checks from the RL INT8 activation. This lets
`NNCostEvaluator` batch these policies with vmap. Previously they fell back to one
forward per candidate. For a 4-64-2 `NonDiffMLPPolicy`, 64 candidates, batch 32,
cross-entropy and `compile_forward=False`, warm evaluation times were:

| Device | 1.1.0 | 1.1.1 |
|---|---:|---:|
| CPU, one thread | 2.36 ms | 0.36 ms |
| RTX 5090 | 148.87 ms | 0.35 ms |

These are averages of two run medians under PyTorch 2.13, with run order reversed.
The baseline is commit `353db52`, which reports version 1.1.0.
Each run used three warmup calls and at least two seconds of timing with
`torch.utils.benchmark.Timer`. Seed 42 and perturbation seed 42 were fixed; candidate
noise had standard deviation 0.1. Both paths matched individual policy forwards.

The stacked rollout evaluator already batched its candidates. Its fused affine
operation reduced a CPU tanh-policy rollout from 361 ms to 141 ms at hidden width
128, 256 candidates, 16 rollouts each and horizon 200. Smaller tanh policies changed
little. Matching the public INT8 policy's scale floor and nonfinite handling added
about 7-17% to the smaller CPU INT8 rollouts tested. The vmap timing above does not
predict the speed of every RL workload.

## CPU threads

PyTorch's default intra-op thread pool can be too large for small steps, especially
when other processes share the CPU. I measured this on eight efficiency cores of
the machine above. The README quickstart takes about 1.6 ms per step on one thread
and slightly longer on four or eight. The full-space 784-64-10 workload in
`python -m polystep.benchmarks.threads` drops from 104 ms on one thread to 63 ms on four
and 54 ms on eight. Other jobs occupied the performance cores during these runs.

The fast test suite and most examples default to one thread;
[CONTRIBUTING](../CONTRIBUTING.md#tests) and the [examples](../examples/README.md) list
the exceptions. The optimizer warns once when the pool covers every core or all but
one.

To choose a count, run `python -m polystep.benchmarks.threads` or time your own workload.
The command times that step at several thread counts in separate processes and
prints the fastest. Call `torch.set_num_threads` before building the optimizer.
`OMP_WAIT_POLICY=PASSIVE`, set before PyTorch starts, keeps idle threads from
spinning; the experiment runners set it.

## Benchmarks

I run one timing job at a time from the repository root:

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

For custom timing loops, use
[`torch.utils.benchmark.Timer`](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
or synchronize CUDA before and after a timed region. Warm up the workload first.
Compare medians and variability, and reverse the run order to check for clock or
load changes. Keep the seed, candidate count, thread count and precision fixed.

For related approaches, [MeZO](https://arxiv.org/abs/2305.17333) uses in-place
perturbations to keep memory near inference cost, while
[EGGROLL](https://arxiv.org/abs/2511.16652) uses low-rank perturbations to make large
candidate batches cheaper. These address different costs: fewer stored weights do
not necessarily mean faster steps. Compare complete steps, peak memory and attained
loss at matched evaluation and time budgets when choosing an evaluation path.

[ZO-Muon](https://arxiv.org/abs/2602.17155) combines subspace gradient estimation
with gradient orthogonalization. [EGGROLL, Unrolled](https://arxiv.org/abs/2609.10980)
analyzes how low-rank perturbations change the update at finite radius and proposes
a leave-one-out estimator. Compare these methods at matched budgets. Their results
do not establish that the same changes improve PolyStep's transport update.
