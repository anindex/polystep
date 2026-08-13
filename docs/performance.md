# Per-step cost

PolyStep runs `P*V*K` forward passes per step (`P` particles x `V` polytope vertices x
`K` probe radii). Measure wall-clock, not forward count: on small nets the cost is
launch/dispatch overhead, not FLOPs.

Timings throughout are from one box (RTX 5090, 24-core CPU). Read the ratios, not the
absolute numbers.

## Which path runs

| model | path | flag |
|---|---|---|
| MLP + `HybridSubspace` + evaluator | `SubspaceDeltaEvaluator` | automatic |
| MLP full space + evaluator | `SparseDeltaEvaluator` | automatic |
| pure `nn.Sequential` MLP (vanilla CE) | `bmm` fast path | automatic |
| conv / attention / custom forward | `SiteVmapEvaluator` | automatic |
| tied weights, fits in memory | `vmap` | `compile_evaluator=True` |
| large net / vmap OOMs | in-place swap | `compile_forward=True` |

Every site-aware path needs `register_evaluator`; without it the step reaches the model
only through `closure()` and materializes every candidate. `api.train()` registers on
every batch. Block-wise mode selects from the same table.

## The evaluator's loss is the objective on every fast chunk

A site-aware chunk scores `evaluator.loss_fn`; a chunk that falls back scores
`closure()`. Give both the same objective. A closure that adds a regularizer the
evaluator does not know about makes one step rank part of its candidates on a different
function, with no error and no shape mismatch.

## Which polytope pays for itself

The simplex is `k+1` vertices, the orthoplex `2k`. Only the orthoplex has antithetic
`+/-` pairs, and only `newton_refinement` still needs them. The quadratic model and
`trust_region` run on any template, because every one of them is a centred tight frame.

At a fixed budget of 300K candidate evaluations, 20-32-3 MLP, 6 seeds, mean loss
reduction (`experiments/scripts/bench_polytope.py`):

| config | loss reduction | steps |
|---|---|---|
| `simplex` (default) | 0.079 | 260 |
| `orthoplex` alone | 0.066 | 195 |
| `orthoplex` + `multifidelity_screen` | 0.064 | 260 |
| `orthoplex` + `use_quadratic_model` + `trust_region`, `num_probe=2` | 0.156 | 173 |
| `orthoplex` + `use_quadratic_model` + `trust_region`, `num_probe=1` | 0.140 | 195 |
| `simplex` + `use_quadratic_model` + `trust_region`, `num_probe=1` | **0.178** | 259 |

The orthoplex alone loses. The quadratic model is what pays, and it pays most on the
cheaper polytope: 2.25x per forward pass against the plain simplex, on `k+1` vertices
instead of `2k`. The configuration to reach for:

```python
PolyStepOptimizer(model, use_quadratic_model=True, trust_region=True, num_probe=1)
```

with the default `polytope_type='simplex'`. Use the orthoplex only for
`newton_refinement`, which reads the antithetic pairs directly.

The trust region scores the previous step's prediction against this step's loss, so it
needs both from the same objective, as in the benchmark above: one fixed batch and
`amortize_steps=1`. Two setups take it out:

- **Amortization.** A momentum step predicts nothing and drops the pending comparison, so
  the next OT step has none to score. The constructor warns. Only the shared `f(X)`
  remains, at +6% wall-clock on `examples/10_cnn_mnist.py` for no accuracy change.
- **A minibatch stream.** Without `objective_token` the ratio compares a prediction made
  on one batch against a loss measured on the next, the noise reads as a bad step every
  time, and the radius sits at `radius_min`. On `examples/02_snn_starter.py` that is
  100% against 68.8%. Passing `objective_token` per batch makes the step decline the
  stale comparison instead, and the run is unchanged.

So it pays on a stationary objective (a fixed dataset, a simulator with a fixed seed, a
combinatorial instance) and does nothing for minibatch SGD-style loops.

## Why the model runs on any polytope

Every template is centred, unit-radius and tight: `sum_v v = 0` and
`sum_v v v^T = (V/d) I`, to 1e-15. The least-squares affine fit is then a closed form:

```text
g       = (d / (V s r)) sum_v L_v v
tr(H)/d = 2 (mean_v L_v - f(X)) / (s r)^2
```

On antipodal pairs `g` reduces to the central difference, and the code takes that form,
so no orthoplex result moves bit for bit. Elsewhere the third moment survives and `g`
carries an `O(r)` bias: 0.67% at `r = 0.01` on the simplex, 6.7% at `r = 0.1`. The curvature is exact at any radius, and it is one
number instead of `d`: the frame is Haar-random every step, so `E[H_jj] = tr(H)/d` for
every `j` and the orthoplex diagonal is `d` noisy estimates of it. Only
`newton_refinement` uses the anisotropy that costs.

The centre is evaluated only when `biased_rotation` or `trust_region` is on, and the
model is built under the same condition: `use_quadratic_model=True` alone computes
nothing. At `num_probe=1` the system is exactly determined, `d+1` equations for `d+1`
unknowns, at `P*V + 1` evaluations against `P*V*K`. The centre is one forward, not one
per particle: a centre candidate is `X` with a row rewritten to the value already there.
The trust-region ratio compares against that same `f(X)` instead of
`mean_i min_v C[i,v]`, which moved by 0.04 across 100 rotations at a fixed point. At
`num_probe >= 2` off the orthoplex there is no centre and no pairs, so the model has
nothing to read and the constructor says so.

## Where a subspace step spends its time

`NNCostEvaluator` evaluates all `N = P*V*K` particles in one vectorized `vmap` sweep
(`torch.func.functional_call` over the candidate axis). `HybridSubspace` then builds each
candidate's weights with a `(N, coords) @ (coords, num_params)` matmul. On a 203K MLP at
batch 256 that is 63% of the step:

| component | ms/step | share |
|---|---|---|
| `reconstruct_batch` | 355.6 | 62.7% |
| forward evaluation | 175.3 | 30.9% |
| everything else | 35.2 | 6.2% |

`SubspaceDeltaEvaluator` cuts it automatically; `FactoredSubspace` is opt-in and trades
accuracy for more.

## Not building the candidate: `SubspaceDeltaEvaluator`

A candidate is the barycenter with one particle row replaced, so its coordinates differ
in `subspace_particle_dim` consecutive entries. When that run sits inside one layer's
coordinate block only that layer's weight moves, and the perturbation is linear in those
coordinates:

```text
W_n = W_bary + reshape(sum_j d_nj * P[:, j])
x @ W_n.T  = x @ W_bary.T + sum_j d_nj * (x @ M_j.T),   M_j = reshape(P[:, j])
```

The `M_j` products are computed once per column and reused by every candidate sharing the
particle. No `(N, d_out, d_in)` tensor is built. For a `SparseRandomProjection` layer
each `M_j` has only `nnz_per_col` nonzeros, so the largest layer is the cheapest.

203K MLP, batch 256, `HybridSubspace(rank=8)`, RTX 5090:

| path | ms/step | peak |
|---|---|---|
| `reconstruct_batch` | 330.8 | 989 MB |
| `SubspaceDeltaEvaluator` | 75.5 | 795 MB |

4.4x, losses identical to five decimals. Needs a plain `nn.Sequential` of Linear and
activation layers and a vanilla mean CE/MSE/L1 loss. Site resolution is per chunk: a run
straddling two coordinate blocks, an unprojected full-width entry, or mixed precision
falls back for that chunk. On the model above 50 of 52 chunks take the fast path.

## In full space: `SparseDeltaEvaluator`

A full-space candidate is the base vector with exactly `particle_dim` scalars replaced,
so its output differs in at most `particle_dim` columns. The delta propagates through the
elementwise layers and goes dense one `Linear` after the perturbed one. Built
automatically for `nn.Sequential` of `Linear` and elementwise layers; falls back per
chunk on a straddling chunk, tied weights, or a layer that mixes features (`LayerNorm`,
`Softmax`, `Conv`).

Perturbed positions follow the particle, not the vertex, so the `V * num_probe`
candidates of one particle share an index row and the evaluator gathers once per group.
784-64-10 MLP, batch 256, `particle_dim=4`, CPU: 595 ms to a 368-410 ms band, cost matrix
unchanged. A screened chunk is not particle-major and degrades to per-candidate.

## Anything traceable: `SiteVmapEvaluator`

The same locality argument holds for any model: a candidate differs inside one parameter
tensor, so batching only that tensor makes the graph ahead of it run once.
`SiteVmapEvaluator` does this with `vmap(in_dims=...)`, needing nothing beyond what
`torch.func` can trace, so it covers conv, normalization, attention and hand-written
`forward`. It picks up the models the delta paths decline outright and the chunks they
decline individually. It declines tied weights.

It runs under `block_strategy='per_layer'` and `'grouped'` too, where the Linear-only
`SparseDeltaEvaluator` is the only other option: a small conv net went 38.6 to 1.0 s per
step (40x), with an identical cost matrix.

In subspace mode the correction for a chunk is `dcoords @ P.t()`. Only `particle_dim`
coordinates per candidate are nonzero, and they sit in one column block per particle
group, so it is a per-group `bmm` rather than a dense product against the whole layer
block. That is `spec_width / particle_dim` fewer FLOPs: 53-96x on the matmul at layer
widths 552 and 1064, and 2.4x on a whole ConvNet step at `subspace_dim=3630`. The gain
scales with `subspace_dim`, so at `max_subspace_dim=256` it is within noise.

## Chunks break where the parameters do

A site-aware path needs every candidate in a chunk to perturb the same parameter. Sized
purely by memory a chunk spans several entries, resolves to no site and falls back, so
the fast paths ran for a small fraction of a sweep. The chunk loop breaks at entry
boundaries (coordinate-block boundaries in subspace mode). CPU, batch 64: a 64-64-10 MLP
went 0.53 to 0.09 s over three steps (5.6x), a small conv net 90.8 to 4.9 s over two
(18.6x). Both match the materializing path to floating-point tolerance.

## Chunk sizing sets peak memory

`chunk` bounds how many candidates are evaluated at once, derived from what a step
allocates per candidate: the config tensor, a full weight set when a subspace runs
without the fused or factored path, and the vmap activations. Budgeting on the config
tensor alone under-counts the other two. 120,010-param MLP (64-1600-10), batch 128, CPU,
one step, peak RSS:

| mode | budget on configs only | budget on what is allocated |
|---|---|---|
| full space | 1661 MB, 43.6 s | 946 MB, 1.4 s |
| `HybridSubspace` | 10565 MB, 436.0 s | 1461 MB, 122.8 s |

Smaller chunks also raise sparse-delta coverage: 1163 of 1166 chunks on this model
against 427 of 430 at the larger chunk. Set `chunk_size` to override.

## A hand-written forward costs you every fast path

The batched evaluators rebuild their layer plan from `named_children()`, so they need
`type(model).forward is nn.Sequential.forward`. A model that is otherwise a plain MLP but
defines its own `forward` fails it and every candidate goes through `vmap`, with no error
and no output difference. 784-128-10 MLP, `HybridSubspace(rank=8)`, batch 512:

| model | CUDA ms/step | CPU ms/step |
|---|---|---|
| custom `nn.Module`, hand-written forward | 171.2 | 43412 |
| `nn.Sequential` subclass | 80.9 | 8106 |

2.1x on CUDA, 5.4x on CPU, same arithmetic. Subclass `nn.Sequential` and pass an
`OrderedDict` to keep the parameter names:

```python
class MNISTNet(nn.Sequential):
    def __init__(self, hidden=128):
        super().__init__(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(784, hidden)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(hidden, 10)),
        ]))
```

Wrapping a `nn.Sequential` in a module whose forward just calls it fails the same check.
The library warns once on this shape, and stays quiet when the children are not the whole
forward, where conversion would change the model rather than speed it up.

## Custom layers on the fast paths

`polystep_elementwise` and `polystep_weight_transform` put a custom layer on the batched
paths; [`api_overview.md`](api_overview.md) has the contracts and the syntax.

The delta path uses the effective delta `Q(w + d) - Q(w)`, not a correction linear in the
perturbation. The difference is nothing for a plain Linear and O(1) for a sign transform:

| path | max abs error vs true per-candidate forward |
|---|---|
| bmm, `Q` on the stacked weight | 4.8e-07 |
| delta, correction linear in the perturbation | **2.78** |
| delta, effective delta `Q(w + d) - Q(w)` | 4.8e-07 |

`SubspaceDeltaEvaluator` and `FactoredEvaluator` decline a declaring layer: their
correction never forms the perturbed weight, which is what they exist to avoid.

Batch 256 on CUDA, before and after the declarations:

| model | blocked by | before | after |
|---|---|---|---|
| `QuantizedMLP` | `QuantizedLinear` | 637 ms/step | 127 ms |
| `BinaryMNISTNet` | `BinaryLinear` | 514 ms/step | 111 ms |
| `StaircaseNet` | `StaircaseActivation` | 709 ms/step | 160 ms |

Out of reach either way: anything mixing across the feature axis (`HardMoENet`,
`SoftMoENet`, `DiscreteAttentionNet`), a stateful timestep loop (every spiking net), and
conv, attention and recurrent models.

## Trading accuracy for speed: `FactoredSubspace`

`FactoredSubspace` perturbs each 2D parameter by `dW = A @ B` with `B` fixed, so
`x (W + A B)^T = x W^T + (x B^T) A^T` and no candidate weight is built:

| subspace | dim | ms/step | peak memory |
|---|---|---|---|
| `HybridSubspace(rank=8)` | 10714 | 626.6 | 19.6 GB |
| `FactoredSubspace(rank=32)` | 8558 | 56.1 | 7.9 GB |

11.2x per step, 8.9x per forward pass. It is a trade, not a free win, and not the
default: `dW = A @ B` confines every perturbation to the `r` input directions `B` spans.
MNIST example, 3 epochs, matched dimension, same schedules:

| subspace | dim | time | best test acc |
|---|---|---|---|
| `HybridSubspace(rank=8)` | 8538 | 38.1 s | **90.87%** |
| `FactoredSubspace(rank=64)` | 8430 | 10.2 s | 81.36% |
| `FactoredSubspace(rank=64, rotation_interval=5)` | 8430 | 11.8 s | 86.74% |
| `FactoredSubspace(rank=64, rotation_interval=1)` | 8430 | 30.5 s | 88.79% |
| `FactoredSubspace(rank=128, rotation_interval=1)` | 16622 | 55.1 s | 90.85% |

`rotation_interval > 0` redraws `B` and recovers most of the accuracy, at an absorb and a
basis rebuild per rotation. Reach for it when a step is dominated by weight
materialization or when memory binds. Requires a plain `nn.Sequential` of Linear and
activation layers. See arXiv:2511.16652.

## Rotation sampling

Every step draws one Haar rotation per particle, and `subspace_particle_dim` defaults to
8, so subspace mode wants thousands of tiny `(d, d)` orthogonal matrices per step.
`get_random_rotation_matrices` picks between the Diaconis-Shahshahani subgroup algorithm
(`d-1` batched Householder reflections) and QR with a Mezzadri sign correction. The
reflections cost `d-1` sequential launches whatever the batch, and batched `geqrf` scales
with the batch, so each wins on one side of a crossover:

| shape | `torch.linalg.qr` | Householder | faster |
|---|---|---|---|
| (1, 8, 8) CPU | 0.007 ms | 0.148 ms | QR 23x |
| (1, 129, 129) CPU | 0.219 ms | 2.869 ms | QR 13x |
| (64, 8, 8) CPU | 0.056 ms | 0.190 ms | QR 3.4x |
| (12723, 8, 8) CPU | 12.2 ms | 3.3 ms | Householder 3.7x |
| (1, 129, 129) CUDA | 0.292 ms | 12.620 ms | QR 43x |
| (1067, 8, 8) CUDA | 18.6 ms | 0.69 ms | Householder 27x |
| (12723, 8, 8) CUDA | 219.9 ms | 0.66 ms | Householder 332x |

QR runs at or below batch 256 on CPU and 16 on CUDA (`geometry._QR_MAX_BATCH`, the last
batch it won at every `d` in 4, 8, 16), reflections above. The wide-batch path is worth
43.7 to 23.9 ms per step on a 784-128-10 MLP, `HybridSubspace(rank=8)`, batch 256; the
narrow-batch path 8 to 15% on examples 07, 08 and 09.

The QR branch runs single-threaded through `thin_qr` for the reason in the next section,
and pins the batched `det` with it: at (16, 8, 8) fp32 CPU the branch costs 0.09 ms on
one thread against 14.99 ms on 16.

Both are Haar on `SO(d)`, each pinned against the Mezzadri reference in
`tests/test_geometry.py`, but they consume the generator stream differently, so a seeded
run either side of the threshold is not comparable. Orthonormality costs the reflections
half a digit, 1.2e-6 against 2.4e-7 in FP32.

## Basis construction

Every subspace builds its projection with a QR of a tall-thin CPU matrix, where LAPACK
spreads the work over every core and the synchronization dominates: `qr` of a `(784, 64)`
fp32 CPU tensor measured 1602 ms on 24 threads against 0.30 ms on one.
`polystep.solvers._shared.thin_qr` pins to a single thread and restores the count on the
way out, which cut `HybridSubspace` setup for the MNIST model from 27.0 s to 97.8 ms.

## Fusing the per-layer projections

`build_fused_projection` block-diagonalizes the dense per-layer projections so
reconstruction is one matmul, at the cost of reading the zero padding per candidate.
Measured at N=64:

| dense blocks | fused size | verdict |
|---|---|---|
| 4 | 12.7 MB | fused 1.16x faster |
| 8 | 25.3 MB | fused 1.34x faster |
| 4 | 94.7 MB | fused 1.49x **slower** |
| 3 | 152.0 MB | fused 1.96x **slower** |

So the fuse is capped at 32 MB and a single dense block is skipped (`block_diag` of one
block is a copy, 1.01x, duplicating 5.7 to 21.8 MB). On the models in `examples/` the
largest layer goes to `SparseRandomProjection` and one dense block is left, so the fuse
declines.

## Spending step budget instead of forward passes: `amortize_steps`

`amortize_steps=k` runs `k-1` momentum steps between OT steps. A momentum step reuses the
EMA transport direction and evaluates no candidates, so the forward-pass budget drops by
roughly `k`. It is the largest single lever in `examples/`, and the one most easily
overdone: it does not save work, it spends step budget, so a run that was already
step-starved gets worse.

Measured at 40 epochs on `examples/10` (LeNet-5, 20k MNIST, batch 256) and 60 epochs on
`examples/11` (selective copy), two seeds each:

| model | config | time | accuracy |
|---|---|---|---|
| LeNet-5 | rank 8, `amortize_steps=3` | 95 s | 93.50 / 95.35 |
| LeNet-5 | rank 4, `amortize_steps=5` | 26 s | 95.00 / 95.25 |
| LeNet-5 | rank 2, `amortize_steps=5` | 13 s | 30.50 / 93.05 |
| LeNet-5 | rank 4, `amortize_steps=8` | 17 s | 77.05 / 95.10 |
| transformer | `amortize_steps=1`, batch 256 (420 steps) | 5.4 s | 100 / 100 |
| transformer | `amortize_steps=5`, batch 256 (420 steps) | 1.1 s | 93.60 / 74.30 |
| transformer | `amortize_steps=5`, batch 128 (900 steps) | 2.3 s | 100 / 100 |

Two things to read off it. The cliff is sharp and it is seed-dependent, so a single seed
will tell you a setting works when it does not. And when a model needs a step count,
halving the batch buys the steps back at no extra forward passes per epoch, which is what
makes the transformer row work.

Rank is the other half of the LeNet-5 result and it does not generalize the same way.
Halving it halves the coordinates and so the candidates per step, and on the CNN it also
scored better, the smaller perturbation carrying less variance per evaluation. On the
784-128-10 MLP of `examples/05` the same cut costs 1.5 points.

## Cutting the candidate count

`multifidelity_screen=True` ranks all `V` directions on a `screen_fidelity` slice of the
batch, then spends full fidelity only on the top `screen_keep_ratio`. Sample-forwards
(candidates x batch rows) drop to about `screen_fidelity/num_probe + screen_keep_ratio`
of the unscreened budget: 0.75x at the defaults, 0.40x at `screen_keep_ratio=0.25` with
`screen_fidelity=0.15`. Skipped with a warning whenever that figure is not below 1.

Ranking is by deviation from the row mean, which on an orthoplex is the antithetic
contrast, so any polytope screens.

Wall-clock follows only when the per-sample cost dominates, since the screen adds one
evaluation call per step. CPU, 2-layer MLP, defaults `0.5 / 0.25`:

| batch | plain | screened | speedup |
|---|---|---|---|
| 64 | 325 ms | 492 ms | 0.66x |
| 1024 | 1458 ms | 1380 ms | 1.06x |
| 8192 | 6741 ms | 6015 ms | 1.12x |

Turn it on for large batches, large models and GPU runs. `optimizer._last_screen_savings`
reports the fraction skipped on the last step.

## Rate-coded SNNs: the input current is constant

A spiking net driven by a static input computes `fc1(x)` inside its timestep loop where
`x` does not change, so the largest matmul runs `num_steps` times for one distinct
result. Hoisting it out is bit-identical, because every call it replaces saw the same
tensor and a fixed-shape GEMM is deterministic. A 784-128-10 LIF net at `num_steps=15`,
batch 128, went 1.87 to 0.72 ms per forward, and `examples/06` from 536 to 250 s with an
unchanged accuracy trajectory.

The hoist does not apply to temporal input: one `(T*B, F)` GEMM blocks differently from
the `T` separate `(B, F)` GEMMs it would replace, so the summation order changes. On a
`(6, 8, 64)` input, 311 of 1536 entries differ on CPU and 1150 on CUDA, up to 4.8e-07.
That is inside any sane tolerance, but a LIF thresholds at `mem >= vth`, so one ULP could
become a whole spike. A 200-seed search found no end-to-end flip, which is why the
per-timestep call stays.

## Compile flags (off by default)

- `compile_evaluator=True` wraps the vmapped forward in `torch.compile(mode="default")`
  (Inductor fusion, no CUDA graphs). Roughly 1.2-1.9x on top of vmap, smallest on
  recurrent SNNs vmap already amortized. `mode="reduce-overhead"` runs over vmap but is
  slower: vmap has already amortized the launches CUDA graphs would remove.
- `compile_forward=True` CUDA-graphs the forward+loss closure for the sequential in-place
  path (auto-selected where a full vmap sweep would OOM) and replays it per candidate. It
  keeps that path competitive with vmap; it does not beat vmap. The in-place path is
  chosen for memory, not speed.

`compile_evaluator` and `candidate_autocast` only pay on the vmap sweep. A model on a
site-aware path barely uses it, and `compile_evaluator` also shrinks the auto chunk to
leave headroom it never spends, so both come out slower there. Check which path the model
takes first.

Both are opt-in because compiling in the inner loop reorders floating-point reductions
(so the OT trajectory can shift), adds first-call latency, and recompiles on shape/dtype
changes. On hard-threshold nets the compiled loss can differ at the discretization scale,
shifting candidate selection. Enable for CUDA and shape-stable runs. Both propagate
through `register_evaluator` and are read by `api.train()`. Reproduce with
`experiments/scripts/bench_forward_backends.py` and `bench_large_net_inplace.py`.

## `candidate_autocast`

`candidate_autocast=True` runs the candidate forward's arithmetic under
`torch.amp.autocast(dtype=bfloat16)` while the parameters keep their own dtype. This is
the opposite trade from `mixed_precision`, which casts the parameters and so rounds away
any perturbation below BF16 resolution. It covers the four paths
`NNCostEvaluator.evaluate` dispatches to; the delta evaluators are excluded because their
correction is a small offset onto a full-scale output and half precision erases it.

The speedup is a CUDA claim and is unmeasured here; on CPU, BF16 autocast buys nothing.
What is checked is that it does not destroy the cost matrix's ranking, the only property
the OT solve reads: `tests/test_candidate_autocast.py` asserts Spearman correlation above
0.99 against the FP32 matrix, plus a bound on the tied-loss rate.

## CPU thread count

Pin it, and pin it below `nproc`:

```python
torch.set_num_threads(1)
```

The optimizer issues many small tensor ops per step, where torch's intra-op pool loses
because the per-op fork/join costs more than the arithmetic. On a 24-core machine, 300
chained `(256, 256)` matmuls took 1.91 s on 24 threads and 0.070 s on one. End to end on
the CPU examples, same box:

| example | 1 thread | 8 | 24 (torch default) |
|---|---|---|---|
| `03_rl_cartpole` | 2.1 s | 2.0 s | 427 s |
| `07_binary_net_no_ste` | 11.3 s | 5.0 s | 72 s |
| `08_direct_loss_minimization` | 3.8 s | 3.5 s | 14 s |
| `09_hard_decision_tree` | 6.0 s | 5.4 s | 89 s |

The last column was measured on a box with other work on it, which is the condition it
degrades under; read it as a range, not a constant.

The blowup is at the core count, not above one. The rotation QR and the `thin_qr` basis
build run single-threaded, which is what keeps the mid-range counts flat: `03` at 16
threads is 2.0 s, against 213 s unpinned. What survives is oversubscription at `nproc`,
below.

The CUDA examples are flat because the threads have little to do (1 thread against 16:
`05` 22.9 s / 22.0 s, `11` 3.8 s / 3.8 s, `06` 208.2 s / 208.5 s). So ten of the eleven
examples pin 1. `07` pins 8: its own objective, `sign()` over a `(258, 400, 32)`
activation, is 3.2 s of its 11 s at one thread and wide enough to pay for the pool.
`POLYSTEP_THREADS` overrides the examples; the test suite reads `POLYSTEP_TEST_THREADS`.

The default is the worst setting available, not merely suboptimal. Full-space step on a
`784-64-10` MLP, batch 64, which is the largest per-op shape PolyStep produces:

| `torch.set_num_threads` | 1 | 4 | 8 | 16 | 20 | 22 | 23 | 24 (default) |
|---|---|---|---|---|---|---|---|---|
| ms/step | 320 | 121 | 84 | 79 | 78 | 74 | 78 - 216 | 171 - 6688 |

At the core count the pool threads and the main thread oversubscribe and the OpenMP
spin-wait takes over: over five runs the 24-thread case ranged from 2x to 80x the
22-thread time, worst when the box was busy, and `OMP_WAIT_POLICY=PASSIVE` brings a bad
6631 ms case back to 150 ms. Full-space shapes this large and objectives like `07`'s are
where a mid-range count beats one thread; raise the count only after measuring that your
per-op shapes are in that range, and stay clear of `nproc`.
`python -m polystep.benchmarks.threads` reproduces the table.

## Starting hyperparameters

MNIST as the sanity check:

| mode | epsilon | `step_radius` | `probe_radius` | other |
|---|---|---|---|---|
| full space | `LinearEpsilon(1.0 -> 0.1)` | 0.15 | 0.12 | |
| `HybridSubspace` | decaying | 4.5 | 2.0 | `rank=4`, `rotation_interval=0` |
| `AdaptiveSubspace` | fixed 0.5 | 10.0 | 2.0 | large `rank` (4096), `use_adaptive_radius=True` |

Per-layer subspaces want a decaying epsilon; a single global projection wants a fixed
epsilon with an adaptive radius, and a larger `step_radius` than a per-layer one.

`step_radius` is measured in coordinates, so it only means a fixed weight step where the
projection has unit gain. Hybrid's QR-orthonormal columns give exactly 1 on the layers
below `sparse_threshold_bytes`. Above it Hybrid routes the layer to
`SparseRandomProjection`, which is only *approximately* orthonormal, so the gain there is
`Pi^T Pi = I` only to about a percent.
