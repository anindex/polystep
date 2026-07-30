# Limitations

What does not work in `polystep`, with source-file references for each entry.

## Drop-in vmap-safe layers

### `VmapSafeMultiHeadAttention` - does NOT support

(per [`src/polystep/layers/attention.py`](src/polystep/layers/attention.py))

- `kdim != embed_dim` - raises `NotImplementedError`
- `vdim != embed_dim` - raises `NotImplementedError`
- `add_bias_kv=True` - raises `NotImplementedError`
- `add_zero_attn=True` - raises `NotImplementedError`
- `batch_first=False` - raises `NotImplementedError`. The implementation assumes
  batch-first layout `(batch, seq, embed_dim)`.
- `forward(..., need_weights=True)` - raises `NotImplementedError`
- `forward(..., is_causal=True)` - raises `NotImplementedError`. Pass an explicit
  triangular `attn_mask` instead.
- `dropout > 0` under `torch.vmap` - works but emits a warning. Call `model.eval()`
  before vmap evaluation.

PyTorch 2.12 fixes native `vmap(nn.MultiheadAttention)` (issue #151558). The wrapper is
retained for the `torch>=2.8` floor; users on 2.12+ may use `nn.MultiheadAttention`
directly when none of the above restrictions apply.

### `VmapSafeLSTM` - does NOT support

(per [`src/polystep/layers/rnn.py`](src/polystep/layers/rnn.py))

- `bidirectional=True` - raises `NotImplementedError`
- `proj_size != 0` - raises `NotImplementedError`
- `batch_first=False` - raises `NotImplementedError`. Assumes
  `(batch, seq_len, input_size)` input layout.
- `forward(PackedSequence)` - raises `NotImplementedError`. Pad to a dense tensor first.
- 2-5x slower than CuDNN: explicit gate computations (`F.linear` + `chunk(4)` +
  sigmoid/tanh) replace the fused CuDNN kernel that fails under vmap (PyTorch issue
  #105982).

## OT solvers

### `SoftmaxSolver`

(per [`src/polystep/solvers/softmax.py`](src/polystep/solvers/softmax.py))

- The target marginal `b` is **silently ignored**: the solver only enforces row sums
  equal the source marginal `a`. Passing a non-uniform `b` triggers a `UserWarning`.
- `epsilon < 1e-6 * max|C|` triggers a `UserWarning` because
`-C/epsilon` may overflow before `torch.softmax` can subtract the row max.
- BF16 / FP16 cost matrices are promoted to FP32 internally; outer `torch.amp.autocast`
  contexts cannot bleed into the softmax.

### `SinkhornSolver`

(per [`src/polystep/solvers/sinkhorn.py`](src/polystep/solvers/sinkhorn.py))

- `omega ∉ [0.5, 1.95]` is rejected by `__post_init__`. Empirically `omega ≤ 1.5` is
  safe; `omega > 1.5` is monitored for divergence and backed off to 1.0 if the iterate
  norm grows more than 5% per check for 3 consecutive checks.
- `anderson_depth > 0` and `adaptive_omega=True` have no effect in
  fixed-iteration mode (`threshold <= 0`); they emit a `UserWarning`.
- BF16 / FP16 cost matrices are promoted to FP32 internally.
- The solver is full-rank only. PolyStep's OT problems are (n particles x m=V vertices)
  with V small, so the cost is `O(n*V)` and a low-rank approximation would not save
  memory or compute.

### Cost sanitization

(per [`src/polystep/solvers/_shared.py`](src/polystep/solvers/_shared.py))

- The `+inf` penalty is `2 * max|finite| + 1` over the whole matrix, not per row. It
  still ranks above every finite entry in every row, so no vertex is misordered against
  its own row. But with `scale_cost='max_cost'` a single masked entry becomes the matrix
  maximum, which roughly halves the contrast for every particle that step, including
  those with nothing masked.

## Subspace and projection

- `HybridSubspace.from_layout(layout, rank=R)` caps a layer's coordinates at its
  parameter count, so `R >= min(d_in, d_out)` saturates to the identity rather than
  over-parameterizing. Reconstruction is exact in that regime, and nothing is gained by
  raising `R` further.
- `SparseRandomProjection`: `subspace_dim / full_dim < 1e-5` triggers
a `UserWarning` because Johnson-Lindenstrauss distance guarantees stop holding for
typical optimization workloads. Projecting models at or above GPT-2 124M scale to a
128-dim subspace falls in this regime and collapses to random predictions.
- `PolyStep` (low-level) has no `block_strategy`: it runs one monolithic step and `run()`
  refuses a configured subspace, which needs `base_params` that only `init_state()`
  accepts. Block-wise modes live on `PolyStepOptimizer`.
- `AdaptiveSubspace` step-0 (no displacement history) falls back to a random rotation:
  deterministic-reproducible with a seeded `torch.Generator`.
- Momentum does not survive a basis change. The velocity is a displacement in the basis
  being replaced, so an absorb or a rotation zeroes it rather than applying it through
  the new one. `AdaptiveSubspace` rotates every step by default (`rotation_interval=1`),
  so `use_momentum` there is inert; raise `rotation_interval` or use a subspace that
  holds its basis.
- `HybridSubspace` clears its displacement history on a basis change, so a rotation costs
  the window rather than mis-attributing it. `AdaptiveSubspace` keeps its history in full
  parameter space and needs no clear. `rotation_interval` defaults to `0` (no rotation),
  which is also the accuracy recommendation.
- `FactoredSubspace` confines every perturbation to the `rank` input directions spanned
  by its fixed `B` factor, so at matched subspace dimension it makes less progress per
  step than `HybridSubspace`'s dense projection. It is 3-11x cheaper per step; on the
  MNIST example at matched dimension it reached 81.4% against 90.9%, or 88.8% with
  `rotation_interval=1`. It is a speed/memory trade, not a drop-in improvement, and is
  not the default. See `docs/performance.md`.
- `use_covariance_adaptation` holds the subspace basis fixed between absorbs. sep-CMA
  learns a per-axis variance for one basis, and a diagonal covariance does not stay
  diagonal under rotation, so the two cannot both run.
- `use_covariance_adaptation` is **rank-one only** by default. The optimizer derives the
  CMA rates at `mu_eff = 1`, where Hansen's
  `c_mu = 2(mu_eff - 2 + 1/mu_eff)/((n+2)^2 + mu_eff)` is exactly zero, so only the `p_c`
  rank-one term shapes the covariance. That is the right `mu_eff` for the evolution
  paths, which consume a single unit-normalised displacement, but not for the rank-mu
  term, whose offspring are the `2*pdim` transport-weighted vertex steps. Set `mu_eff`
  explicitly on `CMAAdaptiveSubspace` to make `c_mu` positive and turn rank-mu on.
- The covariance is renormalised to `trace(C) = n` every step, so `C` carries only shape
  and never scale. `step_radius` sets the step magnitude, not `C`. This departs from Ros
  & Hansen sep-CMA-ES, where the trace is free.
- There is no cumulative step-size adaptation. CSA reads step size from the length of the
  evolution path against the length expected under a *Gaussian random walk*, and an OT
  barycentre is a deterministic descent direction, so that reference does not apply and
  the step size would grow without bound. Use `use_adaptive_radius` for loss-driven step
  adaptation. Two-Point step-size Adaptation (arXiv:0805.0231) is the model-free
  alternative that would fit here, and is nearly free given the orthoplex already
  evaluates antithetic pairs; it is not implemented.

## Optimizer

- `PolyStepOptimizer.step(closure)` requires `closure(batched_params) -> losses`, a 1D
  tensor of shape `(N,)`. Not a drop-in for `torch.optim.LBFGS`-style
  `closure() -> loss`: the closure receives a stacked param dict, not a no-arg callable.
- `subspace` is passed as an instance, not a string enum. A string reaches the step and
  raises `AttributeError: 'str' object has no attribute 'subspace_dim'`.
- A model that computes in more than one dtype needs a per-entry subspace
  (`HybridSubspace`, `LinearSubspace`, `LowRankSubspace`), which gives each parameter its
  own projection and keeps its dtype. Full space and `AdaptiveSubspace` hold every
  parameter in one vector at the layout's `dominant_dtype`, so the minority would be
  optimized at the majority's precision; both raise at construction instead. The
  perturbation itself is at coordinate resolution, which is `dominant_dtype`, since the
  coordinates are a single tensor.
- The `mixed_precision: bool = False` flag runs the model forward and the polytope
  geometry in BF16 while the OT solvers promote the cost to FP32. The barycentric and
  fused-softmax projections, the `HybridSubspace` QR, and the cost evaluator bridge the
  BF16/FP32 boundary, so a step runs end to end. There is no autocast region inside the
  model forward, so a model that needs autocast for its own BF16 numerics must add it.
- `dual_momentum_beta` defaults to `0.0`. Pass `dual_momentum_beta=0.3` to extrapolate
  the warm-started duals across steps.
- `num_probe` defaults to `1` everywhere.
- Any scheduled `step_radius` paired with an SNN model (`lif`, `leaky`, `spik`, `alif` in
  a module class name) emits a `UserWarning`: the combination collapses SNN accuracy from
  ~93% to 10-47%. Pass a flat float.
- The fast candidate evaluators only run when the optimizer holds the evaluator and its
  data. `train()` calls `register_evaluator` on every batch; a hand-rolled loop calling
  `step(closure)` must do the same, or the fused in-place, factored and delta paths stay
  unused.
- `adaptive_probes` reuse is all or nothing on the configuration. A candidate is the
  whole parameter vector with one particle row replaced, so every row of the cost matrix
  depends on every particle's position and one moving particle invalidates all of them.
  Reuse therefore saves forwards only once the whole configuration has settled, which in
  practice means near convergence. `train()` passes a per-batch `objective_token`, so on
  a minibatch objective it saves nothing at all and only the bookkeeping remains.
- `multifidelity_screen` needs a cheap `screen_closure` from the caller,
  `screen_fidelity/num_probe + screen_keep_ratio < 1`, and either
`polytope_type='orthoplex'` or a selection solver (`min_cost_greedy`, `top_k_mean`),
whose screen ranks vertices directly and so needs no antithetic pairing. Outside that it
warns and does not run. The default simplex therefore does not screen. It also only pays
off in wall-clock when the per-sample cost dominates: measured 0.66x at batch 64 and
1.12x at batch 8192 on CPU.
- The CMA scalings read like errors and are not. `trace_scale=n` and the `pdim` factor on
  rank-mu compensate for evolution paths fed unit-normalized innovations, so `E||p_c||^2`
  is about 1 rather than `n`.
- `SinkhornSolver`'s column marginal couples particles through the vertex index. Each
  particle carries its own rotation, so "vertex v" is a different direction per row, and
  the shared `b = 1/V` marginal spreads mass across an index that has no common meaning.
  This is a spreading regularizer rather than a wrong answer, and it applies to the
  full-space default; subspace mode defaults to the independent-row softmax.

### Multi-fidelity screening

(per [`src/polystep/_step_monolithic.py`](src/polystep/_step_monolithic.py))

- Vertices the screen drops are imputed as `screen_cost + (full_mean - screen_mean)`, a
  per-particle additive offset. That is exact only when the gap between the screened and
  full fidelity is the same for every direction within a particle. Under a multiplicative
  gap it under-prices the dropped vertices. They are the high-screen-cost ones and so
  already carry little transport mass, which bounds the effect, but the screened cost
  matrix is not an unbiased estimate of the full one.

## Architectures and benchmarks

### What does NOT work end-to-end

No result files ship for these; they are recorded here, not in
[`experiments/EXPERIMENT_INDEX.md`](experiments/EXPERIMENT_INDEX.md).

- **GPT-2 124M all-parameter fine-tune** (`experiments/runners/run_gpt2_finetune.py`):
  collapses to random predictions at a 128-dim projection (ratio 1e-6, below the 1e-5 JL
  floor above). Head-only works.
- **CIFAR-10**: deferred. Network-type and size scalability is the bottleneck.

### Asymmetric baseline comparisons

- **MAX-SAT 1M SLS comparison** (`run_sls` in `experiments/runners/run_maxsat.py`): the
  SLS heuristic is an in-repo Python WalkSAT, single seed, 50K flips at 1M vars. polystep
  receives `STEP_BUDGETS * popsize` evals; SLS receives only flip budget. **Not a fair
  comparison** to a tuned production solver.
- **SNN Adam-surrogate baseline**: the surrogate-gradient baseline reported in the paper
  (§5.3) is not bundled with this release; the Adam baseline in
  `experiments/results/softmax/main/snn_adam_*.json` uses straight-through gradients
  only.

### Evaluation protocol

The four main runners default to val-selected checkpoints (no test-set leakage). A
test-selected mode is opt-in via `--allow-test-leakage`:

- `experiments/runners/run_mnist.py` - 10% validation slice.
- `experiments/runners/run_moe.py` - 10% validation slice.
- `experiments/runners/run_elevation.py` - 10% validation slice (affects SNN, INT8,
  Argmax, Staircase).
- `experiments/runners/run_timeseries.py` - validation MSE from the Informer-standard val
  split.

A regression test (`tests/test_no_test_set_leakage.py`) verifies all runners expose the
flag.

## Random-seed gotchas

- Tied weights are silently deduplicated in `ParamLayout.from_module` by `data_ptr()`.
  The dedup is logged at INFO level, so it is invisible unless
  `logging.basicConfig(level=logging.INFO)` is called.
- Multi-GPU is not supported by any benchmark, and nothing in the library shards state
  across devices.
