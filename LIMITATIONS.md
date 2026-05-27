# Limitations

What does not work in `polystep`, with source-file references for each entry.

## Drop-in vmap-safe layers

### `VmapSafeMultiHeadAttention` - does NOT support

(per [`src/polystep/layers/attention.py`](src/polystep/layers/attention.py))

- `kdim != embed_dim` - raises `NotImplementedError`
- `vdim != embed_dim` - raises `NotImplementedError`
- `add_bias_kv=True` - raises `NotImplementedError`
- `add_zero_attn=True` - raises `NotImplementedError`
- `batch_first=False` - raises `NotImplementedError`. The implementation
  assumes batch-first layout `(batch, seq, embed_dim)`.
- `forward(..., need_weights=True)` - raises `NotImplementedError`
- `forward(..., is_causal=True)` - raises `NotImplementedError`. Pass
  an explicit triangular `attn_mask` instead.
- `dropout > 0` under `torch.vmap` - works but emits a warning. Call
  `model.eval()` before vmap evaluation.

Note: PyTorch 2.12 fixes native `vmap(nn.MultiheadAttention)` (issue
#151558). The wrapper is retained for the `torch>=2.8` floor; users on
2.12+ may use `nn.MultiheadAttention` directly when none of the above
restrictions apply.

### `VmapSafeLSTM` - does NOT support

(per [`src/polystep/layers/rnn.py`](src/polystep/layers/rnn.py))

- `bidirectional=True` - raises `NotImplementedError`
- `proj_size != 0` - raises `NotImplementedError`
- `batch_first=False` - raises `NotImplementedError`. Assumes
  `(batch, seq_len, input_size)` input layout.
- `forward(PackedSequence)` - raises `NotImplementedError`. Pad to a
  dense tensor first.
- 2-5x slower than CuDNN: explicit gate computations (`F.linear` +
  `chunk(4)` + sigmoid/tanh) replace the fused CuDNN kernel that
  fails under vmap (PyTorch issue #105982).

## OT solvers

### `SoftmaxSolver`

(per [`src/polystep/solvers/softmax.py`](src/polystep/solvers/softmax.py))

- The target marginal `b` is **silently ignored**: the solver only
  enforces row sums equal the source marginal `a`. Passing a
  non-uniform `b` triggers a `UserWarning`.
- `epsilon < 1e-6 * max|C|` triggers a `UserWarning` because
  `-C/epsilon` may overflow before `torch.softmax` can subtract the
  row max.
- BF16 / FP16 cost matrices are promoted to FP32 internally; outer
  `torch.amp.autocast` contexts cannot bleed into the softmax.

### `SinkhornSolver`

(per [`src/polystep/solvers/sinkhorn.py`](src/polystep/solvers/sinkhorn.py))

- `omega ∉ [0.5, 1.95]` is rejected by `__post_init__`. Empirically
  `omega ≤ 1.5` is safe; `omega > 1.5` is monitored for divergence
  and backed off to 1.0 if the iterate norm grows more than 5% per
  check for 3 consecutive checks.
- `anderson_depth > 0` and `adaptive_omega=True` have no effect in
  fixed-iteration mode (`threshold <= 0`) or low-rank mode
  (`rank is not None`). Both emit a `UserWarning`.
- Warm-started duals `init_f, init_g` are **ignored** in low-rank
  mode (a `UserWarning` is emitted). Low-rank Sinkhorn always cold-
  starts from zeros (or `data_dependent_init` cost-mean init).
- BF16 / FP16 cost matrices are promoted to FP32 internally.

## Subspace and projection

- `HybridSubspace.from_layout(layout, rank=R)` produces an
  *over-parameterized* projection when `R >= min(d_in, d_out)`: the
  projection has more coordinates than parameters. Reconstruction is
  exact in this saturated regime.
- `SparseRandomProjection`: `subspace_dim / full_dim < 1e-5` triggers
  a `UserWarning` because Johnson-Lindenstrauss distance guarantees
  stop holding for typical optimization workloads. Projecting models
  at or above GPT-2 124M scale to a 128-dim subspace falls in this
  regime and collapses to random predictions.
- `PolyStep` (low-level) **does not** support `subspace + non-monolithic
  block_strategy`; raises `NotImplementedError`. The high-level
  `PolyStepOptimizer` does support that combination
  (per [`src/polystep/solver.py`](src/polystep/solver.py) - see
  `PolyStep.__post_init__` guard on `subspace` + `block_strategy`).
- `AdaptiveSubspace` step-0 (no displacement history) falls back to a
  random rotation: deterministic-reproducible with a seeded
  `torch.Generator`.

## Optimizer

- `PolyStepOptimizer.step(closure)` requires `closure(batched_params)
  -> losses` (1D tensor of shape `(N,)`). It is **NOT** a drop-in
  for `torch.optim.LBFGS`-style `closure() -> loss`. The closure
  receives a stacked param dict, not a no-arg callable.
- `subspace` is passed as an instance, not a string enum. Typos give
  a `TypeError`, not a clean `ValueError`.
- The `mixed_precision: bool = False` constructor flag casts the model
  to BF16 but does not wire to any autocast region inside the solver;
  the solvers themselves promote BF16 inputs back to FP32. This flag
  is currently a documentation gap.
- `dual_momentum_beta` defaults to `0.0`. Pass
  `dual_momentum_beta=0.3` explicitly to enable dual momentum in
  turbo mode.
- `num_probe` defaults to `1` everywhere.
- `step_radius=CosineEpsilon(...)` paired with an SNN model (LIF /
  Leaky / Spik / ALIF in module class names) emits a `UserWarning`
  because the combination collapses SNN accuracy from ~93% to 10-47%.

## Architectures and benchmarks

### What does NOT work end-to-end

(per [`experiments/EXPERIMENT_INDEX.md`](experiments/EXPERIMENT_INDEX.md))

- **SST-2 transformer from scratch**: collapses to near-random
  accuracy.
- **GPT-2 124M all-parameter fine-tune**
  (`src/polystep/benchmarks/gpt2_feasibility.py`,
  `experiments/runners/run_gpt2_finetune.py`): collapses to random
  predictions when projected to 128-dim subspace (ratio 1e-6, well
  below the 1e-5 floor - see Sparse JL warning above). Only head-only
  fine-tune is functional.
- **CIFAR-10**: deferred. Network-type and size scalability is the
  bottleneck.
- **Bidirectional LSTM, multi-head attention with causal masking,
  packed sequences**: not supported by the vmap-safe drop-in layers
  (see above).

### Asymmetric baseline comparisons

- **MAX-SAT 1M SLS comparison**
  (`experiments/runners/run_maxsat.py:937-1013`): the SLS heuristic
  is an in-repo Python WalkSAT, single seed, 50K flips at 1M vars.
  polystep receives `STEP_BUDGETS * popsize` evals; SLS receives only
  flip budget. **Not a fair comparison** to a tuned production
  solver.
- **SNN Adam-surrogate baseline**: the surrogate-gradient baseline
  reported in the paper (§5.3) is not bundled with this release; the
  Adam baseline in `experiments/results/softmax/main/snn_adam_*.json`
  uses straight-through gradients only.

### Evaluation protocol

The four main runners default to val-selected checkpoints (no
test-set leakage). A test-selected mode is opt-in via
`--allow-test-leakage`:

- `experiments/runners/run_mnist.py` - 10% validation slice.
- `experiments/runners/run_moe.py` - 10% validation slice.
- `experiments/runners/run_elevation.py` - 10% validation slice
  (affects SNN, INT8, Argmax, Staircase).
- `experiments/runners/run_timeseries.py` - validation MSE from the
  Informer-standard val split.

A regression test (`tests/test_no_test_set_leakage.py`) verifies all
runners expose the flag.

## Random-seed gotchas

- The Sinkhorn low-rank SVD initializer uses the optimizer's seed
  (propagated from `PolyStepOptimizer` through
  `solver.solve(seed=...)`).
- Tied weights are silently deduplicated in `ParamLayout.from_module`
  by `data_ptr()`. The dedup is logged at INFO level, so it is
  invisible unless `logging.basicConfig(level=logging.INFO)` is
  called.
- `torch.cuda.manual_seed_all` is never called: multi-GPU runs (not
  currently supported in any benchmark) would diverge per-device.
