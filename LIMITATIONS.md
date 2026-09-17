# Limitations

## Cost and search space

PolyStep needs many candidate forwards per update. Differentiable models usually
train faster with backpropagation. Small subspaces reduce cost but may exclude useful
directions; very small projection ratios can fail on large models.

`HybridSubspace` caps each layer at its parameter count. Large dense projections may
use sparse signed columns, which are not orthonormal. Claims requiring an orthogonal
projection do not apply to those columns.

`FactoredSubspace` restricts perturbations to the input directions in its fixed
factor. Rotation changes those directions. Retune radii when switching subspaces.

Momentum is cleared when the basis changes. `AdaptiveSubspace` rotates every step
by default, so momentum requires a longer rotation interval to accumulate.
Heavy-ball momentum can amplify the barycentric displacement; `step_radius` bounds
the displacement before momentum.

CMA adaptation keeps its basis fixed between absorbs and normalizes covariance to
`trace(C) = n`. Its default `mu_eff=1` gives rank-one adaptation only. Cumulative
step-size adaptation is not implemented.

## Evaluation

- `step()` expects `closure(batched_params) -> losses` with shape `(N,)`.
- Manual loops must register the evaluator and current batch to enable fast paths.
  The evaluator and closure must compute the same objective.
- Mixed-dtype models require `HybridSubspace` or `FactoredSubspace`. Full-space and
  global-subspace coordinates use one dtype and reject mixed parameter dtypes.
- `mixed_precision=True` uses BF16 geometry and forwards; solver costs use FP32.
  Models needing autocast internally must provide it.
- `adaptive_probes` needs an unchanged whole configuration and `objective_token`.
  It cannot reuse losses across changing minibatches.
- Deferred `trust_region` comparisons require the same objective. `train()` rejects
  this option; use manual steps on a stationary objective.
- `trust_region` and `use_adaptive_radius` both scale the step. Choose one controller.
- Multi-fidelity screening imputes dropped costs using an additive offset. This can
  bias the full-fidelity cost estimate and is disabled with quadratic-model options.
- Blockwise mode does not support every monolithic adaptation option; unsupported
  options warn at construction.
- The library does not distribute optimizer state across multiple GPUs.

## Attention and recurrent layers

[`VmapSafeMultiHeadAttention`](src/polystep/layers/attention.py) requires batch-first
inputs and `kdim == vdim == embed_dim`. It does not support `add_bias_kv`,
`add_zero_attn`, returned attention weights, or `is_causal=True`; supply an explicit
causal mask. It returns an output tensor, not an `(output, weights)` tuple.

Boolean masks use `True` for masked positions; float masks are additive. Attention
masks accept `(query, key)`, `(batch * heads, query, key)`, and broadcastable 4D shapes.
The wrapper also accepts `(batch, query, key)` masks.

CPU uses native scaled-dot-product attention; CUDA uses explicit matmuls. CPU vmap
may emit a batching-fallback warning. Evaluate dropout layers in `model.eval()` mode
for deterministic candidate comparisons.

[`VmapSafeLSTM`](src/polystep/layers/rnn.py) requires batch-first dense inputs. It does
not support bidirectional recurrence, projection, or `PackedSequence`. Its explicit
cell computations can be slower than fused cuDNN recurrence.

## Solvers and mathematical scope

`SoftmaxSolver` enforces source row sums only; it ignores the target marginal and
warns for a nonuniform one. Use `SinkhornSolver` to enforce both marginals.
Independent particle rotations mean Sinkhorn's shared column index does not identify
a common direction across particles.

Sinkhorn requires `omega` in `[0.5, 1.95]`. Anderson acceleration and adaptive omega
are inactive in fixed-iteration mode (`threshold <= 0`). Both solver families promote
FP16/BF16 costs to FP32.

All nonfinite costs are invalid. Their finite replacement can affect global cost
scaling, including `scale_cost='max_cost'`.

Quadratic fits are finite-radius models. Simplex gradients can be biased on
anisotropic quadratics, and block models omit cross-block curvature. A fit does not
supply a classical derivative at a discontinuity. The deferred radius controller
does not perform a trust-region acceptance test.

## Benchmark comparisons

The architecture tables use validation-selected checkpoints. `--allow-test-leakage`
is an explicit opt-in for test-selected runs, which standard aggregation rejects.

The SNN surrogate-gradient baseline has substantial variation across seeds. MAX-SAT
search heuristics use flip budgets, which are not equivalent to neural candidate
budgets. Head-only GPT-2 experiments do not establish full-model fine-tuning results.

See the [experiment index](experiments/EXPERIMENT_INDEX.md) for protocols and results.
