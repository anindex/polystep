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

Momentum, the `amortize_steps` coasting direction, and the `biased_rotation`
direction are cleared when the basis changes. `AdaptiveSubspace` rotates every step
by default, so these options need a longer rotation interval to take effect; the
optimizer warns when they meet a rotation every step. Heavy-ball momentum can
amplify the barycentric displacement; `step_radius` bounds the displacement before
momentum.

## Evaluation

- `step()` expects `closure(batched_params) -> losses` with shape `(N,)`.
  The candidate tensors may share storage that the next chunk overwrites, so clone
  any you keep after the closure returns.
- Manual loops must register the evaluator and current batch to enable fast paths.
  The evaluator and closure must compute the same objective.
- Mixed-dtype models require `HybridSubspace` or `FactoredSubspace`. Full-space and
  global-subspace coordinates use one dtype and reject mixed parameter dtypes.
- `mixed_precision=True` uses BF16 geometry and forwards; solver costs use FP32.
  When a candidate's output is FP16 or BF16, whether from the model's dtype or from
  `candidate_autocast`, an `nn.CrossEntropyLoss`, `nn.MSELoss`, or `nn.L1Loss` left
  at its defaults apart from `reduction` reduces it in FP32. Any other loss, including one with class
  weights, label smoothing, or hooks, keeps the dtype it returns, so candidates
  closer than its rounding step tie.
  Models needing autocast internally must provide it.
- With FP16 or BF16 parameters, the subspace-delta and factored evaluators are off
  so that candidates are scored on the weights they round to. `FactoredSubspace`
  then materializes every candidate and runs slower.
- Deferred `trust_region` comparisons require the same objective. `train()` rejects
  this option; use manual steps on a stationary objective.
- `trust_region` and `use_adaptive_radius` both scale the step. Choose one controller.
- Multi-fidelity screening imputes dropped costs using an additive offset. This can
  bias the full-fidelity cost estimate and is disabled with quadratic-model options.
- Full-space blockwise mode does not support every monolithic adaptation option;
  unsupported options warn at construction. With a subspace, `per_layer` and
  `grouped` run the monolithic step and only split Sinkhorn and KL solves into row
  segments.
- The automatic chunk size sees only the outputs of layers that own parameters.
  Attention scores and other intermediates inside a layer are not counted, so set
  `chunk_size` by hand if such a model runs out of memory.
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

All nonfinite costs are invalid. `min_cost_greedy` and `top_k_mean` skip a vertex
with any nonfinite probe, and the solve leaves a particle with no feasible vertex
where it is. The weighted solvers replace each nonfinite cost with a finite penalty,
twice the largest finite magnitude plus one, so an infeasible candidate keeps a small
weight that depends on epsilon. Under `scale_cost='mean'` or `'max_cost'` the penalty
is computed after scaling; otherwise that weight also depends on the cost offset.
Balanced Sinkhorn sends every column its prescribed mass, even when all of that
column's candidates are infeasible. The monolithic step records a vertex's cost as
the mean of its probes with each nonfinite probe already penalized, so with several
probes a vertex with only some nonfinite probes can still set the best-vertex
progress that drives the adaptive radius.

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
