# Changelog

## 1.1.0 - 2026-10-24

### Features and performance

- Add `step_batch(inputs, targets, loss_fn=...)` for supervised training. It shares
  the fast evaluation paths with `train()`; both accept tuple and dict inputs.
  `register_evaluator` now returns a closure for the registered batch.
- Add `is_causal` support to `VmapSafeMultiHeadAttention`.
- Extend fast MLP evaluation to functional losses and traceable custom forwards.
  Reuse unchanged forward work, reduce CPU cross-entropy overhead, and replay
  site-local CUDA forwards with CUDA graphs.
- Measured full-space MLP steps are 1.9–3.3x faster on one CPU thread and 2.6x on a
  shared RTX 5090 than 1.0.0. Tiny tied-weight models can be slower on CUDA. See
  [performance](docs/performance.md) for the evaluation paths and tradeoffs.
- Refresh the examples and their metrics, clarify simulated Loihi adaptation, and
  add SNN training and CartPole GIFs. The MAX-SAT example reaches about 98% SAT with
  600 default steps instead of 1500.

### Fixes

- Restore forward-written buffers between scoring calls and vmap chunks, including
  on failure. Protect frozen embedding weights changed by `max_norm`, reject
  evaluators for another model, and follow model dtype and device changes.
- Preserve finite-difference gradients and curvature at small probe radii. Correct
  KL zero-support handling and marginal diagnostics, including tiny target masses.
- Fix CUDA rotation for large adaptive subspaces, carry FP16/BF16 absorb rounding
  error forward, and correct biased rotation, screening, amortized schedules,
  epsilon restarts, half-precision attention, and loss precision in ES and baselines.
- Fall back to sequential scoring when vmap cannot batch a model, including stock
  `nn.LSTM`.

### Upgrade notes

- Pass the model to `PolyStepOptimizer`, not `model.parameters()`. Rebuild the
  optimizer if its trainable parameter set changes. Separate parameters or buffers
  that share trainable storage now raise `ValueError`; tie weights by assigning
  the same Parameter, or register a clone.
- `FactoredSubspace` redraws `B` every 5 OT steps by default; set
  `rotation_interval=0` for the old behavior. Hybrid and factored seeded bases change.
- `SinkhornSolver` and `KLSoftmaxSolver` default to `threshold=3e-5`, using relative
  marginal error and the dual step divided by epsilon, respectively. Stalled solves
  above the threshold report `converged=False`.
- `train()` uses `step_batch`; `restore_best` compares weights on the first training
  batch. Seeded results can differ from 1.0.
- For compiled hybrid/factored runs, rebuild the optimizer from saved model weights:
  checkpoint parameter names no longer contain the `_orig_mod.` prefix. The
  [upgrade guide](docs/api_overview.md#upgrading-from-10) covers all compatibility
  changes; [reproducibility](docs/reproducibility.md#changes-since-the-published-tables)
  lists the affected published results.

## 1.0.0 - 2026-10-13

### Upgrade notes

- Use `PolyStepOptimizer` for neural networks and `PolyStep` for standalone objectives.
- Replace diagonal CMA with `AdaptiveSubspace`, tempered softmax with `SoftmaxSolver`,
  and `auto_epsilon` with `ProgressiveEpsilon`. Probe reuse has been removed.
- Checkpoints use format 6. Rebuild the optimizer from saved model weights when
  upgrading from 0.13; older optimizer checkpoints cannot be loaded.
- Some low-level exports and experiment helpers moved or were removed. The
  [migration guide](docs/api_overview.md#upgrading-from-013) covers the main API
  replacements and checkpoint requirements.

### Fixes

- Preserve live model weights, buffers, momentum, and trust-region state across
  evaluation, subspace updates, and checkpoint restoration.
- Handle nonfinite costs, tiny transport rows, integer losses, and half-precision
  reductions consistently. Sparse CUDA projections repeat in deterministic mode.
- Accept device strings in rotation sampling and float32 masks in half-precision
  CUDA attention.
- Correct baseline budgets, tied ranks, rollout accounting, and download handling.

### Performance and examples

- Reuse unchanged forward work, update only affected weights, and size candidate
  chunks from measured activations. Speed up blockwise, factored, and adaptive steps.
- Reduce sparse projection storage and host synchronization; compile Sinkhorn loops.
- Refresh all 11 examples, their evaluation paths, and their documented settings.
  See [performance](docs/performance.md) for measurements and tradeoffs.
- Consolidate redundant tests and keep release validation focused on the built wheel.

## 0.13.0 - 2026-09-22

- Fix KL row mass, nonfinite inputs, evaluator hooks, and chunk handling.
- Reuse shared MLP forward work and load only requested MNIST images.
- Fix example checkpoint selection and evaluation budgets; trim demo dependencies.

## 0.12.0 - 2026-09-17

- Fix cross-entropy targets, stale caches, attention masks, nonfinite best values,
  and mixed-dtype evaluation.
- Correct simplex gradients and whole-model trust predictions; migrate checkpoints
  to format 5.
- Reuse MLP prefixes, use native CPU attention, and compile site-local evaluation.
- Average tied baseline ranks; add controlled experiments and forward benchmarks.
- Validate sparse dimensions and factored ranks; simplify documentation.

## 0.11.0 - 2026-08-14

- Remove `LowRankSubspace`, `FactorSpec`, and `LinearSubspace`; use `HybridSubspace`
  or `FactoredSubspace`. Replace `absorb_every` with `absorb_mode`.
- Remove `polystep.benchmarks.baselines`, `evotorch`, `SinkhornSolver.solve(init_eps)`,
  and solver warm-start methods. `CMAAdaptiveSubspace.base` is removed.
- Require `dim` in `probe_scale_of`; correct baseline probe scaling.
- Fix nonfinite costs, zero-mass rows, quadratic evaluation counts, and resume state.
- Add tight-frame quadratic fits and cache radius-jitter sampling.

## 0.10.1 - 2026-08-01

- Require `objective_token` for probe reuse and reject invalid optimizer settings.
- Fix blockwise evaluation, checkpoint aliasing, dtype changes, and tied weights.
- Add site-local evaluation to blockwise mode and reduce subspace allocations.

## 0.10.0 - 2026-07-30

- Default to simplex sampling; retune radii when upgrading.
- Change seeded rotation streams and clear momentum on basis changes.
- Enable forward compilation on the in-place path by default.
- Remove `projection_mode`, `use_csa`, low-level block strategies, and unused exports.
- Move NumPy to optional extras; validate schedules and constructor arguments.

## 0.9.0 - 2026-07-27

- Add sparse-delta evaluation, `FactoredSubspace`, greedy solvers, and evaluator
  registration helpers.
- Fix probe reuse, screening, subspace state, tied weights, and best-weight restoration.
- Extend checkpoint state and reduce projection and evaluation allocations.

## 0.8.0 - 2026-07-24

- Add optimizer checkpointing and forward-compilation controls.
- Fix mixed precision, scheduled epsilon, and invalid solver inputs.

## 0.7.0 - 2026-07-20

- Publish on PyPI and reuse blockwise candidate buffers.
- Fix constant-offset cost stability and biased rotations.

## 0.6.1 - 2026-07-15

- Add the hard decision-tree example.
- Fix mixed precision, barycentric normalization, and candidate memory estimates.
- Cache hybrid projections between rotations.

## 0.6.0 - 2026-07-12

- Add direct F1 optimization and optimizer-variant experiments.
- Fix CMA scaling, solver validation, and trust-region state.

## 0.5.0 - 2026-07-07

- Add `PolyStepES`, `minimize`, and the binary-network example.
- Fix Anderson acceleration, Newton selection, screening, and step adaptation.

## 0.4.0 - 2026-06-21

- Share solver cost handling and remove low-rank Sinkhorn.
- Fix cost scaling, dual recentering, and activation semantics.

## 0.3.0 - 2026-05-27

- Reduce GPU synchronization in solvers and blockwise updates.
- Fix CMA rotation forwarding and API examples.

## 0.2.2 - 2026-05-21

- Use unit-radius vertex generators with keyword-only `radius`.
- Fix CMA transport weighting and stale API examples.

## 0.2.1 - 2026-05-14

- Add the Loihi-style SNN adaptation example.

## 0.2.0 - 2026-05-14

- Require Python 3.11, PyTorch 2.8, and NumPy 2.0 for optional extras.
- Reduce solver synchronization and use inference mode for evaluation.

## 0.1.0 - 2026-05-03

- Initial release: gradient-free training, OT solvers, subspaces, vmap-compatible
  layers, examples, and experiment runners.
