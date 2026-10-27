# Changelog

## 1.1.1 - 2026-10-27

- Fix transport for tiny and zero masses, including warm starts, and prevent
  overflow in FP32 cost scaling.
- Make RL INT8 scoring consistent and compatible with vmap; speed up batched
  policy evaluation.
- Restrict example 06 to simulation and update the examples and guides.
- Update CI actions, reduce scheduled runs, check release dates, and remove Dependabot.

## 1.1.0 - 2026-10-24

- Add `step_batch()` for supervised training, tuple and dict inputs, and a closure
  returned by `register_evaluator`.
- Add `is_causal` support to `VmapSafeMultiHeadAttention`.
- Speed up MLP evaluation and replay site-local CUDA forwards with CUDA graphs.
  See [performance](docs/performance.md) for measurements and tradeoffs.
- Restore forward-written buffers and frozen embedding weights, including on
  failure; follow model dtype and device changes. Score sequentially when vmap
  cannot batch a model, including stock `nn.LSTM`.
- Fix small-radius gradients, KL support, marginal diagnostics, subspace rotation,
  half-precision rounding, screening, and amortized schedules.
- Update example settings and metrics; add SNN and CartPole animations.

### Upgrade notes

- Pass the model to `PolyStepOptimizer`. Rebuild it if the trainable parameter set
  changes. Separate parameters or buffers that share trainable storage are rejected;
  tie weights with the same Parameter, or register a clone.
- `FactoredSubspace` redraws `B` every 5 OT steps by default; set
  `rotation_interval=0` for the old behavior. Hybrid and factored seeded bases change.
- `SinkhornSolver` and `KLSoftmaxSolver` default to `threshold=3e-5`. Stalled solves
  above the threshold report `converged=False`. `train(restore_best=True)` selects
  weights on the first training batch.
- Rebuild compiled hybrid/factored optimizers from saved model weights: checkpoint
  names no longer contain `_orig_mod.`.

The [upgrade guide](docs/api_overview.md#upgrading-from-10) lists all compatibility
changes. Seeded results can differ; see
[reproducibility](docs/reproducibility.md#changes-since-the-published-tables).

## 1.0.0 - 2026-10-13

- Use `PolyStepOptimizer` for neural networks and `PolyStep` for standalone objectives.
- Replace diagonal CMA with `AdaptiveSubspace`, tempered softmax with `SoftmaxSolver`,
  and `auto_epsilon` with `ProgressiveEpsilon`. Probe reuse has been removed.
- Move checkpoints to format 6. Older formats cannot be loaded; rebuild the optimizer
  from saved model weights. Some low-level exports and experiment helpers moved or
  were removed. See the [migration guide](docs/api_overview.md#upgrading-from-013).
- Fix state restoration, nonfinite costs, tiny transport rows, half-precision
  evaluation, deterministic sparse CUDA projections, and baseline accounting.
- Reuse forward work, reduce projection memory and synchronization, and compile
  Sinkhorn loops. Update all 11 examples and validate the built wheel.

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
