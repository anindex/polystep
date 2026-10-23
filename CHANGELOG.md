# Changelog

## 1.1.0 - 2026-10-24

### Upgrade notes

- A buffer or a separate Parameter that shares memory with a trainable parameter now
  raises `ValueError`. Tie weights by assigning the same Parameter, or register a
  clone.
- `step()` raises `ValueError` when the set of `requires_grad` parameters changes
  after construction, and passing `model.parameters()` instead of the module raises
  `TypeError`.
- `FactoredSubspace` redraws `B` every 5 OT steps by default; set
  `rotation_interval=0` for the 1.0 behavior. `HybridSubspace` and `FactoredSubspace`
  draw new seeded bases.
- `SinkhornSolver` and `KLSoftmaxSolver` stop on a relative error with
  `threshold=3e-5` and report `converged=False` when the error stalls.
- `train()` runs each batch through `step_batch`, and `restore_best` compares weights
  by their loss on the first training batch.
- Parameter names of `torch.compile` modules lose the `_orig_mod.` prefix, so 1.0
  optimizer checkpoints of compiled hybrid or factored runs no longer load.
- Seeded runs differ from 1.0. The
  [upgrade guide](docs/api_overview.md#upgrading-from-10) lists every change, new
  warnings and validation included, and the
  [reproducibility guide](docs/reproducibility.md#changes-since-the-published-tables)
  lists the affected published results.

### New

- `PolyStepOptimizer.step_batch(inputs, targets, loss_fn=...)` runs one step on a
  batch through the same fast paths as `train()`. Examples 02, 05, 10 and 11 use it.
- `step_batch`, `train()` and `NNCostEvaluator` accept tuple and dict inputs for
  multi-input models; `train()` reads the target from a dict's `labels` entry.
- Functional and configured cross-entropy, MSE and L1 losses, and custom forwards
  that trace to a chain of Linear layers, use the fast MLP paths.
- `register_evaluator` returns a closure that scores the registered batch, and
  `VmapSafeMultiHeadAttention` accepts `is_causal`.

### Fixes

- Scoring restores every buffer the forward writes, so identical candidates get
  identical costs. Models vmap cannot batch, such as stock `nn.LSTM`, fall back to
  scoring one candidate at a time instead of raising.
- Evaluators follow dtype casts and device moves, and inputs keep their dtype when
  the forward accepts it.
- `AdaptiveSubspace` rotation no longer crashes on CUDA above about 8M parameters,
  and FP16 and BF16 absorbs carry their rounding error forward.
- Smaller fixes cover `biased_rotation`, multifidelity screening, rotation schedules
  under `amortize_steps`, `KLSoftmaxSolver` zero-support columns, `CosineEpsilon`
  restarts, half-precision CUDA attention, and `PolyStepES` and baseline loss
  precision.

### Performance and examples

Measured against 1.0.0 on one CPU thread and a shared RTX 5090; see
[performance](docs/performance.md) for the scoring paths.

- Full-space MLP steps run about 1.9x to 3.3x faster on CPU and 2.6x on CUDA. The
  784-64-10 step at batch 64 that `python -m polystep.benchmarks.threads` times drops
  from about 320 ms to 97 ms on one efficiency core. `per_layer`, `AdaptiveSubspace`
  and `FactoredSubspace` steps run 2x to 4x faster on CPU.
- On CPU, site-local scoring adds each candidate's output change instead of building
  its weights; on CUDA it replays CUDA graphs. Example 05 trains about 2x faster on
  CPU, and example 11 steps 2.2x faster on CUDA.
- A tiny tied-weight `HybridSubspace` model now takes the site-local path and steps
  about 2x slower on CUDA; larger tied models are faster.
- Example 04 runs 600 steps by default instead of 1500. It still reaches about 98%
  SAT, in about a third of the time, and takes a `--device` flag.

## 1.0.0 - 2026-10-13

### Upgrade notes

- Use `PolyStepOptimizer` for neural networks and `PolyStep` for standalone objectives.
- Replace diagonal CMA with `AdaptiveSubspace`, tempered softmax with `SoftmaxSolver`,
  and `auto_epsilon` with `ProgressiveEpsilon`. Probe reuse has been removed.
- Checkpoints use format 6. Rebuild the optimizer from saved model weights when
  upgrading from 0.13; older optimizer checkpoints cannot be loaded.
- Some low-level exports and experiment helpers moved or were removed. The
  [migration guide](docs/api_overview.md#upgrading-from-013) lists replacements and
  changes to evaluation, configuration, and callbacks.

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
