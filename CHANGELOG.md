# Changelog

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
