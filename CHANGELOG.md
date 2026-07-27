# Changelog

## 0.9.0 - 2026-07-27

### Added

- `SparseDeltaEvaluator`: a full-space candidate replaces `particle_dim` scalars, so its
  output differs from the base in at most that many columns. Propagates the delta through
  elementwise layers and goes dense one `Linear` past the perturbed one, instead of a
  dense `bmm` over `N` weight sets. Automatic for `nn.Sequential` of `Linear` and
  elementwise layers, with a per-chunk fallback.
- `FactoredSubspace`: perturbs each 2D parameter by `dW = A @ B` with `B` fixed, so
  `x (W + A B)^T = x W^T + (x B^T) A^T` and no candidate weight is built. Not the
  default: the fixed `B` confines perturbations to `rank` input directions.
- `step(..., objective_token=...)` marks which objective a step measures, so cost rows
  cannot be reused across minibatches. `release_evaluator()`, `resync_from_model()`,
  `MinCostGreedySolver` and `TopKMeanSolver` are public.
- `MANIFEST.in`, so the sdist ships a runnable test suite. CI runs it from the tarball.

### Fixed

- `api.train()` never called `register_evaluator`, the only assignment of
  `_cost_evaluator`, `_fused_inputs`, `_factored_evaluator` and `_sparse_delta_evaluator`,
  so every fast evaluation path was unreachable through the training API.
- Cost-row reuse under `adaptive_probes` was per particle, but a candidate is the whole
  configuration with one row replaced, so a stagnant particle's row was still measured
  against every other particle's position. Reuse is now all or nothing.
- `multifidelity_screen` averaged direction contrast across particles, which carry
  independent rotations, so the kept set was chosen at random. Ranked per particle now.
- Tied weights raised on both evaluation paths. `batch_unflatten` emits the canonical key
  only; `unflatten` still carries aliases for `load_state_dict`.
- `restore_best` keyed on the OT cost when no callback was registered, restoring the step
  with the cheapest probe cloud rather than the best weights. It also snapshots to CPU.
- Chunk size ignored the weight set a non-fused subspace builds per candidate and the vmap
  activations. A 120K-param step peaked at 1661 MB in full space and 10565 MB in subspace
  mode; now 946 MB and 1461 MB.
- `AdaptiveSubspace` and per-layer rotation replaced the projection while the coordinates
  were non-zero, moving the weights with nothing evaluated behind them.
- `CMAAdaptiveSubspace` never rotated or absorbed, left its hyperparameters at `0.0` so
  the covariance update was inert, normalized the evolution paths by `sigma` instead of
  `step_radius` so `C_diag` decayed to its floor, carried state through basis rotations
  untransformed, and accumulated at bf16 where `1 + c_1*x` rounds back to `1`.
- Anderson acceleration subtracted `dX` instead of `dX + dR`, so `anderson_depth > 0`
  converged slower than no acceleration.
- `num_particles = 1` froze the optimizer: balanced OT with one row forces a uniform plan.
  The default now selects `SoftmaxSolver` and warns.
- `scale_cost` divided before recentering, so adding a constant to every cost changed the
  plan. `SinkhornSolver` also reported `ent_reg_cost` in the wrong frame; the plan and
  duals were unaffected. The greedy and top-k solvers had the same reporting bug.
- Softmax paths recentred by the global minimum, so a row above it underflowed to `-inf`
  and returned NaN; the fused kernel did not recenter at all.
- `sanitize_cost` mapped `-inf` to the worst finite value instead of the best, and used an
  absolute penalty that entered the `scale_cost='mean'` reduction.
- `state_dict` captured only `SolverState`, so resume was not exact for `amortize_steps`,
  `adaptive_probes`, `trust_region` or `HybridSubspace`. Format is 3; format 2 loads with
  a warning.
- A stagnation absorb never cleared `stagnation_count`, redrawing the basis every step on
  a plateau. `resync_from_model()` left momentum and rotation state on the old anchor.
- Probe losses accumulated in FP32 unconditionally, so a float64 objective whose costs
  differ below FP32 resolution produced a constant cost matrix.
- Block-wise biased rotation raised on bf16 under `mixed_precision=True`. A zero descent
  direction produced a singular rotation collapsing `+e0` and `-e0`.
- Smaller: the in-place evaluator dropped unknown parameter keys silently; the
  trust-region ratio scored a Newton step that was never applied; displacement history
  mixed bases; randomized PCA drew from the global RNG; rank transitions dropped
  `HybridSubspace` settings; `data_dependent_init` had the wrong sign; a float64 model
  with a subspace raised; a frozen model gave an opaque error; `pyproject.toml` declared
  `torch>=2.4` where 0.2.0 announced `torch>=2.8`.

### Changed

- `adaptive_probes` and `adaptive_num_probe` default on under `block_strategy='monolithic'`.
- `use_csa` is refused with a warning: the OT step is a deterministic descent direction, so
  CSA reads agreement as the normal state and `sigma` grows without bound.
- `multifidelity_screen` cuts the forward budget instead of only reweighting, and runs only
  when `screen_fidelity/num_probe + screen_keep_ratio < 1`. Needs a cheap closure from
  `screen_closure_from()`; `api.train` supplies one.
- Block-wise warns for options it ignores; `grouped` warns when combined with a subspace.
  `LinearSubspace` amplifies rather than dilutes the perturbation, as its docstring said.
- `_step_blockwise` claimed blockwise cuts OT cost from `O(N^2)` to `O(N^2/L)`. The forward
  count is identical; `L` solves over `P/L` rows replace one over `P`.
- Removed `get_sobol_rotation_matrices`, `HybridSubspace.reconstruct_base` and
  `reconstruct_batch_delta`: no callers, no coverage.

### Performance

- `thin_qr` pins basis construction to one thread: `HybridSubspace` setup for the MNIST
  example drops from 27.0 s to 97.8 ms. `apply_biased_rotation` uses modified Gram-Schmidt
  instead of `torch.linalg.qr`, 7-33x faster at these sizes. Compiled kernels take radius
  and epsilon as 0-d tensors, so a scheduled radius no longer recompiles every step.
- Example 06 on an RTX 5090: 1024 s to 845 s, shift-recovery +9.3 pp to +10.7 pp.

### Tests, examples and docs

- 967 tests in 262 s with an intermittent timeout failure, now 941 in 8 s with none.
  `tests/test_subspace_contract.py` replaces per-class copies of the properties every
  subspace shares. Removed tests that could not fail. Suite verified by mutation: eight
  semantic changes to the source are all caught.
- `tests/test_nondiff_data.py` guarded nine tests on the `pysat` import but not on the
  `experiments/` tree they import from, so they failed rather than skipped in the sdist
  when pysat happened to be installed.
- Examples pin CPU threads, check optional imports before use instead of failing after the
  run, and register the evaluator in hand-rolled loops. `python-sat` moved into the
  `examples` extra, which example 04 needs.
- `release.yml` lint and test commands had drifted from `ci.yml` and it lacked the CPU
  torch index; both now match. README and the docs corrected where they had drifted.

## 0.8.0 - 2026-07-24

### Added

- `compile_forward` on `NNCostEvaluator` / `PolyStepOptimizer`: CUDA-graphs the
  forward+loss on the in-place path and replays it per candidate. Rescues the
  memory-forced path; does not beat compiled vmap. See `docs/performance.md`.
- `TrainConfig(restore_best=True)` (default): restore the lowest-loss weights
  before `train()` returns.
- `PolyStepOptimizer.state_dict()` / `load_state_dict()` for checkpoint-resume
  (solver state, `ProgressiveEpsilon` internals, RNG). Model weights save separately.

### Changed

- `compile_evaluator` / `compile_forward` propagate to a registered evaluator via
  `register_evaluator`, so they work outside `api.train()`.
- `train()` skips the extra per-step full-batch forward and its host syncs when no
  callback consumes per-step metrics.

### Fixed

- Standalone `PolyStep` accepts any epsilon scheduler (`CosineEpsilon`,
  `ProgressiveEpsilon`), not only `LinearEpsilon` (was `TypeError`).
- `mixed_precision=True` builds the `ParamLayout` after the BF16 cast, so full-space
  candidate params are BF16 (were FP32).
- `evaluate_subspace_inplace` casts float inputs to the param dtype, fixing a mismatch
  under `mixed_precision` + `HybridSubspace` + FP32 inputs.
- `cost_batch_size` subsampling keeps the seeded RNG and compares the full device on a
  mismatch (was non-deterministic; fixes a `cuda:0`/`cuda:1` `randperm` crash).
- `PolyStep.run()` accepts a 1-D initial point (normalized to one particle).
- `ProgressiveEpsilon` no longer inflates epsilon every step under a fixed-iteration
  Sinkhorn solver (`threshold<=0`).
- Greedy solvers: `MinCostGreedySolver` sanitizes non-finite costs; `TopKMeanSolver`
  rejects `k=0`.
- `PolyStep.step()` uses the shared `sanitize_cost` (was an inline copy that synced and
  could overflow the penalty to `inf`); `state.costs` records the raw mean cost.
- `HybridSubspace` docstring corrected (QR-orthonormal columns; `step_radius` retune).
- `NNCostEvaluator` compile docstrings match the shipped modes; the vmap-fallback
  filter matches functorch markers instead of any "batched" error.

## 0.7.0 - 2026-07-20

### Added

- On PyPI now: `pip install polystep` (or `uv add polystep`).

### Changed

- Wider install range: `torch >= 2.4` (was 2.10) and `numpy >= 1.24` (was 2.0).
- Blockwise and subspace-blockwise steps reuse one buffer for the per-chunk
  configs instead of reallocating it every chunk, so a step allocates less.

### Fixed

- Solvers stay accurate when the cost matrix has a large constant offset
  (`|C|` far above `epsilon`). The cost is recentered before the log-domain
  math, which leaves the transport plan unchanged, so `SinkhornSolver.matrix`,
  `KLSoftmaxSolver(lam=0)`, and the softmax solvers no longer lose precision or
  return NaN in that regime.
- `SinkhornSolver` computes `ent_reg_cost` from the actual plan mass, so it is
  correct before convergence too, and it warns when the two marginals do not
  sum to the same total.
- A 1D biased rotation returns the identity instead of flipping the axis it just
  aligned (`SO(1)` cannot represent a reflection).

## 0.6.1 - 2026-07-15

### Added

- `examples/09_hard_decision_tree.py`: a hard oblique decision tree with strict
  argmax routing and no relaxation, trained on an XOR checkerboard. PolyStep
  optimizes the hard tree directly while OpenAI-ES and SPSA stall on its
  piecewise-constant loss; a soft-tree Adam baseline is scored after hardening.

### Changed

- The HybridSubspace fused block-diagonal projection is rebuilt only when the
  projection rotates. With the default `rotation_interval=0` the projection is
  static, so the previous per-step `block_diag` rebuild was pure overhead.

### Fixed

- `mixed_precision=True` runs end to end. The barycentric and fused-softmax
  projections cast the FP32 OT weights to the BF16 geometry dtype, `HybridSubspace`
  runs its projection QR in FP32 (no BF16 CPU `geqrf`), and the cost evaluator
  matches float inputs to the parameter dtype. Previously the first step raised a
  dtype mismatch.
- The barycentric projection normalizes by the realized transport row sum instead
  of the target marginal, so an unconverged Sinkhorn plan gives a
  translation-invariant step. The softmax path is unchanged.
- Full-space monolithic steps bound the default evaluation chunk to a fixed memory
  budget, avoiding the `O(n_params^2)` config buffer that ran out of memory at the
  default `chunk_size=None`.
- `apply_biased_rotation` realigns the first search axis with the requested bias
  (QR fixes the column only up to sign), so the biased search points toward descent
  rather than ascent, and runs its QR and determinant in FP32 for BF16 inputs.

## 0.6.0 - 2026-07-12

### Added

- `examples/08_direct_loss_minimization.py`: direct F1 minimization on an imbalanced
  checkerboard where PolyStepES beats both a biased gradient (Adam+STE) and OpenAI-ES.
- `experiments/runners/variant_sweep.py`: a reproducible sweep of the optimizer variants (solver, representation, block strategy,
  adaptation flags, schedules, geometry) across nine tasks, ranked by forward-eval budget
  with a state-mutation self-check that flags any variant whose state never changed.

### Changed

- CMA covariance updates use the covariance-metric offspring step `y = sqrt(C_diag) * z`,
  so the evolution paths and rank-mu match the covariance-scaled sampling.
- AdaptiveSubspace runs the per-step basis QR on the GPU in fp32 for CUDA targets, several
  times faster than the previous CPU round-trip for large models.

### Fixed

- `SinkhornSolver` rejects `max_iterations < 1` and validates user-supplied marginals
  (finite, nonnegative, positive mass) instead of clamping to an infeasible plan.
- `PolyStepES` validates `dim`, `num_particles`, `epsilon`, and `step_radius` at construction.
- CMA covariance and step-size adaptation are disabled with a warning under non-monolithic
  block strategies, where they previously never updated.
- `trust_region` now records its predicted improvement independent of `biased_rotation`, so
  the step-radius multiplier adapts whenever `use_quadratic_model` and `num_probe >= 2` hold
  instead of staying frozen at 1.0. `trust_region` also auto-enables the quadratic model.
- Heaviside stall detection uses the correct generation index (`generation + 1`).

### Removed

- Dead `compute_ot_weights` export; the rank-mu path uses the transport masses directly.

## 0.5.0 - 2026-07-07

### Added

- `PolyStepES` and `minimize`: an ask/tell interface so PolyStep drops into
  evolution-strategy harnesses (evosax, NeuroEvoBench) as a gradient-free optimizer.
- `examples/07_binary_net_no_ste.py`: STE-free binary-net training, PolyStepES vs OpenAI-ES.
- `experiments/bench_ask_tell.py`: ask/tell comparison against a Gaussian ES on the synthetic suite.

### Changed

- CMA covariance rank-mu now estimates per-coordinate variance from the
  transport-weighted polytope vertices, and drops the per-step scatter allocation.
- CSA calibrates the step-size against the running norm of the evolution path
  instead of sqrt(n), so sigma no longer collapses on OT-scale steps.
- Anderson acceleration uses a Tikhonov ridge solve and only accepts steps that
  do not regress the dual objective.
- Newton refinement keeps its step only when the quadratic model says it beats
  the plain OT step.
- Amortized momentum coasts along the OT step, not the OT-plus-momentum step.

### Fixed

- Multi-fidelity screening no longer feeds its own dampened cost back in, which
  had permanently starved low-contrast directions.
- Sinkhorn fixed-iteration mode reports convergence on a finite result, so
  ProgressiveEpsilon stops inflating epsilon.
- `adaptive_omega` no longer erases the divergence back-off within the same check.
- Guarded the barycentric projection against a zero source marginal.

## 0.4.0 - 2026-06-21

### Added

- `solvers/_prelude.py`: one shared solver preamble (FP32 promotion, device-side
  non-finite cost sanitization, device/dtype-aligned marginals and warm-start
  duals) used by the Sinkhorn, softmax, and tempered-softmax solvers.
- `experiments/scripts/bench_eps_rescale.py` and `tests/test_hardening.py`.

### Changed

- Fused-softmax now honors every `scale_cost` mode (`'mean'/'max_cost'/float/None`),
  matching `SoftmaxSolver`; the compiled kernel no longer scales internally.
- `TemperedSoftmaxSolver` gains FP32 promotion, non-finite handling, and an
  autocast-disabled softmax (previously absent).
- Warm-start dual re-centering uses the valid gauge `f -> f+c, g -> g-c` instead
  of independent mean subtraction, which perturbed the iterate under `omega != 1`.
- `BatchedLinearEvaluator` applies the real activation modules (exact
  `LeakyReLU.negative_slope`, `GELU.approximate`) rather than hardcoded
  functional defaults; non-default `Flatten` falls back to vmap.
- Low-level `PolyStep.step()` forwards `init_eps` like the optimizer path.

### Removed

- Low-rank Sinkhorn (`SinkhornSolver.{rank,gamma,auto_rank_threshold}`,
  `_solve_low_rank`, `SinkhornResult.{_Q,_R,_g_lr}`, the `solve(seed=...)` arg,
  and the `rank` params on `PolyStepOptimizer` / `PolyStep`).

### Fixed

- Re-validate `epsilon > 0` inside `SinkhornSolver.solve()` (schedules mutate
  it); reject `check_every < 1`, `num_probe < 1`, and `scale_cost` of 0 / non-finite.
- Mezzadri rotation sign correction treats `sign(0)` as `+1` so an underflowed
  QR diagonal can't null a column; dropped a redundant `Q.clone()`.

## 0.3.0 - 2026-05-27

Cleanup. No public API changes.

### Changed

- `solvers/sinkhorn.py`: dropped two GPU->CPU syncs per Sinkhorn solve.
  `cost_scale` (the warm-start `clamp_` bound) stays on-device as a 0-d
  tensor, and `ent_reg_cost = <f,a> + <g,b>` resolves both inner products
  in one host transfer via `torch.stack([...]).sum().item()`. Same pattern
  applied to the low-rank convergence loop and its `err_a/err_b` marginal
  transfers.
- `_step_blockwise.py`: per-block fused-softmax `ent_cost_tensor.item()`
  calls now defer to a single `torch.stack([...]).sum().item()` after the
  per-block loop, mirroring `block_disp_terms` and `block_model_loss_terms`.
  Saves `O(num_blocks)` syncs per step in the fused-softmax path.
- `cma.py`: `update_step_size_csa` now accepts an optional pre-computed
  `p_sigma_norm`; both call sites (`_step_common.update_cma_state`,
  `_step_monolithic.step`) forward the norm they already paid for in
  `compute_heaviside_sigma`, removing one redundant `torch.norm(...).item()`
  per CMA generation.

### Fixed

- `CMAAdaptiveSubspace.rotate()` now forwards `transport_matrix`,
  `X_vertices`, and `X_current` to the wrapped `AdaptiveSubspace`, so the
  `'ot_bias'` rotation mode fires when CMA is enabled (it previously fell
  through to random rotation).
- `docs/api_overview.md`: the `PolyStepOptimizer.step` snippet used a
  zero-argument lambda; replaced with the real
  `closure(batched_params) -> losses` signature via `NNCostEvaluator`. The
  `SparseRandomProjection` example used `input_dim` / `output_dim` keyword
  args; the actual constructor takes `full_dim` / `subspace_dim`.
- Tightened two test tolerances that were 100-1000x looser than the
  solver's convergence threshold (`tests/test_numerical_stress.py` row-sum
  checks; `tests/test_ablation_solvers.py` tempered-softmax-vs-greedy
  match).

## 0.2.2 - 2026-05-21

Codebase cleanup, test modernization, and documentation correctness pass.

### Breaking

- Vertex generator API (`get_orthoplex_vertices`, `get_simplex_vertices`,
  `get_cube_vertices`) signature changed from `(dim, origin, radius)` to
  `(dim, *, radius=1.0)`. The `origin` parameter has been removed - all
  generators now produce unit-radius templates centered at the origin;
  translation is handled by the solver. This affects direct callers only;
  `PolyStep` and `PolyStepOptimizer` are unaffected.

### Fixed

- `compute_ot_weights` in `cma.py`: replaced linear softmax normalization
  with entropy-based OT weight computation matching the theoretical
  derivation.
- `@torch.inference_mode()` migration across `transform.py` (removed
  unused `import warnings`).
- README and API overview: `max_iteration` -> `max_iterations` (matching
  the actual `PolyStep` dataclass field).
- API overview: `AdaptiveSubspace` and `LinearSubspace` code snippets
  updated from stale direct constructors to the `from_layout()` factory.
- API overview: `SolverState(X=X_init)` -> `solver.init_state(X_init)`.

### Test suite

- Modernized all geometry tests to the new `(dim, radius)` API; fixed 17
  failing tests.
- Deleted duplicate `test_softmax_correctness.py` (was a copy of tests
  already in `test_softmax.py`).
- Cleaned up numbered tags and stale pre/post-fix prose across ten
  test files.
- Net: 905 collected tests across 45 files; fast-tier CI suite passes
  880 (+ 2 skipped, 23 deselected) in ~20 s.

### CI

- Removed 11 `--ignore` entries from CI (7 pointed to non-existent files,
  1 to a previously deleted file). CI now runs all non-slow, non-GPU tests
  without explicit ignores.
- Fast experiment-infrastructure tests (`test_baseline_fairness`,
  `test_nondiff_data`, `test_nondiff_models`, `test_rl_benchmarks`) are
  now included in CI (+98 tests, <2 s overhead).

## 0.2.1 - 2026-05-14

### Added

- `examples/06_loihi_snn_polystep.py`: end-to-end skeleton for a
  Loihi 2-style two-stage workflow. Stage 1 pretrains a hard-LIF
  MNIST SNN with PolyStep using `PSTORCH_CONFIGS["snn"]`. Stage 2
  adapts only the writable subset a real Loihi 2 chip exposes at
  runtime (`fc2`, per-population `vth`, and `beta`; about 1.3% of
  model parameters) under an `N(0, 1)` Gaussian input shift. Stage 2
  uses TENT-style safeguards: mixed-batch (half clean / half shifted),
  rank-8 probing on the writable subspace, and two probes per step.
  Both stages use best-test early stopping (patience 4) since
  zeroth-order test curves are noisier. Shifted-test evaluations
  share a fixed seeded noise mask across pre / post / baseline so the
  reported recovery is a paired comparison. On default settings the
  example reaches ~83% best clean accuracy and a +13 pp paired
  shift-recovery over the frozen-readout baseline with near-zero
  clean-accuracy degradation in about 17 minutes on a single GPU.
  Reported numbers shift by a few percentage points run-to-run on
  CUDA because of non-deterministic cuBLAS reductions, but the
  qualitative recovery is robust. The host loop is backend-agnostic:
  `LoihiSpikeEvaluator` is the single swap point against a Lava
  `netx` deployment path.

## 0.2.0 - 2026-05-14

Dependency floor lift, native fast-path adoption, and performance pass for
PyTorch 2.12.

### Changed

- Minimum Python is now 3.11 (was 3.10); minimum PyTorch is now 2.8 (was 2.4);
  NumPy floor is now `>=2.0`.
- Ruff `target-version` bumped to `py311`. Classifiers updated to advertise
  3.11 / 3.12 / 3.13 / 3.14.

### Performance

- `SinkhornSolver` convergence loop: batched the per-check scalar
  measurements (err_a, err_b, dual norm, Lyapunov) into a single
  device-to-host transfer, and switched the Anderson-acceleration
  Lyapunov accept gate to a device-side `torch.where`. Eliminates 4-7
  GPU-CPU syncs per `check_every` interval. Microbench (200 iter,
  n=m=512, fp32 on RTX 5090, full Anderson + adaptive omega): **490 ms
  -> 48 ms (~10x)**.
- Block-wise step (`_step_blockwise.py`): per-block displacement and
  model-loss scalars are accumulated as device tensors and reduced to
  the host once per step instead of twice per block.
- `NNCostEvaluator.evaluate` and `BatchedLinearEvaluator.evaluate` now
  use `@torch.inference_mode()` instead of `@torch.no_grad()` (matches
  the rest of `cost_nn.py`).

### Verified

- Full test suite (CPU + CUDA) green on PyTorch 2.12 / Python 3.14 / CUDA 13.0
  on RTX 5090 (sm_120).
- All five `examples/` reproduce within tolerance (≤1% accuracy / ≤5% returns).
- `torch.compile(fullgraph=True)` (default in `_compiled.py:try_compile`)
  warns on bypass in PyTorch 2.12 and will hard-error in 2.13; behavior
  documented for the next dependency lift.

### Test suite

- Consolidated overlapping test files.
- Net: 935 -> 906 collected tests across 45 files (was 49); fast-tier suite
  passes 885 (+ 2 skipped, 19 deselected) in ~21 s on RTX 5090.

## 0.1.0 - 2026-05-03 (Initial Release)

First public release alongside the arXiv preprint *Training Non-Differentiable
Networks via Optimal Transport*.

### Core

- `PolyStepOptimizer` for gradient-free training of any `nn.Module`
- Softmax solver (default; fast path) and log-domain Sinkhorn solver with
  warm-started dual potentials
- Polytope sampling (orthoplex, simplex, cube) with random rotations
- Vectorised NN evaluation via `torch.func` + optional `torch.vmap`
- High-level `train()` API with callbacks and early stopping

### Subspace compression

- `LinearSubspace` - random projection
- `AdaptiveSubspace` - rotating orthogonal projection
- `HybridSubspace` - per-layer projections (recommended)
- `CMAAdaptiveSubspace` - CMA-ES covariance adaptation
- `SparseRandomProjection` - for models with 100M+ parameters

### Scalability

- Block-wise per-layer OT decomposition
- Chunked cost evaluation for memory control
- `torch.compile` support on GPU hot paths
- Mixed precision (BF16 model + FP32 solver)

### Layers

- `VmapSafeMultiHeadAttention` - attention compatible with `torch.vmap`
- `VmapSafeLSTM` - LSTM compatible with `torch.vmap`

### Examples and experiments

- 5 runnable demos under `examples/` (quickstart, SNN, RL, MAX-SAT, MNIST)
- Paper-reproduction harness under `experiments/runners/` covering SNN,
  INT8, argmax, staircase, hard MoE, MAX-SAT (100K-1M vars), MNIST,
  ETTh1 timeseries, RL, and GPT-2 head-only fine-tuning
- Ablations: OT vs softmax solver, epsilon / radius / particles / subspace
  grid, MAX-SAT scaling

### Verification

- 879 unit tests pass (`pytest -q -m "not slow"`); ruff lint clean
- Reported numbers verified against multi-seed result JSONs (seeds
  `{42, 123, 456, 789, 1337}`)
