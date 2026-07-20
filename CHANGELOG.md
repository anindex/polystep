# Changelog

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
  Headline numbers shift by a few percentage points run-to-run on
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
  INT8, argmax, staircase, hard MoE, MAX-SAT (100K–1M vars), MNIST,
  ETTh1 timeseries, RL, and GPT-2 head-only fine-tuning
- Ablations: OT vs softmax solver, epsilon / radius / particles / subspace
  grid, MAX-SAT scaling

### Verification

- 879 unit tests pass (`pytest -q -m "not slow"`); ruff lint clean
- Headline numbers verified against multi-seed result JSONs (seeds
  `{42, 123, 456, 789, 1337}`)
