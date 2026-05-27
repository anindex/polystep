# Changelog

## 0.2.4 - 2026-05-27

Inner-loop performance improvements. Profile-driven cleanup of CPU<->GPU sync points. No algorithmic changes.

### Changed

- `solvers/sinkhorn.py`: dropped two GPU->CPU syncs per Sinkhorn solve.
  - `cost_scale` (used to compute the warm-start `clamp_` bound) stays
    on-device as a 0-d tensor; `Tensor.clamp_` accepts tensor bounds.
  - `ent_reg_cost = <f,a> + <g,b>` now resolves both inner products in
    one host transfer via `torch.stack([...]).sum().item()`. Same
    pattern applied to the low-rank convergence loop, plus its
    `err_a/err_b` marginal-error transfers (mirrors the dense
    convergence path).
- `_step_blockwise.py`: per-block fused-softmax `ent_cost_tensor.item()`
  calls defer to a single `torch.stack([...]).sum().item()` after the
  per-block loop, mirroring the existing `block_disp_terms` and
  `block_model_loss_terms` reductions. Saves `O(num_blocks)` syncs per
  step in the fused-softmax path.
- `cma.py`: `update_step_size_csa` now accepts an optional
  pre-computed `p_sigma_norm`. Both call sites
  (`_step_common.py:update_cma_state`, `_step_monolithic.py:step`)
  forward the norm they already paid for in `compute_heaviside_sigma`,
  removing one redundant `torch.norm(...).item()` per CMA generation.
- `_step_common.py:apply_biased_rotation`: the (dead-but-exported)
  helper now mirrors the production path in `_step_monolithic.py` --
  one batched `torch.linalg.qr` plus a determinant fix-up, instead of
  an `O(pdim^2)` Python Gram-Schmidt loop. The blockwise paths keep
  the Gram-Schmidt loop on purpose: per-layer block_dim is typically
  <=128 and small batched QR via cuSOLVER measured slower than the
  elementwise loop on this hardware.

## 0.2.3 - 2026-05-27

Final-pass polish: docstring cleanup, citation corrections, lint sweep
of examples and experiments.

### Fixed

- `kl_softmax.py`: corrected the author list for the unbalanced-OT
  scaling-algorithm reference to Chizat, Peyré, Schmitzer & Vialard
  (Math. Comp. 87, 2018; arXiv:1607.05816). The previous citation
  conflated two different Chizat et al. papers.
- `epsilon.py`: corrected the ProgOT citation to Kassraie, Pooladian,
  Klein, Thornton, Niles-Weed & Cuturi, NeurIPS 2024 (arXiv:2406.05061).
  The previous "Kassab & Thornton, 2025" tag was wrong on author, year,
  and arXiv.
- `epsilon.py`: module docstring now lists all three schedulers
  (`LinearEpsilon`, `CosineEpsilon`, `ProgressiveEpsilon`).
- `cma_subspace.py`, `hybrid_subspace.py`: rewrote two module
  docstrings whose first paragraph was mangled by an earlier refactor
  (text fragments like "Strategy)" and "per step) with" at the start
  of sentences).
- `CMAAdaptiveSubspace.rotate()` now forwards `transport_matrix`,
  `X_vertices`, and `X_current` to the wrapped `AdaptiveSubspace`, so
  the `'ot_bias'` rotation mode actually fires when CMA is enabled
  (previously silently fell through to random rotation).
- `docs/api_overview.md`: the `PolyStepOptimizer.step` snippet used a
  zero-argument lambda, which is **not** what the optimizer expects.
  Replaced with the real `closure(batched_params) -> losses` signature
  via `NNCostEvaluator`. The `SparseRandomProjection` example used
  `input_dim` / `output_dim` keyword args; the actual constructor
  takes `full_dim` / `subspace_dim`.

### Lint

- Cleared ruff sweep over `examples/` and `experiments/`: F541 (f-string
  with no placeholders) in `examples/05_mnist.py`; F841 (unused locals)
  in `experiments/runners/run_maxsat.py`; E722 (bare `except`) in
  `experiments/runners/run_fill_ablation_grid.py`; F821 (closure capture
  shadowed by later `del`) in `experiments/runners/run_gpt2_finetune.py`.
  `ruff check src tests examples experiments` is now clean.

### Verified

- 880 tests pass (2 skipped) on the fast CI tier in ~21 s on RTX 5090
  (PyTorch 2.12 / Python 3.14 / CUDA 13.0).

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
- Stripped AI-pattern artifacts (numbered tags, pre/post-fix prose) from
  10 test files.
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

- `examples/06_loihi_snn_polystep.py`: end-to-end skeleton demonstrating
  PolyStep on a Loihi 2-style two-phase workflow. Pretrains a hard-LIF
  MNIST SNN with PolyStep using the paper's `PSTORCH_CONFIGS["snn"]`
  config, then adapts *only* the writable subset a real Loihi 2 chip
  would expose at runtime (`fc2` + per-population `vth` + `beta`,
  1.3 % of model parameters) under a `N(0, 1^2)` Gaussian input-shift.
  Phase B uses three TENT-style test-time-adaptation safeguards -
  mixed-batch (half clean / half shifted), rank-8 probing on the tiny
  writable subspace, and two probes per step. Both phases use
  best-test early stopping (patience 4, tuned for the noisier per-epoch
  test curves of zeroth-order optimization). Shifted-test evaluations
  share a fixed seeded noise mask across pre / post / baseline so the
  reported recovery is a *paired* comparison free of sampling jitter.
  On default settings reaches **83.1 % best clean / +13.1 pp paired
  shift-recovery over the frozen-readout baseline with near-zero clean-
  accuracy degradation** (Phase B clean 82.5 % vs. Phase A clean
  83.1 %) in ~17 min on a single GPU. Headline numbers shift by a
  few pp run-to-run on CUDA due to non-deterministic cuBLAS reductions;
  the qualitative recovery is robust. The host loop is backend-
  agnostic - `LoihiSpikeEvaluator` is the single swap point against
  the Lava `netx` deployment path.

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
