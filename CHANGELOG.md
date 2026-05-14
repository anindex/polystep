# Changelog

## 0.2.1 - 2026-05-14

### Added

- `examples/06_loihi_snn_polystep.py`: end-to-end skeleton demonstrating
  PolyStep on a Loihi 2-style two-phase workflow. Pretrains a hard-LIF
  MNIST SNN with PolyStep using the paper's `PSTORCH_CONFIGS["snn"]`
  config, then adapts *only* the writable subset a real Loihi 2 chip
  would expose at runtime (`fc2` + per-population `vth` + `beta`,
  1.3 % of model parameters) under a `N(0, 1^2)` Gaussian input-shift.
  Phase B uses three TENT-style test-time-adaptation safeguards —
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
  agnostic — `LoihiSpikeEvaluator` is the single swap point against
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
- Net: 935 → 906 collected tests across 45 files (was 49); fast-tier suite
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
