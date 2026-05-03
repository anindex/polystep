# Changelog

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

- 959 unit tests pass (`pytest -q -m "not slow"`); ruff lint clean
- Headline numbers verified against multi-seed result JSONs (seeds
  `{42, 123, 456, 789, 1337}`)
