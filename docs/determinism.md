# Determinism

Reproducibility requires the same configuration, device, software build, and CPU
thread count. A seed alone does not guarantee identical results across systems.

`experiments/runners/common.py` sets:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, unless already set.
- Deterministic PyTorch algorithms with `warn_only=True`.
- Deterministic cuDNN behavior, with benchmarking and TF32 disabled.
- Seeded DataLoader generators and worker initialization.

Warnings remain possible when an operation has no deterministic implementation.
Use `torch.use_deterministic_algorithms(True, warn_only=False)` to make these errors
explicit when validating a workload.

## Rotations

The rotation sampler uses batched QR for small particle batches and Householder
reflections for larger batches. Both sample Haar rotations but consume different
random streams. Changing the particle count or device can change the sequence.

## Checks

```bash
pytest tests/test_determinism.py
pytest tests/test_determinism.py -m gpu
```

Checkpoint continuation also requires the original optimizer configuration and
model weights; see the [API reference](api_overview.md#checkpoints).
