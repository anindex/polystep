# Determinism

Reproducibility requires the same configuration, device, software build, and CPU
thread count. A seed alone does not guarantee identical results across systems.

CPU QR factorizations below 1e8 of `rows * cols**2` work run on one thread. Larger
subspace bases (adaptive, hybrid, factored) use the whole intra-op pool, so their
rounding follows the thread count. Random rotation QRs stay on one thread at any size.

`experiments/runners/common.py` sets:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, unless already set.
- Deterministic PyTorch algorithms with `warn_only=True`.
- Deterministic cuDNN behavior, with benchmarking and TF32 disabled.
- Seeded DataLoader generators and worker initialization.

Warnings remain possible when an operation has no deterministic implementation.
Use `torch.use_deterministic_algorithms(True, warn_only=False)` to make these errors
explicit when validating a workload.

Without deterministic algorithms, CUDA scatter-adds sum repeated entries in whatever
order the threads finish. Hybrid runs with sparse projections use them, so two such
runs can differ in the last bit.

## Random streams

The rotation sampler uses batched QR for small particle batches and Householder
reflections for larger batches. Both sample Haar rotations but consume different
random streams. Changing the particle count or device can change the sequence.

`PolyStepES` with a `seed` draws from its own generator. Without one it draws from
PyTorch's global RNG, so `torch.manual_seed` before construction reproduces the run.

Dense subspace bases depend on the device as well as the seed. A dense
`AdaptiveSubspace` samples its bases on its generator's device, so a seeded CUDA run
draws a different stream from the same seed on the CPU. A dense `HybridSubspace`
basis is drawn on the CPU but orthogonalized by a QR on the model's device, so its
CUDA bases round differently. Sparse projections draw their rows and signs from their
own seeded CPU generator and skip the QR, so every device gets the same projection.

A candidate forward that draws its own random numbers, such as injected noise, is
retried under vmap with `randomness="different"`, which gives each candidate its own
draw. That stream differs from scoring the candidates one at a time, so a run whose
forward falls back to the sequential loop does not match one that stays batched.

## Closure tensors

The candidate tensors passed to a closure are borrowed: the next chunk may overwrite
their storage. A closure that keeps them after it returns, for logging or caching,
must clone them, or it reads later candidates' values.

## Checks

```bash
pytest tests/test_determinism.py
pytest tests/test_determinism.py -m gpu
```

Checkpoint continuation also requires the original optimizer configuration and
model weights; see the [API reference](api_overview.md#checkpoints).
