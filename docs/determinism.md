# Determinism

Two runs of the same seed should produce the same numbers. Seeding alone
does not get you there on CUDA: kernel autotuning, TF32 and the
DataLoader shuffle order all vary independently of the RNG seed. This is
what the repo pins, and what it cannot.

## What is pinned

`experiments/runners/common.py`:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set at **module import time**, not
  inside `set_seed`. cuBLAS reads it once, when the CUDA context is
  created; setting it after any CUDA call is a no-op. Every runner
  imports `common` before it builds a model, so import time is the one
  place that is always early enough. Set via `os.environ.setdefault`, so
  an operator-supplied value wins.
- `set_deterministic()`, called by `set_seed()`:
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
  - `torch.backends.cudnn.deterministic = True`, `benchmark = False`
  - `torch.backends.cuda.matmul.allow_tf32 = False`,
    `torch.backends.cudnn.allow_tf32 = False`. TF32 is on by default on
    Ampere and later, and its truncated mantissa makes the result depend
    on which kernel the autotuner happened to pick.
- Every DataLoader gets `generator=` and `worker_init_fn=` from
  `polystep.benchmarks.utils.seeded_loader_kwargs()`. The generator is
  seeded from `torch.initial_seed()` by default, so shuffle order still
  differs per experiment seed but no longer depends on how much of the
  global RNG the rest of the process consumed first. `worker_init_fn`
  re-seeds numpy and `random` inside worker processes, which torch does
  not do for you.

## Non-deterministic ops

`warn_only=True`, so a non-deterministic op warns instead of raising. As
of this writing nothing in the main experiments trips it: a single
PolyStep step on CUDA under `warn_only=False` runs clean for `MNISTNet`,
`SpikingMNISTNet`, `QuantizedMLP`, `DiscreteAttentionNet`,
`StaircaseNet` and `HardMoENet`, as does an Adam forward/backward/step on
`MNISTNet`.

`warn_only=True` is kept anyway, because a hard failure on a benchmark
outside that set (RL, GPT-2 fine-tuning, the Tonic event datasets) would
abort a long run over a warning. If you add a benchmark, run it once with
`warn_only=False` and record what breaks here.

## Verifying

`tests/test_determinism.py` runs a tiny MNIST PolyStep training twice with
seed 42, in-process, and asserts the loss trajectories are identical. The
CUDA version of the same check is marked `gpu`:

```
pytest tests/test_determinism.py          # CPU
pytest tests/test_determinism.py -m gpu   # CUDA
```

## The rotation sampler switches algorithm on batch size

`polystep.geometry.get_random_rotation_matrices` draws Haar rotations two
ways: a batched QR below `_QR_MAX_BATCH` (256 on CPU, 16 on CUDA) and
Diaconis-Shahshahani Householder reflections above it. Both are Haar and
both are seeded, but they consume the generator stream differently.

Two consequences worth knowing before you compare runs:

- The batch is the particle count, so seed 7 at `particle_dim=16` and seed
  7 at `particle_dim=300` are not the same sequence of rotations. Compare
  across seeds within one configuration, not across configurations.
- The threshold is also keyed on device type, so the same seed gives
  different rotations on CPU and CUDA.

Neither affects a run reproducing itself, which is what the tests check.

## Still not guaranteed

Bit-identical results across *different* GPUs, CUDA/cuDNN versions or
CPU thread counts. Determinism here means the same machine and the same
build reproduce themselves.
