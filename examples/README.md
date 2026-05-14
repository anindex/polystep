# Examples

Run in order. Examples 01–05 run in under 2 minutes on a laptop CPU; example 06 is GPU-friendly and takes ~17 min on CUDA (it pretrains a full SNN with best-test early stopping, then runs an on-chip-style adaptation phase).

| # | File | What it shows | Time |
|---|------|---------------|------|
| 01 | [`01_quickstart_2d.py`](01_quickstart_2d.py) | Polytope-sampling on a 2D staircase objective | ~10s |
| 02 | [`02_snn_starter.py`](02_snn_starter.py) | SNN with hard LIF spikes (non-differentiable) | ~60s |
| 03 | [`03_rl_cartpole.py`](03_rl_cartpole.py) | Direct policy search on CartPole-v1 | ~30s |
| 04 | [`04_maxsat_10k.py`](04_maxsat_10k.py) | Random 3-SAT (10K vars) via gradient-free optimization | ~60s (GPU) |
| 05 | [`05_mnist.py`](05_mnist.py) | MNIST training with the `train()` API | ~2min |
| 06 | [`06_loihi_snn_polystep.py`](06_loihi_snn_polystep.py) | **Loihi 2 skeleton**: MNIST SNN pretrain + on-chip readout adaptation under input shift (**+13.1 pp** paired-noise shift-recovery on a 1.3 % writable subset, *no clean-accuracy degradation*) | ~17min (GPU) |

## Quick start

```bash
pip install -e ".[examples]"
python examples/01_quickstart_2d.py
```

For paper reproduction, see [`experiments/`](../experiments/).
