# Examples

Run in order. Examples 01-05 and 07 run in under a couple of minutes on a laptop CPU; example 06 is GPU-friendly and takes about 17 minutes on CUDA. Example 06 pretrains a full SNN with best-test early stopping and then runs an on-chip-style adaptation stage.

| # | File | What it shows | Time |
|---|------|---------------|------|
| 01 | [`01_quickstart_2d.py`](01_quickstart_2d.py) | Polytope sampling on a 2D staircase objective | ~10 s |
| 02 | [`02_snn_starter.py`](02_snn_starter.py) | SNN with hard LIF spikes (non-differentiable) | ~60 s |
| 03 | [`03_rl_cartpole.py`](03_rl_cartpole.py) | Direct policy search on CartPole-v1 | ~30 s |
| 04 | [`04_maxsat_10k.py`](04_maxsat_10k.py) | Random 3-SAT with 10K variables, gradient-free | ~60 s (GPU) |
| 05 | [`05_mnist.py`](05_mnist.py) | MNIST training with `PolyStepOptimizer` | ~2 min |
| 06 | [`06_loihi_snn_polystep.py`](06_loihi_snn_polystep.py) | Loihi 2 skeleton: MNIST SNN pretrain + on-chip readout adaptation under input shift (~+13 pp paired shift-recovery on a ~1.3% writable subset, with near-zero clean-accuracy degradation) | ~17 min (GPU) |
| 07 | [`07_binary_net_no_ste.py`](07_binary_net_no_ste.py) | STE-free binary (sign-activation) net via ask/tell: PolyStepES vs OpenAI-ES on 0-1 error. Beats OpenAI-ES by ~20 points on the hard XOR-checkerboard boundary | ~15 s |

## Quick start

```bash
pip install -e ".[examples]"
python examples/01_quickstart_2d.py
```

For paper reproduction, see [`experiments/`](../experiments/).
