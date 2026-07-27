# Examples

Run in order. Times are measured on one CPU core; every example pins
`torch.set_num_threads(1)`. Unpinned, these times swing with whatever else holds the
cores: example 02 measured 2.5 s and 105.9 s on back-to-back runs. Set
`POLYSTEP_THREADS` to override. Example 06 is GPU-oriented and takes
about 13 minutes on an RTX 5090.

Optional dependencies (matplotlib, gymnasium, pysat) are checked before use, so a
missing one skips the figure or the render instead of failing after the work is done.

| # | File | What it shows | Time |
|---|------|---------------|------|
| 01 | [`01_quickstart_2d.py`](01_quickstart_2d.py) | Polytope sampling on a 2D staircase objective | ~1 s |
| 02 | [`02_snn_starter.py`](02_snn_starter.py) | SNN with hard LIF spikes (non-differentiable) | ~5 s |
| 03 | [`03_rl_cartpole.py`](03_rl_cartpole.py) | Direct policy search on CartPole-v1 | ~3 s |
| 04 | [`04_maxsat_10k.py`](04_maxsat_10k.py) | Random 3-SAT with 10K variables, gradient-free | ~50 s |
| 05 | [`05_mnist.py`](05_mnist.py) | MNIST training with `PolyStepOptimizer` | ~3 min |
| 06 | [`06_loihi_snn_polystep.py`](06_loihi_snn_polystep.py) | Loihi 2 skeleton: MNIST SNN pretrain, then on-chip readout adaptation under input shift | ~13 min (GPU) |
| 07 | [`07_binary_net_no_ste.py`](07_binary_net_no_ste.py) | STE-free binary (sign-activation) net via ask/tell, scored on 0-1 error against OpenAI-ES | ~9 s |
| 08 | [`08_direct_loss_minimization.py`](08_direct_loss_minimization.py) | Directly maximize a non-decomposable metric (F1) on an imbalanced checkerboard, against Adam+STE and OpenAI-ES over 5 seeds | ~4 s |
| 09 | [`09_hard_decision_tree.py`](09_hard_decision_tree.py) | Train a hard oblique decision tree (strict argmax routing, no relaxation) on an XOR checkerboard, against OpenAI-ES, SPSA and a soft-tree Adam baseline scored after hardening | ~6 s |

## Quick start

```bash
pip install -e ".[examples]"
python examples/01_quickstart_2d.py
```

For paper reproduction, see [`experiments/`](../experiments/).

Numbers comparing PolyStep against the baselines live in
[`experiments/BENCHMARKS.md`](../experiments/BENCHMARKS.md).
