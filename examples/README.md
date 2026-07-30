# Examples

Run in order. Every example pins a thread count, because torch's default of `nproc`
collapses: `03` runs 2.1 s on one thread against 427 s on 24. All pin 1 except `07`,
whose own objective is wide enough to pay for the pool (11.3 s to 4.1 s at 8).
`POLYSTEP_THREADS` overrides. See [`../docs/performance.md`](../docs/performance.md).

`02`, `04`, `05`, `06`, `10` and `11` use CUDA when present and fall back to CPU. `01`,
`03`, `07`, `08` and `09` stay on CPU: their objectives are small enough that
kernel-launch overhead outweighs the device (`03` measured 3.0 s CPU against 4.7 s CUDA).
`03` still takes `--device cuda`.

Times below are indicative, from one GPU box. `05`, `06`, `10` and `11` take
`--device cpu` to force CPU; `04` picks the device itself and takes `--small`.

Optional dependencies (matplotlib, gymnasium, pysat) are checked before use, so a missing
one skips the figure or the render instead of failing after the work is done.

Two levers set the wall-clock in 05, 10 and 11. `amortize_steps` runs momentum steps that
evaluate nothing between OT steps, cutting forward passes by roughly its value; it spends
step budget rather than work, so a step-starved run needs a smaller batch to pay for it
(example 11 halves its batch). Subspace rank cuts candidates per step: it helps the CNN,
whose descent direction is low-rank, and costs the MLP accuracy. Both have a sharp,
seed-dependent cliff just past the shipped settings, tabulated in
[`../docs/performance.md`](../docs/performance.md).

Examples 10 and 11 train a convolutional net and a transformer. Neither fits the
`nn.Sequential`-of-`Linear` shape the delta evaluators need, so both take the site-aware
path: a candidate differs inside one parameter tensor, so only that tensor is batched and
the layers ahead of it run once. `--compare` times it against the materializing path on
identical batches. It takes one `register_evaluator(...)` call; without it the step sees
only `closure()` and materializes.

| # | File | What it shows | Time |
|---|------|---------------|------|
| 01 | [`01_quickstart_2d.py`](01_quickstart_2d.py) | Polytope sampling on a 2D staircase objective | 1 s |
| 02 | [`02_snn_starter.py`](02_snn_starter.py) | SNN with hard LIF spikes (non-differentiable) | 2 s |
| 03 | [`03_rl_cartpole.py`](03_rl_cartpole.py) | Direct policy search on CartPole-v1 | 2 s |
| 04 | [`04_maxsat_10k.py`](04_maxsat_10k.py) | Random 3-SAT with 10K variables, gradient-free | 14 s (GPU) |
| 05 | [`05_mnist.py`](05_mnist.py) | MNIST training with `PolyStepOptimizer` | 22 s (GPU) |
| 06 | [`06_loihi_snn_polystep.py`](06_loihi_snn_polystep.py) | Loihi 2 skeleton: MNIST SNN pretrain, then on-chip readout adaptation under input shift | 3.5 min (GPU) |
| 07 | [`07_binary_net_no_ste.py`](07_binary_net_no_ste.py) | STE-free binary (sign-activation) net via ask/tell, scored on 0-1 error against OpenAI-ES | 4 s |
| 08 | [`08_direct_loss_minimization.py`](08_direct_loss_minimization.py) | Directly maximize a non-decomposable metric (F1) on an imbalanced checkerboard, against Adam+STE and OpenAI-ES over 5 seeds | 4 s |
| 09 | [`09_hard_decision_tree.py`](09_hard_decision_tree.py) | Train a hard oblique decision tree (strict argmax routing, no relaxation) on an XOR checkerboard, against OpenAI-ES, SPSA and a soft-tree Adam baseline scored after hardening | 6 s |
| 10 | [`10_cnn_mnist.py`](10_cnn_mnist.py) | LeNet-5 on MNIST, forward passes only. Convolutions and a hand-written `forward`, which every earlier fast path declines | 36 s (GPU, 40 epochs) |
| 11 | [`11_transformer_selective_copy.py`](11_transformer_selective_copy.py) | Transformer solving a pointer-following task that needs attention, forward passes only | 4 s (GPU) |

## Quick start

```bash
pip install -e ".[examples]"
python examples/01_quickstart_2d.py
```

For paper reproduction, see [`experiments/`](../experiments/).

Numbers comparing PolyStep against the baselines are in the [benchmark
tables](../README.md#benchmarks) and
[`experiments/EXPERIMENT_INDEX.md`](../experiments/EXPERIMENT_INDEX.md).
