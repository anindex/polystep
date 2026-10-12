# Examples

```bash
pip install -e ".[examples]"
python examples/01_quickstart_2d.py
```

| File | Task |
|---|---|
| [01_quickstart_2d.py](01_quickstart_2d.py) | Polytope search on a staircase objective |
| [02_snn_starter.py](02_snn_starter.py) | Spiking network with hard LIF thresholds |
| [03_rl_cartpole.py](03_rl_cartpole.py) | CartPole policy search |
| [04_maxsat_10k.py](04_maxsat_10k.py) | Random 3-SAT |
| [05_mnist.py](05_mnist.py) | MNIST classifier |
| [06_loihi_snn_polystep.py](06_loihi_snn_polystep.py) | SNN readout adaptation under input shift |
| [07_binary_net_no_ste.py](07_binary_net_no_ste.py) | Binary network without a straight-through estimator |
| [08_direct_loss_minimization.py](08_direct_loss_minimization.py) | Direct F1 optimization |
| [09_hard_decision_tree.py](09_hard_decision_tree.py) | Decision tree with hard routing |
| [10_cnn_mnist.py](10_cnn_mnist.py) | Convolutional MNIST classifier |
| [11_transformer_selective_copy.py](11_transformer_selective_copy.py) | Attention-based selective copying |

Examples run on one CPU thread, except 07, which uses eight.
To change the thread count, set `POLYSTEP_THREADS`.
Examples 02, 04, 05, 06, 10, and 11 use CUDA when available. Example 05 uses BF16 on CUDA.
Example 03 accepts `--device cuda`. To skip its GIF, use `--no-render`.
For CPU runs of 02, 05, 06, 10, and 11, use `--device cpu`.
For a smaller MAX-SAT task, use `--small`.

Examples 10 and 11 support `--compare` to check loss agreement and time registered
evaluation against materialized candidates, including first-call setup. See
[performance](../docs/performance.md) for subspace settings and the
[API reference](../docs/api_overview.md#evaluation-controls) for amortization.

Examples 05, 06, 10, and 11 select checkpoints on validation data and report
test accuracy afterward. The Loihi example simulates the SNN in PyTorch on the
selected device; its hardware evaluator is a deployment sketch.

Examples 07 and 08 report the mean over five seeds. Their OpenAI-ES baselines use
the noise scale and learning rate that scored best on 20 seeds the examples do not
report.

For published comparisons, see [experiments](../experiments/README.md).
