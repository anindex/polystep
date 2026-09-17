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

Examples set CPU thread counts; override with `POLYSTEP_THREADS`. Examples 02, 04,
05, 06, 10, and 11 use CUDA when available. Example 03 accepts `--device cuda`.
Use `--device cpu` for 05, 06, 10, and 11; use `--small` for a smaller MAX-SAT task.

Examples 10 and 11 support `--compare` to time registered candidate evaluation against
materialized candidates. Runtime and accuracy depend on subspace dimension and
amortization; see [performance](../docs/performance.md).

For published comparisons, see [experiments](../experiments/README.md).
