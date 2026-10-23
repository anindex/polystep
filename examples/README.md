# Examples

Each script runs PolyStep on a small task and prints the result; 07, 08 and 09 also run
baselines for comparison. Examples 01 to 04 and 06 to 09 save a figure to
`examples/figures/`.

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

## Devices and threads

Examples run on one CPU thread, except 07, which uses eight; set `POLYSTEP_THREADS` to
change it. Examples 02, 04, 05, 06, 10 and 11 use CUDA when available, and 05 runs in
BF16 there; pass `--device cpu` to run them on the CPU. Example 03 runs on the CPU,
accepts `--device cuda`, and skips its GIF with `--no-render`. Example 04 accepts
`--small` for a 2,000-variable task.

Example 10 trains in about 20 seconds on an RTX 5090. On one CPU thread its 40 default
epochs take about an hour, so shorten a CPU run with `--epochs` and `--train-size`.

## Data

Examples 05, 06 and 10 download MNIST to `mnist` under the system temp directory,
`/tmp/mnist` on Linux. Example 06 takes `--data-dir` to use another directory.

## Comparing evaluation paths

Examples 10 and 11 take `--compare`, which scores the same batches with fully built
candidates and with the site-local evaluator, checks that the losses agree, and times
both, first-call setup included. On the CPU, example 10's fully built candidates take
about 15 GB of memory. See [performance](../docs/performance.md) for subspace settings
and the [API reference](../docs/api_overview.md#evaluation-controls) for amortization.

## Reported results

Examples 05, 06, 10 and 11 select checkpoints on validation data and report test
accuracy afterward. The Loihi example simulates the SNN in PyTorch on the selected
device; its hardware evaluator is a deployment sketch.

Examples 07 and 08 report the mean over five seeds. Their OpenAI-ES baselines use the
noise scale and learning rate that scored best on 20 seeds the examples do not report.

For published comparisons, see [experiments](../experiments/README.md).
