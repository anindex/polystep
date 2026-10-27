# Examples

Run these examples from a source checkout. The package's `examples` extra installs
dependencies; the scripts live in this repository:

```bash
pip install -e ".[examples]"
python examples/01_quickstart_2d.py
```

| File | Task |
|---|---|
| [01_quickstart_2d.py](01_quickstart_2d.py) | Polytope search on a staircase objective |
| [02_snn_starter.py](02_snn_starter.py) | Spiking network with hard LIF thresholds |
| [03_rl_cartpole.py](03_rl_cartpole.py) | CartPole policy search with hard left/right actions |
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

For a short CPU check, I use:

```bash
python examples/06_loihi_snn_polystep.py --device cpu --hidden 32 --num-steps 5 --max-train 1024 --max-test 256 --pretrain-epochs 1 --adapt-epochs 1
python examples/10_cnn_mnist.py --device cpu --epochs 2 --train-size 1000 --test-size 256 --batch-size 32
python examples/11_transformer_selective_copy.py --device cpu --epochs 5 --train-size 512 --test-size 256
```

These small budgets check execution; use the defaults to assess learning quality.

## Data

Examples 05, 06 and 10 download MNIST to `mnist` under the system temp directory,
`/tmp/mnist` on Linux. Example 06 takes `--data-dir` to use another directory.

## Comparing evaluation paths

Examples 10 and 11 take `--compare` to score the same batches with fully built
candidates and with the site-local evaluator. This checks that the losses agree and
times both paths, including first-call setup. On the CPU, example 10's fully built
candidates take about 15 GB of memory. See [performance](../docs/performance.md) for subspace settings
and the [API reference](../docs/api_overview.md#evaluation-controls) for amortization.

## Reported results

Examples 05, 06, 10 and 11 select checkpoints on validation data and report test
accuracy afterward. Example 06 runs a PyTorch SNN simulation on the selected device.
It adapts the readout and neuron parameters under Gaussian input noise. Its
one-state neuron uses subtractive reset, so it does not reproduce
[Lava LIF](https://lava-nc.org/lava/lava.proc.lif.html) or Loihi hardware.
`--backend cpu_sim` remains accepted; `--backend loihi2` fails before loading data.

Examples 07 and 08 report the mean over five seeds. Their OpenAI-ES baselines use the
noise scale and learning rate selected on 20 separate tuning seeds.

Examples 07 to 09 measure the objective on the data used for optimization; those
scores demonstrate fitting, not generalization. Their gradient baselines have a
separate training-step budget. Example 08 reports sample standard deviations across
its five seeds.

For published comparisons, see [experiments](../experiments/README.md).

I ran the default budgets on an RTX 5090 with v1.1.1, Python 3.14 and PyTorch 2.13.
Test accuracy reached 94.4% in example 05, 94.35% in example 10, and 100% on example
11's synthetic copy task. Example 06 recovered 13.3 percentage points on
shifted test inputs, while clean accuracy fell from 73.2% to 72.0%. These are the
scripts' default seeds, not multi-seed estimates. The CPU CartPole run reached its
200-step evaluation limit on all 32 episodes.
