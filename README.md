# polystep

[![PyPI](https://img.shields.io/pypi/v/polystep.svg)](https://pypi.org/project/polystep/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01928-b31b1b.svg)](https://arxiv.org/abs/2605.01928)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/anindex/polystep/blob/main/LICENSE)

Gradient-free neural network training via optimal transport.

PolyStep evaluates finite-radius parameter perturbations and updates parameters
through a weighted barycenter. It supports spiking networks, quantized layers, hard
routing, and other models without useful gradients. For differentiable models,
backpropagation is usually faster.

[Paper](https://arxiv.org/abs/2605.01928) ·
[Interactive visualization](https://vietngth.github.io/polystep-visualization/) ·
[API reference](docs/api_overview.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/anindex/polystep/main/docs/figures/method_diagram.png" width="840"
       alt="PolyStep: subspace projection, polytope probes, loss evaluation, and a weighted parameter update.">
</p>

> Want to play around with parameters? [**Viet T. Nguyen**](https://vietngth.github.io/) built a gorgeous interactive walkthrough that animates every step of the method -> **[explore the PolyStep visualization](https://vietngth.github.io/polystep-visualization/)**.

## Installation

```bash
pip install polystep
pip install "polystep[examples]"  # optional demo dependencies
```

Python 3.11+ and PyTorch 2.8+ are required. Choose a CUDA build compatible with your
driver from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
For development, install `pip install -e ".[dev]"` from a checkout.

## Quickstart

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from polystep import HybridSubspace, PolyStepOptimizer, TrainConfig, train
from polystep.transform import ParamLayout

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
loader = DataLoader(
    TensorDataset(torch.randn(256, 16), torch.randint(0, 3, (256,))),
    batch_size=32,
)
subspace = HybridSubspace.from_layout(
    ParamLayout.from_module(model), rank=4, max_subspace_dim=128,
)
optimizer = PolyStepOptimizer(model, subspace=subspace, solver="softmax")
train(model, loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=5))
```

Replace the generated data with your dataset. Limit `max_subspace_dim` to control
the forward budget. In manual loops, call `register_evaluator()` before each batch;
`train()` does this automatically. See [performance](docs/performance.md) for
candidate paths, thread counts, compilation, and memory settings.

### Ask/tell

```python
from polystep import PolyStepES

es = PolyStepES(dim=20, epsilon=0.1, step_radius=0.3)
for _ in range(300):
    candidates = es.ask()
    es.tell(candidates.square().sum(dim=1))
print(es.best_fitness, es.best_solution)
```

### Synthetic objectives

```python
from polystep import Ackley
from polystep.solver import PolyStep

solver = PolyStep.create(Ackley(dim=10), epsilon=0.5, max_iterations=50)
state = solver.run(torch.randn(100, 10))
```

## Method

Each step samples a rotated polytope, evaluates its vertices, computes transport
weights, and moves to their barycenter. Subspaces reduce the number of coordinates
searched. Softmax treats blocks independently; Sinkhorn also constrains target mass.

Finite-radius probes can cross flat regions of discontinuous losses. Performance
depends on the radius, representation, and evaluation budget. PolyStep's optional
quadratic models and radius controllers are implementation heuristics; they do not
inherit classical direct-search or trust-region guarantees.

## Baselines

`polystep.baselines` provides `openai_es`, `spsa`, `mezo`, `random_search`, `eggroll`,
and `cma_es`. One evaluation means one candidate scored, even when candidates are
batched:

```python
from polystep.baselines import Objective, openai_es

objective = Objective(lambda x: x.square().sum(1), dim=64, budget=10_000)
result = openai_es(objective, sigma=0.05, lr=0.1, popsize=32)
```

The paper runners support `--fair` for matched search spaces and evaluation budgets.
EGGROLL uses a factored representation to retain matrix perturbations. See the
[reproduction protocol](docs/reproducibility.md) for tuning and budget details.

## Benchmarks

Five-seed mean ± sample standard deviation. These architecture tables compare
validation-tuned methods at matched optimizer steps. Evaluation- and time-matched
studies are reported separately in the [experiment index](experiments/EXPERIMENT_INDEX.md).

<!-- BENCH:START -->
### Non-differentiable tasks

Test accuracy %. A dash means the cell is not in this release.

| Task | PolyStep | CMA-ES | OpenAI-ES | SPSA | Adam (surrogate) | Non-diff op |
|---|---|---|---|---|---|---|
| SNN/LIF (MNIST) | 93.0 ± 0.2 | 77.1 ± 13.3 | 79.6 ± 5.2 | 53.9 ± 3.0 | 86.3 ± 10.6 | `threshold()` |
| Int8 quantized | 97.0 ± 0.1 | 91.6 ± 0.4 | 94.4 ± 0.2 | 82.5 ± 0.5 | 97.8 ± 0.1 | `round()` |
| Argmax attention | 86.6 ± 0.4 | 79.5 ± 0.5 | 80.0 ± 0.4 | 67.4 ± 1.3 | 88.6 ± 0.1 | `argmax()` |
| Staircase activation | 94.3 ± 0.1 | 89.2 ± 0.4 | 88.9 ± 0.5 | 45.2 ± 4.2 | 97.5 ± 0.1 | `floor()` |
| Hard MoE routing | 90.6 ± 0.2 | 77.8 ± 1.5 | 82.5 ± 0.8 | 25.4 ± 3.5 | - | `argmax()` |

### MAX-SAT (% clauses satisfied)

| Variables | PolyStep | CMA-ES | OpenAI-ES | probSAT | RC2 |
|---|---|---|---|---|---|
| 100 | 98.0 ± 0.7 | 99.0 ± 0.3 | 99.4 ± 0.1 | 99.8 | 99.8 |
| 5,000 | 98.1 ± 0.1 | 95.2 ± 0.2 | 92.9 ± 0.2 | 99.9 | timeout |
| 100,000 | 98.1 ± 0.0 | 90.7 ± 0.1 | 88.9 ± 0.0 | 99.6 | timeout |

### Differentiable sanity checks

| Task | PolyStep | Adam |
|---|---|---|
| MNIST (2-layer MLP) | 96.8 ± 0.1 | 97.7 ± 0.1 |
| ETTh1 (LSTM, MSE; lower is better) | 0.253 ± 0.023 | 0.247 ± 0.023 |
<!-- BENCH:END -->

The SNN surrogate-gradient baseline varies substantially across seeds. Adam leads
on the differentiable tasks, and specialized solvers lead on MAX-SAT. See
[limitations](LIMITATIONS.md) for unsupported configurations and comparison limits.

## Documentation

- [API reference](docs/api_overview.md)
- [Performance](docs/performance.md)
- [Examples](examples/README.md)
- [Experiments](experiments/README.md) and [reproduction](docs/reproducibility.md)
- [Determinism](docs/determinism.md)
- [Contributing](CONTRIBUTING.md) and [changelog](CHANGELOG.md)

## Citation

```bibtex
@article{le2026training,
  title={Training Non-Differentiable Networks via Optimal Transport},
  author={Le, An T},
  journal={arXiv preprint arXiv:2605.01928},
  year={2026}
}
```

## Acknowledgments

A huge thank you to [**Viet**](https://vietngth.github.io/) for building a beautiful interactive [PolyStep visualization](https://vietngth.github.io/polystep-visualization/), it brings the method to life and makes every step click!

## License

[Apache-2.0](LICENSE).
