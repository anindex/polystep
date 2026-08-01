# polystep

[![PyPI](https://img.shields.io/pypi/v/polystep.svg)](https://pypi.org/project/polystep/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01928-b31b1b.svg)](https://arxiv.org/abs/2605.01928)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/anindex/polystep/blob/main/LICENSE)

**Gradient-free neural network training via optimal transport.**

PolyStep optimizes neural networks without backpropagation. At each step, it samples
polytope vertices around current parameters, evaluates losses via forward passes only,
and computes softmax-weighted projections to find descent directions. That trains models
with non-differentiable components (spiking networks, quantized layers, blackbox modules)
where gradients are unavailable or undefined.

Based on the Sinkhorn Step algorithm
([Le et al., NeurIPS 2023](https://arxiv.org/abs/2309.15970)), extended with subspace
compression, a softmax solver, and convergence analysis for piecewise-smooth losses.

## How it works

1. **Sample** polytope vertices around the current parameters in a compressed subspace.
2. **Evaluate** the loss at each vertex (forward pass only, no gradients).
3. **Compute** softmax weights over the cost matrix.
4. **Update** the parameters by barycentric projection from the weighted vertices.

<p align="center">
  <img src="https://raw.githubusercontent.com/anindex/polystep/main/docs/figures/method_diagram.png" width="840"
       alt="One PolyStep: subspace projection, polytope probes, cost matrix, soft entropic-OT assignment, and barycentric projection, plus the softmax-to-full-OT solver continuum.">
</p>

## Installation

```bash
pip install polystep                  # from PyPI (core: torch only)
uv add polystep                       # or with uv
```

From source:

```bash
pip install -e .                      # core library (torch only)
pip install -e ".[examples]"          # + numpy, torchvision, matplotlib, Pillow, gymnasium, python-sat
pip install -e ".[dev]"               # + numpy, pytest (+ plugins), ruff
pip install -e ".[experiments]"       # + numpy, pandas, python-sat, torchvision, cma, evotorch, snntorch, datasets, gymnasium
pip install -e ".[rl]"                # + gymnasium[box2d], stable-baselines3 (Box2D envs)
```

GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu130`, or pick the
CUDA build matching your driver from the
[PyTorch install page](https://pytorch.org/get-started/locally/).

## Quickstart

### Synthetic optimization

```python
import torch
from polystep import Ackley
from polystep.solver import PolyStep

solver = PolyStep.create(Ackley(dim=10), epsilon=0.5, max_iterations=50)
state = solver.run(torch.randn(100, 10))
print(f"Best cost: {min(state.costs):.4f}")
```

### Neural network training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from polystep import PolyStepOptimizer, train, TrainConfig
from polystep.epsilon import CosineEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

# Replace this with your real DataLoader.
train_loader = DataLoader(
    TensorDataset(torch.randn(1024, 784), torch.randint(0, 10, (1024,))),
    batch_size=64,
)

# HybridSubspace compresses the parameter space per layer. Cost per step scales with
# subspace_dim, and rank alone does not bound it: rank=4 on this 101k-parameter model
# gives subspace_dim=4338 and 600 ms/step on one CPU core at batch 64. Cap it
# directly instead; max_subspace_dim=256 gives 26 ms/step.
layout = ParamLayout.from_module(model)
subspace = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=256)

# Cosine schedules: broad exploration -> fine exploitation.
optimizer = PolyStepOptimizer(
    model, subspace=subspace, solver="softmax",
    epsilon=CosineEpsilon(init=10.0, target=0.5, decay=0.02),
    step_radius=CosineEpsilon(init=5.0, target=1.0, decay=0.008),
    probe_radius=CosineEpsilon(init=10.0, target=2.0, decay=0.016),
)

train(model, train_loader, nn.CrossEntropyLoss(), optimizer, TrainConfig(epochs=5))
```

Gradient-free training is forward-pass bound: expect this to be much slower per step
than backprop, and prefer a GPU build of PyTorch for anything beyond a smoke test.

Two things dominate wall-clock in practice:

- **Pin the CPU thread count below `nproc`.** PolyStep's per-step ops are small, so
  torch's default core-count intra-op pool oversubscribes and OpenMP spin-wait takes
  over: on a 24-core box `examples/03` runs 2.1 s at `torch.set_num_threads(1)` against
  427 s at the default. One thread suits most models; raise it only if your own
  objective issues wide ops. See
  [`docs/performance.md`](https://github.com/anindex/polystep/blob/main/docs/performance.md).
- **Register the evaluator if you write your own loop.** `train()` does it for you.
  A bare `optimizer.step(closure)` leaves `closure()` as the only route to the objective,
  so the fused in-place, factored and sparse-delta evaluators never run. Call
  `optimizer.register_evaluator(evaluator, inputs, targets)` before each step.

### Drop-in gradient-free optimizer (ask/tell)

PolyStep also exposes the `ask`/`tell` interface used by evolution-strategy libraries, so
it drops into ES benchmark harnesses (evosax, NeuroEvoBench) and any black-box loop:

```python
import torch
from polystep import PolyStepES

es = PolyStepES(dim=20, epsilon=0.1, step_radius=0.3, x0=torch.full((20,), 2.0))
for _ in range(300):
    candidates = es.ask()           # (popsize, dim) points to evaluate
    es.tell(objective(candidates))  # lower fitness is better
print(es.best_fitness, es.mean)
```

[`experiments/bench_ask_tell.py`](https://github.com/anindex/polystep/blob/main/experiments/bench_ask_tell.py)
compares it head-to-head with a Gaussian ES on the standard synthetic suite under a
matched evaluation budget.

See [`examples/`](https://github.com/anindex/polystep/tree/main/examples/) for eleven
runnable demos.

<table>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/anindex/polystep/main/docs/figures/polystep_snn_progress.gif" width="330"><br><sub>Spiking net (hard LIF thresholds) trained with forward passes only</sub></td>
    <td align="center"><img src="https://raw.githubusercontent.com/anindex/polystep/main/docs/figures/rl_cartpole_policy.gif" width="330"><br><sub>CartPole policy search: no value function, no gradients</sub></td>
  </tr>
</table>

## When to use PolyStep

PolyStep is designed for models where gradients are **unavailable or unreliable**:

- **Spiking neural networks**: hard LIF thresholds, discrete spike events
- **Quantized layers**: int8 weights, binary/ternary networks
- **Blackbox modules**: external simulators, API-based models, hardware-in-the-loop
- **Hard routing**: argmax gating, hard mixture-of-experts
- **Combinatorial optimization**: MAX-SAT, discrete assignment problems

If your model is fully differentiable, Adam/SGD will be faster and more accurate.

## How PolyStep relates to other gradient-free methods

Zeroth-order optimization splits into two camps. Memory-efficient fine-tuning (MeZO and
successors) skips the backprop tape but still assumes a useful local gradient. Evolution
strategies and SPSA (CMA-ES, OpenAI-ES) estimate one from small perturbations.

PolyStep is a randomized direct search: it probes a finite-radius polytope around the
current parameters and moves toward the lowest-cost vertices through an optimal-transport
barycenter. Where the loss is piecewise-constant, the local gradient is zero almost
everywhere, so finite differences carry no signal and ES/SPSA stall while a finite radius
steps across the flat regions. Example 09 shows the separation on a hard decision tree.

Classical direct search (generalized pattern search, MADS) motivates that finite radius,
but PolyStep does not inherit its convergence guarantees: those rest on evaluating the
incumbent, accepting only improving steps, and refining the mesh on a failed poll.
PolyStep evaluates no incumbent (probes exclude scale 0), moves to the barycenter every
step, and drives its radius from the epsilon schedule. Its guarantees come from the
Sinkhorn Step analysis instead.

## Benchmarks

5-seed mean ± std (seeds: 42, 123, 456, 789, 1337). Hardware: NVIDIA RTX 5090.

### Non-differentiable tasks

Adam is gradient-based (backprop, on a smoothed surrogate where the task is
non-differentiable). It is a reference upper bound, not a gradient-free peer of PolyStep,
CMA-ES, OpenAI-ES and SPSA, and is shown only to bound the gap to a gradient method. Bold
marks the best method on each row; a dash means the run is not in this release.

| Task | PolyStep | Adam (surrogate) | CMA-ES | OpenAI-ES | SPSA | Non-diff op |
|------|----------|------------------|--------|-----------|------|-------------|
| SNN/LIF (MNIST) | **93.4 ± 0.3** | 80.5 ± 13.1 | 16.2 ± 8.9 | 33.1 ± 5.5 | 29.4 ± 5.9 | threshold() |
| Int8 quantized | 97.1 ± 0.1 | **98.1 ± 0.0** | 80.7 ± 1.7 | 78.1 ± 0.7 | 91.2 ± 0.1 | round() |
| Argmax attention | 86.8 ± 0.4 | **89.1 ± 0.2** | 72.6 ± 0.6 | 75.7 ± 0.3 | 77.7 ± 0.2 | argmax() |
| Staircase activation | 93.2 ± 0.3 | **97.6 ± 0.1** | 72.8 ± 3.1 | 85.5 ± 0.2 | 49.3 ± 4.7 | floor() |
| Hard MoE routing | **90.7 ± 0.2** | - | 62.8 ± 2.1 | 63.5 ± 6.4 | 69.3 ± 2.2 | argmax() |
| MAX-SAT 100K vars | **98.0 ± 0.01** | - | 90.1 ± 0.04 | 88.9 ± 0.01 | - | round() |
| MAX-SAT 1M vars | **92.6 ± 0.02** | - | - | 87.8 ± 0.00 | - | round() |

### Differentiable sanity checks

| Task | PolyStep | Adam | Architecture |
|------|---------|------|--------------|
| MNIST | 96.0% ± 0.1 | **97.9% ± 0.0** | 2-layer MLP (101K) |
| ETTh1 timeseries | **MSE 0.121 ± 0.004** | MSE 0.187 | LSTM (23K) |

### SNN memory scaling (forward-only vs. BPTT)

| Timesteps | PolyStep | BPTT (surrogate) | Savings |
|-----------|---------|------------------|---------|
| T=25 | 31.8 MB | 132 MB | 4.2x |
| T=400 | 51.6 MB | 1,538 MB | **29.8x** |

PolyStep leads every gradient-free row here, at 60x (SNN) to 13,000x (MAX-SAT 1M) the
evaluation budget of the ES and SPSA baselines. Against the gradient surrogate it wins
where the non-differentiability is hard (SNN LIF, hard MoE routing) and loses where an
accurate smooth surrogate exists (int8, argmax, staircase), so the niche is hard
non-differentiability, not non-differentiability in general. On MAX-SAT the domain
solvers win outright: probSAT reaches about 99.6% at 100K variables and 98.9% at 1M.

## Features

- **OT solvers**: entropic Sinkhorn (full-space default), softmax (subspace default),
  KL-softmax interpolation, and greedy selection.
- **Subspace compression**: `HybridSubspace` (recommended), `FactoredSubspace`,
  `AdaptiveSubspace`, and sparse projection for very large models.
- **Candidate evaluation without materializing weights**: for `nn.Sequential` MLPs the
  step scores every candidate through a shared base forward plus a low-rank correction,
  4.4x on a 203K MLP. Automatic. See
  [`docs/performance.md`](https://github.com/anindex/polystep/blob/main/docs/performance.md).
- **Sub-linear memory**: forward-only evaluation, no BPTT activation tape (~30x savings
  at long SNN horizons).
- **Block-wise OT** for per-layer decomposition.
- **Vmap-safe layers**: drop-in attention and LSTM that work under `torch.vmap`.
- **`torch.compile`** opt-in on the OT kernels (`compile=True`) and on the candidate
  forward (`compile_evaluator`; `compile_forward` auto-enables on the in-place path and
  takes an explicit `False` to opt out).
- **Ask/tell API**: `PolyStepES` drops into evosax / NeuroEvoBench-style ES harnesses and
  black-box loops.

## Limitations

- **Compute cost.** Roughly tens of millions of forward passes (on the SNN benchmark,
  around 30M) vs. tens of thousands of Adam gradient steps for the same MNIST accuracy.
  This is inherent to zeroth-order methods.
- **High-dimensional NLP.** All-parameter fine-tuning of GPT-2 124M through a 128-dim
  projection collapses to random predictions; the projection ratio is far below the
  Johnson-Lindenstrauss floor.
- **Adam baseline.** Where a smooth surrogate exists, Adam wins (see the benchmark
  tables). The stronger surrogate-gradient / BPTT baseline for SNNs (paper §5.3) is not
  bundled with this release; see the arXiv preprint.

Full discussion in
[`LIMITATIONS.md`](https://github.com/anindex/polystep/blob/main/LIMITATIONS.md).

## Documentation

| Resource | Description |
|----------|-------------|
| [`examples/`](https://github.com/anindex/polystep/tree/main/examples/) | 11 runnable demos: quickstart, SNN, RL, MAX-SAT, MNIST, Loihi 2, STE-free binary net, direct loss minimization, hard oblique decision tree, CNN, transformer |
| [`experiments/`](https://github.com/anindex/polystep/tree/main/experiments/) | Paper reproduction: runners, results, baselines |
| [`docs/api_overview.md`](https://github.com/anindex/polystep/blob/main/docs/api_overview.md) | API reference |
| [`docs/performance.md`](https://github.com/anindex/polystep/blob/main/docs/performance.md) | Per-step cost, the fast evaluation paths, thread count, compile flags |
| [`docs/reproducibility.md`](https://github.com/anindex/polystep/blob/main/docs/reproducibility.md) | Reproducing the paper results |
| [`LIMITATIONS.md`](https://github.com/anindex/polystep/blob/main/LIMITATIONS.md) | Known limitations |
| [`CONTRIBUTING.md`](https://github.com/anindex/polystep/blob/main/CONTRIBUTING.md) | Contribution guidelines |
| [`CHANGELOG.md`](https://github.com/anindex/polystep/blob/main/CHANGELOG.md) | Release history |

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{le2026training,
  title={Training Non-Differentiable Networks via Optimal Transport},
  author={Le, An T},
  journal={arXiv preprint arXiv:2605.01928},
  year={2026}
}
```

## Acknowledgments

Thanks to [**Viet T. Nguyen**](https://vietngth.github.io/) for the interactive
[PolyStep visualization](https://vietngth.github.io/polystep-visualization/), which
animates every step of the method.

## License

Apache License 2.0. See [LICENSE](https://github.com/anindex/polystep/blob/main/LICENSE).
