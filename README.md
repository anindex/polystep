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

> Want to play around with parameters? [**Viet T. Nguyen**](https://vietngth.github.io/) built a gorgeous interactive walkthrough that animates every step of the method -> **[explore the PolyStep visualization](https://vietngth.github.io/polystep-visualization/)**.

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
pip install -e ".[experiments]"       # + numpy, pandas, python-sat, torchvision, cma, snntorch, datasets, transformers, gymnasium
pip install -e ".[rl]"                # + gymnasium[box2d], stable-baselines3, swig (Box2D envs)
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

Expect this to cost far more per step than backprop. Use a GPU build.

Two things dominate wall-clock:

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
but PolyStep does not inherit its guarantees: it evaluates no incumbent, moves to the
barycenter every step, and drives its radius from the epsilon schedule.

## Gradient-free baselines

`polystep.baselines` runs six gradient-free methods against the same protocol PolyStep
uses, so a comparison is not confounded by the search space, the probe radius, the
minibatch stream or the evaluation budget.

| Method | Cost per iteration | Source |
|--------|-------------------|--------|
| `openai_es` | `popsize` | Salimans et al. 2017 (arXiv:1703.03864) |
| `spsa` | 2 | Spall 1992 |
| `mezo` | 2 | Malladi et al. 2023 (arXiv:2305.17333) |
| `random_search` | 1 | control: random direction, keep it if the loss drops |
| `eggroll` | `popsize` | Sarkar et al. (arXiv:2511.16652) |
| `cma_es` | `popsize` | pycma (`pip install cma`) |

```python
from polystep.baselines import Objective, openai_es, random_search

# fn: (N, dim) -> (N,) losses, lower is better. One call per generation, so a
# stochastic fn draws one minibatch per call and every candidate sees the same data.
obj = Objective(fn, dim=64, budget=10_000)
result = openai_es(obj, sigma=0.05, lr=0.1, popsize=32)
result.best_loss, result.evals   # evals is never above budget
```

To search a subspace instead, only the objective changes:

```python
obj = Objective.from_subspace(hybrid, base_sd, loss_batch, budget=10_000)
result = random_search(obj, sigma=0.1)   # isolates the subspace from the update rule
```

**One evaluation means one candidate scored.** `Objective` counts rows, so a method that
vmaps 32 candidates behind one call spends 32, exactly like one that makes 32 calls. It
refuses a batch that would exceed the budget, so no method can overspend.

Note on EGGROLL: its rank-`r` perturbations `A B^T` are defined on weight *matrices*, and
`Objective.shapes` says where those are. `Objective.from_layout` preserves the per-tensor
shapes; `Objective.from_subspace` cannot, because `HybridSubspace` coordinates carry no
matrix structure, so EGGROLL there degenerates to dense Gaussian ES on the coordinates.
`FactoredSubspace` is the subspace that already implements the `A B^T` parameterization.

### Fairness mode

The paper runners take `--fair`, which hands every gradient-free method in a table the
same subspace (same class, rank and seed), the same candidate budget derived from what
PolyStep spends over its configured epochs, the same minibatch stream and the same probe
radius. Each result JSON records `subspace_class`, `subspace_rank`, `eval_budget` and
`evals_used`, plus a per-generation trajectory against *cumulative candidate
evaluations* for the accuracy-vs-evaluations figure. EGGROLL is the one recorded
exception: it gets `FactoredSubspace`, for the reason above.

```bash
python experiments/runners/run_mnist.py --fair --methods polystep openai_es spsa mezo random_search eggroll cma_es
# tuning cost, in the same units, for both sides of the table
python experiments/runners/variant_sweep.py --envs mnist_mlp --stage screen   # writes tuning_cost.json
python experiments/runners/variant_sweep.py --envs mnist_mlp --baseline spsa  # appends to it
```

`--theory-mode` runs the unaccelerated reference configuration instead of the tuned one:
jitter `0.05` on the probe and step radii with the smooth mollifier density, independently sampled
rotations, flat epsilon, step radius `r_0 (t+1)^-(1/2+0.1)` (`polystep.PowerDecay`),
orthoplex, `HybridSubspace`, and no momentum, amortization or Anderson acceleration.
Jitter and the amortization heuristics are mutually exclusive and `PolyStepOptimizer`
enforces that itself: `probe_radius_jitter > 0` turns off `adaptive_probes` and
`amortize_steps`, because jitter makes the per-step cost a noisy estimate that those
heuristics read as progress.

## Benchmarks

5-seed mean ± std, every method tuned on validation at an equal budget and matched on
optimizer steps. Protocol and per-experiment notes:
[`experiments/EXPERIMENT_INDEX.md`](https://github.com/anindex/polystep/blob/main/experiments/EXPERIMENT_INDEX.md).
Tables are generated from the result JSONs, not hand-maintained.

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

### SNN memory scaling (forward-only vs. BPTT)

| Timesteps | PolyStep | BPTT (surrogate) | Savings |
|-----------|---------|------------------|---------|
| T=25 | 31.8 MB | 132 MB | 4.2x |
| T=400 | 51.6 MB | 1,538 MB | **29.8x** |

PolyStep leads every gradient-free baseline on all five non-differentiable tasks, and
beats the gradient surrogate only where the non-differentiability is hard. It does not
beat Adam on differentiable problems, or domain solvers on MAX-SAT.

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
- **Adam baseline.** The stronger surrogate-gradient / BPTT baseline for SNNs is not
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

Citation:

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

Apache License 2.0. See [LICENSE](https://github.com/anindex/polystep/blob/main/LICENSE).
