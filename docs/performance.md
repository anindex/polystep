# Per-step cost and the compile flags

PolyStep runs `P*V*K` forward passes per step (`P` particles x `V` polytope
vertices x `K` probe radii). Measure wall-clock, not forward count: on small
nets the cost is launch/dispatch overhead, not FLOPs.

## Particles are batched

`NNCostEvaluator` evaluates all `N = P*V*K` particles in one vectorized `vmap`
sweep (`torch.func.functional_call` over the candidate axis), so per-kernel
launch overhead is amortized across particles. This is the default path.

## Two opt-in compile flags (off by default)

- `compile_evaluator=True`: wraps the vmapped forward in
  `torch.compile(mode="default")` (Inductor fusion, no CUDA graphs). Roughly
  1.2-1.9x on top of vmap, largest on FLOP-heavy nets (CNN), smallest on
  recurrent SNNs that vmap already amortized. Best backend when it fits.
- `compile_forward=True`: for the sequential in-place path (auto-selected for
  >500K-param GPU models, or `use_inplace=True`, where a full vmap sweep would
  OOM), CUDA-graphs the forward+loss closure and replays it per candidate. It
  keeps the memory-forced path competitive with vmap; it does not beat vmap.
  The in-place path is chosen for memory, not speed.

Both are opt-in because compiling in the inner loop reorders floating-point
reductions (loss values change slightly, so the OT trajectory can shift), adds
first-call compile latency, and recompiles on shape/dtype changes. On
hard-threshold nets (SNN) the compiled loss can differ at the discretization
scale (~1.6e-3), shifting candidate selection. Enable for CUDA + shape-stable runs.

## Which path runs

| model                                 | path            | flag                     |
|---------------------------------------|-----------------|--------------------------|
| pure `nn.Sequential` MLP (vanilla CE) | `bmm` fast path | none (already fast)      |
| custom forward, fits in memory        | `vmap`          | `compile_evaluator=True` |
| large net / vmap OOMs                 | in-place swap   | `compile_forward=True`   |

Both flags propagate to a registered evaluator via `register_evaluator` and are
read by `api.train()`, so `PolyStepOptimizer(compile_evaluator=True)` works on
every path. Reproduce with `experiments/scripts/bench_forward_backends.py` and
`experiments/scripts/bench_large_net_inplace.py`.
