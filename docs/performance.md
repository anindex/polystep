# Per-step cost & the compile flags

PolyStep spends `P·V·K` forward passes per optimizer step (`P` particles ×
`V` polytope vertices × `K` probe radii). This page is the honest, **measured**
guide to what that costs and which knobs move it. Reproduce everything with
`experiments/scripts/bench_forward_backends.py` (RTX 5090, torch 2.13, N=128
candidates, batch 128, 3 seeds; medians are illustrative — pin GPU clocks for
headline-grade numbers).

## Currency rule

Measure **wall-clock** (and, if you want a hardware-independent number,
dense-MACs / synaptic-ops). **Never** compare methods by forward count: on a
small net the cost is dominated by kernel-launch / dispatch overhead, not FLOPs,
so "fewer forwards" and "faster" are not the same thing.

## The candidate forwards are already batched

The single most important fact: `NNCostEvaluator` evaluates all `N = P·V·K`
candidates in **one vectorized `vmap` sweep** (`torch.func.functional_call`
over the candidate axis), not one forward at a time. `vmap` turns `N` launches
of a size-`X` kernel into **one** launch of a size-`N·X` kernel, so the
per-kernel launch overhead is amortized across all candidates. Measured on the
recurrent SNN, the `vmap` sweep is **~30–40× faster** than the equivalent
sequential per-candidate loop (3.4 ms vs 122 ms for 128 candidates). In other
words, the launch-bound problem is *already solved* on PolyStep's default path.

> Aside: a single un-batched forward of a small recurrent net is heavily
> launch-bound, and `torch.compile(mode="reduce-overhead")` (CUDA graphs) can
> speed *that* up ~10×. That figure does **not** apply to PolyStep — PolyStep
> never evaluates a single forward; it batches candidates, which captures the
> same win structurally.

## Two compile flags (both opt-in, off by default)

### `compile_evaluator=True`  (→ `NNCostEvaluator(compile_vmap=True)`)

Wraps the vmapped forward in `torch.compile(mode="default")` — **Inductor
kernel fusion only, no CUDA graphs**. Measured speedup on top of `vmap`,
across four non-differentiable architectures:

| architecture              | fusion speedup | note                                   |
|---------------------------|:--------------:|----------------------------------------|
| BinaryCIFAR10Net (CNN)    |     1.86×      | largest — FLOP/bandwidth-heavy         |
| DiscreteAttentionNet      |     1.47×      | MLP + argmax routing                   |
| HardMoENet                |     1.46×      | MLP + hard top-1 gating                |
| SpikingMNISTNet (T=15)    |     1.20×      | smallest — vmap already amortized it   |

The gradient is the honest story: fusion helps **most** where there is real
compute to fuse (CNN) and **least** on the recurrent SNN, because `vmap` has
*already* amortized the SNN's many tiny launches. Expect a modest, universal
~1.2–1.9× — not a 10× — and it is the best backend of those tested on all four.
`reduce-overhead` (CUDA graphs) on the *vmapped* path was measured **not** to
beat `default`: once the sweep is vmap-amortized, graph capture buys nothing (and
Inductor tends to skip it). That is why `compile_vmap` deliberately ships
`mode="default"`.

Off by default because compiling arbitrary user modules in the inner loop has
real costs: fusion reorders floating-point reductions, which slightly changes
loss values and can therefore shift the OT trajectory (reproducibility); plus
first-call compile latency, recompiles when `N`/batch/dtype change, and silent
graph breaks on data-dependent control flow. Turn it on for CUDA + shape-stable
runs.

### `compile_forward=True`  (→ `NNCostEvaluator(compile_forward=True)`)

For the **sequential in-place path** (auto-selected for >500K-param GPU models,
or forced with `use_inplace=True`, where a full `vmap` sweep would OOM), this
compiles the forward+loss closure with `torch.compile(mode="reduce-overhead")` —
**CUDA graphs** — and replays it per candidate while the swap loop mutates
`param.data` in place (`.data.copy_` preserves storage addresses, so graph
replay reads the fresh weights; `torch._foreach_copy_` fuses the swap). CUDA
graphs are safe here, unlike on the vmap path, because there is no chunk-concat.

Measured **within the in-place path** (inplace + graphs vs inplace + eager):

| architecture           | in-place graph speedup |
|------------------------|:----------------------:|
| SpikingMNISTNet (T=15) |         ~6.1×          |
| HardMoENet             |         ~2.0×          |
| DiscreteAttentionNet   |         ~1.4×          |
| BinaryCIFAR10Net (CNN) |         ~1.1×          |

The win is proportional to how launch-bound the forward is: large for the
recurrent SNN, negligible for the denser CNN. **Caveat — do not extrapolate the
6×:** it was measured on a *small* SNN. The in-place path is only used for nets
large enough to OOM `vmap`, and such nets have big, dense kernels where launch
overhead is already a small fraction — so the realistic graph win there is closer
to the CNN's ~1.1× than the SNN's 6×. A genuinely-large-net measurement is future
work.

## Which path runs, and what to flip

| your model                              | eval path            | recommendation                            |
|-----------------------------------------|----------------------|-------------------------------------------|
| pure `nn.Sequential` MLP (vanilla CE)   | `bmm` fast path      | already fast; compile flags don't apply   |
| custom-forward net, fits in memory      | `vmap`               | `compile_evaluator=True` → ~1.2–1.9×      |
| large net / `vmap` OOMs                 | in-place swap loop   | `compile_forward=True` (launch-bound nets)|
| activation-heavy (e.g. CNN, big batch)  | `vmap` **can lose**  | try `use_inplace=True` + `compile_forward` |

That last row is real: `vmap` over a conv net lowers to a grouped conv that
materializes `N×` activations, so on the CNN the plain in-place loop (19 ms) can
**beat** `eager_vmap` (25 ms). It is the concrete reason the in-place path (and
its compile flag) exist beyond just OOM avoidance.

Both optimizer flags propagate to a registered evaluator via
`register_evaluator`, and `api.train()` reads them, so
`PolyStepOptimizer(compile_evaluator=True)` works on every path (not only the
high-level trainer).

## Why this is the remaining per-step lever

PolyStep's zero-order *algorithm* levers are closed: the number of vertices is
floored at `k+1` (the regular simplex is the minimal positive basis — a
deterministic probe set that guarantees a descent direction must positively span
`R^k`), and dropping/racing vertices provably fails because each step takes an
entropic-OT *barycenter* (a weighted mean over the vertices), not a single
vertex. With the counts fixed, the only remaining per-step lever is the cost of
one forward — i.e. this page: kernel fusion (`compile_evaluator`) and, on the
sequential path, CUDA-graph launch elimination (`compile_forward`).

## Future lever (not implemented)

For the memory-bounded **chunked** regime (very large `N`, where `vmap` runs in
`chunk_size` slices), CUDA graphs are blocked by the chunk-concat. Padding the
candidate batch to a fixed multiple of `chunk_size` (and masking the padded
losses before the OT solve) would give static shapes and let
`reduce-overhead` capture the whole chunk. Recorded here as a documented option;
the in-place `compile_forward` path reaches the launch-bound win with far less
surgery, so this is not built.
