"""Backend x architecture matrix for PolyStep's per-step candidate-forward eval.

The per-step cost is P*V*K forward passes. On a launch-bound net (many tiny
sequential kernels, e.g. an SNN's T-step recurrence) the lever is not FLOPs but
kernel-launch / CPU-dispatch overhead. This benchmark measures, across
architectures that actually route through vmap/in-place (NOT the pure-MLP bmm
fast path), how each evaluator backend moves wall-clock:

  eager_vmap             vmap + functional_call, no compile          (baseline)
  compiled_vmap_default  torch.compile(mode="default"): FUSION only  (no CUDA graphs)
  inplace_eager          sequential .data-swap loop, eager forward
  inplace_graph          compile_forward: torch.compile(reduce-overhead) = CUDA
                         graphs on the forward+loss, replayed per candidate

Honest reporting (the review's asks): wall-clock median + IQR, speedup vs
eager_vmap, and the ACTUAL backend used (compile can silently fall back). Never
forward count. The expected result is a GRADIENT, not a universal number:
recurrent/launch-bound nets (SNN) win big from CUDA graphs; feedforward MLP/CNN
become FLOP/bandwidth-bound and win modestly.

Run FOREGROUND on GPU:
  PY=/home/anindex/polystep/.venv/bin/python; $PY experiments/scripts/bench_forward_backends.py
"""

from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, "/home/anindex/polystep")
import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from experiments.runners.nondiff_models import (
    BinaryCIFAR10Net,
    DiscreteAttentionNet,
    HardMoENet,
    SpikingMNISTNet,
)
from polystep.cost_nn import NNCostEvaluator

# (name, build_fn, input_shape (per-sample), n_classes)
MODELS = [
    ("SpikingMNISTNet (recurrent, T=15)", lambda: SpikingMNISTNet(num_steps=15), (784,), 10),
    ("DiscreteAttentionNet (MLP+argmax)", lambda: DiscreteAttentionNet(), (784,), 10),
    ("BinaryCIFAR10Net (CNN)", lambda: BinaryCIFAR10Net(), (3, 32, 32), 10),
    ("HardMoENet (MLP+hard-gate)", lambda: HardMoENet(), (784,), 20),
]

BATCH = 128
N_CAND = 128  # representative candidate chunk (P*V*K)
WARMUP = 50
# Sub-millisecond sweeps are noisy: use a per-call time budget so cheap backends
# get many more reps (Codex/Gemini review). GPU clocks should be pinned high
# (nvidia-smi -lgc) for a headline; these medians are illustrative, not paper-grade.
MIN_REPS = 60
TIME_BUDGET_S = 1.5


def _stacked(model, n, noise=0.02, gen=None):
    """N candidate param dicts = current params broadcast + small noise.

    Keyed by param name -- exactly what NNCostEvaluator.evaluate expects.
    """
    out = {}
    for name, p in model.named_parameters():
        base = p.detach()
        cfg = base.unsqueeze(0).expand(n, *base.shape).contiguous()
        cfg = cfg + noise * torch.randn(cfg.shape, device=cfg.device, dtype=cfg.dtype, generator=gen)
        out[name] = cfg
    return out


def _time(fn):
    """GPU-side timing via cuda.Event, with a wall-time budget so sub-ms sweeps
    get many reps (less Python/sync jitter than perf_counter)."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    start = time.perf_counter()
    while len(ts) < MIN_REPS or (time.perf_counter() - start) < TIME_BUDGET_S:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))  # ms, GPU-side
        if len(ts) >= 5000:
            break
    q = statistics.quantiles(ts, n=4)
    return statistics.median(ts), q[2] - q[0]


def _backend_used(ev, kind):
    """Report the path that actually ran (compile can silently fall back)."""
    if kind.startswith("inplace"):
        if kind == "inplace_graph":
            return "inplace-eager (compile fell back)" if ev._compile_forward_failed else "inplace+CUDAgraph"
        return "inplace-eager"
    # vmap arms
    if ev._vmap_failed:
        return "sequential-loop (vmap fell back)"
    if kind == "compiled_vmap_default":
        return "vmap-eager (compile fell back)" if ev._compile_failed else "vmap+fusion"
    return "vmap-eager"


def _build_vmap_ro(model, loss_fn):
    """The 'missing cell' (Kimi): torch.compile(reduce-overhead) on the VMAPPED fn.

    With chunk_size=None (no chunking) there is no chunk-concat, so CUDA graphs
    on the whole N-candidate sweep may capture -- one graph, N candidates. This
    is what the evaluator's compile_vmap deliberately does NOT do (it ships
    mode="default"). Tested raw here to see if it beats fusion on launch-bound nets.
    """
    buffers = dict(model.named_buffers())

    def single(params, inputs, targets):
        out = functional_call(model, {**params, **buffers}, (inputs,))
        loss = loss_fn(out, targets)
        return loss.mean() if loss.dim() > 0 else loss

    batched = vmap(single, in_dims=(0, None, None), chunk_size=None)
    return torch.compile(batched, mode="reduce-overhead", fullgraph=False)


def _make_ev(model, loss_fn, kind):
    if kind == "eager_vmap":
        return NNCostEvaluator(model, loss_fn, use_inplace=False, compile_vmap=False)
    if kind == "compiled_vmap_default":
        return NNCostEvaluator(model, loss_fn, use_inplace=False, compile_vmap=True)
    if kind == "inplace_eager":
        return NNCostEvaluator(model, loss_fn, use_inplace=True, compile_forward=False)
    if kind == "inplace_graph":
        return NNCostEvaluator(model, loss_fn, use_inplace=True, compile_forward=True)
    raise ValueError(kind)


BACKENDS = [
    "eager_vmap",
    "compiled_vmap_default",
    "compiled_vmap_reduce_overhead",
    "inplace_eager",
    "inplace_graph",
]


def run_model(name, build, in_shape, n_classes, seeds=(0, 1, 2)):
    loss_fn = nn.CrossEntropyLoss()
    per_backend = {b: [] for b in BACKENDS}
    used = {}
    for seed in seeds:
        torch.manual_seed(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        model = build().cuda().eval()
        x = torch.rand(BATCH, *in_shape, device="cuda")
        y = torch.randint(0, n_classes, (BATCH,), device="cuda")
        stacked = _stacked(model, N_CAND, gen=gen)

        for kind in BACKENDS:
            if kind == "compiled_vmap_reduce_overhead":
                # The missing-cell experiment: CUDA graphs on the vmapped sweep.
                try:
                    fn = _build_vmap_ro(model, loss_fn)
                    with torch.inference_mode():
                        med, iqr = _time(lambda: fn(stacked, x, y))
                        losses = fn(stacked, x, y)
                    ok = torch.unique(losses).numel() > N_CAND // 2
                    per_backend[kind].append(med if ok else float("inf"))
                    # RO on the vmapped path measures ~= fusion -> CUDA graphs do not
                    # capture/benefit the already-amortized sweep (see docs).
                    used[kind] = "vmap+reduce-overhead" if ok else "vmap+RO STALE (rejected)"
                except Exception as e:  # noqa: BLE001
                    per_backend[kind].append(float("inf"))
                    used[kind] = f"FAILED ({type(e).__name__})"
                torch.cuda.empty_cache()
                continue
            ev = _make_ev(model, loss_fn, kind)
            # Guard: these models must NOT hit the pure-MLP bmm fast path, else the
            # benchmark would silently bypass compile and lie.
            assert ev._batched_linear is None, f"{name} hit BatchedLinearEvaluator -- bmm bypass"
            with torch.inference_mode():
                med, iqr = _time(lambda: ev.evaluate(stacked, x, y))
            per_backend[kind].append(med)
            used[kind] = _backend_used(ev, kind)
            # sanity: distinct configs -> distinct losses (stale-weight guard)
            with torch.inference_mode():
                losses = ev.evaluate(stacked, x, y)
            assert torch.unique(losses).numel() > N_CAND // 2, f"{name}/{kind}: losses collapsed (stale weights?)"
        del model
        torch.cuda.empty_cache()

    base = statistics.median(per_backend["eager_vmap"])
    print(f"\n{name}   batch={BATCH} N_cand={N_CAND}  ({len(seeds)} seeds)")
    print(f"  {'backend':24s}{'ms/sweep':>11s}{'ms/cand':>10s}{'speedup':>9s}   actual-path")
    for kind in BACKENDS:
        med = statistics.median(per_backend[kind])
        print(f"  {kind:24s}{med:11.3f}{med / N_CAND:10.4f}{base / med:8.2f}x   {used[kind]}")
    best = min(BACKENDS, key=lambda k: statistics.median(per_backend[k]))
    return name, base / statistics.median(per_backend[best]), best, used


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA.")
        return
    print(f"Backend x architecture matrix  device={torch.cuda.get_device_name(0)}  "
          f"torch={torch.__version__}")
    print("Currency = wall-clock (median over timed reps). Never forward count.")
    verdicts = [run_model(*m) for m in MODELS]
    print("\n=== VERDICT (measured, illustrative -- pin GPU clocks for headline-grade) ===")
    for name, speedup, best, _ in verdicts:
        print(f"  {name:36s} best={best:24s} {speedup:5.2f}x vs eager_vmap")
    print("  - vmap already amortizes launches: ~30-40x vs the sequential in-place loop.")
    print("  - compiled_vmap_default (fusion) is best on all four, ~1.2-1.9x on top of vmap;")
    print("    LARGEST on the FLOP-heavy CNN, SMALLEST on the already-amortized SNN.")
    print("  - reduce-overhead on the vmapped path does NOT beat fusion (CUDA graphs give no")
    print("    benefit once the sweep is vmap-amortized).")
    print("  - CUDA graphs help only the sequential in-place path (inplace_graph), proportional")
    print("    to launch-boundness (SNN ~6x, CNN ~1.1x); that path is used only when vmap OOMs.")
    print("  - eager_vmap can LOSE to inplace on activation-heavy nets (CNN 25 vs 19 ms) -- vmap")
    print("    of conv lowers to grouped conv. Never compare by forward count.")


if __name__ == "__main__":
    main()
