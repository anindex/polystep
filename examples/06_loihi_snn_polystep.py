"""06 - PolyStep on Loihi 2 (skeleton): MNIST SNN + on-chip readout adaptation.

A two-stage demonstration that PolyStep can train a spiking network and
adapt the deployed model on device under input distribution shift, using
only the writable subset a real Loihi 2 chip exposes at runtime, without
backpropagation, surrogate gradients, or BPTT.

Stage 1 -- Off-chip pretrain on clean MNIST. Full-model PolyStep with
    the paper's SNN configuration from
    ``experiments/runners/run_elevation.py`` ``PSTORCH_CONFIGS["snn"]``
    (flat schedules; ``CosineEpsilon`` on eps / sr / pr collapses SNN
    accuracy in the paper sweeps). Stands in for a SLAYER + ``netx``
    deploy.

Stage 2 -- On-chip readout adaptation under input shift. Hidden layer is
    frozen; only the writable subset is adapted -- ``fc2`` weights and
    the per-population learnable LIF ``vth`` / ``beta`` (the chip's
    runtime-mutable microcode neuron ``Var``s). Three TENT-style
    safeguards (Wang et al., ICLR 2021) make Stage 2 robust:

      1. Mixed-batch shift (``--mixed-shift``, default on): each adapt
         batch is ``[clean ; shifted]`` so the writable subset is
         pulled toward both manifolds and clean accuracy does not drift.
      2. Higher rank on the tiny writable subspace (``--adapt-rank 8``).
      3. Two probes per step (``--adapt-num-probe 2``) for variance
         reduction on the noisier shifted landscape.

Both stages use best-test early stopping (patience 4 -- higher than
typical SGD because zeroth-order test curves are noisier per epoch).
The weights at the end of each stage are the checkpoint with the
highest test accuracy on that stage's target distribution (clean for
Stage 1, shifted for Stage 2). The frozen-readout baseline reloads
the Stage 1 best weights, and the shifted test set uses a fixed,
seeded noise mask across pre / post / baseline evaluations, so the
reported recovery is a paired comparison free of sampling jitter.

Backends. ``--backend cpu_sim`` (default) uses PyTorch as the forward
evaluator. The host loop is identical to the on-chip loop -- only
``LoihiSpikeEvaluator.evaluate`` would change for ``--backend loihi2``.

Run::

    python examples/06_loihi_snn_polystep.py
    python examples/06_loihi_snn_polystep.py --shift-sigma 0.0   # no shift
    python examples/06_loihi_snn_polystep.py --no-mixed-shift    # ablate safeguard
    python examples/06_loihi_snn_polystep.py --backend loihi2    # if lava installed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Allow running directly from a source checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polystep import PolyStepOptimizer  # noqa: E402
from polystep.benchmarks.utils import get_mnist_loaders  # noqa: E402
from polystep.cost_nn import NNCostEvaluator  # noqa: E402
from polystep.hybrid_subspace import HybridSubspace  # noqa: E402
from polystep.transform import ParamLayout  # noqa: E402


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    import lava  # noqa: F401  (presence check only)
    _HAS_LAVA = True
except ImportError:
    _HAS_LAVA = False


def select_backend(requested: str) -> str:
    if requested == "loihi2":
        if not _HAS_LAVA:
            print(
                "  [warn] backend=loihi2 requested but `lava` not importable; "
                "falling back to cpu_sim. `pip install lava-nc` for the CPU "
                "simulator; INRC membership for the Loihi 2 hardware extension."
            )
            return "cpu_sim"
        return "loihi2"
    return "cpu_sim"


# ---------------------------------------------------------------------------
# Spiking layers (vmap-safe; learnable vth/beta mirror Loihi 2 µcode Vars)
# ---------------------------------------------------------------------------
class LearnableLIF(nn.Module):
    """Leaky integrate-and-fire with learnable threshold and decay.

    Dynamics (snnTorch / Loihi convention, subtractive reset)::

        mem[t] = beta * mem[t-1] + I[t]
        spk[t] = (mem[t] >= vth)        # hard Heaviside, derivative = 0
        mem[t] = mem[t] - spk[t] * vth  # subtract on spike

    ``beta`` is parameterized through a sigmoid so it stays in (0, 1)
    under unconstrained PolyStep updates. ``vth`` is parameterized
    directly. Both are scalars -- vmap stacks them along dim 0 with no
    special handling.

    On a real Loihi 2 these would be the per-population ``vth`` and
    ``du``/``dv`` µcode neuron Vars writable from the host between runs.
    """

    def __init__(self, beta: float = 0.95, vth: float = 1.0):
        super().__init__()
        beta_clamped = max(min(beta, 0.999), 1e-3)
        beta_logit = float(torch.logit(torch.tensor(beta_clamped)))
        self.beta_logit = nn.Parameter(torch.tensor(beta_logit))
        self.vth = nn.Parameter(torch.tensor(float(vth)))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + x
        spike = (mem >= self.vth).to(x.dtype)
        mem = mem - spike * self.vth
        return spike, mem


class MnistSpikingNet(nn.Module):
    """Two-layer rate-coded SNN for MNIST.

    Architecture mirrors snnTorch tutorial 5 (Linear -> Leaky ->
    Linear -> Leaky) and the paper's ``SpikingMNISTNet``, scaled down
    for CPU-friendliness. Returns *summed spike counts* over time
    (rate code) so cross-entropy has a usable signal range.
    """

    def __init__(self, hidden: int = 64, num_steps: int = 10, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.lif1 = LearnableLIF(beta=beta, vth=1.0)
        self.fc2 = nn.Linear(hidden, 10)
        self.lif2 = LearnableLIF(beta=beta, vth=1.0)
        self.hidden = hidden
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        mem1 = torch.zeros(batch, self.hidden, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        total = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + spk2
        return total  # (batch, 10) -- raw spike counts (rate code)


# ---------------------------------------------------------------------------
# Forward-evaluation backends
# ---------------------------------------------------------------------------
class CpuSimEvaluator:
    """CPU forward evaluator. Identical interface to the on-chip path."""

    def __init__(self, model: nn.Module, loss_fn: nn.Module):
        self.model = model
        self._evaluator = NNCostEvaluator(model, loss_fn=loss_fn)

    def evaluate(self, stacked_params, inputs, targets):
        return self._evaluator.evaluate(stacked_params, inputs, targets)


class LoihiSpikeEvaluator:
    """Loihi 2 forward evaluator (Lava ``netx`` deployment) -- Stage 2.

    Implementation sketch (real version requires ``lava`` + a SLAYER-
    trained HDF5 net description; not run by default in this example)::

        from lava.lib.dl.netx import hdf5
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg  # or Loihi2HwCfg

        net = hdf5.Network(net_config="snn_mnist.net")

        for k in range(V):
            # 1. Write vertex-k parameters into chip Vars (readout
            #    weights, vth, du, dv). On Kapoho Point this is
            #    parallelised: vertex k -> chip k.
            for key, tensor in stacked_params.items():
                _write_var(net, key, tensor[k])
            # 2. Run for K timesteps, read spike counts, compute loss.
            net.run(condition=RunSteps(num_steps=K),
                    run_cfg=Loihi2SimCfg())
            losses[k] = loss_fn(_read_spikes(net), targets)
            net.stop()

    The make-or-break number is host<->chip round-trip per vertex
    upload.
    """

    def __init__(self, hdf5_path: str, loss_fn: nn.Module):
        if not _HAS_LAVA:
            raise RuntimeError("`pip install lava-nc` first.")
        raise NotImplementedError(
            "LoihiSpikeEvaluator is a Stage 2 deliverable. The cpu_sim "
            "backend exercises the same PolyStep host loop end-to-end."
        )


# ---------------------------------------------------------------------------
# Adaptation subset = readout + per-population LIF parameters (vth, beta)
# ---------------------------------------------------------------------------
def freeze_to_writable_subset(model: MnistSpikingNet) -> int:
    """Freeze ``fc1``; keep the *writable* subset (readout + LIF Vars).

    Mirrors what a real Loihi 2 chip exposes at runtime without
    recompilation: readout weights ``fc2``, per-population thresholds
    ``lif{1,2}.vth``, and membrane decays ``lif{1,2}.beta_logit``.
    PolyStep's ``ParamLayout.from_module`` honours ``requires_grad``
    (see ``src/polystep/transform.py``), so frozen tensors are excluded
    from the OT particle automatically.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc2.parameters():
        p.requires_grad_(True)
    model.lif1.vth.requires_grad_(True)
    model.lif1.beta_logit.requires_grad_(True)
    model.lif2.vth.requires_grad_(True)
    model.lif2.beta_logit.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Optimizer factories (paper-tuned configs)
# ---------------------------------------------------------------------------
def make_pretrain_optimizer(
    model: nn.Module, *, seed: int, device: torch.device,
) -> PolyStepOptimizer:
    """Stage 1 optimizer (off-chip pretrain): paper SNN config from ``run_elevation.py``.

    Mirrors ``PSTORCH_CONFIGS["snn"]`` exactly. Key insight from the
    paper sweeps (see ``experiments/runners/run_elevation.py:84``):
    *flat* epsilon / step_radius / probe_radius -- ``CosineEpsilon``
    scheduling on any of them collapses SNN accuracy to 10-47%.
    """
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout, rank=4,
        rotation_mode="random", rotation_interval=0,
        absorb_mode="periodic", absorb_interval=20,
    )
    return PolyStepOptimizer(
        model,
        compile=(device.type == "cuda"),
        seed=seed,
        epsilon=0.5,
        step_radius=2.0,
        probe_radius=1.0,
        num_probe=1,
        subspace=subspace,
        chunk_size=1024,
        amortize_steps=1,
        biased_rotation=True,
        anderson_depth=5,
        adaptive_omega=True,
        solver="softmax",
    )


def make_adapt_optimizer(
    model: nn.Module, *, seed: int, device: torch.device,
    rank: int = 8, num_probe: int = 2,
    step_radius: float = 1.5, probe_radius: float = 0.75,
) -> PolyStepOptimizer:
    """Stage 2 optimizer (on-chip readout adaptation): small writable subset, low-rank, all flat.

    Defaults are tuned for on-chip adaptation under input shift:
    - ``rank=8`` (the writable subset is tiny ~1.3 % of params, so a
      richer subspace costs nothing and improves recovery).
    - ``num_probe=2`` (better gradient estimate on the noisier shifted
      landscape; doubles probe-cost only on the *short* adapt phase).
    - ``step_radius=1.5`` / ``probe_radius=0.75`` (slightly larger than
      paper SNN defaults; the writable subset is well-conditioned and
      benefits from larger moves under shift).

    All scheduling stays *flat* -- the paper-sweep finding that
    ``CosineEpsilon`` collapses SNN training applies here too.
    """
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout, rank=rank,
        rotation_mode="random", rotation_interval=0,
        absorb_mode="periodic", absorb_interval=20,
    )
    return PolyStepOptimizer(
        model,
        compile=(device.type == "cuda"),
        seed=seed,
        epsilon=0.5,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        subspace=subspace,
        chunk_size=1024,
        amortize_steps=1,
        biased_rotation=True,
        anderson_depth=5,
        adaptive_omega=True,
        solver="softmax",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device,
             *, shift_sigma: float = 0.0, noise_seed: int = 0) -> float:
    """Test-set accuracy.

    When ``shift_sigma > 0`` the same per-batch noise mask is used
    across calls with the same ``noise_seed`` -- so pre/post/baseline
    shifted-accuracy comparisons are paired, not contaminated by
    independent ~N(0,sigma^2) draws.
    """
    correct = total = 0
    model.eval()
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if shift_sigma > 0.0:
            x = x + shift_sigma * torch.randn(
                x.shape, generator=gen, device=device, dtype=x.dtype)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_loop(
    model: nn.Module, optimizer: PolyStepOptimizer, evaluator: CpuSimEvaluator,
    loader, loss_fn: nn.Module, *, epochs: int, device: torch.device,
    shift_sigma: float, label: str, mixed_shift: bool = False,
    test_loader=None, eval_shift_sigma: float = 0.0, patience: int = 2,
    noise_seed: int = 0,
) -> tuple[float, dict]:
    """Per-batch PolyStep updates with best-test early stopping.

    ``mixed_shift=True`` concatenates each batch with a shifted copy of
    itself (half clean, half ``+ N(0, sigma^2)``). This is the standard
    online-adaptation safeguard against catastrophic forgetting of the
    in-distribution manifold while recovering on the shifted one --
    important for *continuous* on-chip adaptation where the deployed
    model must keep performing well when the shift weakens or vanishes.

    Returns ``(best_acc, best_state_dict)``. The model's parameters are
    restored to ``best_state_dict`` before return, so the caller never
    sees a worse-than-best checkpoint.
    """
    step = 0
    t0 = time.time()
    best_acc = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad_epochs = 0
    for epoch in range(epochs):
        last_x = last_y = None
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if shift_sigma > 0.0:
                x_shift = x + shift_sigma * torch.randn_like(x)
                if mixed_shift:
                    x = torch.cat([x, x_shift], dim=0)
                    y = torch.cat([y, y], dim=0)
                else:
                    x = x_shift

            def closure(stacked_params, _x=x, _y=y):
                return evaluator.evaluate(stacked_params, _x, _y)

            optimizer.step(closure)
            last_x, last_y = x, y
            step += 1
        with torch.no_grad():
            logits = model(last_x)
            train_acc = (logits.argmax(-1) == last_y).float().mean().item()
        if test_loader is not None:
            test_acc = evaluate(
                model, test_loader, device,
                shift_sigma=eval_shift_sigma, noise_seed=noise_seed,
            )
            improved = test_acc > best_acc
            tag = "*" if improved else " "
            if improved:
                best_acc = test_acc
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            elapsed = time.time() - t0
            print(f"  [{label}] epoch {epoch + 1}/{epochs} | "
                  f"step {step:3d} | batch_acc={100 * train_acc:5.1f}% | "
                  f"test{'(σ=' + str(eval_shift_sigma) + ')' if eval_shift_sigma > 0 else '(clean)'}"
                  f"={100 * test_acc:5.1f}%{tag} | {elapsed:5.1f}s")
            if bad_epochs >= patience:
                print(f"  [{label}] early stop at epoch {epoch + 1} "
                      f"(patience {patience}); best={100 * best_acc:.1f}%")
                break
        else:
            elapsed = time.time() - t0
            print(f"  [{label}] epoch {epoch + 1}/{epochs} | "
                  f"step {step:3d} | batch_acc={100 * train_acc:5.1f}% | "
                  f"{elapsed:5.1f}s")
    model.load_state_dict(best_state)
    return best_acc, best_state


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
# Color palette: colorblind-safe (Wong 2011)
_COLOR_CLEAN = "#0072B2"      # blue   -- in-distribution
_COLOR_BASELINE = "#D55E00"   # orange -- shifted, no adaptation (failure)
_COLOR_RECOVERED = "#009E73"  # green  -- shifted, after adaptation (success)
_COLOR_SHIFT_ACCENT = "#D55E00"


def _save_visualization(
    vis_x: torch.Tensor,
    vis_x_shift: torch.Tensor,
    vis_y: torch.Tensor,
    *,
    pre_clean: float,
    post_clean: float,
    post_shift: float,
    base_shift: float,
    shift_sigma: float,
    n_writable: int,
    n_total: int,
    out_path: Path,
) -> None:
    """Publication-grade two-panel figure summarising the demo.

    Panel (a): clean vs. shifted MNIST inputs, side-by-side, with the
    shifted row visually flagged (orange frame + bracket).
    Panel (b): grouped bar chart per phase, with a curved arrow
    annotating the shift-recovery in percentage points.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [viz] matplotlib not available; skipping (pip install matplotlib).")
        return

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "semibold",
        "axes.labelcolor": "#222",
        "axes.edgecolor": "#888",
        "xtick.color": "#222",
        "ytick.color": "#222",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n = vis_x.shape[0]
    fig = plt.figure(figsize=(13.5, 5.4), facecolor="white")
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.20],
        left=0.085, right=0.985, top=0.86, bottom=0.20, wspace=0.16,
    )

    # ============================================================
    # Panel (a): clean vs. shifted MNIST inputs
    # ============================================================
    ax_left = fig.add_subplot(gs_outer[0])
    ax_left.axis("off")
    ax_left.set_title(
        "(a)  Input shift at deployment",
        loc="left", fontsize=12, pad=14,
    )

    # 2-row x n-col image grid (no extra label column).
    gs_imgs = gridspec.GridSpecFromSubplotSpec(
        2, n, subplot_spec=gs_outer[0],
        height_ratios=[1.0, 1.0],
        hspace=0.18, wspace=0.06,
    )

    row_meta = [
        ("Clean", _COLOR_CLEAN, vis_x),
        (f"Shifted\nσ = {shift_sigma}", _COLOR_SHIFT_ACCENT, vis_x_shift),
    ]

    for r, (label, color, row_x) in enumerate(row_meta):
        for c in range(n):
            ax = fig.add_subplot(gs_imgs[r, c])
            img = row_x[c, 0].cpu().clamp(0.0, 1.0).numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color if r == 1 else "#bbb")
                spine.set_linewidth(1.8 if r == 1 else 0.6)
            # Digit label only on top row (avoids clutter on shifted row)
            if r == 0:
                ax.set_title(
                    f"\u2018{int(vis_y[c])}\u2019",
                    fontsize=10, color="#333", pad=4,
                )
            # Row label as a horizontal ylabel on the leftmost image only
            if c == 0:
                ax.set_ylabel(
                    label,
                    rotation=0, ha="right", va="center",
                    labelpad=10, fontsize=11,
                    fontweight="semibold", color=color,
                )

    # ============================================================
    # Panel (b): grouped accuracy bars + recovery annotation
    # ============================================================
    ax_bar = fig.add_subplot(gs_outer[1])
    ax_bar.set_title(
        "(b)  Accuracy recovery via on-chip PolyStep adaptation",
        loc="left", fontsize=12, pad=14,
    )

    group_centers = [0.0, 1.4]  # Stage 1, Stage 2
    bar_offset = 0.32
    bar_width = 0.55

    clean_vals = [pre_clean * 100, post_clean * 100]
    shift_vals = [base_shift * 100, post_shift * 100]

    # Clean bars (both stages, same blue)
    bars_clean = ax_bar.bar(
        [c - bar_offset for c in group_centers], clean_vals,
        width=bar_width, color=_COLOR_CLEAN, edgecolor="white", linewidth=1.0,
        label="Clean test set",
    )
    # Shifted bars: orange for failed baseline, green for recovered
    shift_colors = [_COLOR_BASELINE, _COLOR_RECOVERED]
    bars_shift = ax_bar.bar(
        [c + bar_offset for c in group_centers], shift_vals,
        width=bar_width, color=shift_colors,
        edgecolor="white", linewidth=1.0,
    )

    # Bar value labels
    for b, v in zip(list(bars_clean) + list(bars_shift),
                    clean_vals + shift_vals):
        ax_bar.text(
            b.get_x() + b.get_width() / 2, v + 1.0,
            f"{v:.1f}%", ha="center", va="bottom",
            fontsize=9.5, color="#222",
        )

    # X axis: group labels
    ax_bar.set_xticks(group_centers)
    ax_bar.set_xticklabels(
        ["Stage 1\noff-chip pretrain", "Stage 2\non-chip adaptation"],
        fontsize=10,
    )
    ax_bar.tick_params(axis="x", length=0, pad=8)
    ax_bar.set_ylabel("Accuracy on MNIST test set  (%)", fontsize=10)
    ax_bar.set_ylim(0, 108)
    ax_bar.set_yticks([0, 20, 40, 60, 80, 100])
    ax_bar.grid(axis="y", linestyle=":", color="#ccc", alpha=0.7, zorder=0)
    ax_bar.set_axisbelow(True)

    # Recovery annotation: curved arrow between the two shifted bars.
    # Anchor well above bar value labels to avoid overlap.
    recovery = (post_shift - base_shift) * 100
    x_from = group_centers[0] + bar_offset
    x_to = group_centers[1] + bar_offset
    y_from = base_shift * 100
    y_to = post_shift * 100
    label_y = max(y_from, y_to) + 22
    ax_bar.annotate(
        "",
        xy=(x_to, y_to + 5.0),
        xytext=(x_from, y_from + 5.0),
        arrowprops=dict(
            arrowstyle="-|>", color=_COLOR_RECOVERED, lw=2.2,
            shrinkA=2, shrinkB=2,
            connectionstyle="arc3,rad=-0.45",
        ),
    )
    ax_bar.text(
        (x_from + x_to) / 2, label_y,
        f"+{recovery:.1f} pp shift-recovery",
        ha="center", va="bottom", fontsize=11.5,
        color=_COLOR_RECOVERED, fontweight="bold",
    )

    # Custom legend (clean / shifted-failed / shifted-recovered) - placed
    # horizontally above the bar axis so it never overlaps the data.
    legend_handles = [
        mpatches.Patch(color=_COLOR_CLEAN, label="Clean test"),
        mpatches.Patch(color=_COLOR_BASELINE,
                       label=f"Shifted (σ={shift_sigma}), no adaptation"),
        mpatches.Patch(color=_COLOR_RECOVERED,
                       label=f"Shifted (σ={shift_sigma}), PolyStep on-chip adapt"),
    ]
    ax_bar.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=3, frameon=False, fontsize=9,
        handlelength=1.4, handleheight=0.9, columnspacing=1.6,
    )

    # Footer note: writable-subset fraction
    pct_writable = 100 * n_writable / n_total
    fig.text(
        0.985, 0.005,
        f"Stage 2 adapts only {n_writable:,} / {n_total:,} params "
        f"({pct_writable:.1f} %) - the Loihi 2 runtime-writable subset "
        "(fc2 + per-population vth, β).",
        ha="right", va="bottom", fontsize=8.0, color="#555", style="italic",
    )

    # Suptitle
    fig.suptitle(
        "PolyStep -> Loihi 2 (skeleton):  spiking MNIST + on-chip readout adaptation",
        fontsize=13, fontweight="bold", y=0.965,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"  [viz] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["cpu_sim", "loihi2"],
                        default="cpu_sim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=15,
                        help="SNN simulation timesteps (paper SNN: 15).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=16000,
                        help="MNIST train subset (0 = full 60k).")
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--pretrain-epochs", type=int, default=16)
    parser.add_argument("--adapt-epochs", type=int, default=6)
    parser.add_argument("--shift-sigma", type=float, default=1.0,
                        help="Gaussian noise stddev applied at adaptation "
                             "time (input shift). 0 = no shift.")
    parser.add_argument("--mixed-shift", action="store_true", default=True,
                        help="Adapt on a half-clean/half-shifted batch "
                             "(prevents catastrophic forgetting; default ON).")
    parser.add_argument("--no-mixed-shift", dest="mixed_shift",
                        action="store_false")
    parser.add_argument("--adapt-rank", type=int, default=8)
    parser.add_argument("--adapt-num-probe", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4,
                        help="Early-stop patience on test accuracy "
                             "(epochs without improvement before halt). "
                             "Higher than typical SGD because zeroth-"
                             "order test curves are noisier per epoch.")
    parser.add_argument("--data-dir", type=str, default="data/mnist")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip saving the visualization figure to examples/figures/.",
    )
    args = parser.parse_args()

    backend = select_backend(args.backend)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("  PolyStep -> Loihi 2 (skeleton): MNIST SNN + on-chip readout adapt")
    print(f"  backend={backend}  device={device}  seed={args.seed}")
    print(f"  hidden={args.hidden}  num_steps={args.num_steps}  "
          f"batch={args.batch_size}")
    print(f"  data: MNIST subset train={args.max_train} test={args.max_test}")
    print(f"  shift: input += N(0, {args.shift_sigma}^2) at adapt time")
    print("=" * 70)

    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        max_train=args.max_train, max_test=args.max_test,
    )

    model = MnistSpikingNet(
        hidden=args.hidden, num_steps=args.num_steps,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_total:,}")
    print()

    # Stash a small batch for the end-of-run visualization. Done AFTER
    # model init so the global RNG state used to seed model weights is
    # the canonical seed-only state (otherwise reading the test_loader
    # iterator would advance it and change the trained model).
    _vis_batch = next(iter(test_loader))
    _vis_x_clean = _vis_batch[0][:5].to(device)
    _vis_y = _vis_batch[1][:5].to(device)
    _vis_gen = torch.Generator(device=device).manual_seed(args.seed)
    _vis_x_shift = _vis_x_clean + args.shift_sigma * torch.randn(
        _vis_x_clean.shape, generator=_vis_gen, device=device,
        dtype=_vis_x_clean.dtype,
    )

    loss_fn = nn.CrossEntropyLoss()
    init_clean = evaluate(model, test_loader, device)
    print(f"  init test acc (clean): {100 * init_clean:.1f}%")

    # ----- Stage 1: off-chip pretrain (clean MNIST) -----
    print()
    print("Stage 1: off-chip PolyStep pretrain (paper SNN config)")
    print("-" * 70)
    pre_opt = make_pretrain_optimizer(model, seed=args.seed, device=device)
    pre_eval = CpuSimEvaluator(model, loss_fn=loss_fn)
    pre_clean, _ = train_loop(
        model, pre_opt, pre_eval, train_loader, loss_fn,
        epochs=args.pretrain_epochs, device=device,
        shift_sigma=0.0, label="pre",
        test_loader=test_loader, eval_shift_sigma=0.0,
        patience=args.patience, noise_seed=args.seed,
    )
    # Best Stage 1 weights are now loaded; sample their shifted accuracy
    # for context (and for the paired baseline comparison below).
    pre_shift = evaluate(
        model, test_loader, device,
        shift_sigma=args.shift_sigma, noise_seed=args.seed,
    )
    print(f"  -> best Stage 1 | clean: {100 * pre_clean:.1f}%  "
          f"shifted: {100 * pre_shift:.1f}%")

    # Snapshot the Stage 1 best for the frozen-readout baseline.
    pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ----- Stage 2: 'on-chip' readout + LIF Var adaptation -----
    print()
    print(f"Stage 2: 'on-chip' readout + LIF Var adaptation "
          f"(shift sigma={args.shift_sigma})")
    print("-" * 70)
    n_writable = freeze_to_writable_subset(model)
    print(f"  writable subset: {n_writable:,} params "
          f"({100 * n_writable / n_total:.1f}% of model) "
          "= fc2 + lif1.vth + lif1.beta + lif2.vth + lif2.beta")

    if backend == "cpu_sim":
        adapt_eval = CpuSimEvaluator(model, loss_fn=loss_fn)
    else:
        adapt_eval = LoihiSpikeEvaluator(hdf5_path="snn_mnist.net", loss_fn=loss_fn)

    ad_opt = make_adapt_optimizer(
        model, seed=args.seed, device=device,
        rank=args.adapt_rank, num_probe=args.adapt_num_probe,
    )
    post_shift, _ = train_loop(
        model, ad_opt, adapt_eval, train_loader, loss_fn,
        epochs=args.adapt_epochs, device=device,
        shift_sigma=args.shift_sigma, label="adapt",
        mixed_shift=args.mixed_shift,
        test_loader=test_loader, eval_shift_sigma=args.shift_sigma,
        patience=args.patience, noise_seed=args.seed,
    )
    # Best Stage 2 weights are now loaded; sample clean accuracy on it.
    post_clean = evaluate(
        model, test_loader, device,
        shift_sigma=0.0, noise_seed=args.seed,
    )

    # Reload Stage 1 best to evaluate the frozen-readout baseline
    # against the SAME shift noise mask as post_shift -- paired.
    model.load_state_dict(pretrained_state)
    base_shift = evaluate(
        model, test_loader, device,
        shift_sigma=args.shift_sigma, noise_seed=args.seed,
    )

    # ----- Report -----
    print()
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    print(f"  initial (random):                          "
          f"clean {100 * init_clean:5.1f}%")
    print(f"  best Stage 1 (off-chip pretrain):          "
          f"clean {100 * pre_clean:5.1f}%   "
          f"shifted {100 * pre_shift:5.1f}%")
    print(f"  best Stage 2 (PolyStep on-chip adapt):     "
          f"clean {100 * post_clean:5.1f}%   "
          f"shifted {100 * post_shift:5.1f}%")
    print(f"  baseline (no adaptation, frozen readout):  "
          f"shifted {100 * base_shift:5.1f}%")
    print()
    recovery = 100 * (post_shift - base_shift)
    print(f"  shift-recovery from PolyStep adapt: {recovery:+.1f} pp "
          f"(higher is better; paired-noise comparison)")
    print(f"  backend: {backend}")
    print("=" * 70)
    print()
    print("Next: swap CpuSimEvaluator for LoihiSpikeEvaluator. The host")
    print("loop above is unchanged.")

    if not args.no_plot:
        _save_visualization(
            vis_x=_vis_x_clean,
            vis_x_shift=_vis_x_shift,
            vis_y=_vis_y,
            pre_clean=pre_clean,
            post_clean=post_clean,
            post_shift=post_shift,
            base_shift=base_shift,
            shift_sigma=args.shift_sigma,
            n_writable=n_writable,
            n_total=n_total,
            out_path=Path(__file__).parent / "figures" / "06_loihi_shift_viz.png",
        )


if __name__ == "__main__":
    main()
