"""06 - Simulated SNN readout adaptation under input shift.

I pretrain a rate-coded SNN on clean MNIST, then freeze its first linear layer.
PolyStep adapts the readout and LIF threshold/decay under Gaussian input shift.
This PyTorch simulation uses one voltage state and subtractive reset. It does
not reproduce Loihi's fixed-point arithmetic or Lava's current state and reset.

Run::

    python examples/06_loihi_snn_polystep.py
    python examples/06_loihi_snn_polystep.py --device cpu --hidden 32 --num-steps 5 --max-train 1024 --max-test 256 --pretrain-epochs 1 --adapt-epochs 1
    python examples/06_loihi_snn_polystep.py --shift-sigma 0.0
    python examples/06_loihi_snn_polystep.py --no-mixed-shift
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

import _env

_env.setup()
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from polystep import PolyStepOptimizer
from polystep.benchmarks.utils import get_mnist_loaders
from polystep.cost_nn import NNCostEvaluator
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


class LearnableLIF(nn.Module):
    """Leaky integrate-and-fire with learnable threshold and decay (subtractive reset).

    Each call is one discrete timestep. ``beta`` is a sigmoid decay factor;
    the effective ``vth`` is at least one dtype epsilon above the zero rest state.
    """

    def __init__(self, beta: float = 0.95, vth: float = 1.0):
        super().__init__()
        if not math.isfinite(beta) or not 0 < beta < 1:
            raise ValueError("beta must be finite and between 0 and 1")
        if not math.isfinite(vth) or vth <= 0:
            raise ValueError("vth must be positive and finite")
        beta_clamped = max(min(beta, 0.999), 1e-3)
        beta_logit = float(torch.logit(torch.tensor(beta_clamped)))
        self.beta_logit = nn.Parameter(torch.tensor(beta_logit))
        self.vth = nn.Parameter(torch.tensor(float(vth)))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + x
        threshold = self.vth.clamp_min(torch.finfo(mem.dtype).eps)
        spike = (mem >= threshold).to(x.dtype)
        mem = mem - spike * threshold
        return spike, mem


class MnistSpikingNet(nn.Module):
    """Two-layer rate-coded SNN for MNIST (Linear -> LIF -> Linear -> LIF).

    Returns summed spike counts over time (rate code) so cross-entropy
    has a usable signal range.
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
        cur1 = self.fc1(x)  # static input: same injected current at every timestep
        mem1 = torch.zeros(batch, self.hidden, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        total = torch.zeros(batch, 10, device=x.device, dtype=x.dtype)
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(cur1, mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            total = total + spk2
        return total  # (batch, 10): raw spike counts (rate code)


def freeze_to_writable_subset(model: MnistSpikingNet) -> int:
    """Freeze ``fc1``; adapt the readout and both neuron populations.

    Leaves ``fc2`` plus per-population ``lif{1,2}.vth`` / ``beta_logit``
    trainable. ``ParamLayout.from_module`` honours ``requires_grad``, so
    frozen tensors stay out of the OT particle.
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


def make_optimizer(
    model: nn.Module, *, seed: int, rank: int, num_probe: int, step_radius: float, probe_radius: float
) -> PolyStepOptimizer:
    """PolyStep with flat schedules on a ``HybridSubspace`` absorbed every 20 steps.

    Stage 1 uses the SNN settings of ``experiments/runners/run_elevation.py`` without
    its ``biased_rotation``. Stage 2 updates about 1.3% of the default model's
    parameters, using ``rank=8`` and a second probe.
    """
    subspace = HybridSubspace.from_layout(
        ParamLayout.from_module(model), rank=rank, absorb_mode="periodic", absorb_interval=20
    )
    return PolyStepOptimizer(
        model,
        seed=seed,
        epsilon=0.5,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        subspace=subspace,
        solver="softmax",
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, *, shift_sigma: float = 0.0, noise_seed: int = 0) -> float:
    """Test-set accuracy.

    With ``shift_sigma > 0`` the noise mask is fixed by ``noise_seed``,
    so pre/post/baseline shifted accuracies are paired comparisons.
    """
    correct = total = 0
    model.eval()
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if shift_sigma > 0.0:
            x = x + shift_sigma * torch.randn(x.shape, generator=gen, device=device, dtype=x.dtype)
        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_loop(
    model: nn.Module,
    optimizer: PolyStepOptimizer,
    evaluator: NNCostEvaluator,
    loader,
    *,
    epochs: int,
    device: torch.device,
    shift_sigma: float,
    label: str,
    mixed_shift: bool = False,
    val_loader,
    eval_shift_sigma: float = 0.0,
    patience: int = 2,
    noise_seed: int = 0,
) -> None:
    """Per-batch PolyStep updates with validation-based early stopping.

    ``mixed_shift=True`` concatenates each batch with a shifted copy of itself.
    Checkpoint selection uses validation accuracy at ``eval_shift_sigma``.
    Restores the selected parameters before return.
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

            optimizer.step(optimizer.register_evaluator(evaluator, x, y))
            last_x, last_y = x, y
            step += 1
        with torch.no_grad():
            logits = model(last_x)
            train_acc = (logits.argmax(-1) == last_y).float().mean().item()
        val_acc = evaluate(
            model,
            val_loader,
            device,
            shift_sigma=eval_shift_sigma,
            noise_seed=noise_seed,
        )
        improved = val_acc > best_acc
        tag = "*" if improved else " "
        if improved:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        elapsed = time.time() - t0
        print(
            f"  [{label}] epoch {epoch + 1}/{epochs} | "
            f"step {step:3d} | batch_acc={100 * train_acc:5.1f}% | "
            f"val{'(σ=' + str(eval_shift_sigma) + ')' if eval_shift_sigma > 0 else '(clean)'}"
            f"={100 * val_acc:5.1f}%{tag} | {elapsed:5.1f}s"
        )
        if bad_epochs >= patience:
            print(f"  [{label}] early stop at epoch {epoch + 1} (patience {patience}); best={100 * best_acc:.1f}%")
            break
    model.load_state_dict(best_state)
    optimizer.release_evaluator()


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
    """Show the input shift and paired test accuracies before and after adaptation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping the figure.")
        return

    n = vis_x.shape[0]
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    grid = fig.add_gridspec(2, n + 5)
    for row, (label, images) in enumerate((("Clean", vis_x), (f"Shifted, σ={shift_sigma}", vis_x_shift))):
        for col in range(n):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow((images[col, 0].cpu() * 0.3081 + 0.1307).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(str(int(vis_y[col])))
            if col == 0:
                ax.set_ylabel(label)

    ax = fig.add_subplot(grid[:, n:])
    for offset, values, label, color in (
        (-0.2, (pre_clean, post_clean), "Clean test", "#0072B2"),
        (0.2, (base_shift, post_shift), "Shifted test", "#D55E00"),
    ):
        bars = ax.bar([offset, 1 + offset], [100 * v for v in values], width=0.38, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks([0, 1], ["Pretrained", "Adapted"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Shift recovery: {100 * (post_shift - base_shift):+.1f} percentage points")
    ax.legend(loc="lower right")
    fig.suptitle(f"PyTorch SNN simulation: adapt {n_writable:,} of {n_total:,} parameters")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=["cpu_sim", "loihi2"], default="cpu_sim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=15, help="SNN simulation timesteps (paper SNN: 15).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=16000, help="MNIST train subset (0 = full 60k).")
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--pretrain-epochs", type=int, default=16)
    parser.add_argument("--adapt-epochs", type=int, default=6)
    parser.add_argument(
        "--shift-sigma",
        type=float,
        default=1.0,
        help="Gaussian noise stddev applied at adaptation time (input shift). 0 = no shift.",
    )
    parser.add_argument(
        "--mixed-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt on a half-clean/half-shifted batch to limit loss of clean accuracy.",
    )
    parser.add_argument("--adapt-rank", type=int, default=8)
    parser.add_argument("--adapt-num-probe", type=int, default=2)
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early-stop patience on validation accuracy "
        "(epochs without improvement before halt). "
        "Higher than typical SGD because zeroth-"
        "order validation curves are noisier per epoch.",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the visualization figure to examples/figures/.",
    )
    args = parser.parse_args()

    if args.backend != "cpu_sim":
        parser.error("This example is a PyTorch simulation; the Loihi hardware backend is not implemented.")
    if min(args.hidden, args.num_steps, args.batch_size, args.pretrain_epochs, args.adapt_epochs, args.patience) < 1:
        parser.error("hidden size, timesteps, batch size, epochs and patience must be positive")
    if args.max_train < 0 or 0 < args.max_train < 10 or args.max_test < 0:
        parser.error("max-train must be 0 or at least 10; max-test must be nonnegative")
    if not math.isfinite(args.shift_sigma) or args.shift_sigma < 0:
        parser.error("shift-sigma must be finite and nonnegative")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("  PolyStep: simulated MNIST SNN readout adaptation")
    print(f"  backend=cpu_sim  device={device}  seed={args.seed}")
    print(f"  hidden={args.hidden}  num_steps={args.num_steps}  batch={args.batch_size}")
    print(f"  data: MNIST subset train={args.max_train} test={args.max_test}")
    print(f"  shift: input += N(0, {args.shift_sigma}^2) at adapt time")
    print("=" * 70)

    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_train=args.max_train,
        max_test=args.max_test,
    )

    train, val = random_split(train_loader.dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=train_loader.generator)
    val_loader = DataLoader(val, batch_size=256, generator=torch.Generator().manual_seed(args.seed))

    model = MnistSpikingNet(
        hidden=args.hidden,
        num_steps=args.num_steps,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_total:,}")
    print()

    # Grab the visualization batch AFTER model init: reading the loader
    # earlier would advance the global RNG and change the seeded weights.
    _vis_batch = next(iter(test_loader))
    _vis_x_clean = _vis_batch[0][:5].to(device)
    _vis_y = _vis_batch[1][:5].to(device)
    _vis_gen = torch.Generator(device=device).manual_seed(args.seed)
    _vis_x_shift = _vis_x_clean + args.shift_sigma * torch.randn(
        _vis_x_clean.shape,
        generator=_vis_gen,
        device=device,
        dtype=_vis_x_clean.dtype,
    )

    loss_fn = nn.CrossEntropyLoss()
    init_clean = evaluate(model, test_loader, device)
    print(f"  init test acc (clean): {100 * init_clean:.1f}%")

    print()
    print("Stage 1: PolyStep pretraining")
    print("-" * 70)
    pre_opt = make_optimizer(model, seed=args.seed, rank=4, num_probe=1, step_radius=2.0, probe_radius=1.0)
    pre_eval = NNCostEvaluator(model, loss_fn=loss_fn)
    train_loop(
        model,
        pre_opt,
        pre_eval,
        train_loader,
        epochs=args.pretrain_epochs,
        device=device,
        shift_sigma=0.0,
        label="pre",
        val_loader=val_loader,
        eval_shift_sigma=0.0,
        patience=args.patience,
        noise_seed=args.seed,
    )
    pre_clean = evaluate(model, test_loader, device)
    pre_shift = evaluate(
        model,
        test_loader,
        device,
        shift_sigma=args.shift_sigma,
        noise_seed=args.seed,
    )
    print(f"  -> best Stage 1 | clean: {100 * pre_clean:.1f}%  shifted: {100 * pre_shift:.1f}%")

    pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print()
    print(f"Stage 2: readout and LIF adaptation (shift sigma={args.shift_sigma})")
    print("-" * 70)
    n_writable = freeze_to_writable_subset(model)
    print(
        f"  writable subset: {n_writable:,} params "
        f"({100 * n_writable / n_total:.1f}% of model) "
        "= fc2 + lif1.vth + lif1.beta + lif2.vth + lif2.beta"
    )

    adapt_eval = NNCostEvaluator(model, loss_fn=loss_fn)

    ad_opt = make_optimizer(
        model,
        seed=args.seed,
        rank=args.adapt_rank,
        num_probe=args.adapt_num_probe,
        step_radius=1.5,
        probe_radius=0.75,
    )
    train_loop(
        model,
        ad_opt,
        adapt_eval,
        train_loader,
        epochs=args.adapt_epochs,
        device=device,
        shift_sigma=args.shift_sigma,
        label="adapt",
        mixed_shift=args.mixed_shift,
        val_loader=val_loader,
        eval_shift_sigma=args.shift_sigma,
        patience=args.patience,
        noise_seed=args.seed,
    )
    post_shift = evaluate(model, test_loader, device, shift_sigma=args.shift_sigma, noise_seed=args.seed)
    post_clean = evaluate(
        model,
        test_loader,
        device,
        shift_sigma=0.0,
        noise_seed=args.seed,
    )

    # Frozen-readout baseline: same noise mask as post_shift, so paired.
    model.load_state_dict(pretrained_state)
    base_shift = evaluate(
        model,
        test_loader,
        device,
        shift_sigma=args.shift_sigma,
        noise_seed=args.seed,
    )

    print()
    print("=" * 70)
    print("  Results")
    print("=" * 70)
    print(f"  initial (random):                          clean {100 * init_clean:5.1f}%")
    print(
        f"  best Stage 1 (pretraining):               clean {100 * pre_clean:5.1f}%   shifted {100 * pre_shift:5.1f}%"
    )
    print(
        f"  best Stage 2 (simulated PolyStep adapt):   "
        f"clean {100 * post_clean:5.1f}%   "
        f"shifted {100 * post_shift:5.1f}%"
    )
    print(f"  baseline (no adaptation, frozen readout):  shifted {100 * base_shift:5.1f}%")
    print()
    recovery = 100 * (post_shift - base_shift)
    print(f"  shift-recovery from PolyStep adapt: {recovery:+.1f} pp (higher is better; paired-noise comparison)")
    print("  backend: cpu_sim (PyTorch)")
    print("=" * 70)
    print()

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
