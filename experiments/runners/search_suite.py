"""Small, download-free workloads and experimental search rules for search benchmarks.

These are architecture and algorithm probes, not substitutes for the paper tasks.
"""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from polystep import PolyStepOptimizer
from polystep.baselines.methods import _lowrank_noise
from polystep.cost_nn import NNCostEvaluator
from polystep.factored_subspace import FactoredSubspace
from polystep.geometry import get_simplex_vertices
from polystep.hybrid_subspace import HybridSubspace
from polystep.layers import VmapSafeMultiHeadAttention
from polystep.quadratic_model import update_trust_region
from polystep.transform import ParamLayout
from experiments.runners.nondiff_models import HardMoENet, LIFNeuron, QuantizedMLP


class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = VmapSafeMultiHeadAttention(16, 2)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        return self.head(self.attn(x, x, x).mean(1))


class SpikeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.lif = LIFNeuron(threshold=0.5)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        current = self.fc(x)
        mem, total = torch.zeros_like(current), torch.zeros_like(current)
        for _ in range(5):
            spike, mem = self.lif(current, mem)
            total = total + spike
        return self.head(total / 5)


WORKLOADS = {
    "mlp": (lambda: nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 3)), (16,)),
    "cnn": (
        lambda: nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(256, 3)),
        (1, 8, 8),
    ),
    "attention": (AttentionClassifier, (6, 16)),
    "spiking": (SpikeClassifier, (16,)),
    "routing": (lambda: HardMoENet(input_dim=16, hidden_dim=16, num_classes=3, num_experts=2), (16,)),
    "quantized": (lambda: QuantizedMLP(input_dim=16, hidden=16, output=3), (16,)),
}


def task(name, seed, device="cpu", batch=32, train_batches=1):
    """Identical initialization and held-out splits across variants; no test-set tuning."""
    build, shape = WORKLOADS[name]
    torch.manual_seed(seed)
    model = build().to(device).eval()
    gen = torch.Generator().manual_seed(seed + 10000)
    train_size = batch * train_batches
    x = torch.randn(train_size + 256, *shape, generator=gen)
    # Shared linear teacher makes the labels learnable (unlike independent random labels).
    features = x.mean(1) if name == "attention" else x.flatten(1)
    teacher = torch.randn(features.shape[1], 3, generator=gen)
    y = (features @ teacher).argmax(1)
    splits = [
        (x[:train_size], y[:train_size]),
        (x[train_size : train_size + 128], y[train_size : train_size + 128]),
        (x[-128:], y[-128:]),
    ]
    return model, [(a.to(device), b.to(device)) for a, b in splits]


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# Existing variants plus global-subspace prototypes. Each prototype uses one
# whole-model perturbation per vertex, instead of one full sweep per parameter block.
ARMS = (
    "hybrid",
    "hybrid_orthoplex",
    "hybrid_deferred",
    "hybrid_biased",
    "hybrid_displacement",
    "hybrid_screen",
    "hybrid_amortized",
    "factored",
    "factored_rotating",
    "global_simplex",
    "global_orthoplex",
    "lowrank",
    "momentum_subspace",
    "same_batch_accept",
)


def draw_directions(layout, q, generator, device, dtype, arm, momentum=None):
    d = layout.total_params
    q = min(q, d)
    if arm == "lowrank":
        axes = _lowrank_noise([e.shape for e in layout.entries], 2, q, d, generator, device, dtype)
        # Unit norm per candidate matches probe energy, while retaining low rank.
        axes = axes / axes.norm(dim=1, keepdim=True).clamp_min(1e-30)
    else:
        raw = torch.randn(d, q, generator=generator, device=device, dtype=dtype)
        if arm == "momentum_subspace" and momentum is not None and momentum.norm() > 1e-12:
            raw[:, 0] = momentum / momentum.norm()
        axes = torch.linalg.qr(raw, mode="reduced")[0].t()
    if arm == "global_simplex":
        return get_simplex_vertices(q, device=device, dtype=dtype) @ axes, axes
    return torch.cat((axes, -axes)), axes


@torch.inference_mode()
def run_search(name, arm, seed, *, device="cpu", budget=4096, seconds=None, batch=32, target=None, streaming=False):
    """Run to a candidate or wall-clock budget; all training evaluations are charged.

    Validation after each step is timed but counted separately. Test data is touched
    only once, at the validation-selected final checkpoint. No arm tunes on test.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    model, (train, valid, test) = task(name, seed, device, batch, train_batches=4 if streaming else 1)
    x, y = train[0][:batch], train[1][:batch]
    loss_fn = nn.CrossEntropyLoss()
    evaluator = NNCostEvaluator(model, loss_fn, use_inplace=False)
    layout = ParamLayout.from_module(model)
    initial = layout.flatten(model).reshape(-1)[: layout.total_params].clone()
    gen = torch.Generator(device=device).manual_seed(seed + 20000)
    opt = None
    if arm.startswith(("hybrid", "factored")):
        sub = (
            FactoredSubspace.from_layout(layout, rank=2, rotation_interval=5 if arm == "factored_rotating" else 0)
            if arm.startswith("factored")
            else HybridSubspace.from_layout(layout, rank=2, max_subspace_dim=128)
        )
        if arm == "hybrid_displacement":
            sub.rotation_interval = 5
            sub.svd_ratio_init = sub.svd_ratio_final = 0.5
        opts = dict(polytope_type="orthoplex") if arm == "hybrid_orthoplex" else {}
        if arm == "hybrid_biased":
            opts.update(biased_rotation=True, use_quadratic_model=True)
        if arm == "hybrid_deferred":
            opts["trust_region"] = True
        if arm == "hybrid_screen":
            opts.update(multifidelity_screen=True, screen_fidelity=0.25, screen_keep_ratio=0.5)
        if arm == "hybrid_amortized":
            opts["amortize_steps"] = 3
        # Bound the simultaneous full-model move by the same .1 used by global variants.
        opt = PolyStepOptimizer(
            model,
            subspace=sub,
            solver="softmax",
            epsilon=1.0,
            ent_epsilon=0.1,
            step_radius=0.1 / math.sqrt(math.ceil(sub.subspace_dim / 8)),
            probe_radius=0.6,
            use_momentum=False,
            adaptive_probes=False,
            compile=False,
            seed=seed + 20000,
            scale_cost=None,
            **opts,
        )
        opt.register_evaluator(evaluator, x, y)

    def evaluate(points):
        return evaluator.evaluate(layout.batch_unflatten(points), x, y)

    def closure(params, inputs=None, targets=None):
        return evaluator.evaluate(params, x if inputs is None else inputs, y if targets is None else targets)

    current, momentum, radius = initial, None, 1.0
    spent, steps, accepted, validations = 0, 0, 0, 0
    best_val = float(loss_fn(model(valid[0]), valid[1]))
    start_val = best_val
    best = {k: v.clone() for k, v in model.state_dict().items()}
    target_time = None
    history = []
    synchronize(device)
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if seconds is not None and elapsed >= seconds:
            break
        if opt is not None:
            p = opt.state.X.shape[0]
            v = opt._polytope_vertices.shape[0]
            required = p * v + int(opt.trust_region or opt.biased_rotation)
            if opt.amortize_steps > 1 and opt._amortize_counter % opt.amortize_steps:
                required = 0  # coasting uses no objective forwards
        else:
            q = min(8, layout.total_params)
            required = (q + 1 if arm == "global_simplex" else 2 * q) + (2 if arm == "same_batch_accept" else 0)
        if spent + required > budget:
            break
        if streaming:
            offset = (steps % 4) * batch
            x, y = train[0][offset : offset + batch], train[1][offset : offset + batch]
        if opt is not None:
            opt.register_evaluator(evaluator, x, y)
            opt.step(
                closure,
                screen_closure=opt.screen_closure_from(closure, x, y),
                objective_token=steps if streaming else 0,
            )
            spent = sum(opt.state.evals)  # includes fractional sample-forwards for screening
        else:
            vertices, axes = draw_directions(layout, q, gen, device, current.dtype, arm, momentum)
            losses = evaluate(current + 0.3 * vertices)
            displacement = (0.1 * radius) * (torch.softmax(-losses / 0.1, 0) @ vertices)
            trial = current + displacement
            if arm == "same_batch_accept":
                incumbent, proposed = evaluate(torch.stack((current, trial)))
                gradient = (losses[:q] - losses[q:]) / 0.6
                predicted = gradient @ (axes @ displacement)
                actual = proposed - incumbent
                ratio = actual / predicted if predicted < -1e-12 else actual.new_tensor(-1.0)
                if actual <= 0 and ratio >= 0.1:
                    current = trial
                    accepted += 1
                radius = update_trust_region(predicted, actual, radius)
            else:
                current = trial
                accepted += 1
            momentum = displacement if arm != "same_batch_accept" else None
            spent += required
            params = dict(model.named_parameters())
            for key, value in layout.batch_unflatten(current[None]).items():
                params[key].copy_(value[0])
        steps += 1
        val = float(loss_fn(model(valid[0]), valid[1]))
        validations += len(valid[0])
        if val < best_val:
            best_val = val
            best = {k: v.clone() for k, v in model.state_dict().items()}
        synchronize(device)
        elapsed = time.perf_counter() - start
        if target is not None and target_time is None and best_val <= target:
            target_time = elapsed
        history.append(dict(evaluations=spent, seconds=elapsed, validation_loss=best_val))
    elapsed = time.perf_counter() - start
    model.load_state_dict(best)
    test_out = model(test[0])
    return dict(
        workload=name,
        arm=arm,
        seed=seed,
        device=str(device),
        torch=torch.__version__,
        budget=budget,
        seconds_budget=seconds,
        streaming=streaming,
        candidates=spent,
        sample_forwards=spent * batch,
        validation_sample_forwards=validations,
        steps=steps,
        accepted=accepted if opt is None else None,
        seconds=elapsed,
        start_validation_loss=start_val,
        best_validation_loss=best_val,
        test_loss=float(loss_fn(test_out, test[1])),
        test_accuracy=float((test_out.argmax(1) == test[1]).float().mean()),
        target=target,
        time_to_target=target_time,
        history=history,
    )
