"""High-level training API for polystep."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils.data import DataLoader

from .optimizer import PolyStepOptimizer


class TrainCallback:
    """Training hooks; return ``True`` from ``on_step_end`` to stop early."""

    def on_step_end(self, metrics: dict) -> bool:
        """Called after each step; return True to stop training."""
        return False

    def on_epoch_end(self, metrics: dict) -> None:
        """Called after each epoch completes."""
        pass


@dataclass
class TrainConfig:
    """Configuration for the ``train()`` function; optimizer params live on ``PolyStepOptimizer``.

    ``restore_best`` snapshots the weights at the lowest loss and restores them, since the last step of a noisy run
    is not the best. The loss is in-sample: it scores each step's weights on the first training batch after
    ``cost_batch_size`` subsampling, so scores compare across steps, with the candidates' input cast, autocast and loss
    reduction. Callbacks still receive the loss on each step's own batch.
    """

    epochs: int = 10
    callbacks: Optional[List[TrainCallback]] = None
    restore_best: bool = True

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.callbacks is None:
            self.callbacks = []


def _batch_inputs_targets(batch):
    """Split a loader batch into model inputs and the target, or None for an unsupervised batch."""
    if isinstance(batch, torch.Tensor):
        return batch, None
    if isinstance(batch, Mapping):
        return {k: v for k, v in batch.items() if k != "labels"}, batch.get("labels")
    if isinstance(batch, (tuple, list)) and batch:
        if len(batch) == 1:
            return batch[0], None
        if len(batch) == 2:
            return batch[0], batch[1]
        return tuple(batch[:-1]), batch[-1]
    raise TypeError(
        "train() expects each batch to be a tensor, a tuple or list that ends with the target, "
        f"or a dict whose 'labels' entry is the target; got {type(batch).__name__}."
    )


def _batch_loss(evaluator, inputs, targets, device) -> float:
    """Loss of the live weights on one batch, with the candidates' input cast and autocast; buffers and RNG kept."""
    with torch.inference_mode(), evaluator._probe_frame(device), evaluator._autocast(device):
        x = evaluator.cast_inputs(inputs, evaluator._input_dtype(evaluator._param_dict_cache))
        return evaluator._forward_loss(x, targets).item()


def _same_state(saved, live) -> bool:
    """Whether a snapshot entry matches the live one, nested tensors included."""
    try:
        (saved, saved_spec), (live, live_spec) = tree_flatten(saved), tree_flatten(live)
        return saved_spec == live_spec and all(
            a.dtype == b.dtype and torch.equal(a.cpu(), b.detach().cpu())
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            else not isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor) and bool(a == b)
            for a, b in zip(saved, live)
        )
    except Exception:
        return False


def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: PolyStepOptimizer,
    config: TrainConfig,
) -> nn.Module:
    """Train a model with the PolyStep optimizer; builds the OT closure internally."""
    module = getattr(model, "_orig_mod", model)
    if getattr(optimizer, "model", module) is not module:
        raise ValueError("train() got a different model than the optimizer was built on.")
    if getattr(optimizer, "trust_region", False):
        raise ValueError(
            "trust_region is not supported by train(): its ratio compares a prediction "
            "against the loss on the next batch, which is a different objective. "
            "Drop trust_region, or drive optimizer.step() on a fixed batch."
        )
    was_training = module.training
    module.eval()
    callbacks = list(config.callbacks)
    global_step = 0
    stop = False

    try:
        device = next(module.parameters()).device
    except StopIteration:
        raise ValueError("Model has no trainable parameters")

    # Track the lowest-loss weights so a noisy run isn't left at a late step.
    best_loss = float("inf")
    best_state = None
    reference = None

    try:
        for epoch in range(config.epochs):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for batch in dataloader:
                inputs, targets = tree_map(
                    lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, _batch_inputs_targets(batch)
                )
                optimizer.step_batch(inputs, targets, loss_fn=loss_fn)

                state = optimizer.state
                ot_cost = state.costs[-1] if state.costs else 0.0

                # Score the updated weights; the probe cost describes the pre-step neighborhood.
                evaluator = optimizer._step_batch_evaluator
                if callbacks:
                    train_loss = _batch_loss(evaluator, optimizer._fused_inputs, optimizer._fused_targets, device)
                else:
                    # Nothing reads a per-step loss; skip the forward and its host sync.
                    train_loss = ot_cost
                if config.restore_best:
                    if reference is None:
                        reference = tree_map(
                            lambda t: t.clone() if isinstance(t, torch.Tensor) else t,
                            (optimizer._fused_inputs, optimizer._fused_targets),
                        )
                    reference_loss = _batch_loss(evaluator, *reference, device)

                if callbacks:
                    metrics = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": train_loss,
                        "ot_cost": ot_cost,
                        "displacement": (state.displacement_sqnorms[-1] if state.displacement_sqnorms else 0.0),
                        "velocity_mag": (torch.norm(state.velocity).item() if state.velocity is not None else 0.0),
                        "converged": (state.linear_convergence[-1] if state.linear_convergence else False),
                        "absorb_count": state.absorb_count,
                    }
                else:
                    metrics = None

                epoch_loss_sum += train_loss
                epoch_loss_count += 1

                # Snapshot before callbacks run: they may write to the live model.
                if config.restore_best and reference_loss < best_loss:
                    best_loss = reference_loss
                    # Keep the snapshot on CPU.
                    best_state = {
                        k: v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                        for k, v in module.state_dict().items()
                    }

                if metrics is not None:
                    for cb in callbacks:
                        if cb.on_step_end(metrics):
                            stop = True
                            break

                if stop:
                    break

                global_step += 1

            if stop:
                break

            avg_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0
            epoch_metrics = {
                "epoch": epoch,
                "avg_loss": avg_loss,
            }
            for cb in callbacks:
                cb.on_epoch_end(epoch_metrics)

    finally:
        # Re-anchor so the next step starts from the restored weights.
        if config.restore_best and best_state is not None:
            live = module.state_dict()
            if any(not _same_state(v, live[k]) for k, v in best_state.items()):
                module.load_state_dict(best_state)
                optimizer.resync_from_model()
        # An exception would otherwise leave the last batch alive on the optimizer.
        optimizer.release_evaluator()
        # NNCostEvaluator switched to eval so candidates score against frozen statistics.
        module.train(was_training)

    return model


class LoggingCallback(TrainCallback):
    """Prints step metrics at a configurable interval."""

    def __init__(self, log_every: int = 10):
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}.")
        self.log_every = log_every

    def on_step_end(self, metrics: dict) -> bool:
        if metrics["step"] % self.log_every == 0:
            print(
                f"[Step {metrics['step']}] "
                f"loss={metrics['loss']:.4f} "
                f"ot_cost={metrics['ot_cost']:.4f} "
                f"disp_sq={metrics['displacement']:.3e} "
                f"converged={metrics['converged']}"
            )
        return False

    def on_epoch_end(self, metrics: dict) -> None:
        print(f"--- Epoch {metrics['epoch']} complete | avg_loss={metrics['avg_loss']:.4f} ---")


class EarlyStoppingCallback(TrainCallback):
    """Stops training when loss stagnates for ``patience`` steps."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0

    def on_step_end(self, metrics: dict) -> bool:
        loss = metrics["loss"]
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"Early stopping at step {metrics['step']}")
            return True
        return False


def get_diagnostics(optimizer: PolyStepOptimizer) -> dict:
    """Diagnostic summary of the optimizer's history."""
    state = optimizer.state
    return {
        "costs": list(state.costs),
        "ess": list(state.ess),
        "rho": list(state.rho),
        "evals": list(state.evals),
        "displacement_sqnorms": list(state.displacement_sqnorms),
        "convergence": list(state.linear_convergence),
        "velocity_magnitude": (torch.norm(state.velocity).item() if state.velocity is not None else None),
        "iteration_count": state.iteration_count,
        "epsilon": state.epsilon,
        "radius_multiplier": state.radius_multiplier,
        "absorb_count": state.absorb_count,
    }
