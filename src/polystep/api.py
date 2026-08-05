"""High-level training API for polystep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .cost_nn import NNCostEvaluator
from .optimizer import PolyStepOptimizer


class TrainCallback:
    """Base class for training callbacks; override ``on_step_end``/``on_epoch_end``, return ``True`` to stop early."""

    def on_step_end(self, metrics: dict) -> bool:
        """Called after each step; return True to stop training."""
        return False

    def on_epoch_end(self, metrics: dict) -> None:
        """Called after each epoch completes."""
        pass


@dataclass
class TrainConfig:
    """Configuration for the ``train()`` function; optimizer params live on ``PolyStepOptimizer``.

    ``restore_best`` snapshots the weights at the lowest minibatch loss and restores them, since the last step of a noisy run is not the best.
    """

    epochs: int = 10
    batch_size: int = 32
    log_every: int = 10
    callbacks: Optional[List[TrainCallback]] = None
    restore_best: bool = True

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be > 0, got {self.log_every}")
        if self.callbacks is None:
            self.callbacks = []


def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: PolyStepOptimizer,
    config: TrainConfig,
) -> nn.Module:
    """Train a model with the Sinkhorn Step optimizer; builds the OT closure internally."""
    if getattr(optimizer, "model", model) is not model:
        raise ValueError("train() got a different model than the optimizer was built on.")
    if getattr(optimizer, "trust_region", False):
        # The optimizer clears the pending pair on every objective_token change, so the multiplier would sit at 1.0.
        raise ValueError(
            "trust_region is not supported by train(): its ratio compares a prediction "
            "against the loss on the next batch, which is a different objective. "
            "Drop trust_region, or drive optimizer.step() on a fixed batch."
        )
    was_training = model.training
    evaluator = NNCostEvaluator(
        model,
        loss_fn=loss_fn,
        compile_vmap=getattr(optimizer, "compile_evaluator", False),
        compile_forward=getattr(optimizer, "compile_forward", False),
        autocast_dtype=getattr(optimizer, "_candidate_autocast_dtype", None),
    )
    callbacks = list(config.callbacks)
    global_step = 0
    stop = False

    try:
        device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("Model has no trainable parameters")

    # Track the lowest-loss weights so a noisy run isn't left at a late step.
    best_loss = float("inf")
    best_state = None

    try:
        for epoch in range(config.epochs):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Micro-batch: subsample for cost evaluation.
                cost_bs = getattr(optimizer, "cost_batch_size", None)
                if cost_bs is not None and cost_bs < inputs.shape[0]:
                    gen = getattr(optimizer, "_generator", None)
                    if gen is not None:
                        if gen.device == inputs.device:
                            idx = torch.randperm(inputs.shape[0], device=inputs.device, generator=gen)[:cost_bs]
                        else:
                            # randperm needs the generator and tensor on one device; sample on the generator's device to keep the seed, then move the index.
                            idx = torch.randperm(inputs.shape[0], device=gen.device, generator=gen)[:cost_bs].to(
                                inputs.device
                            )
                    else:
                        idx = torch.randperm(inputs.shape[0], device=inputs.device)[:cost_bs]
                    cost_inputs = inputs[idx]
                    cost_targets = targets[idx]
                else:
                    cost_inputs = inputs
                    cost_targets = targets

                def closure(batched_params, _in=cost_inputs, _tgt=cost_targets):
                    return evaluator.evaluate(batched_params, _in, _tgt)

                # Without this the fused in-place, factored and sparse-delta evaluators never run.
                optimizer.register_evaluator(evaluator, cost_inputs, cost_targets)

                # objective_token: every batch is a different objective, so cached rows from the last step don't describe this one.
                optimizer.step(
                    closure,
                    screen_closure=optimizer.screen_closure_from(closure, cost_inputs, cost_targets),
                    objective_token=global_step,
                )

                state = optimizer.state
                ot_cost = state.costs[-1] if state.costs else 0.0

                # ot_cost averages probes around the pre-step weights, so it doesn't measure the current weights; restore_best keyed on it would snapshot the cheapest probe cloud.
                if callbacks or config.restore_best:
                    with torch.inference_mode():
                        output = model(cost_inputs)
                        train_loss = loss_fn(output, cost_targets).item()
                else:
                    # Nothing reads a per-step loss; skip the forward and its host sync.
                    train_loss = ot_cost

                if callbacks:
                    metrics = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": train_loss,
                        "ot_cost": ot_cost,
                        "displacement": (state.displacement_sqnorms[-1] if state.displacement_sqnorms else 0.0),
                        "velocity_mag": (torch.norm(state.velocity).item() if state.velocity is not None else 0.0),
                        "converged": (state.linear_convergence[-1] if state.linear_convergence else False),
                        "absorb_count": getattr(state, "absorb_count", 0),
                    }
                else:
                    metrics = None

                epoch_loss_sum += train_loss
                epoch_loss_count += 1

                # Snapshot before callbacks run: they may write to the live model.
                if config.restore_best and train_loss < best_loss:
                    best_loss = train_loss
                    # Keep the snapshot on CPU.
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

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
        # Re-anchor the optimizer on the restored weights; otherwise the next step writes the last-step weights back over them.
        if config.restore_best and best_state is not None:
            model.load_state_dict(best_state)
            optimizer.resync_from_model()
        # An exception would otherwise leave the last batch alive on the optimizer.
        optimizer.release_evaluator()
        # NNCostEvaluator switched to eval so candidates score against frozen statistics.
        model.train(was_training)

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
                f"disp={metrics['displacement']:.6f} "
                f"converged={metrics['converged']}"
            )
        return False

    def on_epoch_end(self, metrics: dict) -> None:
        print(f"--- Epoch {metrics['epoch']} complete | avg_loss={metrics['avg_loss']:.4f} ---")


class EarlyStoppingCallback(TrainCallback):
    """Stops training when loss stagnates for ``patience`` steps."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
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
        "absorb_count": getattr(state, "absorb_count", 0),
    }
