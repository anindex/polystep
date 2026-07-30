"""High-level training API for polystep.

Provides a single-call ``train()`` function that wraps PolyStepOptimizer
with an epoch-based training loop, callback system, and diagnostics.
Users can train any ``nn.Module`` without manually managing closures,
epoch iteration, or batch handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .cost_nn import NNCostEvaluator
from .optimizer import PolyStepOptimizer


class TrainCallback:
    """Base class for training callbacks.

    Subclass and override ``on_step_end`` and/or ``on_epoch_end`` to
    hook into the training loop. Return ``True`` from ``on_step_end``
    to signal early stopping.

    Example::

        class LossThresholdCallback(TrainCallback):
            def __init__(self, threshold: float = 0.01):
                self.threshold = threshold

            def on_step_end(self, metrics: dict) -> bool:
                if metrics['loss'] < self.threshold:
                    print(f"Target loss reached at step {metrics['step']}")
                    return True  # stop training
                return False
    """

    def on_step_end(self, metrics: dict) -> bool:
        """Called after each step; return True to stop training.

        ``metrics`` holds 'step', 'epoch', 'loss', 'ot_cost', 'displacement',
        'velocity_mag' and 'converged'.
        """
        return False

    def on_epoch_end(self, metrics: dict) -> None:
        """Called after each epoch completes.

        Args:
            metrics: Dict with keys 'epoch', 'avg_loss'.
        """
        pass


@dataclass
class TrainConfig:
    """Configuration for the ``train()`` function.

    Training loop parameters only. Optimizer-specific parameters (epsilon,
    step_radius, subspace, block_strategy, etc.) are set on
    ``PolyStepOptimizer``.

    Args:
        epochs: Number of training epochs. Must be > 0.
        batch_size: Convenience/documentation field. ``train()`` uses
            the provided DataLoader directly and does NOT build one
            from this value.
        log_every: Step interval for built-in logging. Must be > 0.
        callbacks: List of ``TrainCallback`` instances. ``None`` is
            normalized to an empty list.
        restore_best: If ``True`` (default), snapshot the model weights
            whenever the tracked loss reaches a new minimum and restore that
            best snapshot before ``train()`` returns. Gradient-free search is
            noisy and updates the model in place, so the *last* step is not
            necessarily the best; this reverts a late divergence. The tracked loss
            is the exact minibatch loss, one extra forward per step, which
            ``restore_best`` and any callback both require.
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
    """Train a model using the Sinkhorn Step optimizer.

    Builds the OT closure internally from the model, loss function, and
    each mini-batch. Users never handle closures directly. For each batch,
    the function constructs a closure that evaluates the model at multiple
    candidate parameter configurations via ``NNCostEvaluator``, then calls
    ``optimizer.step(closure)`` to update the model weights.

    Example::

        from polystep import PolyStepOptimizer, train, TrainConfig, LoggingCallback

        optimizer = PolyStepOptimizer(model, epsilon=0.1)
        config = TrainConfig(epochs=5, callbacks=[LoggingCallback()])
        trained_model = train(model, dataloader, loss_fn, optimizer, config)

    See Also:
        ``PolyStepOptimizer`` for optimizer configuration (epsilon,
        step_radius, subspace, block_strategy, etc.).

    Args:
        model: The ``nn.Module`` to train. Updated in-place.
        dataloader: PyTorch DataLoader yielding ``(inputs, targets)`` batches.
        loss_fn: Loss function with signature ``loss_fn(output, targets) -> scalar``.
        optimizer: A ``PolyStepOptimizer`` instance (already configured).
        config: ``TrainConfig`` with loop hyperparameters and callbacks.

    Returns:
        The same model object (mutated in-place by the optimizer).
    """
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

    # Detect model device for automatic batch transfer
    try:
        device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("Model has no trainable parameters")

    # Track the lowest-loss weights so a noisy run is not left at a late step.
    best_loss = float("inf")
    best_state = None

    try:
        for epoch in range(config.epochs):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Micro-batch: subsample for cost evaluation if configured.
                cost_bs = getattr(optimizer, "cost_batch_size", None)
                if cost_bs is not None and cost_bs < inputs.shape[0]:
                    gen = getattr(optimizer, "_generator", None)
                    if gen is not None:
                        if gen.device == inputs.device:
                            idx = torch.randperm(inputs.shape[0], device=inputs.device, generator=gen)[:cost_bs]
                        else:
                            # Devices differ (seeded CPU generator with CUDA inputs,
                            # or cuda:0 vs cuda:1). randperm needs the generator and
                            # tensor on the same device: sample on the generator's
                            # device to keep the seed, then move the index.
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

                # Without this the step reaches the objective only through closure(), so the
                # fused in-place, factored and sparse-delta evaluators never run.
                optimizer.register_evaluator(evaluator, cost_inputs, cost_targets)

                # multifidelity_screen ranks directions on a slice of the same batch
                # before spending full-fidelity forwards; returns None when off.
                # objective_token: every batch is a different objective, so cached cost
                # rows from the previous step do not describe this one.
                optimizer.step(
                    closure,
                    screen_closure=optimizer.screen_closure_from(closure, cost_inputs, cost_targets),
                    objective_token=global_step,
                )

                state = optimizer.state
                ot_cost = state.costs[-1] if state.costs else 0.0

                # ot_cost averages probe points around the pre-step weights, so it does not
                # measure the weights the model now holds. restore_best keyed on it would
                # snapshot the cheapest probe cloud. Costs one forward per 2*d*K probes.
                if callbacks or config.restore_best:
                    with torch.no_grad():
                        output = model(inputs)
                        train_loss = loss_fn(output, targets).item()
                else:
                    # Nothing reads a per-step loss: skip the forward and its host sync.
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

                # Snapshot at the lowest tracked loss, before any callback runs. Callbacks
                # receive the live model and may write to it, so snapshotting afterwards can
                # store weights that do not correspond to the loss they are keyed on.
                if config.restore_best and train_loss < best_loss:
                    best_loss = train_loss
                    # Keep the snapshot on CPU. A CUDA copy is a second full parameter set
                    # resident on the device for the whole run.
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

            # Epoch-level metrics
            avg_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0
            epoch_metrics = {
                "epoch": epoch,
                "avg_loss": avg_loss,
            }
            for cb in callbacks:
                cb.on_epoch_end(epoch_metrics)

        # Restore the best-seen weights (load_state_dict copies in place, keeping dtype/device).
        # The optimizer still holds the particle state of the LAST step, so re-anchor it on
        # the restored weights; otherwise the next step() ends in _sync_model and writes the
        # last-step weights straight back over the restored ones.
        if config.restore_best and best_state is not None:
            model.load_state_dict(best_state)
            optimizer.resync_from_model()

    finally:
        # finally: an exception from the loss, optimizer or a callback would
        # otherwise leave the last batch alive on the optimizer for its lifetime.
        optimizer.release_evaluator()

    return model


class LoggingCallback(TrainCallback):
    """Prints step metrics at a configurable interval.

    Args:
        log_every: Print every *log_every* steps (default 10).
    """

    def __init__(self, log_every: int = 10):
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
    """Stops training when loss stagnates for *patience* steps.

    Args:
        patience: Number of steps without improvement before stopping.
        min_delta: Minimum loss decrease to count as improvement.
    """

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
    """Diagnostic summary of the optimizer's history.

    Every list-valued entry has length ``iteration_count``, so they zip. The fields
    themselves are documented on :class:`~polystep.solver.SolverState`; ``evals`` is
    zero on an amortized step, which spends no forward passes at all.
    """
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
