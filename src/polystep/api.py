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
        """Called after each optimizer step.

        Args:
            metrics: Dict with keys 'step', 'epoch', 'loss', 'ot_cost',
                'displacement', 'velocity_mag', 'converged'.

        Returns:
            True to stop training, False to continue.
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
        eval_every: Epoch interval for full-training-set loss evaluation.
            0 = disabled.
        callbacks: List of ``TrainCallback`` instances. None is
            normalized to an empty list.
    """

    epochs: int = 10
    batch_size: int = 32
    log_every: int = 10
    eval_every: int = 0
    callbacks: Optional[List[TrainCallback]] = None

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
        model, loss_fn=loss_fn,
        compile_vmap=getattr(optimizer, '_compile_evaluator', False),
    )
    callbacks = list(config.callbacks)
    global_step = 0
    stop = False

    # Detect model device for automatic batch transfer
    try:
        device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("Model has no trainable parameters")

    for epoch in range(config.epochs):
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Micro-batch: subsample for cost evaluation if configured
            cost_bs = getattr(optimizer, 'cost_batch_size', None)
            if cost_bs is not None and cost_bs < inputs.shape[0]:
                idx = torch.randperm(inputs.shape[0], device=inputs.device)[:cost_bs]
                cost_inputs = inputs[idx]
                cost_targets = targets[idx]
            else:
                cost_inputs = inputs
                cost_targets = targets

            # Build closure for this batch using default-argument capture
            def closure(batched_params, _in=cost_inputs, _tgt=cost_targets):
                return evaluator.evaluate(batched_params, _in, _tgt)

            optimizer.step(closure)

            # Compute training loss separately (not the OT cost)
            with torch.no_grad():
                output = model(inputs)
                train_loss = loss_fn(output, targets).item()

            epoch_loss_sum += train_loss
            epoch_loss_count += 1

            # Build metrics dict
            state = optimizer.state
            metrics = {
                'step': global_step,
                'epoch': epoch,
                'loss': train_loss,
                'ot_cost': state.costs[-1] if state.costs else 0.0,
                'displacement': (
                    state.displacement_sqnorms[-1]
                    if state.displacement_sqnorms
                    else 0.0
                ),
                'velocity_mag': (
                    torch.norm(state.velocity).item()
                    if state.velocity is not None
                    else 0.0
                ),
                'converged': (
                    state.linear_convergence[-1]
                    if state.linear_convergence
                    else True
                ),
                'absorb_count': getattr(state, 'absorb_count', 0),
            }

            # Invoke on_step_end callbacks
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
            'epoch': epoch,
            'avg_loss': avg_loss,
        }
        for cb in callbacks:
            cb.on_epoch_end(epoch_metrics)

    return model


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------


class LoggingCallback(TrainCallback):
    """Prints step metrics at a configurable interval.

    Args:
        log_every: Print every *log_every* steps (default 10).
    """

    def __init__(self, log_every: int = 10):
        self.log_every = log_every

    def on_step_end(self, metrics: dict) -> bool:
        if metrics['step'] % self.log_every == 0:
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
        self.best_loss: float = float('inf')
        self.counter: int = 0

    def on_step_end(self, metrics: dict) -> bool:
        loss = metrics['loss']
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"Early stopping at step {metrics['step']}")
            return True
        return False


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------


def get_diagnostics(optimizer: PolyStepOptimizer) -> dict:
    """Extract diagnostic summary from optimizer state.

    Args:
        optimizer: A ``PolyStepOptimizer`` instance (may have been stepped).

    Returns:
        Dict with keys: 'costs', 'displacement_sqnorms', 'convergence',
        'velocity_magnitude', 'iteration_count', 'epsilon', 'radius_multiplier',
        'absorb_count'.
    """
    state = optimizer.state
    return {
        'costs': list(state.costs),
        'displacement_sqnorms': list(state.displacement_sqnorms),
        'convergence': list(state.linear_convergence),
        'velocity_magnitude': (
            torch.norm(state.velocity).item()
            if state.velocity is not None
            else None
        ),
        'iteration_count': state.iteration_count,
        'epsilon': state.epsilon,
        'radius_multiplier': state.radius_multiplier,
        'absorb_count': getattr(state, 'absorb_count', 0),
    }
