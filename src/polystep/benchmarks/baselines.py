"""EvoTorch CMA-ES baseline wrapper for benchmarks.

Shared implementation used across the benchmark comparison scripts. Uses
``separable=True`` for models above 10K params (O(n) memory instead of O(n^2)),
runs on GPU when available, and reports ``function_evals = generations * popsize``.

    from polystep.benchmarks.baselines import train_cmaes
    result = train_cmaes(model, train_data, train_labels, test_data, test_labels)

Needs ``pip install evotorch``.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .utils import BenchmarkResult


_HAS_EVOTORCH = False
_EVOTORCH_ERROR = None
try:
    from evotorch import Problem
    from evotorch.algorithms import CMAES

    _HAS_EVOTORCH = True
except ImportError as e:
    _EVOTORCH_ERROR = str(e)


def has_evotorch() -> bool:
    """Check if EvoTorch is available."""
    return _HAS_EVOTORCH


if _HAS_EVOTORCH:

    class NNOptimizationProblem(Problem):
        """EvoTorch Problem wrapper for neural network parameter optimization.

        This wraps a PyTorch model for use with EvoTorch's CMA-ES optimizer.
        The fitness function evaluates classification accuracy on a batch of data.

        Adapted from CMA-ES baseline.

        Args:
            model: PyTorch model to optimize.
            train_data: Training data tensor, shape (N, ...).
            train_labels: Training labels tensor, shape (N,).
            batch_size: Batch size for fitness evaluation.
            device: Device for computation ('cpu' or 'cuda').
            fitness_fn: Optional custom fitness function. If None, uses accuracy.

        Example:
            >>> model = MNISTNet()
            >>> problem = NNOptimizationProblem(model, X_train, y_train)
            >>> searcher = CMAES(problem, popsize=16, stdev_init=0.5)
            >>> searcher.step()
        """

        def __init__(
            self,
            model: nn.Module,
            train_data: Tensor,
            train_labels: Tensor,
            batch_size: int = 512,
            device: str = "cpu",
            fitness_fn: Optional[Callable[[nn.Module, Tensor, Tensor], float]] = None,
        ):
            self._model = model
            self._param_count = sum(p.numel() for p in model.parameters())
            self._param_shapes = [p.shape for p in model.parameters()]

            self._train_data = train_data.to(device)
            self._train_labels = train_labels.to(device)
            self._batch_size = batch_size
            self._fitness_fn = fitness_fn
            self._device_str = device
            self._batch_indices = None

            super().__init__(
                objective_sense="max",  # Maximize accuracy
                solution_length=self._param_count,
                dtype=torch.float32,
                device=device,
                initial_bounds=(-1.0, 1.0),
            )

        def _load_weights(self, model: nn.Module, solution: Tensor) -> None:
            """Load flattened solution vector into model parameters."""
            offset = 0
            with torch.no_grad():
                for p in model.parameters():
                    numel = p.numel()
                    p.data.copy_(solution[offset : offset + numel].view(p.shape))
                    offset += numel

        def _evaluate(self, solution) -> None:
            """Evaluate a single solution (parameter vector).

            EvoTorch calls this with a Solution object, so set the
            evaluation on the solution, not return a value.

            Args:
                solution: EvoTorch Solution object containing parameter values.
            """
            values = solution.values
            device = values.device

            batch_size = min(self._batch_size, len(self._train_data))

            # One batch per generation, shared by the population: a fresh draw per
            # candidate would rank it on different data, unlike polystep's fixed closure.
            if self._batch_indices is None or len(self._batch_indices) != batch_size:
                self._batch_indices = torch.randperm(len(self._train_data), device=device)[:batch_size]
            indices = self._batch_indices
            batch_data = self._train_data[indices]
            batch_labels = self._train_labels[indices]

            # Clone model architecture (create fresh instance)
            model = type(self._model)(*self._get_model_init_args()).to(device)
            model.eval()

            self._load_weights(model, values)

            if self._fitness_fn is not None:
                fitness = self._fitness_fn(model, batch_data, batch_labels)
            else:
                # Default: accuracy
                with torch.no_grad():
                    outputs = model(batch_data)
                    preds = outputs.argmax(dim=-1)
                    fitness = (preds == batch_labels).float().mean().item()

            solution.set_evaluation(fitness)

        def _get_model_init_args(self):
            """Get initialization arguments for model cloning.

            Override this for models with custom initialization.
            Returns empty tuple for default (no-arg) constructors.
            """
            return ()


def train_cmaes(
    model: nn.Module,
    train_data: Tensor,
    train_labels: Tensor,
    test_data: Tensor,
    test_labels: Tensor,
    generations: int = 200,
    popsize: int = 16,
    stdev_init: float = 0.5,
    batch_size: int = 512,
    device: str = "cuda",
    log_interval: int = 10,
    verbose: bool = True,
    model_init_args: tuple = (),
) -> BenchmarkResult:
    """Train model using EvoTorch CMA-ES.

    Uses separable=True for models >10K params (O(n) memory instead of O(n^2)).
    Reports function_evals = generations * popsize.

    Args:
        model: PyTorch model to optimize.
        train_data: Training data tensor, shape (N, ...).
        train_labels: Training labels tensor, shape (N,).
        test_data: Test data tensor, shape (M, ...).
        test_labels: Test labels tensor, shape (M,).
        generations: Number of CMA-ES generations.
        popsize: Population size.
        stdev_init: Initial standard deviation.
        batch_size: Batch size for fitness evaluation.
        device: Device for computation.
        log_interval: Print stats every N generations.
        verbose: Print progress.
        model_init_args: Arguments for model constructor (for cloning).

    Returns:
        BenchmarkResult with training metrics.

    Raises:
        ImportError: If EvoTorch is not installed.
    """
    if not _HAS_EVOTORCH:
        raise ImportError(
            f"EvoTorch is required for CMA-ES baseline. Install with: pip install evotorch\n"
            f"Import error: {_EVOTORCH_ERROR}"
        )

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    class ModelProblem(NNOptimizationProblem):
        def _get_model_init_args(self):
            return model_init_args

    problem = ModelProblem(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        batch_size=batch_size,
        device=device,
    )

    # Use separable CMA-ES for large models (O(n) vs O(n^2) memory)
    use_separable = param_count > 10000
    if verbose and use_separable:
        print(f"  Using separable (diagonal) CMA-ES for {param_count:,} params")

    searcher = CMAES(
        problem,
        popsize=popsize,
        stdev_init=stdev_init,
        separable=use_separable,
    )

    # Convert test data to tensors on device
    test_data_t = test_data.to(device)
    test_labels_t = test_labels.to(device)

    # Training loop
    epoch_logs = []
    best_test_acc = 0.0
    # final_accuracy reports this: selecting by test accuracy is test-set selection,
    # and polystep reports its actual final model. best_test_acc stays separate.
    last_solution = None
    start_time = time.time()

    # Build the eval model once and reuse across calls; rebuilding it on
    # every log_interval was a hot-path allocation under the original
    # benchmark configuration (generations=200, log_interval=10 ⇒ 21 rebuilds).
    eval_model = type(model)(*model_init_args).to(device)
    eval_model.eval()

    def evaluate_on_test(solution) -> float:
        """Evaluate ``solution`` on the full test set."""
        if solution is None:
            return 0.0
        values = solution.values if hasattr(solution, "values") else solution

        offset = 0
        with torch.no_grad():
            for p in eval_model.parameters():
                numel = p.numel()
                p.data.copy_(values[offset : offset + numel].view(p.shape))
                offset += numel

        # Evaluate in batches
        correct = 0
        total = 0
        eval_batch_size = 512

        with torch.no_grad():
            for i in range(0, len(test_data_t), eval_batch_size):
                batch_data = test_data_t[i : i + eval_batch_size]
                batch_labels = test_labels_t[i : i + eval_batch_size]
                outputs = eval_model(batch_data)
                preds = outputs.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

        return correct / total if total > 0 else 0.0

    for gen in range(generations):
        # New batch per generation, shared within it. See _evaluate.
        problem._batch_indices = None
        searcher.step()

        status = searcher.status
        pop_best_fitness = float(status.get("pop_best_eval", 0.0))
        mean_fitness = float(status.get("mean_eval", 0.0))
        sigma = float(status.get("stepsize", stdev_init))

        # Evaluate on test set periodically
        if (gen + 1) % log_interval == 0 or gen == generations - 1:
            pop_best_sol = status.get("pop_best", None)
            if pop_best_sol is not None:
                test_acc = evaluate_on_test(pop_best_sol)
                last_solution = pop_best_sol.clone() if hasattr(pop_best_sol, "clone") else pop_best_sol
                best_test_acc = max(best_test_acc, test_acc)
            else:
                test_acc = 0.0

            epoch_logs.append(
                {
                    "epoch": gen + 1,
                    "generation": gen + 1,
                    "accuracy": test_acc,
                    "test_accuracy": test_acc,
                    "pop_best_fitness": pop_best_fitness,
                    "mean_fitness": mean_fitness,
                    "loss": -pop_best_fitness,
                    "sigma": sigma,
                    "time": time.time() - start_time,
                }
            )

            if verbose:
                print(
                    f"  Gen {gen + 1:4d} | best_fit={pop_best_fitness:.4f} | "
                    f"mean_fit={mean_fitness:.4f} | sigma={sigma:.6f} | "
                    f"test_acc={test_acc * 100:.1f}%"
                )

    elapsed = time.time() - start_time
    total_evals = generations * popsize

    # On the last solution, not the test-best one.
    final_test_acc = evaluate_on_test(last_solution) if last_solution is not None else 0.0

    peak_memory = 0.0
    if device == "cuda" and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    return BenchmarkResult(
        optimizer="cmaes",
        seed=0,  # CMA-ES uses internal randomness
        final_accuracy=final_test_acc,
        best_accuracy=best_test_acc,
        final_loss=None,  # CMA-ES maximizes accuracy, no loss
        wall_time_seconds=elapsed,
        peak_gpu_memory_mb=peak_memory,
        total_steps=generations,
        function_evals=total_evals,
        convergence_epoch=None,
        epoch_logs=epoch_logs,
    )
