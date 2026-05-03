"""Benchmark utilities for polystep experiments.

Provides shared model factories, data loaders, baselines, and evaluation
helpers used by the experiment runners and examples.
"""

from .utils import (
    SEEDS,
    get_mnist_loaders,
    get_cifar10_loaders,
    get_sst2_loaders,
    get_nmnist_loaders,
    get_dvs_gesture_loaders,
    MNISTNet,
    CIFAR10Net,
    create_model,
    LIFNeuron,
    SpikingNet,
    evaluate_accuracy,
    evaluate_snn_accuracy,
    BenchmarkResult,
    get_environment_info,
    save_results_json,
    format_results_table,
    compute_summary_stats,
)

from .baselines import (
    has_evotorch,
    has_nevergrad,
    train_cmaes,
    train_nevergrad,
    check_gradient_free_deps,
    get_available_optimizers,
)

__all__ = [
    "SEEDS",
    "get_mnist_loaders",
    "get_cifar10_loaders",
    "get_sst2_loaders",
    "get_nmnist_loaders",
    "get_dvs_gesture_loaders",
    "MNISTNet",
    "CIFAR10Net",
    "create_model",
    "LIFNeuron",
    "SpikingNet",
    "evaluate_accuracy",
    "evaluate_snn_accuracy",
    "BenchmarkResult",
    "get_environment_info",
    "save_results_json",
    "format_results_table",
    "compute_summary_stats",
    "has_evotorch",
    "has_nevergrad",
    "train_cmaes",
    "train_nevergrad",
    "check_gradient_free_deps",
    "get_available_optimizers",
]
