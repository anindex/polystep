"""Benchmark utilities for polystep experiments.

Provides shared model factories, data loaders, baselines, and evaluation
helpers used by the experiment runners and examples.
"""

from .utils import (
    get_mnist_loaders,
    get_cifar10_loaders,
    MNISTNet,
    CIFAR10Net,
    LIFNeuron,
    SpikingNet,
    evaluate_accuracy,
    BenchmarkResult,
    get_environment_info,
)

from .baselines import (
    has_evotorch,
    train_cmaes,
)

__all__ = [
    "get_mnist_loaders",
    "get_cifar10_loaders",
    "MNISTNet",
    "CIFAR10Net",
    "LIFNeuron",
    "SpikingNet",
    "evaluate_accuracy",
    "BenchmarkResult",
    "get_environment_info",
    "has_evotorch",
    "train_cmaes",
]
