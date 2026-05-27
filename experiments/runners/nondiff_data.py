"""Data generators for the non-differentiable experiment runners.

- MAX-SAT: random 3-SAT instances via PySAT with clause tensors
- Permutation learning: random sequences with target sort permutations
- Multi-domain: combined MNIST + Fashion-MNIST (20 classes)

Consumed by ``run_maxsat.py``, ``run_elevation.py``, ``run_mnist.py``,
and ``run_moe.py`` under ``experiments/runners/``.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


__all__ = [
    "generate_maxsat_instance",
    "generate_sorting_data",
    "generate_multidomain_data",
]


def generate_maxsat_instance(
    num_vars: int = 100,
    num_clauses: Optional[int] = None,
    k: int = 3,
    ratio: float = 4.27,
    seed: int = 42,
) -> Dict:
    """Generate a random k-SAT instance using PySAT.

    Creates a random k-SAT formula at the specified clause-to-variable ratio
    and converts it to PyTorch tensors for use with neural MAX-SAT solvers.

    Args:
        num_vars: Number of Boolean variables.
        num_clauses: Number of clauses. If None, computed as
            ``int(num_vars * ratio)`` (critical ratio for 3-SAT ~ 4.27).
        k: Number of literals per clause (default 3 for 3-SAT).
        ratio: Clause-to-variable ratio (used when num_clauses is None).
        seed: Random seed for reproducible clause generation.

    Returns:
        Dict with keys:
            - 'clause_vars': Tensor of shape (num_clauses, k), dtype long,
              containing 0-indexed variable indices.
            - 'clause_signs': Tensor of shape (num_clauses, k), dtype float,
              1.0 for positive literals, 0.0 for negated literals.
            - 'num_vars': int, number of variables.
            - 'num_clauses': int, number of clauses.
            - 'cnf': PySAT CNF object (for reference solvers like RC2/WalkSAT).
    """
    from pysat.formula import CNF

    if num_clauses is None:
        num_clauses = int(num_vars * ratio)

    rng = random.Random(seed)
    cnf = CNF()

    for _ in range(num_clauses):
        # Sample k distinct variables (1-indexed for PySAT convention)
        vars_chosen = rng.sample(range(1, num_vars + 1), k)
        # Randomly negate each literal
        clause = [v if rng.random() < 0.5 else -v for v in vars_chosen]
        cnf.append(clause)

    # Convert to tensors
    clause_vars_list = []
    clause_signs_list = []
    for clause in cnf.clauses:
        vars_row = [abs(lit) - 1 for lit in clause]  # 0-indexed
        signs_row = [1.0 if lit > 0 else 0.0 for lit in clause]
        clause_vars_list.append(vars_row)
        clause_signs_list.append(signs_row)

    clause_vars = torch.tensor(clause_vars_list, dtype=torch.long)
    clause_signs = torch.tensor(clause_signs_list, dtype=torch.float)

    return {
        "clause_vars": clause_vars,
        "clause_signs": clause_signs,
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "cnf": cnf,
    }


def generate_sorting_data(
    N: int = 50,
    num_samples: int = 1000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random sequences and their target sort permutations.

    Creates random float sequences and computes the permutation that sorts
    each sequence in ascending order. Used for permutation learning experiments.

    Args:
        N: Length of each sequence.
        num_samples: Number of sequence/permutation pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sequences, permutations):
            - sequences: Tensor of shape (num_samples, N), random floats in [0, 1).
            - permutations: Tensor of shape (num_samples, N), each row is the
              permutation of indices 0..N-1 that sorts the corresponding sequence.
    """
    gen = torch.Generator().manual_seed(seed)
    sequences = torch.rand(num_samples, N, generator=gen)
    permutations = sequences.argsort(dim=-1)
    return sequences, permutations


class _LabelOffsetDataset(Dataset):
    """Wrapper that adds a fixed offset to dataset labels.

    Used to shift Fashion-MNIST labels from 0-9 to 10-19 when combining
    with MNIST for multi-domain classification.
    """

    def __init__(self, dataset: Dataset, offset: int):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.offset


def generate_multidomain_data(
    data_dir: str = "data/",
    batch_size: int = 512,
) -> Dict:
    """Generate combined MNIST + Fashion-MNIST dataset with 20 classes.

    Loads both MNIST (labels 0-9) and Fashion-MNIST (labels offset to 10-19),
    concatenates them into a single 20-class classification problem.

    Args:
        data_dir: Directory to store/load dataset files.
        batch_size: Batch size for DataLoaders.

    Returns:
        Dict with keys:
            - 'train_loader': DataLoader for combined training set.
            - 'test_loader': DataLoader for combined test set.
            - 'num_classes': 20 (MNIST 0-9, Fashion-MNIST 10-19).
    """
    from experiments.runners.common import load_mnist, load_fashion_mnist

    # Load both datasets (returns DataLoaders; we need the underlying datasets)
    # Use the underlying torchvision datasets directly for concatenation
    from torchvision import datasets, transforms
    import os

    # MNIST transform (matching common.py load_mnist normalization)
    mnist_dir = os.path.join(data_dir, "mnist")
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    mnist_train = datasets.MNIST(mnist_dir, train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST(mnist_dir, train=False, download=True, transform=mnist_transform)

    # Fashion-MNIST transform
    fmnist_dir = os.path.join(data_dir, "fashion_mnist")
    fmnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    fmnist_train = datasets.FashionMNIST(fmnist_dir, train=True, download=True, transform=fmnist_transform)
    fmnist_test = datasets.FashionMNIST(fmnist_dir, train=False, download=True, transform=fmnist_transform)

    # Offset Fashion-MNIST labels by 10
    fmnist_train_offset = _LabelOffsetDataset(fmnist_train, offset=10)
    fmnist_test_offset = _LabelOffsetDataset(fmnist_test, offset=10)

    # Combine datasets
    combined_train = ConcatDataset([mnist_train, fmnist_train_offset])
    combined_test = ConcatDataset([mnist_test, fmnist_test_offset])

    train_loader = DataLoader(
        combined_train, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        combined_test, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "num_classes": 20,
    }
