"""MNIST tensors from the shared benchmark loader, and the LeNet-5 that examples 10 and 13 train."""

import torch
import torch.nn as nn

from polystep.benchmarks.utils import get_mnist_loaders


def get_mnist_tensors(n_train: int = 0, n_test: int = 0):
    """Normalized train/test tensors; a zero size loads the full split."""
    train, test = get_mnist_loaders(max_train=n_train, max_test=n_test)
    return (*train.dataset.tensors, *test.dataset.tensors)


class LeNet5(nn.Module):
    """LeCun et al. 1998, with ReLU and max-pool. 61,706 parameters."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)  # 1 @ 28x28 -> 6 @ 28x28
        self.c3 = nn.Conv2d(6, 16, 5)  # 6 @ 14x14 -> 16 @ 10x10
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        pool = nn.functional.max_pool2d
        x = pool(torch.relu(self.c1(x)), 2)
        x = pool(torch.relu(self.c3(x)), 2)
        x = torch.relu(self.f5(x.flatten(1)))
        return self.out(torch.relu(self.f6(x)))
