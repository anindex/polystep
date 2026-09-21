"""MNIST tensors from the shared benchmark loader; no torchvision required."""

from polystep.benchmarks.utils import get_mnist_loaders


def get_mnist_tensors(n_train: int = 0, n_test: int = 0, data_dir: str = "/tmp/mnist"):
    """Normalized train/test tensors; a zero size loads the full split."""
    train, test = get_mnist_loaders(data_dir=data_dir, max_train=n_train, max_test=n_test)
    return (*train.dataset.tensors, *test.dataset.tensors)
