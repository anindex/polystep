"""MNIST download and tensor loading; reads the IDX files directly, no torchvision."""

from __future__ import annotations

import gzip
import os
import struct as pystruct
from urllib.request import urlretrieve

import numpy as np
import torch

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MEAN, STD = 0.1307, 0.3081


def download_mnist(data_dir: str = "/tmp/mnist") -> None:
    os.makedirs(data_dir, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"  Downloading {filename}...")
            urlretrieve(MNIST_URL + filename, path)


def load_mnist_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        _magic, num, rows, cols = pystruct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def load_mnist_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        _magic, _num = pystruct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def get_mnist_tensors(n_train: int = 0, n_test: int = 0, data_dir: str = "/tmp/mnist"):
    """Normalized ``(train_x, train_y, test_x, test_y)``, images shaped (N, 1, 28, 28).
    ``n_train``/``n_test`` cap the split; 0 takes all of it.
    """
    download_mnist(data_dir)
    train_x = (load_mnist_images(os.path.join(data_dir, MNIST_FILES["train_images"])) - MEAN) / STD
    test_x = (load_mnist_images(os.path.join(data_dir, MNIST_FILES["test_images"])) - MEAN) / STD
    train_y = load_mnist_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    test_y = load_mnist_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))
    tr = slice(None) if n_train == 0 else slice(n_train)
    te = slice(None) if n_test == 0 else slice(n_test)
    return (
        torch.from_numpy(train_x[tr].copy()),
        torch.from_numpy(train_y[tr].copy()),
        torch.from_numpy(test_x[te].copy()),
        torch.from_numpy(test_y[te].copy()),
    )
