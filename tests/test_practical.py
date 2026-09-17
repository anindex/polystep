"""Configuration-grid and exact hard-forward checks; no dataset or GPU is needed."""

import pytest
import torch

pytest.importorskip("experiments.runners")

from experiments.runners.run_practical import points
from experiments.runners.controlled_data import self_check


def test_grids_and_hard_models():
    for task in ("snn", "dvs", "int8"):
        for method in (
            "polystep_tuned",
            "polystep",
            "openai_es",
            "eggroll",
            "spsa",
            "mezo",
            "random_search",
            "adam_full",
            "adam_subspace",
        ):
            cells = list(points(task, method))
            assert len(cells) == 27
            assert len({str(c) for c in cells}) == 27
    from polystep.baselines.core import centered_rank

    values = torch.tensor([2.0, 1.0, 1.0, 3.0])
    expected = torch.tensor([1 / 6, -1 / 3, -1 / 3, 0.5])
    assert torch.allclose(centered_rank(values), expected)
    permutation = torch.tensor([3, 0, 2, 1])
    assert torch.equal(centered_rank(values[permutation]), centered_rank(values)[permutation])
    assert torch.equal(centered_rank(torch.ones(32)), torch.zeros(32))
    self_check()


if __name__ == "__main__":
    test_grids_and_hard_models()
    print("practical grids and hard-forward controls passed")
