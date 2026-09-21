"""Practical-study contracts, with dataset/CUDA integration behind the gpu marker."""

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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "task,method",
    [
        (task, method)
        for task in ("snn", "dvs", "int8")
        for method in ("polystep_tuned", "polystep", "openai_es", "eggroll", "spsa", "mezo", "random_search")
        if task != "int8" or method in ("polystep_tuned", "polystep", "mezo")
    ],
)
@torch.inference_mode()
def test_practical_gpu_update(task, method):
    from experiments.runners.run_practical import host_data, make_engine

    data = host_data(task)[0]
    n = 32 if task == "dvs" else 128
    batch = (data[0][:n].cuda().float(), data[1][:n].cuda())
    torch.manual_seed(2026)
    tuned = method == "polystep_tuned"
    cfg = list(points(task, method))[13]
    if not tuned and "population" in cfg:
        cfg["population"] = 32
    engine = make_engine(task, method, cfg, 2026, 512 if tuned else 128, tuned)
    before = {k: p.clone() for k, p in engine.params().items()}
    old_z = None if tuned else engine.z.clone()
    for j in range(2 if tuned else 1):
        engine.progress = 0.5 * j
        assert engine.step(batch) == engine.candidates_per_step
    assert all(torch.isfinite(p).all() for p in engine.params().values())
    if tuned:
        assert any(not torch.equal(before[k], p) for k, p in engine.params().items())
    else:
        assert torch.equal(engine.z[engine.active :], old_z[engine.active :])
        assert all(torch.equal(p, engine.base[k]) for k, p in engine.model.named_parameters())
