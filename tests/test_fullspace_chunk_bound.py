"""Full-space monolithic chunk-size default.

In full-space mode the config buffer is (chunk, P, pdim). A default chunk of
total_evals is ~O(n_params^2) and OOMs on large models, so the default is bounded
to a fixed memory budget. Chunking only splits the loop, so results are unchanged.
"""

import torch
import torch.nn as nn

from polystep import PolyStepOptimizer
from polystep.cost_nn import NNCostEvaluator


def _model():
    # ~11K params: large enough that the default-chunk memory bound engages
    # (config buffer would otherwise be total_evals * n_params floats).
    return nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 10))


def _one_step(chunk_size):
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(16, 100)
    y = torch.randint(0, 10, (16,))
    ev = NNCostEvaluator(model, nn.CrossEntropyLoss())
    opt = PolyStepOptimizer(
        model,
        subspace=None,
        solver="softmax",
        epsilon=0.5,
        step_radius=0.1,
        probe_radius=0.5,
        chunk_size=chunk_size,
        seed=123,
    )
    return float(opt.step(lambda s: ev.evaluate(s, x, y)))


def test_fullspace_default_chunk_matches_explicit_chunk():
    """Bounded default chunk must produce the same result as an explicit chunk."""
    loss_default = _one_step(chunk_size=None)
    loss_explicit = _one_step(chunk_size=500)
    assert abs(loss_default - loss_explicit) < 1e-4


if __name__ == "__main__":
    test_fullspace_default_chunk_matches_explicit_chunk()
    print("ok")
