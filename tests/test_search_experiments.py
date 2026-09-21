"""Budget and geometry contracts for the experimental search methods."""

import pytest
import torch
from torch import nn

pytest.importorskip("experiments.runners")

from experiments.runners.search_suite import ARMS, draw_directions, run_search
from polystep.transform import ParamLayout


def test_search_cli_uses_one_budget_protocol_and_saves_baseline_only(tmp_path, monkeypatch):
    import json
    import sys
    from experiments.scripts.bench_polytope import main

    output = tmp_path / "search.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--workloads",
            "mlp",
            "--arms",
            "hybrid",
            "--seeds",
            "0",
            "--budget",
            "160",
            "--batch",
            "8",
            "--output",
            str(output),
        ],
    )
    main()
    rows = json.loads(output.read_text())
    assert len(rows) == 1
    assert rows[0]["arm"] == "hybrid"
    assert 0 < rows[0]["candidates"] <= 160
    with pytest.raises(ValueError, match="Unknown arm"):
        run_search("mlp", "hybrid_typo", 0)


@pytest.mark.parametrize("arm", ARMS)
def test_search_respects_budget_and_keeps_test_data_out_of_selection(arm):
    result = run_search("mlp", arm, 0, budget=512, batch=8, streaming=True)
    assert 0 < result["candidates"] <= 512
    assert result["sample_forwards"] == result["candidates"] * 8
    assert result["steps"] > 0
    assert result["best_validation_loss"] <= result["start_validation_loss"]
    assert all("test_loss" not in h for h in result["history"])


def test_global_search_geometry_and_momentum_direction():
    layout = ParamLayout.from_module(nn.Linear(6, 5, bias=False))
    gen = torch.Generator().manual_seed(3)
    momentum = torch.randn(layout.total_params, generator=gen, dtype=torch.float64)
    vertices, axes = draw_directions(layout, 4, gen, "cpu", torch.float64, "momentum_subspace", momentum)
    torch.testing.assert_close(axes @ axes.t(), torch.eye(4, dtype=torch.float64))
    torch.testing.assert_close(vertices[:4], -vertices[4:])
    assert (axes[0] @ momentum / momentum.norm()).abs().item() == pytest.approx(1.0)
    vertices, _ = draw_directions(layout, 4, gen, "cpu", torch.float64, "global_simplex")
    torch.testing.assert_close(vertices.mean(0), torch.zeros(layout.total_params, dtype=torch.float64))
    torch.testing.assert_close(vertices.norm(dim=1), torch.ones(5, dtype=torch.float64))
    vertices, _ = draw_directions(layout, 4, gen, "cpu", torch.float64, "lowrank")
    assert torch.linalg.matrix_rank(vertices.reshape(-1, 5, 6)).max() <= 2
    torch.testing.assert_close(vertices.norm(dim=1), torch.ones(8, dtype=torch.float64))
