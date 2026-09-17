"""Small runnable checks for controlled-study identities and candidate reconstruction."""

import pytest
import torch

pytest.importorskip("experiments.runners")

from experiments.runners.run_controlled import (
    RULES,
    block_candidates,
    directions,
    draw_vertices,
    grid,
    make_projection,
    reconstruct,
)


def test_controlled():
    gen = torch.Generator().manual_seed(13)
    u = draw_vertices(3, "orthoplex", gen, "cpu").double()
    assert torch.allclose(u[:, :8] @ u[:, :8].transpose(1, 2), torch.eye(8).double().expand(3, 8, 8), atol=1e-6)
    assert torch.equal(u.sum(1), torch.zeros(3, 8).double()) or u.sum(1).abs().max() < 1e-6
    cost = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]).expand(
        3, -1
    )
    for rule in RULES:
        assert len(list(grid(rule))) == 27
        for normalize in (False, True):
            d = directions(cost.double(), u, rule, 0.1, normalize)
            shift = directions(cost.double() + 7, u, rule, 0.1, normalize)
            assert torch.allclose(d, shift, atol=1e-12)
            assert torch.equal(directions(torch.ones_like(cost).double(), u, rule, 0.1, normalize), torch.zeros_like(d))
            if normalize:
                assert torch.allclose(d.norm(dim=-1), torch.ones(3).double())
    expected_top = (u[:, :3].sum(1) + u[:, 3:6].sum(1) / 3) / 4
    assert torch.allclose(directions(cost.double(), u, "top4", 0.1), expected_top)
    expected_min = u[:, :3].mean(1)
    assert torch.allclose(directions(cost.double(), u, "greedy", 0.1), expected_min)
    normal = torch.randn(3, 8, generator=gen).double()
    normal /= normal.norm(dim=-1, keepdim=True)
    s = torch.einsum("pvd,pd->pv", u, normal)
    d = directions((s >= 0).double() * 2, u, "softmax", 0.4)
    exact = -torch.tanh(torch.tensor(2 / 0.8)) * s[:, :8].abs().mean(1)
    assert torch.allclose((d * normal).sum(1), exact, atol=1e-7)
    model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
    base = {k: p.detach() for k, p in model.named_parameters()}
    sub, projection, _ = make_projection(model, 5, "cpu")
    active = sub.subspace_dim // 8 * 8
    z = torch.randn(sub.subspace_dim, generator=gen) * 0.01
    params = reconstruct(base, sub, projection, z)
    offsets = torch.randn(active // 8, 16, 8, generator=gen) * 0.01
    batch = block_candidates(params, sub, projection, 0, offsets)
    for i in range(active // 8):
        for j in (0, 7, 15):
            candidate = z.clone()
            candidate[i * 8 : (i + 1) * 8] += offsets[i, j]
            exact_params = reconstruct(base, sub, projection, candidate)
            for key in base:
                assert torch.allclose(batch[key][16 * i + j], exact_params[key], atol=1e-6)


if __name__ == "__main__":
    test_controlled()
    print("controlled-study checks passed")
