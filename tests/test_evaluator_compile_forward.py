"""Guards the in-place ``compile_forward`` (CUDA-graph) evaluator path.

The launch-bound win comes from ``torch.compile(mode="reduce-overhead")`` (CUDA
graphs) capturing the forward+loss once and replaying it per candidate while the
swap loop mutates ``param.data`` in place. The one silent-wrongness failure mode
(flagged by every reviewer): if the graph pools parameters into a static buffer,
``.data.copy_`` writes the wrong storage and every candidate replays the SAME
weights: distinct configs then yield IDENTICAL losses and OT sees zero contrast,
with no error. So the test is: distinct configs -> distinct losses.
"""

import torch
import torch.nn as nn
import pytest

from polystep.cost_nn import NNCostEvaluator
from polystep.transform import ParamLayout


class _CustomForwardNet(nn.Module):
    """Custom forward so it routes through in-place/vmap, not the bmm fast path."""

    def __init__(self, d_in=16, h=32, c=4):
        super().__init__()
        self.a = nn.Linear(d_in, h)
        self.b = nn.Linear(h, c)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)) * 1.0)


def _stacked_configs(net, n, scale=0.1, seed=0):
    torch.manual_seed(seed)
    layout = ParamLayout.from_module(net)
    flat = torch.randn(n, layout.total_params, device=next(net.parameters()).device) * scale
    return layout.batch_unflatten(flat)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile_forward needs CUDA")
def test_compile_forward_distinct_configs_distinct_losses():
    """Stale-weight guard: N distinct configs must give N distinct losses."""
    net = _CustomForwardNet().cuda()
    x = torch.rand(8, 16, device="cuda")
    y = torch.randint(0, 4, (8,), device="cuda")
    n = 6
    stacked = _stacked_configs(net, n)

    ev = NNCostEvaluator(net, nn.CrossEntropyLoss(), use_inplace=True, compile_forward=True)
    with torch.inference_mode():
        losses = ev.evaluate(stacked, x, y)

    assert losses.shape == (n,)
    assert torch.unique(losses).numel() == n, (
        "STALE-WEIGHT BUG: compiled/graphed forward replayed identical weights "
        f"across distinct configs (losses={losses.tolist()})"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile_forward needs CUDA")
def test_compile_forward_matches_eager():
    """The compiled in-place path must match the eager in-place path to fp tol."""
    net = _CustomForwardNet().cuda()
    x = torch.rand(8, 16, device="cuda")
    y = torch.randint(0, 4, (8,), device="cuda")
    stacked = _stacked_configs(net, 6)

    eager = NNCostEvaluator(net, nn.CrossEntropyLoss(), use_inplace=True, compile_forward=False)
    comp = NNCostEvaluator(net, nn.CrossEntropyLoss(), use_inplace=True, compile_forward=True)
    with torch.inference_mode():
        le = eager.evaluate(stacked, x, y)
        lc = comp.evaluate(stacked, x, y)
    assert torch.allclose(le, lc, atol=1e-4), f"compiled != eager (max diff {(le - lc).abs().max().item()})"


def test_compile_forward_falls_back_on_cpu():
    """On CPU (no CUDA graphs) compile_forward must silently use the eager forward
    and still return correct distinct losses: never crash, never go stale."""
    net = _CustomForwardNet()  # CPU
    x = torch.rand(8, 16)
    y = torch.randint(0, 4, (8,))
    n = 5
    stacked = _stacked_configs(net, n)
    ev = NNCostEvaluator(net, nn.CrossEntropyLoss(), use_inplace=True, compile_forward=True)
    with torch.inference_mode():
        losses = ev.evaluate(stacked, x, y)
    assert torch.unique(losses).numel() == n


@pytest.mark.parametrize(
    "use_inplace, compile_forward, expected",
    [
        (True, None, True),
        (False, None, False),
        (True, False, False),
        (False, True, True),
    ],
)
def test_compile_forward_defaults_to_the_inplace_path(use_inplace, compile_forward, expected):
    """CUDA graphs only help the in-place path, which is the only place they apply.

    The in-place path is a Python loop of N sequential forwards, so it is
    launch-bound. Leaving the flag off by default meant the fix never reached the
    only code that needs it. An explicit value still wins.
    """
    ev = NNCostEvaluator(
        nn.Sequential(nn.Linear(8, 4)),
        nn.MSELoss(),
        use_inplace=use_inplace,
        compile_forward=compile_forward,
    )
    assert ev._compile_forward is expected
