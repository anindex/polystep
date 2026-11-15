"""``ProbeGradient``: gradient estimates from forward passes on layer outputs."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from polystep import ProbeGradient


def _per_sample(output, target):
    return nn.functional.cross_entropy(output, target, reduction="none")


def _reference(model, x, y):
    model.zero_grad()
    with torch.enable_grad():
        nn.functional.cross_entropy(model(x), y).backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    return grads


def _estimate(model, x, y, **kwargs):
    kwargs.setdefault("probes", 4096)
    kwargs.setdefault("probe_radius", 1e-4)
    ProbeGradient(model, seed=0, **kwargs)(x, y, loss_fn=_per_sample)
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def _data(shape, classes=3, batch=12):
    gen = torch.Generator().manual_seed(1)
    return torch.randn(batch, *shape, generator=gen, dtype=torch.float64), torch.randint(
        0, classes, (batch,), generator=gen
    )


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 3)

    def forward(self, x):
        h = torch.tanh(self.a(x))
        return self.c(h + torch.tanh(self.b(h)))


class _Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.rec, self.out = nn.Linear(8, 6), nn.Linear(6, 6), nn.Linear(6, 3)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        state = torch.zeros(x.shape[0], 6, dtype=x.dtype, device=x.device)
        for _ in range(3):
            state = torch.tanh(self.rec(state) + h)
        return self.out(state)


class _Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second, self.out = nn.Linear(6, 6), nn.Linear(6, 6), nn.Linear(6, 3)
        self.second.weight = self.first.weight

    def forward(self, x):
        return self.out(torch.tanh(self.second(torch.tanh(self.first(x)))))


class _BatchReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner, self.out = nn.Linear(8, 6), nn.Linear(6, 3)

    def forward(self, x):
        z = self.inner(x)
        return self.out(z.reshape(x.shape[0], -1).softmax(-1).reshape(-1, 6))


class _InputMutated(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner, self.out = nn.Linear(8, 6), nn.Linear(6, 3)

    def forward(self, x):
        h = x.clone()
        z = self.inner(h)
        if not torch.is_grad_enabled():
            h.add_(10)
        return self.out(torch.tanh(z))


class _Ragged(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared, self.out = nn.Linear(8, 5), nn.Linear(5, 3)

    def forward(self, x):
        return self.out(torch.tanh(self.shared(x[:, :1])).sum(1) + torch.tanh(self.shared(x)).sum(1))


def _hooked():
    model = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 3))
    model[0].register_forward_hook(lambda mod, args, out: 2 * out)
    model[2].register_forward_hook(lambda mod, args, out: 3 * out)
    return model


def _frozen_weights():
    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Tanh(), nn.Flatten(), nn.Linear(72, 5), nn.Tanh(), nn.Linear(5, 3))
    model[0].weight.requires_grad_(False)
    model[3].weight.requires_grad_(False)
    return model


MODELS = {
    "mlp": (lambda: nn.Sequential(nn.Linear(8, 10), nn.Tanh(), nn.Linear(10, 7), nn.Tanh(), nn.Linear(7, 3)), (8,)),
    "cnn": (
        lambda: nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(3, 4, 3, stride=2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(36, 3),
        ),
        (1, 8, 8),
    ),
    "sequence": (lambda: nn.Sequential(nn.Linear(8, 5), nn.Tanh(), nn.Flatten(), nn.Linear(20, 3)), (4, 8)),
    "residual": (_Residual, (8,)),
    "recurrent": (_Recurrent, (8,)),
    "tied": (_Tied, (6,)),
    "inplace": (
        lambda: nn.Sequential(nn.Linear(8, 6), nn.ReLU(inplace=True), nn.Linear(6, 3), nn.ReLU(inplace=True)),
        (8,),
    ),
    "input_mutated": (_InputMutated, (8,)),
    "ragged": (_Ragged, (4, 8)),
    "hooked": (_hooked, (8,)),
    "frozen_weights": (_frozen_weights, (1, 8, 8)),
}


@pytest.mark.parametrize("name", list(MODELS))
def test_complete_frame_matches_backprop(name):
    make, shape = MODELS[name]
    torch.manual_seed(0)
    model = make().double()
    x, y = _data(shape)
    expected = _reference(model, x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _estimate(model, x, y)

    assert got.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "make, shape",
    [(lambda: nn.Linear(6, 8), (6,)), (lambda: nn.Conv2d(1, 2, 2), (1, 3, 3))],
    ids=["linear", "conv"],
)
def test_few_probes_average_to_gradient(make, shape):
    """Each sample probes a random slice of the units, and the ``units / probes`` scale keeps the mean unbiased."""
    torch.manual_seed(0)
    model = make().double()
    x = torch.randn(16, *shape, dtype=torch.float64)

    def loss_fn(output):
        return output.square().flatten(1).mean(1)

    ProbeGradient(model, probes=4096, probe_radius=1e-4, seed=0)(x, loss_fn=loss_fn)
    expected = [p.grad.clone() for p in model.parameters()]
    totals = [torch.zeros_like(p) for p in model.parameters()]
    for seed in range(400):
        ProbeGradient(model, probes=2, probe_radius=1e-4, seed=seed)(x, loss_fn=loss_fn)
        for total, p in zip(totals, model.parameters()):
            total += p.grad

    for total, reference in zip(totals, expected):
        assert (total / 400 - reference).norm() < 0.2 * reference.norm()


def test_probe_chunks_do_not_change_estimate():
    make, shape = MODELS["cnn"]
    torch.manual_seed(0)
    model = make().double()
    x, y = _data(shape)
    whole = _estimate(model, x, y, probes=6)
    chunked = _estimate(model, x, y, probes=6, max_probe_batch=24)

    for key in whole:
        torch.testing.assert_close(chunked[key], whole[key], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("inference", [False, True])
def test_last_layer_needs_no_model_forward(inference):
    """After the first call has checked the shortcut, the last layer is probed on the recorded outputs."""
    calls = []
    model = nn.Sequential(nn.Linear(8, 3)).double()
    model.register_forward_hook(lambda *_: calls.append(1))
    x, y = _data((8,))
    estimator = ProbeGradient(model, probe_radius=1e-4, seed=0)
    with torch.inference_mode(inference):
        estimator(x, y, loss_fn=_per_sample)
        calls.clear()
        estimator(x, y, loss_fn=_per_sample)

    assert len(calls) == 1
    torch.testing.assert_close(model[0].weight.grad, _reference(model, x, y)["0.weight"], rtol=1e-5, atol=1e-7)


def test_expanded_forward_runs_prefix_once():
    """Layers ahead of the probed one see the plain batch; only the layers after it see the vertices."""
    seen = []
    model = nn.Sequential(nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    model[0].register_forward_hook(lambda mod, args, out: seen.append(args[0].shape[0]))
    x, y = _data((8,))
    estimator = ProbeGradient(model, probes=2, seed=0)
    estimator(x, y, loss_fn=_per_sample)
    seen.clear()
    estimator(x, y, loss_fn=_per_sample)

    assert set(seen) == {12}


def test_model_state_and_weights_untouched():
    make, shape = MODELS["mlp"]
    model = make().double().train()
    model[1].eval()
    modes = [m.training for m in model.modules()]
    before = [p.detach().clone() for p in model.parameters()]
    x, y = _data(shape)
    loss = ProbeGradient(model, seed=0)(x, y, loss_fn=_per_sample)

    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    torch.testing.assert_close(loss, _per_sample(model.eval()(x), y).mean())


def test_finite_radius_crosses_hard_threshold():
    """Backprop sees a zero slope through a sign unit; a probe wider than the unit's margin sees the step."""

    class Sign(nn.Module):
        def forward(self, x):
            return torch.sign(x)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), Sign(), nn.Linear(16, 3)).double()
    x, y = _data((8,), batch=64)
    estimator = ProbeGradient(model, probes=16, probe_radius=1.0, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    first = estimator(x, y, loss_fn=_per_sample)
    assert model[0].weight.grad.abs().max() > 0
    for _ in range(60):
        optimizer.step()
        last = estimator(x, y, loss_fn=_per_sample)

    assert last < 0.5 * first


def test_frozen_and_unsupported_layers_get_no_gradient():
    model = nn.Sequential(nn.Conv2d(1, 2, 3, groups=1), nn.Flatten(), nn.Linear(72, 4), nn.Linear(4, 3)).double()
    model[2].weight.requires_grad_(False)
    model[2].bias.requires_grad_(False)
    x, y = _data((1, 8, 8))
    ProbeGradient(model, seed=0)(x, y, loss_fn=_per_sample)

    assert model[2].weight.grad is None and model[2].bias.grad is None
    assert model[0].weight.grad is not None and model[3].weight.grad is not None


def test_reduced_loss_rejected():
    model = nn.Sequential(nn.Linear(8, 3)).double()
    x, y = _data((8,))
    with pytest.raises(ValueError, match="one loss per sample"):
        ProbeGradient(model, seed=0)(x, y, loss_fn=nn.CrossEntropyLoss())


@pytest.mark.parametrize("kwargs", [dict(probes=0), dict(probe_radius=0.0), dict(max_probe_batch=1)])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        ProbeGradient(nn.Linear(2, 2), **kwargs)


def test_model_without_linear_layers_rejected():
    with pytest.raises(ValueError, match="no trainable"):
        ProbeGradient(nn.Sequential(nn.Tanh()))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name", ["cnn", "recurrent"])
def test_cuda_fp32_estimate_follows_backprop(name):
    make, shape = MODELS[name]
    torch.manual_seed(0)
    model = make().cuda()
    x, y = _data(shape, batch=32)
    x, y = x.float().cuda(), y.cuda()
    expected = _reference(model, x, y)
    got = _estimate(model, x, y, probe_radius=1e-2)

    for key in expected:
        cosine = nn.functional.cosine_similarity(got[key].flatten(), expected[key].flatten(), dim=0)
        assert cosine > 0.999


def test_tuple_inputs_reach_the_model():
    class TwoInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.left, self.right, self.out = nn.Linear(8, 5), nn.Linear(4, 5), nn.Linear(5, 3)

        def forward(self, a, b):
            return self.out(torch.tanh(self.left(a) + self.right(b)))

    torch.manual_seed(0)
    model = TwoInputs().double()
    gen = torch.Generator().manual_seed(1)
    a, b = (
        torch.randn(12, 8, generator=gen, dtype=torch.float64),
        torch.randn(12, 4, generator=gen, dtype=torch.float64),
    )
    y = torch.randint(0, 3, (12,), generator=gen)
    model.zero_grad()
    with torch.enable_grad():
        nn.functional.cross_entropy(model(a, b), y).backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters()}
    ProbeGradient(model, probes=64, probe_radius=1e-4, seed=0)((a, b), y, loss_fn=_per_sample)

    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-5, atol=1e-7)


def test_named_padding_convolution_skipped_with_warning():
    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding="same"), nn.Flatten(), nn.Linear(128, 3)).double()
    x, y = _data((1, 8, 8))
    with pytest.warns(UserWarning, match="gets no gradient"):
        ProbeGradient(model, seed=0)(x, y, loss_fn=_per_sample)

    assert model[0].weight.grad is None and model[2].weight.grad is not None


def test_other_batch_size_layer_warns():
    class Flattened(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.out = nn.Linear(8, 5), nn.Linear(20, 3)

        def forward(self, x):
            return self.out(torch.tanh(self.inner(x.reshape(-1, 8))).reshape(x.shape[0], -1))

    model = Flattened().double()
    x, y = _data((4, 8))
    with pytest.warns(UserWarning, match="batch size other than the input's"):
        ProbeGradient(model, seed=0)(x, y, loss_fn=_per_sample)

    assert model.inner.weight.grad is None and model.out.weight.grad is not None


def test_biasless_conv_frame_counts_weights():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 2, 3, bias=False), nn.Tanh(), nn.Flatten(), nn.Linear(72, 3)).double()
    x, y = _data((1, 8, 8))
    expected = _reference(model, x, y)
    got = _estimate(model, x, y, probes=18)

    torch.testing.assert_close(got["0.weight"], expected["0.weight"], rtol=1e-5, atol=1e-7)


def test_zero_outputs_still_get_gradient():
    model = nn.Linear(8, 3)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    x, y = _data((8,))
    x = x.float()
    expected = _reference(model, x, y)
    got = _estimate(model, x, y, probe_radius=0.1)

    torch.testing.assert_close(got["weight"], expected["weight"], rtol=1e-2, atol=1e-4)


def test_unused_layer_gradient_cleared():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.which = nn.Linear(8, 3), nn.Linear(8, 3), "a"

        def forward(self, x):
            return getattr(self, self.which)(x)

    model = Branch().double()
    x, y = _data((8,))
    estimator = ProbeGradient(model, seed=0)
    estimator(x, y, loss_fn=_per_sample)
    model.which = "b"
    estimator(x, y, loss_fn=_per_sample)

    assert model.a.weight.grad is None and model.a.bias.grad is None
    assert model.b.weight.grad is not None


def test_sample_coupled_loss_warns():
    model = nn.Sequential(nn.Linear(8, 3)).double()
    x, y = _data((8,))
    with pytest.warns(UserWarning, match="mixes samples"):
        ProbeGradient(model, seed=0)(x, y, loss_fn=lambda out, tgt: _per_sample(out - out.mean(0), tgt))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_moved_after_construction_still_probes():
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 3))
    estimator = ProbeGradient(model, probes=4, seed=0)
    model.cuda()
    x, y = _data((8,))
    estimator(x.float().cuda(), y.cuda(), loss_fn=_per_sample)

    assert model[0].weight.grad.device.type == "cuda"


def test_sparse_loss_reshape_falls_back():
    """One scored sample is enough to tell a wrong expanded forward from the repeated one."""
    torch.manual_seed(0)
    model = _BatchReshape().double()
    x, y = _data((8,), batch=16)
    mask = torch.zeros(16, dtype=torch.float64)
    mask[0] = 1

    def loss_fn(output, target):
        return _per_sample(output, target[0]) * target[1]

    model.zero_grad()
    with torch.enable_grad():
        loss_fn(model(x), (y, mask)).mean().backward()
    expected = model.inner.weight.grad.clone()
    ProbeGradient(model, probes=4096, probe_radius=1e-4, seed=0)(x, (y, mask), loss_fn=loss_fn)

    torch.testing.assert_close(model.inner.weight.grad, expected, rtol=1e-5, atol=1e-9)


def test_batch_size_change_falls_back():
    """A lone sample broadcasts against its vertices under a skip connection; a larger batch cannot."""

    class Skip(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.out = nn.Linear(8, 8), nn.Linear(8, 3)

        def forward(self, x):
            return self.out(torch.tanh(self.inner(x)) + x)

    torch.manual_seed(0)
    model = Skip().double()
    x, y = _data((8,))
    estimator = ProbeGradient(model, probes=4096, probe_radius=1e-4, seed=0)
    estimator(x[:1], y[:1], loss_fn=_per_sample)
    estimator(x, y, loss_fn=_per_sample)

    torch.testing.assert_close(model.inner.weight.grad, _reference(model, x, y)["inner.weight"], rtol=1e-5, atol=1e-7)


def test_untracked_output_write_disables_shortcut():
    class DataWrite(nn.Module):
        def __init__(self):
            super().__init__()
            self.out = nn.Linear(8, 3)

        def forward(self, x):
            z = self.out(x)
            z.data.mul_(2)
            return z

    torch.manual_seed(0)
    model = DataWrite().double()
    x, y = _data((8,))
    with torch.enable_grad():
        nn.functional.cross_entropy(2 * model.out(x), y).backward()
    expected = model.out.weight.grad.clone()
    got = _estimate(model, x, y)

    torch.testing.assert_close(got["out.weight"], expected, rtol=1e-5, atol=1e-7)


def test_check_stays_within_probe_radius():
    """A forward defined on positive outputs survives the first call when every probe keeps them positive."""

    class Positive(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(1, 1, bias=False)
            nn.init.uniform_(self.inner.weight, 0.5, 2.0)

        def forward(self, x):
            return torch.linalg.cholesky(self.inner(x).unsqueeze(-1)).squeeze(-1)

    for seed in range(20):
        torch.manual_seed(seed)
        model = Positive().double()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ProbeGradient(model, probes=1, probe_radius=0.01, seed=seed)(
                torch.ones(2, 1, dtype=torch.float64), loss_fn=lambda out: out.square().sum(1)
            )
        torch.testing.assert_close(model.inner.weight.grad, torch.ones(1, 1, dtype=torch.float64), rtol=1e-3, atol=0)


def test_pairwise_mixing_forward_warns():
    class Pairs(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.out = nn.Linear(8, 6), nn.Linear(6, 3)

        def forward(self, x):
            z = self.inner(x)
            return self.out(torch.tanh(z + 0.5 * z.reshape(-1, 2, 6).flip(1).reshape_as(z)))

    model = Pairs().double()
    x, y = _data((8,), batch=64)
    with pytest.warns(UserWarning, match="mixes samples"):
        ProbeGradient(model, seed=0)(x, y, loss_fn=_per_sample)


@pytest.mark.parametrize("batch", [1, 2])
def test_knife_edge_threshold_does_not_warn(batch):
    """A sign unit sitting on its threshold flips with the batch size, which is rounding and not mixing."""

    class Calibrated(nn.Module):
        def __init__(self, x):
            super().__init__()
            self.a, self.b = nn.Linear(17, 33), nn.Linear(33, 1)
            self.b.requires_grad_(False)
            with torch.no_grad():
                self.register_buffer("threshold", self.b(self.a(x[:1])).clone())

        def forward(self, x):
            return (self.b(self.a(x)) - self.threshold).sign()

    for seed in range(30):
        torch.manual_seed(seed)
        x = torch.randn(1, 17).repeat(batch, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ProbeGradient(Calibrated(x), probes=2, seed=seed)(x, loss_fn=lambda out: out.sum(-1))
