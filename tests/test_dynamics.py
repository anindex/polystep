"""Momentum and adaptive radius dynamics."""

import pytest
import torch

from polystep.dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
    update_radius_multiplier,
    update_stagnation,
)


@pytest.mark.parametrize(
    "iteration, max_iterations, momentum_init, momentum_final, expected",
    [
        (0, 100, 0.5, 0.95, 0.5),
        (49, 100, 0.5, 0.95, 0.5 + (49 / 99) * 0.45),  # midpoint of the warm-up
        (99, 100, 0.5, 0.95, 0.95),
        (200, 100, 0.5, 0.95, 0.95),  # capped past the end
        (0, 1, 0.5, 0.95, 0.5),  # max_iterations=1 must not divide by zero
        (9, 10, 0.0, 1.0, 1.0),
    ],
)
def test_warms_up_linearly(iteration, max_iterations, momentum_init, momentum_final, expected):
    beta = compute_momentum_coefficient(
        iteration,
        max_iterations=max_iterations,
        momentum_init=momentum_init,
        momentum_final=momentum_final,
    )
    assert beta == pytest.approx(expected)


def test_default_endpoints():
    """The 0.5 to 0.95 default range, pinned so a changed default is visible."""
    assert compute_momentum_coefficient(0, 100) == pytest.approx(0.5)
    assert compute_momentum_coefficient(99, 100) == pytest.approx(0.95)


class TestApplyMomentum:
    def test_zero_velocity(self):
        """From rest the step is the plain displacement, scaled by velocity_lr."""
        X_old = torch.tensor([[1.0, 2.0]])
        X_bary = torch.tensor([[3.0, 4.0]])
        X_new, v_new = apply_momentum(X_old, X_bary, torch.zeros_like(X_old), beta=0.9)

        assert torch.allclose(v_new, X_bary - X_old)
        assert torch.allclose(X_new, X_old + (X_bary - X_old))

    def test_accumulation(self):
        """Velocity carries beta of the previous step into the next."""
        X0 = torch.tensor([[0.0, 0.0]])
        X1, v1 = apply_momentum(X0, torch.tensor([[1.0, 0.0]]), torch.zeros_like(X0), beta=0.5)
        assert torch.allclose(v1, torch.tensor([[1.0, 0.0]]))

        # v2 = 0.5 * [1,0] + [1,0]
        _, v2 = apply_momentum(X1, torch.tensor([[2.0, 0.0]]), v1, beta=0.5)
        assert torch.allclose(v2, torch.tensor([[1.5, 0.0]]))

    @pytest.mark.parametrize("velocity_lr, expected_x", [(1.0, 2.0), (0.5, 1.0)])
    def test_velocity_lr_scales_the_move(self, velocity_lr, expected_x):
        X_old = torch.tensor([[0.0, 0.0]])
        X_new, _ = apply_momentum(
            X_old, torch.tensor([[2.0, 0.0]]), torch.zeros_like(X_old), beta=0.0, velocity_lr=velocity_lr
        )
        assert torch.allclose(X_new, torch.tensor([[expected_x, 0.0]]))


# (kwargs, expected subset of the (radius_multiplier, stagnation_count, prev_loss) return).
# The radius grows on stagnation and decays on improvement, the opposite of a trust
# region; see update_radius_multiplier.
_RADIUS_CASES = [
    ("a tiny relative change counts as stagnation", dict(current_loss=1.0, prev_loss=1.0 + 1e-6), {"sc": 1}),
    ("a large change resets the counter", dict(current_loss=0.5, prev_loss=1.0, stagnation_count=5), {"sc": 0}),
    # Straddle the threshold: the counter must turn over within a factor of two of it,
    # or absorb_mode='stagnation' fires on a run that is still descending. Defaults are
    # left in place so a changed default shows up here.
    (
        "just under the default threshold still stagnates",
        dict(current_loss=1.0, prev_loss=1.0 + 0.9e-4, stagnation_count=3),
        {"sc": 4},
    ),
    (
        "just over the default threshold resets",
        dict(current_loss=1.0, prev_loss=1.0 + 1.1e-4, stagnation_count=3),
        {"sc": 0},
    ),
    (
        "reaching the default patience boosts the radius and clears the counter",
        dict(current_loss=1.0, prev_loss=1.0 + 1e-6, stagnation_count=9),
        {"rm": 1.5, "sc": 0},
    ),
    (
        "one short of the default patience does not boost",
        dict(current_loss=1.0, prev_loss=1.0, stagnation_count=8),
        {"rm": 1.0, "sc": 9},
    ),
    ("an improving step decays the radius", dict(current_loss=0.5, prev_loss=1.0), {"rm": 0.9}),
    (
        "the boost clamps at the default radius_max (2.5 * 1.5 = 3.75)",
        dict(current_loss=1.0, prev_loss=1.0 + 1e-6, stagnation_count=9, radius_multiplier=2.5),
        {"rm": 3.0},
    ),
    (
        "the decay clamps at the default radius_min (0.55 * 0.9 = 0.495)",
        dict(current_loss=0.5, prev_loss=1.0, radius_multiplier=0.55),
        {"rm": 0.5},
    ),
    (
        "the first step has no history, so nothing adapts",
        dict(current_loss=1.0, prev_loss=float("inf")),
        {"rm": 1.0, "sc": 0},
    ),
    ("the loss comes back for the caller to store", dict(current_loss=42.0, prev_loss=100.0), {"pl": 42.0}),
]


@pytest.mark.parametrize("label, kwargs, expected", _RADIUS_CASES, ids=[c[0] for c in _RADIUS_CASES])
def test_radius_and_stagnation(label, kwargs, expected):
    """Drive the pair the way both step paths do: stagnation first, then the radius."""
    current_loss = kwargs.pop("current_loss")
    prev_loss = kwargs.pop("prev_loss")
    stagnation_count = kwargs.pop("stagnation_count", 0)
    radius_multiplier = kwargs.pop("radius_multiplier", 1.0)
    threshold = {"stagnation_threshold": kwargs.pop("stagnation_threshold")} if "stagnation_threshold" in kwargs else {}

    count, returned_loss = update_stagnation(current_loss, prev_loss, stagnation_count, **threshold)
    multiplier, count = update_radius_multiplier(current_loss, prev_loss, count, radius_multiplier, **kwargs)

    if "sc" in expected:
        assert count == expected["sc"]
    if "rm" in expected:
        assert multiplier == pytest.approx(expected["rm"])
    if "pl" in expected:
        assert returned_loss == pytest.approx(expected["pl"])
