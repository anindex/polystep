"""Vectorized CartPole-v1 evaluator for PolyStep RL benchmarks.

Pure analytic dynamics - no Gymnasium dependency at evaluation time. The
dynamics, thresholds, and initial state distribution match Gymnasium's
``CartPole-v1`` exactly so SB3 baselines trained on the Gym env transfer.

Reward: +1 per step while alive; episode terminates when ``|x| > 2.4`` or
``|theta| > 12 deg``. Max horizon 500 ⇒ max return 500.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch

from .policies import count_stacked_candidates


OBS_DIM = 4
ACTION_DIM = 2
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5  # actually half the pole's length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates
THETA_THRESHOLD = 12 * 2 * math.pi / 360
X_THRESHOLD = 2.4
INIT_RANGE = 0.05
DEFAULT_HORIZON = 500


def sample_initial_states(
    num: int, *, seed: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Uniform [-0.05, 0.05]^4 init matching Gym's ``CartPole-v1.reset``."""

    device = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    states = (torch.rand(num, OBS_DIM, generator=gen) * 2 - 1) * INIT_RANGE
    return states.to(device)


def cartpole_step(
    states: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized CartPole transition.

    Args:
        states: ``(N, 4)`` float tensor [x, x_dot, theta, theta_dot].
        actions: ``(N,)`` long tensor in {0, 1}.

    Returns:
        ``(next_states, reward, done)`` where reward is +1 (alive) and ``done``
        flags physically-terminated episodes.
    """

    x = states[..., 0]
    x_dot = states[..., 1]
    theta = states[..., 2]
    theta_dot = states[..., 3]

    force = torch.where(
        actions.bool(),
        torch.full_like(x, FORCE_MAG),
        torch.full_like(x, -FORCE_MAG),
    )
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    next_x = x + TAU * x_dot
    next_x_dot = x_dot + TAU * xacc
    next_theta = theta + TAU * theta_dot
    next_theta_dot = theta_dot + TAU * thetaacc

    next_states = torch.stack(
        [next_x, next_x_dot, next_theta, next_theta_dot], dim=-1
    )
    done = (
        (next_x.abs() > X_THRESHOLD)
        | (next_theta.abs() > THETA_THRESHOLD)
    )
    reward = torch.ones_like(x)
    return next_states, reward, done


@dataclass
class CartPoleRolloutResult:
    returns: torch.Tensor
    lengths: torch.Tensor
    successes: torch.Tensor  # episode survived full horizon


def _batched_mlp_logits(
    obs: torch.Tensor,
    stacked_params: Dict[str, torch.Tensor],
    n_candidates: int,
    rollouts: int,
) -> torch.Tensor:
    """Apply N policies to (N, R, 4) observations via bmm. Returns (N, R, 2)."""

    x = obs  # (N, R, 4)
    linear_indices = sorted(
        int(k.split(".")[1]) for k in stacked_params if k.endswith(".weight")
    )
    for j, idx in enumerate(linear_indices):
        w = stacked_params[f"net.{idx}.weight"]  # (N, out, in)
        b = stacked_params[f"net.{idx}.bias"]    # (N, out)
        x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
        if j < len(linear_indices) - 1:
            x = torch.tanh(x)
    return x


class CartPoleEvaluator:
    """Vectorized evaluator for stacked MLP policies on CartPole-v1."""

    env_id = "CartPole-v1"
    obs_dim = OBS_DIM
    action_dim = ACTION_DIM
    action_type = "discrete"

    def __init__(
        self,
        rollouts_per_candidate: int = 32,
        horizon: int = DEFAULT_HORIZON,
        device: str = "cpu",
    ):
        self.rollouts_per_candidate = int(rollouts_per_candidate)
        self.horizon = int(horizon)
        self.device = torch.device(device)

    def rollout_stacked_params(
        self,
        stacked_params: Dict[str, torch.Tensor],
        *,
        seed: int,
        step: int,
    ) -> CartPoleRolloutResult:
        n_candidates = count_stacked_candidates(stacked_params)
        R = self.rollouts_per_candidate
        # CRN: same initial states for every candidate within a step; vary across steps.
        states = sample_initial_states(
            R, seed=int(seed) + 1009 * int(step), device=self.device,
        ).unsqueeze(0).expand(n_candidates, R, OBS_DIM).contiguous()

        returns = torch.zeros(n_candidates, R, device=self.device)
        lengths = torch.zeros(n_candidates, R, device=self.device)
        active = torch.ones(n_candidates, R, dtype=torch.bool, device=self.device)
        # Send stacked params to device.
        sp = {k: v.to(self.device) for k, v in stacked_params.items()}

        # Check ``active.any()`` only every ``early_stop_check`` steps to
        # amortize the GPU-CPU sync cost. CartPole rewards stop accumulating
        # once an env is inactive, so running a few extra "dead" iterations
        # is cheap and keeps the inner loop fully on-device.
        early_stop_check = max(1, self.horizon // 8)
        for t in range(self.horizon):
            logits = _batched_mlp_logits(states, sp, n_candidates, R)  # (N, R, 2)
            actions = logits.argmax(dim=-1)  # (N, R)
            flat_states = states.reshape(-1, OBS_DIM)
            flat_actions = actions.reshape(-1)
            next_flat, rewards_flat, done_flat = cartpole_step(flat_states, flat_actions)
            next_states = next_flat.reshape(n_candidates, R, OBS_DIM)
            rewards = rewards_flat.reshape(n_candidates, R)
            done = done_flat.reshape(n_candidates, R)

            returns = returns + torch.where(active, rewards, torch.zeros_like(rewards))
            lengths = lengths + active.float()
            states = torch.where(active.unsqueeze(-1), next_states, states)
            active = active & ~done
            if (t + 1) % early_stop_check == 0 and not bool(active.any()):
                break

        successes = active  # survived all horizon steps
        return CartPoleRolloutResult(returns=returns, lengths=lengths, successes=successes)

    def loss_for_stacked_params(
        self,
        stacked_params: Dict[str, torch.Tensor],
        seed: int,
        step: int,
    ) -> torch.Tensor:
        result = self.rollout_stacked_params(stacked_params, seed=seed, step=step)
        return -result.returns.mean(dim=1).to(dtype=torch.float32)

    def summarize_stacked_params(
        self,
        stacked_params: Dict[str, torch.Tensor],
        *,
        seed: int,
        step: int = 0,
    ) -> Dict[str, float]:
        result = self.rollout_stacked_params(stacked_params, seed=seed, step=step)
        return {
            "mean_return": float(result.returns.mean().item()),
            "success_rate": float(result.successes.float().mean().item()),
            "episode_length": float(result.lengths.mean().item()),
        }


def evaluate_policy_module(
    module: torch.nn.Module,
    *,
    seed: int,
    episodes: int = 100,
    horizon: int = DEFAULT_HORIZON,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a single ``DiscreteMLPPolicy`` module deterministically."""

    from .policies import stack_module_params

    evaluator = CartPoleEvaluator(
        rollouts_per_candidate=int(episodes), horizon=horizon, device=device,
    )
    return evaluator.summarize_stacked_params(
        stack_module_params(module, 1), seed=seed,
    )


def random_policy_baseline(
    *, seed: int, episodes: int = 100, horizon: int = DEFAULT_HORIZON,
    device: str = "cpu",
) -> Dict[str, float]:
    """Uniform-random action baseline for CartPole."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    states = sample_initial_states(int(episodes), seed=seed, device=device)
    returns = torch.zeros(int(episodes), device=device)
    lengths = torch.zeros(int(episodes), device=device)
    active = torch.ones(int(episodes), dtype=torch.bool, device=device)
    for _ in range(int(horizon)):
        actions = torch.randint(0, ACTION_DIM, (int(episodes),), generator=gen).to(device)
        next_states, rewards, done = cartpole_step(states, actions)
        returns = returns + torch.where(active, rewards, torch.zeros_like(rewards))
        lengths = lengths + active.float()
        states = torch.where(active.unsqueeze(-1), next_states, states)
        active = active & ~done
        if not bool(active.any()):
            break
    return {
        "mean_return": float(returns.mean().item()),
        "success_rate": float(active.float().mean().item()),
        "episode_length": float(lengths.mean().item()),
    }
