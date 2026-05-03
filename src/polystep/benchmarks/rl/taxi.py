"""Vectorized Taxi-v3 evaluator and lightweight tabular baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .policies import count_stacked_candidates, taxi_logits_from_stacked_params


NUM_STATES = 500
NUM_ACTIONS = 6
TAXI_LOCATIONS = torch.tensor([[0, 0], [0, 4], [4, 0], [4, 3]], dtype=torch.long)
TAXI_DESC = (
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
)


def encode_taxi_state(row: torch.Tensor, col: torch.Tensor, passenger: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    """Encode Taxi-v3 state components into integer state ids."""

    return (((row * 5 + col) * 5 + passenger) * 4 + dest).long()


def decode_taxi_state(states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode integer state ids into row, col, passenger, destination tensors."""

    dest = states % 4
    passenger = (states // 4) % 5
    col = (states // 20) % 5
    row = states // 100
    return row.long(), col.long(), passenger.long(), dest.long()


def _movement_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    east = torch.zeros(5, 5, dtype=torch.bool, device=device)
    west = torch.zeros(5, 5, dtype=torch.bool, device=device)
    for row in range(5):
        for col in range(5):
            east[row, col] = col < 4 and TAXI_DESC[1 + row][2 * col + 2] == ":"
            west[row, col] = col > 0 and TAXI_DESC[1 + row][2 * col] == ":"
    return east, west


def sample_initial_states(num: int, *, seed: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample deterministic Taxi initial states with passenger != destination."""

    device = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    row = torch.randint(0, 5, (num,), generator=gen)
    col = torch.randint(0, 5, (num,), generator=gen)
    passenger = torch.randint(0, 4, (num,), generator=gen)
    dest = torch.randint(0, 4, (num,), generator=gen)
    same = dest == passenger
    dest = torch.where(same, (dest + 1) % 4, dest)
    return encode_taxi_state(row, col, passenger, dest).to(device)


def taxi_step(states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized Taxi transition.

    Returns ``(next_state, reward, done, illegal)``.
    """

    device = states.device
    row, col, passenger, dest = decode_taxi_state(states)
    next_row = row.clone()
    next_col = col.clone()
    next_passenger = passenger.clone()
    reward = torch.full(states.shape, -1.0, dtype=torch.float32, device=device)
    done = torch.zeros(states.shape, dtype=torch.bool, device=device)
    east_open, west_open = _movement_masks(device)

    south = actions == 0
    north = actions == 1
    east = actions == 2
    west = actions == 3
    pickup = actions == 4
    dropoff = actions == 5

    next_row = torch.where(south & (row < 4), row + 1, next_row)
    next_row = torch.where(north & (row > 0), row - 1, next_row)
    next_col = torch.where(east & east_open[row, col], col + 1, next_col)
    next_col = torch.where(west & west_open[row, col], col - 1, next_col)

    loc_rows = TAXI_LOCATIONS[:, 0].to(device)
    loc_cols = TAXI_LOCATIONS[:, 1].to(device)
    at_passenger = passenger < 4
    passenger_row = loc_rows[passenger.clamp_max(3)]
    passenger_col = loc_cols[passenger.clamp_max(3)]
    can_pickup = pickup & at_passenger & (row == passenger_row) & (col == passenger_col)
    bad_pickup = pickup & ~can_pickup
    next_passenger = torch.where(can_pickup, torch.full_like(next_passenger, 4), next_passenger)

    dest_row = loc_rows[dest]
    dest_col = loc_cols[dest]
    can_dropoff = dropoff & (passenger == 4) & (row == dest_row) & (col == dest_col)
    bad_dropoff = dropoff & ~can_dropoff
    done = can_dropoff
    next_passenger = torch.where(can_dropoff, dest, next_passenger)
    reward = torch.where(can_dropoff, torch.full_like(reward, 20.0), reward)

    illegal = bad_pickup | bad_dropoff
    reward = torch.where(illegal, torch.full_like(reward, -10.0), reward)
    return encode_taxi_state(next_row, next_col, next_passenger, dest), reward, done, illegal


@dataclass
class TaxiRolloutResult:
    returns: torch.Tensor
    lengths: torch.Tensor
    successes: torch.Tensor
    illegal_actions: torch.Tensor


class TaxiEvaluator:
    """Vectorized evaluator for stacked Taxi policy parameters."""

    env_id = "Taxi-v3"
    obs_dim = NUM_STATES
    action_dim = NUM_ACTIONS
    action_type = "discrete"

    def __init__(self, rollouts_per_candidate: int = 128, horizon: int = 200, device: str = "cpu"):
        self.rollouts_per_candidate = int(rollouts_per_candidate)
        self.horizon = int(horizon)
        self.device = torch.device(device)

    def rollout_stacked_params(self, stacked_params: Dict[str, torch.Tensor], *, seed: int, step: int) -> TaxiRolloutResult:
        n_candidates = count_stacked_candidates(stacked_params)
        params = {k: v.to(self.device) for k, v in stacked_params.items()}
        states = sample_initial_states(
            n_candidates * self.rollouts_per_candidate,
            seed=int(seed) + 1009 * int(step),
            device=self.device,
        ).view(n_candidates, self.rollouts_per_candidate)
        returns = torch.zeros_like(states, dtype=torch.float32)
        lengths = torch.zeros_like(states, dtype=torch.float32)
        successes = torch.zeros_like(states, dtype=torch.bool)
        illegal_count = torch.zeros_like(states, dtype=torch.float32)
        active = torch.ones_like(states, dtype=torch.bool)

        for _ in range(self.horizon):
            logits = taxi_logits_from_stacked_params(params, states)
            actions = logits.argmax(dim=-1)
            next_states, rewards, done, illegal = taxi_step(states, actions)
            returns = returns + torch.where(active, rewards, torch.zeros_like(rewards))
            lengths = lengths + active.float()
            illegal_count = illegal_count + (active & illegal).float()
            successes = successes | (active & done)
            states = torch.where(active, next_states, states)
            active = active & ~done
            if not bool(active.any()):
                break

        return TaxiRolloutResult(
            returns=returns,
            lengths=lengths,
            successes=successes,
            illegal_actions=illegal_count,
        )

    def loss_for_stacked_params(self, stacked_params: Dict[str, torch.Tensor], seed: int, step: int) -> torch.Tensor:
        result = self.rollout_stacked_params(stacked_params, seed=seed, step=step)
        return -result.returns.mean(dim=1).to(dtype=torch.float32)

    def summarize_stacked_params(self, stacked_params: Dict[str, torch.Tensor], *, seed: int, step: int = 0) -> Dict[str, float]:
        result = self.rollout_stacked_params(stacked_params, seed=seed, step=step)
        return {
            "mean_return": float(result.returns.mean().item()),
            "success_rate": float(result.successes.float().mean().item()),
            "episode_length": float(result.lengths.mean().item()),
            "illegal_action_rate": float((result.illegal_actions / result.lengths.clamp_min(1)).mean().item()),
        }


class TabularQModule(torch.nn.Module):
    """Thin nn.Module wrapper around a 500×6 Q-table for PolyStep compatibility.

    PolyStep requires an ``nn.Module`` to extract ``ParamLayout``. This module
    holds a single ``(500, 6)`` parameter tensor that represents Q(s, a) values.
    The greedy policy is ``argmax_a Q(s, a)``.
    """

    def __init__(self, num_states: int = NUM_STATES, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.q_table = torch.nn.Parameter(torch.zeros(num_states, num_actions))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values for given state indices."""
        return self.q_table[states.long()]


class TabularTaxiEvaluator:
    """Vectorized evaluator for stacked tabular Q-table parameters.

    Unlike ``TaxiEvaluator`` which expects MLP params, this evaluator
    works with a single ``q_table`` parameter of shape ``(N, 500, 6)``
    (stacked across candidates). The greedy policy is ``argmax Q[s, :]``.
    """

    env_id = "Taxi-v3"
    obs_dim = NUM_STATES
    action_dim = NUM_ACTIONS
    action_type = "discrete"

    def __init__(self, rollouts_per_candidate: int = 128, horizon: int = 200, device: str = "cpu"):
        self.rollouts_per_candidate = int(rollouts_per_candidate)
        self.horizon = int(horizon)
        self.device = torch.device(device)

    def rollout_stacked_params(
        self,
        stacked_params: Dict[str, torch.Tensor],
        *,
        seed: int,
        step: int,
    ) -> TaxiRolloutResult:
        """Rollout greedy Q-table policies for N candidates in parallel."""

        q_tables = stacked_params["q_table"].to(self.device)  # (N, 500, 6)
        n_candidates = q_tables.shape[0]
        R = self.rollouts_per_candidate

        states = sample_initial_states(
            n_candidates * R, seed=int(seed) + 1009 * int(step), device=self.device,
        ).view(n_candidates, R)

        returns = torch.zeros(n_candidates, R, dtype=torch.float32, device=self.device)
        lengths = torch.zeros(n_candidates, R, dtype=torch.float32, device=self.device)
        successes = torch.zeros(n_candidates, R, dtype=torch.bool, device=self.device)
        illegal_count = torch.zeros(n_candidates, R, dtype=torch.float32, device=self.device)
        active = torch.ones(n_candidates, R, dtype=torch.bool, device=self.device)

        for _ in range(self.horizon):
            # Q-table lookup: q_tables[n, states[n, r], :] → (N, R, 6)
            # Use advanced indexing:
            q_vals = q_tables[
                torch.arange(n_candidates, device=self.device).unsqueeze(1).expand_as(states),
                states.long(),
            ]  # (N, R, 6)
            actions = q_vals.argmax(dim=-1)  # (N, R)

            # Flatten for taxi_step, then reshape back
            flat_states = states.view(-1)
            flat_actions = actions.view(-1)
            next_flat, rewards_flat, done_flat, illegal_flat = taxi_step(flat_states, flat_actions)
            next_states = next_flat.view(n_candidates, R)
            rewards = rewards_flat.view(n_candidates, R)
            done = done_flat.view(n_candidates, R)
            illegal = illegal_flat.view(n_candidates, R)

            returns += torch.where(active, rewards, torch.zeros_like(rewards))
            lengths += active.float()
            illegal_count += (active & illegal).float()
            successes = successes | (active & done)
            states = torch.where(active, next_states, states)
            active = active & ~done
            if not bool(active.any()):
                break

        return TaxiRolloutResult(
            returns=returns, lengths=lengths,
            successes=successes, illegal_actions=illegal_count,
        )

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
            "illegal_action_rate": float((result.illegal_actions / result.lengths.clamp_min(1)).mean().item()),
        }


def train_q_learning_taxi(
    *,
    seed: int,
    episodes: int = 50_000,
    horizon: int = 200,
    learning_rate: float = 0.7,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
) -> tuple[torch.Tensor, int]:
    """Train a small tabular Q-learning baseline for Taxi."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    q_values = torch.zeros(NUM_STATES, NUM_ACTIONS, dtype=torch.float32)
    env_steps = 0

    for episode in range(int(episodes)):
        state_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
        state = sample_initial_states(1, seed=state_seed)[0]
        eps_frac = episode / max(1, episodes - 1)
        epsilon = epsilon_start + eps_frac * (epsilon_final - epsilon_start)
        for _ in range(int(horizon)):
            if float(torch.rand((), generator=gen).item()) < epsilon:
                action = int(torch.randint(0, NUM_ACTIONS, (1,), generator=gen).item())
            else:
                action = int(q_values[state].argmax().item())
            next_state, reward, done, _ = taxi_step(state.view(1), torch.tensor([action]))
            target = reward[0] + gamma * q_values[next_state[0]].max() * (~done[0]).float()
            q_values[state, action] = q_values[state, action] + learning_rate * (target - q_values[state, action])
            state = next_state[0]
            env_steps += 1
            if bool(done[0]):
                break

    return q_values, env_steps


def evaluate_q_table(
    q_values: torch.Tensor,
    *,
    seed: int,
    episodes: int = 100,
    horizon: int = 200,
) -> Dict[str, float]:
    """Evaluate a greedy Taxi Q-table."""

    states = sample_initial_states(int(episodes), seed=seed)
    returns = torch.zeros(int(episodes), dtype=torch.float32)
    lengths = torch.zeros(int(episodes), dtype=torch.float32)
    successes = torch.zeros(int(episodes), dtype=torch.bool)
    illegal = torch.zeros(int(episodes), dtype=torch.float32)
    active = torch.ones(int(episodes), dtype=torch.bool)
    for _ in range(int(horizon)):
        actions = q_values[states].argmax(dim=-1)
        next_states, rewards, done, illegal_step = taxi_step(states, actions)
        returns = returns + torch.where(active, rewards, torch.zeros_like(rewards))
        lengths = lengths + active.float()
        successes = successes | (active & done)
        illegal = illegal + (active & illegal_step).float()
        states = torch.where(active, next_states, states)
        active = active & ~done
        if not bool(active.any()):
            break
    return {
        "mean_return": float(returns.mean().item()),
        "success_rate": float(successes.float().mean().item()),
        "episode_length": float(lengths.mean().item()),
        "illegal_action_rate": float((illegal / lengths.clamp_min(1)).mean().item()),
    }
