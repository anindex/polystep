"""Generic vectorized Gymnasium evaluator for stacked MLP policies.

Wraps any discrete-action Gymnasium environment via ``SyncVectorEnv`` and
evaluates ``N`` stacked candidate policies in parallel by running ``N * R``
parallel envs and dispatching each env to its assigned candidate. The API
mirrors :class:`polystep.benchmarks.rl.cartpole.CartPoleEvaluator` so it can be
dropped into the same training loop in ``experiments/runners/run_rl.py``.

Notes
-----
- Works with any Gymnasium ``Discrete``-action environment whose observation
  space is a ``Box`` (or anything yielding a fixed-size ``ndarray``).
- For analytic, fully GPU-vectorizable envs (e.g. CartPole), prefer the
  task-specific evaluator; this class is the right choice when you need
  Gymnasium's exact dynamics or there is no analytic form.
- Uses CRN (common random numbers): every candidate sees the same per-(step,
  rollout) seed in a given outer iteration, so evaluations within a step
  differ only by the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .policies import count_stacked_candidates


@dataclass
class GymRolloutResult:
    returns: torch.Tensor   # (N, R)
    lengths: torch.Tensor   # (N, R)
    successes: torch.Tensor  # (N, R) bool - "success" flag, env-specific


def _make_vector_env(env_id: str, total_envs: int):
    """Create a SyncVectorEnv with ``total_envs`` instances of ``env_id``."""

    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv

    def thunk():
        return gym.make(env_id)

    return SyncVectorEnv([thunk for _ in range(int(total_envs))])


def _batched_mlp_logits(
    obs: torch.Tensor,
    stacked_params: Dict[str, torch.Tensor],
    activation: str = "tanh",
) -> torch.Tensor:
    """Apply N stacked policies to obs of shape (N, R, obs_dim).

    ``activation`` selects the inner activation between linear layers:

    - ``"tanh"`` - standard :class:`DiscreteMLPPolicy`.
    - ``"int8"`` - per-tensor symmetric INT8 quantize/dequantize (no STE).
    - ``"binary"`` - ``sign(x)`` activation (no STE).
    - ``"float32"`` - identity.
    """

    x = obs  # (N, R, obs_dim)
    linear_indices = sorted(
        int(k.split(".")[1]) for k in stacked_params if k.endswith(".weight")
    )
    for j, idx in enumerate(linear_indices):
        w = stacked_params[f"net.{idx}.weight"]  # (N, out, in)
        b = stacked_params[f"net.{idx}.bias"]    # (N, out)
        x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
        if j < len(linear_indices) - 1:
            if activation == "tanh":
                x = torch.tanh(x)
            elif activation == "int8":
                # Per-candidate per-tensor quantization to mirror NonDiffActivation.
                max_abs = x.detach().abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
                scale = max_abs / 127.0
                x = torch.round(x / scale) * scale
            elif activation == "binary":
                x = torch.sign(x)
            elif activation == "float32":
                pass
            else:
                raise ValueError(f"Unknown activation {activation!r}")
    return x


class GymVectorEvaluator:
    """Vectorized evaluator for stacked discrete-action MLP policies on any Gymnasium env.

    Parameters
    ----------
    env_id:
        Gymnasium env identifier (e.g. ``"Acrobot-v1"``, ``"LunarLander-v3"``).
    rollouts_per_candidate:
        Number of parallel episodes per candidate (``R``).
    horizon:
        Maximum steps per episode. Defaults to the env's ``spec.max_episode_steps``.
    device:
        Device for policy forward passes. Env stays on CPU.
    success_fn:
        Optional callable ``(returns: Tensor (N,R), lengths: Tensor (N,R)) ->
        Tensor[bool] (N,R)`` mapping per-episode return and length to a
        "success" flag for logging. Defaults to "survived to ``horizon``"
        (i.e., ``lengths == horizon``), which is correct for fixed-horizon
        balancing tasks like CartPole. For environments where "success"
        means reaching a goal in less than ``horizon`` steps (e.g.,
        Acrobot, MountainCar), pass a custom ``success_fn``.
    activation:
        Inner activation between linear layers. ``"tanh"`` (default) for the
        standard :class:`DiscreteMLPPolicy`; ``"int8"`` / ``"binary"`` /
        ``"float32"`` for the non-differentiable variants used to motivate
        gradient-free training (see :class:`NonDiffMLPPolicy`).
    """

    def __init__(
        self,
        env_id: str,
        *,
        rollouts_per_candidate: int = 16,
        horizon: Optional[int] = None,
        device: str = "cpu",
        success_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        activation: str = "tanh",
    ):
        import gymnasium as gym

        self.env_id = str(env_id)
        self.rollouts_per_candidate = int(rollouts_per_candidate)
        self.device = torch.device(device)
        # Probe env for spec metadata.
        probe = gym.make(self.env_id)
        self.obs_dim = int(np.prod(probe.observation_space.shape))
        if probe.action_space.__class__.__name__ != "Discrete":
            probe.close()
            raise ValueError(
                f"GymVectorEvaluator requires Discrete action space; got "
                f"{probe.action_space} for {env_id}"
            )
        self.action_dim = int(probe.action_space.n)
        spec_horizon = getattr(probe.spec, "max_episode_steps", None) or 500
        probe.close()
        self.horizon = int(horizon) if horizon is not None else int(spec_horizon)
        self.action_type = "discrete"
        self.success_fn = success_fn
        self.activation = str(activation)
        # Cache vector env across calls; lazily (re)created when n_candidates changes.
        self._cached_n_candidates: Optional[int] = None
        self._venv = None

    def _ensure_venv(self, n_candidates: int) -> None:
        n_total = int(n_candidates) * self.rollouts_per_candidate
        if self._venv is None or self._cached_n_candidates != n_candidates:
            if self._venv is not None:
                try:
                    self._venv.close()
                except Exception:  # noqa: BLE001
                    pass
            self._venv = _make_vector_env(self.env_id, n_total)
            self._cached_n_candidates = int(n_candidates)

    def close(self) -> None:
        if self._venv is not None:
            try:
                self._venv.close()
            except Exception:  # noqa: BLE001
                pass
            self._venv = None

    def __del__(self):  # best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Core rollout
    # ------------------------------------------------------------------
    def rollout_stacked_params(
        self,
        stacked_params: Dict[str, torch.Tensor],
        *,
        seed: int,
        step: int,
    ) -> GymRolloutResult:
        n_candidates = count_stacked_candidates(stacked_params)
        R = self.rollouts_per_candidate
        N = n_candidates
        self._ensure_venv(N)

        # Per-env seed: deterministic per (step, env_index) so candidates share CRN
        # within a step and seeds rotate across steps.
        base_seed = int(seed) + 1009 * int(step)
        seeds = [base_seed + i for i in range(N * R)]
        obs_np, _ = self._venv.reset(seed=seeds)

        # State tensors (on CPU then move to device per step for policy inference).
        returns = torch.zeros(N, R, dtype=torch.float32)
        lengths = torch.zeros(N, R, dtype=torch.float32)
        active = np.ones(N * R, dtype=bool)

        sp = {k: v.to(self.device) for k, v in stacked_params.items()}

        for _t in range(self.horizon):
            obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(self.device)
            obs_t = obs_t.view(N, R, self.obs_dim)
            with torch.no_grad():
                logits = _batched_mlp_logits(obs_t, sp, activation=self.activation)  # (N, R, action_dim)
                actions = logits.argmax(dim=-1)  # (N, R)
            actions_np = actions.cpu().numpy().reshape(N * R).astype(np.int64)
            obs_np, rewards_np, terminated_np, truncated_np, _ = self._venv.step(actions_np)
            done_np = np.logical_or(terminated_np, truncated_np)

            r_active = active.astype(np.float32)
            rew_active = rewards_np.astype(np.float32) * r_active
            returns += torch.from_numpy(rew_active.reshape(N, R))
            lengths += torch.from_numpy(r_active.reshape(N, R))

            # Mark envs that just terminated as inactive going forward; further
            # rewards from auto-reset are masked out by the `active` flag.
            active = active & ~done_np
            if not active.any():
                break

        # Successes: env-specific. ``success_fn`` lets the caller supply
        # env-aware logic; the default "survived to horizon" is appropriate
        # for fixed-horizon balancing tasks (CartPole) and clearly wrong for
        # goal-reaching tasks (Acrobot), which is why custom ``success_fn``
        # is encouraged.
        if self.success_fn is not None:
            successes = self.success_fn(returns, lengths)
        else:
            successes = lengths >= float(self.horizon)

        return GymRolloutResult(
            returns=returns.to(self.device),
            lengths=lengths.to(self.device),
            successes=successes.to(self.device),
        )

    # ------------------------------------------------------------------
    # Loss + summary helpers (matches CartPoleEvaluator interface)
    # ------------------------------------------------------------------
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


def random_policy_baseline(
    env_id: str,
    *,
    seed: int,
    episodes: int = 100,
    horizon: Optional[int] = None,
) -> Dict[str, float]:
    """Uniform-random action baseline for any Gymnasium discrete-action env."""

    import gymnasium as gym

    rng = np.random.default_rng(int(seed))
    venv = _make_vector_env(env_id, int(episodes))
    obs_np, _ = venv.reset(seed=[int(seed) + i for i in range(int(episodes))])
    if horizon is None:
        probe = gym.make(env_id)
        horizon = getattr(probe.spec, "max_episode_steps", None) or 500
        probe.close()
    n_actions = venv.single_action_space.n
    returns = np.zeros(int(episodes), dtype=np.float64)
    lengths = np.zeros(int(episodes), dtype=np.float64)
    active = np.ones(int(episodes), dtype=bool)
    for _ in range(int(horizon)):
        actions = rng.integers(0, n_actions, size=int(episodes))
        obs_np, rewards_np, terminated_np, truncated_np, _ = venv.step(actions)
        done_np = np.logical_or(terminated_np, truncated_np)
        returns += rewards_np * active
        lengths += active.astype(np.float64)
        active = active & ~done_np
        if not active.any():
            break
    venv.close()
    return {
        "mean_return": float(returns.mean()),
        "success_rate": float((~active).mean()),
        "episode_length": float(lengths.mean()),
    }
