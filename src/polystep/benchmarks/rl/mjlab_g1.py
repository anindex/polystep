"""Optional MJWarp/mjlab Unitree G1 locomotion adapter.

This module is intentionally importable without mjlab installed so the default
test suite and base package stay lightweight. Full G1 runs require the separate
MJWarp/mjlab installation path documented in the experiment runner.

Key design: candidate policies are evaluated **in parallel** via batched matrix
multiplication (``torch.bmm``), the same pattern as ``BatchedLinearEvaluator``
in ``cost_nn.py``. The mjlab environment is created with
``num_envs = N_candidates × envs_per_candidate`` and each candidate's policy is
applied to its own slice of environments simultaneously. This avoids the
sequential bottleneck of evaluating N candidates one at a time.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


MJLAB_TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"


class MjlabUnavailableError(RuntimeError):
    """Raised when the optional mjlab/MJWarp stack is not installed."""


def _require_mjlab():
    try:
        import mjlab  # type: ignore
    except ImportError as exc:
        raise MjlabUnavailableError(
            "mjlab is required for Unitree G1 locomotion runs. Install the optional "
            "rl-mjlab stack recommended by the mjlab/MJWarp documentation."
        ) from exc
    return mjlab


def check_mjlab_available() -> tuple[bool, str]:
    """Check whether the mjlab/MJWarp stack is importable.

    Returns ``(available, reason)`` where reason is an empty string on success.
    """

    import importlib
    import importlib.util

    if importlib.util.find_spec("mjlab") is None:
        return False, "mjlab is not installed; install the optional rl-mjlab stack to run MJWarp G1 sweeps."
    try:
        importlib.import_module("mjlab.tasks")
    except Exception as exc:
        return False, f"mjlab import failed during task registration: {type(exc).__name__}: {exc}"
    return True, ""


@dataclass
class MjlabBaselineCommand:
    """Subprocess command for in-ecosystem mjlab baselines."""

    argv: list[str]
    env: Optional[Dict[str, str]] = None


def _batched_mlp_forward(
    obs: torch.Tensor,
    stacked_params: dict[str, torch.Tensor],
    n_candidates: int,
    envs_per_candidate: int,
    activation: str = "elu",
) -> torch.Tensor:
    """Apply N different MLP policies to N slices of observations via bmm.

    This is the core parallel evaluation primitive. Instead of looping over
    candidates, all N forward passes are computed in a single set of batched
    matrix multiplications.

    Args:
        obs: Observations ``(total_envs, obs_dim)`` where
            ``total_envs = n_candidates × envs_per_candidate``.
        stacked_params: Parameter dict with shape ``(N, *param_shape)`` per key.
        n_candidates: Number of candidate policies N.
        envs_per_candidate: Environments per candidate.
        activation: Hidden activation (``'elu'`` or ``'tanh'``).

    Returns:
        Actions ``(total_envs, action_dim)``.
    """
    act_fn = {"elu": torch.nn.functional.elu, "tanh": torch.tanh}.get(
        activation.lower(), torch.nn.functional.elu
    )
    # Reshape obs: (N*E, obs_dim) → (N, E, obs_dim)
    x = obs.reshape(n_candidates, envs_per_candidate, -1)

    # Find linear layer indices by scanning for weight keys
    linear_indices = sorted(
        int(k.split(".")[1]) for k in stacked_params if k.endswith(".weight")
    )

    for j, idx in enumerate(linear_indices):
        w = stacked_params[f"net.{idx}.weight"]  # (N, out, in)
        b = stacked_params[f"net.{idx}.bias"]    # (N, out)
        # bmm: (N, E, in) @ (N, in, out) → (N, E, out)
        x = torch.bmm(x, w.transpose(1, 2))
        x = x + b.unsqueeze(1)  # broadcast (N, 1, out)
        # Apply activation: configurable for hidden, tanh for output (action clipping)
        if j < len(linear_indices) - 1:
            x = act_fn(x)
        else:
            x = torch.tanh(x)

    # Reshape back: (N, E, act_dim) → (N*E, act_dim)
    return x.reshape(n_candidates * envs_per_candidate, -1)


class MjlabG1Evaluator:
    """In-process mjlab G1 evaluator for PolyStep direct policy search.

    Creates a ``ManagerBasedRlEnv`` and evaluates candidate policies in
    parallel via batched matrix multiplication (``torch.bmm``). The environment
    is sized to ``max_candidates × envs_per_candidate`` so all candidates can
    be rolled out simultaneously in a single env.step() call.
    """

    env_id = MJLAB_TASK_ID
    action_type = "continuous"

    # Discovered via env introspection
    OBS_DIM = 99    # actor observation dimension
    ACTION_DIM = 29  # joint position action dimension

    def __init__(
        self,
        num_envs: int = 4096,
        horizon: int = 24,
        device: str = "cuda",
        max_candidates: int = 512,
        activation: str = "elu",
        eval_horizon: int | None = None,
    ):
        self.mjlab = _require_mjlab()
        self.envs_per_candidate = max(1, int(num_envs))
        self.horizon = int(horizon)
        # Headline-evaluation horizon (full episode by default). Decoupled from
        # training horizon so PolyStep can train on cheap short rollouts but be
        # *measured* under PPO's full-episode regime (mjlab episode_length_s=20s,
        # decimation=4, sim dt=0.005 ⇒ ~1000 control steps).
        self.eval_horizon = int(eval_horizon) if eval_horizon is not None else self.horizon
        self.device = torch.device(device)
        self.max_candidates = int(max_candidates)
        self.activation = activation
        self._env = None
        self._env_size = 0  # actual env size (may differ from requested)

    def _get_env(self, total_envs: int | None = None):
        """Lazily create or resize the mjlab environment.

        The environment is created once with the maximum needed size. If a
        larger size is requested later, the env is recreated.
        """

        needed = total_envs or self.envs_per_candidate
        if self._env is not None and self._env_size >= needed:
            return self._env

        if self._env is not None:
            self._env.close()
            self._env = None

        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        cfg = load_env_cfg(self.env_id)
        cfg.scene.num_envs = needed
        self._env = ManagerBasedRlEnv(cfg, device=str(self.device))
        self._env_size = needed
        return self._env

    def rollout_single_policy(
        self,
        policy_fn,
        *,
        seed: int,
        num_rollouts: int | None = None,
        horizon: int | None = None,
    ) -> Dict[str, float]:
        """Run rollouts for a single policy function and return summary metrics.

        Args:
            policy_fn: Callable that maps obs tensor ``(N, obs_dim)`` to actions ``(N, act_dim)``.
            seed: Random seed for initial state reset.
            num_rollouts: Number of parallel environments to use (defaults to envs_per_candidate).
            horizon: Override the rollout horizon (defaults to ``self.horizon``).
        """

        n = num_rollouts or self.envs_per_candidate
        env = self._get_env(n)
        H = int(horizon) if horizon is not None else self.horizon

        try:
            obs_dict, _ = env.reset(seed=int(seed))
        except TypeError:
            obs_dict, _ = env.reset()
        obs = obs_dict["actor"][:n]  # (n, 99)

        total_reward = torch.zeros(n, device=self.device)
        total_steps = torch.zeros(n, device=self.device)
        active = torch.ones(n, dtype=torch.bool, device=self.device)

        for _ in range(H):
            with torch.no_grad():
                actions = policy_fn(obs)
            obs_dict, rewards, terminated, truncated, info = env.step(
                # Pad actions to full env size if needed
                torch.cat([actions, torch.zeros(
                    self._env_size - n, self.ACTION_DIM, device=self.device
                )]) if self._env_size > n else actions
            )
            obs = obs_dict["actor"][:n]
            rewards = rewards[:n]
            terminated = terminated[:n]
            truncated = truncated[:n]

            total_reward += torch.where(active, rewards, torch.zeros_like(rewards))
            total_steps += active.float()

            done = terminated | truncated
            active = active & ~done
            if not active.any():
                break

        return {
            "mean_return": float(total_reward.mean().item()),
            "episode_length": float(total_steps.mean().item()),
            "fall_rate": float(terminated.float().mean().item()) if terminated is not None else 0.0,
        }

    def loss_for_stacked_params(
        self,
        stacked_params: dict[str, torch.Tensor],
        seed: int,
        step: int,
    ) -> torch.Tensor:
        """Evaluate stacked candidate policies via batched bmm and return negative mean rewards.

        All N candidates are evaluated **in parallel**: the environment has
        ``N × envs_per_candidate`` instances, and each candidate's MLP policy
        is applied to its slice via ``torch.bmm``. This is O(1) in N for the
        policy forward pass (GPU-parallel), with only the env.step() scaling
        linearly with total_envs (also GPU-parallel in MJWarp).

        Args:
            stacked_params: Dict of tensors with shape ``(N, *param_shape)``.
            seed: Random seed for rollout reproducibility.
            step: Current optimizer step (for logging/seed variation).

        Returns:
            Tensor of shape ``(N,)`` with negative mean return per candidate.
        """
        from .policies import count_stacked_candidates

        n_candidates = count_stacked_candidates(stacked_params)
        epc = self.envs_per_candidate
        total_envs = n_candidates * epc

        env = self._get_env(total_envs)

        # Reset all environments. Use the same seed across candidates within a
        # step (common random numbers) so per-candidate returns differ only due
        # to policy parameters, not initial state. ``step`` perturbs the seed
        # across optimizer iterations so we don't overfit to one init.
        try:
            obs_dict, _ = env.reset(seed=int(seed) + 7919 * int(step))
        except TypeError:
            obs_dict, _ = env.reset()
        obs = obs_dict["actor"][:total_envs]  # (N*epc, 99)

        # Track per-env rewards
        total_reward = torch.zeros(total_envs, device=self.device)
        active = torch.ones(total_envs, dtype=torch.bool, device=self.device)

        for _ in range(self.horizon):
            with torch.no_grad():
                # Batched forward: all N policies applied in parallel via bmm
                actions = _batched_mlp_forward(
                    obs, stacked_params, n_candidates, epc, self.activation,
                )

            # Pad actions if env is larger than total_envs
            if self._env_size > total_envs:
                actions = torch.cat([
                    actions,
                    torch.zeros(self._env_size - total_envs, self.ACTION_DIM, device=self.device),
                ])

            obs_dict, rewards, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["actor"][:total_envs]
            rewards = rewards[:total_envs]
            terminated_slice = terminated[:total_envs]
            truncated_slice = truncated[:total_envs]

            total_reward += torch.where(active, rewards, torch.zeros_like(rewards))
            done = terminated_slice | truncated_slice
            active = active & ~done

        # Reshape rewards: (N*epc,) → (N, epc) → mean over envs → (N,)
        per_candidate_return = total_reward.reshape(n_candidates, epc).mean(dim=1)
        return -per_candidate_return  # negative because PolyStep minimizes

    def summarize_stacked_params(
        self,
        stacked_params: dict[str, torch.Tensor],
        *,
        seed: int,
        step: int = 0,
        horizon: int | None = None,
    ) -> Dict[str, float]:
        """Evaluate a single policy (first candidate) and return summary metrics.

        By default uses ``self.eval_horizon`` (full episode) - this is the
        headline metric reported in step_logs and the final summary. Pass
        ``horizon=self.horizon`` to use the training-horizon view instead.
        """

        candidate_params = {k: v[0] for k, v in stacked_params.items()}
        H = int(horizon) if horizon is not None else self.eval_horizon
        act_fn = {"elu": torch.nn.functional.elu, "tanh": torch.tanh}.get(
            self.activation.lower(), torch.nn.functional.elu
        )

        def make_policy_fn(params):
            linear_indices = sorted(
                int(k.split(".")[1]) for k in params if k.endswith(".weight")
            )

            def policy_fn(obs):
                x = obs
                for j, idx in enumerate(linear_indices):
                    w = params[f"net.{idx}.weight"]
                    b = params[f"net.{idx}.bias"]
                    x = x @ w.T + b
                    if j < len(linear_indices) - 1:
                        x = act_fn(x)
                    else:
                        x = torch.tanh(x)  # output clipping
                return x
            return policy_fn

        policy_fn = make_policy_fn(candidate_params)
        return self.rollout_single_policy(
            policy_fn, seed=int(seed) + 7919 * int(step), horizon=H,
        )

    def close(self):
        """Close the underlying mjlab environment."""

        if self._env is not None:
            self._env.close()
            self._env = None


def build_rsl_rl_ppo_command(
    *,
    num_envs: int = 4096,
    max_iterations: int = 5000,
    seed: int = 42,
    experiment_name: str = "g1_velocity",
    run_name: str | None = None,
    logger: str = "tensorboard",
) -> MjlabBaselineCommand:
    """Build the mjlab RSL-RL PPO command for G1 velocity tracking.

    Defaults to ``--agent.logger tensorboard`` so per-iteration ``Train/mean_reward``
    can be parsed post-hoc into the polystep RL JSON schema.
    """

    argv = [
        "uv",
        "run",
        "train",
        MJLAB_TASK_ID,
        "--env.scene.num-envs",
        str(int(num_envs)),
        "--agent.max-iterations",
        str(int(max_iterations)),
        "--agent.seed",
        str(int(seed)),
        "--agent.logger",
        str(logger),
        "--agent.experiment-name",
        str(experiment_name),
    ]
    if run_name is not None:
        argv.extend(["--agent.run-name", str(run_name)])
    return MjlabBaselineCommand(
        argv=argv,
        env={"MUJOCO_GL": "egl"},
    )


def run_mjlab_command(command: MjlabBaselineCommand, *, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run an mjlab command wrapper and return the completed process."""

    import os

    env = os.environ.copy()
    if command.env:
        env.update(command.env)
    return subprocess.run(command.argv, cwd=cwd, env=env, check=False, capture_output=True, text=True)
