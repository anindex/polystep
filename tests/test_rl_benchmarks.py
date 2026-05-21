"""Smoke tests for RL benchmark helpers and runner."""

from __future__ import annotations

import json
import os

import pytest
import torch


def test_rl_metrics_include_save_result_required_keys():
    from polystep.benchmarks.rl.metrics import build_rl_metrics

    metrics = build_rl_metrics(
        final_return=1.0,
        best_return=2.0,
        normalized_score=0.25,
        wall_time_seconds=3.0,
        peak_gpu_memory_mb=0.0,
        function_evals=4,
        total_steps=5,
        rl_env_steps=6,
    )

    for key in [
        "final_accuracy",
        "best_accuracy",
        "wall_time_seconds",
        "peak_gpu_memory_mb",
        "function_evals",
        "total_steps",
        "final_return",
        "best_return",
        "normalized_score",
        "rl_env_steps",
    ]:
        assert key in metrics
    assert metrics["final_accuracy"] == pytest.approx(0.25)
    assert metrics["best_accuracy"] == pytest.approx(0.25)


@pytest.mark.parametrize("env_id", ["Acrobot-v1"])
def test_gym_vector_evaluator_basic(env_id):
    pytest.importorskip("gymnasium")
    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator
    from polystep.benchmarks.rl.policies import DiscreteMLPPolicy, stack_module_params

    ev = GymVectorEvaluator(env_id, rollouts_per_candidate=2, horizon=20)
    pol = DiscreteMLPPolicy(ev.obs_dim, 8, ev.action_dim)
    sp = stack_module_params(pol, num_candidates=3, noise_scale=0.1, seed=0)

    losses = ev.loss_for_stacked_params(sp, seed=0, step=0)
    assert losses.shape == (3,)
    assert torch.isfinite(losses).all()

    summary = ev.summarize_stacked_params(sp, seed=0, step=0)
    assert {"mean_return", "success_rate", "episode_length"} <= set(summary.keys())
    ev.close()


def test_gym_vector_evaluator_crn_determinism():
    pytest.importorskip("gymnasium")
    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator
    from polystep.benchmarks.rl.policies import DiscreteMLPPolicy, stack_module_params

    ev = GymVectorEvaluator("Acrobot-v1", rollouts_per_candidate=2, horizon=15)
    pol = DiscreteMLPPolicy(ev.obs_dim, 8, ev.action_dim)
    sp = stack_module_params(pol, num_candidates=2, noise_scale=0.0, seed=0)

    a = ev.loss_for_stacked_params(sp, seed=7, step=3)
    b = ev.loss_for_stacked_params(sp, seed=7, step=3)
    assert torch.allclose(a, b)
    ev.close()


def test_nondiff_policy_zero_grad():
    """NonDiffMLPPolicy must give zero gradient through the non-diff op (no STE)."""
    from polystep.benchmarks.rl.policies import NonDiffMLPPolicy

    for mode in ("int8", "binary"):
        pol = NonDiffMLPPolicy(obs_dim=4, hidden=8, action_dim=2, mode=mode)
        x = torch.randn(3, 4)
        logits = pol(x)
        loss = logits.sum()
        loss.backward()
        # First Linear sits below the non-diff op -> must have zero (or None) grad.
        first_w_grad = pol.net[0].weight.grad
        assert first_w_grad is not None
        assert torch.allclose(first_w_grad, torch.zeros_like(first_w_grad)), (
            f"mode={mode}: expected zero grad through non-diff op, got "
            f"max|g|={first_w_grad.abs().max().item()}"
        )


def test_gym_evaluator_nondiff_acrobot():
    """Binary-activation evaluator returns finite losses; logits differ from tanh."""
    pytest.importorskip("gymnasium")
    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator, _batched_mlp_logits
    from polystep.benchmarks.rl.policies import DiscreteMLPPolicy, stack_module_params

    pol = DiscreteMLPPolicy(6, 8, 3)
    sp = stack_module_params(pol, num_candidates=2, noise_scale=0.5, seed=0)

    # Direct logit comparison - activation choice must change forward outputs.
    obs = torch.randn(2, 4, 6)  # (N=2, R=4, obs_dim=6)
    logits_t = _batched_mlp_logits(obs, sp, activation="tanh")
    logits_b = _batched_mlp_logits(obs, sp, activation="binary")
    assert torch.isfinite(logits_t).all() and torch.isfinite(logits_b).all()
    assert not torch.allclose(logits_t, logits_b), "binary activation must yield different logits than tanh"

    # End-to-end: binary evaluator returns finite losses.
    ev_b = GymVectorEvaluator("Acrobot-v1", rollouts_per_candidate=2, horizon=20, activation="binary")
    losses_b = ev_b.loss_for_stacked_params(sp, seed=0, step=0)
    assert torch.isfinite(losses_b).all()
    ev_b.close()


@pytest.mark.parametrize("method", ["ppo", "dqn"])
def test_sb3_nondiff_zero_grad(method):
    """SB3 PPO/DQN with the new non-diff harness must produce zero gradient
    on every trainable parameter (full-backbone collapse, not just feature-extractor)."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    import gymnasium as gym
    from stable_baselines3 import DQN, PPO
    from experiments.runners.run_rl import (
        _apply_nondiff_to_sb3_policy,
        _make_nondiff_activation_fn,
    )

    algo_cls = {"ppo": PPO, "dqn": DQN}[method]
    env = gym.make("CartPole-v1")
    env.reset(seed=0)
    policy_kwargs = {
        "net_arch": [16, 16],
        "activation_fn": _make_nondiff_activation_fn("binary"),
    }
    model = algo_cls("MlpPolicy", env, seed=0, verbose=0, policy_kwargs=policy_kwargs, device="cpu")
    _apply_nondiff_to_sb3_policy(model, method, "binary")

    # Forward + a synthetic loss + backward.
    obs, _ = env.reset(seed=0)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    if method == "ppo":
        # forward returns (actions, values, log_prob)
        _, values, log_prob = model.policy(obs_t)
        loss = (values.sum() + log_prob.sum())
    else:
        q = model.q_net(obs_t)
        loss = q.sum()
    model.policy.zero_grad(set_to_none=True)
    loss.backward()

    # Every trainable parameter must have either no grad or all-zero grad.
    nonzero = []
    for name, p in model.policy.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        max_abs = float(p.grad.abs().max().item())
        if max_abs > 0.0:
            nonzero.append((name, max_abs))
    env.close()
    assert not nonzero, f"{method} non-diff harness leaks grad on: {nonzero[:5]}"


def test_hardened_env_smoke():
    """Hardened wrappers register, reset, step, and produce quantized obs / bucketed reward."""
    pytest.importorskip("gymnasium")
    import gymnasium as gym
    import numpy as np
    from experiments.runners.hardened_env import (
        QuantizedObsWrapper, SparseRewardWrapper, make_hardened_env,
        register_hardened_envs, HARDENED_GYM_IDS,
    )

    # Direct quantizer: only 4 distinct values per channel (bin centres).
    env = QuantizedObsWrapper(gym.make("CartPole-v1"), bins=4,
                              low=[-2.4, -3.0, -0.21, -3.5],
                              high=[2.4, 3.0, 0.21, 3.5])
    obs0, _ = env.reset(seed=0)
    obs1, _ = env.reset(seed=1)
    # Quantizer outputs should be one of 4 bin centres per channel.
    assert obs0.shape == (4,) and obs0.dtype == np.float32
    assert np.unique(np.concatenate([obs0, obs1])).size <= 8  # ≤ 4 bins × 2 resets
    env.close()

    # Reward bucketing: |r| < deadband zeros out.
    env = SparseRewardWrapper(gym.make("CartPole-v1"), bucket=10.0, deadband=2.0)
    env.reset(seed=0)
    _, r, _, _, _ = env.step(0)
    assert r == 0.0  # CartPole step reward is 1 < deadband 2 -> bucketed to 0
    env.close()

    # End-to-end factory + Gym registration
    for s in ("cartpole_hard", "acrobot_hard"):
        e = make_hardened_env(s)
        obs, _ = e.reset(seed=0)
        assert np.isfinite(obs).all()
        e.close()
    register_hardened_envs()
    for gid in HARDENED_GYM_IDS.values():
        e = gym.make(gid)
        e.reset(seed=0)
        e.close()
