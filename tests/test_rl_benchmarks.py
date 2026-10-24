"""Smoke tests for RL benchmark helpers and runner."""

from __future__ import annotations


import pytest
import torch


def test_rl_metrics_include_save_result_keys():
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


def test_gym_evaluator_reuses_envs_per_size():
    pytest.importorskip("gymnasium")
    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator
    from polystep.benchmarks.rl.policies import DiscreteMLPPolicy, stack_module_params

    ev = GymVectorEvaluator("Acrobot-v1", rollouts_per_candidate=2, horizon=5)
    pol = DiscreteMLPPolicy(ev.obs_dim, 8, ev.action_dim)
    three = stack_module_params(pol, num_candidates=3, noise_scale=0.1, seed=0)
    a = ev.loss_for_stacked_params(three, seed=0, step=0)
    ev.summarize_stacked_params(stack_module_params(pol, 1), seed=0)
    venvs = dict(ev._venvs)
    b = ev.loss_for_stacked_params(three, seed=0, step=0)

    assert sorted(venvs) == [1, 3] and all(ev._venvs[n] is v for n, v in venvs.items())
    assert torch.equal(a, b)
    assert ev.env_steps == 2 * 3 * 2 * 5
    ev.close()
    assert not ev._venvs and all(v.closed for v in venvs.values())


def test_cartpole_evaluator_activation_and_training_steps():
    from polystep.benchmarks.rl.cartpole import CartPoleEvaluator
    from polystep.benchmarks.rl.policies import DiscreteMLPPolicy, stack_module_params

    sp = stack_module_params(DiscreteMLPPolicy(4, 8, 2), num_candidates=3, noise_scale=0.5, seed=0)
    ev = CartPoleEvaluator(rollouts_per_candidate=2, horizon=30, activation="binary")
    ev.summarize_stacked_params(sp, seed=0)
    assert ev.env_steps == 0
    ev.loss_for_stacked_params(sp, seed=0, step=1)
    assert ev.env_steps == int(ev.rollout_stacked_params(sp, seed=0, step=1).lengths.sum())

    with pytest.raises(ValueError, match="Unknown activation"):
        CartPoleEvaluator(rollouts_per_candidate=2, horizon=5, activation="relu").loss_for_stacked_params(
            sp, seed=0, step=0
        )


def test_es_cartpole_counts_training_steps(require_experiments, tmp_path):
    pytest.importorskip("gymnasium")
    import json

    from experiments.runners.run_rl import run_es_gym

    run_es_gym(
        env_short="cartpole",
        seed=0,
        generations=2,
        popsize=4,
        rollouts_per_candidate=2,
        horizon=50,
        results_dir=str(tmp_path),
    )
    (path,) = tmp_path.glob("*.json")
    record = json.loads(path.read_text())
    steps = [row["env_steps_cumulative"] for row in record["step_logs"]]
    assert steps[0] == 0 < steps[1] < steps[2] == record["metrics"]["rl_env_steps"] < 2 * 4 * 2 * 50


def test_cartpole_routes_to_cartpole_evaluator(require_experiments, monkeypatch, tmp_path):
    pytest.importorskip("gymnasium")
    import sys

    from experiments.runners import run_rl
    from polystep.benchmarks.rl.cartpole import CartPoleEvaluator

    activations = []
    loss = CartPoleEvaluator.loss_for_stacked_params

    def record(self, stacked_params, **kwargs):
        activations.append(self.activation)
        return loss(self, stacked_params, **kwargs)

    monkeypatch.setattr(CartPoleEvaluator, "loss_for_stacked_params", record)
    run_rl.run_polystep_gym(
        env_short="cartpole",
        seed=0,
        steps=1,
        rollouts_per_candidate=2,
        horizon=5,
        max_subspace_dim=4,
        results_dir=str(tmp_path / "polystep"),
        nondiff_mode="binary",
    )
    assert activations and set(activations) == {"binary"}

    calls = []
    monkeypatch.setattr(run_rl, "run_random_cartpole", lambda **kwargs: calls.append(kwargs))
    argv = ["run_rl", "--env", "cartpole", "--methods", "random_policy", "--nondiff-mode", "binary"]
    monkeypatch.setattr(sys, "argv", [*argv, "--seeds", "0", "--results-dir", str(tmp_path)])
    run_rl.main()
    assert calls == [{"seed": 0, "results_dir": str(tmp_path)}]


def test_es_keeps_theta_when_returns_tie(require_experiments, monkeypatch, tmp_path):
    pytest.importorskip("gymnasium")
    from experiments.runners.run_rl import run_es_gym
    from polystep.benchmarks.rl.cartpole import CartPoleEvaluator

    seen = []
    summarize = CartPoleEvaluator.summarize_stacked_params

    def record(self, stacked_params, **kwargs):
        seen.append({name: t.clone() for name, t in stacked_params.items()})
        return summarize(self, stacked_params, **kwargs)

    monkeypatch.setattr(CartPoleEvaluator, "summarize_stacked_params", record)
    run_es_gym(
        env_short="cartpole",
        seed=0,
        generations=2,
        popsize=4,
        rollouts_per_candidate=2,
        horizon=1,
        results_dir=str(tmp_path),
    )
    assert len(seen) > 1
    for params in seen[1:]:
        assert all(torch.equal(params[name], seen[0][name]) for name in params)


def test_es_climbs_toward_lower_loss(require_experiments, monkeypatch, tmp_path):
    pytest.importorskip("gymnasium")
    from experiments.runners.run_rl import run_es_gym
    from polystep.benchmarks.rl.cartpole import CartPoleEvaluator

    seen = []
    summarize = CartPoleEvaluator.summarize_stacked_params

    def record(self, stacked_params, **kwargs):
        seen.append(float(sum(t.sum() for t in stacked_params.values())))
        return summarize(self, stacked_params, **kwargs)

    def loss(self, stacked_params, **kwargs):
        return -sum(t.flatten(1).sum(1) for t in stacked_params.values())

    monkeypatch.setattr(CartPoleEvaluator, "summarize_stacked_params", record)
    monkeypatch.setattr(CartPoleEvaluator, "loss_for_stacked_params", loss)
    run_es_gym(
        env_short="cartpole",
        seed=0,
        generations=2,
        popsize=4,
        rollouts_per_candidate=2,
        horizon=5,
        results_dir=str(tmp_path),
    )
    assert seen[-1] > seen[0]


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
            f"mode={mode}: expected zero grad through non-diff op, got max|g|={first_w_grad.abs().max().item()}"
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

    ev_b = GymVectorEvaluator("Acrobot-v1", rollouts_per_candidate=2, horizon=20, activation="binary")
    losses_b = ev_b.loss_for_stacked_params(sp, seed=0, step=0)
    assert torch.isfinite(losses_b).all()
    ev_b.close()


def test_hardened_env_smoke(require_experiments):
    """Hardened wrappers register, reset, step, and produce quantized obs / bucketed reward."""
    pytest.importorskip("gymnasium")
    import gymnasium as gym
    import numpy as np
    from experiments.runners.hardened_env import (
        QuantizedObsWrapper,
        SparseRewardWrapper,
        make_hardened_env,
        register_hardened_envs,
        HARDENED_GYM_IDS,
    )

    # Direct quantizer: only 4 distinct values per channel (bin centers).
    env = QuantizedObsWrapper(
        gym.make("CartPole-v1"), bins=4, low=[-2.4, -3.0, -0.21, -3.5], high=[2.4, 3.0, 0.21, 3.5]
    )
    obs0, _ = env.reset(seed=0)
    obs1, _ = env.reset(seed=1)
    assert obs0.shape == (4,) and obs0.dtype == np.float32
    assert np.unique(np.concatenate([obs0, obs1])).size <= 8  # <= 4 bins x 2 resets
    env.close()

    # Reward bucketing: |r| < deadband zeros out.
    env = SparseRewardWrapper(gym.make("CartPole-v1"), bucket=10.0, deadband=2.0)
    env.reset(seed=0)
    _, r, _, _, _ = env.step(0)
    assert r == 0.0  # CartPole step reward is 1 < deadband 2 -> bucketed to 0
    env.close()

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


def test_acrobot_success_means_goal_reached():
    """Acrobot success must mean reaching the goal, not timing out at the horizon.

    The default flag is ``lengths >= horizon``, which is right for balancing
    tasks and exactly backwards for goal-reaching ones.
    """
    pytest.importorskip("gymnasium")
    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator

    acrobot = GymVectorEvaluator("Acrobot-v1", rollouts_per_candidate=2)
    lengths = torch.tensor([[81.0, float(acrobot.horizon)]])
    returns = torch.tensor([[-80.0, -float(acrobot.horizon)]])
    flags = acrobot.success_fn(returns, lengths)
    assert flags.tolist() == [[True, False]], flags

    # Balancing tasks keep the survived-to-horizon default.
    assert GymVectorEvaluator("CartPole-v1", rollouts_per_candidate=2).success_fn is None

    from polystep.benchmarks.rl.gym_evaluator import random_policy_baseline

    assert random_policy_baseline("Acrobot-v1", seed=0, episodes=4, horizon=5)["success_rate"] == 0.0
    assert random_policy_baseline("CartPole-v1", seed=0, episodes=4, horizon=5)["success_rate"] == 1.0
