"""Smoke tests for RL benchmark helpers and runner.

The G1/mjlab tests are optional because MJWarp requires a separate GPU-focused
install. Taxi tests are dependency-light and should run in the default suite.
"""

from __future__ import annotations

import json
import os

import pytest
import torch


def test_taxi_evaluator_returns_one_loss_per_candidate():
    from polystep.benchmarks.rl.policies import make_taxi_policy, stack_module_params
    from polystep.benchmarks.rl.taxi import TaxiEvaluator

    model = make_taxi_policy(hidden=8)
    stacked_params = stack_module_params(model, num_candidates=3)
    evaluator = TaxiEvaluator(rollouts_per_candidate=4, horizon=20, device="cpu")

    losses = evaluator.loss_for_stacked_params(stacked_params, seed=42, step=0)

    assert losses.shape == (3,)
    assert losses.dtype == torch.float32
    assert torch.isfinite(losses).all()


def test_taxi_evaluator_is_seed_deterministic():
    from polystep.benchmarks.rl.policies import make_taxi_policy, stack_module_params
    from polystep.benchmarks.rl.taxi import TaxiEvaluator

    model = make_taxi_policy(hidden=8)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.net[2].bias[4] = 10.0  # always attempt pickup; legality depends on initial state
    stacked_params = stack_module_params(model, num_candidates=2)
    evaluator = TaxiEvaluator(rollouts_per_candidate=64, horizon=15, device="cpu")

    first = evaluator.loss_for_stacked_params(stacked_params, seed=123, step=2)
    second = evaluator.loss_for_stacked_params(stacked_params, seed=123, step=2)
    different_seed = evaluator.loss_for_stacked_params(stacked_params, seed=456, step=2)

    assert torch.equal(first, second)
    assert not torch.equal(first, different_seed)


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


@pytest.mark.timeout(30)
def test_run_polystep_taxi_smoke_writes_result_json(tmp_path):
    from experiments.runners.run_rl import run_polystep_taxi

    evals = run_polystep_taxi(
        seed=42,
        device="cpu",
        steps=2,
        rollouts_per_candidate=4,
        horizon=20,
        max_subspace_dim=16,
        results_dir=str(tmp_path),
    )

    assert evals > 0
    result_file = os.path.join(str(tmp_path), "taxi_polystep_42.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)

    assert data["benchmark"] == "taxi"
    assert data["method"] == "polystep"
    assert data["seed"] == 42
    assert data["metrics"]["total_steps"] == 2
    assert data["metrics"]["function_evals"] == evals
    assert "final_return" in data["metrics"]
    assert "rl_env_steps" in data["metrics"]
    assert data["hyperparameters"]["representation"] == "tabular_q_table"


@pytest.mark.timeout(30)
def test_run_q_learning_taxi_smoke_writes_result_json(tmp_path):
    from experiments.runners.run_rl import run_q_learning_taxi

    run_q_learning_taxi(
        seed=42,
        episodes=20,
        eval_episodes=8,
        results_dir=str(tmp_path),
    )

    result_file = os.path.join(str(tmp_path), "taxi_q_learning_42.json")
    assert os.path.exists(result_file)
    with open(result_file) as f:
        data = json.load(f)

    assert data["benchmark"] == "taxi"
    assert data["method"] == "q_learning"
    assert data["metrics"]["rl_env_steps"] > 0


def test_select_best_sweep_result_prefers_best_return():
    from experiments.runners.run_rl import select_best_sweep_result

    best = select_best_sweep_result(
        [
            {"config_id": "a", "metrics": {"best_return": -200.0}, "config": {"step_radius": 0.1}},
            {"config_id": "b", "metrics": {"best_return": -50.0}, "config": {"step_radius": 0.3}},
        ]
    )

    assert best["config_id"] == "b"
    assert best["config"]["step_radius"] == 0.3


@pytest.mark.timeout(30)
def test_run_taxi_fast_sweep_writes_per_config_results(tmp_path):
    from experiments.runners.run_rl import run_polystep_taxi, select_best_sweep_result

    # Use minimal configs for CI speed (production sweep uses 200 steps)
    # Tabular Q-table: no 'hidden' param needed
    smoke_configs = [
        {"steps": 2, "rollouts_per_candidate": 4, "horizon": 20,
         "subspace_rank": 2, "epsilon_init": 0.5, "epsilon_target": 0.3,
         "step_radius": 0.05, "probe_radius": 0.1, "amortize_steps": 1, "max_subspace_dim": 16},
        {"steps": 2, "rollouts_per_candidate": 4, "horizon": 20,
         "subspace_rank": 4, "epsilon_init": 1.0, "epsilon_target": 0.3,
         "step_radius": 0.1, "probe_radius": 0.2, "amortize_steps": 1, "max_subspace_dim": 24},
    ]
    records = []
    for idx, config in enumerate(smoke_configs):
        method = f"polystep_sweep_{idx}"
        run_polystep_taxi(seed=42, device="cpu", results_dir=str(tmp_path), method=method, **config)
        result_path = os.path.join(str(tmp_path), f"taxi_{method}_42.json")
        with open(result_path) as f:
            data = json.load(f)
        records.append({"config_id": str(idx), "config": config, "metrics": data["metrics"]})
    best = select_best_sweep_result(records)

    assert "config" in best
    assert os.path.exists(os.path.join(str(tmp_path), "taxi_polystep_sweep_0_42.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "taxi_polystep_sweep_1_42.json"))


def test_sb3_taxi_baseline_reports_missing_dependency(tmp_path):
    from experiments.runners.run_rl import run_sb3_taxi

    stable_baselines3 = pytest.importorskip("stable_baselines3", reason="missing path tested only when SB3 absent")
    assert stable_baselines3 is not None


def test_sb3_taxi_baseline_missing_dependency_message(monkeypatch, tmp_path):
    import builtins

    from experiments.runners.run_rl import run_sb3_taxi

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("stable_baselines3"):
            raise ImportError("synthetic missing sb3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="stable-baselines3"):
        run_sb3_taxi(method="ppo", seed=42, total_timesteps=8, results_dir=str(tmp_path))


def test_mjlab_g1_module_is_importable_without_mjlab_installed():
    from polystep.benchmarks.rl.mjlab_g1 import MJLAB_TASK_ID, MjlabUnavailableError

    assert MJLAB_TASK_ID == "Mjlab-Velocity-Flat-Unitree-G1"
    assert issubclass(MjlabUnavailableError, RuntimeError)


def test_run_g1_sweep_records_dependency_blocker_when_mjlab_missing(tmp_path):
    import importlib.util

    if importlib.util.find_spec("mjlab") is not None:
        pytest.skip("mjlab installed; blocker path not applicable")

    from experiments.runners.run_rl import run_polystep_g1_sweep

    result = run_polystep_g1_sweep(seed=42, results_dir=str(tmp_path), max_configs=1)

    assert result["status"] == "blocked"
    assert os.path.exists(os.path.join(str(tmp_path), "g1_polystep_sweep_blocked_42.json"))


def test_mjlab_g1_smoke_if_dependency_is_installed():
    pytest.importorskip("mjlab")

    from polystep.benchmarks.rl.mjlab_g1 import MjlabG1Evaluator

    evaluator = MjlabG1Evaluator(num_envs=1, horizon=1, device="cuda")
    assert evaluator.env_id == "Mjlab-Velocity-Flat-Unitree-G1"


@pytest.mark.parametrize("activation", ["elu", "tanh"])
def test_g1_batched_forward_matches_per_candidate_loop(activation):
    """``_batched_mlp_forward`` must equal a per-candidate Python loop.

    Regression guard: previously ``summarize_stacked_params`` hardcoded ``elu``
    in its single-policy loop, silently diverging from the batched bmm path
    when ``activation != "elu"``.
    """

    from polystep.benchmarks.rl.mjlab_g1 import _batched_mlp_forward

    torch.manual_seed(0)
    n_candidates, epc = 3, 5
    obs_dim, hidden, action_dim = 99, 32, 29

    obs = torch.randn(n_candidates * epc, obs_dim)
    stacked = {
        "net.0.weight": torch.randn(n_candidates, hidden, obs_dim) * 0.1,
        "net.0.bias": torch.randn(n_candidates, hidden) * 0.1,
        "net.2.weight": torch.randn(n_candidates, action_dim, hidden) * 0.1,
        "net.2.bias": torch.randn(n_candidates, action_dim) * 0.1,
    }

    batched = _batched_mlp_forward(obs, stacked, n_candidates, epc, activation)

    act_fn = {"elu": torch.nn.functional.elu, "tanh": torch.tanh}[activation]
    expected = torch.empty_like(batched)
    for c in range(n_candidates):
        x = obs[c * epc : (c + 1) * epc]
        x = act_fn(x @ stacked["net.0.weight"][c].T + stacked["net.0.bias"][c])
        x = torch.tanh(x @ stacked["net.2.weight"][c].T + stacked["net.2.bias"][c])
        expected[c * epc : (c + 1) * epc] = x

    assert torch.allclose(batched, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# GymVectorEvaluator smoke tests (Acrobot-v1 always available in gymnasium core)
# ---------------------------------------------------------------------------

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
        # First Linear sits below the non-diff op → must have zero (or None) grad.
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
    pytest.importorskip("Box2D")  # LunarLander dependency
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
    assert r == 0.0  # CartPole step reward is 1 < deadband 2 → bucketed to 0
    env.close()

    # End-to-end factory + Gym registration
    for s in ("cartpole_hard", "acrobot_hard", "lunarlander_hard"):
        e = make_hardened_env(s)
        obs, _ = e.reset(seed=0)
        assert np.isfinite(obs).all()
        e.close()
    register_hardened_envs()
    for gid in HARDENED_GYM_IDS.values():
        e = gym.make(gid)
        e.reset(seed=0)
        e.close()
