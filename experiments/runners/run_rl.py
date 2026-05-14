#!/usr/bin/env python
"""Run focused RL benchmarks for PolyStep paper experiments.

Covers CartPole-v1 (analytic + Gym), Acrobot-v1, and the hardened (quantized
obs + bucketed reward) variants. PolyStep, OpenAI-ES, SB3 PPO/DQN, and a
uniform random baseline are supported per env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from experiments.runners.common import SEEDS, save_result, set_seed, track_gpu_memory
from polystep import PolyStepOptimizer
from polystep.benchmarks.rl.metrics import build_rl_metrics, normalize_score
from polystep.benchmarks.rl.policies import (
    DiscreteMLPPolicy,
    NonDiffMLPPolicy,
    stack_module_params,
)
from polystep.benchmarks.rl.cartpole import (
    CartPoleEvaluator,
    DEFAULT_HORIZON as CARTPOLE_HORIZON,
    OBS_DIM as CARTPOLE_OBS_DIM,
    ACTION_DIM as CARTPOLE_ACTION_DIM,
    random_policy_baseline as cartpole_random_baseline,
)
from polystep.epsilon import CosineEpsilon
from polystep.hybrid_subspace import HybridSubspace
from polystep.transform import ParamLayout


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "softmax",
    "rl",
)


class CountingClosure:
    """Wrap a PolyStep closure and count candidate policy evaluations."""

    def __init__(self, closure):
        self.closure = closure
        self.count = 0

    def __call__(self, stacked_params):
        losses = self.closure(stacked_params)
        self.count += int(losses.shape[0])
        return losses


# Multi-seed eval offsets used by multi_seed_summary() to produce CIs on the
# final reported return without adding measurable wall time.
_FINAL_EVAL_SEED_OFFSETS = (20_000, 30_000, 40_000)


def multi_seed_summary(evaluator, stacked_params, *, seed: int, step: int,
                       offsets: tuple[int, ...] = _FINAL_EVAL_SEED_OFFSETS) -> dict:
    """Average evaluator.summarize_stacked_params across several deterministic eval seeds.

    Returns a dict with all original summary keys plus *_std variants for
    mean_return / success_rate / episode_length / fall_rate when present.
    """
    import statistics as _st
    summaries = [
        evaluator.summarize_stacked_params(stacked_params, seed=seed + off, step=step)
        for off in offsets
    ]
    out: dict = {}
    keys = set().union(*(s.keys() for s in summaries))
    for k in keys:
        vals = [float(s[k]) for s in summaries if k in s and s[k] is not None]
        if not vals:
            continue
        out[k] = sum(vals) / len(vals)
        if len(vals) >= 2:
            out[f"{k}_std"] = _st.stdev(vals)
        else:
            out[f"{k}_std"] = 0.0
    out["_n_eval_seeds"] = len(summaries)
    return out


def _sb3_periodic_eval_callback(eval_env_factory, *, n_eval_episodes: int, n_eval_points: int,
                                 total_timesteps: int, deterministic: bool = True,
                                 eval_seed_base: int = 0):
    """Build an SB3 callback that runs deterministic eval every total/n_eval_points steps.

    Records a per-eval curve (env_steps_cumulative, mean_return) so PolyStep and SB3 share
    a comparable X-axis. Returns (callback, curve_list).
    """
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np

    eval_freq = max(1, int(total_timesteps) // max(1, int(n_eval_points)))
    curve: list[dict] = []

    class _PeriodicEval(BaseCallback):
        def __init__(self):
            super().__init__()
            self._next_eval = eval_freq
            self._eval_idx = 0
            self._did_step0 = False

        def _run_eval(self) -> None:
            env = eval_env_factory()
            returns_, lengths_ = [], []
            for ep in range(int(n_eval_episodes)):
                obs, _ = env.reset(seed=eval_seed_base + 50_000 + ep)
                done, truncated = False, False
                total, length = 0.0, 0
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    obs, reward, done, truncated, _ = env.step(action)
                    total += float(reward)
                    length += 1
                returns_.append(total)
                lengths_.append(length)
            env.close()
            mean_ret = float(np.mean(returns_)) if returns_ else 0.0
            mean_len = float(np.mean(lengths_)) if lengths_ else 0.0
            curve.append({
                "step": self._eval_idx,  # 0 for the step-0 anchor; 1, 2, ... otherwise
                "epoch": self._eval_idx,
                "env_steps_cumulative": int(self.num_timesteps),
                "mean_return": mean_ret,
                "episode_length": mean_len,
                "loss": -mean_ret,
                "time": float(self.num_timesteps),
            })
            self._eval_idx += 1

        def _on_step(self) -> bool:
            # Step-0 anchor: emit one eval at num_timesteps == 0 (first call).
            if not self._did_step0:
                self._did_step0 = True
                # Save current num_timesteps which will be small but not 0 (at least 1
                # env step has occurred). Force x-coord to 0 for a true anchor.
                _saved_idx = self._eval_idx
                self._run_eval()
                # Patch the just-appended record to env_steps=0.
                if curve:
                    curve[-1]["env_steps_cumulative"] = 0
                    curve[-1]["time"] = 0.0
                    curve[-1]["step"] = 0
                    curve[-1]["epoch"] = 0
            if self.num_timesteps >= self._next_eval:
                self._run_eval()
                self._next_eval += eval_freq
            return True

    return _PeriodicEval(), curve


def _cartpole_normalized_score(mean_return: float) -> float:
    return max(0.0, min(1.0, normalize_score(mean_return, random_return=22.0, reference_return=500.0)))


CARTPOLE_POLYSTEP_FINAL_CONFIG: dict[str, Any] = {
    "steps": 200,
    "rollouts_per_candidate": 32,
    "horizon": CARTPOLE_HORIZON,
    "hidden": 16,
    "subspace_rank": 4,
    "epsilon_init": 2.0,
    "epsilon_target": 0.3,
    "step_radius": 0.1,
    "probe_radius": 1.5,
    "amortize_steps": 3,
    "max_subspace_dim": 24,
    "selected_from": "hyperparameter sweep",
}


def run_polystep_cartpole(
    *,
    seed: int,
    device: str = "cpu",
    steps: int = 100,
    rollouts_per_candidate: int = 32,
    horizon: int = CARTPOLE_HORIZON,
    hidden: int = 16,
    results_dir: str | None = None,
    subspace_rank: int = 4,
    epsilon_init: float = 1.0,
    epsilon_target: float = 0.3,
    step_radius: float = 0.1,
    probe_radius: float = 0.4,
    amortize_steps: int = 1,
    max_subspace_dim: int | None = None,
    method: str = "polystep",
) -> int:
    """Run PolyStep direct policy search on CartPole-v1."""

    set_seed(seed)
    model = DiscreteMLPPolicy(
        obs_dim=CARTPOLE_OBS_DIM, hidden=hidden, action_dim=CARTPOLE_ACTION_DIM,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    evaluator = CartPoleEvaluator(
        rollouts_per_candidate=rollouts_per_candidate, horizon=horizon, device=device,
    )
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(layout, rank=subspace_rank, max_subspace_dim=max_subspace_dim)
    total_steps = max(1, int(steps))
    print(f"  [CartPole] hidden={hidden} params={param_count} subspace_dim={subspace.subspace_dim} rank={subspace_rank}")

    optimizer = PolyStepOptimizer(
        model,
        solver="softmax",
        subspace=subspace,
        epsilon=CosineEpsilon(
            init=epsilon_init,
            target=epsilon_target,
            decay=(epsilon_init - epsilon_target) / total_steps,
        ),
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=1,
        amortize_steps=amortize_steps,
        chunk_size=256,
        seed=seed,
    )

    step_logs: List[Dict[str, Any]] = []
    best_return = float("-inf")
    best_summary: Dict[str, float] = {}
    start = time.time()
    eval_interval = 1

    def closure(stacked_params):
        step = optimizer.state.iteration_count if optimizer.state is not None else len(step_logs)
        return evaluator.loss_for_stacked_params(stacked_params, seed=seed, step=step)

    counted = CountingClosure(closure)

    # Step-0 anchor.
    init_summary = evaluator.summarize_stacked_params(
        stack_module_params(model, 1), seed=seed + 10_000, step=0,
    )
    step_logs.append({
        "step": 0,
        "epoch": 0,
        "accuracy": _cartpole_normalized_score(init_summary["mean_return"]),
        "mean_return": init_summary["mean_return"],
        "success_rate": init_summary["success_rate"],
        "episode_length": init_summary["episode_length"],
        "loss": -init_summary["mean_return"],
        "time": 0.0,
        "step_wall_time": 0.0,
        "candidates_evaluated": 0,
        "env_steps_cumulative": 0,
    })

    with track_gpu_memory() as mem:
        for step in range(1, total_steps + 1):
            step_start = time.time()
            optimizer.step(counted)
            step_wall = time.time() - step_start

            if step == 1 or step == total_steps or step % eval_interval == 0:
                summary = evaluator.summarize_stacked_params(
                    stack_module_params(model, 1), seed=seed + 10_000, step=0,
                )
                mean_return = summary["mean_return"]
                if mean_return > best_return:
                    best_return = mean_return
                    best_summary = summary
                step_logs.append({
                    "step": step,
                    "epoch": step,
                    "accuracy": _cartpole_normalized_score(mean_return),
                    "mean_return": mean_return,
                    "success_rate": summary["success_rate"],
                    "episode_length": summary["episode_length"],
                    "loss": -mean_return,
                    "time": time.time() - start,
                    "step_wall_time": step_wall,
                    "candidates_evaluated": counted.count,
                    "env_steps_cumulative": counted.count * rollouts_per_candidate * horizon,
                })
                print(f"  [CartPole step {step}/{total_steps}] return={mean_return:.1f} "
                      f"success={summary['success_rate']:.3f} best={best_return:.1f} "
                      f"wall={time.time()-start:.0f}s")

    final_summary = multi_seed_summary(
        evaluator, stack_module_params(model, 1), seed=seed, step=total_steps,
    )
    best_return = max(best_return, final_summary["mean_return"])
    if not best_summary:
        best_summary = final_summary
    metrics = build_rl_metrics(
        final_return=final_summary["mean_return"],
        best_return=best_return,
        normalized_score=_cartpole_normalized_score(final_summary["mean_return"]),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=mem["peak_gpu_memory_mb"],
        function_evals=counted.count,
        total_steps=total_steps,
        rl_env_steps=counted.count * rollouts_per_candidate * horizon,
        success_rate=final_summary["success_rate"],
        episode_length=final_summary["episode_length"],
        best_success_rate=best_summary.get("success_rate", 0.0),
    )
    metrics["final_return_std"] = final_summary.get("mean_return_std", 0.0)
    metrics["final_eval_seeds"] = final_summary.get("_n_eval_seeds", 1)
    epoch_logs = [
        {"epoch": row["step"], "accuracy": row["accuracy"], "loss": row["loss"], "time": row["time"]}
        for row in step_logs
    ]
    save_result(
        benchmark="cartpole",
        method=method,
        seed=seed,
        metrics=metrics,
        hyperparameters={
            "hidden": hidden,
            "steps": total_steps,
            "rollouts_per_candidate": rollouts_per_candidate,
            "horizon": horizon,
            "subspace_rank": subspace_rank,
            "epsilon_init": epsilon_init,
            "epsilon_target": epsilon_target,
            "step_radius": step_radius,
            "probe_radius": probe_radius,
            "amortize_steps": amortize_steps,
            "max_subspace_dim": max_subspace_dim,
            "param_count": param_count,
            "subspace_dim": subspace.subspace_dim,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )
    return counted.count


def run_random_cartpole(
    *, seed: int, eval_episodes: int = 256, horizon: int = CARTPOLE_HORIZON,
    results_dir: str | None = None,
) -> None:
    """Uniform-random-action CartPole baseline."""

    start = time.time()
    summary = cartpole_random_baseline(seed=seed, episodes=eval_episodes, horizon=horizon)
    metrics = build_rl_metrics(
        final_return=summary["mean_return"],
        best_return=summary["mean_return"],
        normalized_score=_cartpole_normalized_score(summary["mean_return"]),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=0.0,
        function_evals=1,
        total_steps=1,
        rl_env_steps=eval_episodes * horizon,
        success_rate=summary["success_rate"],
        episode_length=summary["episode_length"],
    )
    save_result(
        "cartpole", "random_policy", seed, metrics,
        {"eval_episodes": eval_episodes, "horizon": horizon, "action_selection": "uniform_random"},
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )


def run_sb3_cartpole(
    *, method: str, seed: int, total_timesteps: int = 50_000,
    eval_episodes: int = 50, results_dir: str | None = None,
    net_arch: tuple[int, ...] = (16,),
) -> None:
    """Stable-Baselines3 DQN/PPO baseline on CartPole-v1.

    net_arch defaults to (16,) to match PolyStep's DiscreteMLPPolicy(hidden=16).
    """

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for CartPole dqn/ppo baselines") from exc
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required for CartPole baselines") from exc
    import numpy as np

    algo_cls = {"dqn": DQN, "ppo": PPO}.get(method)
    if algo_cls is None:
        raise ValueError(f"Unsupported SB3 CartPole method: {method}")

    start = time.time()
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    policy_kwargs = {"net_arch": list(net_arch)}
    # Force CPU: small MLP, SB3 recommends CPU for non-CNN policies.
    model = algo_cls("MlpPolicy", env, seed=seed, verbose=0, policy_kwargs=policy_kwargs, device="cpu")
    param_count = sum(p.numel() for p in model.policy.parameters())

    def _eval_env_factory():
        return gym.make("CartPole-v1")

    cb, curve = _sb3_periodic_eval_callback(
        _eval_env_factory, n_eval_episodes=min(20, eval_episodes), n_eval_points=80,
        total_timesteps=int(total_timesteps), eval_seed_base=int(seed),
    )
    model.learn(total_timesteps=int(total_timesteps), callback=cb)

    eval_env = gym.make("CartPole-v1")
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(int(eval_episodes)):
        obs, _ = eval_env.reset(seed=seed + 40_000 + ep)
        total = 0.0
        length = 0
        for _ in range(CARTPOLE_HORIZON):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(int(action))
            total += float(reward)
            length += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(length)
    eval_env.close()
    env.close()

    mean_return = float(np.mean(returns)) if returns else 0.0
    best_return = max([mean_return] + [c["mean_return"] for c in curve], default=mean_return)
    metrics = build_rl_metrics(
        final_return=mean_return,
        best_return=best_return,
        normalized_score=_cartpole_normalized_score(mean_return),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=0.0,
        function_evals=int(total_timesteps),
        total_steps=int(total_timesteps),
        rl_env_steps=int(total_timesteps) + int(eval_episodes) * CARTPOLE_HORIZON,
        success_rate=float(sum(1 for length in lengths if length >= CARTPOLE_HORIZON) / max(1, len(lengths))),
        episode_length=float(np.mean(lengths)) if lengths else 0.0,
    )
    save_result(
        "cartpole", method, seed, metrics,
        {"total_timesteps": int(total_timesteps), "eval_episodes": int(eval_episodes),
         "net_arch": list(net_arch), "param_count": int(param_count)},
        epoch_logs=[{"epoch": c["epoch"], "accuracy": 0.0, "loss": c["loss"], "time": c["time"]} for c in curve],
        step_logs=curve,
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )


def _run_cartpole_polystep_full(*, seed: int, device: str, results_dir: str, args) -> None:
    config = CARTPOLE_POLYSTEP_FINAL_CONFIG.copy()
    run_polystep_cartpole(
        seed=seed,
        device=device,
        steps=args.steps or config["steps"],
        rollouts_per_candidate=args.rollouts_per_candidate or config["rollouts_per_candidate"],
        horizon=args.horizon or config["horizon"],
        hidden=args.hidden or config["hidden"],
        subspace_rank=config["subspace_rank"],
        epsilon_init=config["epsilon_init"],
        epsilon_target=config["epsilon_target"],
        step_radius=config["step_radius"],
        probe_radius=args.probe_radius if args.probe_radius is not None else config["probe_radius"],
        amortize_steps=config["amortize_steps"],
        max_subspace_dim=args.max_subspace_dim or config.get("max_subspace_dim"),
        results_dir=results_dir,
    )


# Per-env registry for generic Gymnasium envs.
GYM_ENV_REGISTRY: dict[str, dict[str, Any]] = {
    "cartpole": {
        "env_id": "CartPole-v1",
        "short": "cartpole",
        "horizon": CARTPOLE_HORIZON,
        "random_return": 22.0,
        "reference_return": 500.0,
        "obs_dim": CARTPOLE_OBS_DIM,
        "action_dim": CARTPOLE_ACTION_DIM,
        "hidden": 16,
        "rollouts_per_candidate": 16,
        "polystep": {
            "steps": 200,
            "subspace_rank": 4,
            "epsilon_init": 2.0,
            "epsilon_target": 0.3,
            "step_radius": 0.1,
            "probe_radius": 1.5,
            "amortize_steps": 3,
            "max_subspace_dim": 24,
        },
        "sb3_total_timesteps": {"sweep": 10_000, "full": 1_000_000},
    },
    "acrobot": {
        "env_id": "Acrobot-v1",
        "short": "acrobot",
        "horizon": 500,
        "random_return": -500.0,
        "reference_return": -80.0,
        "obs_dim": 6,
        "action_dim": 3,
        "hidden": 16,
        "rollouts_per_candidate": 16,
        "polystep": {
            "steps": 200,
            "subspace_rank": 4,
            "epsilon_init": 2.0,
            "epsilon_target": 0.3,
            "step_radius": 0.1,
            "probe_radius": 2.0,
            "amortize_steps": 1,
            "max_subspace_dim": 24,
        },
        "sb3_total_timesteps": {"sweep": 10_000, "full": 500_000},
    },
}


# Hardened-env variants: quantized obs + bucketed reward via hardened_env.py.
def _register_hardened_envs_if_needed() -> None:
    try:
        from experiments.runners.hardened_env import register_hardened_envs
    except Exception:  # pragma: no cover - keep import guard cheap
        return
    register_hardened_envs()


_register_hardened_envs_if_needed()

# Hardened-env entries inherit hidden / rollouts / polystep-config from their
# vanilla parents but adjust scoring anchors (random/reference returns drop
# under bucketed sparse reward).
_HARDENED_OVERRIDES = {
    "cartpole_hard": dict(env_id="CartPoleHard-v1", random_return=18.0, reference_return=475.0),
    "acrobot_hard": dict(env_id="AcrobotHard-v1", random_return=-500.0, reference_return=-90.0),
}
for _short, _ovr in _HARDENED_OVERRIDES.items():
    _parent = _short.replace("_hard", "")
    _entry = dict(GYM_ENV_REGISTRY[_parent])
    _entry.update(_ovr)
    _entry["short"] = _short
    GYM_ENV_REGISTRY[_short] = _entry


def _gym_normalized_score(env_short: str, mean_return: float) -> float:
    cfg = GYM_ENV_REGISTRY[env_short]
    return max(0.0, min(1.0, normalize_score(
        mean_return,
        random_return=cfg["random_return"],
        reference_return=cfg["reference_return"],
    )))


def run_polystep_gym(
    *,
    env_short: str,
    seed: int,
    device: str = "cpu",
    steps: int | None = None,
    rollouts_per_candidate: int | None = None,
    horizon: int | None = None,
    hidden: int | None = None,
    results_dir: str | None = None,
    subspace_rank: int | None = None,
    epsilon_init: float | None = None,
    epsilon_target: float | None = None,
    step_radius: float | None = None,
    probe_radius: float | None = None,
    amortize_steps: int | None = None,
    max_subspace_dim: int | None = None,
    method: str = "polystep",
    nondiff_mode: str = "float32",
) -> int:
    """Run PolyStep direct policy search on a generic discrete-action Gym env.

    ``nondiff_mode`` ∈ {``"float32"``, ``"int8"``, ``"binary"``} swaps the inner
    activation for the corresponding non-differentiable op. PolyStep is unaffected
    (it sees only forward losses); PPO/DQN trained with the same mode collapse.
    """

    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator

    cfg = GYM_ENV_REGISTRY[env_short]
    pcfg = cfg["polystep"]
    env_id = cfg["env_id"]

    steps = int(steps) if steps is not None else int(pcfg["steps"])
    rollouts_per_candidate = int(
        rollouts_per_candidate if rollouts_per_candidate is not None else cfg["rollouts_per_candidate"]
    )
    horizon = int(horizon) if horizon is not None else int(cfg["horizon"])
    hidden = int(hidden) if hidden is not None else int(cfg["hidden"])
    subspace_rank = int(subspace_rank) if subspace_rank is not None else int(pcfg["subspace_rank"])
    epsilon_init = float(epsilon_init) if epsilon_init is not None else float(pcfg["epsilon_init"])
    epsilon_target = float(epsilon_target) if epsilon_target is not None else float(pcfg["epsilon_target"])
    step_radius = float(step_radius) if step_radius is not None else float(pcfg["step_radius"])
    probe_radius = float(probe_radius) if probe_radius is not None else float(pcfg["probe_radius"])
    amortize_steps = int(amortize_steps) if amortize_steps is not None else int(pcfg["amortize_steps"])
    max_subspace_dim = (
        int(max_subspace_dim) if max_subspace_dim is not None else pcfg.get("max_subspace_dim")
    )

    set_seed(seed)
    if nondiff_mode == "float32":
        model = DiscreteMLPPolicy(
            obs_dim=int(cfg["obs_dim"]), hidden=hidden, action_dim=int(cfg["action_dim"]),
        ).to(device)
        eval_activation = "tanh"
    else:
        model = NonDiffMLPPolicy(
            obs_dim=int(cfg["obs_dim"]), hidden=hidden, action_dim=int(cfg["action_dim"]),
            mode=nondiff_mode,
        ).to(device)
        eval_activation = nondiff_mode
    param_count = sum(p.numel() for p in model.parameters())
    evaluator = GymVectorEvaluator(
        env_id, rollouts_per_candidate=rollouts_per_candidate, horizon=horizon, device=device,
        activation=eval_activation,
    )
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout, rank=subspace_rank, max_subspace_dim=max_subspace_dim,
    )
    total_steps = max(1, int(steps))
    print(f"  [{env_short}] env={env_id} hidden={hidden} params={param_count} "
          f"subspace_dim={subspace.subspace_dim} rank={subspace_rank}")

    optimizer = PolyStepOptimizer(
        model,
        solver="softmax",
        subspace=subspace,
        epsilon=CosineEpsilon(
            init=epsilon_init,
            target=epsilon_target,
            decay=(epsilon_init - epsilon_target) / total_steps,
        ),
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=1,
        amortize_steps=amortize_steps,
        chunk_size=256,
        seed=seed,
    )

    step_logs: List[Dict[str, Any]] = []
    best_return = float("-inf")
    best_summary: Dict[str, float] = {}
    start = time.time()
    eval_interval = 1

    def closure(stacked_params):
        step = optimizer.state.iteration_count if optimizer.state is not None else len(step_logs)
        return evaluator.loss_for_stacked_params(stacked_params, seed=seed, step=step)

    counted = CountingClosure(closure)

    # Step-0 anchor.
    init_summary = evaluator.summarize_stacked_params(
        stack_module_params(model, 1), seed=seed + 10_000, step=0,
    )
    step_logs.append({
        "step": 0,
        "epoch": 0,
        "accuracy": _gym_normalized_score(env_short, init_summary["mean_return"]),
        "mean_return": init_summary["mean_return"],
        "success_rate": init_summary["success_rate"],
        "episode_length": init_summary["episode_length"],
        "loss": -init_summary["mean_return"],
        "time": 0.0,
        "step_wall_time": 0.0,
        "candidates_evaluated": 0,
        "env_steps_cumulative": 0,
    })

    with track_gpu_memory() as mem:
        for step in range(1, total_steps + 1):
            step_start = time.time()
            optimizer.step(counted)
            step_wall = time.time() - step_start

            if step == 1 or step == total_steps or step % eval_interval == 0:
                summary = evaluator.summarize_stacked_params(
                    stack_module_params(model, 1), seed=seed + 10_000, step=0,
                )
                mean_return = summary["mean_return"]
                if mean_return > best_return:
                    best_return = mean_return
                    best_summary = summary
                step_logs.append({
                    "step": step,
                    "epoch": step,
                    "accuracy": _gym_normalized_score(env_short, mean_return),
                    "mean_return": mean_return,
                    "success_rate": summary["success_rate"],
                    "episode_length": summary["episode_length"],
                    "loss": -mean_return,
                    "time": time.time() - start,
                    "step_wall_time": step_wall,
                    "candidates_evaluated": counted.count,
                    "env_steps_cumulative": counted.count * rollouts_per_candidate * horizon,
                })
                print(f"  [{env_short} step {step}/{total_steps}] return={mean_return:.1f} "
                      f"success={summary['success_rate']:.3f} best={best_return:.1f} "
                      f"wall={time.time()-start:.0f}s")

    final_summary = multi_seed_summary(
        evaluator, stack_module_params(model, 1), seed=seed, step=total_steps,
    )
    best_return = max(best_return, final_summary["mean_return"])
    if not best_summary:
        best_summary = final_summary
    metrics = build_rl_metrics(
        final_return=final_summary["mean_return"],
        best_return=best_return,
        normalized_score=_gym_normalized_score(env_short, final_summary["mean_return"]),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=mem["peak_gpu_memory_mb"],
        function_evals=counted.count,
        total_steps=total_steps,
        rl_env_steps=counted.count * rollouts_per_candidate * horizon,
        success_rate=final_summary["success_rate"],
        episode_length=final_summary["episode_length"],
        best_success_rate=best_summary.get("success_rate", 0.0),
    )
    metrics["final_return_std"] = final_summary.get("mean_return_std", 0.0)
    metrics["final_eval_seeds"] = final_summary.get("_n_eval_seeds", 1)
    epoch_logs = [
        {"epoch": row["step"], "accuracy": row["accuracy"], "loss": row["loss"], "time": row["time"]}
        for row in step_logs
    ]
    save_result(
        benchmark=env_short,
        method=method,
        seed=seed,
        metrics=metrics,
        hyperparameters={
            "env_id": env_id,
            "hidden": hidden,
            "steps": total_steps,
            "nondiff_mode": nondiff_mode,
            "rollouts_per_candidate": rollouts_per_candidate,
            "horizon": horizon,
            "subspace_rank": subspace_rank,
            "epsilon_init": epsilon_init,
            "epsilon_target": epsilon_target,
            "step_radius": step_radius,
            "probe_radius": probe_radius,
            "amortize_steps": amortize_steps,
            "max_subspace_dim": max_subspace_dim,
            "param_count": param_count,
            "subspace_dim": subspace.subspace_dim,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )
    evaluator.close()
    return counted.count


def run_random_gym(
    *, env_short: str, seed: int, eval_episodes: int = 256,
    horizon: int | None = None, results_dir: str | None = None,
) -> None:
    """Uniform-random-action baseline for any registered Gym env."""

    from polystep.benchmarks.rl.gym_evaluator import random_policy_baseline

    cfg = GYM_ENV_REGISTRY[env_short]
    env_id = cfg["env_id"]
    horizon = int(horizon) if horizon is not None else int(cfg["horizon"])

    start = time.time()
    summary = random_policy_baseline(
        env_id, seed=seed, episodes=int(eval_episodes), horizon=horizon,
    )
    metrics = build_rl_metrics(
        final_return=summary["mean_return"],
        best_return=summary["mean_return"],
        normalized_score=_gym_normalized_score(env_short, summary["mean_return"]),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=0.0,
        function_evals=1,
        total_steps=1,
        rl_env_steps=int(eval_episodes) * horizon,
        success_rate=summary["success_rate"],
        episode_length=summary["episode_length"],
    )
    save_result(
        env_short, "random_policy", seed, metrics,
        {"env_id": env_id, "eval_episodes": int(eval_episodes), "horizon": horizon,
         "action_selection": "uniform_random"},
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )


def run_sb3_gym(
    *, env_short: str, method: str, seed: int, total_timesteps: int,
    eval_episodes: int = 50, results_dir: str | None = None,
    net_arch: tuple[int, ...] | None = None,
) -> None:
    """Stable-Baselines3 DQN/PPO baseline on a generic Gym env."""

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for SB3 gym baselines") from exc
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("gymnasium is required for SB3 gym baselines") from exc
    import numpy as np

    cfg = GYM_ENV_REGISTRY[env_short]
    env_id = cfg["env_id"]
    horizon = int(cfg["horizon"])
    if net_arch is None:
        net_arch = (int(cfg["hidden"]),)

    algo_cls = {"dqn": DQN, "ppo": PPO}.get(method)
    if algo_cls is None:
        raise ValueError(f"Unsupported SB3 method: {method}")

    start = time.time()
    env = gym.make(env_id)
    env.reset(seed=seed)
    policy_kwargs = {"net_arch": list(net_arch)}
    model = algo_cls(
        "MlpPolicy", env, seed=seed, verbose=0, policy_kwargs=policy_kwargs, device="cpu",
    )
    param_count = sum(p.numel() for p in model.policy.parameters())

    def _eval_env_factory():
        return gym.make(env_id)

    cb, curve = _sb3_periodic_eval_callback(
        _eval_env_factory, n_eval_episodes=min(20, eval_episodes), n_eval_points=80,
        total_timesteps=int(total_timesteps), eval_seed_base=int(seed),
    )
    model.learn(total_timesteps=int(total_timesteps), callback=cb)

    eval_env = gym.make(env_id)
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(int(eval_episodes)):
        obs, _ = eval_env.reset(seed=seed + 40_000 + ep)
        total = 0.0
        length = 0
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(int(action))
            total += float(reward)
            length += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(length)
    eval_env.close()
    env.close()

    mean_return = float(np.mean(returns)) if returns else 0.0
    best_return = max([mean_return] + [c["mean_return"] for c in curve], default=mean_return)
    metrics = build_rl_metrics(
        final_return=mean_return,
        best_return=best_return,
        normalized_score=_gym_normalized_score(env_short, mean_return),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=0.0,
        function_evals=int(total_timesteps),
        total_steps=int(total_timesteps),
        rl_env_steps=int(total_timesteps) + int(eval_episodes) * horizon,
        success_rate=float(sum(1 for length in lengths if length >= horizon) / max(1, len(lengths))),
        episode_length=float(np.mean(lengths)) if lengths else 0.0,
    )
    save_result(
        env_short, method, seed, metrics,
        {"env_id": env_id, "total_timesteps": int(total_timesteps),
         "eval_episodes": int(eval_episodes),
         "net_arch": list(net_arch), "param_count": int(param_count)},
        epoch_logs=[{"epoch": c["epoch"], "accuracy": 0.0, "loss": c["loss"], "time": c["time"]} for c in curve],
        step_logs=curve,
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )


def _make_nondiff_activation_fn(mode: str):
    """Return a partial that constructs ``NonDiffActivation(mode)`` with no args.

    SB3 calls ``activation_fn()`` with no arguments inside ``MlpExtractor``;
    wrapping our 1-arg ``NonDiffActivation`` in a ``functools.partial`` makes it
    drop-in compatible while preserving the configured non-diff ``mode``.
    """

    import functools
    from polystep.benchmarks.rl.policies import NonDiffActivation

    return functools.partial(NonDiffActivation, mode=mode)


def _wrap_module_output_with_nondiff(module, mode: str):
    """Return ``nn.Sequential(module, NonDiffActivation(mode))``.

    Used to clamp the *final* policy / value / Q outputs through a non-diff op
    so that backprop through the action distribution / Bellman target returns
    zero gradient all the way to the upstream linear weights - the only design
    that genuinely collapses PPO/DQN learning (a single trainable linear head
    on top of binary features can still solve CartPole).
    """

    from torch import nn
    from polystep.benchmarks.rl.policies import NonDiffActivation
    if module is None:
        return module
    return nn.Sequential(module, NonDiffActivation(mode))


def _apply_nondiff_to_sb3_policy(model, method: str, mode: str) -> None:
    """Wrap every output head of an SB3 PPO/DQN policy with ``NonDiffActivation``.

    Together with ``activation_fn = NonDiffActivation`` in ``policy_kwargs``,
    this makes *every* activation in the network non-differentiable: hidden
    activations kill grad to all upstream linears, and the post-output wrap
    kills grad to the final head itself. The result is total gradient collapse
    (verifiable via ``test_sb3_nondiff_zero_grad``).
    """

    if mode == "float32":
        return
    if method == "ppo":
        model.policy.action_net = _wrap_module_output_with_nondiff(model.policy.action_net, mode)
        model.policy.value_net = _wrap_module_output_with_nondiff(model.policy.value_net, mode)
    elif method == "dqn":
        # SB3 DQN: model.q_net is QNetwork; its trailing Linear is `q_net.q_net`.
        # Wrap the inner Linear so the final Q-values are quantized/binarized.
        if hasattr(model.q_net, "q_net"):
            model.q_net.q_net = _wrap_module_output_with_nondiff(model.q_net.q_net, mode)
        if hasattr(model, "q_net_target") and hasattr(model.q_net_target, "q_net"):
            model.q_net_target.q_net = _wrap_module_output_with_nondiff(
                model.q_net_target.q_net, mode
            )


def run_sb3_gym_nondiff(
    *, env_short: str, method: str, seed: int, total_timesteps: int,
    nondiff_mode: str = "binary",
    eval_episodes: int = 50, results_dir: str | None = None,
    features_dim: int | None = None,
) -> None:
    """SB3 PPO/DQN baseline trained through a fully non-differentiable policy.

    The non-diff op is applied at *every* hidden activation (via SB3
    ``activation_fn``) **and** wrapped around every output head (action_net /
    value_net for PPO; q_net / q_net_target for DQN). Backprop through any of
    these returns zero gradient (no STE), so all trainable linears stagnate at
    init. Logged under ``method = f"{method}_nondiff_{mode}"``.
    """

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for SB3 non-diff baselines") from exc
    import gymnasium as gym

    cfg = GYM_ENV_REGISTRY[env_short]
    env_id = cfg["env_id"]
    horizon = int(cfg["horizon"])
    hidden = int(features_dim) if features_dim is not None else int(cfg["hidden"])

    algo_cls = {"dqn": DQN, "ppo": PPO}.get(method)
    if algo_cls is None:
        raise ValueError(f"Unsupported SB3 method: {method}")

    activation_fn = _make_nondiff_activation_fn(nondiff_mode)

    start = time.time()
    env = gym.make(env_id)
    env.reset(seed=seed)
    # Two-hidden-layer architecture so multiple non-diff activations sit
    # between every pair of trainable linears (single-hidden would leave the
    # input-Linear's weights still trainable through one non-diff hop).
    policy_kwargs = {
        "net_arch": [hidden, hidden],
        "activation_fn": activation_fn,
    }
    model = algo_cls(
        "MlpPolicy", env, seed=seed, verbose=0, policy_kwargs=policy_kwargs, device="cpu",
    )
    _apply_nondiff_to_sb3_policy(model, method, nondiff_mode)
    param_count = sum(p.numel() for p in model.policy.parameters())

    def _eval_env_factory():
        return gym.make(env_id)

    cb, curve = _sb3_periodic_eval_callback(
        _eval_env_factory, n_eval_episodes=min(20, eval_episodes), n_eval_points=80,
        total_timesteps=int(total_timesteps), eval_seed_base=int(seed),
    )
    model.learn(total_timesteps=int(total_timesteps), callback=cb)

    eval_env = gym.make(env_id)
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(int(eval_episodes)):
        obs, _ = eval_env.reset(seed=seed + 40_000 + ep)
        total = 0.0
        length = 0
        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(int(action))
            total += float(reward)
            length += 1
            if term or trunc:
                break
        returns.append(total)
        lengths.append(length)
    eval_env.close()
    env.close()

    import numpy as _np
    mean_return = float(_np.mean(returns)) if returns else 0.0
    best_return = max([mean_return] + [c["mean_return"] for c in curve], default=mean_return)
    metrics = build_rl_metrics(
        final_return=mean_return,
        best_return=best_return,
        normalized_score=_gym_normalized_score(env_short, mean_return),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=0.0,
        function_evals=int(total_timesteps),
        total_steps=int(total_timesteps),
        rl_env_steps=int(total_timesteps) + int(eval_episodes) * horizon,
        success_rate=float(sum(1 for length in lengths if length >= horizon) / max(1, len(lengths))),
        episode_length=float(_np.mean(lengths)) if lengths else 0.0,
    )
    save_result(
        env_short, f"{method}_nondiff_{nondiff_mode}", seed, metrics,
        {"env_id": env_id, "total_timesteps": int(total_timesteps),
         "eval_episodes": int(eval_episodes),
         "hidden": hidden, "param_count": int(param_count),
         "nondiff_mode": nondiff_mode},
        epoch_logs=[{"epoch": c["epoch"], "accuracy": 0.0, "loss": c["loss"], "time": c["time"]} for c in curve],
        step_logs=curve,
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )


def run_es_gym(
    *,
    env_short: str,
    seed: int,
    device: str = "cpu",
    generations: int | None = None,
    popsize: int = 32,
    sigma_init: float = 0.1,
    sigma_target: float = 0.02,
    lr: float = 0.05,
    rollouts_per_candidate: int | None = None,
    horizon: int | None = None,
    hidden: int | None = None,
    results_dir: str | None = None,
    nondiff_mode: str = "float32",
) -> None:
    """Hand-rolled OpenAI-ES baseline (antithetic sampling, rank centering, cosine sigma decay).

    Evaluates the whole population in parallel via :class:`GymVectorEvaluator`'s
    stacked-params interface. This is the *fair* gradient-free reference: like
    PolyStep it cannot exploit gradients, so a PolyStep win over ES isolates the
    benefit of OT-guided steps over plain Gaussian smoothing.

    ``nondiff_mode`` ∈ {``"float32"``, ``"int8"``, ``"binary"``}: ES is gradient-free
    so handles non-diff policies natively; included for completeness in the
    non-diff sweep.
    """

    from polystep.benchmarks.rl.gym_evaluator import GymVectorEvaluator

    cfg = GYM_ENV_REGISTRY[env_short]
    env_id = cfg["env_id"]
    horizon = int(horizon) if horizon is not None else int(cfg["horizon"])
    hidden = int(hidden) if hidden is not None else int(cfg["hidden"])
    rollouts_per_candidate = int(
        rollouts_per_candidate if rollouts_per_candidate is not None else cfg["rollouts_per_candidate"]
    )
    # Default generations: match PolyStep's per-env step budget.
    if generations is None:
        generations = int(cfg["polystep"]["steps"])
    # Popsize must be even for antithetic sampling.
    popsize = int(popsize)
    if popsize % 2 != 0:
        popsize += 1

    set_seed(seed)
    if nondiff_mode == "float32":
        model = DiscreteMLPPolicy(
            obs_dim=int(cfg["obs_dim"]), hidden=hidden, action_dim=int(cfg["action_dim"]),
        ).to(device)
        eval_activation = "tanh"
    else:
        model = NonDiffMLPPolicy(
            obs_dim=int(cfg["obs_dim"]), hidden=hidden, action_dim=int(cfg["action_dim"]),
            mode=nondiff_mode,
        ).to(device)
        eval_activation = nondiff_mode
    param_count = sum(p.numel() for p in model.parameters())
    evaluator = GymVectorEvaluator(
        env_id, rollouts_per_candidate=rollouts_per_candidate, horizon=horizon, device=device,
        activation=eval_activation,
    )
    print(f"  [{env_short}/ES] env={env_id} hidden={hidden} params={param_count} "
          f"popsize={popsize} generations={generations}")

    # theta = current mean parameters as a list of named (name, shape).
    base_params: dict[str, torch.Tensor] = {
        n: p.detach().clone().to(device) for n, p in model.named_parameters()
    }
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    half = popsize // 2
    step_logs: list[dict[str, Any]] = []
    best_return = float("-inf")
    best_summary: dict[str, float] = {}
    start = time.time()
    eval_interval = 1
    env_steps_cumulative = 0

    def _make_stacked(noise: dict[str, torch.Tensor], sigma: float) -> dict[str, torch.Tensor]:
        # Stacked params: theta_i = theta + sigma * noise_i, shape (popsize, *param_shape).
        stacked: dict[str, torch.Tensor] = {}
        for name, theta in base_params.items():
            stacked[name] = theta.unsqueeze(0) + sigma * noise[name]
        return stacked

    # Step-0 anchor.
    eval_stacked0 = {n: t.unsqueeze(0) for n, t in base_params.items()}
    init_summary = evaluator.summarize_stacked_params(
        eval_stacked0, seed=seed + 10_000, step=0,
    )
    step_logs.append({
        "step": 0,
        "epoch": 0,
        "accuracy": _gym_normalized_score(env_short, init_summary["mean_return"]),
        "mean_return": init_summary["mean_return"],
        "success_rate": init_summary["success_rate"],
        "episode_length": init_summary["episode_length"],
        "loss": -init_summary["mean_return"],
        "time": 0.0,
        "env_steps_cumulative": 0,
        "sigma": float(sigma_init),
    })

    for gen in range(1, generations + 1):
        # Cosine-annealed sigma.
        import math as _math
        progress = (gen - 1) / max(1, generations - 1)
        sigma = sigma_target + 0.5 * (sigma_init - sigma_target) * (1 + _math.cos(_math.pi * progress))

        # Antithetic noise: half random, half mirrored.
        noise: dict[str, torch.Tensor] = {}
        for name, theta in base_params.items():
            eps_half = torch.randn(
                (half,) + theta.shape, generator=rng, dtype=theta.dtype,
            ).to(device)
            noise[name] = torch.cat([eps_half, -eps_half], dim=0)  # (popsize, *)

        stacked = _make_stacked(noise, sigma)
        # Use evaluator: returns (popsize, R) → mean over R is fitness per candidate.
        result = evaluator.rollout_stacked_params(stacked, seed=seed, step=gen)
        fitness = result.returns.mean(dim=1).detach().cpu()  # (popsize,)
        env_steps_cumulative += int(popsize) * rollouts_per_candidate * horizon

        # Rank-centered weights (standardized -> sum to 0).
        ranks = torch.empty_like(fitness)
        ranks[fitness.argsort()] = torch.arange(popsize, dtype=fitness.dtype)
        centered = (ranks - (popsize - 1) / 2.0) / max(1.0, (popsize - 1) / 2.0)  # in [-1, 1]

        # Update theta: theta <- theta + (lr / (popsize * sigma)) * sum_i (w_i * noise_i)
        scale = lr / (popsize * max(sigma, 1e-8))
        for name, theta in base_params.items():
            # weighted sum across population: einsum 'p,p...->...'
            w = centered.to(theta.device, dtype=theta.dtype)
            update = torch.einsum("p,p...->...", w, noise[name])
            base_params[name] = theta + scale * update

        # Periodic checkpoint eval at theta (mean params).
        if gen == 1 or gen == generations or gen % eval_interval == 0:
            with torch.no_grad():
                # Build a 1-candidate stacked dict from the current theta.
                eval_stacked = {n: t.unsqueeze(0) for n, t in base_params.items()}
            summary = evaluator.summarize_stacked_params(
                eval_stacked, seed=seed + 10_000, step=0,
            )
            mean_return = summary["mean_return"]
            if mean_return > best_return:
                best_return = mean_return
                best_summary = summary
            step_logs.append({
                "step": gen,
                "epoch": gen,
                "accuracy": _gym_normalized_score(env_short, mean_return),
                "mean_return": mean_return,
                "success_rate": summary["success_rate"],
                "episode_length": summary["episode_length"],
                "loss": -mean_return,
                "time": time.time() - start,
                "env_steps_cumulative": env_steps_cumulative,
                "sigma": float(sigma),
            })
            print(f"  [{env_short}/ES gen {gen}/{generations}] return={mean_return:.1f} "
                  f"sigma={sigma:.3f} best={best_return:.1f} wall={time.time()-start:.0f}s")

    # Final multi-seed eval at the mean params.
    eval_stacked = {n: t.unsqueeze(0) for n, t in base_params.items()}
    final_summary = multi_seed_summary(
        evaluator, eval_stacked, seed=seed, step=generations,
    )
    best_return = max(best_return, final_summary["mean_return"])
    if not best_summary:
        best_summary = final_summary
    metrics = build_rl_metrics(
        final_return=final_summary["mean_return"],
        best_return=best_return,
        normalized_score=_gym_normalized_score(env_short, final_summary["mean_return"]),
        wall_time_seconds=time.time() - start,
        peak_gpu_memory_mb=0.0,
        function_evals=int(generations) * popsize,
        total_steps=int(generations),
        rl_env_steps=env_steps_cumulative,
        success_rate=final_summary["success_rate"],
        episode_length=final_summary["episode_length"],
        best_success_rate=best_summary.get("success_rate", 0.0),
    )
    metrics["final_return_std"] = final_summary.get("mean_return_std", 0.0)
    metrics["final_eval_seeds"] = final_summary.get("_n_eval_seeds", 1)
    method_name = "es" if nondiff_mode == "float32" else f"es_nondiff_{nondiff_mode}"
    save_result(
        env_short, method_name, seed, metrics,
        {
            "env_id": env_id, "hidden": hidden, "param_count": int(param_count),
            "popsize": popsize, "generations": int(generations),
            "sigma_init": sigma_init, "sigma_target": sigma_target, "lr": lr,
            "rollouts_per_candidate": rollouts_per_candidate, "horizon": horizon,
            "nondiff_mode": nondiff_mode,
        },
        epoch_logs=[
            {"epoch": r["step"], "accuracy": r["accuracy"], "loss": r["loss"], "time": r["time"]}
            for r in step_logs
        ],
        step_logs=step_logs,
        results_dir=results_dir or DEFAULT_RESULTS_DIR,
    )
    evaluator.close()


# Method dispatch tables for new envs (DQN excluded for LunarLander per plan).
_ACROBOT_METHODS = {"polystep", "random_policy", "dqn", "ppo", "es"}
_LUNARLANDER_METHODS = {"polystep", "random_policy", "ppo", "es"}
# Hardened-env method sets mirror their vanilla parents.
_HARDENED_METHODS = {
    "cartpole_hard": {"polystep", "random_policy", "dqn", "ppo", "es"},
    "acrobot_hard": _ACROBOT_METHODS,
}

_CARTPOLE_METHODS = {
    "polystep",
    "random_policy",
    "dqn",
    "ppo",
    "es",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sweep", "full"], default="full")
    parser.add_argument("--env", choices=[
        "cartpole", "acrobot", "cartpole_hard", "acrobot_hard",
    ], default="cartpole")
    parser.add_argument("--methods", nargs="+", default=["polystep"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--rollouts-per-candidate", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-subspace-dim", type=int, default=None)
    parser.add_argument("--probe-radius", type=float, default=None,
                        help="Override probe_radius in FINAL_CONFIG for PolyStep runs.")
    parser.add_argument("--nondiff-mode", choices=["float32", "int8", "binary"], default="float32",
                        help="Non-differentiable activation for the policy. PolyStep + ES "
                             "handle int8/binary natively; PPO/DQN under int8/binary collapse "
                             "(no STE) - used to motivate gradient-free training.")
    args = parser.parse_args()

    for seed in args.seeds:
        for method in args.methods:
            print(f"\n{'='*60}")
            print(f"[{args.mode}] env={args.env}  method={method}  seed={seed}")
            print(f"{'='*60}")

            if args.env == "cartpole":
                if method not in _CARTPOLE_METHODS:
                    raise ValueError(f"Unknown CartPole method: {method!r}. Valid: {sorted(_CARTPOLE_METHODS)}")
                # Non-differentiable CartPole runs route through the generic
                # GymVectorEvaluator path; float32 keeps the fast analytic path.
                if args.nondiff_mode != "float32":
                    if method == "polystep":
                        run_polystep_gym(
                            env_short="cartpole", seed=seed, device=args.device,
                            results_dir=args.results_dir,
                            nondiff_mode=args.nondiff_mode,
                            method=f"polystep_nondiff_{args.nondiff_mode}",
                        )
                    elif method == "es":
                        run_es_gym(
                            env_short="cartpole", seed=seed, device=args.device,
                            results_dir=args.results_dir,
                            nondiff_mode=args.nondiff_mode,
                        )
                    elif method == "random_policy":
                        run_random_gym(env_short="cartpole", seed=seed, results_dir=args.results_dir)
                    elif method in {"dqn", "ppo"}:
                        cfg = GYM_ENV_REGISTRY["cartpole"]
                        total_timesteps = cfg["sb3_total_timesteps"][args.mode]
                        run_sb3_gym_nondiff(
                            env_short="cartpole", method=method, seed=seed,
                            total_timesteps=int(total_timesteps),
                            nondiff_mode=args.nondiff_mode,
                            results_dir=args.results_dir,
                        )
                    continue
                if method == "polystep":
                    _run_cartpole_polystep_full(
                        seed=seed, device=args.device, results_dir=args.results_dir, args=args,
                    )
                elif method == "random_policy":
                    run_random_cartpole(seed=seed, results_dir=args.results_dir)
                elif method == "es":
                    run_es_gym(
                        env_short="cartpole", seed=seed, device=args.device,
                        results_dir=args.results_dir,
                    )
                elif method in {"dqn", "ppo"}:
                    run_sb3_cartpole(
                        method=method, seed=seed,
                        total_timesteps=10_000 if args.mode == "sweep" else 1_000_000,
                        results_dir=args.results_dir,
                    )

            elif args.env in {"acrobot", "cartpole_hard", "acrobot_hard"}:
                env_short = args.env
                allowed = _HARDENED_METHODS.get(env_short, _ACROBOT_METHODS)
                if method not in allowed:
                    raise ValueError(
                        f"Unknown {env_short} method: {method!r}. Valid: {sorted(allowed)}"
                    )
                cfg = GYM_ENV_REGISTRY[env_short]
                if method == "polystep":
                    run_polystep_gym(
                        env_short=env_short,
                        seed=seed,
                        device=args.device,
                        steps=args.steps,
                        rollouts_per_candidate=args.rollouts_per_candidate,
                        horizon=args.horizon,
                        hidden=args.hidden,
                        probe_radius=args.probe_radius,
                        max_subspace_dim=args.max_subspace_dim,
                        results_dir=args.results_dir,
                        nondiff_mode=args.nondiff_mode,
                        method=("polystep" if args.nondiff_mode == "float32"
                                else f"polystep_nondiff_{args.nondiff_mode}"),
                    )
                elif method == "random_policy":
                    run_random_gym(env_short=env_short, seed=seed, results_dir=args.results_dir)
                elif method == "es":
                    run_es_gym(
                        env_short=env_short, seed=seed, device=args.device,
                        results_dir=args.results_dir,
                        nondiff_mode=args.nondiff_mode,
                    )
                elif method in {"dqn", "ppo"}:
                    total_timesteps = cfg["sb3_total_timesteps"][args.mode]
                    if args.nondiff_mode == "float32":
                        run_sb3_gym(
                            env_short=env_short, method=method, seed=seed,
                            total_timesteps=int(total_timesteps),
                            results_dir=args.results_dir,
                        )
                    else:
                        run_sb3_gym_nondiff(
                            env_short=env_short, method=method, seed=seed,
                            total_timesteps=int(total_timesteps),
                            nondiff_mode=args.nondiff_mode,
                            results_dir=args.results_dir,
                        )


if __name__ == "__main__":
    main()
