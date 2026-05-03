"""03 - RL starter: gradient-free policy training on CartPole.

PolyStep optimizes the policy directly against the (non-differentiable)
total episode return. No policy-gradient theorem, no value baselines, no
Gym dependency: the reward signal *is* the optimization target.

Why this is a fair use case:
  Standard RL pipelines apply a smoothing trick somewhere (likelihood-ratio
  trick, advantage estimation, GAE, etc.) to get a usable gradient. PolyStep
  treats the environment as a black-box objective and probes around the
  current policy to find descent directions. The CartPole-v1 dynamics here
  are vectorized in pure PyTorch (no Gymnasium), matching the official Gym
  thresholds and reset distribution.

What you should see:
  Mean episode return rises from ~10-40 (random policy) toward 200+ over
  ~80 PolyStep steps. CartPole-v1's max return is 500; we use a reduced
  horizon of 200 to keep the demo under one minute on CPU.

Output:
  examples/figures/rl_cartpole.png

Run:
  python examples/03_rl_cartpole.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

# Allow running directly from a source checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polystep import PolyStepOptimizer  # noqa: E402
from polystep.benchmarks.rl.cartpole import (  # noqa: E402
    CartPoleEvaluator,
    evaluate_policy_module,
    random_policy_baseline,
)
from polystep.benchmarks.rl.policies import DiscreteMLPPolicy  # noqa: E402
from polystep.epsilon import CosineEpsilon  # noqa: E402
from polystep.hybrid_subspace import HybridSubspace  # noqa: E402
from polystep.transform import ParamLayout  # noqa: E402


def main():
    seed = 42
    device = "cpu"
    target_steps = 80
    rollouts_per_candidate = 16
    horizon = 200  # below the 500 max so the demo runs in <60s on CPU
    eval_episodes = 32

    torch.manual_seed(seed)

    print("=" * 60)
    print("CartPole-v1 direct policy search with PolyStep")
    print("=" * 60)

    policy = DiscreteMLPPolicy(obs_dim=4, hidden=16, action_dim=2)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  policy params: {num_params}")

    evaluator = CartPoleEvaluator(
        rollouts_per_candidate=rollouts_per_candidate,
        horizon=horizon,
        device=device,
    )

    # Recipe mirrors experiments/runners/run_rl.py::run_polystep_cartpole.
    # HybridSubspace + softmax solver + small radii are the configuration
    # that actually drives policy improvement on CartPole.
    layout = ParamLayout.from_module(policy)
    subspace = HybridSubspace.from_layout(layout, rank=4)

    epsilon_init, epsilon_target = 1.0, 0.3
    optimizer = PolyStepOptimizer(
        policy,
        solver="softmax",
        subspace=subspace,
        compile=False,
        seed=seed,
        epsilon=CosineEpsilon(
            init=epsilon_init,
            target=epsilon_target,
            decay=(epsilon_init - epsilon_target) / target_steps,
        ),
        step_radius=0.1,
        probe_radius=0.4,
        num_probe=1,
        chunk_size=256,
    )

    rand_summary = random_policy_baseline(
        seed=seed, episodes=eval_episodes, horizon=horizon, device=device,
    )
    init_summary = evaluate_policy_module(
        policy, seed=seed + 10_000, episodes=eval_episodes,
        horizon=horizon, device=device,
    )
    print(f"  random policy mean return:  {rand_summary['mean_return']:.1f}")
    print(f"  initial policy mean return: {init_summary['mean_return']:.1f}")
    print()

    return_log: list[float] = []
    step_log: list[int] = []

    print("training...")
    start = time.time()

    # Single closure shared across steps; uses the optimizer's own iteration
    # counter so the CRN seed advances correctly each step.
    def closure(stacked_params):
        step = optimizer.state.iteration_count if optimizer.state is not None else 0
        return evaluator.loss_for_stacked_params(
            stacked_params, seed=seed, step=step,
        )

    for step in range(target_steps):
        optimizer.step(closure)

        if step % 4 == 0 or step == target_steps - 1:
            summary = evaluate_policy_module(
                policy, seed=seed + 10_000, episodes=eval_episodes,
                horizon=horizon, device=device,
            )
            return_log.append(summary["mean_return"])
            step_log.append(step)
            if step % 16 == 0 or step == target_steps - 1:
                print(
                    f"  step {step:3d} | "
                    f"mean_return={summary['mean_return']:6.1f} "
                    f"success={100 * summary['success_rate']:.0f}%"
                )

    elapsed = time.time() - start
    final_summary = evaluate_policy_module(
        policy, seed=seed + 10_000, episodes=eval_episodes,
        horizon=horizon, device=device,
    )

    print()
    print("=" * 60)
    print(f"  initial mean return: {init_summary['mean_return']:.1f}")
    print(f"  final   mean return: {final_summary['mean_return']:.1f} "
          f"(success {100 * final_summary['success_rate']:.0f}%)")
    print(f"  random baseline:     {rand_summary['mean_return']:.1f}")
    print(f"  wallclock: {elapsed:.1f}s ({target_steps} steps)")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8), constrained_layout=True)
    ax.plot(step_log, return_log, color="#0072B2", lw=1.5, marker="o",
            markersize=3, label="PolyStep policy")
    ax.axhline(rand_summary["mean_return"], color="#999999", ls="--", lw=1.0,
               label=f"random ({rand_summary['mean_return']:.0f})")
    ax.axhline(horizon, color="#009E73", ls=":", lw=1.0,
               label=f"horizon cap ({horizon})")
    ax.set_xlabel("PolyStep step")
    ax.set_ylabel(f"Mean return over {eval_episodes} episodes")
    ax.set_title("CartPole-v1: gradient-free direct policy search", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)

    out = Path(__file__).parent / "figures" / "rl_cartpole.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
