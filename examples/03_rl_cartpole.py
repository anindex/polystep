"""03 - RL starter: gradient-free policy training on CartPole.

PolyStep optimizes the policy directly against the (non-differentiable)
total episode return. The CartPole-v1 dynamics are vectorized in pure
PyTorch and match the official Gym thresholds; after training, a
Gymnasium render window shows the trained policy.

Output:
  examples/figures/rl_cartpole.png

Run:
  python examples/03_rl_cartpole.py
  python examples/03_rl_cartpole.py --no-render   # headless
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch

import _env  # noqa: E402

_env.setup()


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


def visualize_policy(policy, num_episodes: int = 3, horizon: int = 500):
    """Run the trained policy in Gymnasium and save a GIF.

    Uses ``render_mode="rgb_array"``: the ``"human"`` mode needs a GLX
    context that fails on headless setups.
    """
    import gymnasium as gym
    from PIL import Image

    # Override the 500-step truncation to show long-term stability.
    env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=horizon)
    frames: list = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(horizon):
            frame = env.render()
            frames.append(frame)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=next(policy.parameters()).device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(obs_t)
                action = int(logits.argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"  episode {ep + 1}: return = {total_reward:.0f}")
    env.close()

    # Save as GIF (sub-sampled to keep file size reasonable).
    out = Path(__file__).parent / "figures" / "rl_cartpole_policy.gif"
    os.makedirs(out.parent, exist_ok=True)
    step = max(1, len(frames) // 200)  # cap at ~200 frames
    imgs = [Image.fromarray(f) for f in frames[::step]]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=33, loop=0)
    print(f"  saved visualization: {out}")


def main():
    parser = argparse.ArgumentParser(description="CartPole policy search with PolyStep")
    parser.add_argument("--no-render", action="store_true", help="skip Gymnasium visualization after training")
    # CPU by default: the rollout loop is sequential and each step is tiny,
    # so device launch overhead dominates.
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seed = 42
    device = args.device
    target_steps = 40
    rollouts_per_candidate = 16
    horizon = 200  # below the 500 max to keep the demo quick on CPU
    eval_episodes = 32

    torch.manual_seed(seed)

    print("=" * 60)
    print("CartPole-v1 direct policy search with PolyStep")
    print("=" * 60)

    policy = DiscreteMLPPolicy(obs_dim=4, hidden=16, action_dim=2).to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  policy params: {num_params}")

    evaluator = CartPoleEvaluator(
        rollouts_per_candidate=rollouts_per_candidate,
        horizon=horizon,
        device=device,
    )

    # Mirrors experiments/runners/run_rl.py::run_polystep_cartpole:
    # HybridSubspace + softmax solver + small radii.
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
        seed=seed,
        episodes=eval_episodes,
        horizon=horizon,
        device=device,
    )
    init_summary = evaluate_policy_module(
        policy,
        seed=seed + 10_000,
        episodes=eval_episodes,
        horizon=horizon,
        device=device,
    )
    print(f"  random policy mean return:  {rand_summary['mean_return']:.1f}")
    print(f"  initial policy mean return: {init_summary['mean_return']:.1f}")
    print()

    return_log: list[float] = []
    step_log: list[int] = []

    print("training...")
    start = time.time()

    # Shared closure; the optimizer's own iteration counter advances the
    # CRN seed correctly each step.
    def closure(stacked_params):
        step = optimizer.state.iteration_count if optimizer.state is not None else 0
        return evaluator.loss_for_stacked_params(
            stacked_params,
            seed=seed,
            step=step,
        )

    for step in range(target_steps):
        optimizer.step(closure)

        if step % 4 == 0 or step == target_steps - 1:
            summary = evaluate_policy_module(
                policy,
                seed=seed + 10_000,
                episodes=eval_episodes,
                horizon=horizon,
                device=device,
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
        policy,
        seed=seed + 10_000,
        episodes=eval_episodes,
        horizon=horizon,
        device=device,
    )

    print()
    print("=" * 60)
    print(f"  initial mean return: {init_summary['mean_return']:.1f}")
    print(
        f"  final   mean return: {final_summary['mean_return']:.1f} "
        f"(success {100 * final_summary['success_rate']:.0f}%)"
    )
    print(f"  random baseline:     {rand_summary['mean_return']:.1f}")
    print(f"  wallclock: {elapsed:.1f}s ({target_steps} steps)")
    print("=" * 60)

    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib not installed; skipping the figure (pip install matplotlib).")
    else:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8), constrained_layout=True)
        ax.plot(step_log, return_log, color="#0072B2", lw=1.5, marker="o", markersize=3, label="PolyStep policy")
        ax.axhline(
            rand_summary["mean_return"],
            color="#999999",
            ls="--",
            lw=1.0,
            label=f"random ({rand_summary['mean_return']:.0f})",
        )
        ax.axhline(horizon, color="#009E73", ls=":", lw=1.0, label=f"horizon cap ({horizon})")
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

    # Gymnasium is only needed for rendering; training uses the internal dynamics.
    if args.no_render:
        print("(skipping Gymnasium render; pass without --no-render to visualize)")
    elif importlib.util.find_spec("gymnasium") is None:
        print("gymnasium not installed; skipping the render (pip install gymnasium).")
    else:
        print()
        print("launching Gymnasium CartPole-v1 visualization...")
        visualize_policy(policy, num_episodes=1, horizon=2000)


if __name__ == "__main__":
    main()
