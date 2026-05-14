"""Hardened Gymnasium env wrappers for the PolyStep RL story.

Two stackable wrappers that *structurally disadvantage* gradient-based methods
without changing the underlying control problem:

* :class:`QuantizedObsWrapper` - channel-wise observation binning into ``K``
  levels. The forward map ``floor((obs - lo) / (hi - lo) * K)`` is piecewise
  constant, so PPO/DQN value-function gradients are zero almost everywhere
  with respect to the (already-evaluated) observations the network conditions
  on. PolyStep, treating the env as a black box, is unaffected.

* :class:`SparseRewardWrapper` - replace dense reward ``r`` with
  ``sign(r) * floor(|r| / bucket)`` and zero whenever ``|r| < deadband``.
  Inflates policy-gradient advantage variance dramatically; PolyStep's
  *episodic-return* objective is unchanged.

Both wrappers are composable. The hardened registry below combines them with
the per-env observation bounds we know from the Gymnasium spec.

Conventions:

* Wrappers are **deterministic** functions of (obs, reward) and therefore do
  not change reset seeds.
* Observation bounds clip values outside ``[lo, hi]`` *before* binning so a
  rare out-of-bounds observation does not destabilise the bin grid.
* The ``register_hardened_envs()`` function registers ``CartPole-Hard-v1``,
  ``Acrobot-Hard-v1`` env IDs that bake in the
  per-env defaults; the underlying class definitions remain reusable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Observation quantization
# ---------------------------------------------------------------------------
def _make_obs_quantizer(low: np.ndarray, high: np.ndarray, bins: int):
    lo = np.asarray(low, dtype=np.float32)
    hi = np.asarray(high, dtype=np.float32)
    bins = int(bins)
    span = np.where(hi > lo, hi - lo, np.float32(1.0))

    def quantize(obs: np.ndarray) -> np.ndarray:
        clipped = np.clip(obs.astype(np.float32), lo, hi)
        # Map into [0, bins-1] integer bins, then back to bin centre in [lo, hi].
        u = (clipped - lo) / span  # in [0, 1]
        # floor into [0, bins-1]
        idx = np.minimum(np.floor(u * bins).astype(np.int32), bins - 1)
        # Bin centre representative value.
        centre = lo + (idx.astype(np.float32) + 0.5) * (span / bins)
        return centre

    return quantize


def QuantizedObsWrapper(env, *, bins: int = 4,
                        low: Optional[Sequence[float]] = None,
                        high: Optional[Sequence[float]] = None):
    """Return a Gymnasium ``ObservationWrapper`` that channel-wise bins obs.

    If ``low``/``high`` are ``None`` the wrapper uses ``env.observation_space``
    bounds, falling back to ``[-10, 10]`` for unbounded channels.
    """

    import gymnasium as gym

    space = env.observation_space
    space_low = np.asarray(space.low, dtype=np.float32)
    space_high = np.asarray(space.high, dtype=np.float32)

    # Replace ±inf with sane defaults - many gym envs leave velocity components
    # unbounded (e.g. CartPole pole velocity) which makes binning meaningless.
    finite_low = np.where(np.isfinite(space_low), space_low, -10.0).astype(np.float32)
    finite_high = np.where(np.isfinite(space_high), space_high, 10.0).astype(np.float32)
    if low is not None:
        finite_low = np.asarray(low, dtype=np.float32)
    if high is not None:
        finite_high = np.asarray(high, dtype=np.float32)

    quantize = _make_obs_quantizer(finite_low, finite_high, bins)

    class _QuantizedObs(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            # Observation space stays the same shape & dtype - quantized values
            # still lie inside the original bounds.
            self.observation_space = gym.spaces.Box(
                low=finite_low, high=finite_high, shape=space.shape, dtype=np.float32,
            )

        def observation(self, obs):
            return quantize(np.asarray(obs))

    return _QuantizedObs(env)


# ---------------------------------------------------------------------------
# Sparse / bucketed reward
# ---------------------------------------------------------------------------
def SparseRewardWrapper(env, *, bucket: float = 1.0, deadband: float = 0.0):
    """Return a Gymnasium ``RewardWrapper`` that buckets and dead-bands rewards.

    Step reward ``r`` is replaced by::

        r' = 0                           if |r| < deadband
             sign(r) * floor(|r| / bucket) * bucket   otherwise
    """

    import gymnasium as gym

    bucket = float(bucket)
    deadband = float(deadband)

    class _SparseReward(gym.RewardWrapper):
        def reward(self, reward):
            r = float(reward)
            if abs(r) < deadband:
                return 0.0
            sign = 1.0 if r > 0 else -1.0
            return sign * float(np.floor(abs(r) / bucket)) * bucket

    return _SparseReward(env)


# ---------------------------------------------------------------------------
# Hardened-env factory + Gymnasium registration
# ---------------------------------------------------------------------------
HARDENED_DEFAULTS = {
    # Each entry: base_env_id, obs_bins, reward_bucket, reward_deadband,
    # optional explicit (low, high) obs bounds for the quantizer.
    "cartpole_hard": dict(
        base="CartPole-v1", bins=4, bucket=1.0, deadband=0.0,
        low=[-2.4, -3.0, -0.21, -3.5], high=[2.4, 3.0, 0.21, 3.5],
    ),
    "acrobot_hard": dict(
        base="Acrobot-v1", bins=4, bucket=1.0, deadband=0.0,
        low=None, high=None,  # Acrobot bounds are finite already.
    ),
}


def make_hardened_env(env_short: str):
    """Construct a fresh hardened env by short-name (``cartpole_hard`` etc)."""

    import gymnasium as gym

    if env_short not in HARDENED_DEFAULTS:
        raise KeyError(f"unknown hardened env {env_short!r}; "
                       f"choices: {sorted(HARDENED_DEFAULTS)}")
    spec = HARDENED_DEFAULTS[env_short]
    env = gym.make(spec["base"])
    env = QuantizedObsWrapper(env, bins=spec["bins"], low=spec["low"], high=spec["high"])
    env = SparseRewardWrapper(env, bucket=spec["bucket"], deadband=spec["deadband"])
    return env


_REGISTERED = False


def register_hardened_envs() -> None:
    """Register ``cartpole_hard`` / ``acrobot_hard`` Gym IDs.

    Idempotent - safe to call multiple times.
    """

    global _REGISTERED
    if _REGISTERED:
        return
    import gymnasium as gym

    id_map = {
        "cartpole_hard": "CartPoleHard-v1",
        "acrobot_hard": "AcrobotHard-v1",
    }
    for short, gym_id in id_map.items():
        spec = HARDENED_DEFAULTS[short]
        try:
            gym.register(
                id=gym_id,
                entry_point=lambda spec=spec: SparseRewardWrapper(
                    QuantizedObsWrapper(
                        gym.make(spec["base"]),
                        bins=spec["bins"], low=spec["low"], high=spec["high"],
                    ),
                    bucket=spec["bucket"], deadband=spec["deadband"],
                ),
            )
        except gym.error.Error:
            # Already registered (e.g. previous import).
            pass
    _REGISTERED = True


HARDENED_GYM_IDS = {
    "cartpole_hard": "CartPoleHard-v1",
    "acrobot_hard": "AcrobotHard-v1",
}
