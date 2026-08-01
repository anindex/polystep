"""Checkpoint format for :class:`~polystep.optimizer.PolyStepOptimizer`.

``state_dict`` / ``load_state_dict`` plus the schema version and the list of
optimizer-owned attributes that steer the next step. Mixed into the optimizer;
kept here so the step orchestration is not interleaved with serialization.
"""

import warnings

import torch


# state_dict schema version. Bump when a key changes meaning; load rejects anything
# newer and fills the gaps for anything older. 1: SolverState only. 2: adds "control".
# 3: adds rotation-bias and quadratic-model state. See CHANGELOG.md.
_STATE_DICT_FORMAT = 3

# Optimizer attributes outside SolverState that steer the *next* step. Every one of
# these is read before it is written on a step, so dropping it changes the resumed
# trajectory.
_SPARSE_TAG = "__sparse_projection__"

_CONTROL_STATE_KEYS = (
    "_amortize_counter",
    "_transport_direction_ema",
    "_trust_region_multiplier",
    "_prev_predicted_improvement",
    "_prev_pre_step_loss",
    "_prev_cost_matrix",
    "_prev_rot_mats",
    "_losses_3d",
    "_prev_X",
    "_prev_k_eff",
    "_prev_step_r",
    "_prev_probe_r",
    # Read before they are written on the next step: the rotation bias picks the
    # search frame, the Newton direction is what an amortized step coasts along, and
    # the decreasing-run counter decides whether the probe count drops.
    "_prev_descent_direction",
    "_prev_descent_direction_finite",
    "_prev_block_descent_directions",
    "_newton_direction",
    "_loss_decreasing_count",
    "_prev_objective_token",
    # Which estimator _prev_pre_step_loss came from. Without it a resume can compare a
    # centre-based baseline against the min-over-vertices proxy.
    "_prev_loss_from_center",
    # Which rank the restored X and projections belong to.
    "_applied_rank",
)


def _ser(v):
    """Detach tensors, recurse into containers, pass everything else through."""
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, dict):
        return {k: _ser(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        out = [_ser(x) for x in v]
        return tuple(out) if isinstance(v, tuple) else out
    return v


class SerializationMixin:
    """``state_dict`` / ``load_state_dict`` for the optimizer."""

    def state_dict(self) -> dict:
        """Serialize the resumable optimizer state (NOT the model weights).

        Captures the solver state (particle positions, warm-start duals,
        momentum velocity, adaptive radius, subspace projection / CMA evolution
        paths and rolling histories, and step counters), the
        ``ProgressiveEpsilon`` scheduler internals, and the RNG generator
        state. Save the model weights separately with ``model.state_dict()``.

        Restore with :meth:`load_state_dict` onto an optimizer built with the
        same configuration and (weight-loaded) model to resume a run. Every value
        that steers the next step is captured, including the optimizer-owned
        control state that lives outside ``SolverState`` (amortization phase,
        transport-direction memory, adaptive-probe reuse caches, trust-region
        multiplier and pending prediction, blockwise dual-momentum history), so
        resume is bit-exact for every configuration.
        """

        from .projection import SparseRandomProjection

        def _ser(v):
            if isinstance(v, torch.Tensor):
                return v.detach().to("cpu").clone()
            if isinstance(v, SparseRandomProjection):
                # A pure function of these four numbers; by reference it would keep CUDA
                # storage in the file and mutate after the save.
                return {
                    _SPARSE_TAG: True,
                    "full_dim": v.full_dim,
                    "subspace_dim": v.subspace_dim,
                    "density": v.density,
                    "seed": v.seed,
                }
            if isinstance(v, dict):
                return {k: _ser(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                out = [_ser(x) for x in v]
                return tuple(out) if isinstance(v, tuple) else out
            return v

        solver_state = {}
        for name in type(self._state).__dataclass_fields__:
            if name == "subspace":
                continue  # object; re-linked from construction on load
            solver_state[name] = _ser(getattr(self._state, name))

        sd = {"format": _STATE_DICT_FORMAT, "solver_state": solver_state}

        # Attached to the state object by the blockwise step rather than declared
        # as a dataclass field, so the loop above cannot see it.
        sd["prev_prev_block_duals"] = _ser(getattr(self._state, "_prev_prev_block_duals", None))

        # Optimizer-owned control state. Omitting any of these makes the next step
        # after a resume differ from the uninterrupted run: the amortization phase
        # decides whether a step evaluates at all, the reuse caches decide which
        # particles are re-evaluated, and the trust-region pair rescales the radius.
        sd["control"] = {name: _ser(getattr(self, name, None)) for name in _CONTROL_STATE_KEYS}
        sd["control"]["_ot_step_costs"] = list(self._ot_step_costs)

        if self._progressive_epsilon is not None:
            sd["progressive_epsilon"] = {
                "current": self._progressive_epsilon._current,
                "smoothed": self._progressive_epsilon._smoothed,
            }
        if self._generator is not None:
            # torch.Generator state is a CPU ByteTensor for CPU *and* CUDA generators.
            sd["generator_state"] = self._generator.get_state().clone()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        """Restore optimizer state saved by :meth:`state_dict` (in place).

        The optimizer must have been constructed with the same configuration
        and model as the one that produced ``sd``. Restored tensors are moved
        onto the current optimizer device; the linked subspace object is
        preserved (only its tensor state is restored).
        """
        fmt = sd.get("format")
        if fmt is None or fmt > _STATE_DICT_FORMAT:
            raise ValueError(
                f"Unsupported optimizer state_dict format {fmt!r}; this build writes and "
                f"reads format {_STATE_DICT_FORMAT}. The checkpoint was written by a newer "
                "polystep."
            )
        device = self._state.X.device
        valid = set(type(self._state).__dataclass_fields__)

        # Rebuild the subspace to the rank the checkpoint was written at, before any
        # of its tensors land. state_dict() carries no subspace object, so without
        # this a run saved after a rank transition restores wide coordinates into the
        # narrow subspace the fresh optimizer was constructed with.
        saved_rank = sd.get("control", {}).get("_applied_rank")
        if saved_rank is not None and saved_rank != self._applied_rank and self.subspace is not None:
            self._transition_rank(saved_rank)

        from .projection import SparseRandomProjection

        def _de(v):
            if isinstance(v, torch.Tensor):
                # clone: a same-device .to() returns the checkpoint's own tensor, which
                # displacement_history then writes in place.
                return v.to(device).clone()
            if isinstance(v, dict) and v.get(_SPARSE_TAG):
                return SparseRandomProjection(
                    full_dim=v["full_dim"],
                    subspace_dim=v["subspace_dim"],
                    density=v["density"],
                    seed=v["seed"],
                )
            if isinstance(v, dict):
                return {k: _de(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                out = [_de(x) for x in v]
                return tuple(out) if isinstance(v, tuple) else out
            return v

        for name, value in sd.get("solver_state", {}).items():
            if name == "subspace" or name not in valid:
                continue
            setattr(self._state, name, _de(value))

        self._state._prev_prev_block_duals = _de(sd.get("prev_prev_block_duals"))

        pe = sd.get("progressive_epsilon")
        if pe is not None and self._progressive_epsilon is not None:
            self._progressive_epsilon._current = pe["current"]
            self._progressive_epsilon._smoothed = pe["smoothed"]

        gs = sd.get("generator_state")
        if gs is not None and self._generator is not None:
            self._generator.set_state(gs.to("cpu"))

        # Restore the optimizer-owned control state. Format-0 checkpoints predate
        # this block; fall back to dropping the caches, which is what those runs did.
        control = sd.get("control")
        if control is None:
            self._invalidate_reuse_cache()
            self._transport_direction_ema = None
        else:
            for name in _CONTROL_STATE_KEYS:
                # `in`, not .get(): an older format missing a key would write None over
                # the live value, and a None _applied_rank fires a spurious transition.
                if name in control:
                    setattr(self, name, _de(control[name]))
            self._ot_step_costs.clear()
            self._ot_step_costs.extend(control.get("_ot_step_costs", []))
            if fmt < 3:
                # Format 2 did not carry the steering state below, so it comes back as
                # None and the first resumed step picks a different search frame.
                warnings.warn(
                    "Loading a pre-format-3 optimizer state_dict. It predates the rotation-bias "
                    "and quadratic-model steering state, so the first step after resume can "
                    "differ from the uninterrupted run when biased_rotation, "
                    "adaptive_num_probe or use_quadratic_model is enabled. Re-checkpoint to "
                    "get exact resume.",
                    stacklevel=2,
                )
                self._prev_descent_direction_finite = False
                self._loss_decreasing_count = 0

        # Recomputed at the start of every step, so it carries nothing across a resume.
        self._sampling_projection = None

        # The fused block-diagonal projection is a cache of the *step-0* basis. The
        # restored checkpoint carries a later basis, so leaving the old one in place
        # evaluates probes through one basis while _sync_model writes another.
        if self._hybrid and hasattr(self.subspace, "build_fused_projection"):
            projections = self._state.hybrid_projections
            if projections:
                self.subspace.build_fused_projection(projections)
