"""Checkpoint format for :class:`~polystep.optimizer.PolyStepOptimizer`."""

import warnings

import torch


# state_dict schema version. Bump when a key changes meaning.
# 1: SolverState only. 2: adds "control". 3: rotation-bias and quadratic-model state.
# 4: jitter pool and CMA sampling-projection cache.
# 5: whole-model (summed) trust prediction and incumbent-only comparisons.
_STATE_DICT_FORMAT = 5

# Optimizer attributes outside SolverState that steer the *next* step.
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
    # Read before they are written on the next step.
    "_prev_descent_direction",
    "_prev_descent_direction_finite",
    "_prev_block_descent_directions",
    "_newton_direction",
    "_loss_decreasing_count",
    "_prev_objective_token",
    # Which estimator _prev_pre_step_loss came from.
    "_prev_loss_from_center",
    # Which rank the restored X and projections belong to.
    "_applied_rank",
    # Batched jitter draws not yet consumed.
    "_jitter_pool",
    # What the cached sampling projection was built from.
    "_sampling_C_diag",
    "_sampling_proj_src",
)


class SerializationMixin:
    """``state_dict`` / ``load_state_dict`` for the optimizer."""

    def state_dict(self) -> dict:
        """Serialize the resumable optimizer state (not the model weights)."""

        from .projection import SparseRandomProjection

        def _ser(v):
            if isinstance(v, torch.Tensor):
                return v.detach().to("cpu").clone()
            if isinstance(v, SparseRandomProjection):
                # A pure function of these four numbers; by reference it would keep CUDA storage in the file.
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
                continue  # object; re-linked from construction on load.
            solver_state[name] = _ser(getattr(self._state, name))

        sd = {"format": _STATE_DICT_FORMAT, "solver_state": solver_state}

        # Attached by the blockwise step, not a dataclass field, so the loop above can't see it.
        sd["prev_prev_block_duals"] = _ser(getattr(self._state, "_prev_prev_block_duals", None))

        # Optimizer-owned control state; omitting any of these changes the resumed step.
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
        """Restore optimizer state saved by :meth:`state_dict` (in place)."""
        fmt = sd.get("format")
        if fmt is None or fmt > _STATE_DICT_FORMAT:
            raise ValueError(
                f"Unsupported optimizer state_dict format {fmt!r}; this build writes and "
                f"reads format {_STATE_DICT_FORMAT}. The checkpoint was written by a newer "
                "polystep."
            )
        device = self._state.X.device
        valid = set(type(self._state).__dataclass_fields__)

        # Rebuild the subspace to the checkpoint's rank before any tensors land.
        saved_rank = sd.get("control", {}).get("_applied_rank")
        if saved_rank is not None and saved_rank != self._applied_rank and self.subspace is not None:
            self._transition_rank(saved_rank)

        from .projection import SparseRandomProjection

        def _de(v):
            if isinstance(v, torch.Tensor):
                # clone: a same-device .to() returns the checkpoint's own tensor, which gets written in place.
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

        # Restore the optimizer-owned control state; format-0 checkpoints drop the caches.
        control = sd.get("control")
        if control is None:
            self._invalidate_reuse_cache()
            self._transport_direction_ema = None
        else:
            for name in _CONTROL_STATE_KEYS:
                # `in`, not .get(): an older format missing a key would write None over the live value.
                if name in control:
                    setattr(self, name, _de(control[name]))
            self._ot_step_costs.clear()
            self._ot_step_costs.extend(control.get("_ot_step_costs", []))
            if fmt < 3:
                # Format 2 did not carry the steering state below.
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

        if fmt < 5:
            if self._prev_loss_from_center and self._prev_predicted_improvement is not None:
                self._prev_predicted_improvement = self._prev_predicted_improvement.sum()
            else:
                self._prev_predicted_improvement = None
                self._prev_pre_step_loss = None

        # Rebuild the scaled projection the restored cache describes. The saved copies
        # stay distinct objects from the live state: _update_sampling_projection skips
        # its work on an identity match, and CMA rebinds C_diag every step, so making
        # them identical here would make the first resumed step skip an absorb the
        # uninterrupted run performs.
        src, C = getattr(self, "_sampling_proj_src", None), getattr(self, "_sampling_C_diag", None)
        if isinstance(src, torch.Tensor) and C is not None:
            self._sampling_projection = self.subspace.apply_covariance_scaling(src, C)
        else:
            if fmt < 4 and self.use_covariance_adaptation:
                warnings.warn(
                    "pre-format-4 checkpoint under use_covariance_adaptation: it carries no "
                    "sampling-projection cache, so the resumed run samples from a projection "
                    "the saved coordinates were not measured under. Re-checkpoint for exact resume.",
                    stacklevel=2,
                )
            self._sampling_projection = None

        # The fused block-diagonal projection caches the step-0 basis; the checkpoint carries a later one.
        if self._hybrid and hasattr(self.subspace, "build_fused_projection"):
            projections = self._state.hybrid_projections
            if projections:
                self.subspace.build_fused_projection(projections)
