"""Checkpoint format for :class:`~polystep.optimizer.PolyStepOptimizer`."""

import torch


# state_dict schema version. Bump when a key changes meaning.
_STATE_DICT_FORMAT = 6

# Optimizer attributes outside SolverState that steer the *next* step.
_SPARSE_TAG = "__sparse_projection__"

_CONTROL_STATE_KEYS = (
    "_amortize_counter",
    "_transport_direction_ema",
    "_trust_region_multiplier",
    "_prev_predicted_improvement",
    "_prev_pre_step_loss",
    # Read before they are written on the next step.
    "_prev_descent_direction",
    "_prev_block_descent_directions",
    "_newton_direction",
    "_loss_decreasing_count",
    "_prev_objective_token",
    # Which rank the restored X and projections belong to.
    "_applied_rank",
    # Batched jitter draws not yet consumed.
    "_jitter_pool",
    "candidate_evals",
)


class SerializationMixin:
    """``state_dict`` / ``load_state_dict`` for the optimizer."""

    def state_dict(self) -> dict:
        """Serialize the resumable optimizer state (not the model weights)."""

        from .projection import SparseRandomProjection

        def _ser(v):
            if isinstance(v, torch.Tensor):
                return v.detach().to("cpu", copy=True)
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
        if fmt != _STATE_DICT_FORMAT:
            raise ValueError(
                f"Unsupported optimizer state_dict format {fmt!r}; this build reads format "
                f"{_STATE_DICT_FORMAT} only. Rebuild the optimizer from the saved model weights "
                "to continue a run saved by another polystep release."
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
                # copy: a same-device .to() returns the checkpoint's own tensor, which gets written in place.
                return v.to(device, copy=True)
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

        # Restore the optimizer-owned control state.
        control = sd.get("control", {})
        for name in _CONTROL_STATE_KEYS:
            # `in`, not .get(): a missing key would write None over the live value.
            if name in control:
                setattr(self, name, _de(control[name]))
        self._ot_step_costs.clear()
        self._ot_step_costs.extend(control.get("_ot_step_costs", []))
        if self._blocks is None:
            self._state.block_duals = None
            self._state._prev_prev_block_duals = None
            self._prev_block_descent_directions = None

        # The fused block-diagonal projection caches the step-0 basis; the checkpoint carries a later one.
        if self._hybrid and hasattr(self.subspace, "build_fused_projection"):
            projections = self._state.hybrid_projections
            if projections:
                self.subspace.build_fused_projection(projections)
