"""Monolithic step: a single OT solve over all particles."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Callable

import torch

from .cost_nn import _reuse_prefixes
from .costs import scale_cost_matrix
from .solvers._shared import loss_buffer_dtype, recenter_cost, sanitize_cost, solver_health
from .epsilon import feed_solver_stats, radius_epsilon_factor
from ._step_core import (
    invalidate_for_basis_change,
    record_saturation,
    update_amortized_direction,
    update_stagnation_and_radius,
    warn_all_nonfinite,
)
from .dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
)
from .geometry import apply_biased_rotation, get_random_rotation_matrices
from .solvers.greedy import MinCostGreedySolver, TopKMeanSolver
from .solvers.base import SolverResult
from .cma import (
    update_evolution_path_sigma,
    compute_heaviside_sigma,
    update_evolution_path_c,
    update_covariance_diagonal,
)

logger = logging.getLogger(__name__)


def _fill_screened_losses(screen_cost, kept, sel_idx, keep_mask, P_active, V, K_eff, mask_dropped=False):
    """Assemble the (P_active*V*K_eff,) loss vector from a screened evaluation.

    Kept vertices carry full-fidelity values; dropped ones get a calibrated cheap
    value for weighted solvers, or ``+inf`` (``mask_dropped``) for selection solvers.
    """
    losses = screen_cost.new_empty(P_active * V * K_eff)
    losses[sel_idx] = kept

    full_3d = losses.reshape(P_active, V, K_eff)
    mask_3d = keep_mask.unsqueeze(-1)
    if mask_dropped:
        return torch.where(mask_3d, full_3d, torch.full_like(full_3d, float("inf"))).reshape(-1)
    # Zero dropped positions before summing, or 0 * inf becomes NaN.
    full_kept = torch.where(mask_3d, full_3d, torch.zeros_like(full_3d))
    n_kept = keep_mask.sum(dim=1, keepdim=True).to(screen_cost.dtype).clamp(min=1)
    full_mean = full_kept.sum(dim=(1, 2)).unsqueeze(1) / (n_kept * K_eff)
    screen_mean = (screen_cost * keep_mask).sum(dim=1, keepdim=True) / n_kept
    calibrated = (screen_cost + (full_mean - screen_mean)).unsqueeze(-1).expand(P_active, V, K_eff)

    return torch.where(mask_3d, full_3d, calibrated).reshape(-1)


def _chunk_spans(n_cand: int, chunk: int, bounds=None):
    """Yield (start, end) chunks of at most ``chunk``, also breaking at ``bounds``.

    Cutting at entry boundaries lets the site-aware paths resolve a site.
    """
    stops = sorted(set(bounds or ()) | {n_cand})
    start = 0
    for stop in stops:
        if stop <= start:
            continue
        for chunk_start in range(start, stop, chunk):
            yield chunk_start, min(chunk_start + chunk, stop)
        start = stop


def _rotation_due(adaptive_sub, iteration: int) -> bool:
    """Whether the basis rotates on this step."""
    interval = getattr(adaptive_sub, "rotation_interval", 1)
    if interval <= 0:
        return False
    return iteration % interval == 0


def record_displacement(state, adaptive_sub, pre_step_sub_coords, proj_used) -> None:
    """Append this step's subspace displacement to the rolling history."""
    post_step_sub_coords = state.X.reshape(-1)[: adaptive_sub.subspace_dim]
    displacement = post_step_sub_coords - pre_step_sub_coords

    idx = state.displacement_history_idx
    state.displacement_history[idx] = displacement
    # Sparse projections rotate by reseeding, so they never read this history.
    if isinstance(proj_used, torch.Tensor):
        if state.displacement_history_full is None:
            state.displacement_history_full = displacement.new_zeros(
                adaptive_sub.displacement_history_size, proj_used.shape[0]
            )
        state.displacement_history_full[idx] = (proj_used @ displacement.to(proj_used.dtype)).to(
            state.displacement_history_full.dtype
        )
    state.displacement_history_idx = (idx + 1) % adaptive_sub.displacement_history_size
    state.displacement_history_count = min(
        state.displacement_history_count + 1,
        adaptive_sub.displacement_history_size,
    )


def maintain_per_layer_subspace(opt, state, cost_mean: float, pre_step_sub_coords) -> bool:
    """Displacement tracking, absorb, and rotation for a per-layer subspace.

    Returns whether the basis changed, so callers can drop warm-started duals.
    """
    if not opt._per_layer_projections:
        return False
    hybrid_sub = opt.subspace
    _basis_before = state.hybrid_projections

    post_step_sub_coords = state.X.reshape(-1)[: hybrid_sub.subspace_dim]
    displacement = post_step_sub_coords - pre_step_sub_coords

    idx = state.displacement_history_idx
    state.displacement_history[idx] = displacement
    state.displacement_history_idx = (idx + 1) % hybrid_sub.displacement_history_size
    state.displacement_history_count = min(
        state.displacement_history_count + 1,
        hybrid_sub.displacement_history_size,
    )

    should_absorb = hybrid_sub.should_absorb(
        state.stagnation_count,
        state.iteration_count,
    )

    if should_absorb:
        full_flat_sub = state.X.reshape(-1)[: hybrid_sub.subspace_dim]
        new_base, _zeroed = hybrid_sub.absorb(
            state.hybrid_projections,
            state.base_params,
            full_flat_sub,
        )
        state.base_params = new_base
        state.X = torch.zeros_like(state.X)
        # init_projections seeds step=0, so absorbing returns the same basis;
        # absorb_aligned_active biases it toward productive directions via displacement-SVD.
        if getattr(hybrid_sub, "absorb_aligned_active", False) and state.displacement_history_count > 0:
            hist = state.displacement_history[: state.displacement_history_count]
            svd_ratio = hybrid_sub.get_svd_ratio(state.iteration_count, opt.max_iterations or 1)
            state.hybrid_projections = hybrid_sub._rotate_all_displacement(
                state.hybrid_projections,
                hist,
                svd_ratio,
                state.X.device,
                state.X.dtype,
                state.iteration_count,
            )
        else:
            state.hybrid_projections = hybrid_sub.init_projections(
                state.X.device,
                state.X.dtype,
            )
        # Rebuild the fused matrix for the new basis, unless the basis came back by identity.
        if state.hybrid_projections is not _basis_before and hasattr(hybrid_sub, "build_fused_projection"):
            hybrid_sub.build_fused_projection(state.hybrid_projections)
        state.displacement_history.zero_()
        if state.displacement_history_full is not None:
            state.displacement_history_full.zero_()
        state.displacement_history_idx = 0
        state.displacement_history_count = 0
        invalidate_for_basis_change(opt, state)
        state.absorb_count += 1
        # Otherwise a stagnation absorb re-triggers every step on a plateau.
        state.stagnation_count = 0
        state.prev_loss = cost_mean
    else:
        hist = (
            state.displacement_history[: state.displacement_history_count]
            if state.displacement_history_count > 0
            else None
        )
        new_projections = hybrid_sub.rotate_all(
            state.hybrid_projections,
            step=state.iteration_count,
            total_steps=opt.max_iterations,
            displacement_history=hist,
        )
        # Only act when projections changed; identity means the block_diag rebuild
        # below can be skipped.
        if new_projections is not state.hybrid_projections:
            # Re-anchor first: swapping P at non-zero coords moves the weights unevaluated.
            state.base_params, _ = hybrid_sub.absorb(
                state.hybrid_projections,
                state.base_params,
                state.X.reshape(-1)[: hybrid_sub.subspace_dim],
            )
            state.X = torch.zeros_like(state.X)
            # Rows index the replaced basis, so the next rotation would read phantom
            # directions.
            state.displacement_history.zero_()
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            invalidate_for_basis_change(opt, state)
            state.hybrid_projections = new_projections
            if hasattr(hybrid_sub, "build_fused_projection"):
                hybrid_sub.build_fused_projection(new_projections)
    # An absorb re-anchors the origin even when the basis is unchanged, so identity
    # alone would under-report it and leave duals warm-started against the old origin.
    return should_absorb or state.hybrid_projections is not _basis_before


def step_monolithic(opt, closure: Callable, screen_closure: Callable | None = None) -> float:
    """Monolithic step: one OT solve over all particles."""
    state = opt._state
    X = state.X  # (P, particle_dim)
    iteration = state.iteration_count
    device = X.device

    opt._update_sampling_projection()

    current_eps = opt._get_epsilon(iteration)
    if opt.use_adaptive_radius:
        radius_mult = state.radius_multiplier
    else:
        radius_mult = 1.0
    # Scheduled radii are physical distances; scalar radii are multipliers on epsilon.
    _sr = opt._get_step_radius(iteration)
    _pr = opt._get_probe_radius(iteration)
    _sr_eps = radius_epsilon_factor(opt.step_radius, current_eps)
    _pr_eps = radius_epsilon_factor(opt.probe_radius, current_eps)
    if opt.trust_region:
        step_r = _sr * opt._trust_region_multiplier * _sr_eps * radius_mult
    else:
        step_r = _sr * _sr_eps * radius_mult
    probe_r = _pr * _pr_eps * radius_mult

    # Probe-radius jitter; no-op at 0.
    probe_r = opt._apply_probe_radius_jitter(probe_r)

    if X.dim() == 1:
        X = X.unsqueeze(0)
    P, pdim = X.shape

    if opt._polytope_vertices.device != device or opt._polytope_vertices.dtype != X.dtype:
        opt._polytope_vertices = opt._polytope_vertices.to(device=device, dtype=X.dtype)
    polytope_verts = opt._polytope_vertices
    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes
    V = polytope_verts.shape[0]  # num vertices
    K = probes.shape[0]  # num probes

    K_eff = K  # effective probe count for this step
    if opt.adaptive_num_probe and iteration >= opt._adaptive_probe_warmup:
        # Drop K to 1 once the last 3 OT-step costs are strictly decreasing. Gate on
        # finiteness, not positivity: negative objectives descend just as much.
        costs_history = opt._ot_step_costs
        if len(costs_history) >= 3:
            recent = list(costs_history)[-3:]
            if all(math.isfinite(c) for c in recent) and all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                opt._loss_decreasing_count += 1
            else:
                opt._loss_decreasing_count = 0

            if opt._loss_decreasing_count >= 3:
                K_eff = 1

    if K_eff < K:
        probes = probes[K // 2 : K // 2 + 1]  # center scale, shape (1,)

    # All or nothing on X: a candidate replaces one row, so any particle moving
    # invalidates every row.
    _can_reuse = (
        opt._adaptive_probes
        # Reuse needs a stationary objective; the token is the caller's only way to
        # assert that.
        and opt._prev_objective_token is not None
        and opt._prev_cost_matrix is not None
        and opt._prev_cost_matrix.shape == (P, V)
        and opt._prev_rot_mats is not None
        and opt._prev_rot_mats.shape == (P, pdim, pdim)
        and opt._prev_X is not None
        and opt._prev_X.shape == X.shape
        and opt._prev_k_eff == K_eff
        and opt._prev_step_r == step_r
        # Cached costs were measured at probe_r, so it must match too.
        and opt._prev_probe_r == probe_r
        and bool(torch.sum((X - opt._prev_X) ** 2) < opt._adaptive_probes_threshold)
    )
    P_active = 0 if _can_reuse else P

    if _can_reuse:
        rot_mats = opt._prev_rot_mats
    else:
        rot_mats = get_random_rotation_matrices(
            P,
            pdim,
            device=device,
            dtype=X.dtype,
            generator=opt._generator,
        )

        if (
            opt.biased_rotation
            and opt._prev_descent_direction is not None
            and opt._prev_descent_direction.shape == (P, pdim)
            and opt._prev_descent_direction_finite
        ):
            rot_mats = apply_biased_rotation(rot_mats, opt._prev_descent_direction)

    X_vertices, rotated = opt._compiled.rotate_and_translate(
        rot_mats,
        polytope_verts,
        X,
        step_r,
    )

    X_probe = opt._compiled.compute_probe_points(
        X,
        rotated,
        probes,
        probe_r,
    )

    # The configuration has not moved since the cached matrix was measured, so every
    # row still describes its vertices.
    _evals_this_step = 0
    # Cleared per step, or a stale flag would read as this step's verdict.
    opt._all_nonfinite = None
    if _can_reuse:
        # An alias, not a copy: the cached matrix is already sanitized and no consumer
        # writes into it.
        cost_matrix = opt._prev_cost_matrix
    else:
        # Every (particle, vertex, probe) candidate, evaluated in chunks to bound memory.
        total_evals = P_active * V * K_eff
        _evals_this_step = total_evals

        _is_subspace = opt.subspace is not None
        _sub_dim = state.subspace.subspace_dim if _is_subspace else 0

        # In-place forward path: avoids materialising N full weight dicts.
        _use_fused_inplace = (
            opt._hybrid
            and hasattr(state.subspace, "apply_perturbation_inplace")
            and getattr(getattr(opt, "_cost_evaluator", None), "_use_inplace", False)
        )

        # Factored subspace scores through the low-rank identity, never building a
        # candidate weight.
        _factored_eval = getattr(opt, "_factored_evaluator", None) if opt._factored else None

        # Full space perturbs one contiguous run, so only that tensor needs batching.
        _sparse_delta = getattr(opt, "_sparse_delta_evaluator", None) if not _is_subspace else None
        # Same locality, no module-set assumption, covers what the delta path declines.
        _site_vmap = getattr(opt, "_site_vmap_evaluator", None) if not _is_subspace else None
        _pdim_arange = torch.arange(pdim, device=device)
        if _sparse_delta is not None or _site_vmap is not None:
            _base_sd = opt.layout.unflatten(X)

        _fused_inputs = getattr(opt, "_fused_inputs", None)
        _fused_targets = getattr(opt, "_fused_targets", None)
        _subspace_delta = getattr(opt, "_subspace_delta_evaluator", None) if opt._hybrid else None
        # The same site argument in coordinate space: a per-layer block maps to one
        # parameter, so only that one is batched.
        _subspace_site = getattr(opt, "_site_vmap_evaluator", None) if opt._hybrid else None
        _bary_sd = None
        if _fused_inputs is not None and (_subspace_delta is not None or _subspace_site is not None):
            _bary_sd = state.subspace.apply_perturbation(
                state.hybrid_projections,
                state.base_params,
                X.reshape(-1)[:_sub_dim],
            )
            _bary_coords = X.reshape(-1)[:_sub_dim]
        else:
            _subspace_delta = _subspace_site = None
            _bary_coords = None

        _evaluator_native = (
            _use_fused_inplace
            or _sparse_delta is not None
            or _factored_eval is not None
            or _subspace_delta is not None
            or _subspace_site is not None
        )
        # One incumbent loss serves every parameter block, at every probe count.
        _center_wanted = opt.use_quadratic_model and (opt.biased_rotation or opt.trust_region)
        screen_inputs, screen_targets = opt._screen_data(_evaluator_native)
        # A selection solver ranks vertices by their own screened cost, so it needs no
        # antithetic pairing; the contrast-ranked branch below still does.
        _selection_solver = isinstance(opt.solver, (MinCostGreedySolver, TopKMeanSolver))
        _screen_ready = (
            opt.multifidelity_screen
            and screen_closure is not None
            # Every remaining branch ranks on the screened costs alone.
            and opt.screen_keep_ratio < 1.0
            and not (opt.use_quadratic_model or opt._newton_refinement or opt.trust_region)
            # Only pays when screen_fidelity/K_eff + keep_ratio < 1.
            and opt.screen_fidelity / K_eff + opt.screen_keep_ratio < 1.0
        )
        if opt.multifidelity_screen and not _screen_ready and not getattr(opt, "_screen_warned", False):
            opt._screen_warned = True
            warnings.warn(
                "multifidelity_screen=True but the screen did not run, so no forward "
                "evaluations are saved this step. It needs a cheap screen_closure passed "
                "to step() (api.train builds one), screen_fidelity/num_probe + "
                "screen_keep_ratio < 1 (above that the screen costs more work than it "
                "saves), and none of use_quadratic_model / newton_refinement / "
                "trust_region enabled.",
                stacklevel=3,
            )

        # Budget on what a candidate allocates; chunking only splits the loop.
        if opt.chunk_size:
            chunk = opt.chunk_size
        else:
            per_candidate = max(1, X.numel())  # P * pdim
            # Any chunk can straddle a block and fall back, so cover a materialized chunk.
            if _is_subspace and not _use_fused_inplace and _factored_eval is None:
                per_candidate += opt.layout.total_params
            if _fused_inputs is not None and opt.layout.entries:
                widest = max(e.shape[0] for e in opt.layout.entries if e.shape)
                per_candidate += _fused_inputs.shape[0] * widest
            chunk = min(total_evals, max(1, (1 << 26) // per_candidate))

        _group = V * K_eff
        _site_bounds = None
        if _subspace_delta is not None or _sparse_delta is not None or _site_vmap is not None or _subspace_site:
            # Keep whole particle groups in one chunk, or the (G, C, pdim) reshape
            # misaligns.
            chunk = max(_group, (chunk // _group) * _group)
        # Break chunks where sites do, or a chunk spans entries and resolves to no site.
        if _sparse_delta is not None or _site_vmap is not None:
            _starts = [e.offset for e in opt.layout.entries]
        elif _subspace_delta is not None or _subspace_site is not None:
            _starts = [s.flat_start for s in state.subspace.specs]
        else:
            _starts = []
        if _starts:
            # Each start contributes its own group and the next, isolating the
            # straddling particle in its own chunk.
            _site_bounds = sorted(
                {min(b, total_evals) for s in _starts for b in ((s // pdim) * _group, -(-s // pdim) * _group)}
            )

        # Shape-determined and fully overwritten each step, so cached on the optimizer.
        _loss_dtype = loss_buffer_dtype(X.dtype)
        _buf_key = (total_evals, chunk, V, K_eff, X.shape[0], X.shape[1], device, X.dtype)
        _bufs = getattr(opt, "_step_buffers", None)
        if _bufs is None or _bufs[0] != _buf_key:
            _all = torch.arange(total_evals, device=device)
            # The (i, v, k) candidate list is a pure function of the key, so build it
            # with the buffers.
            _vk = _all % (V * K_eff)
            _bufs = [
                _buf_key,
                torch.empty(total_evals, dtype=_loss_dtype, device=device),
                None,
                torch.arange(chunk, device=device) if chunk <= total_evals else None,
                _all // (V * K_eff),
                _vk // K_eff,
                _vk % K_eff,
            ]
            opt._step_buffers = _bufs
        _, losses, _, _local_range_full, _i_all, _v_all, _k_all = _bufs
        _configs_wanted = chunk <= total_evals

        @_reuse_prefixes(_sparse_delta, _subspace_delta)
        def _evaluate_candidates(
            i_all,
            v_all,
            k_all,
            closure_fn,
            fused_inputs,
            fused_targets,
            out,
            sanitize=True,
            dense=False,
            probes_src=None,
            track_nonfinite=True,
        ):
            """Evaluate an explicit list of (particle, vertex, probe) candidates.

            ``dense`` marks the full list, whose particle-major grouping the delta
            paths need; a screened subset has none.
            """
            probe_src = X_probe if probes_src is None else probes_src
            n_cand = i_all.shape[0]
            # Restore only the rows the previous chunk dirtied (O(chunk*pdim) vs a full
            # copy); None means copy it all first.
            dirty_rows = None
            for chunk_start, chunk_end in _chunk_spans(n_cand, chunk, _site_bounds if dense else None):
                chunk_size_actual = chunk_end - chunk_start

                i_idx = i_all[chunk_start:chunk_end]
                v_idx = v_all[chunk_start:chunk_end]
                k_idx = k_all[chunk_start:chunk_end]

                # Sparse-delta: each candidate perturbs one contiguous run, so no
                # candidate config is built at all.
                _site = None
                if (_sparse_delta is not None or _site_vmap is not None) and fused_inputs is not None:
                    # Positions follow the particle, so gather once per group; a screened
                    # chunk has no grouping.
                    _offsets = i_idx * pdim
                    _cand = _group if dense else 1
                    _span = (
                        ((chunk_start // _group) * pdim, ((chunk_end - 1) // _group) * pdim + pdim) if dense else None
                    )
                    _owner = _sparse_delta if _sparse_delta is not None else _site_vmap
                    _site = _owner.resolve_site(_offsets, pdim, _span)
                    if _site is None and _sparse_delta is not None and _site_vmap is not None:
                        # Sparse-delta is confined to Linear layers; vmap shares the graph
                        # ahead of any entry.
                        _owner = _site_vmap
                        _site = _site_vmap.resolve_site(_offsets, pdim, _span)
                if _site is not None:
                    _local_idx = _offsets[::_cand].unsqueeze(1) + _pdim_arange - _site.offset
                    _values = probe_src[i_idx, v_idx, k_idx].reshape(chunk_size_actual // _cand, _cand, pdim)
                    _key = _site if _owner is _site_vmap else _site.key
                    out[chunk_start:chunk_end] = _owner.evaluate(
                        _base_sd, _key, _local_idx, _values, fused_inputs, fused_targets
                    ).to(out.dtype)
                    continue

                # Subspace delta: the dense list is particle-major, so the coordinate
                # span follows from arithmetic with no device read-back.
                if _subspace_delta is not None and dense and fused_inputs is not None:
                    _g0, _g1 = chunk_start // _group, chunk_end // _group
                    _spec = _subspace_delta.resolve_site(state.subspace, _g0 * pdim, _g1 * pdim, _sub_dim)
                    if _spec is not None:
                        _d = (probe_src[i_idx, v_idx, k_idx] - X[i_idx]).reshape(_g1 - _g0, _group, pdim)
                        out[chunk_start:chunk_end] = _subspace_delta.evaluate(
                            state.subspace,
                            state.hybrid_projections,
                            _bary_sd,
                            _spec,
                            _g0 * pdim - _spec.flat_start,
                            _d,
                            fused_inputs,
                            fused_targets,
                        ).to(out.dtype)
                        continue

                # Same locality in coordinate space: build only the one perturbed
                # parameter and share the rest.
                if _subspace_site is not None and dense and fused_inputs is not None:
                    _g0, _g1 = chunk_start // _group, chunk_end // _group
                    _spec = _subspace_site.resolve_spec(state.subspace, _g0 * pdim, _g1 * pdim, _sub_dim)
                    if _spec is not None:
                        # An offset from the barycentre, which _bary_sd already carries.
                        _d = (probe_src[i_idx, v_idx, k_idx] - X[i_idx]).reshape(_g1 - _g0, _group, pdim)
                        out[chunk_start:chunk_end] = _subspace_site.evaluate_subspace(
                            state.hybrid_projections,
                            _bary_sd,
                            _spec,
                            _g0 * pdim - _spec.flat_start,
                            _d,
                            fused_inputs,
                            fused_targets,
                        ).to(out.dtype)
                        continue

                if _configs_wanted and chunk_size_actual == chunk:
                    local_range = _local_range_full
                    if _bufs[2] is None:
                        # Built from X, so it needs no restore on this pass.
                        _bufs[2] = X.unsqueeze(0).expand(chunk, -1, -1).clone()
                        batch_configs = _bufs[2]
                    else:
                        batch_configs = _bufs[2]
                        if dirty_rows is None:
                            batch_configs.copy_(X)  # copy_ broadcasts
                        else:
                            batch_configs[local_range, dirty_rows] = X[dirty_rows]
                    dirty_rows = i_idx
                else:
                    batch_configs = X.unsqueeze(0).expand(chunk_size_actual, -1, -1).clone()
                    local_range = torch.arange(chunk_size_actual, device=device)
                batch_configs[local_range, i_idx] = probe_src[i_idx, v_idx, k_idx]

                flat_configs = batch_configs.reshape(chunk_size_actual, -1)

                if _use_fused_inplace and _is_subspace and opt._hybrid:
                    # One config at a time via in-place weight swap, never the full
                    # (N, *param_shape) stack.
                    flat_sub = flat_configs[:, :_sub_dim]
                    chunk_losses = opt._cost_evaluator.evaluate_subspace_inplace(
                        state.subspace,
                        state.hybrid_projections,
                        state.base_params,
                        flat_sub,
                        fused_inputs,
                        fused_targets,
                    )
                elif _factored_eval is not None:
                    chunk_losses = _factored_eval.evaluate(
                        state.subspace,
                        state.hybrid_projections,
                        state.base_params,
                        flat_configs[:, :_sub_dim],
                        fused_inputs,
                        fused_targets,
                    )
                elif _is_subspace:
                    flat_sub = flat_configs[:, :_sub_dim]
                    if opt._mixed_precision and state.projection is not None:
                        flat_sub = flat_sub.to(dtype=state.projection.dtype)
                    if opt._adaptive:
                        chunk_params = state.subspace.reconstruct_batch(
                            opt._sampling_projection,
                            state.base_params,
                            flat_sub,
                        )
                    elif opt._per_layer_projections:
                        chunk_params = state.subspace.reconstruct_batch(
                            state.hybrid_projections,
                            state.base_params,
                            flat_sub,
                        )
                    else:
                        chunk_params = state.subspace.reconstruct_batch(
                            state.base_params,
                            flat_sub,
                        )
                    chunk_losses = closure_fn(chunk_params)
                else:
                    layout_flat = opt.layout.padded_size
                    if flat_configs.shape[1] >= layout_flat:
                        flat_for_layout = flat_configs[:, :layout_flat]
                    else:
                        flat_for_layout = torch.nn.functional.pad(
                            flat_configs,
                            (0, layout_flat - flat_configs.shape[1]),
                        )
                    chunk_params = opt.layout.batch_unflatten(flat_for_layout)
                    chunk_losses = closure_fn(chunk_params)

                if chunk_losses.shape != (chunk_end - chunk_start,):
                    raise ValueError(
                        f"the closure must return one loss per candidate, shape "
                        f"({chunk_end - chunk_start},), got {tuple(chunk_losses.shape)}."
                    )
                out[chunk_start:chunk_end] = chunk_losses.to(out.dtype)

            # Over every candidate at once, not per chunk: the 2*max|finite|+1 penalty
            # is relative to what it is handed, so per-chunk could rank an infeasible
            # vertex above a legal one from another chunk.
            # AND across calls, and outside the sanitize gate: the screened path fills
            # the cost matrix in two passes and the promoted one does not sanitize, so
            # gating on sanitize would leave the flag reporting the screen alone.
            # Centre evaluations are not part of the cost matrix and opt out.
            if track_nonfinite:
                seen = opt._all_nonfinite
                flag = ~torch.isfinite(out).any()
                opt._all_nonfinite = flag if seen is None else seen & flag
            if sanitize:
                out.copy_(sanitize_cost(out))
            return out

        if not _screen_ready:
            _evaluate_candidates(_i_all, _v_all, _k_all, closure, _fused_inputs, _fused_targets, losses, dense=True)
        else:
            # Stage 1: rank every direction on the cheap fidelity, one probe scale.
            k_center = K_eff // 2
            _center = _k_all == k_center
            _i_center = _i_all[_center]
            screen_flat = _evaluate_candidates(
                _i_center,
                _v_all[_center],
                torch.full_like(_i_center, k_center),
                screen_closure,
                screen_inputs,
                screen_targets,
                X_probe.new_empty(P_active * V, dtype=_loss_dtype),
            )
            screen_cost = screen_flat.reshape(P_active, V)

            # Stage 2: full fidelity only where it can still matter, ranked per particle.
            keep_mask = torch.zeros(P_active, V, dtype=torch.bool, device=device)
            if _selection_solver:
                # A floor at k, not a cap: dropped vertices sanitize back to a finite
                # penalty, so TopKMeanSolver would average ones never scored.
                keep_v = min(V, max(1, int(round(V * opt.screen_keep_ratio)), min(getattr(opt.solver, "k", 1), V)))
                keep_idx = torch.topk(screen_cost, keep_v, dim=1, largest=False).indices
                keep_mask.scatter_(1, keep_idx, True)
            elif opt.polytope_type == "orthoplex":
                # A weighted mean is moved by the contrast between antithetic partners,
                # so rank by that and keep both signs.
                pdim_local = V // 2
                contrast = (screen_cost[:, :pdim_local] - screen_cost[:, pdim_local:]).abs()
                keep_k = max(1, min(pdim_local, int(round(pdim_local * opt.screen_keep_ratio))))
                keep_dirs = torch.topk(contrast, keep_k, dim=1).indices  # (P_active, keep_k)
                keep_mask.scatter_(1, keep_dirs, True)
                keep_mask.scatter_(1, keep_dirs + pdim_local, True)
            else:
                # sum_v v = 0, so a vertex at the row mean carries weight 1/V and moves the
                # barycentre by nothing. Deviation from the mean is the contrast above,
                # written for a frame with no partner to subtract.
                deviation = (screen_cost - screen_cost.mean(dim=1, keepdim=True)).abs()
                keep_v = max(1, min(V, int(round(V * opt.screen_keep_ratio))))
                keep_mask.scatter_(1, torch.topk(deviation, keep_v, dim=1).indices, True)

            sel_idx = torch.nonzero(keep_mask[_i_all, _v_all], as_tuple=True)[0]
            kept = _evaluate_candidates(
                _i_all[sel_idx],
                _v_all[sel_idx],
                _k_all[sel_idx],
                closure,
                _fused_inputs,
                _fused_targets,
                X_probe.new_empty(sel_idx.numel(), dtype=_loss_dtype),
                sanitize=False,
            )

            # Offset the dropped entries by the per-particle difference on the kept ones,
            # or the solve ranks by fidelity rather than by cost.
            losses = _fill_screened_losses(
                screen_cost, kept, sel_idx, keep_mask, P_active, V, K_eff, mask_dropped=_selection_solver
            )
            # One penalty over the merged population, or the two fidelities get different
            # 2*max|finite|+1 substitutes and compare wrong.
            losses = sanitize_cost(losses)
            # Screen forwards see part of the data, so weight them by the realized slice
            # (or the configured fraction when there is none).
            _fidelity = (
                screen_inputs.shape[0] / _fused_inputs.shape[0]
                if screen_inputs is not None and _fused_inputs is not None
                else opt.screen_fidelity
            )
            _evals_this_step = sel_idx.numel() + _fidelity * P_active * V
            opt._last_screen_savings = 1.0 - _evals_this_step / max(1, total_evals)
            logger.debug(
                "multifidelity_screen: %d/%d full-fidelity forwards + %d screen forwards",
                sel_idx.numel(),
                total_evals,
                P_active * V,
            )

        losses_3d_full = losses.reshape(P, V, K_eff)
        cost_matrix = losses.reshape(P, V) if K_eff == 1 else losses_3d_full.mean(dim=-1)
        opt._center_loss = None
        if opt.use_quadratic_model:
            # The step buffer is overwritten next time; the model owns a snapshot.
            opt._losses_3d = losses_3d_full.detach().clone()
        if _center_wanted:
            # Every centre candidate equals X, so evaluate once, not once per block.
            _one = torch.zeros(1, dtype=torch.long, device=device)
            _center = _evaluate_candidates(
                _one,
                _one,
                _one,
                closure,
                _fused_inputs,
                _fused_targets,
                X.new_empty(1, dtype=_loss_dtype),
                probes_src=X.reshape(P, 1, 1, -1),
                sanitize=False,
                track_nonfinite=False,
            ).detach()
            opt._center_loss = _center.expand(P).clone()
            _evals_this_step += 1

    # Charged after the quadratic model's centre evaluations, which also add to
    # _evals_this_step and would otherwise be undercounted here.
    opt.candidate_evals += int(_evals_this_step)

    # No sanitize here: `losses` was sanitized per chunk, a mean of finite values is
    # finite, and the reuse path holds an already-sanitized matrix.
    raw_cost_matrix = cost_matrix

    # Deferred: the matrix is measured at the position the previous step produced, so
    # comparing it against that step's prediction gives a real ratio. Both ends must
    # use the same estimator, so a step with a centre and one without never compare.
    _center_now = opt._center_loss is not None
    if (
        opt.trust_region
        # A reuse step re-reports the cached loss, so the ratio would be 0 by construction.
        and not _can_reuse
        and opt._prev_predicted_improvement is not None
        and opt._prev_pre_step_loss is not None
        and _center_now
        and opt._prev_loss_from_center
    ):
        current_loss = opt._center_loss[0].item()
        # Negative means loss decreased.
        actual_improvement = torch.tensor([current_loss - opt._prev_pre_step_loss])
        from .quadratic_model import update_trust_region

        opt._trust_region_multiplier = update_trust_region(
            opt._prev_predicted_improvement.sum(),
            actual_improvement,
            opt._trust_region_multiplier,
            min_radius=0.1,
            max_radius=3.0,
        )
        state.trust_region_multipliers.append(opt._trust_region_multiplier)
        opt._prev_predicted_improvement = None
        opt._prev_pre_step_loss = None

    ent_eps = opt._get_ent_epsilon(iteration)
    ot_epsilon = ent_eps if ent_eps is not None else current_eps

    # A >2x epsilon jump invalidates the dual-momentum history: the prev values were
    # measured at the old epsilon and extrapolate to a bad warm start.
    if state.last_solve_eps is not None and (
        ot_epsilon / state.last_solve_eps > 2.0 or state.last_solve_eps / ot_epsilon > 2.0
    ):
        state.prev_prev_f = None
        state.prev_prev_g = None

    init_f_for_solve = state.f
    init_g_for_solve = state.g
    if (
        opt._dual_momentum_beta > 0.0
        and state.f is not None
        and state.prev_prev_f is not None
        and state.prev_prev_g is not None
    ):
        beta = opt._dual_momentum_beta
        init_f_for_solve = state.f + beta * (state.f - state.prev_prev_f)
        init_g_for_solve = state.g + beta * (state.g - state.prev_prev_g)
        # No clamp: SinkhornSolver bounds any warm start by 10*max|C_scaled|, and an
        # epsilon-scaled bound would truncate valid extrapolations.

    opt.solver.epsilon = ot_epsilon
    if opt._use_fused_softmax:
        # Scale outside the kernel so every scale_cost mode matches the non-fused solvers.
        scaled_cost = scale_cost_matrix(recenter_cost(cost_matrix)[0], opt.scale_cost)
        X_new_fused, transport_matrix = opt._compiled.fused_softmax_project(
            scaled_cost,
            ot_epsilon,
            state.a,
            opt._polytope_vertices,
            rot_mats,
            step_r,
            X,
            scale_cost_mean=False,
        )
        # state.costs tracks cost_matrix.mean, so the OT cost is unused here.
        ot_result = SolverResult(
            matrix=transport_matrix,
            cost=0.0,
            f=None,
            g=None,
            converged=True,
            n_iters=1,
            ent_reg_cost=0.0,
        )
    else:
        # prev_epsilon lets the solver rescale warm-started duals across an epsilon change.
        solve_kwargs = dict(
            cost_matrix=cost_matrix,
            init_f=init_f_for_solve,
            init_g=init_g_for_solve,
            scale_cost=opt.scale_cost,
        )
        ot_result = opt.solver.solve(**solve_kwargs)
    state.last_solve_eps = ot_epsilon

    feed_solver_stats(opt._progressive_epsilon, opt.solver, ot_result.n_iters, ot_result.converged)

    # Pre-step subspace coords for the displacement history and CMA evolution paths.
    _pre_step_sub_coords = None

    # CMA uses the same displacement/rotate path, so it needs these with both flags off.
    if opt._adaptive or opt._per_layer_projections:
        sub_dim = opt.subspace.subspace_dim
        _pre_step_sub_coords = state.X.reshape(-1)[:sub_dim].clone()

    # The finite-difference model feeds biased_rotation and trust_region, so it is
    # built whenever either is active.
    _center_loss = getattr(opt, "_center_loss", None)
    if _center_loss is not None and _center_loss.shape != (P,):
        _center_loss = None
    # The gradient is a closed form on any centred tight frame. Curvature needs either a
    # shared f(X) or a K>=2 regression, and the regression is per coordinate, so it needs
    # the antipodal pairs only the orthoplex has.
    # Not V == 2*pdim: a 2D cube has 4 vertices too, and its halves are not antipodal.
    _antithetic = opt.polytope_type == "orthoplex"
    _quad_ready = (
        opt.use_quadratic_model
        and opt._losses_3d is not None
        and opt._losses_3d.shape == (P, V, K_eff)
        and (_center_loss is None or bool(torch.isfinite(_center_loss).all()))
        and (_center_loss is not None or (K_eff >= 2 and _antithetic))
    )
    _fd_grad = _fd_hess = None
    if not _quad_ready:
        # Stale otherwise: adaptive_num_probe can drop K_eff to 1 mid-run.
        opt._newton_direction = None
    if (opt.biased_rotation or opt.trust_region) and _quad_ready:
        from .quadratic_model import (
            compute_newton_step,
            extract_fd_gradient,
            extract_fd_hessian_diag,
            extract_fd_hessian_diag_centered,
            extract_iso_curvature,
        )

        _fd_grad = extract_fd_gradient(opt._losses_3d, probes, probe_r, pdim, None if _antithetic else polytope_verts)
        if _center_loss is None:
            _fd_hess = extract_fd_hessian_diag(opt._losses_3d, probes, probe_r, pdim)  # (P, pdim)
        elif _antithetic:
            _fd_hess = extract_fd_hessian_diag_centered(opt._losses_3d, probes, probe_r, pdim, _center_loss)
        else:
            # No antipodal pairs, so the frame measures the trace instead of the diagonal.
            _fd_hess = extract_iso_curvature(opt._losses_3d, _center_loss, probes, probe_r)  # (P, 1)

        if opt.biased_rotation:
            # Descent direction = negative gradient in original space (rot_mats @ .).
            newton_rot = compute_newton_step(_fd_grad, _fd_hess, max_step_norm=step_r, hessian_reg=1e-4)
            grad_orig = torch.einsum("bij,bj->bi", rot_mats, _fd_grad)  # (P, pdim)
            opt._prev_descent_direction = -grad_orig.detach()
            opt._prev_descent_direction_finite = bool(torch.isfinite(grad_orig).all())
            newton_orig = torch.einsum("bij,bj->bi", rot_mats, newton_rot)
            opt._newton_direction = newton_orig.detach()
    elif opt.biased_rotation:
        # OT descent direction when the quadratic model is unavailable, so warn rather
        # than let the caller assume the finite-difference path is running.
        if not getattr(opt, "_biased_rotation_fallback_warned", False):
            opt._biased_rotation_fallback_warned = True
            warnings.warn(
                "biased_rotation is aligning the chart to the OT barycenter's displacement, "
                "not to a finite-difference descent/Newton direction. The FD path needs "
                "use_quadratic_model=True; num_probe=1 is enough on any polytope, the step "
                "then evaluates one shared f(X) for the curvature.",
                stacklevel=3,
            )
        transport_weights = ot_result.matrix  # (P, V)
        vertex_offsets = X_vertices - X.unsqueeze(1)  # (P, V, pdim)
        weight_sums = transport_weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
        normalized_weights = transport_weights / weight_sums  # (P, V)
        weighted_dir = (normalized_weights.unsqueeze(-1) * vertex_offsets).sum(dim=1)  # (P, pdim)
        opt._prev_descent_direction = weighted_dir.detach()
        opt._prev_descent_direction_finite = bool(torch.isfinite(weighted_dir).all())

    if opt._use_fused_softmax:
        X_bary = X_new_fused
    else:
        X_bary = opt._compiled.barycentric_projection(
            ot_result.matrix,
            X_vertices,
        )

    record_saturation(opt, state, ot_result.matrix, step_r)

    # Jitter per particle so at least one block's displacement has a density along any
    # wall normal; a shared radius gives one chance where this gives P.
    X_bary = opt._apply_particle_step_jitter(X, X_bary)

    # Reduced from tensors already in hand; syncs with the cost mean below.
    _ess_tensor, _rho_tensor = solver_health(ot_result.matrix, X_bary - X, step_r)

    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(
            iteration,
            opt.max_iterations,
            opt.momentum_init,
            opt.momentum_final,
        )
        X_new, vel_new = apply_momentum(
            X,
            X_bary,
            state.velocity,
            beta,
            opt.velocity_lr,
        )
        state.velocity = vel_new
        state.X = X_new
    else:
        state.X = X_bary

    # Post-OT correction; tracks whether the refinement makes the solve's duals stale.
    _duals_invalidated = False
    if opt._newton_refinement and opt._losses_3d is not None and K_eff >= 2 and opt.polytope_type == "orthoplex":
        from .quadratic_model import apply_newton_refinement

        X_refined = apply_newton_refinement(
            # X_bary, not state.X: the accept test scores the pure OT step, and the
            # post-momentum position would double-count momentum.
            X_bary=X_bary,
            losses_3d=opt._losses_3d,
            scales=probes,
            probe_radius=probe_r,
            pdim=pdim,
            rot_mats=rot_mats,
            X_current=X,
            alpha=opt._newton_refinement_alpha,
            max_step_norm=step_r * 0.5,
            hessian_reg=1e-4,
        )
        if torch.isfinite(X_refined).all():
            state.X = X_refined
            # Refinement replaced the post-momentum position, so re-derive the velocity from
            # the move that actually happened. apply_momentum's identity is X = X_old + lr*v.
            if opt.use_momentum and state.velocity is not None:
                state.velocity = (
                    (X_refined - X) / opt.velocity_lr if opt.velocity_lr != 0 else torch.zeros_like(state.velocity)
                )
            # The duals now encode old positions; mark stale so the save below skips them.
            _duals_invalidated = True

    # state.X is final here, so the model scores the realized move, not a Newton step
    # that was never applied.
    if opt.trust_region and _fd_grad is not None and _center_loss is not None:
        from .quadratic_model import compute_predicted_improvement

        realized_rot = torch.einsum("bij,bj->bi", rot_mats.transpose(-1, -2), state.X - X)
        # Rows are disjoint parameter blocks of one objective, not independent losses.
        opt._prev_predicted_improvement = compute_predicted_improvement(_fd_grad, _fd_hess, realized_rot).sum().detach()
        opt._prev_loss_from_center = True
        opt._prev_pre_step_loss = _center_loss[0].item()

    if opt.use_covariance_adaptation:
        cma_sub = opt.subspace  # CMAAdaptiveSubspace
        sub_dim = cma_sub.subspace_dim

        # From X_bary not state.X: the paths and rank-mu must describe the same
        # offspring distribution, without the momentum/Newton correction.
        post_step_coords = X_bary.reshape(-1)[:sub_dim]
        raw_displacement = post_step_coords - _pre_step_sub_coords

        # Transport row normalized to sum to 1 per particle.
        transport = ot_result.matrix  # (P, V)
        recomb = transport / transport.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # step_r, not sigma: dividing by sigma alone leaves a (step_radius*epsilon)^2
        # factor in every second moment.
        step_scale = max(abs(step_r), 1e-12)

        # Direction only: with unit-norm innovations the stationary ||p_sigma|| is 1 in
        # any dimension, so the ratio is pure agreement.
        z_displacement = raw_displacement / step_scale
        z_norm = torch.linalg.vector_norm(z_displacement)
        # Below this the step is rounding, and normalising feeds the paths noise.
        if bool(z_norm > 1e-12):
            z_direction = z_displacement / z_norm
            # p_sigma lives on whitened z, which z_direction already is, so no C round-trip.
            sqrt_C = torch.sqrt(torch.clamp(state.C_diag, min=opt._cma_params["cov_min"]))
            y_displacement = sqrt_C * z_direction

            state.p_sigma = update_evolution_path_sigma(
                p_sigma=state.p_sigma,
                displacement=z_direction,
                C_diag=None,
                c_sigma=opt._cma_params["c_sigma"],
                mu_eff=opt._cma_params["mu_eff"],
            )
        else:
            y_displacement = torch.zeros_like(state.C_diag)
            state.p_sigma = (1.0 - opt._cma_params["c_sigma"]) * state.p_sigma

        # Stall test against the same unit reference.
        p_sigma_norm = torch.norm(state.p_sigma).item()
        h_sigma = compute_heaviside_sigma(
            p_sigma_norm=p_sigma_norm,
            expected_norm=1.0,
            n=sub_dim,
            c_sigma=opt._cma_params["c_sigma"],
            generation=state.generation,
        )

        state.p_c = update_evolution_path_c(
            p_c=state.p_c,
            displacement=y_displacement,
            h_sigma=h_sigma,
            c_c=opt._cma_params["c_c"],
            mu_eff=opt._cma_params["mu_eff"],
        )

        if opt.use_covariance_adaptation:
            # c_mu is 0 at mu_eff = 1, so rank-mu would be multiplied by zero.
            _c_mu = opt._cma_params["c_mu"]
            if _c_mu > 0.0:
                # Offspring are the vertices weighted by transport mass; the pdim factor
                # keeps the update trace-preserving.
                vertex_steps = (X_vertices - X.unsqueeze(1)) / step_scale  # (P, V, pdim)
                rank_mu_local = (recomb.unsqueeze(-1) * vertex_steps**2).sum(dim=1)  # (P, pdim)
                # rank-mu is sum_v w_v * y_v^2 = C_diag * sum_v w_v * z_v^2.
                rank_mu = state.C_diag * pdim * rank_mu_local.reshape(-1)[:sub_dim]
            else:
                rank_mu = torch.zeros_like(state.C_diag)

            state.C_diag = update_covariance_diagonal(
                C_diag=state.C_diag,
                p_c=state.p_c,
                rank_mu=rank_mu,
                c_1=opt._cma_params["c_1"],
                c_mu=_c_mu,
                h_sigma=h_sigma,
                c_c=opt._cma_params["c_c"],
                # p_c is unwhitened, so this scale sets rank-one strength only; the
                # renormalisation below absorbs the trace mismatch.
                trace_scale=float(sub_dim),
                cov_min=opt._cma_params["cov_min"],
                cov_max=opt._cma_params["cov_max"],
                # C carries shape, the step radius carries scale.
                trace=float(sub_dim),
            )

        state.generation += 1

    # One transfer for all three, not three device syncs a step.
    _nonfinite = opt._all_nonfinite
    _cost_mean, _ess, _rho, _all_nonfinite, _progress = torch.stack(
        [
            cost_matrix.mean(),
            _ess_tensor,
            _rho_tensor,
            _nonfinite.to(cost_matrix.dtype) if _nonfinite is not None else cost_matrix.new_zeros(()),
            # Mean of each particle's best vertex; the full-matrix mean would partly
            # measure the radius controller's own last move.
            cost_matrix.min(dim=1).values.mean(),
        ]
    ).tolist()
    if _all_nonfinite:
        warn_all_nonfinite(opt)
    # Candidate evaluations this step, net of probe reuse and screening.
    state.record_solver_health(_ess, _rho, _evals_this_step)
    # A reuse step re-reports the cached cost, so the stagnation counter would climb on
    # a step that measured nothing.
    if not _can_reuse:
        update_stagnation_and_radius(opt, state, _progress)

    _nan_reverted = False
    if not torch.isfinite(state.X).all():
        # Revert to pre-step X if it was finite, else to the coordinate origin.
        state.X = X.clone() if torch.isfinite(X).all() else torch.zeros_like(X)
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # The CMA block consumed the same non-finite X, so reset its state too or
        # sqrt(C_diag) stays poisoned and the run cannot recover.
        if opt.use_covariance_adaptation:
            if state.p_c is not None:
                state.p_c = torch.zeros_like(state.p_c)
            if state.p_sigma is not None:
                state.p_sigma = torch.zeros_like(state.p_sigma)
            if state.C_diag is not None:
                state.C_diag = torch.ones_like(state.C_diag)
        _nan_reverted = True

    if _nan_reverted and opt.biased_rotation:
        opt._prev_descent_direction = None
        opt._prev_descent_direction_finite = False
    if _nan_reverted:
        opt._newton_direction = None

    # From X_bary, not state.X: the coasting direction must not re-carry momentum/Newton.
    update_amortized_direction(opt, (X_bary - X).detach(), _nan_reverted)

    # Trust-region update is deferred to the next step's pre-OT measurement.

    per_particle_disp_sqnorms = torch.sum((state.X - X) ** 2, dim=-1)  # (P,)
    disp_sqnorm_tensor = torch.mean(per_particle_disp_sqnorms)
    state.costs.append(_cost_mean)
    state.linear_convergence.append(ot_result.converged)
    state.displacement_sqnorms.append(disp_sqnorm_tensor.item())
    state.iteration_count += 1
    opt._ot_step_costs.append(_cost_mean)

    # Cache the matrix with the configuration it was measured at; a reuse step leaves
    # the anchor where it is.
    if opt._adaptive_probes and not _can_reuse:
        opt._prev_X = X.detach().clone()
        opt._prev_cost_matrix = raw_cost_matrix.detach()
        opt._prev_rot_mats = rot_mats.detach()
        opt._prev_k_eff = K_eff
        opt._prev_step_r = step_r
        opt._prev_probe_r = probe_r
    # Save duals for warm-starting; reset if NaN reverted or refinement moved particles.
    if _nan_reverted or _duals_invalidated:
        state.f = None
        state.g = None
        if opt._dual_momentum_beta > 0.0:
            state.prev_prev_f = None
            state.prev_prev_g = None
    else:
        if opt._dual_momentum_beta > 0.0:
            state.prev_prev_f = state.f.clone() if state.f is not None else None
            state.prev_prev_g = state.g.clone() if state.g is not None else None
        state.f = ot_result.f.detach() if ot_result.f is not None else None
        state.g = ot_result.g.detach() if ot_result.g is not None else None
    state.epsilon = current_eps

    # Displacement tracking, absorb, and rotation. CMAAdaptiveSubspace composes (not
    # inherits) AdaptiveSubspace, so it must be tested separately.
    if opt._adaptive:
        adaptive_sub = opt.subspace
        # Use the cached covariance-scaled projection, or the sqrt(C_diag) factor is
        # dropped and the weights move.
        proj_used = opt._sampling_projection if opt._sampling_projection is not None else state.projection

        record_displacement(state, adaptive_sub, _pre_step_sub_coords, proj_used)

        should_absorb = adaptive_sub.should_absorb(
            state.stagnation_count,
            state.iteration_count,  # already incremented above
        )

        if should_absorb:
            full_flat_sub = state.X.reshape(-1)[: adaptive_sub.subspace_dim]
            new_base, _zeroed = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                full_flat_sub,
            )
            state.base_params = new_base
            # Zero rather than re-project: after a random redraw the bases are largely
            # uncorrelated, and the next OT solve finds a fresh descent direction.
            state.X = torch.zeros_like(state.X)
            from .projection import SparseRandomProjection

            if isinstance(state.projection, SparseRandomProjection):
                new_seed = state.projection.seed + state.absorb_count + 1000
                state.projection = SparseRandomProjection(
                    full_dim=state.projection.full_dim,
                    subspace_dim=state.projection.subspace_dim,
                    seed=new_seed,
                )
            else:
                state.projection = adaptive_sub.init_projection(
                    generator=opt._generator,
                    device=state.X.device,
                    dtype=state.X.dtype,
                )
            state.displacement_history.zero_()
            if state.displacement_history_full is not None:
                state.displacement_history_full.zero_()
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            invalidate_for_basis_change(opt, state)
            state.absorb_count += 1
            # Clear the stagnation counter, or a plateau redraws the basis every step;
            # the absorb re-anchors the origin, so the old loss history no longer applies.
            state.stagnation_count = 0
            state.prev_loss = _cost_mean
            if opt.use_covariance_adaptation:
                state.p_c = torch.zeros_like(state.p_c)
                state.p_sigma = torch.zeros_like(state.p_sigma)
                state.C_diag = torch.ones_like(state.C_diag)
                # Keep the generation counter to preserve cumulation history.
        elif opt.use_covariance_adaptation:
            # C_diag and the paths index axes of THIS basis, and a diagonal covariance
            # does not stay diagonal under rotation, so hold the basis between absorbs.
            pass
        elif _rotation_due(adaptive_sub, state.iteration_count):
            # Re-anchor first: the point is base + P @ coords, so swapping P at non-zero
            # coords moves the weights unevaluated. The basis QR/SVD runs in fp32.
            state.base_params, _ = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                state.X.reshape(-1)[: adaptive_sub.subspace_dim],
            )
            state.X = torch.zeros_like(state.X)

            from .projection import SparseRandomProjection

            if isinstance(state.projection, SparseRandomProjection):
                new_seed = state.projection.seed + state.iteration_count
                state.projection = SparseRandomProjection(
                    full_dim=state.projection.full_dim,
                    subspace_dim=state.projection.subspace_dim,
                    seed=new_seed,
                )
            else:
                # Dense projection: pass the full-space history so each entry keeps the
                # frame it was measured in.
                hist = (
                    state.displacement_history_full[: state.displacement_history_count]
                    if state.displacement_history_count > 0 and state.displacement_history_full is not None
                    else None
                )

                state.projection = adaptive_sub.rotate(
                    state.projection,
                    step=state.iteration_count,
                    total_steps=opt.max_iterations,
                    displacement_history=hist,
                    generator=opt._generator,
                    history_is_full=True,
                )
            invalidate_for_basis_change(opt, state)

    maintain_per_layer_subspace(opt, state, _cost_mean, _pre_step_sub_coords)

    # Against the applied rank, not subspace_dim: the latter sums coordinate counts, so
    # comparing them would rebuild the subspace from scratch every step.
    if opt._rank_schedule is not None and opt.subspace is not None:
        current_rank = opt._rank_schedule.at(state.iteration_count)
        if current_rank != opt._applied_rank:
            opt._transition_rank(current_rank)

    opt._sync_model()

    return _cost_mean
