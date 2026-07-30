"""Monolithic step: a single OT solve over all particles.

Called from ``PolyStepOptimizer.step()``.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Callable

import torch

from .costs import scale_cost_matrix
from .solvers._shared import loss_buffer_dtype, recenter_cost, sanitize_cost, solver_health
from .epsilon import feed_solver_stats
from ._step_core import invalidate_for_basis_change
from .dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
    update_radius_multiplier,
    update_stagnation,
)
from .geometry import apply_biased_rotation, get_random_rotation_matrices
from .solvers import SinkhornSolver
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

    Kept vertices carry their full-fidelity values. ``keep_mask`` is (P_active, V): each
    particle keeps its own vertex set.

    What happens to the dropped ones depends on what the solver does with the cost matrix.
    A weighted mean reads every entry, so a dropped vertex must carry a usable value: its
    cheap-fidelity value plus a per-particle offset, estimated as the mean difference
    between the two fidelities on that particle's kept vertices. A SELECTION solver
    (argmin, top-k) instead reads only the winner, and an imputed value can win, sending
    the step to a vertex that was never scored at full fidelity. For those solvers the
    dropped entries are made ineligible instead (``mask_dropped=True``), which is the same
    ``+inf`` convention ``sanitize_cost`` already uses for a failed evaluation.
    """
    losses = screen_cost.new_empty(P_active * V * K_eff)
    losses[sel_idx] = kept

    full_3d = losses.reshape(P_active, V, K_eff)
    mask_3d = keep_mask.unsqueeze(-1)
    if mask_dropped:
        return torch.where(mask_3d, full_3d, torch.full_like(full_3d, float("inf"))).reshape(-1)
    # Dropped positions hold uninitialized memory, so zero them before summing
    # rather than weighting them: 0 * inf is NaN.
    full_kept = torch.where(mask_3d, full_3d, torch.zeros_like(full_3d))
    n_kept = keep_mask.sum(dim=1, keepdim=True).to(screen_cost.dtype).clamp(min=1)
    full_mean = full_kept.sum(dim=(1, 2)).unsqueeze(1) / (n_kept * K_eff)
    screen_mean = (screen_cost * keep_mask).sum(dim=1, keepdim=True) / n_kept
    calibrated = (screen_cost + (full_mean - screen_mean)).unsqueeze(-1).expand(P_active, V, K_eff)

    return torch.where(mask_3d, full_3d, calibrated).reshape(-1)


def _chunk_spans(n_cand: int, chunk: int, bounds=None):
    """``(start, end)`` chunks of at most ``chunk``, also breaking at ``bounds``.

    ``bounds`` are candidate indices where a new parameter entry begins. Cutting there
    keeps every candidate in a chunk inside one entry, which is what lets the
    site-aware paths resolve a site instead of falling back.
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
    """Whether the basis rotates on this step.

    The default of 1 rotates every step, matching what the OT solve extracts. Raising
    it amortizes the absorb plus QR/SVD, which dominate at large ``full_dim``.
    """
    interval = getattr(adaptive_sub, "rotation_interval", 1)
    if interval <= 0:
        return False
    return iteration % interval == 0


def record_displacement(state, adaptive_sub, pre_step_sub_coords, proj_used) -> None:
    """Append this step's subspace displacement to the rolling history.

    Stores the full-space image under the current basis: the basis rotates every
    step, so mapping old coordinates through the latest one misattributes them.
    """
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

    Shared with the blockwise step, which otherwise never rotates or absorbs.
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
        # init_projections seeds step=0, so every absorb returns the same basis and the
        # run stays in one affine subspace. Redrawing instead loses accuracy on SNNs.
        # absorb_aligned_active biases the new basis toward the window's productive
        # directions by displacement-SVD.
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
        # Otherwise the fused matrix still holds the pre-absorb basis and candidates get
        # scored in the old one while _sync_model writes the new. Skipped when the basis
        # came back by identity, since the fused matrix is a function of it alone.
        if state.hybrid_projections is not _basis_before and hasattr(hybrid_sub, "build_fused_projection"):
            hybrid_sub.build_fused_projection(state.hybrid_projections)
        state.displacement_history.zero_()
        if state.displacement_history_full is not None:
            state.displacement_history_full.zero_()
        state.displacement_history_idx = 0
        state.displacement_history_count = 0
        invalidate_for_basis_change(opt, state)
        state.absorb_count += 1
        # Else absorb_mode='stagnation' stays triggered on a plateau and redraws every
        # step; the absorb re-anchors the origin, so the old loss history is void.
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
        # Only act when projections changed (a rotation happened). With the
        # default rotation_interval=0 rotate_all returns the same dict, so the
        # block_diag rebuild below is skipped: it is a full, wasteful rebuild.
        if new_projections is not state.hybrid_projections:
            # Re-anchor first: the point is base + P @ coords, so swapping P at non-zero
            # coords moves the weights with nothing evaluated behind it.
            state.base_params, _ = hybrid_sub.absorb(
                state.hybrid_projections,
                state.base_params,
                state.X.reshape(-1)[: hybrid_sub.subspace_dim],
            )
            state.X = torch.zeros_like(state.X)
            # Rows are coordinates under the basis being replaced, so the next rotation
            # would read them through the new projections and learn phantom directions.
            state.displacement_history.zero_()
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            invalidate_for_basis_change(opt, state)
            state.hybrid_projections = new_projections
            if hasattr(hybrid_sub, "build_fused_projection"):
                hybrid_sub.build_fused_projection(new_projections)
    # An absorb re-anchors the origin even when init_projections hands back the same
    # basis by identity, so identity alone would under-report it and the blockwise
    # caller would keep duals warm-started against the pre-absorb origin.
    return should_absorb or state.hybrid_projections is not _basis_before


def step_monolithic(opt, closure: Callable, screen_closure: Callable | None = None) -> float:
    """Monolithic step: single OT solve over all particles.

    Multi-particle architecture: X is (P, particle_dim) where P is
    num_particles and particle_dim is typically 2. For each particle i,
    polytope vertices are sampled in particle_dim space. The cost for
    entry (i, v, k) is evaluated by constructing the full model config
    with particle i replaced by the probe position. The OT problem is
    (P, V) which is tractable.
    """
    state = opt._state
    X = state.X  # (P, particle_dim)
    iteration = state.iteration_count
    device = X.device

    # Cache the coord->param projection for this step (covariance-scaled for CMA).
    opt._update_sampling_projection()

    current_eps = opt._get_epsilon(iteration)
    if opt.use_adaptive_radius:
        radius_mult = state.radius_multiplier
    else:
        radius_mult = 1.0
    # Scheduled radii: if step_radius/probe_radius has .at(), the schedule
    # handles annealing (no epsilon multiplication). Float values use the
    # original behavior (radius * eps * radius_mult).
    _sr = opt._get_step_radius(iteration)
    _pr = opt._get_probe_radius(iteration)
    _sr_scheduled = hasattr(opt.step_radius, "at")
    _pr_scheduled = hasattr(opt.probe_radius, "at")
    if opt.trust_region:
        step_r = _sr * opt._trust_region_multiplier * (1.0 if _sr_scheduled else current_eps) * radius_mult
    else:
        step_r = _sr * (1.0 if _sr_scheduled else current_eps) * radius_mult
    probe_r = _pr * (1.0 if _pr_scheduled else current_eps) * radius_mult

    # Probe-radius jitter (Thm. 4.2 condition (iv); no-op when probe_radius_jitter == 0).
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
        # Reduce K once the last 3 OT-step costs are strictly decreasing. Finiteness,
        # not positivity: an objective that returns negative values (RL returns, margin
        # losses) is descending just as much, and gating on c > 0 excluded it entirely.
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

    # All or nothing on X: a candidate is X with one row replaced, so any particle
    # moving invalidates every row. _prev_X is where the cached costs were measured, not
    # the last position, so slow drift cannot reset the budget.
    _can_reuse = (
        opt._adaptive_probes
        and opt._prev_cost_matrix is not None
        and opt._prev_cost_matrix.shape == (P, V)
        and opt._prev_rot_mats is not None
        and opt._prev_rot_mats.shape == (P, pdim, pdim)
        and opt._prev_X is not None
        and opt._prev_X.shape == X.shape
        and opt._prev_k_eff == K_eff
        and opt._prev_step_r == step_r
        # Costs are measured at probe_r, so probe_r must match for a cached row to be
        # comparable with a fresh one. A scheduled probe_radius or probe_radius_jitter > 0
        # moves it independently of step_r.
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

    # For each (i, v, k), construct a full (P, pdim) config with row i
    # replaced by X_probe[i, v, k]. Then unflatten to params and evaluate.

    # The configuration has not moved since the cached matrix was measured, so every
    # row still describes its vertices. Zero forwards this step.
    _evals_this_step = 0
    if _can_reuse:
        # An alias, not a copy: no consumer writes into the cost matrix. Every downstream
        # operation (recenter, scale, prepare_cost) allocates a fresh tensor, and this
        # path deliberately skips sanitize because the cached matrix is already sanitized.
        cost_matrix = opt._prev_cost_matrix
    else:
        # More efficient: process all probes for each particle in a batch
        # Every (particle, vertex, probe) candidate, evaluated in chunks to bound memory.
        total_evals = P_active * V * K_eff
        _evals_this_step = total_evals

        _is_subspace = opt.subspace is not None
        _sub_dim = state.subspace.subspace_dim if _is_subspace else 0

        # Fused subspace reconstruct + in-place forward path: avoids
        # materialising N full weight dicts when the evaluator supports it.
        _use_fused_inplace = (
            opt._hybrid
            and hasattr(state.subspace, "apply_perturbation_inplace")
            and getattr(getattr(opt, "_cost_evaluator", None), "_use_inplace", False)
        )

        # Factored subspace: score candidates through the low-rank identity rather
        # than building N weight tensors. Falls back to reconstruct_batch below when
        # the evaluator could not build a plan for this model.
        _factored_eval = getattr(opt, "_factored_evaluator", None) if opt._factored else None

        # Full space: a candidate perturbs one contiguous run of the flat vector, so
        # every layer but that one holds the shared base weight and the per-candidate
        # bmm is unnecessary. Needs the evaluator's data, same as the in-place path.
        _sparse_delta = getattr(opt, "_sparse_delta_evaluator", None) if not _is_subspace else None
        # Same locality argument with no module-set assumption, so it covers the models
        # and the chunks the sparse-delta path declines.
        _site_vmap = getattr(opt, "_site_vmap_evaluator", None) if not _is_subspace else None
        _pdim_arange = torch.arange(pdim, device=device)
        if _sparse_delta is not None or _site_vmap is not None:
            _base_sd = opt.layout.unflatten(X)

        # Needs a cheap closure from the caller; without one there is no cheaper way to
        # rank directions. Off whenever a quadratic-model consumer is active: those need
        # a full (P, V, K) loss tensor at a single fidelity.
        _fused_inputs = getattr(opt, "_fused_inputs", None)
        _fused_targets = getattr(opt, "_fused_targets", None)
        # Every path reading the registered batch needs the sliced copy, or stage one
        # runs at full fidelity while the accounting below charges screen_fidelity.
        _subspace_delta = getattr(opt, "_subspace_delta_evaluator", None) if opt._hybrid else None
        # The same site argument in coordinate space, for the models the delta path
        # declines: a per-layer block maps to one parameter, so only that one is batched.
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
        # At K_eff == 1 the curvature regression has nothing to regress on, so the
        # quadratic model buys a shared f(X) per particle instead of a second scale.
        _center_wanted = (
            K_eff == 1
            and opt.use_quadratic_model
            and (opt.biased_rotation or opt.trust_region)
            and opt.polytope_type == "orthoplex"
        )
        screen_inputs, screen_targets = opt._screen_data(_evaluator_native)
        # A selection solver reads only the winner, so its screen ranks vertices by their own
        # screened cost and never touches an antithetic partner. That is what lets it run on
        # any polytope; the contrast-ranked branch below still needs the orthoplex's pairing.
        _selection_solver = isinstance(opt.solver, (MinCostGreedySolver, TopKMeanSolver))
        _screen_ready = (
            opt.multifidelity_screen
            and screen_closure is not None
            and (_selection_solver or (opt.polytope_type == "orthoplex" and V == 2 * pdim))
            and opt.screen_keep_ratio < 1.0
            and not (opt.use_quadratic_model or opt._newton_refinement or opt.trust_region)
            # Stage one is P_active*V cheap forwards, stage two keep_ratio of the dense
            # P_active*V*K_eff. Only pays when screen_fidelity/K_eff + keep_ratio < 1.
            and opt.screen_fidelity / K_eff + opt.screen_keep_ratio < 1.0
        )
        if opt.multifidelity_screen and not _screen_ready and not getattr(opt, "_screen_warned", False):
            opt._screen_warned = True
            warnings.warn(
                "multifidelity_screen=True but the screen did not run, so no forward "
                "evaluations are saved this step. It needs a cheap screen_closure passed "
                "to step() (api.train builds one), screen_fidelity/num_probe + "
                "screen_keep_ratio < 1 (above that the screen costs more work than it "
                "saves), none of use_quadratic_model / newton_refinement / trust_region "
                "enabled, and either polytope_type='orthoplex' or a selection solver "
                "(min_cost_greedy / top_k_mean), whose screen ranks vertices directly and "
                "so needs no antithetic pairing.",
                stacklevel=3,
            )

        # Budget on what a candidate allocates: the config tensor, a full weight set
        # when a subspace runs without the fused or factored path, and the vmap
        # activations. Chunking only splits the loop, so results are unchanged.
        if opt.chunk_size:
            chunk = opt.chunk_size
        else:
            per_candidate = max(1, X.numel())  # P * pdim
            # The delta path is picked per chunk and any chunk can straddle a coordinate
            # block and fall back, so the budget still has to cover a materialized chunk.
            if _is_subspace and not _use_fused_inplace and _factored_eval is None:
                per_candidate += opt.layout.total_params
            if _fused_inputs is not None and opt.layout.entries:
                widest = max(e.shape[0] for e in opt.layout.entries if e.shape)
                per_candidate += _fused_inputs.shape[0] * widest
            chunk = min(total_evals, max(1, (1 << 26) // per_candidate))

        _group = V * K_eff
        _site_bounds = None
        if _subspace_delta is not None or _sparse_delta is not None or _site_vmap is not None or _subspace_site:
            # Every site-aware path groups a chunk by particle, so a chunk holds whole
            # groups of V*K_eff candidates. A ragged chunk would split one particle's
            # group across two calls and misalign the (G, C, pdim) reshape.
            chunk = max(_group, (chunk // _group) * _group)
        # A site-aware path needs every candidate in the chunk inside one parameter, so
        # the chunk loop breaks where the sites do. Otherwise a chunk spans several,
        # resolves to no site, and the whole sweep materializes.
        if _sparse_delta is not None or _site_vmap is not None:
            _starts = [e.offset for e in opt.layout.entries]
        elif _subspace_delta is not None or _subspace_site is not None:
            _starts = [s.flat_start for s in state.subspace.specs]
        else:
            _starts = []
        if _starts:
            # Sites do not land on particle boundaries, so each start contributes its own
            # group and the next. That isolates the straddling particle in its own chunk.
            _site_bounds = sorted(
                {min(b, total_evals) for s in _starts for b in ((s // pdim) * _group, -(-s // pdim) * _group)}
            )

        # Shape-determined and fully overwritten each step, so cached on the optimizer.
        # The config buffer stays None until a chunk needs it; a step the sparse-delta
        # path covers end to end never builds one.
        _loss_dtype = loss_buffer_dtype(X.dtype)
        _buf_key = (total_evals, chunk, V, K_eff, X.shape[0], X.shape[1], device, X.dtype)
        _bufs = getattr(opt, "_step_buffers", None)
        if _bufs is None or _bufs[0] != _buf_key:
            _all = torch.arange(total_evals, device=device)
            # The (i, v, k) candidate list is a pure function of the key, so it is built
            # with the buffers rather than recomputed from integer division every step.
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
        ):
            """Evaluate an explicit list of (particle, vertex, probe) candidates.

            The list is explicit rather than a dense range so the multi-fidelity
            screen can evaluate a subset of vertices. ``dense`` marks the full list,
            whose particle-major grouping both delta paths need; a screened subset has
            no such grouping.
            """
            probe_src = X_probe if probes_src is None else probes_src
            n_cand = i_all.shape[0]
            # Rows the previous chunk dirtied. Restoring just those is O(chunk * pdim)
            # against O(chunk * P * pdim) for a full copy. None means copy it all first.
            dirty_rows = None
            for chunk_start, chunk_end in _chunk_spans(n_cand, chunk, _site_bounds if dense else None):
                chunk_size_actual = chunk_end - chunk_start

                i_idx = i_all[chunk_start:chunk_end]
                v_idx = v_all[chunk_start:chunk_end]
                k_idx = k_all[chunk_start:chunk_end]

                # Sparse-delta path: every candidate in this chunk perturbs one
                # contiguous run inside a single parameter, so all the other layers
                # keep the shared base weight and no candidate config is built at all.
                _site = None
                if (_sparse_delta is not None or _site_vmap is not None) and fused_inputs is not None:
                    # Perturbed positions follow the particle, so the particle-major list
                    # gathers once per group. A screened chunk has no grouping.
                    _offsets = i_idx * pdim
                    _cand = _group if dense else 1
                    _span = (
                        ((chunk_start // _group) * pdim, ((chunk_end - 1) // _group) * pdim + pdim) if dense else None
                    )
                    _owner = _sparse_delta if _sparse_delta is not None else _site_vmap
                    _site = _owner.resolve_site(_offsets, pdim, _span)
                    if _site is None and _sparse_delta is not None and _site_vmap is not None:
                        # The sparse-delta correction is confined to Linear layers; the
                        # site-aware vmap still shares the graph ahead of any entry.
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

                # Subspace delta path. The dense list is particle-major, so this chunk
                # covers groups [chunk_start // C, chunk_end // C) and its coordinate
                # span follows from arithmetic, with no read back from the device.
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

                # Same locality in coordinate space, for a model the delta path declines:
                # build only the one perturbed parameter and share the rest.
                if _subspace_site is not None and dense and fused_inputs is not None:
                    _g0, _g1 = chunk_start // _group, chunk_end // _group
                    _spec = _subspace_site.resolve_spec(state.subspace, _g0 * pdim, _g1 * pdim, _sub_dim)
                    if _spec is not None:
                        # An offset from the barycentre, which _bary_sd already carries.
                        _dcoords = probe_src.new_zeros(chunk_size_actual, _spec.flat_end - _spec.flat_start)
                        _cols = (
                            torch.arange(_g0, _g1, device=device).unsqueeze(1) * pdim + _pdim_arange
                        ).repeat_interleave(_group, dim=0).reshape(chunk_size_actual, pdim) - _spec.flat_start
                        _dcoords.scatter_(
                            1, _cols, (probe_src[i_idx, v_idx, k_idx] - X[i_idx]).reshape(chunk_size_actual, pdim)
                        )
                        out[chunk_start:chunk_end] = _subspace_site.evaluate_subspace(
                            state.hybrid_projections,
                            _bary_sd,
                            _spec,
                            _dcoords,
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
                    # EGGROLL-inspired: one config at a time via in-place weight swap,
                    # never the full (N, *param_shape) stack. Memory O(1 x activation).
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
                    if opt._adaptive or opt._cma_subspace:
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

                out[chunk_start:chunk_end] = chunk_losses.to(out.dtype)

            # Over every candidate at once, not per chunk: the penalty is
            # 2*max|finite|+1 of what it is handed, so a per-chunk one could rank an
            # infeasible vertex above an expensive-but-legal vertex from another chunk.
            if sanitize:
                out.copy_(sanitize_cost(out))
            return out

        if not _screen_ready:
            _evaluate_candidates(_i_all, _v_all, _k_all, closure, _fused_inputs, _fused_targets, losses, dense=True)
        else:
            # Stage 1: rank every direction on the cheap fidelity, one probe scale.
            # Cost: P_active * V forwards on a fraction of the data.
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

            # Stage 2: full fidelity only where it can still matter. Ranked per particle:
            # each carries its own rotation, so vertex j is a different direction in every
            # row and a statistic averaged down the rows would rank noise.
            keep_mask = torch.zeros(P_active, V, dtype=torch.bool, device=device)
            if _selection_solver:
                # Selection rules need no antithetic pairing, hence no orthoplex here.
                # TopKMeanSolver averages min(k, V) vertices, so keeping fewer would feed
                # it screen-ineligible entries carrying the sanitize penalty.
                keep_v = max(1, min(V, int(round(V * opt.screen_keep_ratio))), min(getattr(opt.solver, "k", 1), V))
                keep_idx = torch.topk(screen_cost, keep_v, dim=1, largest=False).indices
                keep_mask.scatter_(1, keep_idx, True)
            else:
                # A weighted mean is moved by the contrast between antithetic partners, so
                # rank directions by that and keep both signs of each.
                pdim_local = V // 2
                contrast = (screen_cost[:, :pdim_local] - screen_cost[:, pdim_local:]).abs()
                keep_k = max(1, min(pdim_local, int(round(pdim_local * opt.screen_keep_ratio))))
                keep_dirs = torch.topk(contrast, keep_k, dim=1).indices  # (P_active, keep_k)
                keep_mask.scatter_(1, keep_dirs, True)
                keep_mask.scatter_(1, keep_dirs + pdim_local, True)

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

            # The two fidelities sit at different levels. Offset the dropped entries by
            # the per-particle difference on the kept ones, or the solve ranks by
            # fidelity rather than by cost.
            losses = _fill_screened_losses(
                screen_cost, kept, sel_idx, keep_mask, P_active, V, K_eff, mask_dropped=_selection_solver
            )
            # One penalty over the merged population. Sanitizing the screen and the kept
            # vector separately gives them different 2*max|finite|+1 substitutes, which
            # then get compared against each other in the same cost matrix.
            losses = sanitize_cost(losses)
            # Screen forwards see part of the data, so weight them rather than counting
            # them whole. Use the realized slice when the batch is in hand; a
            # closure-only screen has none, so fall back to the configured fraction.
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

        if K_eff == 1:
            cost_matrix = losses.reshape(P, V)
            if opt.use_quadratic_model:
                # clone: `losses` is the persistent step buffer, so a view would
                # silently become the *next* step's values.
                opt._losses_3d = losses.reshape(P, V, 1).detach().clone()
                # One centre per particle replaces the second probe scale the
                # regression needs: P extra evaluations against P*V.
                opt._center_loss = None
                if _center_wanted:
                    _range_P = torch.arange(P, device=device)
                    _zeros_P = torch.zeros_like(_range_P)
                    opt._center_loss = _evaluate_candidates(
                        _range_P,
                        _zeros_P,
                        _zeros_P,
                        closure,
                        _fused_inputs,
                        _fused_targets,
                        X.new_empty(P, dtype=_loss_dtype),
                        probes_src=X.reshape(P, 1, 1, -1),
                    ).detach()
        else:
            losses_3d_full = losses.reshape(P, V, K_eff)
            cost_matrix = losses_3d_full.mean(dim=-1)  # (P, V)
            opt._center_loss = None
            if opt.use_quadratic_model:
                # clone: `losses` is the persistent step buffer, so a view would
                # silently become the *next* step's values.
                opt._losses_3d = losses_3d_full.detach().clone()

    # No sanitize here: `losses` was sanitized per chunk and a mean of finite
    # values is finite, and the reuse path holds an already-sanitized matrix.

    # Raw cost for next step's contrast/trust-region baseline; dampening below
    # only shapes the current OT solve.
    raw_cost_matrix = cost_matrix

    # Deferred: this step's cost matrix is measured at the position the previous step
    # produced, so comparing it against that step's prediction gives a real ratio.
    # Both ends must use the same estimator, so a step that has a centre and one that
    # does not are never compared.
    _center_now = opt._center_loss is not None
    if (
        opt.trust_region
        and opt._prev_predicted_improvement is not None
        and opt._prev_pre_step_loss is not None
        and _center_now == opt._prev_loss_from_center
    ):
        # f(X) where available. The fallback proxy is min-over-vertices under a fresh
        # random rotation, so it moves between steps even at a fixed point.
        current_loss = opt._center_loss.mean().item() if _center_now else cost_matrix.min(dim=1).values.mean().item()
        # Sign convention matches update_trust_region and predicted improvement:
        # negative means loss decreased.
        actual_improvement = torch.tensor([current_loss - opt._prev_pre_step_loss])
        from .quadratic_model import update_trust_region

        opt._trust_region_multiplier = update_trust_region(
            opt._prev_predicted_improvement,
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

    # A >2x epsilon jump invalidates the dual-momentum history: prev_prev_f/g were
    # measured at the old epsilon, and extrapolating across the jump lands the warm
    # start far from the new fixed point.
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
        # No clamp here: SinkhornSolver bounds any warm start it is handed by
        # 10 * max|C_scaled| and zeroes a non-finite one. Duals scale with cost
        # magnitude, so an epsilon-scaled bound truncated valid extrapolations.

    opt.solver.epsilon = ot_epsilon
    if opt._use_fused_softmax:
        # Fused softmax + vertex-free projection. Scale outside the kernel so every
        # scale_cost mode matches the non-fused solvers.
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
        # prev_epsilon lets the solver rescale warm-started duals across an epsilon
        # change. a=None takes the solver's uniform 1/P and skips a host sync.
        solve_kwargs = dict(
            cost_matrix=cost_matrix,
            init_f=init_f_for_solve,
            init_g=init_g_for_solve,
            scale_cost=opt.scale_cost,
        )
        if isinstance(opt.solver, SinkhornSolver) and state.last_solve_eps is not None:
            solve_kwargs["init_eps"] = state.last_solve_eps
        ot_result = opt.solver.solve(**solve_kwargs)
    state.last_solve_eps = ot_epsilon

    feed_solver_stats(opt._progressive_epsilon, opt.solver, ot_result.n_iters, ot_result.converged)

    # Save pre-step subspace coords for AdaptiveSubspace displacement
    # history and CMA evolution paths. Declared here, populated conditionally.
    _pre_step_sub_coords = None

    # CMA takes the same displacement-history and rotate path as a plain
    # AdaptiveSubspace, so it needs these even with both CMA flags off.
    if opt._adaptive or opt._per_layer_projections or opt._cma_subspace:
        sub_dim = opt.subspace.subspace_dim
        _pre_step_sub_coords = state.X.reshape(-1)[:sub_dim].clone()

    # The finite-difference model feeds two independent features: biased_rotation
    # (descent/Newton directions) and trust_region (predicted-vs-actual ratio), so it
    # is built whenever either is active.
    _center_loss = getattr(opt, "_center_loss", None)
    if _center_loss is not None and _center_loss.shape != (P,):
        _center_loss = None
    _quad_ready = (
        opt.use_quadratic_model
        and opt._losses_3d is not None
        and (K_eff >= 2 or _center_loss is not None)
        and opt._losses_3d.shape == (P, V, K_eff)
        and opt.polytope_type == "orthoplex"  # FD extractors assume orthoplex vertex order
    )
    _fd_grad = _fd_hess = None
    if not _quad_ready:
        # Stale otherwise: adaptive_num_probe can drop K_eff to 1 mid-run, and the
        # amortized step prefers this direction over the transport EMA.
        opt._newton_direction = None
    if (opt.biased_rotation or opt.trust_region) and _quad_ready:
        from .quadratic_model import (
            compute_newton_step,
            extract_fd_gradient,
            extract_fd_hessian_diag,
            extract_fd_hessian_diag_centered,
        )

        _fd_grad = extract_fd_gradient(opt._losses_3d, probes, probe_r, pdim)  # (P, pdim)
        if _center_loss is not None:
            _fd_hess = extract_fd_hessian_diag_centered(opt._losses_3d, probes, probe_r, pdim, _center_loss)
        else:
            _fd_hess = extract_fd_hessian_diag(opt._losses_3d, probes, probe_r, pdim)  # (P, pdim)

        if opt.biased_rotation:
            # Descent direction = negative gradient in original space (rot_mats @ .).
            newton_rot = compute_newton_step(_fd_grad, _fd_hess, max_step_norm=step_r, hessian_reg=1e-4)
            grad_orig = torch.einsum("bij,bj->bi", rot_mats, _fd_grad)  # (P, pdim)
            opt._prev_descent_direction = -grad_orig.detach()
            opt._prev_descent_direction_finite = bool(torch.isfinite(grad_orig).all())
            newton_orig = torch.einsum("bij,bj->bi", rot_mats, newton_rot)
            opt._newton_direction = newton_orig.detach()
    elif opt.biased_rotation:
        # OT descent direction when the quadratic model is unavailable. A different
        # mechanism, not a degraded one, so warn rather than let the caller assume the
        # finite-difference path is running.
        if not getattr(opt, "_biased_rotation_fallback_warned", False):
            opt._biased_rotation_fallback_warned = True
            warnings.warn(
                "biased_rotation is aligning the chart to the OT barycenter's displacement, "
                "not to a finite-difference descent/Newton direction. The FD path needs "
                "use_quadratic_model=True and polytope_type='orthoplex'; num_probe=1 is "
                "enough, the step then evaluates one f(X) per particle for the curvature.",
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
        X_bary = X_new_fused  # Already computed by fused function
    else:
        X_bary = opt._compiled.barycentric_projection(
            ot_result.matrix,
            X_vertices,
        )

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

    # Newton refinement: post-OT correction using quadratic model
    # Tracks whether a post-solve move (refinement) makes the solve's duals stale.
    _duals_invalidated = False
    if opt._newton_refinement and opt._losses_3d is not None and K_eff >= 2 and opt.polytope_type == "orthoplex":
        from .quadratic_model import apply_newton_refinement

        X_refined = apply_newton_refinement(
            X_bary=state.X,
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
            # Newton refinement moved particles, so the solve's dual potentials
            # encode the old positions. Mark them stale so the dual-save block
            # below does not warm-start the next step from them.
            _duals_invalidated = True

    # state.X is final here, so the model scores the realized move (transport, momentum
    # and refinement) rather than a Newton step that was never applied.
    if opt.trust_region and _fd_grad is not None:
        from .quadratic_model import compute_predicted_improvement

        realized_rot = torch.einsum("bij,bj->bi", rot_mats.transpose(-1, -2), state.X - X)
        opt._prev_predicted_improvement = compute_predicted_improvement(_fd_grad, _fd_hess, realized_rot).detach()
        opt._prev_loss_from_center = _center_loss is not None
        opt._prev_pre_step_loss = (
            _center_loss.mean().item() if _center_loss is not None else raw_cost_matrix.min(dim=1).values.mean().item()
        )

    if opt._cma_subspace and opt.use_covariance_adaptation:
        cma_sub = opt.subspace  # CMAAdaptiveSubspace
        sub_dim = cma_sub.subspace_dim

        # OT displacement alone, from X_bary not state.X: the paths and rank-mu must
        # describe the same offspring distribution, and state.X carries momentum and
        # the Newton correction the covariance update never saw.
        post_step_coords = X_bary.reshape(-1)[:sub_dim]
        raw_displacement = post_step_coords - _pre_step_sub_coords

        # Recombination weights: the transport row, normalized to sum to 1 per particle.
        transport = ot_result.matrix  # (P, V)
        recomb = transport / transport.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # step_r, not sigma: step_r = step_radius * epsilon * sigma, so dividing by sigma
        # alone leaves a (step_radius * epsilon)^2 factor in every second moment.
        step_scale = max(abs(step_r), 1e-12)

        # Direction only, not magnitude: an OT step over an orthoplex has no fixed norm
        # to compare against, so sqrt(n) does not apply. With unit-norm innovations the
        # stationary ||p_sigma|| is 1 in any dimension, so the ratio is pure agreement.
        z_displacement = raw_displacement / step_scale
        z_norm = torch.linalg.vector_norm(z_displacement)
        # Below this the step is rounding, and normalising it feeds the paths noise.
        if bool(z_norm > 1e-12):
            z_direction = z_displacement / z_norm
            # p_c lives on y = sqrt(C) z; p_sigma lives on whitened z, which z_direction
            # already is, so it takes C_diag=None instead of a cancelling round trip.
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
            # c_mu is exactly 0 at mu_eff = 1, the default the optimizer derives its
            # rates at, so rank-mu would be multiplied by zero. Skip building it.
            _c_mu = opt._cma_params["c_mu"]
            if _c_mu > 0.0:
                # Offspring are the polytope vertices weighted by transport mass. The
                # pdim factor keeps the update trace-preserving: sum_v w_v z_v z_v^T has
                # trace 1 per particle where C has trace pdim.
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
                # p_c is unwhitened, so its squares sum to E[z^T C z] while C_diag sums
                # to sub_dim. The renormalisation below absorbs the mismatch, leaving
                # this scale to set rank-one strength only.
                trace_scale=float(sub_dim),
                cov_min=opt._cma_params["cov_min"],
                cov_max=opt._cma_params["cov_max"],
                # Mean 1: C carries shape, the step radius carries scale.
                trace=float(sub_dim),
            )

        state.generation += 1

    # One transfer for all three: read separately they cost three device syncs a step,
    # and only the cost mean is needed before the step ends.
    _cost_mean, _ess, _rho = torch.stack([cost_matrix.mean(), _ess_tensor, _rho_tensor]).tolist()
    # evals is candidate evaluations this step, net of probe reuse and screening.
    # Multiply by the batch size for sample-forwards.
    state.record_solver_health(_ess, _rho, _evals_this_step)
    # Tracked unconditionally: absorb_mode="stagnation" reads this counter, which
    # use_adaptive_radius (default False) does not gate.
    _prev_loss_for_radius = state.prev_loss
    state.stagnation_count, state.prev_loss = update_stagnation(
        _cost_mean,
        state.prev_loss,
        state.stagnation_count,
        stagnation_threshold=opt.stagnation_threshold,
    )
    if opt.use_adaptive_radius:
        state.radius_multiplier, state.stagnation_count = update_radius_multiplier(
            _cost_mean,
            _prev_loss_for_radius,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )

    _nan_reverted = False
    if not torch.isfinite(state.X).all():
        # Reverting to the pre-step X only helps when that point was itself finite;
        # if the state arrived poisoned, fall back to the coordinate origin, which is
        # the base weights in subspace mode and the layout flatten in full space.
        state.X = X.clone() if torch.isfinite(X).all() else torch.zeros_like(X)
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # The CMA block above consumed the same non-finite X, so reverting X alone leaves
        # sqrt(C_diag) poisoned in the sampling projection and the run cannot recover.
        if opt._cma_subspace and opt.use_covariance_adaptation:
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

    if opt.amortize_steps > 1:
        if _nan_reverted:
            opt._transport_direction_ema = None
        else:
            # Pure OT step from X_bary, not state.X: momentum and Newton are
            # applied separately, so the coasting direction must not re-carry them.
            raw_direction = (X_bary - X).detach()
            alpha = opt.amortize_ema
            if opt._transport_direction_ema is None:
                opt._transport_direction_ema = raw_direction
            else:
                opt._transport_direction_ema = alpha * opt._transport_direction_ema + (1.0 - alpha) * raw_direction

    # NB: trust-region update happens at the start of the next call to
    # ``step`` (deferred), where we have a real post-step pre-OT measurement.

    per_particle_disp_sqnorms = torch.sum((state.X - X) ** 2, dim=-1)  # (P,)
    disp_sqnorm_tensor = torch.mean(per_particle_disp_sqnorms)
    state.costs.append(_cost_mean)
    state.linear_convergence.append(ot_result.converged)
    state.displacement_sqnorms.append(disp_sqnorm_tensor.item())
    state.iteration_count += 1
    opt._ot_step_costs.append(_cost_mean)

    # Cache the matrix with the configuration it was measured at. A reuse step measured
    # nothing, so it leaves the anchor where it is.
    if opt._adaptive_probes and not _can_reuse:
        opt._prev_X = X.detach().clone()
        opt._prev_cost_matrix = raw_cost_matrix.detach()
        opt._prev_rot_mats = rot_mats.detach()
        opt._prev_k_eff = K_eff
        opt._prev_step_r = step_r
        opt._prev_probe_r = probe_r
    # Save duals for warm-starting; reset if the step reverted on NaN or a
    # post-solve refinement moved the particles (duals now encode old positions).
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

    # Adaptive subspace: displacement tracking, absorb, and rotation.
    # CMAAdaptiveSubspace wraps AdaptiveSubspace by composition, not inheritance, so
    # it must be tested separately or CMA runs never rotate or absorb.
    if opt._adaptive or opt._cma_subspace:
        adaptive_sub = opt.subspace
        # Probes and _sync_model both used the cached projection, covariance-scaled
        # under CMA. Folding with state.projection would drop the sqrt(C_diag) factor
        # and move the weights.
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
            # New random projection
            # Sparse projection: create new SparseRandomProjection with fresh seed
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
            # Clear the stagnation counter, else absorb_mode='stagnation' stays
            # triggered on a plateau and redraws the basis every step. The absorb
            # re-anchors the origin, so the old loss history no longer applies.
            state.stagnation_count = 0
            state.prev_loss = _cost_mean
            if opt._cma_subspace and opt.use_covariance_adaptation:
                state.p_c = torch.zeros_like(state.p_c)
                state.p_sigma = torch.zeros_like(state.p_sigma)
                state.C_diag = torch.ones_like(state.C_diag)
                # Keep generation counter (don't reset to preserve cumulation history)
        elif opt._cma_subspace and opt.use_covariance_adaptation:
            # sep-CMA's C_diag and evolution paths index axes of THIS basis, and a
            # diagonal covariance does not stay diagonal under rotation. Hold the basis
            # between absorbs; absorb redraws it and resets the CMA state above.
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
                # Dense projection: use existing displacement-based rotation. Pass the
                # full-space history so each entry keeps the frame it was measured in.
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
    # Periodic absorb for plain subspaces only; adaptive and hybrid handle their own.
    # Fires after every N steps, iteration_count having been incremented above.
    if (
        not opt._adaptive
        and not opt._per_layer_projections
        and not opt._cma_subspace
        and opt.subspace is not None
        and opt.absorb_every > 0
        and state.iteration_count % opt.absorb_every == 0
    ):
        flat_sub = state.X.reshape(-1)[: state.subspace.subspace_dim]
        new_base, _zeroed = state.subspace.absorb(state.base_params, flat_sub)
        state.base_params = new_base
        state.X = torch.zeros_like(state.X)
        state.f = None
        state.g = None
        state.prev_prev_f = None
        state.prev_prev_g = None
        opt._transport_direction_ema = None
        opt._invalidate_reuse_cache()
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)

    # Against the applied rank, not subspace_dim: the schedule yields a per-layer rank
    # while subspace_dim sums the coordinate counts it produces, so comparing the two
    # fires every step and rebuilds the subspace from scratch each time.
    if opt._rank_schedule is not None and opt.subspace is not None:
        current_rank = opt._rank_schedule.at(state.iteration_count)
        if current_rank != opt._applied_rank:
            opt._transition_rank(current_rank)

    opt._sync_model()

    return _cost_mean
