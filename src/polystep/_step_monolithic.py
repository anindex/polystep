"""Monolithic step: one OT solve over all particles, or one per row segment."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Callable

import torch

from ._compiled import normalize_rows
from .cost_nn import SiteInplaceEvaluator, _reuse_prefixes
from .solvers._shared import loss_buffer_dtype, recenter_and_scale, sanitize_cost, solver_health
from .epsilon import feed_solver_stats, radius_epsilon_factor, resolve_epsilon
from ._step_core import (
    activation_elements,
    check_losses,
    chunk_spans,
    epsilon_jumped,
    invalidate_for_basis_change,
    record_saturation,
    site_bounds,
    site_chunks,
    update_amortized_direction,
    update_stagnation_and_radius,
    warn_all_nonfinite,
)
from .dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
)
from .geometry import apply_biased_rotation, get_random_rotation_matrices
from .projection import SparseRandomProjection
from .quadratic_model import (
    apply_newton_refinement,
    compute_newton_step,
    compute_predicted_improvement,
    extract_fd_gradient,
    extract_fd_hessian_diag,
    extract_fd_hessian_diag_centered,
    extract_iso_curvature,
    update_trust_region,
)
from .solvers.greedy import MinCostGreedySolver, TopKMeanSolver
from .solvers.base import SolverResult

logger = logging.getLogger(__name__)


def _fill_screened_losses(screen_cost, kept, sel_idx, keep_mask, P, V, K_eff, mask_dropped=False):
    """Assemble the (P*V*K_eff,) loss vector from a screened evaluation.

    Kept vertices carry full-fidelity values; dropped ones get a cheap value
    calibrated on the kept entries finite at both fidelities for weighted solvers,
    or ``+inf`` (``mask_dropped``) for selection solvers. A dropped vertex with a
    non-finite screen value stays non-finite.
    """
    losses = screen_cost.new_empty(P * V * K_eff)
    losses[sel_idx] = kept

    full_3d = losses.reshape(P, V, K_eff)
    mask_3d = keep_mask.unsqueeze(-1)
    if mask_dropped:
        return torch.where(mask_3d, full_3d, torch.full_like(full_3d, float("inf"))).reshape(-1)
    screen_ok = torch.isfinite(screen_cost)
    ok = mask_3d & torch.isfinite(full_3d) & screen_ok.unsqueeze(-1)
    w = ok.sum(dim=-1).to(screen_cost.dtype) / K_eff
    n_kept = w.sum(dim=1, keepdim=True)
    n_kept = torch.where(n_kept > 0, n_kept, torch.ones_like(n_kept))
    # Zero dropped positions before summing, or 0 * inf becomes NaN.
    full_kept = torch.where(ok, full_3d, torch.zeros_like(full_3d))
    full_mean = full_kept.sum(dim=(1, 2)).unsqueeze(1) / (n_kept * K_eff)
    screen_mean = (torch.where(screen_ok, screen_cost, 0) * w).sum(dim=1, keepdim=True) / n_kept
    calibrated = (screen_cost + (full_mean - screen_mean)).unsqueeze(-1).expand(P, V, K_eff)

    return torch.where(mask_3d, full_3d, calibrated).reshape(-1)


def _solve_segments(solver, segments, cost_matrix, init_f, init_g, scale_cost) -> SolverResult:
    """Solve each row segment on its own; ``g`` stacks one column dual per segment."""
    costs = cost_matrix.tensor_split(segments)
    fs = gs = [None] * segments
    if (
        init_f is not None
        and init_g is not None
        and init_f.shape == cost_matrix.shape[:1]
        and init_g.shape == (segments, cost_matrix.shape[1])
    ):
        fs, gs = init_f.tensor_split(segments), init_g.unbind()
    parts = [solver.solve(cost_matrix=c, init_f=f, init_g=g, scale_cost=scale_cost) for c, f, g in zip(costs, fs, gs)]
    return SolverResult(
        matrix=torch.cat([r.matrix for r in parts]),
        f=torch.cat([r.f for r in parts]),
        g=torch.stack([r.g for r in parts]),
        converged=all(r.converged for r in parts),
        n_iters=max(r.n_iters for r in parts),
    )


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


def finish_absorb(opt, state) -> None:
    """Count an absorb and reset the history, basis caches and stagnation; call after the new basis is drawn."""
    state.displacement_history.zero_()
    if state.displacement_history_full is not None:
        state.displacement_history_full.zero_()
    state.displacement_history_idx = 0
    state.displacement_history_count = 0
    invalidate_for_basis_change(opt, state)
    state.absorb_count += 1
    # Otherwise a stagnation absorb re-triggers every step on a plateau.
    state.stagnation_count = 0


def maintain_per_layer_subspace(opt, state, pre_step_sub_coords) -> None:
    """Displacement tracking, absorb, and rotation for a per-layer subspace."""
    if not opt._per_layer_projections:
        return
    hybrid_sub = opt.subspace

    record_displacement(state, hybrid_sub, pre_step_sub_coords, None)

    should_absorb = hybrid_sub.should_absorb(
        state.stagnation_count,
        state.iteration_count,
    )
    hist = state.displacement_history[: state.displacement_history_count]
    # absorb_aligned_active biases the basis toward productive directions via displacement-SVD.
    if should_absorb and getattr(hybrid_sub, "absorb_aligned_active", False):
        svd_ratio = hybrid_sub.get_svd_ratio(state.iteration_count, opt.max_iterations or 1)
        new_projections = hybrid_sub._rotate_all_displacement(
            state.hybrid_projections,
            hist,
            svd_ratio,
            state.X.device,
            state.X.dtype,
            state.iteration_count,
        )
    else:
        new_projections = hybrid_sub.rotate_all(
            state.hybrid_projections,
            step=state.iteration_count,
            total_steps=opt.max_iterations,
            displacement_history=hist,
        )
    # Identity means nothing rotated, so the block_diag rebuild below can be skipped.
    rotated = new_projections is not state.hybrid_projections
    if not (should_absorb or rotated):
        return
    # Re-anchor first: swapping P at non-zero coords moves the weights unevaluated.
    state.base_params, _ = hybrid_sub.absorb(
        state.hybrid_projections,
        state.base_params,
        state.X.reshape(-1)[: hybrid_sub.subspace_dim],
    )
    state.X = torch.zeros_like(state.X)
    if rotated:
        state.hybrid_projections = new_projections
        if hasattr(hybrid_sub, "build_fused_projection"):
            hybrid_sub.build_fused_projection(new_projections)
    if should_absorb:
        finish_absorb(opt, state)
    else:
        # Rows index the replaced basis, so the next rotation would read phantom
        # directions.
        state.displacement_history.zero_()
        state.displacement_history_idx = 0
        state.displacement_history_count = 0
        invalidate_for_basis_change(opt, state)


def step_monolithic(opt, closure: Callable, screen_closure: Callable | None = None) -> float:
    """Monolithic step: one OT solve over all particles, or one per row segment."""
    state = opt._state
    X = state.X  # (P, particle_dim)
    iteration = state.iteration_count
    device = X.device

    current_eps = resolve_epsilon(opt.epsilon, iteration, "epsilon")
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

    # Every (particle, vertex, probe) candidate, evaluated in chunks to bound memory.
    total_evals = P * V * K_eff
    _evals_this_step = total_evals

    _is_subspace = opt.subspace is not None
    _sub_dim = state.subspace.subspace_dim if _is_subspace else 0

    # In-place forward path: avoids materialising N full weight dicts.
    _use_fused_inplace = (
        opt._hybrid
        and hasattr(state.subspace, "apply_perturbation_inplace")
        and getattr(getattr(opt, "_cost_evaluator", None), "_use_inplace", False)
    )

    _fused_inputs = getattr(opt, "_fused_inputs", None)
    _fused_targets = getattr(opt, "_fused_targets", None)
    # Factored subspace scores through the low-rank identity, never building a
    # candidate weight.
    _factored_eval = getattr(opt, "_factored_evaluator", None) if opt._factored and _fused_inputs is not None else None

    # Full space perturbs one contiguous run, so only that tensor needs batching.
    _sparse_delta = getattr(opt, "_sparse_delta_evaluator", None) if not _is_subspace else None
    # Same locality, no module-set assumption, covers what the delta path declines.
    _site_vmap = getattr(opt, "_site_vmap_evaluator", None) if not _is_subspace else None
    _pdim_arange = torch.arange(pdim, device=device)
    if _sparse_delta is not None or _site_vmap is not None:
        _base_sd = opt.layout.unflatten(X)

    _act = activation_elements(opt, _fused_inputs)
    _subspace_delta = getattr(opt, "_subspace_delta_evaluator", None) if opt._per_layer_projections else None
    if (
        _factored_eval is not None
        and device.type != "cpu"
        and (total_evals <= (1 << 26) // (X.numel() + _act) or (opt.chunk_size or total_evals) * _act < 1 << 24)
    ):
        _subspace_delta = None
    # The same site argument in coordinate space: a per-layer block maps to one
    # parameter, so only that one is batched.
    _subspace_site = getattr(opt, "_site_vmap_evaluator", None) if opt._hybrid else None
    _bary_sd = None
    if _fused_inputs is not None and (_subspace_delta is not None or _subspace_site is not None):
        _bary_coords = X.reshape(-1)[:_sub_dim]
        _half = [state.base_params[s.entry_key].dtype in (torch.float16, torch.bfloat16) for s in state.subspace.specs]
        if not any(_half):
            _bary_sd = state.subspace.apply_perturbation(state.hybrid_projections, state.base_params, _bary_coords)
        elif not (all(_half) and _subspace_delta is None and isinstance(_subspace_site, SiteInplaceEvaluator)):
            _bary_sd = state.subspace.reconstruct_batch(
                state.hybrid_projections, state.base_params, _bary_coords.unsqueeze(0)
            )
            _bary_sd = {k: v[0] for k, v in _bary_sd.items()}
    else:
        _subspace_delta = _subspace_site = None
        _bary_coords = None

    # One incumbent loss serves every parameter block, at every probe count.
    _center_wanted = opt.use_quadratic_model and (opt.biased_rotation or opt.trust_region)
    screen_inputs, screen_targets = opt._screen_data()
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

    _bary_flat = None
    if (
        opt._adaptive
        and device.type == "cpu"
        and not _screen_ready
        and _sub_dim >= pdim
        and isinstance(state.projection, torch.Tensor)
        and state.projection.dtype not in (torch.float16, torch.bfloat16)
    ):
        _bary_flat = torch.addmv(
            torch.cat([state.base_params[s.entry_key].reshape(-1) for s in state.subspace._entry_specs]).to(
                state.projection.dtype
            ),
            state.projection,
            X.reshape(-1)[:_sub_dim],
        )

    # Budget on what a candidate allocates; chunking only splits the loop.
    _local = any(p is not None for p in (_subspace_delta, _sparse_delta, _site_vmap, _subspace_site))
    if opt.chunk_size:
        chunk_mat = chunk = opt.chunk_size
    else:
        per_candidate = max(1, X.numel())  # P * pdim
        # A materialized subspace candidate builds every weight.
        if _is_subspace and not _use_fused_inplace and _factored_eval is None:
            per_candidate += opt.layout.total_params
        chunk_mat = chunk = min(total_evals, max(1, (1 << 26) // (per_candidate + _act)))
        if _local:
            per_site = 2 * _act
            if _site_vmap is not None or _subspace_site is not None:
                per_site += max(e.numel for e in opt.layout.entries)
            chunk = min(total_evals, max(1, (1 << 26) // max(1, per_site)))

    _group = V * K_eff
    if _local or _bary_flat is not None:
        # Keep whole particle groups in one chunk, or the (G, C, pdim) reshape
        # misaligns.
        chunk = max(_group, (chunk // _group) * _group)
        chunk_mat = min(chunk_mat, chunk)
    # Break chunks where sites do, or a chunk spans entries and resolves to no site.
    if _sparse_delta is not None or _site_vmap is not None:
        _sites = [(e.offset, e.key) for e in opt.layout.entries]
    elif _subspace_delta is not None or _subspace_site is not None:
        _sites = [(s.flat_start, s.entry_key) for s in state.subspace.specs]
    elif _bary_flat is not None:
        _sites = [(_sub_dim, None)]
    else:
        _sites = []
    _site_bounds = site_bounds([s for s, _ in _sites], pdim, _group, total_evals)
    _site_chunks = None
    if _local and not opt.chunk_size and _sparse_delta is None and _subspace_delta is None:
        _site_chunks = site_chunks(opt, _fused_inputs, _sites, pdim, _group, total_evals)

    # Shape-determined and fully overwritten each step, so cached on the optimizer.
    _loss_dtype = loss_buffer_dtype(X.dtype)
    _buf_key = (total_evals, chunk_mat, V, K_eff, X.shape[0], X.shape[1], device, X.dtype)
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
            torch.arange(chunk_mat, device=device) if chunk_mat <= total_evals else None,
            _all // (V * K_eff),
            _vk // K_eff,
            _vk % K_eff,
            None,
        ]
        opt._step_buffers = _bufs
    if _bary_flat is not None and _bufs[7] is None:
        _bufs[7] = _bary_flat.new_empty(min(chunk, total_evals) // _group, _group, _bary_flat.numel())
    _, losses, _, _local_range_full, _i_all, _v_all, _k_all, _ = _bufs
    _configs_wanted = chunk_mat <= total_evals

    @_reuse_prefixes(_sparse_delta, _subspace_delta, _site_vmap, _subspace_site, _factored_eval)
    def _evaluate_candidates(
        i_all,
        v_all,
        k_all,
        closure_fn,
        fused_inputs,
        fused_targets,
        out,
        dense=False,
        probes_src=None,
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
        for chunk_start, chunk_end in chunk_spans(
            n_cand, chunk, _site_bounds if dense else None, _site_chunks if dense else None
        ):
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
                _span = ((chunk_start // _group) * pdim, ((chunk_end - 1) // _group) * pdim + pdim) if dense else None
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
                    out[chunk_start:chunk_end] = _subspace_site.evaluate_subspace(
                        state.hybrid_projections,
                        state.base_params,
                        _bary_coords,
                        _bary_sd,
                        _spec,
                        _g0 * pdim - _spec.flat_start,
                        probe_src[i_idx, v_idx, k_idx].reshape(_g1 - _g0, _group, pdim),
                        fused_inputs,
                        fused_targets,
                    ).to(out.dtype)
                    continue

            if _bary_flat is not None and dense:
                _g0, _g1 = chunk_start // _group, chunk_end // _group
                if _g1 * pdim <= _sub_dim:
                    chunk_losses = closure_fn(
                        state.subspace.reconstruct_offsets(
                            state.projection,
                            _bary_flat,
                            _g0 * pdim,
                            (probe_src[i_idx, v_idx, k_idx] - X[i_idx]).reshape(_g1 - _g0, _group, pdim),
                            _bufs[7][: _g1 - _g0],
                        )
                    )
                    check_losses(chunk_losses, chunk_size_actual)
                    out[chunk_start:chunk_end] = chunk_losses.to(out.dtype)
                    continue

            for lo in range(chunk_start, chunk_end, chunk_mat):
                hi = min(lo + chunk_mat, chunk_end)
                if hi - lo < chunk_size_actual:
                    i_idx, v_idx, k_idx = i_all[lo:hi], v_all[lo:hi], k_all[lo:hi]
                if _configs_wanted and hi - lo == chunk_mat:
                    local_range = _local_range_full
                    if _bufs[2] is None:
                        # Built from X, so it needs no restore on this pass.
                        _bufs[2] = X.unsqueeze(0).expand(chunk_mat, -1, -1).clone()
                        batch_configs = _bufs[2]
                    else:
                        batch_configs = _bufs[2]
                        if dirty_rows is None:
                            batch_configs.copy_(X)  # copy_ broadcasts
                        else:
                            batch_configs[local_range, dirty_rows] = X[dirty_rows]
                    dirty_rows = i_idx
                else:
                    batch_configs = X.unsqueeze(0).expand(hi - lo, -1, -1).clone()
                    local_range = torch.arange(hi - lo, device=device)
                batch_configs[local_range, i_idx] = probe_src[i_idx, v_idx, k_idx]

                flat_configs = batch_configs.reshape(hi - lo, -1)

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
                    chunk_params = state.subspace.reconstruct_batch(
                        state.projection if opt._adaptive else state.hybrid_projections,
                        state.base_params,
                        flat_sub,
                        *(() if _bary_flat is None else (_bufs[7].flatten(0, 1)[: hi - lo],)),
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

                check_losses(chunk_losses, hi - lo)
                out[lo:hi] = chunk_losses.to(out.dtype)

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
            X_probe.new_empty(P * V, dtype=_loss_dtype),
        )
        screen_raw = screen_flat.reshape(P, V)
        screen_cost = sanitize_cost(screen_raw)

        # Stage 2: full fidelity only where it can still matter, ranked per particle.
        keep_mask = torch.zeros(P, V, dtype=torch.bool, device=device)
        if _selection_solver:
            # A floor at k, not a cap: TopKMeanSolver skips the dropped vertices, so a
            # cap would leave it fewer than k scored ones to average.
            keep_v = min(V, max(1, int(round(V * opt.screen_keep_ratio)), min(getattr(opt.solver, "k", 1), V)))
            keep_idx = torch.topk(screen_cost, keep_v, dim=1, largest=False).indices
            keep_mask.scatter_(1, keep_idx, True)
        elif opt.polytope_type == "orthoplex":
            # A weighted mean is moved by the contrast between antithetic partners,
            # so rank by that and keep both signs.
            pdim_local = V // 2
            contrast = (screen_cost[:, :pdim_local] - screen_cost[:, pdim_local:]).abs()
            keep_k = max(1, min(pdim_local, int(round(pdim_local * opt.screen_keep_ratio))))
            keep_dirs = torch.topk(contrast, keep_k, dim=1).indices  # (P, keep_k)
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
        )

        # Offset the dropped entries by the per-particle difference on the kept ones,
        # or the solve ranks by fidelity rather than by cost.
        losses = _fill_screened_losses(
            screen_raw, kept, sel_idx, keep_mask, P, V, K_eff, mask_dropped=_selection_solver
        )
        # Screen forwards see part of the data, so weight them by the realized slice
        # (or the configured fraction when there is none).
        _fidelity = (
            screen_inputs.shape[0] / _fused_inputs.shape[0]
            if screen_inputs is not None and _fused_inputs is not None
            else opt.screen_fidelity
        )
        _evals_this_step = sel_idx.numel() + _fidelity * P * V
        opt._last_screen_savings = 1.0 - _evals_this_step / max(1, total_evals)
        logger.debug(
            "multifidelity_screen: %d/%d full-fidelity forwards + %d screen forwards",
            sel_idx.numel(),
            total_evals,
            P * V,
        )

    solve_cost = losses.reshape(P, V) if K_eff == 1 else losses.reshape(P, V, K_eff).mean(dim=-1)
    # One penalty over the merged population, or the two fidelities get different
    # 2*max|finite|+1 substitutes and compare wrong.
    safe = sanitize_cost(losses)
    cost_matrix = safe.reshape(P, V) if K_eff == 1 else sanitize_cost(safe.reshape(P, V, K_eff).mean(dim=-1))
    _nonfinite = ~torch.isfinite(solve_cost).any()
    opt._center_loss = None
    if opt.use_quadratic_model:
        # The step buffer is overwritten next time; the model owns a snapshot.
        opt._losses_3d = safe.reshape(P, V, K_eff).detach()
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
        ).detach()
        opt._center_loss = _center.expand(P).clone()
        _evals_this_step += 1

    # Charged after the quadratic model's centre evaluations, which also add to
    # _evals_this_step and would otherwise be undercounted here.
    opt.candidate_evals += int(_evals_this_step)

    # Deferred: the matrix is measured at the position the previous step produced, so
    # comparing it against that step's prediction gives a real ratio. Both ends must
    # use the same estimator, so a step with a centre and one without never compare.
    _center_now = opt._center_loss is not None
    _center_value = opt._center_loss[0].item() if _center_now else None
    if (
        opt.trust_region
        and opt._prev_predicted_improvement is not None
        and opt._prev_pre_step_loss is not None
        and _center_now
    ):
        # Negative means loss decreased.
        actual_improvement = torch.tensor([_center_value - opt._prev_pre_step_loss])
        opt._trust_region_multiplier = update_trust_region(
            opt._prev_predicted_improvement.sum(),
            actual_improvement,
            opt._trust_region_multiplier,
            min_radius=0.1,
            max_radius=3.0,
        )
        step_r = _sr * opt._trust_region_multiplier * _sr_eps * radius_mult
        X_vertices = rotated * step_r + X.unsqueeze(1)
        state.trust_region_multipliers.append(opt._trust_region_multiplier)
        opt._prev_predicted_improvement = None
        opt._prev_pre_step_loss = None

    ent_eps = resolve_epsilon(opt.ent_epsilon, iteration, "ent_epsilon")
    ot_epsilon = ent_eps if ent_eps is not None else current_eps

    # A >2x epsilon jump invalidates the dual-momentum history: the prev values were
    # measured at the old epsilon and extrapolate to a bad warm start.
    if epsilon_jumped(state, ot_epsilon):
        state.prev_prev_f = None
        state.prev_prev_g = None

    init_f_for_solve = state.f
    init_g_for_solve = state.g
    if (
        opt._dual_momentum_beta > 0.0
        and state.f is not None
        and state.prev_prev_f is not None
        and state.prev_prev_g is not None
        and state.prev_prev_g.shape == state.g.shape
    ):
        beta = opt._dual_momentum_beta
        init_f_for_solve = state.f + beta * (state.f - state.prev_prev_f)
        init_g_for_solve = state.g + beta * (state.g - state.prev_prev_g)
        # No clamp: SinkhornSolver bounds any warm start by 10*max|C_scaled|, and an
        # epsilon-scaled bound would truncate valid extrapolations.

    opt.solver.epsilon = ot_epsilon
    if opt._use_fused_softmax:
        # Scale outside the kernel so every scale_cost mode matches the non-fused solvers.
        solve_safe = cost_matrix if K_eff == 1 else sanitize_cost(solve_cost)
        scaled_cost = recenter_and_scale(solve_safe, solve_cost, opt.scale_cost)[0]
        X_new_fused, transport_matrix = opt._compiled.fused_softmax_project(
            scaled_cost,
            ot_epsilon,
            opt._polytope_vertices,
            rot_mats,
            step_r,
            X,
        )
        # state.costs tracks cost_matrix.mean, so the OT cost is unused here.
        ot_result = SolverResult(matrix=transport_matrix)
    else:
        solve_kwargs = dict(
            cost_matrix=solve_cost,
            init_f=init_f_for_solve,
            init_g=init_g_for_solve,
            scale_cost=opt.scale_cost,
        )
        ot_result = (
            opt.solver.solve(**solve_kwargs)
            if opt._row_segments is None or P < 2
            else _solve_segments(opt.solver, min(opt._row_segments, P), **solve_kwargs)
        )
    state.last_solve_eps = ot_epsilon

    # Pre-step subspace coords for the displacement history.
    _pre_step_sub_coords = None

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
        and (_center_loss is None or math.isfinite(_center_value))
        and (_center_loss is not None or (K_eff >= 2 and _antithetic))
    )
    _fd_grad = _fd_hess = None
    if not _quad_ready:
        # Stale otherwise: adaptive_num_probe can drop K_eff to 1 mid-run.
        opt._newton_direction = None
    if (opt.biased_rotation or opt.trust_region) and _quad_ready:
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
        vertex_offsets = X_vertices - X.unsqueeze(1)  # (P, V, pdim)
        normalized_weights = normalize_rows(ot_result.matrix)  # (P, V)
        weighted_dir = (normalized_weights.unsqueeze(-1) * vertex_offsets).sum(dim=1)  # (P, pdim)
        opt._prev_descent_direction = weighted_dir.detach()

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

    # Post-OT correction, ahead of momentum so the velocity carries it.
    X_target = X_bary
    if opt._newton_refinement and opt._losses_3d is not None and K_eff >= 2 and opt.polytope_type == "orthoplex":
        X_refined = apply_newton_refinement(
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
        X_target = torch.where(torch.isfinite(X_refined).all(-1, keepdim=True), X_refined, X_bary)

    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(
            iteration,
            opt.max_iterations,
            opt.momentum_init,
            opt.momentum_final,
        )
        X_new, vel_new = apply_momentum(
            X,
            X_target,
            state.velocity,
            beta,
            opt.velocity_lr,
        )
        state.velocity = vel_new
        state.X = X_new
    else:
        state.X = X_target

    # state.X is final here, so the model scores the realized move, not a Newton step
    # that was never applied.
    if opt.trust_region and _fd_grad is not None and _center_loss is not None:
        realized_rot = torch.einsum("bij,bj->bi", rot_mats.transpose(-1, -2), state.X - X)
        # Rows are disjoint parameter blocks of one objective, not independent losses.
        opt._prev_predicted_improvement = compute_predicted_improvement(_fd_grad, _fd_hess, realized_rot).sum().detach()
        opt._prev_pre_step_loss = _center_value

    # One transfer for all of these, not a device sync each.
    _cost_mean, _ess, _rho, _all_nonfinite, _progress, _x_finite, _disp = torch.stack(
        [
            cost_matrix.mean(),
            _ess_tensor,
            _rho_tensor,
            _nonfinite.to(cost_matrix.dtype),
            # Mean of each particle's best vertex; the full-matrix mean would partly
            # measure the radius controller's own last move.
            cost_matrix.min(dim=1).values.mean(),
            torch.isfinite(state.X).all(),
            torch.mean(torch.sum((state.X - X) ** 2, dim=-1)),
        ]
    ).tolist()
    _fabricated = bool(_all_nonfinite)
    if _fabricated:
        warn_all_nonfinite(opt)
        _cost_mean = _progress = float("inf")
    _converged = ot_result.converged and not _fabricated
    feed_solver_stats(opt._progressive_epsilon, opt.solver, ot_result.n_iters, _converged)
    # Candidate evaluations this step, net of screening.
    state.record_solver_health(_ess, _rho, _evals_this_step)
    update_stagnation_and_radius(opt, state, _progress)

    _nan_reverted = not _x_finite
    if _nan_reverted:
        # Revert to pre-step X, resetting only its non-finite coordinates to the origin.
        state.X = X.nan_to_num(0.0, 0.0, 0.0)
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        if opt.biased_rotation:
            opt._prev_descent_direction = None
        opt._newton_direction = None
        _disp = torch.mean(torch.sum((state.X - X) ** 2, dim=-1)).item()

    # From X_bary, not state.X: the coasting direction must not re-carry momentum/Newton.
    update_amortized_direction(opt, (X_bary - X).detach(), _nan_reverted)

    # Trust-region update is deferred to the next step's pre-OT measurement.

    state.costs.append(_cost_mean)
    state.linear_convergence.append(_converged)
    state.displacement_sqnorms.append(_disp)
    state.iteration_count += 1
    opt._ot_step_costs.append(_cost_mean)

    # Save duals for warm-starting; reset if NaN reverted, and keep the old ones when
    # the solve saw a fabricated flat matrix.
    if _nan_reverted:
        state.f = None
        state.g = None
        if opt._dual_momentum_beta > 0.0:
            state.prev_prev_f = None
            state.prev_prev_g = None
    elif not _fabricated:
        if opt._dual_momentum_beta > 0.0:
            state.prev_prev_f = state.f.clone() if state.f is not None else None
            state.prev_prev_g = state.g.clone() if state.g is not None else None
        state.f = ot_result.f.detach() if ot_result.f is not None else None
        state.g = ot_result.g.detach() if ot_result.g is not None else None
    state.epsilon = current_eps

    # Displacement tracking, absorb, and rotation.
    if opt._adaptive:
        adaptive_sub = opt.subspace
        proj_used = state.projection

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
            finish_absorb(opt, state)
        elif _rotation_due(adaptive_sub, state.iteration_count):
            # Re-anchor first: the point is base + P @ coords, so swapping P at non-zero
            # coords moves the weights unevaluated. The basis QR/SVD runs in fp32.
            state.base_params, _ = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                state.X.reshape(-1)[: adaptive_sub.subspace_dim],
            )
            state.X = torch.zeros_like(state.X)

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

    maintain_per_layer_subspace(opt, state, _pre_step_sub_coords)

    # Against the applied rank, not subspace_dim: the latter sums coordinate counts, so
    # comparing them would rebuild the subspace from scratch every step.
    if opt._rank_schedule is not None:
        current_rank = opt._rank_schedule.at(state.iteration_count)
        if current_rank != opt._applied_rank:
            opt._transition_rank(current_rank)

    opt._sync_model()

    return _cost_mean
