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
from .solvers._prelude import loss_buffer_dtype, recenter_cost, sanitize_cost
from .dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
    update_radius_multiplier,
    update_stagnation,
)
from .geometry import apply_biased_rotation, get_random_rotation_matrices
from .solvers import SinkhornSolver
from .solvers.base import SolverResult
from .adaptive_subspace import AdaptiveSubspace
from .hybrid_subspace import HybridSubspace
from .cma import (
    update_evolution_path_sigma,
    compute_heaviside_sigma,
    update_evolution_path_c,
    update_covariance_diagonal,
    update_step_size_csa,
)

logger = logging.getLogger(__name__)


def _fill_screened_losses(screen_cost, kept, sel_idx, keep_mask, P_active, V, K_eff):
    """Assemble the (P_active*V*K_eff,) loss vector from a screened evaluation.

    Kept vertices carry their full-fidelity values. Dropped vertices carry their
    cheap-fidelity value plus a per-particle offset, estimated as the mean
    difference between the two fidelities on the kept vertices of that particle.
    ``keep_mask`` is (P_active, V): each particle keeps its own vertex set.
    """
    losses = screen_cost.new_empty(P_active * V * K_eff)
    losses[sel_idx] = kept

    full_3d = losses.reshape(P_active, V, K_eff)
    mask_3d = keep_mask.unsqueeze(-1)
    # Dropped positions hold uninitialized memory, so zero them before summing
    # rather than weighting them: 0 * inf is NaN.
    full_kept = torch.where(mask_3d, full_3d, torch.zeros_like(full_3d))
    n_kept = keep_mask.sum(dim=1, keepdim=True).to(screen_cost.dtype).clamp(min=1)
    full_mean = full_kept.sum(dim=(1, 2)).unsqueeze(1) / (n_kept * K_eff)
    screen_mean = (screen_cost * keep_mask).sum(dim=1, keepdim=True) / n_kept
    calibrated = (screen_cost + (full_mean - screen_mean)).unsqueeze(-1).expand(P_active, V, K_eff)

    return torch.where(mask_3d, full_3d, calibrated).reshape(-1)


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

    # Resolve epsilon and radii
    current_eps = opt._get_epsilon(iteration)
    # Use CSA sigma or heuristic radius_multiplier
    if opt.use_csa and state.use_csa:
        radius_mult = state.sigma
    elif opt.use_adaptive_radius:
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

    # Ensure 2D
    if X.dim() == 1:
        X = X.unsqueeze(0)
    P, pdim = X.shape

    # Move templates to device (cache after first transfer)
    if opt._polytope_vertices.device != device or opt._polytope_vertices.dtype != X.dtype:
        opt._polytope_vertices = opt._polytope_vertices.to(device=device, dtype=X.dtype)
    polytope_verts = opt._polytope_vertices
    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes
    V = polytope_verts.shape[0]  # num vertices
    K = probes.shape[0]  # num probes

    # Adaptive probe count: reduce K during exploitation
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

    # Select reduced probes if K_eff < K
    if K_eff < K:
        probes = probes[K // 2 : K // 2 + 1]  # center scale, shape (1,)

    # Adaptive probes: decide reuse before generating rotations. Every candidate is
    # the whole configuration X with one row replaced, so a cached row for particle i
    # was measured against every other particle's position too. Any particle that moves
    # invalidates all the rows, not just its own: reuse is all or nothing on X. The
    # cached rotations come back with it, so the rows still describe the same vertices.
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
        # Generate rotation matrices: (P, pdim, pdim)
        rot_mats = get_random_rotation_matrices(
            P,
            pdim,
            device=device,
            dtype=X.dtype,
            generator=opt._generator,
        )

        # Transport-biased rotation: replace first column with previous OT descent direction
        if (
            opt.biased_rotation
            and opt._prev_descent_direction is not None
            and opt._prev_descent_direction.shape == (P, pdim)
            and opt._prev_descent_direction_finite
        ):
            rot_mats = apply_biased_rotation(rot_mats, opt._prev_descent_direction)

    # Rotate + translate: X_vertices (P, V, pdim), rotated (P, V, pdim)
    X_vertices, rotated = opt._compiled.rotate_and_translate(
        rot_mats,
        polytope_verts,
        X,
        step_r,
    )

    # Probe generation: X_probe (P, V, K, pdim)
    X_probe = opt._compiled.compute_probe_points(
        X,
        rotated,
        probes,
        probe_r,
    )

    # Build full model configs and evaluate cost
    # For each (i, v, k), construct a full (P, pdim) config with row i
    # replaced by X_probe[i, v, k]. Then unflatten to params and evaluate.

    # The configuration has not moved since the cached matrix was measured, so every
    # row still describes its vertices. Zero forwards this step.
    if _can_reuse:
        cost_matrix = opt._prev_cost_matrix.clone()
    else:
        # More efficient: process all probes for each particle in a batch
        # Total evaluations: P_active * V * K_eff. Each builds full config (P, pdim)
        # flattened to (padded_size,), then batch-unflattened.
        #
        # Strategy: process in chunks to control memory.
        # Build indices for all (active_i, v, k) combinations.
        total_evals = P_active * V * K_eff

        # Cache subspace info for the loop
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
        if _sparse_delta is not None:
            _base_sd = opt.layout.unflatten(X)
            _pdim_arange = torch.arange(pdim, device=device)

        # Two-stage multi-fidelity screening. Requires a cheap closure from the
        # caller; without one there is no cheaper way to rank directions, so the
        # feature degrades to the post-hoc reweighting below (which saves nothing).
        # The quadratic-model consumers need a full (P, V, K) loss tensor measured at
        # a single fidelity, so screening stays off whenever one of them is active.
        _fused_inputs = getattr(opt, "_fused_inputs", None)
        _fused_targets = getattr(opt, "_fused_targets", None)
        screen_inputs, screen_targets = opt._screen_data(_use_fused_inplace)
        _screen_ready = (
            opt.multifidelity_screen
            and screen_closure is not None
            and opt.polytope_type == "orthoplex"
            and V == 2 * pdim
            and opt.screen_keep_ratio < 1.0
            and not (opt.use_quadratic_model or opt._newton_refinement or opt.trust_region)
            # Stage one costs P_active*V forwards on screen_fidelity of the data,
            # stage two keep_ratio * P_active*V*K_eff full ones, against a dense
            # budget of P_active*V*K_eff. Only pays when
            # screen_fidelity/K_eff + keep_ratio < 1.
            and opt.screen_fidelity / K_eff + opt.screen_keep_ratio < 1.0
        )
        if opt.multifidelity_screen and not _screen_ready and not getattr(opt, "_screen_warned", False):
            opt._screen_warned = True
            warnings.warn(
                "multifidelity_screen=True but the screen did not run, so no forward "
                "evaluations are saved this step. It needs a cheap screen_closure passed "
                "to step() (api.train builds one), polytope_type='orthoplex', "
                "screen_fidelity/num_probe + screen_keep_ratio < 1 (above that the screen "
                "costs more work than it saves), and none of use_quadratic_model / "
                "newton_refinement / trust_region enabled.",
                stacklevel=3,
            )

        # Budget on what a candidate actually allocates: the config tensor, a full weight
        # set when a subspace runs without the fused or factored path, and the vmap
        # activations. Counting configs alone missed the other two and peaked at 1.6 GB
        # full-space / 10 GB subspace on a 120k-param model. Chunking only splits the
        # loop, so results are unchanged.
        if opt.chunk_size:
            chunk = opt.chunk_size
        else:
            per_candidate = max(1, X.numel())  # P * pdim
            if _is_subspace and not _use_fused_inplace and _factored_eval is None:
                per_candidate += opt.layout.total_params
            if _fused_inputs is not None and opt.layout.entries:
                widest = max(e.shape[0] for e in opt.layout.entries if e.shape)
                per_candidate += _fused_inputs.shape[0] * widest
            chunk = min(total_evals, max(1, (1 << 26) // per_candidate))

        # These are shape-determined and fully overwritten each step, so they are cached
        # on the optimizer and reused across steps rather than reallocated. The config
        # buffer stays None until a chunk actually needs it: a step the sparse-delta path
        # covers end to end never builds a candidate config at all.
        _loss_dtype = loss_buffer_dtype(X.dtype)
        _buf_key = (total_evals, chunk, X.shape[0], X.shape[1], device, X.dtype)
        _bufs = getattr(opt, "_step_buffers", None)
        if _bufs is None or _bufs[0] != _buf_key:
            _bufs = [
                _buf_key,
                torch.empty(total_evals, dtype=_loss_dtype, device=device),
                None,
                torch.arange(chunk, device=device) if chunk <= total_evals else None,
                torch.arange(total_evals, device=device),
            ]
            opt._step_buffers = _bufs
        _, losses, _, _local_range_full, _all_indices = _bufs
        _configs_wanted = chunk <= total_evals

        def _evaluate_candidates(i_all, v_all, k_all, closure_fn, fused_inputs, fused_targets, out):
            """Evaluate an explicit list of (particle, vertex, probe) candidates.

            The candidate list is explicit rather than a dense range so the
            multi-fidelity screen below can evaluate a *subset* of vertices; the
            unscreened path passes the full dense list and behaves exactly as before.
            """
            n_cand = i_all.shape[0]
            # Rows the previous chunk perturbed in the cached buffer. Each candidate
            # differs from X in one row, so restoring those costs O(chunk * pdim)
            # against O(chunk * P * pdim) for a full copy. None means the buffer is
            # not ours yet and needs the full copy first.
            dirty_rows = None
            for chunk_start in range(0, n_cand, chunk):
                chunk_end = min(chunk_start + chunk, n_cand)
                chunk_size_actual = chunk_end - chunk_start

                i_idx = i_all[chunk_start:chunk_end]
                v_idx = v_all[chunk_start:chunk_end]
                k_idx = k_all[chunk_start:chunk_end]

                # Sparse-delta path: every candidate in this chunk perturbs one
                # contiguous run inside a single parameter, so all the other layers
                # keep the shared base weight and no candidate config is built at all.
                _site = None
                if _sparse_delta is not None and fused_inputs is not None:
                    _offsets = i_idx * pdim
                    _site = _sparse_delta.resolve_site(_offsets, pdim)
                if _site is not None:
                    out[chunk_start:chunk_end] = _sparse_delta.evaluate(
                        _base_sd,
                        _site.key,
                        _offsets.unsqueeze(1) + _pdim_arange - _site.offset,
                        X_probe[i_idx, v_idx, k_idx],
                        fused_inputs,
                        fused_targets,
                    ).to(out.dtype)
                    continue

                # Reuse pre-allocated buffer when chunk size matches, otherwise allocate
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
                batch_configs[local_range, i_idx] = X_probe[i_idx, v_idx, k_idx]

                # Flatten each config to (padded_size,)
                flat_configs = batch_configs.reshape(chunk_size_actual, -1)
                # flat_configs: (chunk_size, P * pdim)

                if _use_fused_inplace and _is_subspace and opt._hybrid:
                    # Fused path (EGGROLL-inspired): reconstruct + forward one
                    # config at a time via in-place weight swap. Never materializes
                    # the full (N, *param_shape) stacked dict. Memory: O(1 x activation).
                    flat_sub = flat_configs[:, :_sub_dim]
                    if opt._mixed_precision and getattr(state, "projection", None) is not None:
                        flat_sub = flat_sub.to(dtype=state.projection.dtype)
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
                    # Subspace mode: reconstruct full params from subspace coords
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
                    # Trim to layout padded_size if needed
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

            # Sanitize over every candidate at once, not per chunk: the +inf penalty is
            # 2*max|finite|+1 of whatever it is handed, so a per-chunk penalty can rank
            # an infeasible vertex above a legitimate expensive one from another chunk.
            # Doing it here also keeps NaN out of the quadratic model, which reads these.
            out.copy_(sanitize_cost(out))
            return out

        # Dense candidate list, computed once per step: (i, v, k) in the row-major
        # order the reshape below expects.
        _i_all = _all_indices // (V * K_eff)
        _vk_all = _all_indices % (V * K_eff)
        _v_all = _vk_all // K_eff
        _k_all = _vk_all % K_eff

        if not _screen_ready:
            _evaluate_candidates(_i_all, _v_all, _k_all, closure, _fused_inputs, _fused_targets, losses)
        else:
            # Stage 1: rank every direction on the cheap fidelity, one probe scale.
            # Cost: P_active * V forwards on a fraction of the data.
            k_center = K_eff // 2
            screen_flat = _evaluate_candidates(
                _i_all[_k_all == k_center],
                _v_all[_k_all == k_center],
                torch.full_like(_i_all[_k_all == k_center], k_center),
                screen_closure,
                screen_inputs,
                screen_targets,
                X_probe.new_empty(P_active * V, dtype=_loss_dtype),
            )
            screen_cost = screen_flat.reshape(P_active, V)

            # Stage 2: keep the highest-contrast antithetic pairs and spend the full
            # fidelity only on those. Both signs of a direction are kept together so
            # the orthoplex stays antithetic. Ranked per particle: each carries its own
            # rotation, so vertex index j is a different direction in every row and a
            # contrast averaged down the rows would rank noise.
            pdim_local = V // 2
            contrast = (screen_cost[:, :pdim_local] - screen_cost[:, pdim_local:]).abs()
            keep_k = max(1, min(pdim_local, int(round(pdim_local * opt.screen_keep_ratio))))
            keep_dirs = torch.topk(contrast, keep_k, dim=1).indices  # (P_active, keep_k)
            keep_mask = torch.zeros(P_active, V, dtype=torch.bool, device=device)
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
            )

            # The two fidelities estimate the same loss on different data, so they
            # sit at different levels. Offset the dropped entries by the per-particle
            # difference measured on the kept vertices, or the OT solve would rank
            # vertices by which fidelity they came from rather than by cost.
            losses = _fill_screened_losses(screen_cost, kept, sel_idx, keep_mask, P_active, V, K_eff)
            # Screen forwards see only screen_fidelity of the data, so weight them
            # rather than counting them as full evaluations.
            opt._last_screen_savings = 1.0 - (sel_idx.numel() + opt.screen_fidelity * P_active * V) / max(
                1, total_evals
            )
            logger.debug(
                "multifidelity_screen: %d/%d full-fidelity forwards + %d screen forwards",
                sel_idx.numel(),
                total_evals,
                P_active * V,
            )

        if K_eff == 1:
            # K=1 fast path: no averaging needed
            cost_matrix = losses.reshape(P, V)
            if opt.use_quadratic_model:
                # clone: `losses` is the persistent step buffer, so a view would
                # silently become the *next* step's values.
                opt._prev_losses_3d = losses.reshape(P, V, 1).detach().clone()
        else:
            losses_3d_full = losses.reshape(P, V, K_eff)
            cost_matrix = losses_3d_full.mean(dim=-1)  # (P, V)
            if opt.use_quadratic_model:
                # clone: `losses` is the persistent step buffer, so a view would
                # silently become the *next* step's values.
                opt._prev_losses_3d = losses_3d_full.detach().clone()

    # Sanitize cost (FP32 promote + finite penalty), branch-free / no host sync.
    # Protects the downstream trust-region / multifidelity readers and BOTH
    # solver paths: the fused softmax scales this same sanitized tensor.
    cost_matrix = sanitize_cost(cost_matrix)

    # Raw cost for next step's contrast/trust-region baseline; dampening below
    # only shapes the current OT solve.
    raw_cost_matrix = cost_matrix

    # Deferred trust-region update: the cost matrix at this step is evaluated
    # at the particle position produced by the *previous* step. Comparing the
    # prediction stored on the previous step against this step's pre-OT min cost
    # gives a real predicted-vs-actual reduction ratio.
    if opt.trust_region and opt._prev_predicted_improvement is not None and opt._prev_pre_step_loss is not None:
        current_loss_proxy = cost_matrix.min(dim=1).values.mean().item()
        # Sign convention matches update_trust_region and predicted improvement:
        # negative means loss decreased.
        actual_improvement = torch.tensor([current_loss_proxy - opt._prev_pre_step_loss])
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

    # If the schedule jumped epsilon by more than 2x in one step,
    # invalidate the dual-momentum history. prev_prev_f / prev_prev_g
    # were computed at the old epsilon; extrapolating across a big
    # epsilon change pushes the warm-start far from the new fixed
    # point and the next solve has to undo the bad init.
    if state.last_solve_eps is not None and (
        ot_epsilon / state.last_solve_eps > 2.0 or state.last_solve_eps / ot_epsilon > 2.0
    ):
        state.prev_prev_f = None
        state.prev_prev_g = None

    # Dual potential momentum: extrapolate warm-start duals
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
        # Clamp to prevent overflow (same bounds as warm-start validation)
        max_abs = 80.0 * max(ot_epsilon, 0.01)
        init_f_for_solve = init_f_for_solve.clamp(-max_abs, max_abs)
        init_g_for_solve = init_g_for_solve.clamp(-max_abs, max_abs)

    opt.solver.epsilon = ot_epsilon
    if opt._use_fused_softmax:
        # Fused path: softmax + vertex-free projection in one compiled call
        # Scale the cost outside the compiled kernel so every scale_cost mode
        # ('mean'/'max_cost'/float/None) matches the non-fused solvers; the
        # kernel then runs on the already-scaled cost (scale_cost_mean=False).
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
        # Forward the previous solve's epsilon so SinkhornSolver
        # can rescale the warm-started duals when epsilon changed.
        # a defaults to uniform 1/P inside the solver, which equals state.a; pass
        # None so the solver skips the host-syncing user-marginal validation.
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

    # Update progressive epsilon from solver stats. Skip in fixed-iteration
    # Sinkhorn mode (threshold <= 0): there the solver always returns
    # converged=True with n_iters == max_iterations, so the ratio is 1.0 and the
    # increase branch would push epsilon to max_epsilon every step. Progressive
    # epsilon is only paired with SinkhornSolver, so threshold always exists.
    if opt._progressive_epsilon is not None and getattr(opt.solver, "threshold", 1.0) > 0:
        opt._progressive_epsilon.update(
            n_iters=ot_result.n_iters,
            max_iterations=getattr(opt.solver, "max_iterations", 1),
            converged=ot_result.converged,
        )

    # Save pre-step subspace coords for AdaptiveSubspace displacement
    # history and CMA evolution paths. Declared here, populated conditionally.
    _pre_step_sub_coords = None

    # CMA takes the same displacement-history and rotate path as a plain
    # AdaptiveSubspace, so it needs these even with both CMA flags off.
    if opt._adaptive or opt._per_layer_projections or opt._cma_subspace:
        sub_dim = opt.subspace.subspace_dim
        _pre_step_sub_coords = state.X.reshape(-1)[:sub_dim].clone()

    # Compute rotation bias direction
    # The finite-difference model feeds two independent features: biased_rotation
    # (descent/Newton directions) and trust_region (predicted-vs-actual ratio), so it
    # is built whenever either is active.
    _quad_ready = (
        opt.use_quadratic_model
        and opt._prev_losses_3d is not None
        and K_eff >= 2
        and opt._prev_losses_3d.shape == (P, V, K_eff)
        and opt.polytope_type == "orthoplex"  # FD extractors assume orthoplex vertex order
    )
    _fd_grad = _fd_hess = None
    if (opt.biased_rotation or opt.trust_region) and _quad_ready:
        from .quadratic_model import compute_newton_step, extract_fd_gradient, extract_fd_hessian_diag

        # FD gradient and diagonal Hessian in the rotated frame.
        _fd_grad = extract_fd_gradient(opt._prev_losses_3d, probes, probe_r, pdim)  # (P, pdim)
        _fd_hess = extract_fd_hessian_diag(opt._prev_losses_3d, probes, probe_r, pdim)  # (P, pdim)

        if opt.biased_rotation:
            # Descent direction = negative gradient in original space (rot_mats @ .).
            newton_rot = compute_newton_step(_fd_grad, _fd_hess, max_step_norm=step_r, hessian_reg=1e-4)
            grad_orig = torch.einsum("bij,bj->bi", rot_mats, _fd_grad)  # (P, pdim)
            opt._prev_descent_direction = -grad_orig.detach()
            opt._prev_descent_direction_finite = bool(torch.isfinite(grad_orig).all())
            newton_orig = torch.einsum("bij,bj->bi", rot_mats, newton_rot)
            opt._newton_direction = newton_orig.detach()
    elif opt.biased_rotation:
        # Fallback: OT descent direction when the quadratic model is unavailable.
        transport_weights = ot_result.matrix  # (P, V)
        vertex_offsets = X_vertices - X.unsqueeze(1)  # (P, V, pdim)
        weight_sums = transport_weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
        normalized_weights = transport_weights / weight_sums  # (P, V)
        weighted_dir = (normalized_weights.unsqueeze(-1) * vertex_offsets).sum(dim=1)  # (P, pdim)
        opt._prev_descent_direction = weighted_dir.detach()
        opt._prev_descent_direction_finite = bool(torch.isfinite(weighted_dir).all())

    # Barycentric projection: (P, pdim)
    if opt._use_fused_softmax:
        X_bary = X_new_fused  # Already computed by fused function
    else:
        X_bary = opt._compiled.barycentric_projection(
            ot_result.matrix,
            state.a,
            X_vertices,
        )

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
    if opt._newton_refinement and opt._prev_losses_3d is not None and K_eff >= 2 and opt.polytope_type == "orthoplex":
        from .quadratic_model import apply_newton_refinement

        X_refined = apply_newton_refinement(
            X_bary=state.X,
            losses_3d=opt._prev_losses_3d,
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

    # Trust region: predict the improvement of the step actually taken.
    # state.X is final here, so the model scores the realized move (transport +
    # momentum + refinement) instead of a Newton step that was never applied.
    # Otherwise the ratio compares two different displacements.
    if opt.trust_region and _fd_grad is not None:
        from .quadratic_model import compute_predicted_improvement

        realized_rot = torch.einsum("bij,bj->bi", rot_mats.transpose(-1, -2), state.X - X)
        opt._prev_predicted_improvement = compute_predicted_improvement(_fd_grad, _fd_hess, realized_rot).detach()
        opt._prev_pre_step_loss = raw_cost_matrix.min(dim=1).values.mean().item()

    # CMA-ES updates
    if opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
        cma_sub = opt.subspace  # CMAAdaptiveSubspace
        sub_dim = cma_sub.subspace_dim

        # Displacement of the OT step alone, taken from X_bary rather than state.X.
        # The evolution paths and the rank-mu term below must describe the same
        # offspring distribution, and rank-mu uses the raw OT vertex steps; reading
        # state.X here would fold in momentum velocity and the Newton correction, so
        # the paths would track a step the covariance update never saw.
        post_step_coords = X_bary.reshape(-1)[:sub_dim]
        raw_displacement = post_step_coords - _pre_step_sub_coords

        # Recombination weights: the transport row, normalized to sum to 1 per particle.
        transport = ot_result.matrix  # (P, V)
        recomb = transport / transport.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # step_r, not sigma: step_r = step_radius * epsilon * sigma, so dividing by sigma
        # alone leaves a (step_radius * epsilon)^2 factor in every second moment.
        step_scale = max(abs(step_r), 1e-12)

        # Paths take the step direction, not its magnitude. An OT step over an orthoplex
        # has no fixed norm to compare against (antithetic vertex pairs cancel toward
        # zero), so sqrt(n) does not apply and any measured reference lags and drives
        # sigma into its clamp. With unit-norm innovations the stationary ||p_sigma|| is
        # exactly 1 in any dimension, so the ratio reads pure directional agreement.
        z_displacement = raw_displacement / step_scale
        z_norm = torch.linalg.vector_norm(z_displacement).clamp(min=1e-12)
        z_direction = z_displacement / z_norm
        # Sampling scales the projection columns by sqrt(C_diag), so the offspring step
        # in covariance-metric coordinates is y = sqrt(C_diag) * z. The evolution paths
        # are defined on y, not the raw z.
        sqrt_C = torch.sqrt(torch.clamp(state.C_diag, min=opt._cma_params["cov_min"]))
        y_displacement = sqrt_C * z_direction

        # Update p_sigma; its internal C^{-1/2} whitens y back to the unit direction.
        state.p_sigma = update_evolution_path_sigma(
            p_sigma=state.p_sigma,
            displacement=y_displacement,
            C_diag=state.C_diag,
            c_sigma=opt._cma_params["c_sigma"],
            mu_eff=1.0,
            cov_min=opt._cma_params["cov_min"],
        )

        # Stall test against the same unit reference.
        p_sigma_norm = torch.norm(state.p_sigma).item()
        h_sigma = compute_heaviside_sigma(
            p_sigma_norm=p_sigma_norm,
            expected_norm=1.0,
            n=sub_dim,
            c_sigma=opt._cma_params["c_sigma"],
            generation=state.generation,
        )

        # Update p_c (covariance evolution path) on the y-scaled step.
        state.p_c = update_evolution_path_c(
            p_c=state.p_c,
            displacement=y_displacement,
            h_sigma=h_sigma,
            c_c=opt._cma_params["c_c"],
            mu_eff=1.0,
        )

        # Update diagonal covariance.
        if opt.use_covariance_adaptation:
            # c_mu = 2*(mu_eff - 2 + 1/mu_eff)/(...) is exactly 0 at mu_eff = 1, which is
            # what the optimizer derives its rates at unless the subspace was given an
            # explicit mu_eff. So the default configuration is rank-one only and the
            # rank-mu term below would be multiplied by zero; skip building it.
            _c_mu = opt._cma_params["c_mu"]
            if _c_mu > 0.0:
                # Rank-mu offspring are the polytope vertices, weighted by transport mass.
                # Each coordinate belongs to one particle, so its variance is the weighted
                # mean of its V squared vertex steps, laid out to match the flattened
                # C_diag. The pdim factor makes the update trace-preserving:
                # sum_v w_v z_v z_v^T has trace 1 per particle while C restricted to that
                # particle has trace pdim.
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
                # p_c accumulates sqrt(C_diag) * z_direction with no whitening, so its
                # squares sum to E[z^T C z], which is 1 only for isotropic C, while
                # C_diag sums to sub_dim. The renormalisation below absorbs the
                # residual mismatch, so this scale only sets the rank-one strength.
                trace_scale=float(sub_dim),
                cov_min=opt._cma_params["cov_min"],
                cov_max=opt._cma_params["cov_max"],
            )
            state.C_diag = torch.clamp(
                state.C_diag,
                opt._cma_params["cov_min"],
                opt._cma_params["cov_max"],
            )
            # Mean 1 so C carries shape and sigma carries scale. Otherwise both grow along
            # the same direction and the effective step, sigma * sqrt(C), overshoots.
            state.C_diag = state.C_diag * (sub_dim / state.C_diag.sum().clamp(min=1e-12))

        # Update step-size via CSA (if enabled), against the same unit reference.
        if opt.use_csa:
            state.sigma = update_step_size_csa(
                sigma=state.sigma,
                p_sigma=state.p_sigma,
                c_sigma=opt._cma_params["c_sigma"],
                d_sigma=opt._cma_params["d_sigma"],
                n=sub_dim,
                p_sigma_norm=p_sigma_norm,
                expected_norm=1.0,
            )
            # Floor sigma to prevent collapse to zero
            state.sigma = max(state.sigma, 1e-6)

        state.generation += 1

    # Cache cost_matrix mean as tensor (defer .item() sync until needed)
    _cost_mean_tensor = cost_matrix.mean()

    # Adaptive radius (use model loss, not OT regularized cost)
    # Single GPU-CPU sync point for cost mean
    _cost_mean = _cost_mean_tensor.item()
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

    # NaN-safe state update - revert X, velocity, and duals if NaN after projection
    _nan_reverted = False
    if not torch.isfinite(state.X).all():
        # Reverting to the pre-step X only helps when that point was itself finite;
        # if the state arrived poisoned, fall back to the coordinate origin, which is
        # the base weights in subspace mode and the layout flatten in full space.
        state.X = X.clone() if torch.isfinite(X).all() else torch.zeros_like(X)
        # Reset velocity to prevent NaN propagation through momentum
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # The CMA block above consumed the same non-finite X, so reverting X alone leaves
        # sqrt(C_diag) poisoned in the sampling projection and the run cannot recover.
        if opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
            if state.p_c is not None:
                state.p_c = torch.zeros_like(state.p_c)
            if state.p_sigma is not None:
                state.p_sigma = torch.zeros_like(state.p_sigma)
            if state.C_diag is not None:
                state.C_diag = torch.ones_like(state.C_diag)
            state.sigma = 1.0
        _nan_reverted = True

    # Clear biased rotation descent direction and Newton direction on NaN revert
    if _nan_reverted and opt.biased_rotation:
        opt._prev_descent_direction = None
        opt._prev_descent_direction_finite = False
    if _nan_reverted:
        opt._newton_direction = None

    # Capture transport direction for amortized OT (after NaN check)
    if opt.amortize_steps > 1:
        if _nan_reverted:
            opt._transport_direction = None  # NaN step, no valid direction
            opt._transport_direction_ema = None
        else:
            # Pure OT step from X_bary, not state.X: momentum and Newton are
            # applied separately, so the coasting direction must not re-carry them.
            raw_direction = (X_bary - X).detach()
            opt._transport_direction = raw_direction
            # EMA blend: smooth transport direction across OT steps
            alpha = opt.amortize_ema
            if opt._transport_direction_ema is None:
                opt._transport_direction_ema = raw_direction
            else:
                opt._transport_direction_ema = alpha * opt._transport_direction_ema + (1.0 - alpha) * raw_direction

    # NB: trust-region update happens at the start of the next call to
    # ``step`` (deferred), where we have a real post-step pre-OT measurement.

    # Update diagnostics (defer GPU-CPU sync for displacement)
    per_particle_disp_sqnorms = torch.sum((state.X - X) ** 2, dim=-1)  # (P,)
    disp_sqnorm_tensor = torch.mean(per_particle_disp_sqnorms)
    state.costs.append(_cost_mean)
    state.linear_convergence.append(ot_result.converged)
    state.displacement_sqnorms.append(disp_sqnorm_tensor.item())
    state.iteration_count += 1
    # Track OT-step costs separately for adaptive_num_probe (excludes momentum steps)
    opt._ot_step_costs.append(_cost_mean)

    # Adaptive probes: cache the matrix with the configuration it was measured at, so
    # the next step can tell whether it still describes the vertices it will score.
    if opt._adaptive_probes:
        opt._prev_X = X.detach().clone()
        opt._prev_cost_matrix = raw_cost_matrix.detach()
        opt._prev_rot_mats = rot_mats.detach()
        opt._prev_k_eff = K_eff
        opt._prev_step_r = step_r
        opt._prev_probe_r = probe_r
    # Multi-fidelity screening: store cost matrix for next step's direction contrast
    # analysis (independent of adaptive probes)
    if opt.multifidelity_screen and not opt._adaptive_probes:
        opt._prev_cost_matrix = raw_cost_matrix.detach()
        opt._prev_k_eff = K_eff
        opt._prev_step_r = step_r
        opt._prev_probe_r = probe_r
    # Save duals for warm-starting; reset if the step reverted on NaN or a
    # post-solve refinement moved the particles (duals now encode old positions).
    if _nan_reverted or _duals_invalidated:
        state.f = None
        state.g = None
        # Also clear momentum history - can't extrapolate from invalid state
        if opt._dual_momentum_beta > 0.0:
            state.prev_prev_f = None
            state.prev_prev_g = None
    else:
        # Dual momentum: save previous duals before overwriting
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

        # Compute displacement in subspace coords
        # The displacement is the change in the flattened subspace coordinate
        # vector after the barycentric projection + momentum update.
        post_step_sub_coords = state.X.reshape(-1)[: adaptive_sub.subspace_dim]
        displacement = post_step_sub_coords - _pre_step_sub_coords

        # Update displacement history (rolling buffer). Also store the full-space
        # image under the basis in use right now: the basis rotates every step, so
        # mapping all stored coordinates through the latest basis would attribute old
        # displacements to directions they were never measured along.
        idx = state.displacement_history_idx
        state.displacement_history[idx] = displacement
        # Only dense bases take the displacement-SVD rotation; a sparse projection
        # rotates by reseeding, so it has no use for the full-space history.
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

        # Check for absorb trigger
        should_absorb = adaptive_sub.should_absorb(
            state.stagnation_count,
            state.iteration_count,  # already incremented above
        )

        if should_absorb:
            # Absorb: fold perturbation into base, zero coords, new random basis.
            full_flat_sub = state.X.reshape(-1)[: adaptive_sub.subspace_dim]
            new_base, _zeroed = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                full_flat_sub,
            )
            state.base_params = new_base
            # Zero the subspace coordinates rather than re-projecting onto
            # the new basis. The new and old bases are largely uncorrelated
            # after a random redraw, so re-projection adds complexity for
            # little benefit; the next OT solve will discover a fresh
            # descent direction.
            state.X = torch.zeros_like(state.X)
            # New random projection
            # Sparse projection: create new SparseRandomProjection with fresh seed
            from .projection import SparseRandomProjection

            if isinstance(state.projection, SparseRandomProjection):
                # Increment seed based on absorb count for fresh random basis
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
            # Reset duals (cost landscape changed)
            state.f = None
            state.g = None
            state.prev_prev_f = None
            state.prev_prev_g = None
            # Invalidate EMA transport direction (cost geometry changed)
            opt._transport_direction_ema = None
            opt._transport_direction = None
            state.absorb_count += 1
            # Clear the stagnation counter, else absorb_mode='stagnation' stays
            # triggered on a plateau and redraws the basis every step. The absorb
            # re-anchors the origin, so the old loss history no longer applies.
            state.stagnation_count = 0
            state.prev_loss = _cost_mean
            # Invalidate cached cost/probe state (cost landscape changed after absorb)
            opt._invalidate_reuse_cache()
            opt._newton_direction = None
            opt._prev_descent_direction = None
            opt._prev_descent_direction_finite = False
            # Momentum velocity lives in the pre-absorb basis; zero it.
            if opt.use_momentum and state.velocity is not None:
                state.velocity = torch.zeros_like(state.velocity)
            # CMA-ES: Reset evolution paths and covariance after absorb
            if opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
                state.p_c = torch.zeros_like(state.p_c)
                state.p_sigma = torch.zeros_like(state.p_sigma)
                state.C_diag = torch.ones_like(state.C_diag)
                state.sigma = 1.0
                # Keep generation counter (don't reset to preserve cumulation history)
        elif opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
            # sep-CMA learns a per-axis variance for THIS basis, so rotating the basis
            # under it would invalidate everything it has accumulated: C_diag[j] and the
            # evolution paths index axis j, and a diagonal covariance does not stay
            # diagonal under an arbitrary rotation. Hold the basis fixed between absorbs
            # and let the covariance scaling shape the search instead; absorb still
            # redraws the basis and resets the CMA state above.
            pass
        else:
            # Re-anchor the coordinate origin, then rotate the basis for next step.
            # The represented point is base + P @ coords, so replacing P while coords are
            # non-zero moves the weights with nothing evaluated behind it. Folding coords
            # into base first keeps the point fixed. Only the origin moves, so duals,
            # momentum and CMA state stay as they are (duals are invalidated below).
            #
            # Rotate every step rather than every N: the OT solve already extracts one
            # step's worth of information. For large full_dim the basis QR/SVD dominates
            # the step and far exceeds the tiny-V Sinkhorn solve, so it runs in fp32.
            state.base_params, _ = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                state.X.reshape(-1)[: adaptive_sub.subspace_dim],
            )
            state.X = torch.zeros_like(state.X)

            # Sparse projection: use seed increment instead of QR rotation
            from .projection import SparseRandomProjection

            if isinstance(state.projection, SparseRandomProjection):
                # Sparse projection doesn't support QR rotation; increment seed
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
                )
            # Reset duals after rotation (cost geometry changed)
            state.f = None
            state.g = None
            state.prev_prev_f = None
            state.prev_prev_g = None
            # Invalidate EMA transport direction and reuse cache (geometry changed)
            opt._transport_direction_ema = None
            opt._transport_direction = None
            opt._invalidate_reuse_cache()

    # Per-layer projections (Hybrid / Factored): displacement tracking, absorb, rotation.
    if opt._per_layer_projections:
        hybrid_sub = opt.subspace

        # Compute displacement in subspace coords
        post_step_sub_coords = state.X.reshape(-1)[: hybrid_sub.subspace_dim]
        displacement = post_step_sub_coords - _pre_step_sub_coords

        # Update displacement history (rolling buffer)
        idx = state.displacement_history_idx
        state.displacement_history[idx] = displacement
        state.displacement_history_idx = (idx + 1) % hybrid_sub.displacement_history_size
        state.displacement_history_count = min(
            state.displacement_history_count + 1,
            hybrid_sub.displacement_history_size,
        )

        # Check for absorb trigger
        should_absorb = hybrid_sub.should_absorb(
            state.stagnation_count,
            state.iteration_count,
        )

        if should_absorb:
            # Absorb: fold perturbation into base, zero coords, new projections
            full_flat_sub = state.X.reshape(-1)[: hybrid_sub.subspace_dim]
            new_base, _zeroed = hybrid_sub.absorb(
                state.hybrid_projections,
                state.base_params,
                full_flat_sub,
            )
            state.base_params = new_base
            # Reset subspace coordinates to zero
            state.X = torch.zeros_like(state.X)
            # Regenerate ALL per-layer projections. Deliberately NOT a fresh draw:
            # init_projections seeds every layer with step=0, so each absorb returns a
            # bit-identical basis and the run stays in one fixed affine subspace.
            # Redrawing per absorb cost 0.61 -> 0.33 median test accuracy on a 101k
            # hard-threshold LIF SNN, since it also discards the state reset below.
            #
            # absorb_aligned_active (opt-in) instead biases the new basis toward the
            # window's productive directions by displacement-SVD, before the history is
            # zeroed. Safe here because the duals are reset in this same block.
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
            state.displacement_history.zero_()
            if state.displacement_history_full is not None:
                state.displacement_history_full.zero_()
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            # Reset duals (cost landscape changed)
            state.f = None
            state.g = None
            state.prev_prev_f = None
            state.prev_prev_g = None
            # Invalidate EMA transport direction (cost geometry changed)
            opt._transport_direction_ema = None
            opt._transport_direction = None
            state.absorb_count += 1
            # Clear the stagnation counter, else absorb_mode='stagnation' stays
            # triggered on a plateau and redraws the basis every step. The absorb
            # re-anchors the origin, so the old loss history no longer applies.
            state.stagnation_count = 0
            state.prev_loss = _cost_mean
            # Invalidate cached cost/probe state (cost landscape changed after absorb)
            opt._invalidate_reuse_cache()
            opt._newton_direction = None
            opt._prev_descent_direction = None
            opt._prev_descent_direction_finite = False
            # Momentum velocity lives in the pre-absorb basis; zero it.
            if opt.use_momentum and state.velocity is not None:
                state.velocity = torch.zeros_like(state.velocity)
        else:
            # Rotate all per-layer projections for next step
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
                # Re-anchor before swapping the basis. The represented point is
                # base + P @ coords, so replacing P while coords are non-zero moves the
                # weights with nothing evaluated behind it. Fold the coordinates in
                # under the OLD projections first, then zero them.
                state.base_params, _ = hybrid_sub.absorb(
                    state.hybrid_projections,
                    state.base_params,
                    state.X.reshape(-1)[: hybrid_sub.subspace_dim],
                )
                state.X = torch.zeros_like(state.X)
                state.f = None
                state.g = None
                state.prev_prev_f = None
                state.prev_prev_g = None
                # Invalidate EMA transport direction and reuse cache (geometry changed)
                opt._transport_direction_ema = None
                opt._transport_direction = None
                opt._invalidate_reuse_cache()
                opt._newton_direction = None
                opt._prev_descent_direction = None
                opt._prev_descent_direction_finite = False
                # Momentum velocity lives in the pre-rotation basis.
                if opt.use_momentum and state.velocity is not None:
                    state.velocity = torch.zeros_like(state.velocity)
                state.hybrid_projections = new_projections
                if hasattr(hybrid_sub, "build_fused_projection"):
                    hybrid_sub.build_fused_projection(new_projections)

    # Periodic absorb: fold perturbation into base, zero subspace
    # Only for non-adaptive/non-hybrid subspaces; their absorb handled separately.
    # Semantics: absorb AFTER every N steps (iteration_count already incremented above).
    # E.g., absorb_every=10 triggers at iteration_count=10,20,30,...
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
        # Zero the particle array (subspace coords reset)
        state.X = torch.zeros_like(state.X)
        # Reset dual potentials since cost landscape changed
        state.f = None
        state.g = None
        state.prev_prev_f = None
        state.prev_prev_g = None
        # Invalidate EMA transport direction and reuse cache (geometry changed)
        opt._transport_direction_ema = None
        opt._transport_direction = None
        opt._invalidate_reuse_cache()
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)

    # Rank schedule transition check
    if opt._rank_schedule is not None and opt.subspace is not None:
        current_rank = opt._rank_schedule.at(state.iteration_count)
        prev_rank = opt._rank_schedule.at(state.iteration_count - 1)
        if current_rank != prev_rank:
            opt._transition_rank(current_rank)

    # Write particles back to model
    opt._sync_model()

    return _cost_mean
