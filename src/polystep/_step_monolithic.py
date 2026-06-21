"""Monolithic step: single OT solve over all particles.

Extracted from optimizer.py for maintainability. Called via delegation
from PolyStepOptimizer.step().
"""
from __future__ import annotations

import logging
from typing import Callable

import torch

from .costs import scale_cost_matrix
from .solvers._prelude import sanitize_cost
from .dynamics import apply_momentum, compute_momentum_coefficient, update_adaptive_radius
from .geometry import get_random_rotation_matrices
from .solvers import SinkhornSolver
from .solvers.base import SolverResult
from .adaptive_subspace import AdaptiveSubspace
from .hybrid_subspace import HybridSubspace
from .cma import (
    update_evolution_path_sigma,
    compute_heaviside_sigma,
    update_evolution_path_c,
    update_covariance_diagonal,
    compute_ot_weights,
    update_step_size_csa,
)

logger = logging.getLogger(__name__)


def step_monolithic(opt, closure: Callable) -> float:
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

    # 1. Resolve epsilon and radii
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
    _sr_scheduled = hasattr(opt.step_radius, 'at')
    _pr_scheduled = hasattr(opt.probe_radius, 'at')
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
        # Check if last 3 OT-step costs are all decreasing (and all > 0)
        costs_history = opt._ot_step_costs
        if len(costs_history) >= 3:
            recent = list(costs_history)[-3:]
            if all(c > 0 for c in recent) and all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                opt._loss_decreasing_count += 1
            else:
                opt._loss_decreasing_count = 0

            if opt._loss_decreasing_count >= 3:
                K_eff = 1

    # Select reduced probes if K_eff < K
    if K_eff < K:
        probes = probes[K // 2:K // 2 + 1]  # center scale, shape (1,)

    # 2. Generate rotation matrices: (P, pdim, pdim)
    rot_mats = get_random_rotation_matrices(
        P, pdim, device=device, dtype=X.dtype, generator=opt._generator,
    )

    # 2b. Transport-biased rotation: replace first column with previous OT descent direction
    if (opt.biased_rotation
            and opt._prev_descent_direction is not None
            and opt._prev_descent_direction.shape == (P, pdim)
            and opt._prev_descent_direction_finite):
        bias_dir = opt._prev_descent_direction  # (P, pdim)
        bias_norms = torch.norm(bias_dir, dim=-1, keepdim=True).clamp(min=1e-10)
        bias_dir_norm = bias_dir / bias_norms  # (P, pdim)
        # Save original rotation matrices BEFORE modification for fallback
        rot_mats_orig = rot_mats.clone()
        # Replace column 0 with normalized bias direction
        rot_mats[:, :, 0] = bias_dir_norm
        # Re-orthogonalize via QR decomposition (vectorized, replaces Python loop)
        # QR on the transposed matrix: columns of rot_mats are rows of rot_mats^T
        # QR preserves column 0 direction (bias), orthogonalizes the rest
        Q, _ = torch.linalg.qr(rot_mats)
        # Fall back to original for any batch element where QR produced NaN
        # (degenerate case: bias direction nearly parallel to other columns)
        valid = torch.isfinite(Q).all(dim=-1).all(dim=-1)  # (P,)
        rot_mats = torch.where(valid[:, None, None], Q, rot_mats_orig)

        # Fix determinant: QR can produce det=-1 matrices.
        # Correct by flipping the last column for matrices with det < 0.
        dets = torch.det(rot_mats)
        flip = (dets < 0).unsqueeze(-1)  # (P, 1)
        rot_mats[:, :, -1] = torch.where(flip, -rot_mats[:, :, -1], rot_mats[:, :, -1])

    # 3. Rotate + translate: X_vertices (P, V, pdim), rotated (P, V, pdim)
    X_vertices, rotated = opt._compiled.rotate_and_translate(
        rot_mats, polytope_verts, X, step_r,
    )

    # 4. Probe generation: X_probe (P, V, K, pdim)
    X_probe = opt._compiled.compute_probe_points(
        X, rotated, probes, probe_r,
    )

    # 5. Build full model configs and evaluate cost
    # For each (i, v, k), construct a full (P, pdim) config with row i
    # replaced by X_probe[i, v, k]. Then unflatten to params and evaluate.

    # Adaptive probes: Determine which particles are stagnant.
    # Stagnant particles reuse the previous step's cost matrix row instead
    # of recomputing. This saves V*K forward passes per stagnant particle.
    # We use cost-row reuse (rather than per-particle probe count) because
    # the cost evaluation loop builds flat (P*V*K) indices, making per-particle
    # probe counts require invasive restructuring of the evaluation batching.
    # Cost-row reuse is simpler and achieves the same forward-pass savings.
    stagnant_mask = opt._get_stagnant_mask(P)
    _can_reuse = (
        stagnant_mask is not None
        and opt._prev_cost_matrix is not None
        and opt._prev_cost_matrix.shape == (P, V)
        and opt._prev_k_eff == K_eff
        and opt._prev_step_r == step_r
    )

    if _can_reuse:
        # Only evaluate active (non-stagnant) particles
        active_mask = ~stagnant_mask
        active_indices = torch.where(active_mask)[0]
        P_active = active_indices.shape[0]
    else:
        active_indices = None
        P_active = P

    # All-stagnant shortcut: reuse entire previous cost matrix, skip evaluation
    if _can_reuse and P_active == 0:
        cost_matrix = opt._prev_cost_matrix.clone()
    else:
        chunk = opt.chunk_size or (P_active * V * K_eff)

        # More efficient: process all probes for each particle in a batch
        # Total evaluations: P_active * V * K_eff. Each builds full config (P, pdim)
        # flattened to (padded_size,), then batch-unflattened.
        #
        # Strategy: process in chunks to control memory.
        # Build indices for all (active_i, v, k) combinations.
        total_evals = P_active * V * K_eff
        # Pre-allocate losses tensor to avoid list+cat overhead
        losses = torch.empty(total_evals, dtype=torch.float32, device=device)

        # Pre-allocate batch_configs buffer for reuse across chunks
        _batch_configs_buf = X.unsqueeze(0).expand(chunk, -1, -1).clone() if chunk <= total_evals else None
        # Pre-allocate local range tensors (avoid repeated arange calls)
        _local_range_full = torch.arange(chunk, device=device) if chunk <= total_evals else None

        # Cache subspace info for the loop
        _is_subspace = opt.subspace is not None
        _sub_dim = state.subspace.subspace_dim if _is_subspace else 0

        # Fused subspace reconstruct + in-place forward path: avoids
        # materialising N full weight dicts when the evaluator supports it.
        _use_fused_inplace = (
            opt._hybrid
            and hasattr(state.subspace, 'apply_perturbation_inplace')
            and hasattr(opt, '_cost_evaluator')
            and getattr(opt._cost_evaluator, '_use_inplace', False)
        )

        _all_indices = torch.arange(total_evals, device=device)

        for chunk_start in range(0, total_evals, chunk):
            chunk_end = min(chunk_start + chunk, total_evals)
            chunk_size_actual = chunk_end - chunk_start

            # Vectorized index computation (replaces Python for-loop)
            global_indices = _all_indices[chunk_start:chunk_end]  # view, no alloc
            active_idx = global_indices // (V * K_eff)
            vk = global_indices % (V * K_eff)
            v_idx = vk // K_eff
            k_idx = vk % K_eff

            if active_indices is not None:
                i_idx = active_indices[active_idx]
            else:
                i_idx = active_idx

            # Reuse pre-allocated buffer when chunk size matches, otherwise allocate
            if _batch_configs_buf is not None and chunk_size_actual == chunk:
                batch_configs = _batch_configs_buf.copy_(X.unsqueeze(0).expand(chunk, -1, -1))
                local_range = _local_range_full
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
                if opt._mixed_precision and getattr(state, 'projection', None) is not None:
                    flat_sub = flat_sub.to(dtype=state.projection.dtype)
                chunk_losses = opt._cost_evaluator.evaluate_subspace_inplace(
                    state.subspace, state.hybrid_projections,
                    state.base_params, flat_sub,
                    opt._fused_inputs, opt._fused_targets,
                )
            elif _is_subspace:
                # Subspace mode: reconstruct full params from subspace coords
                flat_sub = flat_configs[:, :_sub_dim]
                if opt._mixed_precision and state.projection is not None:
                    flat_sub = flat_sub.to(dtype=state.projection.dtype)
                if opt._adaptive or opt._cma_subspace:
                    chunk_params = state.subspace.reconstruct_batch(
                        state.projection, state.base_params, flat_sub,
                    )
                elif opt._hybrid:
                    chunk_params = state.subspace.reconstruct_batch(
                        state.hybrid_projections, state.base_params, flat_sub,
                    )
                else:
                    chunk_params = state.subspace.reconstruct_batch(
                        state.base_params, flat_sub,
                    )
                chunk_losses = closure(chunk_params)
            else:
                # Trim to layout padded_size if needed
                layout_flat = opt.layout.padded_size
                if flat_configs.shape[1] >= layout_flat:
                    flat_for_layout = flat_configs[:, :layout_flat]
                else:
                    flat_for_layout = torch.nn.functional.pad(
                        flat_configs, (0, layout_flat - flat_configs.shape[1]),
                    )
                chunk_params = opt.layout.batch_unflatten(flat_for_layout)
                chunk_losses = closure(chunk_params)

            # Write directly into pre-allocated tensor (FP32 for Sinkhorn stability)
            losses[chunk_start:chunk_end] = chunk_losses.float()

        if _can_reuse and P_active < P:
            # Build full cost matrix: reuse previous rows for stagnant particles,
            # fill active rows from fresh evaluations
            if K_eff == 1:
                # K=1 fast path: no averaging needed
                active_cost = losses.reshape(P_active, V)
            else:
                active_losses_3d = losses.reshape(P_active, V, K_eff)
                active_cost = active_losses_3d.mean(dim=-1)  # (P_active, V)
            cost_matrix = opt._prev_cost_matrix.clone()
            cost_matrix[active_indices] = active_cost
            # Retain full losses_3d for quadratic model (merge active into prev)
            if opt.use_quadratic_model and opt._prev_losses_3d is not None:
                losses_3d_full = opt._prev_losses_3d.clone()
                if K_eff == 1:
                    losses_3d_full[active_indices] = losses.reshape(P_active, V, 1)
                else:
                    losses_3d_full[active_indices] = active_losses_3d
                opt._prev_losses_3d = losses_3d_full.detach()
            elif opt.use_quadratic_model:
                opt._prev_losses_3d = None  # no full data yet
        else:
            if K_eff == 1:
                # K=1 fast path: no averaging needed
                cost_matrix = losses.reshape(P, V)
                if opt.use_quadratic_model:
                    opt._prev_losses_3d = losses.reshape(P, V, 1).detach()
            else:
                losses_3d_full = losses.reshape(P, V, K_eff)
                cost_matrix = losses_3d_full.mean(dim=-1)  # (P, V)
                if opt.use_quadratic_model:
                    opt._prev_losses_3d = losses_3d_full.detach()

    # Sanitize cost (FP32 promote + finite penalty), branch-free / no host sync.
    # Protects the downstream trust-region / multifidelity readers and BOTH
    # solver paths -- the fused softmax scales this same sanitized tensor.
    cost_matrix = sanitize_cost(cost_matrix)

    # Deferred trust-region update: the cost matrix at this step is evaluated
    # at the particle position produced by the *previous* step. Comparing the
    # prediction stored on the previous step against this step's pre-OT min cost
    # gives a real predicted-vs-actual reduction ratio.
    if (opt.trust_region
            and opt._prev_predicted_improvement is not None
            and opt._prev_pre_step_loss is not None):
        current_loss_proxy = cost_matrix.min(dim=1).values.mean().item()
        actual_improvement = torch.tensor(
            [opt._prev_pre_step_loss - current_loss_proxy]
        )
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

    # Multi-fidelity screening: dampen low-contrast vertex directions
    # Uses previous step's cost to identify uninformative directions and
    # blend their cost toward the row mean, making OT focus on informative ones.
    if (opt.multifidelity_screen
            and opt._prev_cost_matrix is not None
            and opt.polytope_type == 'orthoplex'):
        prev_cost = opt._prev_cost_matrix  # (P, V)
        if prev_cost.shape == (P, V):
            pdim_local = V // 2
            # Cost contrast: |L(+e_i) - L(-e_i)| for each direction
            dir_contrast = (prev_cost[:, :pdim_local] - prev_cost[:, pdim_local:2 * pdim_local]).abs()
            # Mean contrast across particles per direction
            mean_contrast = dir_contrast.mean(dim=0)  # (pdim_local,)
            # Threshold: keep top screen_keep_ratio directions at full weight
            keep_k = max(1, int(pdim_local * opt.screen_keep_ratio))
            if keep_k < pdim_local:
                threshold = mean_contrast.topk(keep_k)[0][-1]
                # Dampen low-contrast directions: push their cost toward row mean
                dampen_mask = mean_contrast < threshold  # (pdim_local,) bool
                # Expand to vertex mask: dampen both +e_i and -e_i
                vertex_dampen = torch.zeros(V, dtype=torch.bool, device=cost_matrix.device)
                vertex_dampen[:pdim_local] = dampen_mask
                vertex_dampen[pdim_local:] = dampen_mask
                # Dampen: blend dampened vertices toward row mean
                row_mean = cost_matrix.mean(dim=1, keepdim=True)  # (P, 1)
                dampen_factor = 0.8  # blend 80% toward mean
                cost_matrix[:, vertex_dampen] = (
                    (1 - dampen_factor) * cost_matrix[:, vertex_dampen]
                    + dampen_factor * row_mean.expand_as(cost_matrix)[:, vertex_dampen]
                )

    # 6. Resolve OT epsilon
    ent_eps = opt._get_ent_epsilon(iteration)
    ot_epsilon = ent_eps if ent_eps is not None else current_eps

    # If the schedule jumped epsilon by more than 2x in one step,
    # invalidate the dual-momentum history. prev_prev_f / prev_prev_g
    # were computed at the old epsilon; extrapolating across a big
    # epsilon change pushes the warm-start far from the new fixed
    # point and the next solve has to undo the bad init.
    if (state.last_solve_eps is not None
            and (ot_epsilon / state.last_solve_eps > 2.0
                 or state.last_solve_eps / ot_epsilon > 2.0)):
        state.prev_prev_f = None
        state.prev_prev_g = None

    # 7. Dual potential momentum: extrapolate warm-start duals
    init_f_for_solve = state.f
    init_g_for_solve = state.g
    if (opt._dual_momentum_beta > 0.0
            and state.f is not None
            and state.prev_prev_f is not None
            and state.prev_prev_g is not None):
        beta = opt._dual_momentum_beta
        init_f_for_solve = state.f + beta * (state.f - state.prev_prev_f)
        init_g_for_solve = state.g + beta * (state.g - state.prev_prev_g)
        # Clamp to prevent overflow (same bounds as warm-start validation)
        max_abs = 80.0 * max(ot_epsilon, 0.01)
        init_f_for_solve = init_f_for_solve.clamp(-max_abs, max_abs)
        init_g_for_solve = init_g_for_solve.clamp(-max_abs, max_abs)

    # 7b. OT solve
    opt.solver.epsilon = ot_epsilon
    if opt._use_fused_softmax:
        # Fused path: softmax + vertex-free projection in one compiled call
        # Scale the cost outside the compiled kernel so every scale_cost mode
        # ('mean'/'max_cost'/float/None) matches the non-fused solvers; the
        # kernel then runs on the already-scaled cost (scale_cost_mean=False).
        scaled_cost = scale_cost_matrix(cost_matrix, opt.scale_cost)
        X_new_fused, transport_matrix, ent_cost_tensor = opt._compiled.fused_softmax_project(
            scaled_cost, ot_epsilon, state.a,
            opt._polytope_vertices, rot_mats, step_r, X,
            scale_cost_mean=False,
        )
        ent_cost = ent_cost_tensor.item()  # .item() OUTSIDE compiled boundary
        ot_result = SolverResult(
            matrix=transport_matrix, cost=ent_cost,
            f=None, g=None, converged=True, n_iters=1,
            ent_reg_cost=ent_cost,
        )
    else:
        # Forward the previous solve's epsilon so SinkhornSolver
        # can rescale the warm-started duals when epsilon changed.
        solve_kwargs = dict(
            cost_matrix=cost_matrix,
            a=state.a,
            init_f=init_f_for_solve,
            init_g=init_g_for_solve,
            scale_cost=opt.scale_cost,
        )
        if isinstance(opt.solver, SinkhornSolver) and state.last_solve_eps is not None:
            solve_kwargs["init_eps"] = state.last_solve_eps
        ot_result = opt.solver.solve(**solve_kwargs)
    state.last_solve_eps = ot_epsilon

    # Auto-epsilon feedback: update progressive epsilon from solver stats
    if opt._progressive_epsilon is not None:
        opt._progressive_epsilon.update(
            n_iters=ot_result.n_iters,
            max_iterations=getattr(opt.solver, 'max_iterations', 1),
            converged=ot_result.converged,
        )

    # 7b. Save pre-step subspace coords for displacement tracking (+ 13)
    # Needed for: AdaptiveSubspace displacement history AND CMA evolution paths
    # Declare unconditionally, populate conditionally
    _pre_step_sub_coords = None
    _pre_step_particle_coords = None  # Per-particle coords for rank-mu

    if opt._adaptive or opt._hybrid or (opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa)):
        sub_dim = opt.subspace.subspace_dim
        _pre_step_sub_coords = state.X.reshape(-1)[:sub_dim].clone()

    # For rank-mu: store per-particle subspace coordinates
    # Each particle's contribution to the subspace coordinate vector
    if opt._cma_subspace and opt.use_covariance_adaptation:
        # state.X is (P, particle_dim), reshape to get subspace view
        # The subspace coords are the flattened X truncated to sub_dim
        # For per-particle tracking, we need each particle's position
        # We store the full X for per-particle displacement computation
        _pre_step_particle_coords = state.X.clone()  # (P, particle_dim)

    # 7c. Capture X_vertices and X for OT-bias rotation
    # ORDERING: This capture MUST happen:
    #   - AFTER X_vertices computation (step 3, line ~427)
    #   - AFTER OT solve (step 7, line ~586) so ot_result.matrix exists
    #   - BEFORE barycentric projection (step 8, line ~614) which updates state.X
    _X_vertices_for_ot_bias = X_vertices  # (P, V, pdim) - already computed at step 3
    _X_pre_barycentric = X.clone()  # (P, pdim) - clone to preserve pre-update state

    # 7d. Compute rotation bias direction
    if opt.biased_rotation:
        if (opt.use_quadratic_model and opt._prev_losses_3d is not None and K_eff >= 2
                and opt._prev_losses_3d.shape == (P, V, K_eff)):
            # Use FD gradient from quadratic model (better signal than OT descent)
            from .quadratic_model import extract_fd_gradient
            fd_grad = extract_fd_gradient(
                opt._prev_losses_3d, probes, probe_r, pdim,
            )  # (P, pdim) in rotated frame
            # Transform to original space: grad_orig = rot_mats @ fd_grad
            # rot_mats is (P, pdim, pdim), fd_grad is (P, pdim)
            grad_orig = torch.einsum("bij,bj->bi", rot_mats, fd_grad)  # (P, pdim)
            # Descent direction = negative gradient
            opt._prev_descent_direction = -grad_orig.detach()
            opt._prev_descent_direction_finite = True
            # Compute Newton direction for momentum steps
            from .quadratic_model import extract_fd_hessian_diag, compute_newton_step
            fd_hess = extract_fd_hessian_diag(
                opt._prev_losses_3d, probes, probe_r, pdim,
            )  # (P, pdim) diagonal Hessian in rotated frame
            newton_rot = compute_newton_step(
                fd_grad, fd_hess,
                max_step_norm=step_r,
                hessian_reg=1e-4,
            )  # (P, pdim) Newton step in rotated frame
            # Transform to original space
            newton_orig = torch.einsum("bij,bj->bi", rot_mats, newton_rot)
            opt._newton_direction = newton_orig.detach()
            # Store predicted improvement and pre-step proxy loss. Both are
            # consumed at the start of the next ``step`` call, where the
            # actual loss reduction can be measured from the new cost matrix.
            if opt.trust_region:
                from .quadratic_model import compute_predicted_improvement
                opt._prev_predicted_improvement = compute_predicted_improvement(
                    fd_grad, fd_hess, newton_rot,
                ).detach()  # (P,)
                opt._prev_pre_step_loss = cost_matrix.min(dim=1).values.mean().item()
        else:
            # Fallback: OT descent direction (original biased_rotation)
            transport_weights = ot_result.matrix  # (P, V)
            vertex_offsets = X_vertices - X.unsqueeze(1)  # (P, V, pdim)
            weight_sums = transport_weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
            normalized_weights = transport_weights / weight_sums  # (P, V)
            weighted_dir = (normalized_weights.unsqueeze(-1) * vertex_offsets).sum(dim=1)  # (P, pdim)
            opt._prev_descent_direction = weighted_dir.detach()
            opt._prev_descent_direction_finite = True

    # 8. Barycentric projection: (P, pdim)
    if opt._use_fused_softmax:
        X_bary = X_new_fused  # Already computed by fused function
    else:
        X_bary = opt._compiled.barycentric_projection(
            ot_result.matrix, state.a, X_vertices,
        )

    # 9. Momentum
    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(
            iteration, opt.max_iterations,
            opt.momentum_init, opt.momentum_final,
        )
        X_new, vel_new = apply_momentum(
            X, X_bary, state.velocity, beta, opt.velocity_lr,
        )
        state.velocity = vel_new
        state.X = X_new
    else:
        state.X = X_bary

    # 9a. Newton refinement: post-OT correction using quadratic model
    if (opt._newton_refinement
            and opt._prev_losses_3d is not None
            and K_eff >= 2
            and opt.polytope_type == 'orthoplex'):
        from .quadratic_model import apply_newton_refinement
        X_refined = apply_newton_refinement(
            X_bary=state.X,
            losses_3d=opt._prev_losses_3d,
            scales=probes,
            probe_radius=probe_r,
            pdim=pdim,
            rot_mats=rot_mats,
            alpha=opt._newton_refinement_alpha,
            max_step_norm=step_r * 0.5,
            hessian_reg=1e-4,
        )
        if torch.isfinite(X_refined).all():
            state.X = X_refined
            # Newton refinement moved particles - dual potentials from
            # the previous Sinkhorn solve encode the old positions and
            # are now stale. Reset to avoid warm-starting from wrong
            # potentials on the next step.
            state.f = None
            state.g = None

    # 9b. CMA-ES updates
    if opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
        cma_sub = opt.subspace  # CMAAdaptiveSubspace
        sub_dim = cma_sub.subspace_dim

        # Compute mean displacement in subspace coordinates
        post_step_coords = state.X.reshape(-1)[:sub_dim]
        raw_displacement = post_step_coords - _pre_step_sub_coords

        # Normalize displacement for CMA-ES evolution paths.
        #
        # CMA-ES expects ||normalized_displacement|| ≈ sqrt(n) for well-calibrated
        # step-size. OT-based optimization produces much smaller displacements
        # (typically 1-5% of polytope size), so we need to scale appropriately.
        #
        # For OT-based optimization, the standard CMA-ES evolution path
        # mechanism doesn't work well because:
        # 1. OT displacement is much smaller than CMA mutations
        # 2. The displacement-sigma relationship is nonlinear in OT
        #
        # We use a simplified approach: normalize displacement by sigma
        # as in standard CMA-ES, and let the CSA update rule work with
        # bounded adjustments. The exponent clamp in cma.py prevents
        # extreme sigma changes.
        normalized_displacement = raw_displacement / state.sigma

        # 1. Update p_sigma (step-size evolution path) using normalized displacement
        state.p_sigma = update_evolution_path_sigma(
            p_sigma=state.p_sigma,
            displacement=normalized_displacement,
            C_diag=state.C_diag,
            c_sigma=opt._cma_params['c_sigma'],
            mu_eff=opt._cma_params['mu_eff'],
        )

        # 2. Compute Heaviside for stall detection
        p_sigma_norm = torch.norm(state.p_sigma).item()
        h_sigma = compute_heaviside_sigma(
            p_sigma_norm=p_sigma_norm,
            expected_norm=opt._cma_params['expected_norm'],
            n=sub_dim,
            c_sigma=opt._cma_params['c_sigma'],
            generation=state.generation,
        )

        # 3. Update p_c (covariance evolution path) using normalized displacement
        state.p_c = update_evolution_path_c(
            p_c=state.p_c,
            displacement=normalized_displacement,
            h_sigma=h_sigma,
            c_c=opt._cma_params['c_c'],
            mu_eff=opt._cma_params['mu_eff'],
        )

        # 4. Update diagonal covariance with full rank-mu (if enabled)
        if opt.use_covariance_adaptation:
            # FULL RANK-MU: Compute actual per-particle displacements
            # Each particle i moved from _pre_step_particle_coords[i] to state.X[i]
            # These per-particle displacements inform the rank-mu covariance update
            P_count = state.X.shape[0]
            pdim_local = state.X.shape[1]

            # Per-particle displacement: (P, particle_dim)
            particle_displacements = state.X - _pre_step_particle_coords

            # Project per-particle displacements to subspace dimension
            # The subspace coordinate vector is (sub_dim,) flattened from (P, pdim)
            # Each particle contributes pdim elements, so we reshape accordingly
            # For rank-mu, we need (P, sub_dim) displacement vectors
            # Strategy: Each particle's displacement in its particle_dim space
            # contributes to the overall subspace displacement pattern

            # Compute per-particle contribution to subspace displacement
            # Tile particle displacements to match subspace dimension per particle
            # sub_dim = P * pdim (approximately, truncated)
            if sub_dim >= P_count * pdim_local:
                # Full coverage: each particle's displacement maps directly
                # Pad to sub_dim per particle for consistent shape
                per_particle_sub_disp_full = torch.zeros(
                    P_count, sub_dim, device=state.X.device, dtype=state.X.dtype
                )
                # Vectorized scatter: build (P, pdim_local) index tensor for column positions
                row_idx = torch.arange(P_count, device=state.X.device)
                col_offsets = torch.arange(pdim_local, device=state.X.device).unsqueeze(0)  # (1, pdim_local)
                col_idx = row_idx.unsqueeze(1) * pdim_local + col_offsets  # (P, pdim_local)
                # Clamp to sub_dim and mask out-of-bounds columns
                valid_mask = col_idx < sub_dim
                col_idx = col_idx.clamp(max=sub_dim - 1)
                src = particle_displacements[:, :pdim_local] * valid_mask
                per_particle_sub_disp_full.scatter_(1, col_idx, src)
            else:
                # sub_dim < P*pdim: distribute particles across available dimensions
                per_particle_sub_disp_full = torch.zeros(
                    P_count, sub_dim, device=state.X.device, dtype=state.X.dtype
                )
                dims_per_particle = max(1, sub_dim // P_count)
                # Vectorized scatter: build (P, dims_per_particle) index tensor
                row_idx = torch.arange(P_count, device=state.X.device)
                col_offsets = torch.arange(dims_per_particle, device=state.X.device).unsqueeze(0)
                col_idx = row_idx.unsqueeze(1) * dims_per_particle + col_offsets  # (P, dims_per_particle)
                # Clamp to sub_dim and mask out-of-bounds columns
                valid_mask = col_idx < sub_dim
                col_idx = col_idx.clamp(max=sub_dim - 1)
                src = particle_displacements[:, :dims_per_particle] * valid_mask
                per_particle_sub_disp_full.scatter_(1, col_idx, src)

            # Normalize per-particle displacements by sigma (same as mean displacement)
            normalized_per_particle_disp = per_particle_sub_disp_full / state.sigma

            # Compute OT-informed weights for rank-mu update
            # Particles that transported more mass should have higher influence
            ot_weights = compute_ot_weights(ot_result.matrix)

            state.C_diag = update_covariance_diagonal(
                C_diag=state.C_diag,
                p_c=state.p_c,
                displacements=normalized_per_particle_disp,  # (P, sub_dim) normalized by sigma
                weights=ot_weights,
                c_1=opt._cma_params['c_1'],
                c_mu=opt._cma_params['c_mu'],
                h_sigma=h_sigma,
                c_c=opt._cma_params['c_c'],
            )
            # Enforce bounds
            state.C_diag = torch.clamp(
                state.C_diag,
                opt._cma_params['cov_min'],
                opt._cma_params['cov_max'],
            )

        # 5. Update step-size via CSA (if enabled). Pass p_sigma_norm
        # from the Heaviside check above to skip a redundant .item() sync.
        if opt.use_csa:
            state.sigma = update_step_size_csa(
                sigma=state.sigma,
                p_sigma=state.p_sigma,
                c_sigma=opt._cma_params['c_sigma'],
                d_sigma=opt._cma_params['d_sigma'],
                n=sub_dim,
                p_sigma_norm=p_sigma_norm,
            )
            # Floor sigma to prevent collapse to zero
            state.sigma = max(state.sigma, 1e-6)

        # 6. Increment generation
        state.generation += 1

    # Cache cost_matrix mean as tensor (defer .item() sync until needed)
    _cost_mean_tensor = cost_matrix.mean()

    # 10. Adaptive radius (use model loss, not OT regularized cost)
    # Single GPU-CPU sync point for cost mean
    _cost_mean = _cost_mean_tensor.item()
    if opt.use_adaptive_radius:
        rm, sc, pl = update_adaptive_radius(
            _cost_mean,
            state.prev_loss,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_threshold=opt.stagnation_threshold,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )
        state.radius_multiplier = rm
        state.stagnation_count = sc
        state.prev_loss = pl

    # NaN-safe state update - revert X, velocity, and duals if NaN after projection
    _nan_reverted = False
    if not torch.isfinite(state.X).all():
        state.X = X.clone()
        # Reset velocity to prevent NaN propagation through momentum
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
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
            raw_direction = (state.X - X).detach()
            opt._transport_direction = raw_direction
            # EMA blend: smooth transport direction across OT steps
            alpha = opt.amortize_ema
            if opt._transport_direction_ema is None:
                opt._transport_direction_ema = raw_direction
            else:
                opt._transport_direction_ema = (
                    alpha * opt._transport_direction_ema + (1.0 - alpha) * raw_direction
                )

    # NB: trust-region update happens at the start of the next call to
    # ``step`` (deferred), where we have a real post-step pre-OT measurement.

    # 11. Update diagnostics (defer GPU-CPU sync for displacement)
    per_particle_disp_sqnorms = torch.sum((state.X - X) ** 2, dim=-1)  # (P,)
    disp_sqnorm_tensor = torch.mean(per_particle_disp_sqnorms)
    state.costs.append(_cost_mean)
    state.linear_convergence.append(ot_result.converged)
    state.displacement_sqnorms.append(disp_sqnorm_tensor.item())
    state.iteration_count += 1
    # Track OT-step costs separately for adaptive_num_probe (excludes momentum steps)
    opt._ot_step_costs.append(_cost_mean)

    # 11-AP. Adaptive probes: store per-particle displacement and cost matrix
    # for next step's stagnation detection and cost-row reuse
    if opt._adaptive_probes:
        opt._prev_displacement_sqnorms = per_particle_disp_sqnorms.detach()
        opt._prev_cost_matrix = cost_matrix.detach()
        opt._prev_k_eff = K_eff
        opt._prev_step_r = step_r
    # 11-MF. Multi-fidelity screening: store cost matrix for next step's
    # direction contrast analysis (independent of adaptive probes)
    if opt.multifidelity_screen and not opt._adaptive_probes:
        opt._prev_cost_matrix = cost_matrix.detach()
        opt._prev_k_eff = K_eff
        opt._prev_step_r = step_r
    # Save duals for warm-starting; reset if NaN reversion occurred
    if _nan_reverted:
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

    # 11a. Adaptive subspace: displacement tracking, absorb, and rotation
    if opt._adaptive and isinstance(opt.subspace, AdaptiveSubspace):
        adaptive_sub = opt.subspace

        # 1. Compute displacement in subspace coords
        # The displacement is the change in the flattened subspace coordinate
        # vector after the barycentric projection + momentum update.
        post_step_sub_coords = state.X.reshape(-1)[:adaptive_sub.subspace_dim]
        displacement = post_step_sub_coords - _pre_step_sub_coords

        # 2. Update displacement history (rolling buffer)
        idx = state.displacement_history_idx
        state.displacement_history[idx] = displacement
        state.displacement_history_idx = (
            (idx + 1) % adaptive_sub.displacement_history_size
        )
        state.displacement_history_count = min(
            state.displacement_history_count + 1,
            adaptive_sub.displacement_history_size,
        )

        # 3. Check for absorb trigger
        should_absorb = adaptive_sub.should_absorb(
            state.stagnation_count,
            state.iteration_count,  # already incremented above
        )

        if should_absorb:
            # Absorb: fold perturbation into base, zero coords, new random basis.
            full_flat_sub = state.X.reshape(-1)[:adaptive_sub.subspace_dim]
            new_base, _zeroed = adaptive_sub.absorb(
                state.projection, state.base_params, full_flat_sub,
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
            # Reset displacement history
            state.displacement_history.zero_()
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
            # Increment absorb count
            state.absorb_count += 1
            # Invalidate cached cost/probe state (cost landscape changed after absorb)
            opt._prev_cost_matrix = None
            opt._prev_losses_3d = None
            opt._prev_displacement_sqnorms = None
            opt._prev_k_eff = None
            opt._prev_step_r = None
            opt._newton_direction = None
            opt._prev_descent_direction = None
            opt._prev_descent_direction_finite = False
            # CMA-ES: Reset evolution paths and covariance after absorb
            if opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
                state.p_c = torch.zeros_like(state.p_c)
                state.p_sigma = torch.zeros_like(state.p_sigma)
                state.C_diag = torch.ones_like(state.C_diag)
                state.sigma = 1.0
                # Keep generation counter (don't reset to preserve cumulation history)
        else:
            # 4. Rotate projection basis for next step
            # Design decision: rotate EVERY step (not every N steps) because
            # the OT solve already provides a natural "information extraction"
            # per step, and rotating ensures maximum exploration of the full
            # parameter space. The computational cost of QR decomposition is
            # negligible compared to the Sinkhorn solve.

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
                # Dense projection: use existing displacement-based rotation
                hist = (
                    state.displacement_history[:state.displacement_history_count]
                    if state.displacement_history_count > 0
                    else None
                )

                # Pass OT info for ot_bias mode
                ot_kwargs = {}
                if hasattr(adaptive_sub, 'rotation_mode') and adaptive_sub.rotation_mode == 'ot_bias':
                    ot_kwargs = {
                        'transport_matrix': ot_result.matrix,
                        'X_vertices': _X_vertices_for_ot_bias,
                        'X_current': _X_pre_barycentric,
                    }

                state.projection = adaptive_sub.rotate(
                    state.projection,
                    step=state.iteration_count,
                    total_steps=opt.max_iterations,
                    displacement_history=hist,
                    generator=opt._generator,
                    **ot_kwargs,
                )
            # Reset duals after rotation (cost geometry changed)
            state.f = None
            state.g = None
            state.prev_prev_f = None
            state.prev_prev_g = None
            # Invalidate EMA transport direction (cost geometry changed)
            opt._transport_direction_ema = None
            opt._transport_direction = None

    # 11a-2. HybridSubspace: displacement tracking, absorb, and rotation
    # Similar to AdaptiveSubspace but uses per-layer projections dict
    if opt._hybrid and isinstance(opt.subspace, HybridSubspace):
        hybrid_sub = opt.subspace

        # 1. Compute displacement in subspace coords
        post_step_sub_coords = state.X.reshape(-1)[:hybrid_sub.subspace_dim]
        displacement = post_step_sub_coords - _pre_step_sub_coords

        # 2. Update displacement history (rolling buffer)
        idx = state.displacement_history_idx
        state.displacement_history[idx] = displacement
        state.displacement_history_idx = (
            (idx + 1) % hybrid_sub.displacement_history_size
        )
        state.displacement_history_count = min(
            state.displacement_history_count + 1,
            hybrid_sub.displacement_history_size,
        )

        # 3. Check for absorb trigger
        should_absorb = hybrid_sub.should_absorb(
            state.stagnation_count,
            state.iteration_count,
        )

        if should_absorb:
            # Absorb: fold perturbation into base, zero coords, new projections
            full_flat_sub = state.X.reshape(-1)[:hybrid_sub.subspace_dim]
            new_base, _zeroed = hybrid_sub.absorb(
                state.hybrid_projections, state.base_params, full_flat_sub,
            )
            state.base_params = new_base
            # Reset subspace coordinates to zero
            state.X = torch.zeros_like(state.X)
            # Regenerate ALL per-layer projections
            state.hybrid_projections = hybrid_sub.init_projections(
                state.X.device, state.X.dtype,
            )
            # Reset displacement history
            state.displacement_history.zero_()
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
            # Increment absorb count
            state.absorb_count += 1
            # Invalidate cached cost/probe state (cost landscape changed after absorb)
            opt._prev_cost_matrix = None
            opt._prev_losses_3d = None
            opt._prev_displacement_sqnorms = None
            opt._prev_k_eff = None
            opt._prev_step_r = None
            opt._newton_direction = None
            opt._prev_descent_direction = None
            opt._prev_descent_direction_finite = False
        else:
            # 4. Rotate all per-layer projections for next step
            hist = (
                state.displacement_history[:state.displacement_history_count]
                if state.displacement_history_count > 0
                else None
            )
            new_projections = hybrid_sub.rotate_all(
                state.hybrid_projections,
                step=state.iteration_count,
                total_steps=opt.max_iterations,
                displacement_history=hist,
            )
            # Reset duals only if projections actually changed (rotation happened)
            if new_projections is not state.hybrid_projections:
                state.f = None
                state.g = None
                state.prev_prev_f = None
                state.prev_prev_g = None
                # Invalidate EMA transport direction (cost geometry changed)
                opt._transport_direction_ema = None
                opt._transport_direction = None
            state.hybrid_projections = new_projections
            # Build fused block-diagonal projection for fast reconstruct_batch
            if hasattr(hybrid_sub, 'build_fused_projection'):
                hybrid_sub.build_fused_projection(new_projections)

    # 11b. Periodic absorb: fold perturbation into base, zero subspace
    # Only for non-adaptive/non-hybrid subspaces; their absorb handled separately.
    # Semantics: absorb AFTER every N steps (iteration_count already incremented above).
    # E.g., absorb_every=10 triggers at iteration_count=10,20,30,...
    if (not opt._adaptive and not opt._hybrid and opt.subspace is not None
            and opt.absorb_every > 0
            and state.iteration_count % opt.absorb_every == 0):
        flat_sub = state.X.reshape(-1)[:state.subspace.subspace_dim]
        new_base, _zeroed = state.subspace.absorb(state.base_params, flat_sub)
        state.base_params = new_base
        # Zero the particle array (subspace coords reset)
        state.X = torch.zeros_like(state.X)
        # Reset dual potentials since cost landscape changed
        state.f = None
        state.g = None
        state.prev_prev_f = None
        state.prev_prev_g = None
        # Invalidate EMA transport direction (cost geometry changed)
        opt._transport_direction_ema = None
        opt._transport_direction = None

    # 11c. Rank schedule transition check
    if opt._rank_schedule is not None and opt.subspace is not None:
        current_rank = opt._rank_schedule.at(state.iteration_count)
        prev_rank = opt._rank_schedule.at(state.iteration_count - 1)
        if current_rank != prev_rank:
            opt._transition_rank(current_rank)

    # 12. Write particles back to model
    opt._sync_model()

    return _cost_mean
