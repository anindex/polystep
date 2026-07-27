"""Block-wise step methods: per-block and subspace+block OT solves."""

from __future__ import annotations

import logging
from typing import Callable

import torch

from .blockwise import (
    split_particles,
    reassemble_blocks,
    split_subspace_to_blocks,
    reassemble_blocks_to_subspace,
    layout_flat_to_block_flat,
    blocks_to_layout_flat,
    blocks_to_layout_flat_batch,
)
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

logger = logging.getLogger(__name__)


def step_blockwise(opt, closure: Callable) -> float:
    """Block-wise step: per-block OT solve with full-model closure calls.

    Each block has its own polytope, rotation, and OT solve in
    particle_dim space (typically 2D). For cost evaluation, the full
    model config is reconstructed by replacing the probed particle row
    within that block. Uses chunked evaluation to bound memory.
    """
    state = opt._state
    X = state.X  # (total_particles, particle_dim)
    iteration = state.iteration_count
    device = X.device
    blocks = opt._blocks

    # Resolve epsilon and radii (scheduled radii bypass epsilon multiplication)
    current_eps = opt._get_epsilon(iteration)
    radius_mult = state.radius_multiplier if opt.use_adaptive_radius else 1.0
    _sr = opt._get_step_radius(iteration)
    _pr = opt._get_probe_radius(iteration)
    step_r = _sr * (1.0 if hasattr(opt.step_radius, "at") else current_eps) * radius_mult
    probe_r = _pr * (1.0 if hasattr(opt.probe_radius, "at") else current_eps) * radius_mult

    # Probe-radius jitter (Thm. 4.2 condition (iv); no-op when probe_radius_jitter == 0).
    probe_r = opt._apply_probe_radius_jitter(probe_r)

    # state.X is always 2-D (num_particles, particle_dim); no reshape needed.

    # Convert layout-indexed flat to block-indexed flat before splitting.
    # Per-layer blocks pad each entry independently, creating different
    # offsets from ParamLayout (contiguous concat + single end pad).
    total_flat_size = sum(b.flat_end - b.flat_start for b in blocks)
    block_flat = layout_flat_to_block_flat(
        X.reshape(-1),
        blocks,
        opt.layout,
    )
    block_X_2d = block_flat.reshape(-1, opt._particle_dim)
    all_block_particles = split_particles(block_X_2d, blocks)

    ent_eps = opt._get_ent_epsilon(iteration)
    ot_epsilon = ent_eps if ent_eps is not None else current_eps

    updated_block_particles = []
    new_block_duals = []
    new_block_descent_dirs = []  # For biased rotation in next step
    total_ent_cost = 0.0
    # Per-block scalars accumulated as device tensors and summed once
    # after the loop, to avoid one GPU->CPU sync per block per step.
    block_disp_terms: list = []
    block_model_loss_terms: list = []
    all_converged = True
    total_particles = 0
    num_blocks_counted = 0

    # Per-block descent directions for biased rotation (populated from previous step)
    _block_descent_dirs = getattr(opt, "_prev_block_descent_directions", None)

    # Cache the transfer instead of calling .to() every step, matching the monolithic path.
    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes
    chunk = opt.chunk_size or 1024  # default chunk for block-wise

    # base_flat holds every block at its current value and is invariant across
    # the block loop (updates are collected and applied after it). Build it once
    # and reuse a single scatter buffer for every block and chunk.
    base_flat = reassemble_blocks(all_block_particles, blocks, total_flat_size)
    base_batch_buf = base_flat.new_empty((chunk, total_flat_size))

    # Drop per-block dual momentum history across an epsilon jump so the
    # warm-start isn't extrapolated over a large epsilon change (matches monolithic).
    if state.last_solve_eps is not None and (
        ot_epsilon / state.last_solve_eps > 2.0 or state.last_solve_eps / ot_epsilon > 2.0
    ):
        state._prev_prev_block_duals = None

    for block_idx, block in enumerate(blocks):
        block_X = all_block_particles[block_idx]
        block_dim = block.particle_dim

        if block_X.dim() == 1:
            block_X = block_X.unsqueeze(0)
        P_block = block_X.shape[0]

        # Per-block polytope
        block_polytope_verts = opt._block_polytopes[block_idx]
        if block_polytope_verts.device != device or block_polytope_verts.dtype != X.dtype:
            block_polytope_verts = block_polytope_verts.to(device=device, dtype=X.dtype)
            opt._block_polytopes[block_idx] = block_polytope_verts

        rot_mats = get_random_rotation_matrices(
            P_block,
            block_dim,
            device=device,
            dtype=X.dtype,
            generator=opt._generator,
        )

        if (
            opt.biased_rotation
            and _block_descent_dirs is not None
            and block_idx < len(_block_descent_dirs)
            and _block_descent_dirs[block_idx] is not None
            and _block_descent_dirs[block_idx].shape == (P_block, block_dim)
        ):
            rot_mats = apply_biased_rotation(rot_mats, _block_descent_dirs[block_idx])

        X_vertices, rotated = opt._compiled.rotate_and_translate(
            rot_mats,
            block_polytope_verts,
            block_X,
            step_r,
        )

        # Probe generation
        X_probe = opt._compiled.compute_probe_points(
            block_X,
            rotated,
            probes,
            probe_r,
        )

        # Build full params with only this block varying.
        # For each probe (i, v, k), construct full flat config by
        # assembling all blocks and replacing particle i in this block.
        P, V, K, D = X_probe.shape
        total_evals = P * V * K

        losses = X_probe.new_empty(total_evals, dtype=loss_buffer_dtype(X_probe.dtype))
        _all_indices = torch.arange(total_evals, device=device)
        _d_offsets = torch.arange(D, device=device)
        for chunk_start in range(0, total_evals, chunk):
            chunk_end = min(chunk_start + chunk, total_evals)
            chunk_size_actual = chunk_end - chunk_start

            # Refill the reuse buffer with base_flat instead of allocating a
            # fresh (chunk x total_flat_size) tensor each chunk.
            base_batch = base_batch_buf[:chunk_size_actual]
            base_batch.copy_(base_flat)

            global_indices = _all_indices[chunk_start:chunk_end]  # view, no alloc
            i_idx = global_indices // (V * K)
            vk = global_indices % (V * K)
            v_idx = vk // K
            k_idx = vk % K

            # Replace probed particle rows in each config, all D coordinates in one
            # scatter instead of one indexed assignment per coordinate.
            row_starts = block.flat_start + i_idx * D
            col_idx = row_starts.unsqueeze(1) + _d_offsets  # (chunk, D)
            base_batch.scatter_(1, col_idx, X_probe[i_idx, v_idx, k_idx])

            # Map block-indexed flat vector to layout-indexed flat vector.
            # Per-layer blocks pad each entry independently, creating
            # different offsets from ParamLayout.
            batch_for_layout = blocks_to_layout_flat_batch(
                base_batch,
                blocks,
                opt.layout,
            )

            # Convert to param dicts and call closure
            batched_params = opt.layout.batch_unflatten(batch_for_layout)
            chunk_losses = closure(batched_params)
            # Ensure FP32 for Sinkhorn solver numerical stability
            losses[chunk_start:chunk_end] = chunk_losses.to(losses.dtype)

        if K == 1:
            # K=1 fast path: no averaging needed
            cost_matrix = losses.reshape(P, V)
        else:
            cost_matrix = losses.reshape(P, V, K).mean(dim=-1)

        # Sanitize cost (FP32 promote + finite penalty), branch-free / no sync.
        cost_matrix = sanitize_cost(cost_matrix)

        # Per-block OT solve with dual momentum extrapolation
        opt.solver.epsilon = ot_epsilon
        block_a = torch.ones(P_block, device=device, dtype=X.dtype) / P_block
        if opt._use_fused_softmax:
            # Fused path: softmax + vertex-free projection in one compiled call.
            scaled_cost = scale_cost_matrix(recenter_cost(cost_matrix)[0], opt.scale_cost)
            X_new_block, transport_matrix = opt._compiled.fused_softmax_project(
                scaled_cost,
                ot_epsilon,
                block_a,
                block_polytope_verts,
                rot_mats,
                step_r,
                block_X,
                scale_cost_mean=False,
            )
            # Per-block model loss is tracked below; the OT cost is unused here.
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
            init_f, init_g = state.block_duals[block_idx]
            # Apply dual momentum per block
            if (
                opt._dual_momentum_beta > 0.0
                and init_f is not None
                and hasattr(state, "_prev_prev_block_duals")
                and state._prev_prev_block_duals is not None
                and block_idx < len(state._prev_prev_block_duals)
            ):
                ppf, ppg = state._prev_prev_block_duals[block_idx]
                if ppf is not None and ppg is not None:
                    beta_dm = opt._dual_momentum_beta
                    init_f = init_f + beta_dm * (init_f - ppf)
                    init_g = init_g + beta_dm * (init_g - ppg)
                    max_abs = 80.0 * max(ot_epsilon, 0.01)
                    init_f = init_f.clamp(-max_abs, max_abs)
                    init_g = init_g.clamp(-max_abs, max_abs)

            # a defaults to uniform 1/P_block inside the solver (equals block_a);
            # pass None to skip the host-syncing user-marginal validation.
            solve_bw_kwargs = dict(
                cost_matrix=cost_matrix,
                init_f=init_f,
                init_g=init_g,
                scale_cost=opt.scale_cost,
            )
            if isinstance(opt.solver, SinkhornSolver):
                # Forward previous solve's epsilon so warm-started duals get
                # rescaled when the epsilon schedule moves.
                last_eps = state.last_solve_eps
                if last_eps is not None:
                    solve_bw_kwargs["init_eps"] = last_eps
            ot_result = opt.solver.solve(**solve_bw_kwargs)

            X_new_block = opt._compiled.barycentric_projection(
                ot_result.matrix,
                block_a,
                X_vertices,
            )
            # Non-fused solvers return a Python-float ent_reg_cost already.
            total_ent_cost += ot_result.ent_reg_cost

        # Track displacement and descent direction for biased rotation
        block_descent = (X_new_block - block_X).detach()
        block_disp_terms.append(torch.sum(block_descent**2, dim=-1).sum())
        total_particles += P_block
        new_block_descent_dirs.append(block_descent)

        updated_block_particles.append(X_new_block)
        new_block_duals.append(
            (
                ot_result.f.detach() if ot_result.f is not None else None,
                ot_result.g.detach() if ot_result.g is not None else None,
            )
        )
        block_model_loss_terms.append(cost_matrix.mean().detach())
        num_blocks_counted += 1
        all_converged = all_converged and ot_result.converged

    # Save per-block descent directions for biased rotation in next step
    if opt.biased_rotation:
        opt._prev_block_descent_directions = new_block_descent_dirs

    # Reassemble and convert back to layout-indexed format
    full_flat = reassemble_blocks(updated_block_particles, blocks, total_flat_size)
    layout_flat_new = blocks_to_layout_flat(full_flat, blocks, opt.layout)
    X_new_full = layout_flat_new.reshape(X.shape)

    # Momentum (on full particles)
    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(
            iteration,
            opt.max_iterations,
            opt.momentum_init,
            opt.momentum_final,
        )
        X_final, vel_new = apply_momentum(
            X,
            X_new_full,
            state.velocity,
            beta,
            opt.velocity_lr,
        )
        state.velocity = vel_new
        state.X = X_final
    else:
        state.X = X_new_full

    # NaN-safe state update - revert X, velocity, and duals if NaN after projection
    _blockwise_nan_reverted = False
    if not torch.isfinite(state.X).all():
        state.X = X.clone()
        state.block_duals = [(None, None) for _ in blocks]
        # Reset velocity to prevent NaN propagation through momentum
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # Clear cached state that could propagate the NaN-producing direction
        opt._transport_direction_ema = None
        opt._prev_descent_direction = None
        opt._prev_descent_direction_finite = False
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = None
        if opt.biased_rotation:
            opt._prev_block_descent_directions = None
        _blockwise_nan_reverted = True

    # Capture transport direction for amortized OT.
    if opt.amortize_steps > 1:
        if _blockwise_nan_reverted:
            opt._transport_direction = None
            opt._transport_direction_ema = None
        else:
            # Pure OT step, taken before momentum. Reading state.X here would fold in
            # beta * velocity, which the next cheap step re-applies on top of its own
            # momentum. The monolithic step captures X_bary - X for the same reason.
            raw_direction = (X_new_full - X).detach()
            opt._transport_direction = raw_direction
            alpha = opt.amortize_ema
            if opt._transport_direction_ema is None:
                opt._transport_direction_ema = raw_direction
            else:
                opt._transport_direction_ema = alpha * opt._transport_direction_ema + (1.0 - alpha) * raw_direction

    # Reduce per-block accumulators with one host transfer each.
    total_model_loss = torch.stack(block_model_loss_terms).sum().item() if block_model_loss_terms else 0.0
    # On a revert state.X is back at X, so the per-block accumulator describes a move
    # that did not happen; reporting it puts an inf in the displacement history.
    total_disp = (
        0.0 if _blockwise_nan_reverted else (torch.stack(block_disp_terms).sum().item() if block_disp_terms else 0.0)
    )

    # Adaptive radius (use model loss, not OT regularized cost)
    avg_model_loss = total_model_loss / num_blocks_counted if num_blocks_counted > 0 else total_ent_cost
    # Tracked unconditionally: absorb_mode="stagnation" reads this counter, which
    # use_adaptive_radius (default False) does not gate.
    _prev_loss_for_radius = state.prev_loss
    state.stagnation_count, state.prev_loss = update_stagnation(
        avg_model_loss,
        state.prev_loss,
        state.stagnation_count,
        stagnation_threshold=opt.stagnation_threshold,
    )
    if opt.use_adaptive_radius:
        state.radius_multiplier, state.stagnation_count = update_radius_multiplier(
            avg_model_loss,
            _prev_loss_for_radius,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )

    # Update diagnostics
    disp_sqnorm = total_disp / total_particles if total_particles > 0 else 0.0
    state.costs.append(avg_model_loss)
    state.linear_convergence.append(all_converged)
    state.displacement_sqnorms.append(disp_sqnorm)
    state.iteration_count += 1
    # Only update block duals if no NaN revert occurred (otherwise stale NaN-causing
    # duals would overwrite the clean reset done above)
    if not _blockwise_nan_reverted:
        # Save previous block duals for dual momentum extrapolation
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = (
                [
                    (f.clone() if f is not None else None, g.clone() if g is not None else None)
                    for f, g in state.block_duals
                ]
                if state.block_duals is not None
                else None
            )
        state.block_duals = new_block_duals
    state.epsilon = current_eps
    state.last_solve_eps = ot_epsilon

    # Write back to model
    opt._sync_model()

    return avg_model_loss


def step_subspace_blockwise(opt, closure: Callable) -> float:
    """Combined subspace + block-wise step: per-block OT in subspace coords.

    This mode combines the benefits of:
    1. Global subspace projection: Compresses full params (e.g., 100M) to
       subspace coords (e.g., 256), reducing memory and enabling cross-layer
       information sharing via the global projection matrix P.
    2. Per-block OT decomposition: L independent Sinkhorn solves over P/L rows
       each instead of one solve over P. The forward count is unchanged: an
       orthoplex spends 2*subspace_dim*K evaluations either way.

    Algorithm:
    a) Get current subspace coords from state.X (flattened)
    b) Split subspace coords into per-block particles
    c) For each block:
       - Sample polytope vertices in block's subspace particle space
       - Compute cost matrix via GLOBAL evaluation:
         * For each probe: apply global projection P to get full params
         * Call closure to evaluate loss on full model
       - Solve per-block OT
       - Barycentric projection to update block particles
    d) Reassemble updated blocks into new subspace coords
    e) Update state.X
    f) Check synchronized absorb (single rotation for all blocks)
    g) If absorb: rotate projection, reset ALL block coords to zero

    Note on cost evaluation (GLOBAL vs layer-local):
    This implementation uses GLOBAL cost evaluation: each probe perturbs
    ONE block's subspace coords, then applies the global projection P to
    reconstruct full params, and evaluates the full model forward pass.
    This captures cross-block interactions through the complete model.
    """
    state = opt._state
    X = state.X  # (num_sub_particles, subspace_particle_dim)
    iteration = state.iteration_count
    device = X.device
    blocks = opt._subspace_blocks

    # Resolve epsilon and radii (scheduled radii bypass epsilon multiplication)
    current_eps = opt._get_epsilon(iteration)
    # Use CSA sigma or heuristic radius_multiplier
    if opt.use_csa and state.use_csa:
        radius_mult = state.sigma
    elif opt.use_adaptive_radius:
        radius_mult = state.radius_multiplier
    else:
        radius_mult = 1.0
    _sr = opt._get_step_radius(iteration)
    _pr = opt._get_probe_radius(iteration)
    step_r = _sr * (1.0 if hasattr(opt.step_radius, "at") else current_eps) * radius_mult
    probe_r = _pr * (1.0 if hasattr(opt.probe_radius, "at") else current_eps) * radius_mult

    # Probe-radius jitter (Thm. 4.2 condition (iv); no-op when probe_radius_jitter == 0).
    probe_r = opt._apply_probe_radius_jitter(probe_r)

    # state.X is always 2-D (num_sub_particles, subspace_particle_dim); no reshape needed.

    # Get subspace dimension
    sub_dim = opt.subspace.subspace_dim

    # Cache the coord-to-param projection for this step, before any rotation below,
    # so probes and the end-of-step _sync_model share one basis.
    opt._update_sampling_projection()
    proj_used = opt._sampling_projection if opt._sampling_projection is not None else state.projection

    # Save pre-step subspace coords for displacement tracking
    _pre_step_sub_coords = None
    if opt._adaptive or opt._cma_subspace:
        _pre_step_sub_coords = state.X.reshape(-1)[:sub_dim].clone()

    # Split subspace coords into per-block particles
    subspace_coords_flat = X.reshape(-1)[:sub_dim]
    all_block_particles = split_subspace_to_blocks(subspace_coords_flat, blocks)

    ent_eps = opt._get_ent_epsilon(iteration)
    ot_epsilon = ent_eps if ent_eps is not None else current_eps

    updated_block_particles = []
    new_block_duals = []
    new_block_descent_dirs = []  # For biased rotation in next step
    total_ent_cost = 0.0
    # Per-block scalars accumulated as device tensors and summed once after
    # the loop to avoid one GPU->CPU sync per block per step.
    block_disp_terms: list = []
    block_model_loss_terms: list = []
    all_converged = True
    total_particles = 0
    num_blocks_counted = 0

    # Cache the transfer instead of calling .to() every step, matching the monolithic path.
    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes
    chunk = opt.chunk_size or 512  # default chunk for combined mode

    # base_subspace holds every block's coords and is invariant across the block
    # loop (updates are applied after it). Build it once and reuse one scatter
    # buffer for every block and chunk.
    base_subspace = reassemble_blocks_to_subspace(all_block_particles, blocks, sub_dim)
    # One trailing scratch column absorbs writes for the padded tail of the last block,
    # so the perturbation scatter needs no validity mask and no host sync.
    base_batch_buf = base_subspace.new_empty((chunk, sub_dim + 1))

    # Per-block descent directions for biased rotation (populated from previous step)
    _block_descent_dirs = getattr(opt, "_prev_block_descent_directions", None)

    # Drop per-block dual momentum history across an epsilon jump so the
    # warm-start isn't extrapolated over a large epsilon change (matches monolithic).
    if state.last_solve_eps is not None and (
        ot_epsilon / state.last_solve_eps > 2.0 or state.last_solve_eps / ot_epsilon > 2.0
    ):
        state._prev_prev_block_duals = None

    for block_idx, block in enumerate(blocks):
        block_X = all_block_particles[block_idx]
        block_dim = block.particle_dim

        if block_X.dim() == 1:
            block_X = block_X.unsqueeze(0)
        P_block = block_X.shape[0]

        # Per-block polytope (in subspace_particle_dim space)
        block_polytope_verts = opt._subspace_block_polytopes[block_idx]
        if block_polytope_verts.device != device or block_polytope_verts.dtype != X.dtype:
            block_polytope_verts = block_polytope_verts.to(device=device, dtype=X.dtype)
            opt._subspace_block_polytopes[block_idx] = block_polytope_verts

        # Rotation matrices for this block
        rot_mats = get_random_rotation_matrices(
            P_block,
            block_dim,
            device=device,
            dtype=X.dtype,
            generator=opt._generator,
        )

        # Same QR-based biased rotation as step_blockwise() above.
        if (
            opt.biased_rotation
            and _block_descent_dirs is not None
            and block_idx < len(_block_descent_dirs)
            and _block_descent_dirs[block_idx] is not None
            and _block_descent_dirs[block_idx].shape == (P_block, block_dim)
        ):
            rot_mats = apply_biased_rotation(rot_mats, _block_descent_dirs[block_idx])

        X_vertices, rotated = opt._compiled.rotate_and_translate(
            rot_mats,
            block_polytope_verts,
            block_X,
            step_r,
        )

        # Probe generation
        X_probe = opt._compiled.compute_probe_points(
            block_X,
            rotated,
            probes,
            probe_r,
        )

        # Build full params with only this block varying.
        # For each probe (i, v, k):
        # Create full subspace coords by assembling all blocks
        # Replace particle i in this block with probe position
        # Apply global projection P to get full params
        # Evaluate closure on full params
        P, V, K, D = X_probe.shape
        total_evals = P * V * K

        losses = X_probe.new_empty(total_evals, dtype=loss_buffer_dtype(X_probe.dtype))
        _all_indices = torch.arange(total_evals, device=device)
        _d_offsets = torch.arange(D, device=device)
        for chunk_start in range(0, total_evals, chunk):
            chunk_end = min(chunk_start + chunk, total_evals)
            chunk_size_actual = chunk_end - chunk_start

            # Refill the reuse buffer with base_subspace instead of allocating a
            # fresh (chunk x sub_dim) tensor each chunk, then perturb this block.
            padded_batch = base_batch_buf[:chunk_size_actual]
            padded_batch[:, :sub_dim].copy_(base_subspace)

            global_indices = _all_indices[chunk_start:chunk_end]  # view, no alloc
            i_idx = global_indices // (V * K)
            vk = global_indices % (V * K)
            v_idx = vk // K
            k_idx = vk % K

            # Replace particle i in this block, all D coordinates in one scatter.
            # Block flat range: [block.flat_start, block.flat_end)
            # Particle i occupies: [block.flat_start + i*D, block.flat_start + (i+1)*D)
            # Columns past sub_dim are the last block's padded tail and go to the scratch
            # column, sliced off below. A per-d loop with a boolean mask instead costs one
            # host sync per coordinate per chunk per block.
            row_starts = block.flat_start + i_idx * D
            col_idx = row_starts.unsqueeze(1) + _d_offsets  # (chunk, D)
            col_idx = torch.where(col_idx < sub_dim, col_idx, sub_dim)
            padded_batch.scatter_(1, col_idx, X_probe[i_idx, v_idx, k_idx])
            base_batch = padded_batch[:, :sub_dim]

            # Apply global projection to get full params
            # base_batch: (chunk_size, sub_dim)
            # projection: (full_dim, sub_dim)
            # reconstruct_batch needs projection argument for AdaptiveSubspace
            # Match dtype with projection for mixed precision compatibility
            if opt._mixed_precision and proj_used is not None and proj_used.dtype is not None:
                base_batch = base_batch.to(dtype=proj_used.dtype)
            if opt._adaptive or opt._cma_subspace:
                chunk_params = state.subspace.reconstruct_batch(
                    proj_used,
                    state.base_params,
                    base_batch,
                )
            elif opt._per_layer_projections:
                chunk_params = state.subspace.reconstruct_batch(
                    state.hybrid_projections,
                    state.base_params,
                    base_batch,
                )
            else:
                chunk_params = state.subspace.reconstruct_batch(
                    state.base_params,
                    base_batch,
                )

            # Evaluate full model via closure
            chunk_losses = closure(chunk_params)
            # Ensure FP32 for Sinkhorn solver numerical stability
            losses[chunk_start:chunk_end] = chunk_losses.to(losses.dtype)

        if K == 1:
            # K=1 fast path: no averaging needed
            cost_matrix = losses.reshape(P, V)
        else:
            cost_matrix = losses.reshape(P, V, K).mean(dim=-1)

        # Sanitize cost (FP32 promote + finite penalty), branch-free / no sync.
        cost_matrix = sanitize_cost(cost_matrix)

        # Per-block OT solve with dual momentum extrapolation
        opt.solver.epsilon = ot_epsilon
        block_a = torch.ones(P_block, device=device, dtype=X.dtype) / P_block
        if opt._use_fused_softmax:
            scaled_cost = scale_cost_matrix(recenter_cost(cost_matrix)[0], opt.scale_cost)
            X_new_block, transport_matrix = opt._compiled.fused_softmax_project(
                scaled_cost,
                ot_epsilon,
                block_a,
                block_polytope_verts,
                rot_mats,
                step_r,
                block_X,
                scale_cost_mean=False,
            )
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
            init_f, init_g = state.block_duals[block_idx]
            # Apply dual momentum per block
            if (
                opt._dual_momentum_beta > 0.0
                and init_f is not None
                and hasattr(state, "_prev_prev_block_duals")
                and state._prev_prev_block_duals is not None
                and block_idx < len(state._prev_prev_block_duals)
            ):
                ppf, ppg = state._prev_prev_block_duals[block_idx]
                if ppf is not None and ppg is not None:
                    beta_dm = opt._dual_momentum_beta
                    init_f = init_f + beta_dm * (init_f - ppf)
                    init_g = init_g + beta_dm * (init_g - ppg)
                    max_abs = 80.0 * max(ot_epsilon, 0.01)
                    init_f = init_f.clamp(-max_abs, max_abs)
                    init_g = init_g.clamp(-max_abs, max_abs)

            # a defaults to uniform 1/P_block inside the solver (equals block_a);
            # pass None to skip the host-syncing user-marginal validation.
            solve_sbw_kwargs = dict(
                cost_matrix=cost_matrix,
                init_f=init_f,
                init_g=init_g,
                scale_cost=opt.scale_cost,
            )
            if isinstance(opt.solver, SinkhornSolver):
                last_eps = state.last_solve_eps
                if last_eps is not None:
                    solve_sbw_kwargs["init_eps"] = last_eps
            ot_result = opt.solver.solve(**solve_sbw_kwargs)

            # Barycentric projection for this block
            X_new_block = opt._compiled.barycentric_projection(
                ot_result.matrix,
                block_a,
                X_vertices,
            )
            total_ent_cost += ot_result.ent_reg_cost

        # Track displacement and descent direction for biased rotation
        block_descent = (X_new_block - block_X).detach()
        block_disp_terms.append(torch.sum(block_descent**2, dim=-1).sum())
        total_particles += P_block
        new_block_descent_dirs.append(block_descent)

        updated_block_particles.append(X_new_block)
        new_block_duals.append(
            (
                ot_result.f.detach() if ot_result.f is not None else None,
                ot_result.g.detach() if ot_result.g is not None else None,
            )
        )
        block_model_loss_terms.append(cost_matrix.mean().detach())
        num_blocks_counted += 1
        all_converged = all_converged and ot_result.converged

    # Save per-block descent directions for biased rotation in next step
    if opt.biased_rotation:
        opt._prev_block_descent_directions = new_block_descent_dirs

    # Reassemble updated subspace coords from all blocks
    new_subspace_coords = reassemble_blocks_to_subspace(updated_block_particles, blocks, sub_dim)

    # Reshape back to (num_sub_particles, particle_dim) format for state.X
    # Pad to match original X shape
    padded_size = X.numel()
    if new_subspace_coords.numel() < padded_size:
        X_new_flat = torch.zeros(padded_size, device=device, dtype=X.dtype)
        X_new_flat[:sub_dim] = new_subspace_coords
    else:
        X_new_flat = new_subspace_coords[:padded_size]
    X_new_full = X_new_flat.reshape(X.shape)

    # Momentum (on full particles)
    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(
            iteration,
            opt.max_iterations,
            opt.momentum_init,
            opt.momentum_final,
        )
        X_final, vel_new = apply_momentum(
            X,
            X_new_full,
            state.velocity,
            beta,
            opt.velocity_lr,
        )
        state.velocity = vel_new
        state.X = X_final
    else:
        state.X = X_new_full

    # NaN-safe state update - revert X, velocity, and duals if NaN after projection
    _blockwise_nan_reverted = False
    if not torch.isfinite(state.X).all():
        state.X = X.clone()
        state.block_duals = [(None, None) for _ in blocks]
        # Reset velocity to prevent NaN propagation through momentum
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # Clear cached state that could propagate the NaN-producing direction
        opt._transport_direction_ema = None
        opt._prev_descent_direction = None
        opt._prev_descent_direction_finite = False
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = None
        if opt.biased_rotation:
            opt._prev_block_descent_directions = None
        _blockwise_nan_reverted = True

    # Capture transport direction for amortized OT.
    if opt.amortize_steps > 1:
        if _blockwise_nan_reverted:
            opt._transport_direction = None
            opt._transport_direction_ema = None
        else:
            # Pure OT step, taken before momentum. Reading state.X here would fold in
            # beta * velocity, which the next cheap step re-applies on top of its own
            # momentum. The monolithic step captures X_bary - X for the same reason.
            raw_direction = (X_new_full - X).detach()
            opt._transport_direction = raw_direction
            alpha = opt.amortize_ema
            if opt._transport_direction_ema is None:
                opt._transport_direction_ema = raw_direction
            else:
                opt._transport_direction_ema = alpha * opt._transport_direction_ema + (1.0 - alpha) * raw_direction

    # Reduce per-block accumulators with one host transfer each.
    total_model_loss = torch.stack(block_model_loss_terms).sum().item() if block_model_loss_terms else 0.0
    # On a revert state.X is back at X, so the per-block accumulator describes a move
    # that did not happen; reporting it puts an inf in the displacement history.
    total_disp = (
        0.0 if _blockwise_nan_reverted else (torch.stack(block_disp_terms).sum().item() if block_disp_terms else 0.0)
    )

    # Adaptive radius (use model loss, not OT regularized cost)
    avg_model_loss = total_model_loss / num_blocks_counted if num_blocks_counted > 0 else total_ent_cost
    # Tracked unconditionally: absorb_mode="stagnation" reads this counter, which
    # use_adaptive_radius (default False) does not gate.
    _prev_loss_for_radius = state.prev_loss
    state.stagnation_count, state.prev_loss = update_stagnation(
        avg_model_loss,
        state.prev_loss,
        state.stagnation_count,
        stagnation_threshold=opt.stagnation_threshold,
    )
    if opt.use_adaptive_radius:
        state.radius_multiplier, state.stagnation_count = update_radius_multiplier(
            avg_model_loss,
            _prev_loss_for_radius,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )

    # Update diagnostics
    disp_sqnorm = total_disp / total_particles if total_particles > 0 else 0.0
    state.costs.append(avg_model_loss)
    state.linear_convergence.append(all_converged)
    state.displacement_sqnorms.append(disp_sqnorm)
    state.iteration_count += 1
    # Only update block duals if no NaN revert occurred (otherwise stale NaN-causing
    # duals would overwrite the clean reset done above)
    if not _blockwise_nan_reverted:
        # Save previous block duals for dual momentum extrapolation.
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = (
                [
                    (f.clone() if f is not None else None, g.clone() if g is not None else None)
                    for f, g in state.block_duals
                ]
                if state.block_duals is not None
                else None
            )
        state.block_duals = new_block_duals
    state.epsilon = current_eps
    state.last_solve_eps = ot_epsilon

    # Adaptive subspace: displacement tracking, absorb, and rotation.
    # CMAAdaptiveSubspace wraps AdaptiveSubspace by composition, not inheritance, so
    # it must be tested separately or CMA runs never rotate or absorb.
    if opt._adaptive or opt._cma_subspace:
        adaptive_sub = opt.subspace

        # Compute displacement in subspace coords
        post_step_sub_coords = state.X.reshape(-1)[: adaptive_sub.subspace_dim]
        displacement = post_step_sub_coords - _pre_step_sub_coords

        # Update displacement history (rolling buffer)
        idx = state.displacement_history_idx
        state.displacement_history[idx] = displacement
        state.displacement_history_idx = (idx + 1) % adaptive_sub.displacement_history_size
        state.displacement_history_count = min(
            state.displacement_history_count + 1,
            adaptive_sub.displacement_history_size,
        )

        # Check for synchronized absorb trigger
        # In combined mode, absorb resets ALL blocks to zero and rotates global P
        should_absorb = adaptive_sub.should_absorb(
            state.stagnation_count,
            state.iteration_count,
        )

        if should_absorb:
            # SYNCHRONIZED ABSORB: fold perturbation into base, zero ALL block coords
            full_flat_sub = state.X.reshape(-1)[: adaptive_sub.subspace_dim]
            new_base, _zeroed = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                full_flat_sub,
            )
            state.base_params = new_base
            # Reset ALL subspace coordinates (all blocks) to zero
            state.X = torch.zeros_like(state.X)
            # Single global projection rotation
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
            state.displacement_history_idx = 0
            state.displacement_history_count = 0
            # Reset ALL block duals (cost landscape changed)
            state.block_duals = [(None, None) for _ in blocks]
            state.absorb_count += 1
            # Clear the stagnation counter, else absorb_mode='stagnation' stays
            # triggered on a plateau and redraws the basis every step. The absorb
            # re-anchors the origin, so the old loss history no longer applies.
            state.stagnation_count = 0
            state.prev_loss = avg_model_loss
            # Invalidate cached cost/probe state (cost landscape changed after absorb).
            opt._invalidate_reuse_cache()
            opt._newton_direction = None
            opt._prev_descent_direction = None
            opt._prev_descent_direction_finite = False
            # Reset per-block turbo state after absorb (cost landscape changed)
            if opt._dual_momentum_beta > 0.0:
                state._prev_prev_block_duals = None
            if opt.biased_rotation:
                opt._prev_block_descent_directions = None
            # CMA-ES: Reset evolution paths and covariance after absorb
            if opt._cma_subspace and (opt.use_covariance_adaptation or opt.use_csa):
                state.p_c = torch.zeros_like(state.p_c)
                state.p_sigma = torch.zeros_like(state.p_sigma)
                state.C_diag = torch.ones_like(state.C_diag)
                state.sigma = 1.0
        else:
            # Re-anchor the coordinate origin, then rotate the basis for next step.
            # The represented point is base + P @ coords, so replacing P while coords
            # are non-zero moves the weights with no evaluation behind it. Folding
            # coords into base first makes rotation point-preserving.
            state.base_params, _ = adaptive_sub.absorb(
                proj_used,
                state.base_params,
                state.X.reshape(-1)[: adaptive_sub.subspace_dim],
            )
            state.X = torch.zeros_like(state.X)

            # Sparse projection: use seed increment instead of QR rotation
            from .projection import SparseRandomProjection

            if isinstance(state.projection, SparseRandomProjection):
                new_seed = state.projection.seed + state.iteration_count
                state.projection = SparseRandomProjection(
                    full_dim=state.projection.full_dim,
                    subspace_dim=state.projection.subspace_dim,
                    seed=new_seed,
                )
            else:
                hist = (
                    state.displacement_history[: state.displacement_history_count]
                    if state.displacement_history_count > 0
                    else None
                )

                state.projection = adaptive_sub.rotate(
                    state.projection,
                    step=state.iteration_count,
                    total_steps=opt.max_iterations,
                    displacement_history=hist,
                    generator=opt._generator,
                )
            # Reset ALL block duals after rotation (cost geometry changed)
            state.block_duals = [(None, None) for _ in blocks]
            opt._invalidate_reuse_cache()

    # Write back to model
    opt._sync_model()

    return avg_model_loss
