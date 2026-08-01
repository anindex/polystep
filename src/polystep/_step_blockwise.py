"""Block-wise step methods: per-block and subspace+block OT solves.

Both modes run the same sweep: split the particles into blocks, and for each block
sample a polytope, score its probes, solve a small OT problem and move the block to
the barycentre. They differ only in what a candidate is (a full-model configuration
with one block row replaced, or a subspace coordinate vector) and in the basis
maintenance that follows the sweep, so the shared parts live in the helpers here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .blockwise import (
    block_to_layout_columns,
    blocks_to_layout_flat,
    layout_flat_to_block_flat,
    reassemble_blocks,
    reassemble_blocks_to_subspace,
    split_particles,
    split_subspace_to_blocks,
)
from .costs import scale_cost_matrix
from .solvers._shared import loss_buffer_dtype, recenter_cost, sanitize_cost, solver_health
from .epsilon import feed_solver_stats
from .dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
    update_radius_multiplier,
    update_stagnation,
)
from .geometry import apply_biased_rotation, get_random_rotation_matrices
from .solvers import SinkhornSolver
from .solvers.base import SolverResult
from ._step_core import invalidate_for_basis_change
from ._step_monolithic import _rotation_due, maintain_per_layer_subspace, record_displacement

logger = logging.getLogger(__name__)


@dataclass
class _BlockOutcome:
    """One block's contribution to the sweep, reduced together after the loop."""

    particles: torch.Tensor
    descent: torch.Tensor
    duals: tuple
    model_loss: torch.Tensor
    ess: torch.Tensor
    rho: torch.Tensor
    evals: int
    converged: bool
    n_iters: int
    # Candidates behind this block's means. Blocks differ in particle count, so a plain
    # average over blocks lets a one-particle block outvote a 500-particle one.
    weight: int


def _resolve_geometry(opt, state, iteration):
    """Epsilon, OT epsilon and the two radii for this step.

    Scheduled radii carry their own magnitude, so they bypass the epsilon multiply.
    """
    current_eps = opt._get_epsilon(iteration)
    radius_mult = state.radius_multiplier if opt.use_adaptive_radius else 1.0
    step_r = opt._get_step_radius(iteration) * (1.0 if hasattr(opt.step_radius, "at") else current_eps) * radius_mult
    probe_r = opt._get_probe_radius(iteration) * (1.0 if hasattr(opt.probe_radius, "at") else current_eps) * radius_mult
    # Thm. 4.2 condition (iv); no-op when probe_radius_jitter == 0.
    probe_r = opt._apply_probe_radius_jitter(probe_r)
    ent_eps = opt._get_ent_epsilon(iteration)
    return current_eps, ent_eps if ent_eps is not None else current_eps, step_r, probe_r


def _max_group(polytopes, probes) -> int:
    """The widest ``V * K`` candidate group over every block.

    Group width is per block; the candidate buffer is allocated once per sweep.
    """
    if not polytopes:
        return 1
    return max(p.shape[0] for p in polytopes) * probes.numel()


def _prepare_probes(opt, state, block, block_X, polytopes, block_idx, probes, step_r, probe_r, descent_dirs):
    """Rotate the block's polytope, place its vertices and probe points."""
    device, dtype = block_X.device, block_X.dtype
    verts = polytopes[block_idx]
    if verts.device != device or verts.dtype != dtype:
        verts = polytopes[block_idx] = verts.to(device=device, dtype=dtype)

    P_block = block_X.shape[0]
    rot_mats = get_random_rotation_matrices(
        P_block, block.particle_dim, device=device, dtype=dtype, generator=opt._generator
    )
    if (
        opt.biased_rotation
        and descent_dirs is not None
        and block_idx < len(descent_dirs)
        and descent_dirs[block_idx] is not None
        and descent_dirs[block_idx].shape == (P_block, block.particle_dim)
    ):
        rot_mats = apply_biased_rotation(rot_mats, descent_dirs[block_idx])

    X_vertices, rotated = opt._compiled.rotate_and_translate(rot_mats, verts, block_X, step_r)
    X_probe = opt._compiled.compute_probe_points(block_X, rotated, probes, probe_r)
    return verts, rot_mats, X_vertices, X_probe


def _chunk_indices(global_indices, V, K):
    """Split flat candidate ids into ``(particle, vertex, probe)`` components."""
    i_idx = global_indices // (V * K)
    vk = global_indices % (V * K)
    return i_idx, vk // K, vk % K


def _cost_from_losses(losses, P, V, K):
    """Make the losses safe for the solver, then average the probe axis away.

    Sanitize first, matching the monolithic driver: averaging first lets one non-finite
    probe carry its whole vertex to the penalty, where sanitizing first keeps the signal
    from the probes that did evaluate.
    """
    safe = sanitize_cost(losses)
    return safe.reshape(P, V) if K == 1 else safe.reshape(P, V, K).mean(dim=-1)


def _solve_block(opt, state, block_idx, cost_matrix, block_X, verts, rot_mats, X_vertices, step_r, ot_epsilon, evals):
    """Solve one block's OT problem and project it onto the barycentre."""
    device, dtype = block_X.device, block_X.dtype
    P_block = block_X.shape[0]
    opt.solver.epsilon = ot_epsilon

    if opt._use_fused_softmax:
        # Only the fused kernel reads the marginal; barycentric_projection normalises
        # by the realised row sum.
        block_a = torch.ones(P_block, device=device, dtype=dtype) / P_block
        scaled_cost = scale_cost_matrix(recenter_cost(cost_matrix)[0], opt.scale_cost)
        X_new_block, transport_matrix = opt._compiled.fused_softmax_project(
            scaled_cost, ot_epsilon, block_a, verts, rot_mats, step_r, block_X, scale_cost_mean=False
        )
        ot_result = SolverResult(
            matrix=transport_matrix, cost=0.0, f=None, g=None, converged=True, n_iters=1, ent_reg_cost=0.0
        )
    else:
        init_f, init_g = state.block_duals[block_idx]
        if (
            opt._dual_momentum_beta > 0.0
            and init_f is not None
            and getattr(state, "_prev_prev_block_duals", None) is not None
            and block_idx < len(state._prev_prev_block_duals)
        ):
            ppf, ppg = state._prev_prev_block_duals[block_idx]
            if ppf is not None and ppg is not None:
                # The solver bounds any warm start it is handed by 10 * max|C_scaled|.
                init_f = init_f + opt._dual_momentum_beta * (init_f - ppf)
                init_g = init_g + opt._dual_momentum_beta * (init_g - ppg)

        # a defaults to uniform 1/P_block inside the solver; pass None to skip the
        # host-syncing user-marginal validation.
        kwargs = dict(cost_matrix=cost_matrix, init_f=init_f, init_g=init_g, scale_cost=opt.scale_cost)
        if isinstance(opt.solver, SinkhornSolver) and state.last_solve_eps is not None:
            # Rescale warm-started duals when the epsilon schedule moves.
            kwargs["init_eps"] = state.last_solve_eps
        ot_result = opt.solver.solve(**kwargs)
        X_new_block = opt._compiled.barycentric_projection(ot_result.matrix, X_vertices)

    descent = (X_new_block - block_X).detach()
    ess, rho = solver_health(ot_result.matrix, descent, step_r)
    return _BlockOutcome(
        particles=X_new_block,
        descent=descent,
        duals=(
            ot_result.f.detach() if ot_result.f is not None else None,
            ot_result.g.detach() if ot_result.g is not None else None,
        ),
        model_loss=cost_matrix.mean().detach(),
        ess=ess,
        rho=rho,
        evals=evals,
        converged=ot_result.converged,
        n_iters=ot_result.n_iters,
        weight=cost_matrix.numel(),
    )


def _index_buffers(opt, total_evals: int, D: int, device):
    """Cached ``(arange(total_evals), arange(D))`` for a block sweep.

    The monolithic driver keeps its index tensors across steps; blockwise rebuilt them
    once per block per step. Both are read-only, so one buffer per size is enough.
    """
    cache = getattr(opt, "_block_index_buffers", None)
    if cache is None or cache[0] != device:
        cache = opt._block_index_buffers = (device, {}, {})
    _, evals_cache, dim_cache = cache
    idx = evals_cache.get(total_evals)
    if idx is None:
        idx = evals_cache[total_evals] = torch.arange(total_evals, device=device)
    off = dim_cache.get(D)
    if off is None:
        off = dim_cache[D] = torch.arange(D, device=device)
    return idx, off


def _delta_context(opt, attr: str = "_sparse_delta_evaluator"):
    """The delta evaluator and the batch it scores against, or ``None`` for the
    materializing path.

    Both need the batch the caller registered: the evaluators score a model directly
    rather than going through the closure, so without it there is nothing to score.
    """
    fused_inputs = getattr(opt, "_fused_inputs", None)
    if fused_inputs is None:
        return None, None, None
    return getattr(opt, attr, None), fused_inputs, getattr(opt, "_fused_targets", None)


def _reset_dual_momentum_on_epsilon_jump(state, ot_epsilon):
    """A large epsilon change makes the stored duals a bad extrapolation base."""
    if state.last_solve_eps is not None and (
        ot_epsilon / state.last_solve_eps > 2.0 or state.last_solve_eps / ot_epsilon > 2.0
    ):
        state._prev_prev_block_duals = None


def _finish_step(opt, state, X, X_new_full, blocks, outcomes, iteration, current_eps, ot_epsilon) -> float:
    """Momentum, NaN revert, diagnostics and dual bookkeeping shared by both modes."""
    if opt.biased_rotation:
        opt._prev_block_descent_directions = [o.descent for o in outcomes]

    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(iteration, opt.max_iterations, opt.momentum_init, opt.momentum_final)
        state.X, state.velocity = apply_momentum(X, X_new_full, state.velocity, beta, opt.velocity_lr)
    else:
        state.X = X_new_full

    nan_reverted = False
    if not torch.isfinite(state.X).all():
        # Back to the pre-step point, or to the origin when that was non-finite too.
        state.X = X.clone() if torch.isfinite(X).all() else torch.zeros_like(X)
        state.block_duals = [(None, None) for _ in blocks]
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # Clear cached state that would propagate the NaN-producing direction.
        opt._transport_direction_ema = None
        opt._prev_descent_direction = None
        opt._prev_descent_direction_finite = False
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = None
        if opt.biased_rotation:
            opt._prev_block_descent_directions = None
        nan_reverted = True

    if opt.amortize_steps > 1:
        if nan_reverted:
            opt._transport_direction_ema = None
        else:
            # The pure OT move, taken before momentum. Reading state.X would fold in
            # beta * velocity, which the next cheap step re-applies on top of its own.
            raw_direction = (X_new_full - X).detach()
            alpha = opt.amortize_ema
            if opt._transport_direction_ema is None:
                opt._transport_direction_ema = raw_direction
            else:
                opt._transport_direction_ema = alpha * opt._transport_direction_ema + (1.0 - alpha) * raw_direction

    # One host transfer per reduction, weighted by candidate count.
    _w = torch.tensor([float(o.weight) for o in outcomes], device=outcomes[0].model_loss.device if outcomes else None)
    _w = _w / _w.sum().clamp(min=1.0)
    avg_model_loss = (torch.stack([o.model_loss for o in outcomes]) * _w).sum().item() if outcomes else 0.0
    # Tracked unconditionally: absorb_mode="stagnation" reads this counter, which
    # use_adaptive_radius (default False) does not gate.
    prev_loss_for_radius = state.prev_loss
    state.stagnation_count, state.prev_loss = update_stagnation(
        avg_model_loss, state.prev_loss, state.stagnation_count, stagnation_threshold=opt.stagnation_threshold
    )
    if opt.use_adaptive_radius:
        state.radius_multiplier, state.stagnation_count = update_radius_multiplier(
            avg_model_loss,
            prev_loss_for_radius,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )

    # From the realised move: the per-block descents ran before momentum, so they
    # describe a move other than the one taken.
    state.costs.append(avg_model_loss)
    all_converged = all(o.converged for o in outcomes)
    state.linear_convergence.append(all_converged)
    # One update for the whole sweep: the slowest block sets the pace.
    feed_solver_stats(
        opt._progressive_epsilon, opt.solver, max((o.n_iters for o in outcomes), default=0), all_converged
    )
    state.displacement_sqnorms.append(torch.mean(torch.sum((state.X - X) ** 2, dim=-1)).item())
    state.record_solver_health(
        (torch.stack([o.ess for o in outcomes]) * _w).sum().item() if outcomes else None,
        (torch.stack([o.rho for o in outcomes]) * _w).sum().item() if outcomes else None,
        sum(o.evals for o in outcomes),
    )
    state.iteration_count += 1

    # Stale NaN-causing duals must not overwrite the clean reset above.
    if not nan_reverted:
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = (
                [
                    (f.clone() if f is not None else None, g.clone() if g is not None else None)
                    for f, g in state.block_duals
                ]
                if state.block_duals is not None
                else None
            )
        state.block_duals = [o.duals for o in outcomes]
    state.epsilon = current_eps
    state.last_solve_eps = ot_epsilon
    return avg_model_loss


def step_blockwise(opt, closure: Callable) -> float:
    """Per-block OT solve with full-model closure calls.

    Each block gets its own polytope, rotation and OT solve in ``particle_dim`` space.
    A candidate is the full configuration with one probed row of one block replaced,
    evaluated in chunks to bound memory.
    """
    state = opt._state
    X = state.X  # (total_particles, particle_dim)
    iteration = state.iteration_count
    device = X.device
    blocks = opt._blocks

    current_eps, ot_epsilon, step_r, probe_r = _resolve_geometry(opt, state, iteration)

    # Per-layer blocks pad each entry independently, so their offsets differ from
    # ParamLayout, which concatenates every entry and pads once at the end.
    total_flat_size = sum(b.flat_end - b.flat_start for b in blocks)
    block_flat = layout_flat_to_block_flat(X.reshape(-1), blocks, opt.layout)
    all_block_particles = split_particles(block_flat.reshape(-1, opt._particle_dim), blocks)

    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes
    # Budget the chunk on elements, like the monolithic path; a fixed chunk OOMs
    # on large models.
    chunk = opt.chunk_size or max(1, (1 << 26) // max(1, total_flat_size))
    # The delta path rounds its stride up to a whole V*K group, which exceeds chunk when
    # chunk is smaller, so the buffer covers the widest group.
    chunk = max(chunk, _max_group(opt._block_polytopes, probes))

    sparse_delta, fused_inputs, fused_targets = _delta_context(opt)
    # No assumption about the module set, so it covers the blocks the Linear-only
    # sparse delta declines (conv, norm, attention, custom).
    site_vmap = getattr(opt, "_site_vmap_evaluator", None) if fused_inputs is not None else None
    site_owner = sparse_delta if sparse_delta is not None else site_vmap
    base_sd = opt.layout.unflatten(X) if site_owner is not None else None
    pdim_arange = torch.arange(opt._particle_dim, device=device)

    # Candidates are written straight into layout order, so there is one buffer rather
    # than a block-order build plus a permutation copy per chunk. The trailing column
    # absorbs per-block padding, which has no layout counterpart.
    base_layout = blocks_to_layout_flat(
        reassemble_blocks(all_block_particles, blocks, total_flat_size), blocks, opt.layout
    )
    layout_columns = block_to_layout_columns(blocks, opt.layout, device)
    # Zeroed once: the padding columns are never written.
    batch_buf = base_layout.new_zeros((chunk, opt.layout.padded_size + 1))

    _reset_dual_momentum_on_epsilon_jump(state, ot_epsilon)
    descent_dirs = getattr(opt, "_prev_block_descent_directions", None)
    outcomes = []

    for block_idx, block in enumerate(blocks):
        block_X = all_block_particles[block_idx]
        if block_X.dim() == 1:
            block_X = block_X.unsqueeze(0)

        verts, rot_mats, X_vertices, X_probe = _prepare_probes(
            opt, state, block, block_X, opt._block_polytopes, block_idx, probes, step_r, probe_r, descent_dirs
        )

        P, V, K, D = X_probe.shape
        total_evals = P * V * K
        losses = X_probe.new_empty(total_evals, dtype=loss_buffer_dtype(X_probe.dtype))
        all_indices, d_offsets = _index_buffers(opt, total_evals, D, device)

        # The delta path scores a whole particle's group at once, so a chunk holds
        # whole groups: a ragged one would split a group across two calls.
        group = V * K
        # A single-entry block maps block position to layout position by a constant
        # shift, so the run a candidate perturbs stays contiguous and its layout offset
        # is arithmetic rather than a device read. A grouped block can straddle two
        # entries, so it keeps the dense path.
        entry = opt.layout.entries[block.leaf_indices[0]] if len(block.leaf_indices) == 1 else None
        block_chunk = max(group, (chunk // group) * group) if site_owner is not None and entry else chunk

        for chunk_start in range(0, total_evals, block_chunk):
            chunk_end = min(chunk_start + block_chunk, total_evals)
            width = chunk_end - chunk_start
            i_idx, v_idx, k_idx = _chunk_indices(all_indices[chunk_start:chunk_end], V, K)

            if entry is not None and site_owner is not None:
                offsets = entry.offset + i_idx * D
                span = (entry.offset + (chunk_start // group) * D, entry.offset + ((chunk_end - 1) // group) * D + D)
                owner = site_owner
                site = owner.resolve_site(offsets, D, span)
                if site is None and site_vmap is not None and owner is not site_vmap:
                    # The sparse-delta correction is confined to Linear layers; the
                    # site-aware vmap still shares the graph ahead of any entry.
                    owner = site_vmap
                    site = owner.resolve_site(offsets, D, span)
                if site is not None:
                    losses[chunk_start:chunk_end] = owner.evaluate(
                        base_sd,
                        site if owner is site_vmap else site.key,
                        offsets[::group].unsqueeze(1) + pdim_arange - site.offset,
                        X_probe[i_idx, v_idx, k_idx].reshape(width // group, group, D),
                        fused_inputs,
                        fused_targets,
                    ).to(losses.dtype)
                    continue

            batch = batch_buf[:width]
            batch[:, : opt.layout.padded_size].copy_(base_layout)
            # All D coordinates of the probed row in one scatter, mapped to layout
            # columns; a per-coordinate loop costs a host sync each.
            col_idx = layout_columns[(block.flat_start + i_idx * D).unsqueeze(1) + d_offsets]
            batch.scatter_(1, col_idx, X_probe[i_idx, v_idx, k_idx])

            chunk_losses = closure(opt.layout.batch_unflatten(batch[:, : opt.layout.padded_size]))
            losses[chunk_start:chunk_end] = chunk_losses.to(losses.dtype)

        cost_matrix = _cost_from_losses(losses, P, V, K)
        outcomes.append(
            _solve_block(
                opt,
                state,
                block_idx,
                cost_matrix,
                block_X,
                verts,
                rot_mats,
                X_vertices,
                step_r,
                ot_epsilon,
                total_evals,
            )
        )

    full_flat = reassemble_blocks([o.particles for o in outcomes], blocks, total_flat_size)
    X_new_full = blocks_to_layout_flat(full_flat, blocks, opt.layout).reshape(X.shape)

    avg_model_loss = _finish_step(opt, state, X, X_new_full, blocks, outcomes, iteration, current_eps, ot_epsilon)
    opt._sync_model()
    return avg_model_loss


def step_subspace_blockwise(opt, closure: Callable) -> float:
    """Per-block OT in subspace coordinates.

    A global projection compresses the full parameters to subspace coordinates, and
    the blocks partition those coordinates into independent OT solves. The forward
    count is unchanged: an orthoplex spends ``2 * subspace_dim * K`` evaluations
    either way.

    Cost evaluation stays global. Each probe perturbs one block's coordinates, the
    projection reconstructs the full parameters, and the whole model is evaluated, so
    cross-block interaction is captured. Absorption and rotation are synchronised: a
    single basis serves every block, so it is folded and redrawn once per sweep.
    """
    state = opt._state
    X = state.X  # (num_sub_particles, subspace_particle_dim)
    iteration = state.iteration_count
    device = X.device
    blocks = opt._subspace_blocks
    sub_dim = opt.subspace.subspace_dim

    current_eps, ot_epsilon, step_r, probe_r = _resolve_geometry(opt, state, iteration)

    # Cache the coord-to-param projection before any rotation below, so the probes
    # and the closing _sync_model share one basis.
    opt._update_sampling_projection()
    proj_used = opt._sampling_projection if opt._sampling_projection is not None else state.projection

    pre_step_sub_coords = None
    if opt._adaptive or opt._cma_subspace or opt._per_layer_projections:
        pre_step_sub_coords = X.reshape(-1)[:sub_dim].clone()

    all_block_particles = split_subspace_to_blocks(X.reshape(-1)[:sub_dim], blocks)

    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes
    # A chunk costs the coordinate buffer plus one full weight set per candidate, since
    # any chunk the delta path declines falls back to reconstruct_batch.
    chunk = opt.chunk_size or max(1, (1 << 26) // max(1, sub_dim + opt.layout.total_params))
    # See step_blockwise: the dense fallback inherits the group-rounded stride.
    chunk = max(chunk, _max_group(opt._subspace_block_polytopes, probes))

    # A candidate perturbs one particle's coordinate run, so a run that sits inside one
    # layer's coordinate block moves only that layer's weight. The barycentre weights it
    # corrects against are reconstructed once per sweep rather than once per candidate.
    subspace_delta, fused_inputs, fused_targets = _delta_context(opt, "_subspace_delta_evaluator")
    if not opt._hybrid:
        subspace_delta = None
    bary_sd = (
        state.subspace.apply_perturbation(state.hybrid_projections, state.base_params, X.reshape(-1)[:sub_dim])
        if subspace_delta is not None
        else None
    )

    base_subspace = reassemble_blocks_to_subspace(all_block_particles, blocks, sub_dim)
    # One trailing scratch column absorbs writes for the padded tail of the last
    # block, so the scatter needs no validity mask and no host sync.
    batch_buf = base_subspace.new_empty((chunk, sub_dim + 1))

    _reset_dual_momentum_on_epsilon_jump(state, ot_epsilon)
    descent_dirs = getattr(opt, "_prev_block_descent_directions", None)
    outcomes = []

    for block_idx, block in enumerate(blocks):
        block_X = all_block_particles[block_idx]
        if block_X.dim() == 1:
            block_X = block_X.unsqueeze(0)

        verts, rot_mats, X_vertices, X_probe = _prepare_probes(
            opt,
            state,
            block,
            block_X,
            opt._subspace_block_polytopes,
            block_idx,
            probes,
            step_r,
            probe_r,
            descent_dirs,
        )

        P, V, K, D = X_probe.shape
        total_evals = P * V * K
        losses = X_probe.new_empty(total_evals, dtype=loss_buffer_dtype(X_probe.dtype))
        all_indices, d_offsets = _index_buffers(opt, total_evals, D, device)

        # The delta path scores a whole particle's group at once, so a chunk holds
        # whole groups: a ragged one would split a group across two calls.
        group = V * K
        block_chunk = max(group, (chunk // group) * group) if subspace_delta is not None else chunk

        for chunk_start in range(0, total_evals, block_chunk):
            chunk_end = min(chunk_start + block_chunk, total_evals)
            width = chunk_end - chunk_start
            i_idx, v_idx, k_idx = _chunk_indices(all_indices[chunk_start:chunk_end], V, K)

            if subspace_delta is not None:
                # This chunk covers whole groups, so its coordinate span follows from
                # arithmetic with no read back from the device.
                g0, g1 = chunk_start // group, chunk_end // group
                lo = block.flat_start + g0 * D
                spec = subspace_delta.resolve_site(state.subspace, lo, block.flat_start + g1 * D, sub_dim)
                if spec is not None:
                    deltas = (X_probe[i_idx, v_idx, k_idx] - block_X[i_idx]).reshape(g1 - g0, group, D)
                    losses[chunk_start:chunk_end] = subspace_delta.evaluate(
                        state.subspace,
                        state.hybrid_projections,
                        bary_sd,
                        spec,
                        lo - spec.flat_start,
                        deltas,
                        fused_inputs,
                        fused_targets,
                    ).to(losses.dtype)
                    continue

            padded = batch_buf[:width]
            padded[:, :sub_dim].copy_(base_subspace)
            col_idx = (block.flat_start + i_idx * D).unsqueeze(1) + d_offsets
            padded.scatter_(1, torch.where(col_idx < sub_dim, col_idx, sub_dim), X_probe[i_idx, v_idx, k_idx])

            coords = padded[:, :sub_dim]
            if opt._mixed_precision and proj_used is not None and proj_used.dtype is not None:
                coords = coords.to(dtype=proj_used.dtype)
            chunk_losses = closure(_reconstruct(opt, state, proj_used, coords))
            losses[chunk_start:chunk_end] = chunk_losses.to(losses.dtype)

        cost_matrix = _cost_from_losses(losses, P, V, K)
        outcomes.append(
            _solve_block(
                opt,
                state,
                block_idx,
                cost_matrix,
                block_X,
                verts,
                rot_mats,
                X_vertices,
                step_r,
                ot_epsilon,
                total_evals,
            )
        )

    new_coords = reassemble_blocks_to_subspace([o.particles for o in outcomes], blocks, sub_dim)
    padded_size = X.numel()
    if new_coords.numel() < padded_size:
        X_new_flat = torch.zeros(padded_size, device=device, dtype=X.dtype)
        X_new_flat[:sub_dim] = new_coords
    else:
        X_new_flat = new_coords[:padded_size]
    X_new_full = X_new_flat.reshape(X.shape)

    avg_model_loss = _finish_step(opt, state, X, X_new_full, blocks, outcomes, iteration, current_eps, ot_epsilon)
    _maintain_subspace(opt, state, blocks, avg_model_loss, pre_step_sub_coords)
    opt._sync_model()
    return avg_model_loss


def _reconstruct(opt, state, proj_used, coords):
    """Full parameters for a batch of subspace coordinate vectors."""
    if opt._adaptive or opt._cma_subspace:
        return state.subspace.reconstruct_batch(proj_used, state.base_params, coords)
    if opt._per_layer_projections:
        return state.subspace.reconstruct_batch(state.hybrid_projections, state.base_params, coords)
    return state.subspace.reconstruct_batch(state.base_params, coords)


def _maintain_subspace(opt, state, blocks, avg_model_loss, pre_step_sub_coords) -> None:
    """Displacement tracking, absorption and rotation after a subspace sweep.

    CMAAdaptiveSubspace wraps AdaptiveSubspace by composition, not inheritance, so it
    is tested separately or CMA runs never rotate or absorb. A per-layer subspace
    keeps its basis in ``hybrid_projections`` rather than ``state.projection``, so it
    takes the second branch.
    """
    if opt._per_layer_projections:
        if pre_step_sub_coords is not None and maintain_per_layer_subspace(
            opt, state, avg_model_loss, pre_step_sub_coords
        ):
            state.block_duals = [(None, None) for _ in blocks]
        return
    if not (opt._adaptive or opt._cma_subspace):
        return

    from .projection import SparseRandomProjection

    adaptive_sub = opt.subspace
    proj_used = opt._sampling_projection if opt._sampling_projection is not None else state.projection
    record_displacement(state, adaptive_sub, pre_step_sub_coords, proj_used)

    if adaptive_sub.should_absorb(state.stagnation_count, state.iteration_count):
        # Fold the perturbation into the base and zero every block's coordinates.
        state.base_params, _ = adaptive_sub.absorb(
            proj_used, state.base_params, state.X.reshape(-1)[: adaptive_sub.subspace_dim]
        )
        state.X = torch.zeros_like(state.X)
        if isinstance(state.projection, SparseRandomProjection):
            state.projection = SparseRandomProjection(
                full_dim=state.projection.full_dim,
                subspace_dim=state.projection.subspace_dim,
                seed=state.projection.seed + state.absorb_count + 1000,
            )
        else:
            state.projection = adaptive_sub.init_projection(
                generator=opt._generator, device=state.X.device, dtype=state.X.dtype
            )
        state.displacement_history.zero_()
        if state.displacement_history_full is not None:
            state.displacement_history_full.zero_()
        state.displacement_history_idx = 0
        state.displacement_history_count = 0
        invalidate_for_basis_change(opt, state)
        state.absorb_count += 1
        # Clear the stagnation counter, else absorb_mode='stagnation' stays triggered
        # on a plateau and redraws the basis every step. The absorb re-anchors the
        # origin, so the old loss history no longer applies.
        state.stagnation_count = 0
        state.prev_loss = avg_model_loss
        if opt._cma_subspace and opt.use_covariance_adaptation:
            state.p_c = torch.zeros_like(state.p_c)
            state.p_sigma = torch.zeros_like(state.p_sigma)
            state.C_diag = torch.ones_like(state.C_diag)
    elif _rotation_due(adaptive_sub, state.iteration_count):
        # Re-anchor the origin before rotating. The represented point is
        # base + P @ coords, so replacing P while coords are non-zero moves the
        # weights with no evaluation behind it.
        state.base_params, _ = adaptive_sub.absorb(
            proj_used, state.base_params, state.X.reshape(-1)[: adaptive_sub.subspace_dim]
        )
        state.X = torch.zeros_like(state.X)
        if isinstance(state.projection, SparseRandomProjection):
            state.projection = SparseRandomProjection(
                full_dim=state.projection.full_dim,
                subspace_dim=state.projection.subspace_dim,
                seed=state.projection.seed + state.iteration_count,
            )
        else:
            # Full-space history: each entry keeps the basis it was measured in
            # instead of being reprojected through whichever one is current.
            hist: Optional[torch.Tensor] = (
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
