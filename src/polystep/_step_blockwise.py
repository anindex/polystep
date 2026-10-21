"""Block-wise step: one OT solve per block of parameter entries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable

import torch

from .blockwise import block_to_layout_columns, blocks_to_layout_flat, layout_flat_to_block_flat
from .solvers._shared import loss_buffer_dtype, recenter_and_scale, sanitize_cost, solver_health
from .epsilon import feed_solver_stats, radius_epsilon_factor, resolve_epsilon
from .dynamics import (
    apply_momentum,
    compute_momentum_coefficient,
)
from .geometry import apply_biased_rotation, get_random_rotation_matrices
from .solvers.base import SolverResult
from ._step_core import (
    activation_elements,
    check_losses,
    chunk_spans,
    epsilon_jumped,
    record_saturation,
    site_bounds,
    site_chunks,
    update_amortized_direction,
    update_stagnation_and_radius,
    warn_all_nonfinite,
)

logger = logging.getLogger(__name__)


@dataclass
class _BlockOutcome:
    """One block's contribution to the sweep, reduced together after the loop."""

    particles: torch.Tensor
    descent: torch.Tensor
    duals: tuple
    model_loss: torch.Tensor
    progress: torch.Tensor
    ess: torch.Tensor
    rho: torch.Tensor
    evals: int
    converged: bool
    n_iters: int
    # Candidate count behind this block's means; a plain average would let a one-particle block outvote a 500-particle one.
    weight: int


def _resolve_geometry(opt, state, iteration):
    """Epsilon, OT epsilon and the two radii for this step."""
    current_eps = resolve_epsilon(opt.epsilon, iteration, "epsilon")
    radius_mult = state.radius_multiplier if opt.use_adaptive_radius else 1.0
    step_r = opt._get_step_radius(iteration) * radius_epsilon_factor(opt.step_radius, current_eps) * radius_mult
    probe_r = opt._get_probe_radius(iteration) * radius_epsilon_factor(opt.probe_radius, current_eps) * radius_mult
    # Probe-radius jitter; no-op when probe_radius_jitter == 0.
    probe_r = opt._apply_probe_radius_jitter(probe_r)
    ent_eps = resolve_epsilon(opt.ent_epsilon, iteration, "ent_epsilon")
    return current_eps, ent_eps if ent_eps is not None else current_eps, step_r, probe_r


def _prepare_probes(opt, block, block_X, verts, block_idx, probes, step_r, probe_r, descent_dirs):
    """Rotate the polytope onto the block's particles, place its vertices and probe points."""
    device, dtype = block_X.device, block_X.dtype
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
    return rot_mats, X_vertices, X_probe


def _cost_from_losses(opt, losses, P, V, K):
    """Average the probe axis away, record an all-non-finite block.

    A vertex with any non-finite probe keeps a non-finite cost, so the solve reads it as infeasible.
    """
    cost = losses.reshape(P, V) if K == 1 else losses.reshape(P, V, K).mean(dim=-1)
    flag = ~torch.isfinite(cost).any()
    seen = opt._all_nonfinite
    opt._all_nonfinite = flag if seen is None else seen & flag
    return cost


def _solve_block(opt, state, block_idx, losses, P, V, K, block_X, verts, rot_mats, X_vertices, step_r, ot_epsilon):
    """Solve one block's OT problem and project it onto the barycentre."""
    solve_cost = _cost_from_losses(opt, losses, P, V, K)
    opt.solver.epsilon = ot_epsilon
    cost_matrix = sanitize_cost(solve_cost)

    if opt._use_fused_softmax:
        scaled_cost = recenter_and_scale(cost_matrix, solve_cost, opt.scale_cost)[0]
        X_new_block, transport_matrix = opt._compiled.fused_softmax_project(
            scaled_cost, ot_epsilon, verts, rot_mats, step_r, block_X
        )
        ot_result = SolverResult(matrix=transport_matrix)
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

        # a defaults to uniform 1/P_block inside the solver; pass None to skip host-sync validation.
        ot_result = opt.solver.solve(cost_matrix=solve_cost, init_f=init_f, init_g=init_g, scale_cost=opt.scale_cost)
        X_new_block = opt._compiled.barycentric_projection(ot_result.matrix, X_vertices)

    record_saturation(opt, state, ot_result.matrix, step_r)

    # Per-particle step-radius jitter; no-op when step_radius_jitter == 0.
    X_new_block = opt._apply_particle_step_jitter(block_X, X_new_block)

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
        progress=cost_matrix.min(dim=1).values.mean().detach(),
        ess=ess,
        rho=rho,
        evals=P * V * K,
        converged=ot_result.converged,
        n_iters=ot_result.n_iters,
        weight=cost_matrix.numel(),
    )


def _finish_step(opt, state, X, X_new_full, outcomes, iteration, current_eps, ot_epsilon) -> float:
    """Momentum, NaN revert, diagnostics and dual bookkeeping after the block sweep."""
    if opt.biased_rotation:
        opt._prev_block_descent_directions = [o.descent for o in outcomes]

    if opt.use_momentum and state.velocity is not None:
        beta = compute_momentum_coefficient(iteration, opt.max_iterations, opt.momentum_init, opt.momentum_final)
        state.X, state.velocity = apply_momentum(X, X_new_full, state.velocity, beta, opt.velocity_lr)
    else:
        state.X = X_new_full

    # One host transfer for the sweep; the means are weighted by candidate count.
    *per_block, _all_nonfinite, x_finite, disp_sqnorm = torch.stack(
        [
            *(v for o in outcomes for v in (o.model_loss, o.progress, o.ess, o.rho)),
            opt._all_nonfinite,
            torch.isfinite(state.X).all(),
            torch.mean(torch.sum((state.X - X) ** 2, dim=-1)),
        ]
    ).tolist()
    total = sum(o.weight for o in outcomes)
    avg_model_loss, progress, ess, rho = (
        sum(o.weight * v for o, v in zip(outcomes, per_block[i::4])) / total for i in range(4)
    )

    nan_reverted = not x_finite
    if nan_reverted:
        # Back to the pre-step point, with its non-finite coordinates reset to the origin.
        state.X = X.nan_to_num(0.0, 0.0, 0.0)
        state.block_duals = [(None, None) for _ in outcomes]
        if opt.use_momentum and state.velocity is not None:
            state.velocity = torch.zeros_like(state.velocity)
        # Clear cached state that would propagate the NaN-producing direction.
        opt._transport_direction_ema = None
        opt._prev_descent_direction = None
        if opt._dual_momentum_beta > 0.0:
            state._prev_prev_block_duals = None
        if opt.biased_rotation:
            opt._prev_block_descent_directions = None
        disp_sqnorm = torch.mean(torch.sum((state.X - X) ** 2, dim=-1)).item()

    # The pure OT move, before momentum; state.X would fold in beta * velocity.
    update_amortized_direction(opt, (X_new_full - X).detach(), nan_reverted)

    if _all_nonfinite:
        # Every block's cost is the sanitize penalty, so it is not a loss.
        avg_model_loss = progress = float("inf")
        warn_all_nonfinite(opt)
    # Tracked unconditionally: absorb_mode="stagnation" reads this counter.
    update_stagnation_and_radius(opt, state, progress)

    # From the realised move; per-block descents ran before momentum.
    state.costs.append(avg_model_loss)
    all_converged = all(o.converged for o in outcomes) and not _all_nonfinite
    state.linear_convergence.append(all_converged)
    # One update for the whole sweep: the slowest block sets the pace.
    feed_solver_stats(opt._progressive_epsilon, opt.solver, max(o.n_iters for o in outcomes), all_converged)
    state.displacement_sqnorms.append(disp_sqnorm)
    state.record_solver_health(ess, rho, sum(o.evals for o in outcomes))
    state.iteration_count += 1

    # Stale NaN-causing duals must not overwrite the clean reset above. An all-non-finite
    # sweep solved a fabricated flat matrix, so its duals are not a warm start either.
    if not nan_reverted and not _all_nonfinite:
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
    """Per-block OT solve with full-model closure calls."""
    state = opt._state
    # Cleared per step, or a stale flag would read as this step's verdict.
    opt._all_nonfinite = None
    X = state.X  # (total_particles, particle_dim)
    iteration = state.iteration_count
    device = X.device
    blocks = opt._blocks

    current_eps, ot_epsilon, step_r, probe_r = _resolve_geometry(opt, state, iteration)

    # Per-layer blocks pad each entry independently, so offsets differ from ParamLayout, which pads once at the end.
    block_flat = layout_flat_to_block_flat(X.reshape(-1), blocks, opt.layout)
    all_block_particles = block_flat.view(-1, opt._particle_dim).split([b.num_particles for b in blocks])

    if opt._polytope_vertices.device != device or opt._polytope_vertices.dtype != X.dtype:
        opt._polytope_vertices = opt._polytope_vertices.to(device=device, dtype=X.dtype)
    verts = opt._polytope_vertices
    if opt._probes.device != device or opt._probes.dtype != X.dtype:
        opt._probes = opt._probes.to(device=device, dtype=X.dtype)
    probes = opt._probes

    fused_inputs = getattr(opt, "_fused_inputs", None)
    fused_targets = getattr(opt, "_fused_targets", None)
    sparse_delta = getattr(opt, "_sparse_delta_evaluator", None) if fused_inputs is not None else None
    # Covers the blocks the Linear-only sparse delta declines (conv, norm, attention, custom).
    site_vmap = getattr(opt, "_site_vmap_evaluator", None) if fused_inputs is not None else None
    site_owner = sparse_delta if sparse_delta is not None else site_vmap
    base_sd = opt.layout.unflatten(X) if site_owner is not None else None
    pdim_arange = torch.arange(opt._particle_dim, device=device)

    group = verts.shape[0] * probes.numel()
    most = max(b.num_particles for b in blocks) * group
    act = activation_elements(opt, fused_inputs)
    # Budget the chunk on elements; a fixed chunk OOMs on large models.
    chunk = min(opt.chunk_size or max(1, (1 << 26) // (opt.layout.padded_size + act)), most)
    site_chunk = chunk
    if site_owner is not None:
        per_site = 2 * act
        if site_vmap is not None and (sparse_delta is None or device.type == "cpu"):
            per_site += max(e.numel for e in opt.layout.entries)
        site_chunk = min(opt.chunk_size or max(1, (1 << 26) // max(1, per_site)), most)
        # The delta path scores a whole particle's group at once, so a chunk holds whole groups.
        site_chunk = max(group, site_chunk // group * group)

    # Candidates are written straight into layout order; the trailing column absorbs per-block padding.
    base_layout = X.reshape(-1)
    layout_columns = block_to_layout_columns(blocks, opt.layout, device)
    batch_buf = None

    if epsilon_jumped(state, ot_epsilon):
        state._prev_prev_block_duals = None
    descent_dirs = getattr(opt, "_prev_block_descent_directions", None)
    outcomes = []

    for block_idx, block in enumerate(blocks):
        block_X = all_block_particles[block_idx]
        rot_mats, X_vertices, X_probe = _prepare_probes(
            opt, block, block_X, verts, block_idx, probes, step_r, probe_r, descent_dirs
        )

        P, V, K, D = X_probe.shape
        total_evals = P * V * K
        # Counted here because the site-aware evaluators bypass the closure.
        opt.candidate_evals += total_evals
        losses = X_probe.new_empty(total_evals, dtype=loss_buffer_dtype(X_probe.dtype))
        candidates = X_probe.reshape(total_evals, D)
        block_columns = layout_columns[block.flat_start : block.flat_end].view(P, D)
        # Layout entries are contiguous, so a block maps to the layout by a constant shift.
        base = opt.layout.entries[block.leaf_indices[0]].offset
        bounds = chunks = None
        if site_owner is not None:
            ends = list(accumulate(opt.layout.entries[i].numel for i in block.leaf_indices))
            if site_owner is site_vmap and not opt.chunk_size:
                keys = [opt.layout.entries[i].key for i in block.leaf_indices]
                chunks = site_chunks(opt, fused_inputs, zip([0, *ends], keys), D, group, most)
            if block.leaf_indices[-1] == len(opt.layout.entries) - 1:
                ends.pop()
            bounds = site_bounds(ends, D, group, total_evals)

        for chunk_start, chunk_end in chunk_spans(total_evals, site_chunk, bounds, chunks):
            if site_owner is not None:
                g0, g1 = chunk_start // group, chunk_end // group
                offsets = base + D * torch.arange(g0, g1, device=device)
                span = (base + g0 * D, base + g1 * D)
                owner = site_owner
                site = owner.resolve_site(offsets, D, span)
                if site is None and site_vmap is not None and owner is not site_vmap:
                    # The sparse-delta correction is confined to Linear layers; vmap shares the graph ahead of any entry.
                    owner = site_vmap
                    site = owner.resolve_site(offsets, D, span)
                if site is not None:
                    losses[chunk_start:chunk_end] = owner.evaluate(
                        base_sd,
                        site if owner is site_vmap else site.key,
                        (offsets - site.offset).unsqueeze(1) + pdim_arange,
                        candidates[chunk_start:chunk_end].reshape(g1 - g0, group, D),
                        fused_inputs,
                        fused_targets,
                        **({} if owner is site_vmap else {"base": base}),
                    ).to(losses.dtype)
                    continue

            for lo in range(chunk_start, chunk_end, chunk):
                hi = min(lo + chunk, chunk_end)
                if batch_buf is None or batch_buf.shape[0] < hi - lo:
                    batch_buf = base_layout.new_empty((hi - lo, opt.layout.padded_size + 1))
                batch = batch_buf[: hi - lo]
                batch[:, : opt.layout.padded_size].copy_(base_layout)
                # All D coordinates of the probed row in one scatter; a per-coordinate loop costs a host sync each.
                col_idx = block_columns[torch.arange(lo, hi, device=device) // group]
                batch.scatter_(1, col_idx, candidates[lo:hi])

                chunk_losses = closure(opt.layout.batch_unflatten(batch[:, : opt.layout.padded_size]))
                check_losses(chunk_losses, hi - lo)
                losses[lo:hi] = chunk_losses.to(losses.dtype)

        outcomes.append(
            _solve_block(
                opt, state, block_idx, losses, P, V, K, block_X, verts, rot_mats, X_vertices, step_r, ot_epsilon
            )
        )

    new_flat = torch.cat([o.particles for o in outcomes]).reshape(-1)
    X_new_full = blocks_to_layout_flat(new_flat, blocks, opt.layout).view(X.shape)

    avg_model_loss = _finish_step(opt, state, X, X_new_full, outcomes, iteration, current_eps, ot_epsilon)
    opt._sync_model()
    return avg_model_loss
