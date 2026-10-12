"""State reset and chunking shared by the step drivers."""

from __future__ import annotations

import warnings

import torch

from ._compiled import normalize_rows
from .dynamics import update_radius_multiplier, update_stagnation


def epsilon_jumped(state, ot_epsilon: float) -> bool:
    """Whether the solve epsilon moved more than 2x since the last solve, which stales the dual-momentum history."""
    last = state.last_solve_eps
    return last is not None and (ot_epsilon / last > 2.0 or last / ot_epsilon > 2.0)


def warn_all_nonfinite(opt) -> None:
    """Warn once when every vertex cost of a step is NaN or inf."""
    if getattr(opt, "_nonfinite_warned", False):
        return
    opt._nonfinite_warned = True
    warnings.warn(
        "every vertex cost was NaN or inf this step, so the cost matrix "
        "carries no ranking information: the plan is uniform, the step is exactly "
        "zero, and the step records an infinite cost.",
        RuntimeWarning,
        stacklevel=3,
    )


def chunk_spans(n_cand: int, chunk: int, bounds=None, chunks=None):
    """Yield (start, end) chunks of at most ``chunk``, also breaking at ``bounds``.

    Cutting at entry boundaries lets the site-aware paths resolve a site. ``chunks``
    maps the start of a span between bounds to its own chunk size.
    """
    stops = sorted(set(bounds or ()) | {n_cand})
    start = 0
    for stop in stops:
        if stop <= start:
            continue
        size = chunks.get(start, chunk) if chunks else chunk
        for chunk_start in range(start, stop, size):
            yield chunk_start, min(chunk_start + size, stop)
        start = stop


def activation_elements(opt, inputs, key=None) -> int:
    """Activation elements one candidate's forward holds on ``inputs``.

    With ``key``, only the outputs from that parameter's module on count. Scales the
    ratios the rebuild probe measured, falling back to the whole model for a key it
    could not place, or the widest layer when there are none.
    """
    if inputs is None:
        return 0
    ratios = getattr(opt, "_act_ratios", None)
    if ratios is not None:
        return round(ratios.get(key, ratios[None]) * inputs.numel())
    return inputs.shape[0] * max((e.shape[0] for e in opt.layout.entries if e.shape), default=0)


def site_chunks(opt, inputs, sites, pdim: int, group: int, cap: int) -> dict[int, int]:
    """Automatic chunk for each site's span, keyed by the first candidate wholly inside it.

    ``sites`` pairs each site's first coordinate with its parameter name. A site path
    batches only the activations from the perturbed parameter's module on, so each site
    is budgeted on those and on the largest parameter.
    """
    numel = max(e.numel for e in opt.layout.entries)
    chunks = {}
    for start, key in sites:
        chunk = min(cap, (1 << 26) // (2 * activation_elements(opt, inputs, key) + numel))
        chunks[-(-start // pdim) * group] = max(group, chunk // group * group)
    return chunks


def site_bounds(starts, pdim: int, group: int, total: int) -> list[int]:
    """Candidate indices where a chunk must break so each site start lands in its own chunk."""
    # Each start contributes its own group and the next, isolating the
    # straddling particle in its own chunk.
    return sorted({min(b, total) for s in starts for b in ((s // pdim) * group, -(-s // pdim) * group)})


def check_losses(losses, n: int) -> None:
    """Raise unless the closure returned one loss per candidate."""
    if losses.shape != (n,):
        raise ValueError(f"the closure must return one loss per candidate, shape ({n},), got {tuple(losses.shape)}.")


def record_saturation(opt, state, transport, step_radius) -> None:
    """Log how far each transport row is from uniform; off unless asked for."""
    log = getattr(opt, "_saturation_log", None)
    if log is None:
        return
    V = transport.shape[-1]
    p = normalize_rows(transport)
    omega = (p - 1.0 / V).abs().amax(-1) * V
    # At lambda = 0 the row is softmax(-C/eps), so log(p_max/p_min) is the exact cost-row spread.
    # Rows whose smallest weight underflows float32 are counted, not trusted: their spread is a floor.
    p_min = p.amin(-1)
    p_max = p.amax(-1)
    spread = torch.log(p_max.clamp_min(1e-30)) - torch.log(p_min.clamp_min(1e-30))
    underflow = p_min <= 1e-30
    log.append(
        {
            "iteration": int(getattr(state, "iteration_count", len(log))),
            "step_radius": float(step_radius),
            "particles": int(omega.numel()),
            "saturated_frac": float((omega > 1.0).to(torch.float32).mean()),
            "omega_mean": float(omega.mean()),
            "omega_max": float(omega.amax()),
            "spread_mean": float(spread.mean()),
            "spread_max": float(spread.amax()),
            "spread_underflow_frac": float(underflow.to(torch.float32).mean()),
            "spread_sat_frac": float((spread > 1.0).to(torch.float32).mean()),
        }
    )


def update_stagnation_and_radius(opt, state, loss: float) -> None:
    """Advance the stagnation counter and, if enabled, the adaptive radius."""
    prev_loss = state.prev_loss
    state.stagnation_count, state.prev_loss = update_stagnation(
        loss,
        state.prev_loss,
        state.stagnation_count,
        stagnation_threshold=opt.stagnation_threshold,
    )
    if opt.use_adaptive_radius:
        state.radius_multiplier, state.stagnation_count = update_radius_multiplier(
            loss,
            prev_loss,
            state.stagnation_count,
            state.radius_multiplier,
            stagnation_patience=opt.stagnation_patience,
            radius_increase=opt.radius_increase,
            radius_decrease=opt.radius_decrease,
            radius_min=opt.radius_min,
            radius_max=opt.radius_max,
        )


def update_amortized_direction(opt, raw_direction, nan_reverted: bool) -> None:
    """EMA the coasting direction an amortized step reuses, or drop it after a revert."""
    if opt.amortize_steps <= 1:
        return
    if nan_reverted:
        opt._transport_direction_ema = None
        return
    alpha = opt.amortize_ema
    prev = opt._transport_direction_ema
    opt._transport_direction_ema = raw_direction if prev is None else alpha * prev + (1.0 - alpha) * raw_direction


def invalidate_for_basis_change(opt, state) -> None:
    """Drop every cached direction and dual that indexes the outgoing basis; call after an absorb or rotation."""
    state.f = None
    state.g = None
    state.prev_prev_f = None
    state.prev_prev_g = None

    opt._transport_direction_ema = None
    opt._newton_direction = None
    opt._prev_descent_direction = None
    opt._center_loss = None

    if opt.use_momentum and state.velocity is not None:
        state.velocity = torch.zeros_like(state.velocity)
