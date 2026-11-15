"""Epoch loop and evaluation-path comparison shared by examples 10 and 11."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from polystep.cost_nn import NNCostEvaluator


def run_epoch(optimizer, loss_fn, x, y, batch_size):
    """Mean ``step_batch`` loss over one shuffled pass through ``(x, y)``."""
    order = torch.randperm(x.shape[0])
    total, batches = 0.0, 0
    for start in range(0, x.shape[0], batch_size):
        idx = order[start : start + batch_size]
        total += optimizer.step_batch(x[idx], y[idx], loss_fn=loss_fn)
        batches += 1
    return total / max(batches, 1)


def compare_paths(make_model, make_optimizer, x, y, device, batch_size, steps):
    """Time and cross-check the two evaluation paths on the same batches.

    ``make_model()`` builds a fresh model and ``make_optimizer(model)`` its optimizer, once per path.
    """
    results = {}
    for register in (False, True):
        torch.manual_seed(0)
        model = make_model().to(device)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        optimizer = make_optimizer(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = []
        for i in range(steps):
            start = (i * batch_size) % len(x)
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            if register:
                optimizer.register_evaluator(evaluator, xb, yb)
            losses.append(optimizer.step(lambda p, _x=xb, _y=yb: evaluator.evaluate(p, _x, _y)))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        results["site" if register else "materializing"] = (losses, time.perf_counter() - started)

    dense_losses, dense_time = results["materializing"]
    site_losses, site_time = results["site"]
    torch.testing.assert_close(torch.tensor(site_losses), torch.tensor(dense_losses), rtol=1e-4, atol=1e-5)
    drift = max(abs(a - b) for a, b in zip(dense_losses, site_losses))
    print(f"  materializing : {dense_time:6.2f} s for {steps} steps")
    print(f"  site-aware    : {site_time:6.2f} s for {steps} steps   ({dense_time / site_time:.1f}x)")
    print(f"  max loss difference between the two: {drift:.2e}")
