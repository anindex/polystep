"""Shared setup every example needs before it imports polystep."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup(default_threads: int = 1) -> None:
    """Pin the CPU thread count and make a source checkout importable.

    PolyStep's per-step ops are small, so torch's default ``nproc``-sized thread
    pool costs more than it returns. ``POLYSTEP_THREADS`` overrides.
    """
    import torch

    torch.set_num_threads(int(os.environ.get("POLYSTEP_THREADS", 0)) or default_threads)
    src = str(ROOT / "src")
    if (ROOT / "src" / "polystep").is_dir() and src not in sys.path:
        sys.path.insert(0, src)
