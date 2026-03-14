"""Shared utility helpers for conditional generative workflows."""

from __future__ import annotations

import multiprocessing as mp
import os

from joblib import cpu_count as _joblib_cpu_count


def _available_cpu_count() -> int:
    """Return CPU count respecting affinity/cgroup limits when possible."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:
        return max(1, int(_joblib_cpu_count()))
    except Exception:
        return max(1, int(mp.cpu_count()))


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a compact human-readable string."""
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.2f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"
