"""Backward-compatible shim for the persistent JAX compilation cache.

The canonical implementation now lives in :mod:`gpuwrf.runtime.compile_cache`
(v0.12.0), which uses a **portable per-user default** cache dir
(``$HOME/.cache/gpuwrf/jit``) so a fresh clone is fast out of the box instead of
writing into a private ``/mnt/data`` path.

This module is kept only so the older name
``configure_jax_compilation_cache`` and the ``CACHE_STATUS`` object continue to
resolve (e.g. ``proofs/perf/win1_cache_ab_timing.py``). New code should import
from :mod:`gpuwrf.runtime.compile_cache` directly.
"""

from __future__ import annotations

from gpuwrf.runtime.compile_cache import (
    CACHE_STATUS,
    configure_compilation_cache,
)

__all__ = ["CACHE_STATUS", "configure_jax_compilation_cache", "DEFAULT_CACHE_DIR"]

# Historical default (the user's workstation NVMe). No longer the out-of-box
# default -- the portable per-user dir is now preferred -- but exported for any
# caller that referenced the constant.  To get the old behaviour explicitly,
# export ``GPUWRF_JAX_CACHE_DIR=/mnt/data/gpuwrf_jax_cache``.
DEFAULT_CACHE_DIR = "/mnt/data/gpuwrf_jax_cache"


def configure_jax_compilation_cache() -> dict[str, object]:
    """Deprecated alias for
    :func:`gpuwrf.runtime.compile_cache.configure_compilation_cache`."""
    return configure_compilation_cache()
