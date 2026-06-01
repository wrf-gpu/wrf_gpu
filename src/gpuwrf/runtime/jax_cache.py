"""Central JAX persistent-compilation-cache hook (v0.2.0 wall-clock win #1).

WHY
---
A normal gpuwrf run recompiles every XLA executable from scratch (cold compile),
which the GPT wall-clock analysis (``.agent/reviews/2026-06-01-gpt-wallclock-
optimization.md``) measured at ~40%% of the d02 24h wall and ~80-90%% of short
validation jobs. JAX ships a persistent on-disk compilation cache that turns a
cold compile into a disk read on every repeat run (and every re-run of *our own*
validation), with **zero numerics change** -- the cached object is the identical
XLA executable keyed by the program's HLO + backend + flags.

This module enables that cache from a single central place. It is imported by
``gpuwrf/__init__.py`` (the same import hook that enables x64) so EVERY entry
path -- operational/real-case, idealized, tests, scripts -- shares one warm cache
without each launch script having to export env vars.

POLICY
------
- Default cache dir: ``/mnt/data/gpuwrf_jax_cache`` (large local NVMe; 200+ GiB
  free). Override with ``GPUWRF_JAX_CACHE_DIR`` or the standard JAX env var
  ``JAX_COMPILATION_CACHE_DIR`` (if the operator already exported one, we respect
  it and do NOT override).
- ``min_compile_time_secs = 0`` so every compiled executable is cached (the
  default 1.0 s threshold would skip cheap programs and make cache-warming audits
  ambiguous; for our minute-scale compiles 0 is strictly better).
- Disable entirely with ``GPUWRF_JAX_CACHE=0`` (e.g. for a clean cold-compile
  benchmark) -- this is a pure no-op, the cache is simply not configured.
- Fully graceful: if the directory cannot be created or the JAX knobs are
  unavailable, the hook silently leaves JAX at its default (no cache) rather than
  failing import.

This NEVER changes a forecast result. A persistent compilation cache only avoids
re-running the XLA compiler; the executable, and therefore every floating-point
operation, is bit-for-bit the same as a cold compile.
"""

from __future__ import annotations

import os

DEFAULT_CACHE_DIR = "/mnt/data/gpuwrf_jax_cache"

# Module-level record of what was configured, for audit/proof scripts.
CACHE_STATUS: dict[str, object] = {
    "enabled": False,
    "dir": None,
    "min_compile_time_secs": None,
    "source": None,
    "error": None,
}


def _resolve_cache_dir() -> str | None:
    """Return the cache directory to use, or ``None`` if caching is disabled."""

    # Explicit opt-out (used for clean cold-compile benchmarks).
    flag = os.environ.get("GPUWRF_JAX_CACHE", "").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        CACHE_STATUS["source"] = "disabled-by-GPUWRF_JAX_CACHE"
        return None

    # Respect an operator-provided cache dir (either name); do not override.
    env_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.environ.get(
        "GPUWRF_JAX_CACHE_DIR"
    )
    if env_dir:
        CACHE_STATUS["source"] = "env"
        return env_dir

    CACHE_STATUS["source"] = "default"
    return DEFAULT_CACHE_DIR


def configure_jax_compilation_cache() -> dict[str, object]:
    """Enable JAX's persistent compilation cache. Idempotent and best-effort.

    Returns the :data:`CACHE_STATUS` dict describing what (if anything) was
    configured. Safe to call multiple times.
    """

    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return CACHE_STATUS

    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        # Cannot create the directory (read-only mount, permissions, missing
        # parent). Leave JAX at default; do NOT fail import.
        CACHE_STATUS["error"] = f"makedirs failed: {exc}"
        return CACHE_STATUS

    try:
        from jax import config as _jax_config

        _jax_config.update("jax_compilation_cache_dir", cache_dir)
        # Cache every compile, even cheap ones (default threshold is 1.0 s).
        _jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # Also cache small entries (the default min-entry-size can skip tiny
        # programs); harmless and makes warming unambiguous. Guarded because the
        # knob name has varied across JAX versions.
        try:
            _jax_config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        except Exception:  # pragma: no cover - knob absent on some versions
            pass
    except Exception as exc:  # pragma: no cover - JAX knob missing/renamed
        CACHE_STATUS["error"] = f"config.update failed: {exc}"
        return CACHE_STATUS

    CACHE_STATUS["enabled"] = True
    CACHE_STATUS["dir"] = cache_dir
    CACHE_STATUS["min_compile_time_secs"] = 0
    return CACHE_STATUS
