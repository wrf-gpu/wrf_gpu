"""Persistent XLA / JIT compilation cache (v0.12.0 first-run usability win).

WHY
---
A fresh ``gpuwrf`` process compiles every XLA executable from scratch. The
v0.12.0 release critique measured a **~4 min 55 s cold JIT compile on every
fresh process** -- undocumented and painful for a naive user who clones, installs
and runs out of the box. JAX ships a persistent on-disk compilation cache that
turns that cold compile into a disk read on every subsequent run/process, with
**zero numerics change**: the cached object is the *identical* XLA executable,
keyed by the program's HLO + backend + compile flags, so every floating-point
operation is bit-for-bit the same as a cold compile.

This module enables that cache from a single central place. It is invoked by
``gpuwrf/__init__.py`` (the same import hook that enables x64) BEFORE JAX
compiles anything, so EVERY entry path -- operational/real-case, idealized,
tests, scripts -- shares one warm cache without each launch script having to
export env vars.

POLICY (portable-by-default; safe for a clean clone)
----------------------------------------------------
Cache directory precedence (first match wins):

1. ``GPUWRF_JAX_CACHE=0`` (or false/off/no) -> caching fully disabled (pure
   no-op; used for clean cold-compile benchmarks).
2. ``JAX_COMPILATION_CACHE_DIR`` -- the standard upstream JAX env var. If the
   operator already exported one we respect it and do not override.
3. ``GPUWRF_JAX_CACHE_DIR`` -- explicit project override (kept for back-compat
   with the v0.2.0 perf proofs and the user's ``/mnt/data`` workstation layout).
4. ``GPUWRF_CACHE`` -- project cache root; the JIT cache lives at
   ``$GPUWRF_CACHE/jit``.
5. Default: ``$XDG_CACHE_HOME/gpuwrf/jit`` if ``XDG_CACHE_HOME`` is set, else
   ``$HOME/.cache/gpuwrf/jit``.

The default is deliberately a **persistent** per-user cache dir, NOT ``/tmp``
(which is frequently a ``tmpfs`` that is wiped on reboot and would silently
defeat the whole point of a *persistent* cache across runs).

- ``min_compile_time_secs = 0`` so every compiled executable is cached (the JAX
  default 1.0 s threshold would skip cheap programs and make cache-warming
  audits ambiguous; for our minute-scale compiles 0 is strictly better).
- Fully graceful: if the directory cannot be created or a JAX knob is
  unavailable/renamed, the hook silently leaves JAX at its default (no cache)
  rather than failing import.

CAVEAT
------
The persistent cache key includes the jaxlib/XLA version and the backend. After
a ``jax``/``jaxlib`` upgrade (or switching CPU<->GPU) the old entries no longer
match and the first run on the new version pays a one-time cold compile again;
stale entries are simply ignored, never wrong. Clearing the cache dir is always
safe.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CACHE_STATUS",
    "configure_compilation_cache",
    "resolve_cache_dir",
    "cache_env_help",
    "cache_entry_count",
    "warm_hit_for",
    "cache_report",
]

# Falsey values that disable the cache entirely.
_DISABLE_VALUES = {"0", "false", "off", "no"}

# Module-level record of what was configured, for audit/proof scripts.
CACHE_STATUS: dict[str, object] = {
    "enabled": False,
    "dir": None,
    "min_compile_time_secs": None,
    "source": None,
    "error": None,
    "autotune": None,
    "parallel_compile": None,
}


def _default_cache_dir() -> Path:
    """Persistent per-user default: ``$XDG_CACHE_HOME/gpuwrf/jit`` or
    ``$HOME/.cache/gpuwrf/jit``. Never ``/tmp`` (often tmpfs, wiped on reboot)."""
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    return base / "gpuwrf" / "jit"


def resolve_cache_dir() -> Path | None:
    """Return the cache directory to use, or ``None`` if caching is disabled.

    Records the chosen ``source`` in :data:`CACHE_STATUS`. Pure (no side effects
    other than updating ``CACHE_STATUS['source']``).
    """
    flag = os.environ.get("GPUWRF_JAX_CACHE", "").strip().lower()
    if flag in _DISABLE_VALUES:
        CACHE_STATUS["source"] = "disabled-by-GPUWRF_JAX_CACHE"
        return None

    std = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if std:
        CACHE_STATUS["source"] = "env:JAX_COMPILATION_CACHE_DIR"
        return Path(std).expanduser()

    project = os.environ.get("GPUWRF_JAX_CACHE_DIR", "").strip()
    if project:
        CACHE_STATUS["source"] = "env:GPUWRF_JAX_CACHE_DIR"
        return Path(project).expanduser()

    root = os.environ.get("GPUWRF_CACHE", "").strip()
    if root:
        CACHE_STATUS["source"] = "env:GPUWRF_CACHE"
        return Path(root).expanduser() / "jit"

    CACHE_STATUS["source"] = "default"
    return _default_cache_dir()


def configure_compilation_cache() -> dict[str, object]:
    """Enable JAX's persistent compilation cache. Idempotent and best-effort.

    Returns the :data:`CACHE_STATUS` dict describing what (if anything) was
    configured. Safe to call multiple times and never raises.
    """
    cache_dir = resolve_cache_dir()
    if cache_dir is None:
        return CACHE_STATUS

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Cannot create the directory (read-only mount, permissions, missing
        # parent). Leave JAX at default; do NOT fail import.
        CACHE_STATUS["error"] = f"mkdir failed: {exc}"
        return CACHE_STATUS

    try:
        from jax import config as _jax_config

        _jax_config.update("jax_compilation_cache_dir", str(cache_dir))
        # Cache every compile, even cheap ones (default threshold is 1.0 s).
        _jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # Also cache small entries (default min-entry-size can skip tiny
        # programs); harmless and makes warming audits unambiguous. Guarded
        # because the knob name has varied across JAX versions.
        try:
            _jax_config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        except Exception:  # pragma: no cover - knob absent on some versions
            pass
    except Exception as exc:  # pragma: no cover - JAX knob missing/renamed
        CACHE_STATUS["error"] = f"config.update failed: {exc}"
        return CACHE_STATUS

    CACHE_STATUS["enabled"] = True
    CACHE_STATUS["dir"] = str(cache_dir)
    CACHE_STATUS["min_compile_time_secs"] = 0

    # v0.13 #2: also wire the persistent GPU autotune cache + parallel-compile
    # flags from the same central hook. This is HARD-SAFE: configure_autotune_cache
    # is OFF by default (requires an explicit GPUWRF_XLA_AUTOTUNE_CACHE=1 opt-in)
    # and, even when opted in, validates each --xla_gpu_* flag against the
    # installed build in an isolated subprocess before injecting it -- so an
    # unknown flag can never abort this (default GPU) path. Wrapped in try/except
    # as a final belt-and-braces so it can NEVER fail the compile-cache setup or
    # the package import. Numerically inert.
    try:
        from gpuwrf.runtime.xla_autotune import (
            AUTOTUNE_STATUS,
            configure_autotune_cache,
        )

        configure_autotune_cache()
        CACHE_STATUS["autotune"] = dict(AUTOTUNE_STATUS)
    except Exception as exc:  # pragma: no cover - never fail the compile cache
        CACHE_STATUS["autotune"] = {"error": f"{type(exc).__name__}: {exc}"}

    # v0.13 Tier2 #1: STANDALONE parallel-compile knob, wired from the same hook
    # but INDEPENDENT of the autotune cache (a deployment can enable N-way
    # parallel XLA compile without opting into the persistent autotune cache).
    # Same HARD-SAFE construction: OFF by default (needs GPUWRF_XLA_PARALLEL_COMPILE
    # or the legacy GPUWRF_XLA_COMPILE_PARALLELISM), respects the platform pin, and
    # validates its single --xla_gpu_* flag in an isolated subprocess before
    # injecting -- so an unknown flag can never abort this (default GPU) path.
    # Wrapped in try/except as belt-and-braces. Numerically inert.
    try:
        from gpuwrf.runtime.xla_autotune import (
            PARALLEL_COMPILE_STATUS,
            configure_parallel_compile,
        )

        configure_parallel_compile()
        CACHE_STATUS["parallel_compile"] = dict(PARALLEL_COMPILE_STATUS)
    except Exception as exc:  # pragma: no cover - never fail the compile cache
        CACHE_STATUS["parallel_compile"] = {"error": f"{type(exc).__name__}: {exc}"}

    return CACHE_STATUS


def cache_entry_count(cache_dir: Path | str | None = None) -> int:
    """Number of compiled-executable entries currently in the cache.

    JAX writes one ``*-cache`` directory (or file, depending on jaxlib version)
    per distinct compiled program. Counting them lets a caller detect a *warm
    hit* robustly without knowing JAX's opaque HLO cache key: if compiling a
    program adds no new entry, it was served from the cache.

    Returns 0 if caching is disabled or the dir does not exist.
    """
    if cache_dir is None:
        cache_dir = CACHE_STATUS.get("dir")  # type: ignore[assignment]
    if not cache_dir:
        return 0
    p = Path(cache_dir)
    if not p.is_dir():
        return 0
    try:
        # Each cached executable is a child whose name contains "-cache".
        # Fall back to counting all immediate children if the naming changes.
        named = [c for c in p.iterdir() if "-cache" in c.name]
        return len(named) if named else len(list(p.iterdir()))
    except OSError:
        return 0


def warm_hit_for(compile_callable, cache_dir: Path | str | None = None) -> dict[str, object]:
    """Detect whether ``compile_callable()`` was a warm cache hit.

    ``compile_callable`` is a zero-arg callable that performs a
    ``jax.jit(fn).lower(*args).compile()`` for the program under test. This
    helper counts cache entries before and after; **no new entry written =>
    warm hit** (the executable was served from disk). This is the config-keyed
    warm-hit assertion the v0.13 roadmap (#3) calls for, robust to JAX's opaque
    cache key.

    Returns a dict with ``before``, ``after``, ``warm`` (bool), and
    ``compile_seconds``.
    """
    import time as _time

    if cache_dir is None:
        cache_dir = CACHE_STATUS.get("dir")  # type: ignore[assignment]
    before = cache_entry_count(cache_dir)
    t0 = _time.perf_counter()
    compile_callable()
    dt = _time.perf_counter() - t0
    after = cache_entry_count(cache_dir)
    return {
        "before": before,
        "after": after,
        "warm": after == before and before > 0,
        "compile_seconds": dt,
    }


def cache_report() -> dict[str, object]:
    """Human/JSON-friendly snapshot of the compile + autotune cache state."""
    return {
        "compile_cache": dict(CACHE_STATUS),
        "compile_cache_entries": cache_entry_count(),
    }


def cache_env_help() -> str:
    """One-line human summary of the env vars, for ``--help`` / docs / logs."""
    return (
        "Persistent XLA compile cache env vars: "
        "GPUWRF_JAX_CACHE=0 disables; "
        "JAX_COMPILATION_CACHE_DIR / GPUWRF_JAX_CACHE_DIR set an explicit dir; "
        "GPUWRF_CACHE sets the cache root (jit cache at $GPUWRF_CACHE/jit); "
        "default $XDG_CACHE_HOME/gpuwrf/jit or $HOME/.cache/gpuwrf/jit."
    )
