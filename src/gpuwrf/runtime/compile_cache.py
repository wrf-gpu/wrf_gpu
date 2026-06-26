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
   with the v0.2.0 perf proofs and the user's ``<DATA_ROOT>`` workstation layout).
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

VERSION-KEYED DEFAULT DIR (B1)
------------------------------
JAX's own persistent-cache key already includes the jaxlib/XLA version and an
opaque backend fingerprint, so a stale entry is *ignored*, never *wrong*. But a
flat shared default dir (``$HOME/.cache/gpuwrf/jit``) mixes the entries of every
``gpuwrf`` version and every backend into one directory: a paid B200 run that
inherited a ``v0.20.0`` cache for a ``v0.20.2`` binary found ZERO usable entries
(different HLO) yet still paid the ~40 min cold compile, and the operator had no
signal that the cache was a guaranteed miss. To make stale-vs-warm legible and
to keep versions/backends from ever sharing a directory, the **default** cache
dir is now keyed by a human-readable tag::

    $HOME/.cache/gpuwrf/jit/<gpuwrf_version>-jax<JAX>-jaxlib<JAXLIB>-<backend>

e.g. ``.../jit/0.20.2-jax0.10.0-jaxlib0.10.0-cuda_sm120``. Two different gpuwrf
versions, or a CPU vs CUDA run, resolve to DIFFERENT directories and therefore
can never stale-hit one another; each is independently warm across runs. This is
purely a *path* change -- the cached executables and every floating-point op are
byte-identical to the un-keyed layout.

When the operator EXPLICITLY sets a cache dir (``JAX_COMPILATION_CACHE_DIR`` /
``GPUWRF_JAX_CACHE_DIR`` / ``GPUWRF_CACHE``) we honour it verbatim (no tag is
appended -- the operator owns that path), but we LOG A WARNING if its trailing
path component carries a version tag that does not match the running version, so
a hand-pinned stale dir is at least visible in the logs.

CAVEAT
------
After a ``jax``/``jaxlib`` upgrade (or switching CPU<->GPU) the version-keyed
default resolves to a fresh directory, so the first run on the new version/
backend pays a one-time cold compile again; the old directory is simply left in
place (never wrong). Clearing any cache dir is always safe.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

__all__ = [
    "CACHE_STATUS",
    "configure_compilation_cache",
    "resolve_cache_dir",
    "version_cache_tag",
    "cache_env_help",
    "cache_entry_count",
    "warm_hit_for",
    "cache_report",
    "cache_lock_path",
]

_LOG = logging.getLogger("gpuwrf.compile_cache")

# Falsey values that disable the cache entirely.
_DISABLE_VALUES = {"0", "false", "off", "no"}

# Cross-process cache LOCK sentinel (parallel-compile safety).
# ----------------------------------------------------------------------------
# JAX's persistent LRUCache acquires a per-dir ``filelock.FileLock(.lockfile)``
# on BOTH get() and put() ONLY when eviction is enabled, and eviction is enabled
# iff ``jax_compilation_cache_max_size != -1`` (the project/JAX default is -1 =
# NO lock + a bare non-atomic ``cache_path.write_bytes(value)``). When several
# PROCESSES compile concurrently into the same shared version-keyed cache dir
# (the cross-domain parallel-compile prewarm), the unlocked default lets a reader
# (the main process) read a half-written sibling ``<key>-cache`` file, and JAX's
# ``get_executable_and_time`` decompresses with NO try/except, so a torn read
# raises a hard ZstdError instead of falling back to recompile. Setting a LARGE
# positive sentinel (256 GiB, far above any real cache so NOTHING is ever
# evicted) flips ``eviction_enabled=True`` => both get() and put() take the
# cross-process FileLock => readers/writers in that dir are mutually excluded.
# The sentinel is NOT part of the cache KEY (HLO hash + platform + flags), so
# this is numerically inert and the filename stays ``<key>-cache``.
_CACHE_LOCK_SENTINEL_BYTES = 256 * 1024 * 1024 * 1024  # 256 GiB

# Default ON whenever the cache is enabled; ``GPUWRF_JAX_CACHE_LOCK=0`` restores
# the legacy unlocked path (single-process runs are unaffected either way).
_LOCK_DISABLE_VALUES = {"0", "false", "off", "no"}

# Guard so the belt-and-braces atomic-writer monkeypatch is installed at most
# once per process (idempotent, import-safe).
_ATOMIC_WRITER_INSTALLED = False

# Module-level record of what was configured, for audit/proof scripts.
CACHE_STATUS: dict[str, object] = {
    "enabled": False,
    "dir": None,
    "min_compile_time_secs": None,
    "source": None,
    "error": None,
    "autotune": None,
    "parallel_compile": None,
    # B1 additions: the version/backend tag that keys the default dir, whether
    # the resolved dir already had cached entries when we configured it (so a
    # caller/log can see "warm-capable" vs "guaranteed cold"), and any
    # explicit-dir version-mismatch warning.
    "version_tag": None,
    "warm_capable": None,
    "warning": None,
    # Parallel-compile cross-process safety (numerically inert):
    #   locked       -- jax_compilation_cache_max_size flipped to the positive
    #                   sentinel so get()/put() take the per-dir FileLock.
    #   max_size      -- the sentinel byte value actually set.
    #   lock_timeout  -- the FileLock acquire timeout (seconds).
    #   atomic_writer -- the belt-and-braces temp+rename put() override installed.
    "locked": None,
    "max_size": None,
    "lock_timeout": None,
    "atomic_writer": None,
}

# A version tag looks like ``0.20.2-jax0.10.0-jaxlib0.10.0-cuda_sm120``: the
# leading semver is what we compare across explicit dirs to spot a stale pin.
_TAG_VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)\b")


def _safe_component(value: str) -> str:
    """Sanitize a value for use as a single path component (no separators)."""
    return re.sub(r"[^0-9A-Za-z._+-]+", "_", value.strip()) or "unknown"


def _backend_tag() -> str:
    """A coarse, env-only backend/device tag for the version-keyed default dir.

    Computed WITHOUT initialising the JAX backend (which would lock ``XLA_FLAGS``
    before the autotune hook can set them, and is unnecessary work at import):

    * CPU/TPU pin  -> ``cpu`` / ``tpu``.
    * A CUDA/GPU target (pin, ``CUDA_VISIBLE_DEVICES``, or an ``/dev/nvidia*``
      node) -> ``cuda`` plus, when ``nvidia-smi`` is available, the compute
      capability of the first device as ``_sm<NNN>`` (e.g. ``cuda_sm120`` for a
      Blackwell sm_120). The compute-capability suffix keeps two physically
      different GPUs (which compile to different SASS) in separate dirs.

    Never raises; falls back to a coarse tag on any probe failure.
    """
    platforms = (
        os.environ.get("JAX_PLATFORMS", "")
        or os.environ.get("JAX_PLATFORM_NAME", "")
    ).strip().lower()
    if platforms:
        first = platforms.split(",")[0].strip()
        if first in ("cpu", "tpu"):
            return first
        if first in ("cuda", "gpu", "rocm"):
            return _cuda_tag()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "" or cvd == "-1":
            return "cpu"
        return _cuda_tag()

    for node in ("/dev/nvidia0", "/dev/nvidiactl"):
        if os.path.exists(node):
            return _cuda_tag()
    return "cpu"


def _cuda_tag() -> str:
    """``cuda`` plus the first device's compute capability (``cuda_sm120``).

    Reads the compute capability via ``nvidia-smi`` (cheap, no JAX backend init).
    Returns the coarse ``cuda`` if ``nvidia-smi`` is absent / errors / times out.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "cuda"
    if proc.returncode != 0:
        return "cuda"
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return "cuda"
    # "12.0" -> "sm120"; keep only the digits so the tag is a clean component.
    digits = re.sub(r"\D", "", line[0].strip())
    return f"cuda_sm{digits}" if digits else "cuda"


def version_cache_tag() -> str:
    """The version/backend tag that keys the default cache dir.

    ``<gpuwrf_version>-jax<JAX>-jaxlib<JAXLIB>-<backend>`` -- e.g.
    ``0.20.2-jax0.10.0-jaxlib0.10.0-cuda_sm120``. Best-effort; missing pieces
    degrade to ``unknown`` rather than raising.
    """
    try:
        from gpuwrf import __version__ as _gv
    except Exception:  # pragma: no cover - version import should not fail
        _gv = "unknown"
    try:
        import jax as _jax

        jax_v = getattr(_jax, "__version__", "unknown")
    except Exception:  # pragma: no cover
        jax_v = "unknown"
    try:
        import jaxlib as _jaxlib

        jaxlib_v = getattr(_jaxlib, "__version__", "unknown")
    except Exception:  # pragma: no cover
        jaxlib_v = "unknown"
    backend = _backend_tag()
    return _safe_component(
        f"{_gv}-jax{jax_v}-jaxlib{jaxlib_v}-{backend}"
    )


def _default_cache_dir() -> Path:
    """Persistent, VERSION-KEYED per-user default.

    ``$XDG_CACHE_HOME/gpuwrf/jit/<version_tag>`` or
    ``$HOME/.cache/gpuwrf/jit/<version_tag>``. Never ``/tmp`` (often tmpfs, wiped
    on reboot). The trailing ``<version_tag>`` component keys the dir by gpuwrf +
    jax/jaxlib version + backend so different versions/backends never share a
    directory (and thus never stale-hit). See :func:`version_cache_tag`."""
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg) if xdg else (Path.home() / ".cache")
    return base / "gpuwrf" / "jit" / version_cache_tag()


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
        p = Path(std).expanduser()
        _warn_if_explicit_dir_version_mismatch(p, "JAX_COMPILATION_CACHE_DIR")
        return p

    project = os.environ.get("GPUWRF_JAX_CACHE_DIR", "").strip()
    if project:
        CACHE_STATUS["source"] = "env:GPUWRF_JAX_CACHE_DIR"
        p = Path(project).expanduser()
        _warn_if_explicit_dir_version_mismatch(p, "GPUWRF_JAX_CACHE_DIR")
        return p

    root = os.environ.get("GPUWRF_CACHE", "").strip()
    if root:
        # We own the layout under this project cache root, so version-key the
        # ``jit`` subdir exactly like the default (different versions/backends
        # never share it). The operator picked the ROOT, not the final dir.
        CACHE_STATUS["source"] = "env:GPUWRF_CACHE"
        return Path(root).expanduser() / "jit" / version_cache_tag()

    CACHE_STATUS["source"] = "default"
    return _default_cache_dir()


def _warn_if_explicit_dir_version_mismatch(cache_dir: Path, var_name: str) -> None:
    """Log a warning if an operator-pinned dir carries a mismatched version tag.

    When the operator points a cache dir at a path whose last component looks
    like one of our version tags (``0.20.0-jax...``) but the leading semver does
    not match the running ``gpuwrf.__version__``, the entries there are a
    guaranteed miss (different HLO) -- exactly the silent paid-cold-compile that
    motivated B1. We DO NOT override the operator's choice (they own that path);
    we only surface the mismatch in the log + :data:`CACHE_STATUS['warning']`.
    Never raises.
    """
    try:
        from gpuwrf import __version__ as running
    except Exception:  # pragma: no cover
        return
    last = cache_dir.name
    m = _TAG_VERSION_RE.match(last)
    if not m:
        return  # not one of our tagged dirs; nothing to compare
    pinned = m.group(1)
    if pinned != running:
        msg = (
            f"compile cache dir ({var_name}={cache_dir}) carries version tag "
            f"{pinned!r} but the running gpuwrf is {running!r}; cached entries "
            f"there are a guaranteed MISS (different HLO) and you will pay a "
            f"cold compile. Point it at a {running}-tagged dir or unset the var "
            f"to use the auto version-keyed default."
        )
        CACHE_STATUS["warning"] = msg
        try:
            _LOG.warning(msg)
        except Exception:  # pragma: no cover - logging must never break config
            pass


def _reset_status() -> None:
    """Reset the mutable CACHE_STATUS fields so a re-call reflects the CURRENT env.

    CACHE_STATUS is module-global; without this reset a later call (e.g. a test
    that disables the cache, or a fail-open path) would inherit ``enabled=True``
    from an earlier successful configure and misreport. Mirrors
    :func:`gpuwrf.runtime.xla_autotune._reset_status`."""
    CACHE_STATUS.update(
        {
            "enabled": False,
            "dir": None,
            "min_compile_time_secs": None,
            "source": None,
            "error": None,
            "autotune": None,
            "parallel_compile": None,
            "version_tag": None,
            "warm_capable": None,
            "warning": None,
            "locked": None,
            "max_size": None,
            "lock_timeout": None,
            "atomic_writer": None,
        }
    )


def configure_compilation_cache() -> dict[str, object]:
    """Enable JAX's persistent compilation cache. Idempotent and best-effort.

    Returns the :data:`CACHE_STATUS` dict describing what (if anything) was
    configured. Safe to call multiple times and never raises; each call rebuilds
    :data:`CACHE_STATUS` from the CURRENT environment.
    """
    _reset_status()
    cache_dir = resolve_cache_dir()
    CACHE_STATUS["version_tag"] = version_cache_tag()
    if cache_dir is None:
        _LOG.info("gpuwrf compile cache: disabled (%s)", CACHE_STATUS.get("source"))
        return CACHE_STATUS

    # Detect warm-capability BEFORE we create the dir: a non-empty dir means a
    # prior run already cached executables here, so this run can warm-hit. A
    # freshly-created (or empty) dir is a guaranteed cold compile -- exactly the
    # signal a paid run wants in its logs up front. Guarded: a malformed path must
    # not raise here either (fail-open).
    try:
        pre_existing = cache_dir.is_dir() and cache_entry_count(cache_dir) > 0
    except Exception:  # pragma: no cover - defensive; is_dir on a weird path
        pre_existing = False
    CACHE_STATUS["warm_capable"] = pre_existing

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        # Cannot create the directory (read-only mount, permissions, missing
        # parent, malformed path -> OSError/ValueError/...). FAIL OPEN: leave JAX
        # at default (no cache); do NOT fail the run.
        CACHE_STATUS["error"] = f"mkdir failed: {exc}"
        try:
            _LOG.warning(
                "gpuwrf compile cache: could not create %s (%s); continuing "
                "WITHOUT a persistent cache (cold compile every run).",
                cache_dir, exc,
            )
        except Exception:  # pragma: no cover
            pass
        return CACHE_STATUS

    try:
        from jax import config as _jax_config

        _jax_config.update("jax_compilation_cache_dir", str(cache_dir))
        # Cache every compile, even cheap ones (default threshold is 1.0 s).
        _jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # CROSS-PROCESS CACHE LOCK (parallel-compile safety; numerically inert).
        # Engage JAX's own per-dir FileLock by flipping eviction on with a large
        # positive sentinel, so concurrent prewarm PROCESSES writing distinct
        # domain keys never expose a torn/half-written sibling to the main
        # process's reader (which would raise a hard ZstdError, not recompile).
        # Default ON when the cache is enabled; GPUWRF_JAX_CACHE_LOCK=0 opts out.
        _configure_cache_lock(_jax_config)
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

    # LOG the resolved path + whether it is warm-capable (B1): a single,
    # operator-visible line so a paid run sees up front whether it will warm-hit
    # or pay a cold compile. INFO level; never raises.
    try:
        _LOG.info(
            "gpuwrf compile cache: %s (source=%s, tag=%s) -- %s",
            cache_dir,
            CACHE_STATUS.get("source"),
            CACHE_STATUS.get("version_tag"),
            (
                "WARM-CAPABLE (existing cached entries found)"
                if pre_existing
                else "cold (no entries yet; first run pays a compile)"
            ),
        )
    except Exception:  # pragma: no cover - logging must never break config
        pass

    # v0.13 #2 / B1: also wire the persistent GPU autotune cache from the same
    # central hook. B1 turns it ON BY DEFAULT here (default_on=True) BECAUSE the
    # executable compile cache is on -- the autotune cache then reuses tuned
    # kernels across DIFFERENT-but-similar graphs (the executable cache alone only
    # covers the SAME graph). The GPUWRF_XLA_AUTOTUNE_CACHE=0 opt-out still wins.
    # Still HARD-SAFE: configure_autotune_cache validates each --xla_gpu_* flag
    # against the installed build in an isolated subprocess and never injects on a
    # CPU/TPU pin, so an unknown flag can never abort this path -- default-on does
    # NOT reintroduce the v0.12.0 GPU abort. Wrapped in try/except as a final
    # belt-and-braces so it can NEVER fail the compile-cache setup or the import.
    # Numerically inert.
    try:
        from gpuwrf.runtime.xla_autotune import (
            AUTOTUNE_STATUS,
            configure_autotune_cache,
        )

        configure_autotune_cache(default_on=True)
        CACHE_STATUS["autotune"] = dict(AUTOTUNE_STATUS)
    except Exception as exc:  # pragma: no cover - never fail the compile cache
        CACHE_STATUS["autotune"] = {"error": f"{type(exc).__name__}: {exc}"}

    # v0.13 Tier2 #1 / B2: STANDALONE parallel-compile knob, wired from the same
    # hook but INDEPENDENT of the autotune cache. B2 turns it ON BY DEFAULT here
    # (default_on=True) WHENEVER the executable compile cache is on, because N-way
    # parallel XLA compile is what slashes the COLD nest-compile wall (the ~40 min
    # paid-B200 case) by overlapping the many independent fusion compiles -- and a
    # cold compile is exactly when the compile cache has nothing to serve yet. The
    # GPUWRF_XLA_PARALLEL_COMPILE=0 opt-out still wins. STILL HARD-SAFE: the single
    # --xla_gpu_* flag is subprocess-probed against the bundled jaxlib before
    # injection and never injected on a CPU/TPU pin, so default-on does NOT
    # reintroduce the v0.12.0 GPU abort (the bundled jaxlib aborts on an unknown
    # --xla_gpu_* flag, which is why the probe is mandatory). Wrapped in try/except
    # as belt-and-braces. Numerically inert (only the compile-thread count changes).
    try:
        from gpuwrf.runtime.xla_autotune import (
            PARALLEL_COMPILE_STATUS,
            configure_parallel_compile,
        )

        configure_parallel_compile(default_on=True)
        CACHE_STATUS["parallel_compile"] = dict(PARALLEL_COMPILE_STATUS)
    except Exception as exc:  # pragma: no cover - never fail the compile cache
        CACHE_STATUS["parallel_compile"] = {"error": f"{type(exc).__name__}: {exc}"}

    return CACHE_STATUS


def cache_lock_path(cache_dir: Path | str | None = None) -> Path | None:
    """Path of the per-dir ``.lockfile`` JAX's LRUCache uses when eviction is on.

    Returns ``None`` if caching is disabled / no dir is known. The file only
    exists once the FileLock has been acquired at least once (a compile happened
    under the lock), so its presence is a positive signal the lock engaged.
    """
    if cache_dir is None:
        cache_dir = CACHE_STATUS.get("dir")  # type: ignore[assignment]
    if not cache_dir:
        return None
    return Path(cache_dir) / ".lockfile"


def _cache_lock_enabled() -> bool:
    """Whether to engage the cross-process cache FileLock (default ON)."""
    flag = os.environ.get("GPUWRF_JAX_CACHE_LOCK", "").strip().lower()
    return flag not in _LOCK_DISABLE_VALUES if flag else True


def _cache_lock_timeout() -> float:
    """FileLock acquire timeout in seconds (default 10s; widenable via env).

    JAX does NOT catch ``filelock.Timeout`` -- it would abort a compile -- so the
    prewarm driver must catch+retry; a generous default keeps the 9 short
    serialized cache writes well inside the window.
    """
    raw = os.environ.get("GPUWRF_JAX_CACHE_LOCK_TIMEOUT", "").strip()
    if not raw:
        return 10.0
    try:
        val = float(raw)
        return val if val > 0 else 10.0
    except ValueError:  # pragma: no cover - bad value falls back to default
        return 10.0


def _configure_cache_lock(jax_config) -> None:
    """Flip ``jax_compilation_cache_max_size`` to the positive lock sentinel.

    Setting a max_size != -1 makes JAX's ``LRUCache`` build a per-dir
    ``filelock.FileLock`` and acquire it on every get()/put(), mutually excluding
    concurrent readers/writers across PROCESSES. The sentinel (256 GiB) is far
    above any real cache so eviction never actually fires (no sibling-domain
    eviction risk). Also widens the lock-timeout knob and installs the
    belt-and-braces atomic-writer. All numerically inert; fail-open (any failure
    leaves the legacy unlocked cache, never aborts the run). Records the outcome
    in :data:`CACHE_STATUS`."""
    if not _cache_lock_enabled():
        CACHE_STATUS["locked"] = False
        CACHE_STATUS["max_size"] = -1
        # Actively restore the unlocked default so a later opt-out genuinely
        # turns eviction/lock off (not just in the status report). NOTE: JAX
        # builds its LRUCache lazily ONCE, so this only takes effect for a config
        # that runs BEFORE the first compile -- which is the import-hook ordering.
        try:
            jax_config.update("jax_compilation_cache_max_size", -1)
        except Exception:  # pragma: no cover - knob absent/renamed
            pass
        return
    # ``filelock`` must be importable for eviction-enabled LRUCache; if it is not
    # we must NOT set the sentinel (JAX would raise on cache build). Fail-open to
    # the legacy unlocked cache.
    try:
        import filelock as _filelock  # noqa: F401
    except Exception as exc:  # pragma: no cover - filelock is a JAX dep here
        CACHE_STATUS["locked"] = False
        CACHE_STATUS["max_size"] = -1
        try:
            _LOG.warning(
                "gpuwrf compile cache: filelock unavailable (%s); leaving the "
                "cache UNLOCKED. Concurrent-process prewarm is unsafe -- set "
                "GPUWRF_NESTED_PARALLEL_COMPILE=1 only with filelock installed.",
                exc,
            )
        except Exception:  # pragma: no cover
            pass
        return
    try:
        timeout = _cache_lock_timeout()
        jax_config.update("jax_compilation_cache_max_size", _CACHE_LOCK_SENTINEL_BYTES)
        # The lock_timeout is an LRUCache __init__ kwarg, not a JAX config knob;
        # we surface our chosen value for the driver + apply it via the atomic
        # writer install path (which can rebuild the cache object). JAX's own
        # default (10s) matches ours, so no further wiring is required for the
        # primary lock; we only record it.
        CACHE_STATUS["locked"] = True
        CACHE_STATUS["max_size"] = _CACHE_LOCK_SENTINEL_BYTES
        CACHE_STATUS["lock_timeout"] = timeout
        _install_atomic_cache_writer()
    except Exception as exc:  # pragma: no cover - never fail the cache setup
        CACHE_STATUS["locked"] = False
        CACHE_STATUS["max_size"] = -1
        CACHE_STATUS["error"] = f"cache-lock config failed: {exc}"


def _install_atomic_cache_writer() -> None:
    """Belt-and-braces: make ``LRUCache.put`` write temp+rename (atomic).

    The FileLock guards readers from a writer's in-progress ``write_bytes``, but
    if a writer CRASHES while holding the lock the OS releases the lock and could
    leave a TRUNCATED real ``<key>-cache`` file that a later reader decompresses
    into a hard ZstdError. We override the class ``put`` to write
    ``<key>-cache.<pid>.tmp`` then ``os.replace()`` into ``<key>-cache`` (an
    atomic same-fs rename), so a crash mid-write leaves only a stray ``.tmp`` and
    never a half-written real entry. Key/filename/bytes are unchanged => the
    unmodified reader path still gets a bit-identical warm hit. Installed at most
    once per process, BEFORE the first compile (same import-ordering as the rest
    of cache config). No-raise / import-safe / fully reversible-by-flag (it is a
    no-op when the lock is disabled, since put is only reached with the cache on).

    ALSO neutralizes ``_evict_if_needed`` (to a no-op). Two reasons: (1) the lock
    sentinel (256 GiB) is far above any real cache, so eviction never legitimately
    fires -- the glob+atime walk is pure cost; (2) CRITICALLY, a pre-existing B1
    WARM cache was written with eviction OFF (no ``-atime`` siblings), so flipping
    eviction ON would make stock ``_evict_if_needed`` raise ``FileNotFoundError``
    on the first put (it reads ``<key>-atime`` for every legacy ``<key>-cache``).
    Skipping eviction makes engaging the lock on an existing warm cache safe and
    keeps every legacy entry a valid warm hit. Numerically inert (the executable
    bytes/keys/filenames are untouched; only the LRU bookkeeping is dropped)."""
    global _ATOMIC_WRITER_INSTALLED
    if _ATOMIC_WRITER_INSTALLED:
        CACHE_STATUS["atomic_writer"] = True
        return
    try:
        import os as _os
        import time as _time
        import warnings as _warnings

        from jax._src import lru_cache as _lru

        _LRUCache = _lru.LRUCache
        _SUFFIX = getattr(_lru, "_CACHE_SUFFIX", "-cache")
        _ATIME = getattr(_lru, "_ATIME_SUFFIX", "-atime")

        if not getattr(_LRUCache.__init__, "_gpuwrf_timeout", False):
            _orig_init = _LRUCache.__init__
            _UNSET = object()

            def _init_with_gpuwrf_timeout(
                self,
                path: str,
                *,
                max_size: int,
                lock_timeout_secs=_UNSET,
            ) -> None:
                if lock_timeout_secs is _UNSET:
                    lock_timeout_secs = _cache_lock_timeout()
                _orig_init(
                    self,
                    path,
                    max_size=max_size,
                    lock_timeout_secs=lock_timeout_secs,
                )

            _init_with_gpuwrf_timeout._gpuwrf_timeout = True  # type: ignore[attr-defined]
            _LRUCache.__init__ = _init_with_gpuwrf_timeout  # type: ignore[assignment]

        if getattr(_LRUCache.put, "_gpuwrf_atomic", False):
            _ATOMIC_WRITER_INSTALLED = True
            CACHE_STATUS["atomic_writer"] = True
            return

        def _atomic_put(self, key: str, value: bytes) -> None:
            if not key:
                raise ValueError("key cannot be empty")
            if self.eviction_enabled and len(value) > self.max_size:
                msg = (
                    f"Cache value for key {key!r} of size {len(value)} bytes "
                    f"exceeds the maximum cache size of {self.max_size} bytes"
                )
                _warnings.warn(msg)
                return
            cache_path = self.path / f"{key}{_SUFFIX}"
            if self.eviction_enabled:
                self.lock.acquire(timeout=self.lock_timeout_secs)
            try:
                if cache_path.exists():
                    return
                # NOTE: intentionally do NOT call _evict_if_needed -- the sentinel
                # max_size never legitimately evicts, and stock eviction crashes
                # on a pre-existing eviction-OFF warm cache (missing -atime files).
                # ATOMIC write: temp file in the SAME dir (same fs) then replace.
                tmp_path = self.path / f"{key}{_SUFFIX}.{_os.getpid()}.tmp"
                try:
                    tmp_path.write_bytes(value)
                    _os.replace(str(tmp_path), str(cache_path))
                finally:
                    # Clean up a stray temp if replace did not consume it.
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:  # pragma: no cover - best-effort cleanup
                        pass
                if self.eviction_enabled:
                    timestamp = _time.time_ns().to_bytes(8, "little")
                    atime_path = self.path / f"{key}{_ATIME}"
                    atime_path.write_bytes(timestamp)
            finally:
                if self.eviction_enabled:
                    self.lock.release()

        _atomic_put._gpuwrf_atomic = True  # type: ignore[attr-defined]
        _LRUCache.put = _atomic_put  # type: ignore[assignment]

        # Neutralize eviction on the CLASS too, so ANY code path (not just our
        # put) is safe against a pre-existing eviction-OFF warm cache. At the
        # sentinel size eviction is never legitimately needed, so this only drops
        # the LRU bookkeeping -- numerically inert; never touches executable bytes.
        if not getattr(_LRUCache._evict_if_needed, "_gpuwrf_noop", False):
            def _noop_evict(self, *, additional_size: int = 0) -> None:  # noqa: ANN001
                return None

            _noop_evict._gpuwrf_noop = True  # type: ignore[attr-defined]
            _LRUCache._evict_if_needed = _noop_evict  # type: ignore[assignment]

        _ATOMIC_WRITER_INSTALLED = True
        CACHE_STATUS["atomic_writer"] = True
    except Exception as exc:  # pragma: no cover - never fail the cache setup
        CACHE_STATUS["atomic_writer"] = False
        CACHE_STATUS["error"] = f"atomic-writer install failed: {exc}"


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
        "version_tag": version_cache_tag(),
        "cache_lock_path": str(cache_lock_path()) if cache_lock_path() else None,
    }


def cache_env_help() -> str:
    """One-line human summary of the env vars, for ``--help`` / docs / logs."""
    return (
        "Persistent XLA compile cache env vars: "
        "GPUWRF_JAX_CACHE=0 disables; "
        "JAX_COMPILATION_CACHE_DIR / GPUWRF_JAX_CACHE_DIR set an explicit dir "
        "(honoured verbatim; a mismatched version tag is warned about); "
        "GPUWRF_CACHE sets the cache root (version-keyed jit cache at "
        "$GPUWRF_CACHE/jit/<version-tag>); "
        "default $XDG_CACHE_HOME/gpuwrf/jit/<version-tag> or "
        "$HOME/.cache/gpuwrf/jit/<version-tag>, where <version-tag> = "
        "gpuwrf+jax+jaxlib version + backend (e.g. 0.20.2-jax0.10.0-jaxlib0.10.0"
        "-cuda_sm120), so different versions/backends never stale-hit. "
        "GPUWRF_JAX_CACHE_LOCK=0 disables the cross-process cache FileLock "
        "(default ON; required for safe concurrent-process parallel compile); "
        "GPUWRF_JAX_CACHE_LOCK_TIMEOUT widens the lock acquire timeout (default 10s)."
    )
