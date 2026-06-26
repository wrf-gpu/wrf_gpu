"""Top-level (picklable) workers for the cross-domain parallel-compile tests.

These MUST live at module top level (not as test-local closures) so the
``spawn`` ProcessPoolExecutor can pickle + import them in the child. They run on
the CPU box with tiny synthetic graphs -- the goal is to exercise the MECHANISM
(spawned-process compile into the shared LOCKED version-keyed cache + a
main-process warm hit), not the heavy WRF body (whose State is GPU-only).
"""

from __future__ import annotations

import os
import time
from typing import Any


def _configure_child_cache(cache_dir: str) -> None:
    """Point the child at the shared cache dir + engage the lock, like gpuwrf."""
    os.environ["GPUWRF_JAX_CACHE_DIR"] = cache_dir
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from gpuwrf.runtime.compile_cache import configure_compilation_cache

    configure_compilation_cache()


def compile_distinct_graph(args: "tuple[str, int]") -> dict[str, Any]:
    """Compile ONE tiny graph whose HLO is distinct per ``which`` -> distinct key.

    Returns a plain dict (picklable). Never raises across the process boundary.
    """
    cache_dir, which = args
    t0 = time.perf_counter()
    try:
        _configure_child_cache(cache_dir)
        import jax
        import jax.numpy as jnp

        from gpuwrf.runtime.compile_cache import cache_entry_count

        before = cache_entry_count()
        # Distinct constant baked per `which` => distinct HLO => distinct key.
        const = float(which + 1)

        @jax.jit
        def fn(x):
            return x * const + jnp.asarray(which, dtype=jnp.float64)

        fn(jnp.ones((which + 2,), dtype=jnp.float64)).block_until_ready()
        after = cache_entry_count()
        return {
            "which": which,
            "before": before,
            "after": after,
            "wrote": after > before,
            "error": None,
        }
    except BaseException as exc:  # noqa: BLE001
        return {
            "which": which,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - t0,
        }


def crash_midway(args: "tuple[str, int]") -> dict[str, Any]:
    """A worker that deliberately CRASHES (os._exit) to test fail-open."""
    cache_dir, which = args
    _configure_child_cache(cache_dir)
    # Hard-exit before returning a result so the parent sees a broken worker.
    os._exit(7)


# --------------------------------------------------------------------------- #
# Low-level LRUCache.put workers (cache-safety tests): operate on the JAX
# LRUCache directly with distinct keys, under the lock sentinel.
# --------------------------------------------------------------------------- #
def lru_put_distinct(args: "tuple[str, str, int]") -> dict[str, Any]:
    """Open the version-keyed LRUCache (eviction/lock ON) and put one distinct key.

    Engages the SAME max_size sentinel + atomic-writer the runtime uses, then
    writes a distinct ``<key>-cache`` entry. Returns picklable status.
    """
    cache_dir, key, payload_size = args
    try:
        from gpuwrf.runtime import compile_cache as cc
        from jax._src.lru_cache import LRUCache

        # Ensure the atomic writer + lock sentinel are installed in THIS process.
        os.environ["GPUWRF_JAX_CACHE_DIR"] = cache_dir
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        cc._install_atomic_cache_writer()

        cache = LRUCache(cache_dir, max_size=cc._CACHE_LOCK_SENTINEL_BYTES)
        # A real zstd-compressed payload so the default reader can decompress it.
        import zlib  # stdlib; the test reader uses the same to verify intact

        value = zlib.compress(bytes((i % 251 for i in range(payload_size))), 6)
        cache.put(key, value)
        return {"key": key, "size": len(value), "error": None}
    except BaseException as exc:  # noqa: BLE001
        return {"key": key, "error": f"{type(exc).__name__}: {exc}"}


def atomic_write_then_crash(args: "tuple[str, str, int]") -> None:
    """Begin an atomic put then HARD-CRASH while (logically) mid-write.

    Because the override writes ``<key>-cache.<pid>.tmp`` and only ``os.replace``s
    into ``<key>-cache`` at the end, crashing before/at the temp write must NEVER
    leave a truncated real ``<key>-cache`` -- only a stray ``.tmp`` (or nothing).
    """
    cache_dir, key, payload_size = args
    from gpuwrf.runtime import compile_cache as cc
    from jax._src.lru_cache import LRUCache

    cc._install_atomic_cache_writer()
    cache = LRUCache(cache_dir, max_size=cc._CACHE_LOCK_SENTINEL_BYTES)
    # Write the temp file by hand exactly as the atomic put would, then crash
    # BEFORE the os.replace -- the worst case for the real entry's integrity.
    tmp = cache.path / f"{key}-cache.{os.getpid()}.tmp"
    tmp.write_bytes(bytes(payload_size))
    os._exit(9)  # crash mid-write: replace never happens
