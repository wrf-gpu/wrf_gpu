"""Per-test private persistent JAX compilation cache (FINDINGS E127).

Setting ``GPUWRF_JAX_CACHE_DIR`` and calling ``configure_compilation_cache()`` is
NOT enough to give a test its own cache:

* JAX builds its file cache once per process (``jax._src.compilation_cache``);
  eager ops at import already initialise it at the ambient directory, and later
  ``jax_compilation_cache_dir`` updates are ignored until ``reset_cache()``;
* ``resolve_cache_dir`` ranks ``JAX_COMPILATION_CACHE_DIR`` (and the
  ``GPUWRF_JAX_CACHE`` switch) above ``GPUWRF_JAX_CACHE_DIR``;
* the configured directory is part of JAX's cache key (per-fusion autotune cache
  dir option), so a fixed ``--basetemp`` brings back the same key next run.

So entries landed in the ambient (often shared) cache, and from the second run on
a compile became a persistent-cache HIT -- on which a CPU ``aotx.serialize`` drops
the native objects (B12d) and the test fails.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import jax
from jax._src import compilation_cache as jcc

from gpuwrf.runtime import compile_cache as cc

_OUTRANKING_ENV = ("GPUWRF_JAX_CACHE", "JAX_COMPILATION_CACHE_DIR")


@contextlib.contextmanager
def private_jax_cache(monkeypatch, cache_dir: Path):
    """Point JAX's persistent cache at ``cache_dir`` for the duration of a test."""
    previous = jax.config.jax_compilation_cache_dir
    for name in _OUTRANKING_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(cache_dir))
    status = cc.configure_compilation_cache()
    assert jax.config.jax_compilation_cache_dir == str(cache_dir), status
    jcc.reset_cache()  # the next compile re-initialises the file cache at cache_dir
    try:
        yield cache_dir
    finally:
        jcc.reset_cache()
        jax.config.update("jax_compilation_cache_dir", previous)
