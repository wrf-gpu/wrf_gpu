"""v0.13 compile-speed infrastructure tests (CPU, numerically inert).

Covers the three v0.13 roadmap items implemented in this sprint:

#1 AOT precompile entrypoint   -> runtime.aot_precompile
#2 Persistent XLA autotune cache -> runtime.xla_autotune
#3 Hardened persistent compile cache -> runtime.compile_cache

These tests assert the cache/AOT path **loads** and produces **identical
results** to a fresh compile, and that the GPU-only autotune flags are guarded
off on CPU (so a CPU box never fatally aborts on an unknown --xla_gpu_* flag).
All assertions are backend-agnostic; they run on the CPU CI box.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_precompile as aot
from gpuwrf.runtime import compile_cache as cc
from gpuwrf.runtime import xla_autotune as at


# --------------------------------------------------------------------------- #
# #1 AOT precompile entrypoint
# --------------------------------------------------------------------------- #
def _representative_graph(u0):
    """fp64 scan(stencil + GEMM): same XLA compile-cost classes as a WRF step."""

    def step(state, _x):
        u, k = state
        lap = (
            jnp.roll(u, 1, 0)
            + jnp.roll(u, -1, 0)
            + jnp.roll(u, 1, 1)
            + jnp.roll(u, -1, 1)
            - 4.0 * u
        )
        mixed = jnp.tanh(u @ u.T) * 1e-3
        u = u + 0.01 * lap + mixed + jnp.sin(u + k)
        return (u, k + 1.0), None

    (u, _), _ = jax.lax.scan(step, (u0, 0.0), xs=None, length=8)
    return u


def test_precompile_returns_callable_executable_and_runs():
    u0 = jnp.ones((32, 32), dtype=jnp.float64)
    compiled, result = aot.precompile(_representative_graph, u0, key="test-graph")
    assert result.error is None, result.error
    assert compiled is not None
    out = compiled(u0)
    out.block_until_ready()
    assert out.shape == (32, 32)
    assert result.compile_seconds >= 0.0
    assert result.key == "test-graph"


def test_aot_warm_compile_is_cache_hit_and_identical():
    """Compile once (cold), then recompile the IDENTICAL program; the second
    compile must be a warm cache hit (no new on-disk entry) AND yield bit-for-bit
    identical output. This is the in-process analogue of the cross-process proof
    in proofs/v0130/compile_speed.py.

    Uses the cache dir gpuwrf actually configured at import (the persistent
    cache must already be wired by the import hook). The on-disk entry-count
    delta is what makes the warm-hit claim robust to JAX's opaque HLO key. If
    caching is disabled in the environment (GPUWRF_JAX_CACHE=0), skip -- the
    warm-hit-by-disk-entry claim is meaningless without a cache dir."""
    if not cc.CACHE_STATUS.get("enabled") or not cc.CACHE_STATUS.get("dir"):
        pytest.skip("persistent compile cache disabled in this environment")
    cache_dir = cc.CACHE_STATUS["dir"]

    u0 = jnp.full((40, 40), 0.5, dtype=jnp.float64)

    # Use a distinct constant + nonce so this program's HLO is unique to this
    # test run (avoids a pre-existing on-disk entry making the "cold" compile a
    # warm hit and the entry-count delta zero).
    nonce = float(os.getpid() % 9973) + 0.123

    def graph(x):
        return _representative_graph(x) + nonce

    # Cold compile -> writes at least one new on-disk entry.
    n0 = cc.cache_entry_count(cache_dir)
    compiled_cold, res_cold = aot.precompile(graph, u0, key="cold")
    assert res_cold.error is None, res_cold.error
    n1 = cc.cache_entry_count(cache_dir)
    assert n1 > n0, "cold compile should write a cache entry"
    out_cold = np.asarray(compiled_cold(u0))

    # Warm compile of the identical program via the warm-hit detector: no NEW
    # entry should appear (served from disk/in-memory cache).
    info = cc.warm_hit_for(lambda: aot.precompile(graph, u0, key="warm"), cache_dir)
    assert info["after"] == info["before"], (
        f"warm compile must not add a cache entry, got {info}"
    )

    compiled_warm, res_warm = aot.precompile(graph, u0, key="warm2")
    out_warm = np.asarray(compiled_warm(u0))

    # Identical results: the cached executable is the SAME XLA program.
    assert out_cold.shape == out_warm.shape
    np.testing.assert_array_equal(out_cold, out_warm)


def test_config_key_is_stable_and_distinguishes_grids():
    a = aot.GridConfig(name="g", nx=100, ny=100, nz=45, n_domains=1, dt_s=18.0)
    b = aot.GridConfig(name="g", nx=100, ny=100, nz=45, n_domains=1, dt_s=18.0)
    assert aot.config_key(a) == aot.config_key(b)

    # Different shape / dom count / flag => different key.
    assert aot.config_key(a) != aot.config_key(
        aot.GridConfig(name="g", nx=128, ny=100, nz=45)
    )
    assert aot.config_key(a) != aot.config_key(
        aot.GridConfig(name="g", nx=100, ny=100, nz=45, n_domains=2)
    )
    a1 = a.with_flags(cu_physics=1)
    a0 = a.with_flags(cu_physics=0)
    assert aot.config_key(a1) != aot.config_key(a0)

    # with_flags is order-independent (sorted tuple => stable key).
    f1 = a.with_flags(gwd_opt=1, cu_physics=0)
    f2 = a.with_flags(cu_physics=0, gwd_opt=1)
    assert aot.config_key(f1) == aot.config_key(f2)


def test_production_grids_registry_has_canary_chain():
    names = {g.name for g in aot.PRODUCTION_GRIDS}
    assert {"canary-d01-9km", "canary-d02-3km", "canary-d03-1km"} <= names
    # Every entry has a positive shape.
    for g in aot.PRODUCTION_GRIDS:
        assert g.nx > 0 and g.ny > 0 and g.nz > 0 and g.n_domains >= 1


def test_prewarm_uses_injected_spec_provider():
    """prewarm() must NOT hard-import State/IO; it drives a supplied provider."""
    calls = []

    def provider(cfg):
        calls.append(cfg.name)
        u0 = jnp.ones((16, 16), dtype=jnp.float64)
        return (_representative_graph, (u0,), {}, False)

    cfgs = (aot.GridConfig(name="tiny", nx=16, ny=16, nz=4),)
    results = aot.prewarm(provider, configs=cfgs)
    assert calls == ["tiny"]
    assert len(results) == 1
    assert results[0].error is None, results[0].error
    assert results[0].key.startswith("tiny__16x16x4")


def test_prewarm_records_provider_errors_without_crashing():
    def bad_provider(cfg):
        raise RuntimeError("boom")

    cfgs = (aot.GridConfig(name="x", nx=8, ny=8, nz=2),)
    results = aot.prewarm(bad_provider, configs=cfgs)
    assert results[0].error is not None
    assert "boom" in results[0].error


# --------------------------------------------------------------------------- #
# #2 Persistent XLA autotune cache (GPU-only; guarded off on CPU)
# --------------------------------------------------------------------------- #
def test_autotune_cache_disabled_when_cpu_pinned(monkeypatch):
    """On a CPU-pinned box the --xla_gpu_* flags must NOT be injected (a CPU
    jaxlib FATALLY aborts on an unknown --xla_gpu_* flag)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["enabled"] is False
    assert "cpu" in str(status["reason"])
    # XLA_FLAGS unchanged.
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_force_still_respects_cpu_pin(monkeypatch):
    """Even GPUWRF_XLA_AUTOTUNE_CACHE=1 must not inject when cpu is pinned."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["enabled"] is False
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_injects_flags_when_gpu_forced_no_pin(monkeypatch, tmp_path):
    """With no platform pin + force, the autotune-cache + parallel flags are
    injected into XLA_FLAGS with the version-matched flag names, never clobbering
    an operator-set flag. (We do NOT import jax here, so no fatal-abort risk.)"""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", str(tmp_path / "at"))
    monkeypatch.setenv("GPUWRF_XLA_COMPILE_PARALLELISM", "4")
    # Operator pre-set the cache mode; we must NOT re-add it.
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_experimental_autotune_cache_mode=read")

    status = at.configure_autotune_cache()
    assert status["enabled"] is True
    flags = os.environ["XLA_FLAGS"]
    # Operator flag preserved.
    assert "--xla_gpu_experimental_autotune_cache_mode=read" in flags
    # We injected the dir + parallelism (version-matched flag names).
    assert "--xla_gpu_per_fusion_autotune_cache_dir=" in flags
    assert "--xla_gpu_force_compilation_parallelism=4" in flags
    # Never clobbered the pre-set mode (only one occurrence).
    assert flags.count("xla_gpu_experimental_autotune_cache_mode") == 1
    injected = status["injected_flags"]
    assert not any("autotune_cache_mode" in f for f in injected)


def test_autotune_disabled_by_env(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "0")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["enabled"] is False
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_env_help_mentions_key_vars():
    h = at.autotune_env_help()
    for token in ("GPUWRF_XLA_AUTOTUNE_CACHE", "GPUWRF_XLA_COMPILE_PARALLELISM"):
        assert token in h


# --------------------------------------------------------------------------- #
# #3 Hardened persistent compile cache
# --------------------------------------------------------------------------- #
def test_cache_entry_count_zero_when_missing(tmp_path):
    assert cc.cache_entry_count(tmp_path / "does-not-exist") == 0


def test_cache_entry_count_counts_cache_dirs(tmp_path):
    (tmp_path / "jit_foo-abc-cache").mkdir()
    (tmp_path / "jit_bar-def-cache").mkdir()
    (tmp_path / "unrelated.txt").write_text("x")
    # Counts the *-cache entries.
    assert cc.cache_entry_count(tmp_path) == 2


def test_cache_report_shape():
    rep = cc.cache_report()
    assert "compile_cache" in rep
    assert "compile_cache_entries" in rep


def test_configure_compilation_cache_wires_autotune_status(monkeypatch, tmp_path):
    """configure_compilation_cache must also surface the autotune status under
    CACHE_STATUS['autotune'] (single import hook configures both)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path / "jit"))
    status = cc.configure_compilation_cache()
    assert "autotune" in status
    assert isinstance(status["autotune"], dict)


def test_cache_env_help_mentions_disable_var():
    assert "GPUWRF_JAX_CACHE" in cc.cache_env_help()
