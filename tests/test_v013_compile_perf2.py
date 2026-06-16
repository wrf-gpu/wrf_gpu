"""v0.13 Tier2 compile/runtime-hygiene tests (CPU, numerically inert).

Covers the two ADDITIVE, default-OFF knobs added this sprint:

#1 STANDALONE parallel-compile knob  -> runtime.xla_autotune.configure_parallel_compile
   (GPUWRF_XLA_PARALLEL_COMPILE / GPUWRF_XLA_COMPILE_PARALLELISM), INDEPENDENT of
   the persistent autotune cache, composing with the SAME subprocess flag-probe.

#2 Recompile-hygiene of the hot @jit entrypoint pattern (traced start_step,
   n_steps, and cadence) -> measured via jax.jit._cache_size().

REGRESSION GUARD (v0.12.0 revert 969d435): the autotune cache injected --xla_gpu_*
flags into XLA_FLAGS at import that the bundled GPU jaxlib REJECTED -> XLA printed
flag-help and FATALLY ABORTED. The new standalone parallel-compile knob reuses the
same hard-safe construction: OFF by default, respects the platform pin, and
validates its single --xla_gpu_* flag against the build in an isolated subprocess
before injecting -- so an unknown flag can never abort the main process.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import compile_cache as cc
from gpuwrf.runtime import xla_autotune as at


@pytest.fixture
def restore_global_cache_dir():
    """Save + restore the GLOBAL JAX compile-cache dir and CACHE_STATUS.

    Tests that call ``configure_compilation_cache()`` with a tmp_path re-point
    JAX's process-global ``jax_compilation_cache_dir`` (and ``cc.CACHE_STATUS``) to
    a directory pytest deletes on teardown. Without restoring it, a LATER test in
    the same session (e.g. test_v0130_compile_speed::test_aot_warm_...) finds the
    cache dir gone and its warm-hit/entry-count assertions break. This fixture
    restores both so cache-mutating tests are self-contained."""
    saved_status = dict(cc.CACHE_STATUS)
    saved_dir = cc.resolve_cache_dir()
    try:
        yield
    finally:
        cc.CACHE_STATUS.clear()
        cc.CACHE_STATUS.update(saved_status)
        if saved_dir is not None:
            try:
                from jax import config as _jc

                _jc.update("jax_compilation_cache_dir", str(saved_dir))
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# #1 Standalone parallel-compile knob: resolve()
# --------------------------------------------------------------------------- #
def test_parallel_resolve_not_opted_in_by_default(monkeypatch):
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    assert at.resolve_parallel_compile() is None
    assert "not-opted-in" in str(at.PARALLEL_COMPILE_STATUS["source"])


def test_parallel_resolve_explicit_count(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "6")
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    assert at.resolve_parallel_compile() == 6


def test_parallel_resolve_truthy_uses_legacy_count(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "on")
    monkeypatch.setenv("GPUWRF_XLA_COMPILE_PARALLELISM", "3")
    assert at.resolve_parallel_compile() == 3


def test_parallel_resolve_truthy_defaults_to_cpu_count_capped(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "true")
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    n = at.resolve_parallel_compile()
    assert isinstance(n, int) and 1 <= n <= 8


def test_parallel_resolve_legacy_var_alone_opts_in(monkeypatch):
    """GPUWRF_XLA_COMPILE_PARALLELISM alone (the legacy var, no primary var) must
    still opt in -- a deployment that only set the count keeps working."""
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_COMPILE_PARALLELISM", "5")
    assert at.resolve_parallel_compile() == 5
    assert "GPUWRF_XLA_COMPILE_PARALLELISM" in str(at.PARALLEL_COMPILE_STATUS["source"])


def test_parallel_resolve_disabled_by_zero(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "0")
    assert at.resolve_parallel_compile() is None
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "off")
    assert at.resolve_parallel_compile() is None


def test_parallel_resolve_primary_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "2")
    monkeypatch.setenv("GPUWRF_XLA_COMPILE_PARALLELISM", "7")
    assert at.resolve_parallel_compile() == 2  # primary (count) wins


# --------------------------------------------------------------------------- #
# #1 Standalone parallel-compile knob: configure() (hard-safe inject)
# --------------------------------------------------------------------------- #
def test_parallel_default_is_pure_noop(monkeypatch):
    """REGRESSION GUARD: with the parallel-compile knob NOT opted in (default),
    configure_parallel_compile must be a pure no-op even on a GPU box with no
    platform pin -- it must NOT inject any --xla_gpu_* flag. This keeps the default
    GPU path byte-unchanged (independent of the autotune cache)."""
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # GPU "present"
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_parallel_compile()
    assert status["opted_in"] is False
    assert status["enabled"] is False
    assert status["injected_flags"] in (None, [])
    assert os.environ.get("XLA_FLAGS", "") == before


def test_parallel_respects_cpu_pin_even_opted_in(monkeypatch):
    """Even GPUWRF_XLA_PARALLEL_COMPILE=N must not inject when cpu is pinned (a CPU
    jaxlib can fatally abort on an unknown --xla_gpu_* flag)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "4")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_parallel_compile()
    assert status["enabled"] is False
    assert status["opted_in"] is True
    assert "cpu" in str(status["reason"])
    assert os.environ.get("XLA_FLAGS", "") == before


def test_parallel_drops_flag_when_probe_rejects(monkeypatch):
    """THE core regression fix for this knob: when the build-validation probe
    reports the parallel-compile flag is UNSUPPORTED, it must be DROPPED (recorded
    in rejected_flags, NOT injected) and configure must NOT abort. We stub the
    probe to reject (simulating the bundled-GPU-jaxlib-rejects-the-flag case)."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # simulate a GPU target
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(
        at, "probe_flag_supported",
        lambda flag, timeout_s=at._PROBE_TIMEOUT_S: (False, "rejected:unknown flag"),
    )
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_parallel_compile()
    assert status["enabled"] is False
    assert status["injected_flags"] in (None, [])
    assert status["rejected_flags"], "rejected flag must be recorded"
    assert os.environ.get("XLA_FLAGS", "") == before


def test_parallel_injects_when_probe_accepts(monkeypatch):
    """When the probe accepts the flag, inject exactly the one
    --xla_gpu_force_compilation_parallelism flag with the requested count."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(
        at, "probe_flag_supported",
        lambda flag, timeout_s=at._PROBE_TIMEOUT_S: (True, "accepted"),
    )
    status = at.configure_parallel_compile()
    assert status["enabled"] is True
    assert status["injected_flags"] == ["--xla_gpu_force_compilation_parallelism=4"]
    flags = os.environ.get("XLA_FLAGS", "")
    assert "--xla_gpu_force_compilation_parallelism=4" in flags


def test_parallel_does_not_clobber_operator_preset(monkeypatch):
    """If the operator already set --xla_gpu_force_compilation_parallelism, respect
    it: inject nothing, leave a single occurrence."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=2")
    monkeypatch.setattr(
        at, "probe_flag_supported",
        lambda flag, timeout_s=at._PROBE_TIMEOUT_S: (True, "accepted"),
    )
    status = at.configure_parallel_compile()
    assert status["enabled"] is False  # nothing injected
    flags = os.environ["XLA_FLAGS"]
    assert flags.count("force_compilation_parallelism") == 1
    assert "=2" in flags  # operator's value preserved


def test_parallel_disabled_by_env(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "0")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_parallel_compile()
    assert status["enabled"] is False
    assert os.environ.get("XLA_FLAGS", "") == before


def test_parallel_env_help_mentions_key_vars():
    h = at.parallel_compile_env_help()
    for token in ("GPUWRF_XLA_PARALLEL_COMPILE", "GPUWRF_XLA_COMPILE_PARALLELISM",
                  "force_compilation_parallelism"):
        assert token in h


def test_compile_cache_surfaces_parallel_status(monkeypatch, tmp_path, restore_global_cache_dir):
    """The central import hook (configure_compilation_cache -> configure_parallel_compile)
    must surface the parallel-compile status under CACHE_STATUS['parallel_compile']."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path / "jit"))
    status = cc.configure_compilation_cache()
    assert "parallel_compile" in status
    assert isinstance(status["parallel_compile"], dict)


def test_compile_cache_does_not_inject_parallel_flag_by_default(monkeypatch, tmp_path, restore_global_cache_dir):
    """In-process analogue of the import-inertness guard for the parallel knob: the
    hook must NOT mutate XLA_FLAGS in the default case even with a GPU detected and
    no platform pin."""
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # GPU "present"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path / "jit"))
    before = os.environ.get("XLA_FLAGS", "")
    cc.configure_compilation_cache()
    assert os.environ.get("XLA_FLAGS", "") == before


# --------------------------------------------------------------------------- #
# #2 Recompile-hygiene of the hot @jit entrypoint pattern
# --------------------------------------------------------------------------- #
def _chunk_like_advance_chunk():
    """A jit'd loop mirroring operational_mode._advance_chunk.

    ``start_step``, ``n_steps``, and ``cadence`` are traced int32 scalars so varying
    interval lengths/cadences do not create new XLA cache entries.
    """

    @jax.jit
    def chunk(carry, start_step, *, n_steps, cadence):
        start_step = jnp.asarray(start_step, dtype=jnp.int32)
        n_steps = jnp.asarray(n_steps, dtype=jnp.int32)
        cadence = jnp.asarray(cadence, dtype=jnp.int32)

        def body(offset, c):
            step = start_step + offset
            increment = jnp.where(
                jnp.equal(jnp.mod(step, cadence), 0),
                10 * step,
                step,
            )
            return c + increment.astype(c.dtype)

        return jax.lax.fori_loop(jnp.asarray(0, dtype=jnp.int32), n_steps, body, carry)

    return chunk


def test_advance_chunk_pattern_does_not_recompile_across_intervals():
    """The hot-entrypoint pattern reuses one executable across start offsets."""
    chunk = _chunk_like_advance_chunk()
    carry = jnp.ones((16, 16), dtype=jnp.float64)
    chunk.clear_cache()
    for s in (1, 7, 13, 19, 7, 25):
        out = chunk(
            carry,
            jnp.asarray(s, dtype=jnp.int32),
            n_steps=jnp.asarray(4, dtype=jnp.int32),
            cadence=jnp.asarray(6, dtype=jnp.int32),
        )
        out.block_until_ready()
    assert chunk._cache_size() == 1, (
        f"varying traced start_step must not recompile; cache={chunk._cache_size()}"
    )


def test_python_int_start_step_would_add_a_trace():
    """Documents the weak-typing trap: a python-int start_step (instead of an int32
    device array) adds a distinct trace. This is WHY production callers wrap
    start_step in jnp.asarray(..., dtype=jnp.int32) -- the recompile this avoids."""
    chunk = _chunk_like_advance_chunk()
    carry = jnp.ones((16, 16), dtype=jnp.float64)
    chunk.clear_cache()
    # Hygienic int32 caller: 1 trace.
    chunk(
        carry,
        jnp.asarray(1, dtype=jnp.int32),
        n_steps=jnp.asarray(4, dtype=jnp.int32),
        cadence=jnp.asarray(6, dtype=jnp.int32),
    ).block_until_ready()
    assert chunk._cache_size() == 1
    # Naive python-int caller: a 2nd trace appears.
    chunk(
        carry,
        1,
        n_steps=jnp.asarray(4, dtype=jnp.int32),
        cadence=jnp.asarray(6, dtype=jnp.int32),
    ).block_until_ready()
    assert chunk._cache_size() == 2


def test_int32_and_pyint_start_step_give_identical_results():
    """The int32 re-cast inside the chunk normalises start_step, so the two caller
    styles produce bit-for-bit identical results (numerically inert hygiene)."""
    chunk = _chunk_like_advance_chunk()
    carry = jnp.ones((16, 16), dtype=jnp.float64)
    kwargs = {
        "n_steps": jnp.asarray(4, dtype=jnp.int32),
        "cadence": jnp.asarray(6, dtype=jnp.int32),
    }
    out_i = np.asarray(chunk(carry, jnp.asarray(7, dtype=jnp.int32), **kwargs))
    out_p = np.asarray(chunk(carry, 7, **kwargs))
    np.testing.assert_array_equal(out_i, out_p)


def test_dynamic_n_steps_and_cadence_do_not_recompile():
    """Varying interval length/cadence must not mint a new cache entry."""
    chunk = _chunk_like_advance_chunk()
    carry = jnp.ones((16, 16), dtype=jnp.float64)
    chunk.clear_cache()
    for start, n_steps, cadence in ((1, 4, 6), (5, 8, 6), (13, 3, 10), (21, 4, 10)):
        chunk(
            carry,
            jnp.asarray(start, dtype=jnp.int32),
            n_steps=jnp.asarray(n_steps, dtype=jnp.int32),
            cadence=jnp.asarray(cadence, dtype=jnp.int32),
        ).block_until_ready()
    assert chunk._cache_size() == 1


def test_dynamic_loop_matches_static_scan_reference_bitwise():
    """The dynamic loop preserves the same per-step order as the former scan body."""
    chunk = _chunk_like_advance_chunk()
    carry = jnp.ones((16, 16), dtype=jnp.float64)

    def reference(start_step: int, n_steps: int, cadence: int):
        idx = jnp.asarray(start_step, dtype=jnp.int32) + jnp.arange(n_steps, dtype=jnp.int32)

        def body(c, step):
            increment = jnp.where(jnp.equal(jnp.mod(step, cadence), 0), 10 * step, step)
            return c + increment.astype(c.dtype), None

        out, _ = jax.lax.scan(body, carry, idx)
        return out

    for start, n_steps, cadence in ((1, 4, 6), (5, 8, 6), (13, 3, 10), (21, 4, 10)):
        out = chunk(
            carry,
            jnp.asarray(start, dtype=jnp.int32),
            n_steps=jnp.asarray(n_steps, dtype=jnp.int32),
            cadence=jnp.asarray(cadence, dtype=jnp.int32),
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(reference(start, n_steps, cadence)))
