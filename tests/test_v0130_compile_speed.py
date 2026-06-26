"""v0.13 compile-speed infrastructure tests (CPU, numerically inert).

Covers the three v0.13 roadmap items implemented in this sprint:

#1 AOT precompile entrypoint   -> runtime.aot_precompile
#2 Persistent XLA autotune cache -> runtime.xla_autotune
#3 Hardened persistent compile cache -> runtime.compile_cache

These tests assert the cache/AOT path **loads** and produces **identical
results** to a fresh compile, and that the GPU-only autotune flags are guarded
off on CPU (so a CPU box never fatally aborts on an unknown --xla_gpu_* flag).
All assertions are backend-agnostic; they run on the CPU CI box.

REGRESSION GUARD (v0.12.0 revert 969d435): the autotune cache injected
--xla_gpu_* flags into XLA_FLAGS at import that the bundled GPU jaxlib REJECTED,
so XLA printed flag-help and FATALLY ABORTED the GPU path. The re-landed version
is OFF by default (requires an explicit GPUWRF_XLA_AUTOTUNE_CACHE=1 opt-in) and
validates each flag against the installed build in an isolated subprocess before
injecting. The tests below assert: (a) the default path injects NOTHING and is
byte-unchanged; (b) an unsupported flag is dropped + logged, never injected; (c)
import is inert (no XLA_FLAGS mutation) when not opted in or when the probe
rejects the flag.
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
    # warm hit and the entry-count delta zero). The nonce must be unique per RUN,
    # not per pid: B1 made the default cache dir version-keyed and hence stable
    # across runs, so a pid-only nonce could collide with an entry a prior run of
    # this same test left on disk (the "cold" compile would then be a warm hit
    # and the delta zero). Use full uuid4 entropy so back-to-back runs never
    # collide regardless of clock resolution.
    import uuid as _uuid

    nonce = (int(_uuid.uuid4().int) % (2**52)) * 1e-7

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
# #2 Persistent XLA autotune cache (GPU-only, OFF by default; hard-safe inject)
# --------------------------------------------------------------------------- #
def test_autotune_default_is_pure_noop(monkeypatch):
    """REGRESSION GUARD (the v0.12.0 revert root cause): with the autotune cache
    NOT explicitly opted-in (the default), configure_autotune_cache must be a
    pure no-op -- it must NOT inject any --xla_gpu_* flag into XLA_FLAGS, even on
    a box where a GPU is detected. This keeps the default GPU path byte-unchanged
    (the abort was caused by always-on-when-GPU injection at import)."""
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)
    # Pretend a GPU is present + no platform pin: under the OLD (buggy) logic
    # this would have injected flags. Under the fix it must stay a no-op.
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["opted_in"] is False
    assert status["enabled"] is False
    assert status["injected_flags"] in (None, [])
    # XLA_FLAGS byte-unchanged.
    assert os.environ.get("XLA_FLAGS", "") == before
    assert "not-opted-in" in str(status["source"])


def test_autotune_cache_disabled_when_cpu_pinned(monkeypatch):
    """On a CPU-pinned box the --xla_gpu_* flags must NOT be injected (a CPU
    jaxlib can FATALLY abort on an unknown --xla_gpu_* flag)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["enabled"] is False
    # XLA_FLAGS unchanged.
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_force_still_respects_cpu_pin(monkeypatch):
    """Even GPUWRF_XLA_AUTOTUNE_CACHE=1 must not inject when cpu is pinned."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["enabled"] is False
    assert "cpu" in str(status["reason"])
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_injects_only_when_opted_in_and_probe_skipped(monkeypatch, tmp_path):
    """With opt-in + no platform pin + the (slow) build-validation probe DISABLED
    (GPUWRF_XLA_AUTOTUNE_PROBE=0, operator's explicit responsibility), the
    autotune-cache + parallel flags are injected with the version-matched names,
    never clobbering an operator-set flag. The probe is skipped here so the CPU
    test box (no GPU) does not have to actually validate against a CUDA build."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_PROBE", "0")  # trust flags, skip probe
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # simulate a GPU target
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", str(tmp_path / "at"))
    monkeypatch.setenv("GPUWRF_XLA_COMPILE_PARALLELISM", "4")
    # Operator pre-set the cache mode; we must NOT re-add it.
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_experimental_autotune_cache_mode=read")

    status = at.configure_autotune_cache()
    assert status["opted_in"] is True
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


def test_autotune_drops_flag_when_probe_rejects(monkeypatch, tmp_path):
    """THE core regression fix: when the build-validation probe reports a flag is
    UNSUPPORTED, that flag must be DROPPED (logged in status, NOT injected into
    XLA_FLAGS), and configure_autotune_cache must NOT abort. We stub the probe to
    reject everything (simulating the bundled-GPU-jaxlib-rejects-the-flag case
    that caused the v0.12.0 abort)."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # simulate a GPU target
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", str(tmp_path / "at"))
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    # Stub the probe: every flag is rejected by the (simulated) build.
    monkeypatch.setattr(
        at, "probe_flag_supported",
        lambda flag, timeout_s=at._PROBE_TIMEOUT_S: (False, "rejected:unknown flag"),
    )
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    # Did NOT abort; injected NOTHING; recorded the rejection.
    assert status["enabled"] is False
    assert status["injected_flags"] in (None, [])
    assert status["rejected_flags"], "rejected flags must be recorded"
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_injects_only_probe_accepted_flags(monkeypatch, tmp_path):
    """When the probe accepts some flags and rejects others, only the accepted
    ones are injected; the rejected ones are recorded and skipped."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", str(tmp_path / "at"))
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    def fake_probe(flag, timeout_s=at._PROBE_TIMEOUT_S):
        # Accept the per-fusion dir; reject the cache-mode flag.
        if "per_fusion_autotune_cache_dir" in flag:
            return True, "accepted"
        return False, "rejected:unknown flag"

    monkeypatch.setattr(at, "probe_flag_supported", fake_probe)
    status = at.configure_autotune_cache()
    flags = os.environ.get("XLA_FLAGS", "")
    assert "--xla_gpu_per_fusion_autotune_cache_dir=" in flags
    assert "xla_gpu_experimental_autotune_cache_mode" not in flags
    assert status["enabled"] is True
    assert any("cache_mode" in f for f in (status["rejected_flags"] or []))


def test_probe_flag_supported_never_raises_and_isolates(monkeypatch):
    """The probe must be fully isolated: a failed/aborted child returns a False
    verdict, never raises, never aborts the parent. We point sys.executable-style
    invocation at a flag that the running (CPU) build will reject and assert a
    clean (False, detail) verdict -- proving the subprocess isolation works."""
    # On the CPU CI box this probe spawns a child that tries to init CUDA; with
    # no GPU the child fails cleanly and the parent sees ok=False. The KEY claim
    # is "no raise / no parent abort", which holds regardless of GPU presence.
    ok, detail = at.probe_flag_supported(
        "--xla_gpu_this_flag_definitely_does_not_exist=1", timeout_s=60.0
    )
    assert ok is False
    assert isinstance(detail, str) and detail


def test_autotune_disabled_by_env(monkeypatch):
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "0")
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_autotune_cache()
    assert status["enabled"] is False
    assert os.environ.get("XLA_FLAGS", "") == before


def test_autotune_env_help_mentions_key_vars():
    h = at.autotune_env_help()
    for token in (
        "GPUWRF_XLA_AUTOTUNE_CACHE",
        "GPUWRF_XLA_COMPILE_PARALLELISM",
        "GPUWRF_XLA_AUTOTUNE_PROBE",
    ):
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


def test_configure_compilation_cache_autotune_default_on_with_compile_cache(monkeypatch, tmp_path):
    """B1 CONTRACT CHANGE: the import hook (configure_compilation_cache ->
    configure_autotune_cache(default_on=True)) now turns the autotune cache ON by
    default WHEN the compile cache is on -- the autotune status must report
    opted_in=True with activation 'default-on-with-compile-cache' even though no
    GPUWRF_XLA_AUTOTUNE_CACHE env var is set. (Whether any --xla_gpu_* flag is
    actually injected still depends on the per-flag subprocess probe + a GPU
    target; that hard-safety is covered by the dedicated probe tests above and the
    opt-out test below.)"""
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)  # unset => default-on
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # GPU "present"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path / "jit"))
    status = cc.configure_compilation_cache()
    at_status = status.get("autotune") or {}
    assert at_status.get("opted_in") is True
    assert at_status.get("activation") == "default-on-with-compile-cache"


def test_configure_compilation_cache_autotune_opt_out_honored(monkeypatch, tmp_path):
    """The GPUWRF_XLA_AUTOTUNE_CACHE=0 opt-out must still win under B1: with it set,
    the import hook injects NOTHING and the autotune status reports disabled +
    not-opted-in even with a GPU detected and the compile cache on. XLA_FLAGS must
    be byte-unchanged by the autotune path (the regression-guard invariant)."""
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "0")  # explicit opt-out
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # GPU "present"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path / "jit"))
    before = os.environ.get("XLA_FLAGS", "")
    status = cc.configure_compilation_cache()
    at_status = status.get("autotune") or {}
    assert at_status.get("enabled") is False
    assert at_status.get("opted_in") is False
    # The opt-out path injects nothing new (it may not strip an operator/import
    # pre-set flag, but it adds none of its own).
    assert at_status.get("injected_flags") in (None, [])
    assert "disabled-by-GPUWRF_XLA_AUTOTUNE_CACHE" in str(at_status.get("source"))
    # XLA_FLAGS unchanged by THIS call.
    assert os.environ.get("XLA_FLAGS", "") == before


# --------------------------------------------------------------------------- #
# REGRESSION GUARD (the v0.12.0 abort): the REAL package import must NEVER abort,
# regardless of the autotune default. B1 turns the autotune cache ON by default
# (when the compile cache is on), so on a box with a real GPU the probe-accepted
# --xla_gpu_* flag IS now injected at import -- the regression guard is therefore
# "import returns rc=0 and any injected flag is one the build provably accepts",
# NOT "no flag is ever injected". The probe (isolated subprocess) is what keeps
# an UNKNOWN flag from ever aborting; we still assert rc=0 here as the definitive
# no-abort proof, and assert the opt-out (=0) path injects nothing (next test).
# --------------------------------------------------------------------------- #
def _import_gpuwrf_child(extra_env):
    import subprocess
    import sys

    repo_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    child = (
        "import os, sys; "
        "import gpuwrf; "
        "sys.stdout.write('XLA_FLAGS=' + os.environ.get('XLA_FLAGS','') + '\\n'); "
        "from gpuwrf.runtime.compile_cache import CACHE_STATUS; "
        "at = CACHE_STATUS.get('autotune') or {}; "
        "sys.stdout.write('AUTOTUNE_ENABLED=' + str(at.get('enabled')) + '\\n'); "
        "sys.stdout.write('AUTOTUNE_OPTED_IN=' + str(at.get('opted_in')) + '\\n')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("JAX_PLATFORMS", None)
    env.pop("JAX_PLATFORM_NAME", None)
    env.pop("XLA_FLAGS", None)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", child],
        env=env, capture_output=True, text=True, timeout=300,
    )
    return proc


def test_package_import_never_aborts_default_autotune_on():
    """B1: with the autotune cache at its new default (unset => on-with-compile-
    cache) and a GPU "detected", `import gpuwrf` must STILL return rc=0 (never the
    v0.12.0 fatal abort), because every injected flag is probe-validated first. We
    assert no abort, and that the autotune cache reports itself opted-in by the
    default-on path (a flag may or may not be injected depending on whether THIS
    box's build accepts it, which the dedicated probe tests cover)."""
    env = {}
    env.pop("GPUWRF_XLA_AUTOTUNE_CACHE", None)  # unset => default-on
    proc = _import_gpuwrf_child(env)
    assert proc.returncode == 0, (
        f"import gpuwrf aborted rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out = proc.stdout
    # Any flag present must be a recognised autotune-cache flag name (never a
    # rejected/unknown one -- a rejected flag is dropped, not injected).
    flag_line = next(l for l in out.splitlines() if l.startswith("XLA_FLAGS="))
    injected = flag_line[len("XLA_FLAGS="):]
    if "xla_gpu" in injected:
        assert "xla_gpu_per_fusion_autotune_cache_dir" in injected
    # Default-on means the autotune path reports opted_in=True at import.
    assert "AUTOTUNE_OPTED_IN=True" in out


def test_package_import_opt_out_injects_no_autotune_flags():
    """The GPUWRF_XLA_AUTOTUNE_CACHE=0 opt-out at import: `import gpuwrf` must NOT
    inject any --xla_gpu_* autotune flag and the autotune cache must report itself
    disabled + not-opted-in. (No-abort + zero autotune injection -- the operator's
    explicit opt-out is honoured end-to-end through the real import hook.)"""
    proc = _import_gpuwrf_child({"GPUWRF_XLA_AUTOTUNE_CACHE": "0"})
    assert proc.returncode == 0, (
        f"import gpuwrf aborted rc={proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out = proc.stdout
    flag_line = next(l for l in out.splitlines() if l.startswith("XLA_FLAGS="))
    injected = flag_line[len("XLA_FLAGS="):]
    assert "xla_gpu_per_fusion_autotune_cache_dir" not in injected
    assert "xla_gpu_experimental_autotune_cache_mode" not in injected
    assert "AUTOTUNE_ENABLED=False" in out
    assert "AUTOTUNE_OPTED_IN=False" in out


def test_cache_env_help_mentions_disable_var():
    assert "GPUWRF_JAX_CACHE" in cc.cache_env_help()
