"""Sprint B1 tests: version-keyed, default-on, fail-open compile/autotune cache.

These tests cover the B1 operational-safety change (numerically inert -- only
cache config + paths change). The motivating failure: a paid B200 run wasted a
~40 min cold compile because (a) the autotune cache was OFF by default and (b)
the default JIT cache dir was NOT version-keyed, so a v0.20.0 cache stale-missed
for a v0.20.2 binary with no operator signal.

Scope here (all CPU-runnable, no GPU required):

1. default-dir resolution + version-keyed shape;
2. version-keying: two DIFFERENT gpuwrf versions -> DIFFERENT dirs (never
   stale-hit), CPU vs CUDA backend -> different dirs;
3. opt-out env vars honoured (GPUWRF_JAX_CACHE=0; GPUWRF_XLA_AUTOTUNE_CACHE=0);
4. fail-OPEN on an unwritable / bad cache dir (warn + continue, never crash);
5. explicit-dir version-mismatch WARNING (operator-pinned stale tag);
6. autotune default-ON-with-compile-cache (and the =0 opt-out still wins).

The autotune-cache flag *injection* is GPU-only + subprocess-probe-gated; that
hard-safety is covered in test_v0130_compile_speed.py. Here we drive the dir/tag
resolution + status plumbing, which is backend-agnostic and runs on the CPU box.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpuwrf.runtime import compile_cache as cc
from gpuwrf.runtime import xla_autotune as at


@pytest.fixture(autouse=True)
def restore_cache_globals():
    """Save + restore the process-global JAX compile-cache dir + CACHE_STATUS.

    AUTOUSE so EVERY test in this file is self-contained: many B1 tests
    monkeypatch cache env vars and call configure_compilation_cache(), which
    repoints JAX's process-global jax_compilation_cache_dir (and cc.CACHE_STATUS)
    at a tmp dir pytest deletes on teardown. Without restoring BOTH, a LATER test
    in the same session (e.g. test_v0130_compile_speed::test_aot_warm_...) finds
    JAX writing to a deleted dir while it counts entries in the real dir, so its
    warm-hit/entry-count assertions break. Mirrors the fixture in
    test_v013_compile_perf2.py. Captured at setup (before any per-test monkeypatch
    of the cache env), so saved_dir is always the real persistent dir."""
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
# 1. Default-dir resolution + version-keyed shape
# --------------------------------------------------------------------------- #
def test_default_dir_is_version_keyed_under_user_cache(monkeypatch):
    """With no cache env vars, the default dir is
    <user-cache>/gpuwrf/jit/<version-tag>, NOT the flat <user-cache>/gpuwrf/jit."""
    for var in (
        "GPUWRF_JAX_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "GPUWRF_JAX_CACHE_DIR",
        "GPUWRF_CACHE",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    # Pin the backend tag so the test is deterministic regardless of CI host.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    d = cc.resolve_cache_dir()
    assert d is not None
    tag = cc.version_cache_tag()
    assert d.name == tag, f"default dir must end with the version tag, got {d}"
    # Parent is .../gpuwrf/jit
    assert d.parent.name == "jit"
    assert d.parent.parent.name == "gpuwrf"
    assert cc.CACHE_STATUS["source"] == "default"


def test_default_dir_honours_xdg_cache_home(monkeypatch, tmp_path):
    for var in (
        "GPUWRF_JAX_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "GPUWRF_JAX_CACHE_DIR",
        "GPUWRF_CACHE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    d = cc.resolve_cache_dir()
    assert str(d).startswith(str(tmp_path))
    assert d.parent == tmp_path / "gpuwrf" / "jit"


def test_version_tag_shape_and_components():
    """The tag is <gpuwrf>-jax<...>-jaxlib<...>-<backend>, a clean path component."""
    import gpuwrf
    import jax
    import jaxlib

    tag = cc.version_cache_tag()
    assert tag.startswith(gpuwrf.__version__ + "-")
    assert f"jax{jax.__version__}" in tag
    assert f"jaxlib{jaxlib.__version__}" in tag
    # No path separators or whitespace (must be a single dir component).
    assert "/" not in tag and "\\" not in tag and " " not in tag
    assert tag == os.path.basename(tag)


# --------------------------------------------------------------------------- #
# 2. Version-keying: different version / backend -> different dir
# --------------------------------------------------------------------------- #
def test_two_versions_resolve_to_different_dirs(monkeypatch, tmp_path):
    """The CORE B1 guarantee: two different gpuwrf versions resolve to DIFFERENT
    cache dirs, so a stale-version cache can never be hit by a new binary."""
    for var in (
        "GPUWRF_JAX_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "GPUWRF_JAX_CACHE_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GPUWRF_CACHE", str(tmp_path))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    import gpuwrf

    monkeypatch.setattr(gpuwrf, "__version__", "0.20.0", raising=False)
    d_v0 = cc.resolve_cache_dir()

    monkeypatch.setattr(gpuwrf, "__version__", "0.20.2", raising=False)
    d_v2 = cc.resolve_cache_dir()

    assert d_v0 != d_v2, "different versions MUST resolve to different cache dirs"
    assert "0.20.0" in str(d_v0)
    assert "0.20.2" in str(d_v2)


def test_cpu_and_cuda_backends_resolve_to_different_dirs(monkeypatch, tmp_path):
    """A CPU run and a CUDA run must not share a dir (different SASS/executables)."""
    for var in (
        "GPUWRF_JAX_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "GPUWRF_JAX_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GPUWRF_CACHE", str(tmp_path))

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    d_cpu = cc.resolve_cache_dir()

    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    d_gpu = cc.resolve_cache_dir()

    assert d_cpu != d_gpu, "cpu and cuda backends must resolve to different dirs"
    assert d_cpu.name.endswith("cpu")
    assert "cuda" in d_gpu.name


def test_gpuwrf_cache_root_is_version_keyed(monkeypatch, tmp_path):
    """GPUWRF_CACHE is a ROOT we own the layout of, so its jit subdir is
    version-keyed: $GPUWRF_CACHE/jit/<version-tag>."""
    for var in ("GPUWRF_JAX_CACHE", "JAX_COMPILATION_CACHE_DIR", "GPUWRF_JAX_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GPUWRF_CACHE", str(tmp_path))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    d = cc.resolve_cache_dir()
    assert d == tmp_path / "jit" / cc.version_cache_tag()
    assert cc.CACHE_STATUS["source"] == "env:GPUWRF_CACHE"


def test_explicit_dir_is_honoured_verbatim(monkeypatch, tmp_path):
    """An explicit JAX_COMPILATION_CACHE_DIR / GPUWRF_JAX_CACHE_DIR is honoured
    verbatim -- NO version tag is appended (the operator owns that path)."""
    monkeypatch.delenv("GPUWRF_CACHE", raising=False)
    explicit = tmp_path / "my-explicit-cache"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(explicit))
    monkeypatch.delenv("GPUWRF_JAX_CACHE_DIR", raising=False)
    d = cc.resolve_cache_dir()
    assert d == explicit  # verbatim, no tag appended
    assert cc.CACHE_STATUS["source"] == "env:JAX_COMPILATION_CACHE_DIR"


# --------------------------------------------------------------------------- #
# 3. Opt-out env vars honoured
# --------------------------------------------------------------------------- #
def test_gpuwrf_jax_cache_zero_disables(monkeypatch):
    monkeypatch.setenv("GPUWRF_JAX_CACHE", "0")
    assert cc.resolve_cache_dir() is None
    assert "disabled" in str(cc.CACHE_STATUS["source"])


@pytest.mark.parametrize("val", ["0", "false", "off", "no"])
def test_gpuwrf_jax_cache_falsey_values_disable(monkeypatch, val):
    monkeypatch.setenv("GPUWRF_JAX_CACHE", val)
    assert cc.resolve_cache_dir() is None


def test_configure_with_cache_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("GPUWRF_JAX_CACHE", "0")
    status = cc.configure_compilation_cache()
    assert status["enabled"] is False
    assert status["dir"] is None


def test_autotune_opt_out_zero_disables(monkeypatch):
    """GPUWRF_XLA_AUTOTUNE_CACHE=0 disables the autotune cache even when the
    compile-cache hook requests default_on=True."""
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "0")
    assert at.resolve_autotune_cache_dir(default_on=True) is None
    assert "disabled-by-GPUWRF_XLA_AUTOTUNE_CACHE" in str(at.AUTOTUNE_STATUS["source"])


# --------------------------------------------------------------------------- #
# 4. Fail-OPEN on an unwritable / bad cache dir
# --------------------------------------------------------------------------- #
def test_configure_fail_open_on_unwritable_dir(monkeypatch, tmp_path, restore_cache_globals):
    """If the cache dir can't be created (here: a parent that is a FILE, so mkdir
    raises OSError), configure_compilation_cache must log + record the error and
    CONTINUE WITHOUT a cache -- never crash the run."""
    a_file = tmp_path / "not-a-dir"
    a_file.write_text("x")  # a regular file; mkdir under it must fail
    bad = a_file / "jit"  # parent is a file -> OSError on mkdir(parents=True)
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(bad))
    # Avoid GPU autotune side effects on the CI box.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    status = cc.configure_compilation_cache()  # must NOT raise
    assert status["enabled"] is False
    assert status["error"] is not None
    assert "mkdir failed" in str(status["error"])


def test_configure_never_raises_even_on_weird_dir(monkeypatch, restore_cache_globals):
    """Belt-and-braces: a NUL byte in the resolved path makes mkdir raise a
    ValueError (not the OSError the code catches by name); configure must still
    return a status dict, never propagate. We inject the bad path via
    resolve_cache_dir (a NUL cannot be put in os.environ)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(cc, "resolve_cache_dir", lambda: Path("/dev/null/\x00bad/jit"))
    status = cc.configure_compilation_cache()  # must NOT raise
    assert status["enabled"] is False
    assert status["error"] is not None


def test_backend_tag_never_raises_without_nvidia_smi(monkeypatch):
    """_cuda_tag must fail-open to the coarse 'cuda' if nvidia-smi is missing."""
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert cc._cuda_tag() == "cuda"


# --------------------------------------------------------------------------- #
# 5. Explicit-dir version-mismatch WARNING
# --------------------------------------------------------------------------- #
def test_explicit_dir_mismatched_version_tag_warns(monkeypatch, tmp_path, caplog):
    """When the operator pins a dir whose last component carries a DIFFERENT
    version tag than the running gpuwrf, resolve_cache_dir logs a warning + records
    it in CACHE_STATUS['warning'] (but still honours the dir -- operator owns it)."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("GPUWRF_CACHE", raising=False)
    # A dir whose name looks like one of OUR tags but a stale version.
    stale = tmp_path / "0.0.1-jax0.10.0-jaxlib0.10.0-cuda_sm120"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(stale))
    cc.CACHE_STATUS["warning"] = None
    with caplog.at_level("WARNING", logger="gpuwrf.compile_cache"):
        d = cc.resolve_cache_dir()
    assert d == stale  # still honoured
    assert cc.CACHE_STATUS["warning"] is not None
    assert "0.0.1" in cc.CACHE_STATUS["warning"]
    # And it was actually logged at WARNING.
    assert any("guaranteed MISS" in r.getMessage() for r in caplog.records)


def test_explicit_dir_matching_version_tag_no_warning(monkeypatch, tmp_path):
    """A dir tagged with the RUNNING version raises no mismatch warning."""
    import gpuwrf

    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("GPUWRF_CACHE", raising=False)
    matched = tmp_path / f"{gpuwrf.__version__}-jax0.10.0-jaxlib0.10.0-cpu"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(matched))
    cc.CACHE_STATUS["warning"] = None
    cc.resolve_cache_dir()
    assert cc.CACHE_STATUS["warning"] is None


def test_explicit_dir_untagged_no_warning(monkeypatch, tmp_path):
    """An explicit dir whose name is NOT one of our tags (a plain path) gets no
    spurious mismatch warning."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("GPUWRF_CACHE", raising=False)
    plain = tmp_path / "just-my-cache"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(plain))
    cc.CACHE_STATUS["warning"] = None
    cc.resolve_cache_dir()
    assert cc.CACHE_STATUS["warning"] is None


# --------------------------------------------------------------------------- #
# 6. Autotune default-ON-with-compile-cache (the B1 flip)
# --------------------------------------------------------------------------- #
def test_autotune_default_on_when_compile_cache_on(monkeypatch, tmp_path):
    """resolve_autotune_cache_dir(default_on=True) returns a dir even with the env
    var UNSET (the new default), and records activation as default-on."""
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)
    monkeypatch.delenv("GPUWRF_CACHE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", raising=False)
    d = at.resolve_autotune_cache_dir(default_on=True)
    assert d is not None
    assert at.AUTOTUNE_STATUS["activation"] == "default-on-with-compile-cache"


def test_autotune_direct_call_default_off(monkeypatch):
    """A DIRECT call (default_on left False) keeps the historical OFF default, so
    callers bypassing the compile-cache hook are byte-unchanged unless they opt in."""
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)
    assert at.resolve_autotune_cache_dir() is None
    assert "not-opted-in" in str(at.AUTOTUNE_STATUS["source"])


def test_autotune_explicit_opt_in_marked_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", raising=False)
    monkeypatch.delenv("GPUWRF_CACHE", raising=False)
    d = at.resolve_autotune_cache_dir(default_on=True)
    assert d is not None
    assert at.AUTOTUNE_STATUS["activation"] == "explicit-opt-in"


def test_opt_in_tristate(monkeypatch):
    """_opt_in resolves the tri-state correctly."""
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "1")
    assert at._opt_in(default_on=False) is True
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE", "0")
    assert at._opt_in(default_on=True) is False  # explicit opt-out wins
    monkeypatch.delenv("GPUWRF_XLA_AUTOTUNE_CACHE", raising=False)
    assert at._opt_in(default_on=False) is False
    assert at._opt_in(default_on=True) is True


# --------------------------------------------------------------------------- #
# Status / report plumbing
# --------------------------------------------------------------------------- #
def test_cache_report_includes_version_tag():
    rep = cc.cache_report()
    assert "version_tag" in rep
    assert rep["version_tag"] == cc.version_cache_tag()


def test_configure_records_version_tag_and_warm_capable(monkeypatch, tmp_path, restore_cache_globals):
    """After a successful configure, CACHE_STATUS carries the version tag and a
    warm_capable boolean (False for a freshly-created empty dir)."""
    fresh = tmp_path / "fresh-jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(fresh))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    status = cc.configure_compilation_cache()
    assert status["enabled"] is True
    assert status["version_tag"] == cc.version_cache_tag()
    # A brand-new empty dir is NOT warm-capable.
    assert status["warm_capable"] is False


def test_cache_env_help_documents_version_tag():
    h = cc.cache_env_help()
    assert "version-tag" in h or "version-keyed" in h
    assert "GPUWRF_JAX_CACHE" in h


# --------------------------------------------------------------------------- #
# vNext cache-safety: cross-process write LOCK + atomic writer (parallel compile)
#
# When concurrent PROCESSES compile into the shared version-keyed cache, the
# unlocked default (max_size=-1 => no FileLock + a bare non-atomic write_bytes)
# lets the main process read a half-written sibling entry, which JAX decompresses
# with NO try/except -> a hard error instead of recompile. Engaging the lock
# sentinel + the atomic temp+rename writer fixes both hazards. These run on CPU.
# --------------------------------------------------------------------------- #
import multiprocessing  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402


def test_lock_sentinel_enables_filelock_and_records_status(monkeypatch, tmp_path):
    """Default-ON: configure flips max_size to the positive sentinel so JAX builds
    the per-dir FileLock; GPUWRF_JAX_CACHE_LOCK=0 restores the unlocked path."""
    d = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(d))
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("GPUWRF_JAX_CACHE_LOCK", raising=False)
    st = cc.configure_compilation_cache()
    assert st["locked"] is True
    assert st["max_size"] == cc._CACHE_LOCK_SENTINEL_BYTES
    assert st["atomic_writer"] is True

    monkeypatch.setenv("GPUWRF_JAX_CACHE_LOCK", "0")
    st2 = cc.configure_compilation_cache()
    assert st2["locked"] is False
    assert st2["max_size"] == -1


def test_lock_timeout_env_reaches_lru_cache(monkeypatch, tmp_path):
    """GPUWRF_JAX_CACHE_LOCK_TIMEOUT must configure JAX's FileLock timeout."""
    from jax._src.lru_cache import LRUCache

    d = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(d))
    monkeypatch.setenv("GPUWRF_JAX_CACHE_LOCK_TIMEOUT", "42.5")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    st = cc.configure_compilation_cache()
    assert st["locked"] is True
    cache = LRUCache(str(d), max_size=cc._CACHE_LOCK_SENTINEL_BYTES)
    assert cache.lock_timeout_secs == 42.5


def test_concurrent_writers_no_corruption(tmp_path):
    """N spawn procs each LRUCache.put a DISTINCT key under the lock sentinel;
    every entry decompresses intact and the .lockfile exists."""
    from tests import _parallel_compile_workers as W

    d = tmp_path / "jit"
    d.mkdir(parents=True, exist_ok=True)
    keys = [f"k{i}" for i in range(6)]
    ctx = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futs = [pool.submit(W.lru_put_distinct, (str(d), k, 4096 + i)) for i, k in enumerate(keys)]
        for f in as_completed(futs):
            results.append(f.result())
    assert all(r.get("error") is None for r in results), [r for r in results if r.get("error")]
    # Every distinct key landed as an intact, decompressible entry.
    for k in keys:
        entry = d / f"{k}-cache"
        assert entry.is_file(), f"missing entry for {k}"
        zlib.decompress(entry.read_bytes())  # raises if torn/truncated
    assert (d / ".lockfile").exists()


def test_locked_write_is_warm_hit_for_default_reader(tmp_path):
    """An entry written WITH eviction/lock on is a byte-identical entry the
    DEFAULT (unlocked) reader path reads back identically; filename == <key>-cache."""
    from jax._src.lru_cache import LRUCache

    d = tmp_path / "jit"
    d.mkdir(parents=True, exist_ok=True)
    cc._install_atomic_cache_writer()
    payload = zlib.compress(b"hello-executable" * 64, 6)

    locked = LRUCache(str(d), max_size=cc._CACHE_LOCK_SENTINEL_BYTES)
    locked.put("warm", payload)
    entry = d / "warm-cache"
    assert entry.is_file()
    assert entry.read_bytes() == payload  # byte-identical

    # The DEFAULT unlocked reader (max_size=-1, no lock) reads the SAME bytes.
    unlocked = LRUCache(str(d), max_size=-1)
    assert unlocked.get("warm") == payload


def test_atomic_writer_leaves_no_partial_on_crash(tmp_path):
    """Kill a child mid-write: only a stray .tmp may remain, never a truncated
    real <key>-cache. So the default reader never decompresses a torn entry."""
    from tests import _parallel_compile_workers as W

    d = tmp_path / "jit"
    d.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=W.atomic_write_then_crash, args=((str(d), "boom", 8192),))
    p.start()
    p.join()
    assert p.exitcode == 9  # the worker hard-crashed mid-write
    # The REAL entry must NOT exist (replace never happened); a stray .tmp is OK.
    assert not (d / "boom-cache").exists(), "crash left a truncated real entry"
    tmps = list(d.glob("boom-cache.*.tmp"))
    # Either a stray tmp remains (no replace) or nothing -- never a real entry.
    assert all(t.suffix == ".tmp" for t in tmps)
