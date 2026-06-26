"""vNext B2 cache/compile-efficiency tests (CPU-only, numerically inert).

Covers the three B2 deliverables, all of which are COMPILE/CACHE config only --
the compiled numerics are bit-identical regardless of any knob here:

1. Prewarmed cache as a release artifact: the version-keyed (B1) pack/unpack
   tooling + the `aot_precompile` CLI (info/pack/unpack), including the tag-
   mismatch refusal that prevents the silent paid-cold-compile B1 fixed.
2. Parallel XLA compile default-safe: `configure_parallel_compile(default_on=True)`
   now activates N-way parallel compile when the compile cache is on, while a
   DIRECT call stays default-OFF and the `=0` opt-out always wins; the single
   `--xla_gpu_*` flag is still subprocess-probed (fail-open) so an unknown flag
   can never abort.
3. De-fuse opt-in knob: `GPUWRF_NESTED_DEFUSE_COMPILE=1` selects the eager
   per-domain compile path (~K x lower peak compile-RAM) while the fused path
   stays the DEFAULT; numerically identical (reuses the existing eager path).

GPU-gated steps (actual GPU compile / real prewarm of PRODUCTION_GRIDS) are NOT
run here -- they are listed for the manager to run under the GPU lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from gpuwrf.runtime import aot_precompile as aot
from gpuwrf.runtime import compile_cache as cc
from gpuwrf.runtime import domain_tree as dt
from gpuwrf.runtime import xla_autotune as at


# --------------------------------------------------------------------------- #
# Deliverable 1: prewarmed cache as a version-keyed release artifact
# --------------------------------------------------------------------------- #
def _make_fake_cache(cache_dir: Path, n: int = 3) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (cache_dir / f"entry{i}-cache").write_text(f"exe{i}")


def test_cache_artifact_info_is_version_keyed(monkeypatch, tmp_path):
    """`info` reports the B1 version-keyed dir + tag + a tag-named artifact, with
    no GPU / no compile (works on a fresh box)."""
    cache_dir = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(cache_dir))
    _make_fake_cache(cache_dir, 2)
    info = aot.cache_artifact_info()
    assert info["version_tag"] == cc.version_cache_tag()
    assert info["entry_count"] == 2
    assert info["warm_capable"] is True
    assert info["suggested_artifact"] == f"gpuwrf-jitcache-{cc.version_cache_tag()}.tar.gz"


def test_pack_then_unpack_roundtrips_entries(monkeypatch, tmp_path):
    """pack tars the warmed cache under one tagged top-level dir; unpack restores
    the entries into the version-keyed dest. The artifact is the shippable thing a
    fresh machine warm-starts from instead of cold-compiling."""
    src = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(src))
    _make_fake_cache(src, 3)

    art = aot.pack_cache(out_path=str(tmp_path / "artifact.tar.gz"))
    assert art.is_file()
    with tarfile.open(art) as t:
        names = [m.name for m in t.getmembers()]
    top = f"gpuwrf-jitcache-{cc.version_cache_tag()}"
    assert all(n == top or n.startswith(top + "/") for n in names)

    dest = tmp_path / "fresh" / cc.version_cache_tag()
    out = aot.unpack_cache(str(art), cache_dir=str(dest))
    assert Path(out) == dest  # entries land strictly IN dest, not a sibling
    got = sorted(p.name for p in Path(out).iterdir())
    assert got == ["entry0-cache", "entry1-cache", "entry2-cache"]
    # Contents must match what was packed (byte-for-byte), not just names.
    for i in range(3):
        assert (dest / f"entry{i}-cache").read_text() == f"exe{i}"
    # The sibling of dest must NOT have received a stray extracted dir.
    siblings = [p.name for p in dest.parent.iterdir()]
    assert siblings == [dest.name]


def test_pack_refuses_empty_cache(monkeypatch, tmp_path):
    """Nothing to ship if the cache was never warmed -> clear FileNotFoundError."""
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        aot.pack_cache(out_path=str(tmp_path / "x.tar.gz"))


def _make_tagged_artifact(tmp_path: Path, tag: str, entries: dict[str, str]) -> Path:
    """Build a valid one-tagged-top-level-dir artifact for ``tag``."""
    stub = tmp_path / f"stub-{tag}"
    stub.mkdir(parents=True, exist_ok=True)
    for name, content in entries.items():
        (stub / name).write_text(content)
    art = tmp_path / f"art-{tag}.tar.gz"
    with tarfile.open(art, "w:gz") as t:
        t.add(str(stub), arcname=f"gpuwrf-jitcache-{tag}")
    return art


def test_unpack_refuses_version_mismatch_then_force_lands_in_dest(monkeypatch, tmp_path):
    """A mismatched-tag artifact must be REFUSED (it would be a guaranteed cache
    miss -- the exact paid-cold-compile B1 prevents). With --force, the contents
    must land STRICTLY IN dest (non-empty, correct contents), not the running-tag
    path or a sibling (the unpack_cache placement bug)."""
    bad = _make_tagged_artifact(
        tmp_path, "9.9.9-jaxZ-jaxlibZ-cpu", {"x-cache": "exeX", "y-cache": "exeY"}
    )
    dest = tmp_path / "dest"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(dest))

    with pytest.raises(ValueError, match="GUARANTEED cache MISS"):
        aot.unpack_cache(str(bad), cache_dir=str(dest))

    out = aot.unpack_cache(str(bad), cache_dir=str(dest), force=True)
    assert Path(out) == dest
    got = sorted(p.name for p in dest.iterdir())
    assert got == ["x-cache", "y-cache"], "force-mismatch must place entries IN dest"
    assert (dest / "x-cache").read_text() == "exeX"
    assert (dest / "y-cache").read_text() == "exeY"
    # No stray gpuwrf-jitcache-* dir was created beside dest (the placement bug:
    # contents landing in a sibling/running-tag path instead of dest).
    stray = [
        p.name for p in dest.parent.iterdir()
        if p.is_dir() and p.name.startswith("gpuwrf-jitcache-")
    ]
    assert stray == [], f"entries leaked to a sibling dir: {stray}"


def test_unpack_rejects_no_tag_artifact(monkeypatch, tmp_path):
    """An artifact with NO gpuwrf-jitcache-<tag> top-level dir must FAIL CLOSED
    (not silently leave dest empty while extracting elsewhere)."""
    stub = tmp_path / "loose"
    stub.mkdir()
    (stub / "x-cache").write_text("e")
    bad = tmp_path / "notag.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        t.add(str(stub), arcname="random-top")  # not a gpuwrf-jitcache- dir
    dest = tmp_path / "dest"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(dest))
    with pytest.raises(ValueError, match="exactly ONE"):
        aot.unpack_cache(str(bad), cache_dir=str(dest), force=True)
    # dest must NOT have been created-and-populated by a failed unpack.
    assert not dest.exists() or not any(dest.iterdir())


def test_unpack_rejects_multiple_tagged_dirs(monkeypatch, tmp_path):
    """>1 tagged top-level dir is ambiguous -> fail closed."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "x-cache").write_text("e")
    b = tmp_path / "b"
    b.mkdir()
    (b / "y-cache").write_text("e")
    bad = tmp_path / "two.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        t.add(str(a), arcname="gpuwrf-jitcache-1.0.0-cpu")
        t.add(str(b), arcname="gpuwrf-jitcache-2.0.0-cpu")
    dest = tmp_path / "dest"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(dest))
    with pytest.raises(ValueError, match="exactly ONE"):
        aot.unpack_cache(str(bad), cache_dir=str(dest), force=True)


def test_unpack_is_idempotent(monkeypatch, tmp_path):
    """Re-unpacking the same artifact must overwrite cleanly, not raise."""
    tag = cc.version_cache_tag()
    art = _make_tagged_artifact(tmp_path, tag, {"e0-cache": "v0", "e1-cache": "v1"})
    dest = tmp_path / "dest"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(dest))
    aot.unpack_cache(str(art), cache_dir=str(dest))
    aot.unpack_cache(str(art), cache_dir=str(dest))  # second time must not raise
    assert sorted(p.name for p in dest.iterdir()) == ["e0-cache", "e1-cache"]
    assert (dest / "e0-cache").read_text() == "v0"


def test_unpack_rejects_path_traversal(monkeypatch, tmp_path):
    """A valid one-tagged-dir artifact that ALSO smuggles a ../ member must be
    rejected by the path-traversal guard (entries cannot escape the staging dir)."""
    tag = cc.version_cache_tag()
    payload = tmp_path / "ok-payload"
    payload.mkdir()
    (payload / "ok").write_text("ok")
    bad = tmp_path / "evil.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        # a legitimate tagged top-level dir (passes the "exactly ONE" gate) ...
        t.add(str(payload / "ok"), arcname=f"gpuwrf-jitcache-{tag}/ok-cache")
        # ... plus a smuggled escaping member that the traversal guard must catch.
        t.add(str(payload / "ok"), arcname=f"gpuwrf-jitcache-{tag}/../../escape")
    dest = tmp_path / "deep" / "dest"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(dest))
    with pytest.raises(ValueError, match="unsafe path"):
        aot.unpack_cache(str(bad), cache_dir=str(dest), force=True)


def test_cli_info_emits_json(monkeypatch, tmp_path):
    """`python -m gpuwrf.runtime.aot_precompile info` is a fresh-box, no-compile
    JSON report of the version-keyed cache."""
    cache_dir = tmp_path / "jit"
    _make_fake_cache(cache_dir, 1)
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["GPUWRF_JAX_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "gpuwrf.runtime.aot_precompile", "info"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["cache_dir"] == str(cache_dir)
    assert payload["entry_count"] == 1


def test_cli_warm_requires_spec_provider(tmp_path):
    """`warm` without --spec-provider exits 2 with a clear message (no traceback)."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "gpuwrf.runtime.aot_precompile", "warm"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2
    assert "spec-provider" in proc.stderr


def test_load_spec_provider_validates_path():
    with pytest.raises(ValueError):
        aot.load_spec_provider("no_colon_here")


# --------------------------------------------------------------------------- #
# Deliverable 2: parallel XLA compile default-safe (default-on with compile cache)
# --------------------------------------------------------------------------- #
def test_resolve_parallel_default_on_picks_capped_cpu_count(monkeypatch):
    """With both vars UNSET and default_on=True (the import-hook path), parallel
    compile resolves to min(cpu_count, 8) -- the cold-compile-wall lever."""
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    at._reset_parallel_status()
    n = at.resolve_parallel_compile(default_on=True)
    assert n == min(os.cpu_count() or 1, 8)
    assert at.PARALLEL_COMPILE_STATUS["activation"] == "default-on-with-compile-cache"


def test_resolve_parallel_direct_call_stays_off(monkeypatch):
    """A DIRECT call (default_on=False) keeps the historical default-OFF, so any
    caller bypassing the compile-cache hook is byte-unchanged."""
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    at._reset_parallel_status()
    assert at.resolve_parallel_compile(default_on=False) is None
    assert at.resolve_parallel_compile() is None  # no-arg == default_on=False


def test_parallel_opt_out_wins_over_default_on(monkeypatch):
    """GPUWRF_XLA_PARALLEL_COMPILE=0 must disable parallel compile even when the
    import hook passes default_on=True."""
    monkeypatch.setenv("GPUWRF_XLA_PARALLEL_COMPILE", "0")
    at._reset_parallel_status()
    assert at.resolve_parallel_compile(default_on=True) is None


def test_parallel_default_on_respects_cpu_pin(monkeypatch):
    """default_on=True must NOT inject --xla_gpu_* on a cpu pin (a CPU jaxlib can
    fatally abort on an unknown flag)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_parallel_compile(default_on=True)
    assert status["enabled"] is False
    assert "cpu" in str(status["reason"])
    assert os.environ.get("XLA_FLAGS", "") == before


def test_parallel_default_on_probes_and_injects_on_gpu(monkeypatch):
    """default_on=True on a GPU target injects the single parallelism flag ONLY
    after the subprocess probe accepts it (fail-open: probe is the safety gate)."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # simulate a GPU target
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(
        at, "probe_flag_supported",
        lambda flag, timeout_s=at._PROBE_TIMEOUT_S: (True, "accepted"),
    )
    status = at.configure_parallel_compile(default_on=True)
    assert status["enabled"] is True
    n = min(os.cpu_count() or 1, 8)
    assert status["injected_flags"] == [f"--xla_gpu_force_compilation_parallelism={n}"]
    assert f"--xla_gpu_force_compilation_parallelism={n}" in os.environ.get("XLA_FLAGS", "")


def test_parallel_default_on_drops_flag_when_probe_rejects(monkeypatch):
    """THE fail-open guard: when the bundled jaxlib rejects the flag, default-on
    must DROP it (record in rejected_flags, NOT inject) and never abort."""
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_PARALLEL_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_XLA_COMPILE_PARALLELISM", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(
        at, "probe_flag_supported",
        lambda flag, timeout_s=at._PROBE_TIMEOUT_S: (False, "rejected:unknown flag"),
    )
    before = os.environ.get("XLA_FLAGS", "")
    status = at.configure_parallel_compile(default_on=True)
    assert status["enabled"] is False
    assert status["injected_flags"] in (None, [])
    assert status["rejected_flags"], "rejected flag must be recorded"
    assert os.environ.get("XLA_FLAGS", "") == before


def test_parallel_env_help_reflects_default_on():
    h = at.parallel_compile_env_help()
    assert "ON by default" in h
    assert "GPUWRF_XLA_PARALLEL_COMPILE=0" in h


# --------------------------------------------------------------------------- #
# Deliverable 3: de-fuse knob (compile-RAM lever; fused default / de-fuse opt-in)
# --------------------------------------------------------------------------- #
def test_default_is_fused(monkeypatch):
    """The UNSET default is fused; de-fuse remains an explicit low-RAM opt-in."""
    monkeypatch.delenv("GPUWRF_NESTED_DEFUSE_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    rep = dt.nested_defuse_report()
    assert rep["fused"] is True
    assert rep["defused"] is False
    assert rep["source"] == "default-fused"


def test_fuse_force_keeps_fused(monkeypatch):
    """Explicit GPUWRF_NESTED_FUSE=1 keeps the fused cascade."""
    monkeypatch.delenv("GPUWRF_NESTED_DEFUSE_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    monkeypatch.setenv("GPUWRF_NESTED_FUSE", "1")
    rep = dt.nested_defuse_report()
    assert rep["fused"] is True
    assert rep["defused"] is False
    assert rep["source"] == "env:GPUWRF_NESTED_FUSE=1"


@pytest.mark.parametrize("val", ["1", "true", "on", "yes"])
def test_defuse_knob_selects_eager(monkeypatch, val):
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", val)
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    rep = dt.nested_defuse_report()
    assert rep["fused"] is False
    assert rep["defused"] is True
    assert rep["source"] == "env:GPUWRF_NESTED_DEFUSE_COMPILE"


@pytest.mark.parametrize("val", ["0", "false", "off", "no", ""])
def test_defuse_falsey_falls_through_to_default_fused(monkeypatch, val):
    """A FALSEY GPUWRF_NESTED_DEFUSE_COMPILE does not force de-fuse."""
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", val)
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    rep = dt.nested_defuse_report()
    assert rep["fused"] is True
    assert rep["defused"] is False
    assert rep["source"] == "default-fused"


@pytest.mark.parametrize("val", ["0", "false", "off", "no", ""])
def test_fuse_falsey_defuses(monkeypatch, val):
    """GPUWRF_NESTED_FUSE falsey still de-fuses (identity-proof opt-out)."""
    monkeypatch.delenv("GPUWRF_NESTED_DEFUSE_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    monkeypatch.setenv("GPUWRF_NESTED_FUSE", val)
    rep = dt.nested_defuse_report()
    # empty string is FALSEY for GPUWRF_NESTED_FUSE -> source env label
    assert rep["fused"] is False
    assert rep["defused"] is True
    assert rep["source"] == "env:GPUWRF_NESTED_FUSE=0"


def test_defuse_takes_precedence_and_is_recorded(monkeypatch):
    """The compile-RAM de-fuse knob is honoured FIRST and recorded distinctly from
    the identity-proof vars, so logs/A-B attribute the eager path to the right
    cause."""
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", "1")
    monkeypatch.setenv("GPUWRF_NESTED_FUSE", "1")  # would otherwise force fused
    assert dt._nested_fuse_default_enabled() is False
    assert dt.NESTED_DEFUSE_STATUS["source"] == "env:GPUWRF_NESTED_DEFUSE_COMPILE"


def test_defuse_env_help_mentions_compile_ram():
    h = dt.nested_defuse_env_help()
    assert "GPUWRF_NESTED_DEFUSE_COMPILE" in h
    assert "compile-RAM" in h or "compile-RAM".replace("-", " ") in h or "peak compile-RAM" in h
