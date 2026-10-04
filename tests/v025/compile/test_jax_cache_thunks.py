"""Triton-bearing JAX persistent-cache entries are written as compiled thunks (B12n)."""
from types import SimpleNamespace

import pytest
from jax._src import compilation_cache as jcc

from gpuwrf.runtime import aot_cheap_key as keys
from gpuwrf.runtime import aot_executable as ax
from gpuwrf.runtime import jax_cache_thunks as jct

LEGACY = b"\x00\x0a\x05\x0a\x03abc" + jct.TRITON_CUSTOM_CALL


class FakeExecutable:
    def __init__(self, blob):
        self.blob = blob

    def serialize(self):
        return self.blob

    def local_devices(self):
        return ["dev"]

    def get_compiled_memory_stats(self):
        return SimpleNamespace(argument_size_in_bytes=1, output_size_in_bytes=2,
                               alias_size_in_bytes=0, temp_size_in_bytes=3,
                               generated_code_size_in_bytes=4)


@pytest.fixture()
def recorder(monkeypatch):
    """Fresh hook around a recording put; JAX's real put is restored afterwards."""
    calls = []
    monkeypatch.setattr(jcc, "put_executable_and_time",
                        lambda key, name, exe, backend, t: calls.append((name, exe.serialize())))
    monkeypatch.setattr(jct, "STATUS", dict(jct.STATUS, converted=0, kept_legacy=0))
    monkeypatch.delenv("GPUWRF_JAX_CACHE_COMPILED_THUNKS", raising=False)
    assert jct.install()
    assert jct.install()  # idempotent: wrapped once
    return calls


def converter(monkeypatch, reason="converted"):
    seen = []

    def fake(blob, dev, reference):
        seen.append((dev, reference["temp_size_in_bytes"]))
        return (b"THUNKS" if reason == "converted" else blob), {"reason": reason, "seconds": 0.5}

    monkeypatch.setattr(ax, "_convert_to_compiled_thunks", fake)
    return seen


def put(name, blob, platform="gpu"):
    jcc.put_executable_and_time("k", name, FakeExecutable(blob), SimpleNamespace(platform=platform), 1)


def test_triton_gpu_entry_is_converted(recorder, monkeypatch):
    seen = converter(monkeypatch)
    put("jit__coupled_updates", LEGACY)
    assert recorder == [("jit__coupled_updates", b"THUNKS")]
    assert seen == [("dev", 3)]
    assert jct.STATUS["converted"] == 1


@pytest.mark.parametrize("name,blob,platform", [
    ("jit__advance_chunk_fori", LEGACY, "gpu"),   # AOT-managed: its own blob is converted
    ("jit_fused_jit", LEGACY, "gpu"),
    ("jit_add", LEGACY[:-len(jct.TRITON_CUSTOM_CALL)], "gpu"),  # no Triton kernels
    ("jit__coupled_updates", LEGACY, "cpu"),
])
def test_other_entries_keep_their_bytes(recorder, monkeypatch, name, blob, platform):
    seen = converter(monkeypatch)
    put(name, blob, platform)
    assert recorder == [(name, blob)] and seen == []


def test_failed_conversion_writes_the_legacy_entry(recorder, monkeypatch):
    converter(monkeypatch, reason="buffer-sizes-changed")
    put("jit__coupled_updates", LEGACY)
    assert recorder == [("jit__coupled_updates", LEGACY)]
    assert jct.STATUS["kept_legacy"] == 1

    def boom(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(ax, "_convert_to_compiled_thunks", boom)
    put("jit__coupled_updates", LEGACY)
    assert recorder[-1] == ("jit__coupled_updates", LEGACY)


def test_opt_out_leaves_jax_put_untouched(monkeypatch):
    original = lambda *args: None  # noqa: E731
    monkeypatch.setattr(jcc, "put_executable_and_time", original)
    monkeypatch.setenv("GPUWRF_JAX_CACHE_COMPILED_THUNKS", "0")
    assert jct.install() is False
    assert jcc.put_executable_and_time is original


def test_switch_never_changes_keys(monkeypatch):
    dev = SimpleNamespace(platform="gpu", client=None)
    monkeypatch.delenv("GPUWRF_JAX_CACHE_COMPILED_THUNKS", raising=False)
    before = (keys.exec_env_hash(dev), keys.global_trace_env_hash())
    monkeypatch.setenv("GPUWRF_JAX_CACHE_COMPILED_THUNKS", "0")
    assert (keys.exec_env_hash(dev), keys.global_trace_env_hash()) == before


def test_upgrade_rewrites_only_triton_legacy_entries(tmp_path, monkeypatch):
    calls = converter(monkeypatch)
    monkeypatch.setattr(ax, "_memory_stats", lambda exe: {"temp_size_in_bytes": 3})

    def entry(name, blob):
        path = tmp_path / f"{name}-0123abcd-cache"
        path.write_bytes(jcc.compress_executable(jcc.combine_executable_and_time(blob, 7)))
        return path

    triton = entry("jit__coupled_updates", LEGACY)
    managed = entry("jit__advance_chunk_fori", LEGACY)
    plain = entry("jit_add", LEGACY[:-len(jct.TRITON_CUSTOM_CALL)])
    before = {p: p.read_bytes() for p in (managed, plain)}
    dev = SimpleNamespace(client=SimpleNamespace(deserialize_executable=lambda *a: object()))
    report = jct.upgrade_cache_dir(tmp_path, device=dev)
    assert [e["entry"] for e in report["converted"]] == [triton.name] and report["skipped"] == 2
    assert jcc.extract_executable_and_time(jcc.decompress_executable(triton.read_bytes())) == (b"THUNKS", 7)
    assert {p: p.read_bytes() for p in (managed, plain)} == before
    assert calls == [(dev, 3)] and not list(tmp_path.glob("*.tmp"))


def test_module_stays_outside_the_trace_closure():
    """Editing the hook must never shift a cheap key (source fingerprint scope)."""
    from pathlib import Path

    pkg_root = Path(keys.__file__).resolve().parent.parent
    rels = {f.relative_to(pkg_root).as_posix() for f in keys._trace_reachable_source_files(pkg_root)}
    for module in ("runtime/jax_cache_thunks.py", "runtime/aot_executable.py", "runtime/aot_precompile.py"):
        assert module not in rels


def test_aot_machinery_installs_the_writer():
    import gpuwrf.runtime.aot_precompile  # noqa: F401  (installs at import)

    assert jct.STATUS["installed"] and getattr(jcc.put_executable_and_time, "_gpuwrf_thunks", False)
