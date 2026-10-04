"""Wire-format safety gates; actual GPU carry gate is recorded separately."""
from gpuwrf.runtime.aot_executable import _compact_gpu_trace_metadata, _proto_fields


def vi(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def f(number, value):
    return vi(number << 3 | 2) + vi(len(value)) + value


def fields(data):
    return {number: value for number, wire, value in _proto_fields(data)}


def fixture():
    metadata = f(1, b"mul") + f(2, b"physics/opaque/source") + f(3, b"oracle.F") + vi(4 << 3) + vi(123) + vi(15 << 3) + vi(4) + f(10, b"tuned-profile") + f(16, b"scheduled-name")
    instruction = f(1, b"kernel-name") + f(2, b"multiply") + f(3, b"layout") + f(7, metadata) + f(8, b"IEEE-literal") + f(35, b"opaque-backend-config")
    computation = f(1, b"entry") + f(2, instruction) + f(4, b"program-shape")
    module = f(1, b"jit-real") + f(3, computation) + f(17, b"source-stack-index")
    with_config = f(1, module) + f(2, b"module-config")
    gpu = f(1, with_config) + f(2, b"buffer-assignment") + f(3, b"original-PTX") + f(4, b"\x7fELF-original-native-object")
    envelope = f(1, gpu) + f(2, b"full-compile-options")
    return vi(4) + b"IFRT" + envelope


def test_compaction_preserves_every_computational_field_and_native_object():
    original = fixture()
    compact, status = _compact_gpu_trace_metadata(original)
    assert status["reason"] == "source-trace-labels"
    assert compact[:5] == original[:5]
    before = fields(original[5:]); after = fields(compact[5:])
    assert before[2] == after[2]
    bgpu, agpu = fields(before[1]), fields(after[1])
    assert {k: v for k, v in bgpu.items() if k != 1} == {k: v for k, v in agpu.items() if k != 1}
    bcfg, acfg = fields(bgpu[1]), fields(agpu[1])
    assert bcfg[2] == acfg[2]
    bmodule, amodule = fields(bcfg[1]), fields(acfg[1])
    assert 17 in bmodule and 17 not in amodule
    bcomp, acomp = fields(bmodule[3]), fields(amodule[3])
    assert bcomp[1] == acomp[1] and bcomp[4] == acomp[4]
    binst, ainst = fields(bcomp[2]), fields(acomp[2])
    assert {k:v for k,v in binst.items() if k!=7} == {k:v for k,v in ainst.items() if k!=7}
    assert fields(ainst[7]) == {10:b"tuned-profile",16:b"scheduled-name"}


def test_compaction_is_idempotent():
    once, _ = _compact_gpu_trace_metadata(fixture())
    twice, status = _compact_gpu_trace_metadata(once)
    assert twice == once and status["removed_fields"] == 0


def test_unknown_format_is_returned_byte_for_byte():
    for original in [b"", b"bad", vi(4)+b"IFRT"+f(3,b"new-format"), fixture()+f(99,b"future-field")]:
        compact, status = _compact_gpu_trace_metadata(original)
        assert compact == original
        assert status["reason"] != "source-trace-labels"


def test_unverified_jaxlib_version_preserves_original_blob(monkeypatch):
    import jaxlib
    monkeypatch.setattr(jaxlib, "__version__", "future")
    original = fixture()
    compact, status = _compact_gpu_trace_metadata(original)
    assert compact == original
    assert status["reason"] == "unsupported-jaxlib-version"


def fake_compiled():
    from types import SimpleNamespace
    import jax
    tree = jax.tree_util.tree_structure(((), {}))
    return SimpleNamespace(in_tree=tree, out_tree=tree,
        _executable=SimpleNamespace(xla_executable=SimpleNamespace(serialize=fixture)))


def test_gpu_export_compacts_and_hashes_the_published_bytes(monkeypatch):
    from types import SimpleNamespace
    from gpuwrf.runtime import aot_executable as ax
    monkeypatch.delenv("GPUWRF_AOT_KEEP_TRACE_METADATA", raising=False)
    status = {}
    blob, meta = ax.serialize(fake_compiled(), dev=SimpleNamespace(platform="gpu"), hlo_sha256="source-hlo", status=status)
    assert blob != fixture() and meta.trace_metadata == "compact"
    assert meta.blob_sha256 == ax.blob_sha256(blob)
    assert meta.hlo_sha256 == "source-hlo"
    assert status["aot_trace_metadata"] == "compact"
    assert status["aot_trace_compaction_seconds"] >= 0


def test_profiler_rich_opt_out_preserves_original_blob(monkeypatch):
    from types import SimpleNamespace
    from gpuwrf.runtime import aot_executable as ax
    monkeypatch.setenv("GPUWRF_AOT_KEEP_TRACE_METADATA", "1")
    blob, meta = ax.serialize(fake_compiled(), dev=SimpleNamespace(platform="gpu"), hlo_sha256="source-hlo")
    assert blob == fixture() and meta.trace_metadata == "rich"


def test_rich_request_refuses_compacted_cached_blob(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from gpuwrf.runtime import aot_executable as ax
    monkeypatch.delenv("GPUWRF_AOT_KEEP_TRACE_METADATA", raising=False)
    blob, meta = ax.serialize(fake_compiled(), dev=SimpleNamespace(platform="gpu"), hlo_sha256="source-hlo")
    monkeypatch.setenv("GPUWRF_AOT_KEEP_TRACE_METADATA", "1")
    with pytest.raises(RuntimeError, match="profiler-rich"):
        ax.load(blob, meta, SimpleNamespace(platform="gpu"))
