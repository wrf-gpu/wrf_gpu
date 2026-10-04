"""Compiled-thunk export: format attestation, option rewrite and fail-open policy."""
from types import SimpleNamespace

import pytest

from gpuwrf.runtime import aot_executable as ax
from gpuwrf.runtime import aot_cheap_key as keys


def envelope(payload, header=b""):
    assert len(payload) < 128 and len(header) < 128
    return bytes([len(header)]) + header + b"\x0a" + bytes([len(payload)]) + payload


def legacy_blob(debug=b"", gpu_tail=b"\x12\x02ok"):
    config = ax._pack(1, 0, 7) + ax._pack(14, 2, debug) + ax._pack(3, 2, b"keep")
    hwc = ax._pack(1, 2, b"hlo-module") + ax._pack(2, 2, config)
    gpu = ax._pack(1, 2, hwc) + gpu_tail
    return bytes([2]) + b"hd" + ax._pack(1, 2, gpu) + ax._pack(2, 2, b"compile-options")


def debug_options(blob):
    pos = 3
    gpu = list(ax._proto_fields(blob[pos:]))[0][2]
    hwc = list(ax._proto_fields(gpu))[0][2]
    cfg = list(ax._proto_fields(hwc))[1][2]
    return dict((f, v) for f, w, v in ax._proto_fields(list(ax._proto_fields(cfg))[1][2]))


def test_formats_and_invalid_envelope():
    assert ax._gpu_blob_format(envelope(ax._RIEGELI_SIGNATURE)) == ax.GPU_THUNK_FORMAT
    assert ax._gpu_blob_format(envelope(b"\x0a\x00")) == ax.GPU_LEGACY_FORMAT
    assert ax._gpu_blob_format(b"\x80") is None
    assert ax._gpu_blob_format(b"\x00\x0a\x7fshort") is None


def test_option_rewrite_sets_only_debug_435():
    blob = legacy_blob(debug=ax._pack(5, 0, 1) + ax._pack(435, 0, 0))
    out = ax._with_compiled_thunks_option(blob)
    assert debug_options(out) == {5: 1, 435: 1}
    assert out[:3] == blob[:3] and out.endswith(ax._pack(2, 2, b"compile-options"))
    assert ax._with_compiled_thunks_option(out) == out
    with pytest.raises(ValueError):
        ax._with_compiled_thunks_option(b"\x00" + ax._pack(2, 2, b"no-gpu-result"))


def fake_dev(exported, stats=4):
    loaded = SimpleNamespace(
        serialize=lambda: exported,
        get_compiled_memory_stats=lambda: SimpleNamespace(
            argument_size_in_bytes=1, output_size_in_bytes=2, alias_size_in_bytes=0,
            temp_size_in_bytes=stats, generated_code_size_in_bytes=9))
    calls = []

    def deserialize(blob, devices, options):
        calls.append(blob)
        return loaded

    return SimpleNamespace(platform="gpu", client=SimpleNamespace(deserialize_executable=deserialize)), calls


def test_conversion_reexports_patched_blob_and_checks_buffer_sizes(monkeypatch):
    import jaxlib._jax as _jax
    monkeypatch.setattr(_jax, "DeviceList", lambda devices: devices)
    thunks = envelope(ax._RIEGELI_SIGNATURE)
    blob = legacy_blob()
    dev, calls = fake_dev(thunks)
    reference = dict(argument_size_in_bytes=1, output_size_in_bytes=2, alias_size_in_bytes=0,
                     temp_size_in_bytes=4)
    out, status = ax._convert_to_compiled_thunks(blob, dev, reference)
    assert out == thunks and status["reason"] == "converted"
    assert debug_options(calls[0]) == {435: 1} and calls[1] == thunks
    out, status = ax._convert_to_compiled_thunks(blob, dev, dict(reference, temp_size_in_bytes=5))
    assert out == blob and status["reason"] == "buffer-sizes-changed"


def test_conversion_fails_open(monkeypatch):
    import jaxlib._jax as _jax
    monkeypatch.setattr(_jax, "DeviceList", lambda devices: devices)
    blob = legacy_blob()
    dev, _ = fake_dev(b"\x00\x0a\x02\x0a\x00")  # still legacy after export
    assert ax._convert_to_compiled_thunks(blob, dev)[1]["reason"] == "export-not-compiled-thunks"
    broken = SimpleNamespace(platform="gpu", client=SimpleNamespace(
        deserialize_executable=lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))))
    out, status = ax._convert_to_compiled_thunks(blob, broken)
    assert out == blob and status["reason"].startswith("error:RuntimeError")
    thunks = envelope(ax._RIEGELI_SIGNATURE)
    assert ax._convert_to_compiled_thunks(thunks, broken)[1]["reason"] == "not-legacy-format"


def test_unknown_version_refused_before_deserialization():
    client = SimpleNamespace(deserialize_executable=lambda *args: pytest.fail("loaded wrong format"))
    dev = SimpleNamespace(platform="gpu", client=client)
    meta = SimpleNamespace(executable_format="xla-gpu-thunks-riegeli-v2")
    with pytest.raises(RuntimeError, match="format/version mismatch"):
        ax.load(envelope(ax._RIEGELI_SIGNATURE), meta, dev)


def test_export_format_opt_out_never_changes_keys(monkeypatch):
    dev = SimpleNamespace(platform="gpu", client=None)
    monkeypatch.delenv("GPUWRF_AOT_COMPILED_THUNKS", raising=False)
    before = (keys.exec_env_hash(dev), keys.global_trace_env_hash(), ax.target_fingerprint(dev))
    monkeypatch.setenv("GPUWRF_AOT_COMPILED_THUNKS", "0")
    assert (keys.exec_env_hash(dev), keys.global_trace_env_hash(), ax.target_fingerprint(dev)) == before
