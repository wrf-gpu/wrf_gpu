"""Slim retained HloModule for compiled-thunk GPU AOT blobs (aot_slim_module)."""
import re
import struct
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jaxlib._jax as _jax
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as keys
from gpuwrf.runtime import aot_executable as ax
from gpuwrf.runtime import aot_slim_module as asm

# HighwayHash reference test vectors (kExpected64): key = bytes 0..31, data[i] = i.
_HH_TEST_KEY = (0x0706050403020100, 0x0F0E0D0C0B0A0908, 0x1716151413121110, 0x1F1E1D1C1B1A1918)
_HH_EXPECTED = (0x907A56DE22C26E53, 0x7EAB43AAC7CDDD78, 0xB8D0569AB0B53D62, 0x5C6BEFAB8A463D80,
                0xF205A46893007EDA, 0x2B8A1668E4A94541, 0xBD4CCC325BEFCA6F, 0x4D02AE1738F59482)

# Written by XLA's Riegeli writer (a compiled-thunk jit__coupled_updates export, CP93):
# 24-byte block headers (hash of the 16-byte body) and 40-byte chunk headers (hash of 32).
_XLA_BLOCK_HEADERS = (
    "83af70d10d884a3f00000000000000004000000000000000",
    "e3ac37518570b602c0ff000000000000232c010000000000",
    "73ccb67f5d9cc883c0ff010000000000232c000000000000",
    "af099b7205608a77ddd30000000000008856020000000000",
    "2aa364b4e9662308ddd30100000000008856010000000000",
    "2523b8da56e89754ddd30200000000008856000000000000",
)
_XLA_CHUNK_HEADERS = (
    "9e5c01fcc92bbcb58b2b0200000000001d8fb7a808e066807202000000000000ab260b0000000000",
    "b4998ec610168927f52903000000000071c570159d14718c720b00000000000041990f0000000000",
    "75d78dd2fb99847cb6c2070000000000439cb3f2d7c02b9f72010000000000009764190000000000",
)


def test_highwayhash_reference_vectors_and_xla_riegeli_headers():
    for n, expected in enumerate(_HH_EXPECTED):
        assert asm.hh64(bytes(range(n)), _HH_TEST_KEY) == expected
    assert asm.hh64(ax._RIEGELI_SIGNATURE[32:64]) == struct.unpack("<Q", ax._RIEGELI_SIGNATURE[24:32])[0]
    for header in map(bytes.fromhex, _XLA_BLOCK_HEADERS):
        assert asm.hh64(header[8:]) == struct.unpack("<Q", header[:8])[0]
    for header in map(bytes.fromhex, _XLA_CHUNK_HEADERS):
        assert asm.hh64(header[8:40]) == struct.unpack("<Q", header[:8])[0]


def snappy_literal(data):
    """Valid snappy stream made of literals only (lengths up to 2**16)."""
    out = bytearray(ax._varint(len(data)))
    for start in range(0, len(data), 60000):
        part = data[start:start + 60000]
        out += bytes([61 << 2]) + (len(part) - 1).to_bytes(2, "little") + part
    return bytes(out)


def test_snappy_literals_and_copies():
    # "abc" + copy(offset 3, length 9) [1-byte offset] + copy(offset 12, length 5) [2-byte]
    stream = (ax._varint(17) + bytes([(3 - 1) << 2]) + b"abc" + bytes([((9 - 4) << 2) | 1, 3])
              + bytes([((5 - 1) << 2) | 2]) + (12).to_bytes(2, "little"))
    assert asm._snappy_decompress(stream) == b"abcabcabcabc" + b"abcab"
    data = bytes(range(256)) * 300
    assert asm._snappy_decompress(snappy_literal(data)) == data
    with pytest.raises(asm.SlimModuleError):
        asm._snappy_decompress(ax._varint(4) + bytes([((4 - 4) << 2) | 1, 9]))


def check_block_headers(riegeli):
    chunks = asm._chunks(riegeli)
    bounds = [(begin, asm._chunk_end(begin, info["data_size"], info["num_records"]))
              for begin, info, _ in chunks]
    for block in range(0, len(riegeli), 1 << 16):
        header = riegeli[block:block + 24]
        assert asm.hh64(header[8:]) == struct.unpack("<Q", header[:8])[0]
        prev, nxt = struct.unpack("<QQ", header[8:])
        begin, end = [(b, e) for b, e in bounds if b <= block < e or (block == b == 0)][0]
        if block:
            assert (prev, nxt) == (block - begin, end - block)
    return chunks


def test_riegeli_writer_round_trips_across_block_boundaries():
    records = [[bytes([7]) * 200_000], [b"x" * 65_000, b"", b"tail"], [b"y" * 131_000]]
    riegeli = ax._RIEGELI_SIGNATURE
    for recs in records:
        riegeli = asm._append_simple_chunk(riegeli, recs)
    chunks = check_block_headers(riegeli)
    assert len(chunks) == 4
    for (begin, info, data_pos), recs in zip(chunks[1:], records):
        data, _ = asm._read_logical(riegeli, data_pos, info["data_size"])
        assert asm.hh64(data) == info["data_hash"]
        assert asm._decode_simple_chunk(data, info["num_records"]) == recs


def test_snappy_compressed_simple_chunk_decodes():
    records = [b"alpha" * 1000, b"omega"]
    sizes = b"".join(ax._varint(len(r)) for r in records)
    values = b"".join(records)
    sizes_c = ax._varint(len(sizes)) + snappy_literal(sizes)
    values_c = ax._varint(len(values)) + snappy_literal(values)
    data = b"s" + ax._varint(len(sizes_c)) + sizes_c + values_c
    assert asm._decode_simple_chunk(data, 2) == records


def hlo_module_proto():
    def step(carry, flags, scale):
        def body(i, c):
            return c * scale + jnp.sin(c) + i.astype(c.dtype)
        out = jax.lax.fori_loop(0, 3, body, carry)
        return out, flags.sum(), out.max() > 0, jnp.int32(4) * flags

    carry = jnp.ones((4, 5), jnp.float32)
    flags = jnp.arange(3, dtype=jnp.int32)
    compiled = jax.jit(step, donate_argnums=0).lower(carry, flags, jnp.float32(0.5)).compile()
    return compiled.runtime_executable().hlo_modules()[0].as_serialized_hlo_module_proto()


def signature(text):
    """HLO module header (name, entry layout, input/output alias) minus the schedule flag."""
    return text.splitlines()[0].replace("is_scheduled=true, ", "")


def test_slim_hlo_module_keeps_entry_signature_alias_and_config_fields():
    module = hlo_module_proto()
    full = _jax.HloModule.from_serialized_hlo_module_proto(module).to_string()
    slim, stats = asm.slim_hlo_module(module)
    text = _jax.HloModule.from_serialized_hlo_module_proto(slim).to_string()
    assert stats["computations"] > 1 and stats["outputs"] == 4 and stats["parameters"] == 3
    assert sum(f == 3 for f, _, _ in ax._proto_fields(slim)) == 1
    assert not any(f == 7 for f, _, _ in ax._proto_fields(slim))
    assert signature(text) == signature(full)
    assert "is_scheduled=true" in full.splitlines()[0]
    assert len(re.findall(r"= \w+\[[0-9,]*\]\S* broadcast\(", text)) == 4
    assert re.search(r"ROOT \S+ = \(.*\) tuple\(", text)
    kept = [(f, v) for f, _, v in ax._proto_fields(module) if f not in (3, 7)]
    assert [(f, v) for f, _, v in ax._proto_fields(slim) if f != 3] == kept
    assert len(slim) < len(module)


def synthetic_blob(module, prefix_records=(b"kernels" * 9000,)):
    with_config = ax._pack(1, 2, module) + ax._pack(2, 2, b"module-config")
    rest = ax._pack(1, 2, with_config) + ax._pack(5, 2, b"program-shape")
    riegeli = asm._append_simple_chunk(ax._RIEGELI_SIGNATURE, list(prefix_records))
    riegeli = asm._append_simple_chunk(riegeli, [rest])
    head = bytes([2]) + b"hd"
    return head + ax._pack(1, 2, riegeli) + ax._pack(2, 2, b"compile-options"), riegeli


def test_slim_blob_rewrites_only_the_module():
    module = hlo_module_proto()
    blob, riegeli = synthetic_blob(module)
    out, stats = asm.slim_blob(blob)
    assert ax._gpu_blob_format(out) == ax.GPU_THUNK_FORMAT
    head = 3
    env = list(ax._proto_fields(out[head:]))
    assert out[:head] == blob[:head] and env[1] == (2, 2, b"compile-options")
    new = env[0][2]
    chunks = check_block_headers(new)
    old_last = asm._chunks(riegeli)[-1][0]
    assert new[:old_last] == riegeli[:old_last]  # kernels/thunk chunks copied verbatim
    begin, info, data_pos = chunks[-1]
    data, _ = asm._read_logical(new, data_pos, info["data_size"])
    assert asm.hh64(data) == info["data_hash"]
    (rest,) = asm._decode_simple_chunk(data, 1)
    rest_fields = list(ax._proto_fields(rest))
    assert rest_fields[1] == (5, 2, b"program-shape")
    with_config = list(ax._proto_fields(rest_fields[0][2]))
    assert with_config[1] == (2, 2, b"module-config")
    assert with_config[0][2] == asm.slim_hlo_module(module)[0]
    assert stats["riegeli_bytes"][1] < stats["riegeli_bytes"][0]


def test_slim_blob_refuses_unknown_layouts():
    with pytest.raises(asm.SlimModuleError):
        asm.slim_blob(bytes([2]) + b"hd" + ax._pack(1, 2, b"\x0a\x00legacy"))
    module = hlo_module_proto()
    blob, riegeli = synthetic_blob(module)
    corrupt = bytearray(blob)
    corrupt[blob.index(ax._RIEGELI_SIGNATURE) + 64] ^= 1  # stored hash of the first data chunk header
    with pytest.raises(asm.SlimModuleError, match="header hash"):
        asm.slim_blob(bytes(corrupt))


STATS = dict(argument_size_in_bytes=1, output_size_in_bytes=2, alias_size_in_bytes=0,
             temp_size_in_bytes=4)


def fake_dev(temp=4, fail=False):
    calls = []

    def deserialize(blob, devices, options):
        calls.append(blob)
        if fail:
            raise RuntimeError("load failed")
        return SimpleNamespace(get_compiled_memory_stats=lambda: SimpleNamespace(
            **dict(STATS, temp_size_in_bytes=temp), generated_code_size_in_bytes=9))

    return SimpleNamespace(platform="gpu", client=SimpleNamespace(deserialize_executable=deserialize)), calls


def test_slim_retained_module_reloads_and_fails_open(monkeypatch):
    monkeypatch.setattr(_jax, "DeviceList", lambda devices: devices)
    blob, _ = synthetic_blob(hlo_module_proto())
    dev, calls = fake_dev()
    out, status = ax._slim_retained_module(blob, dev, STATS)
    assert status["reason"] == "slimmed" and out == asm.slim_blob(blob)[0] and calls == [out]
    out, status = ax._slim_retained_module(blob, fake_dev(temp=5)[0], STATS)
    assert out == blob and status["reason"] == "buffer-sizes-changed"
    out, status = ax._slim_retained_module(blob, fake_dev(fail=True)[0], STATS)
    assert out == blob and status["reason"].startswith("error:RuntimeError")
    legacy = bytes([2]) + b"hd" + ax._pack(1, 2, b"\x0a\x00")
    out, status = ax._slim_retained_module(legacy, dev, STATS)
    assert out == legacy and status["reason"] == "not-compiled-thunks"


def fake_compiled(blob):
    tree = jax.tree_util.tree_structure(((), {}))
    return SimpleNamespace(in_tree=tree, out_tree=tree, _executable=SimpleNamespace(
        xla_executable=SimpleNamespace(serialize=lambda: blob)))


@pytest.mark.parametrize("env, slimmed", [({}, True), ({"GPUWRF_AOT_SLIM_MODULE": "0"}, False),
                                          ({"GPUWRF_AOT_KEEP_TRACE_METADATA": "1"}, False)])
def test_serialize_slims_compiled_thunk_exports(monkeypatch, env, slimmed):
    for name in ("GPUWRF_AOT_SLIM_MODULE", "GPUWRF_AOT_KEEP_TRACE_METADATA", "GPUWRF_AOT_COMPILED_THUNKS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    thunks, _ = synthetic_blob(hlo_module_proto())
    monkeypatch.setattr(ax, "_memory_stats", lambda executable: dict(STATS))
    monkeypatch.setattr(ax, "_convert_to_compiled_thunks",
                        lambda blob, dev, reference: (thunks, {"reason": "converted"}))
    calls = []

    def slim(blob, dev, reference):
        calls.append(blob)
        return b"slim:" + blob[:8], {"reason": "slimmed"}

    monkeypatch.setattr(ax, "_slim_retained_module", slim)
    status = {}
    blob, meta = ax.serialize(fake_compiled(b"legacy"), dev=SimpleNamespace(platform="gpu"),
                              hlo_sha256="h", status=status)
    assert (calls == [thunks]) is slimmed
    assert meta.retained_module == ("slim" if slimmed else "full")
    assert blob == (b"slim:" + thunks[:8] if slimmed else thunks)
    assert meta.blob_sha256 == ax.blob_sha256(blob)


def test_rich_request_refuses_slim_blob(monkeypatch):
    monkeypatch.setenv("GPUWRF_AOT_KEEP_TRACE_METADATA", "1")
    blob, _ = synthetic_blob(hlo_module_proto())
    meta = SimpleNamespace(executable_format=ax.GPU_THUNK_FORMAT, trace_metadata="rich",
                           retained_module="slim")
    with pytest.raises(RuntimeError, match="slimmed"):
        ax.load(blob, meta, SimpleNamespace(platform="gpu"))


def test_old_metas_unpickle_without_the_field():
    import pickle
    tree = jax.tree_util.tree_structure(((), {}))
    meta = ax.AotMeta(in_tree=tree, out_tree=tree, kept_var_idx=None, in_avals=(),
                      hlo_sha256=None, fingerprint={})
    state = dict(meta.__dict__)
    state.pop("retained_module")
    old = object.__new__(ax.AotMeta)
    object.__setattr__(old, "__dict__", state)
    assert pickle.loads(pickle.dumps(old)).retained_module is None


def test_slim_opt_out_never_changes_keys_and_module_is_outside_the_closure(monkeypatch):
    from pathlib import Path
    dev = SimpleNamespace(platform="gpu", client=None)
    monkeypatch.delenv("GPUWRF_AOT_SLIM_MODULE", raising=False)
    before = (keys.exec_env_hash(dev), keys.global_trace_env_hash(), ax.target_fingerprint(dev))
    monkeypatch.setenv("GPUWRF_AOT_SLIM_MODULE", "0")
    assert (keys.exec_env_hash(dev), keys.global_trace_env_hash(), ax.target_fingerprint(dev)) == before
    pkg_root = Path(asm.__file__).resolve().parents[1]
    rels = {f.relative_to(pkg_root).as_posix() for f in keys._trace_reachable_source_files(pkg_root)}
    assert "runtime/aot_slim_module.py" not in rels and "runtime/aot_executable.py" not in rels


@pytest.mark.parametrize("env, expected", [
    ({}, True),
    ({"GPUWRF_AOT_DIRECT_THUNKS": "0"}, False),
    ({"GPUWRF_AOT_SLIM_MODULE": "0"}, False),
    ({"GPUWRF_AOT_COMPILED_THUNKS": "0"}, False),
    ({"GPUWRF_AOT_KEEP_TRACE_METADATA": "1"}, False),
])
def test_export_compile_options_request_direct_thunks_only_with_slim(monkeypatch, env, expected):
    for name in ("GPUWRF_AOT_DIRECT_THUNKS", "GPUWRF_AOT_SLIM_MODULE", "GPUWRF_AOT_COMPILED_THUNKS",
                 "GPUWRF_AOT_KEEP_TRACE_METADATA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    options = ax.export_compile_options()
    assert options == ({"xla_gpu_experimental_aot_compiled_thunks": True} if expected else None)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert ax.export_compile_options() is None


def test_direct_thunks_opt_out_never_changes_keys(monkeypatch):
    dev = SimpleNamespace(platform="gpu", client=None)
    monkeypatch.delenv("GPUWRF_AOT_DIRECT_THUNKS", raising=False)
    before = (keys.exec_env_hash(dev), keys.global_trace_env_hash(), ax.target_fingerprint(dev))
    monkeypatch.setenv("GPUWRF_AOT_DIRECT_THUNKS", "0")
    assert (keys.exec_env_hash(dev), keys.global_trace_env_hash(), ax.target_fingerprint(dev)) == before


def test_domain_tree_compiles_every_exported_program_with_export_options():
    from pathlib import Path
    source = (Path(asm.__file__).resolve().parent / "domain_tree.py").read_text()
    calls = re.findall(r"lowered\.compile\(([^)]*)\)", source)
    assert len(calls) == 2 and all(c == "_aotx_export.export_compile_options(" for c in calls), calls
