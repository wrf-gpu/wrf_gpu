"""Slim the retained HloModule of a compiled-thunk GPU AOT blob.

A compiled-thunk executable (XLA ``GpuExecutableProto``) runs from its own thunk
sequence, kernels, buffer allocations, output info and program shape. It still
carries the whole optimized ``HloModule`` (field 1, ``hlo_module_with_config``):
about 125-140 MB and 1,500-3,700 computations per PROD domain. Pinned XLA
(b6f37ab) cannot drop it: ``Executable::module()`` CHECKs that it is non-null, and
PJRT reads ``result_shape()`` on load and ``input_output_alias_config()`` on every
execute. It does pay for the module body, though:

* ``GpuExecutable::ExecuteAsyncOnStreamImpl`` calls
  ``ShouldCollectiveUseMinimalResource(module())`` on every execute. That walks
  ``MakeNonfusionComputations`` (dispatch D10: about 2 ms of host time per root);
* each load parses the module (``HloModule::CreateFromProtoWithConfig``) and
  builds profiler ``ModuleAnnotations`` from it.

:func:`slim_blob` keeps every ``HloModuleProto`` field except ``computations``
and ``schedule`` byte for byte: name, ids, host program shape,
input/output alias, buffer donor, shardings, stack frames and frontend
attributes. It also keeps the ``HloModuleConfig`` byte for byte. The
computations become ONE entry computation with the original name, id and
program shape: the original parameter instructions plus, per output leaf,
``constant(s32 0) -> convert -> broadcast`` to the original output shape and
layout, under the original root shape.

The blob is an IFRT envelope around a Riegeli split-proto file (64 KiB blocks,
HighwayHash-64 keyed checksums). The rest record carrying the module is the
last chunk. Earlier chunks (kernels, thunks, constants) are copied verbatim;
only the 40-byte chunk headers are verified here. The device reload in
``aot_executable._slim_retained_module`` is the end-to-end check, and XLA
verifies every chunk hash on load. Pure Python, no JAX import: the trace
closure is unchanged.
"""

from __future__ import annotations

import struct
from typing import Any

from gpuwrf.runtime.aot_executable import _RIEGELI_SIGNATURE, _pack, _proto_fields, _varint

__all__ = ["SlimModuleError", "slim_blob", "slim_hlo_module", "hh64"]


class SlimModuleError(ValueError):
    """The blob or module layout is not one this rewriter understands."""


# ---------------------------------------------------------------------------
# HighwayHash-64 (portable reference algorithm), keyed as Riegeli records.
_M64 = (1 << 64) - 1
_INIT0 = (0xDBE6D5D5FE4CCE2F, 0xA4093822299F31D0, 0x13198A2E03707344, 0x243F6A8885A308D3)
_INIT1 = (0x3BD39E10CB0EF593, 0xC0ACF169B5F18A8C, 0xBE5466CF34E90C6C, 0x452821E638D01377)
_RIEGELI_KEY = (0x2F696C6567656952, 0x0A7364726F636572, 0x2F696C6567656952, 0x0A7364726F636572)


def _zipper(v1: int, v0: int) -> tuple[int, int]:
    a0 = ((((v0 & 0xFF000000) | (v1 & 0xFF00000000)) >> 24)
          | (((v0 & 0xFF0000000000) | (v1 & 0xFF000000000000)) >> 16)
          | (v0 & 0xFF0000) | ((v0 & 0xFF00) << 32)
          | ((v1 & 0xFF00000000000000) >> 8) | ((v0 << 56) & _M64))
    a1 = ((((v1 & 0xFF000000) | (v0 & 0xFF00000000)) >> 24)
          | (v1 & 0xFF0000) | ((v1 & 0xFF0000000000) >> 16)
          | ((v1 & 0xFF00) << 24) | ((v0 & 0xFF000000000000) >> 8)
          | ((v1 & 0xFF) << 48) | (v0 & 0xFF00000000000000))
    return a1 & _M64, a0 & _M64


class _HighwayHash:
    def __init__(self, key: tuple[int, int, int, int]) -> None:
        self.mul0 = list(_INIT0)
        self.mul1 = list(_INIT1)
        self.v0 = [self.mul0[i] ^ key[i] for i in range(4)]
        self.v1 = [self.mul1[i] ^ (((key[i] >> 32) | (key[i] << 32)) & _M64) for i in range(4)]

    def _update(self, lanes: Any) -> None:
        v0, v1, mul0, mul1 = self.v0, self.v1, self.mul0, self.mul1
        for i in range(4):
            v1[i] = (v1[i] + mul0[i] + lanes[i]) & _M64
            mul0[i] ^= ((v1[i] & 0xFFFFFFFF) * (v0[i] >> 32)) & _M64
            v0[i] = (v0[i] + mul1[i]) & _M64
            mul1[i] ^= ((v0[i] & 0xFFFFFFFF) * (v1[i] >> 32)) & _M64
        a1, a0 = _zipper(v1[1], v1[0]); v0[1] = (v0[1] + a1) & _M64; v0[0] = (v0[0] + a0) & _M64
        a1, a0 = _zipper(v1[3], v1[2]); v0[3] = (v0[3] + a1) & _M64; v0[2] = (v0[2] + a0) & _M64
        a1, a0 = _zipper(v0[1], v0[0]); v1[1] = (v1[1] + a1) & _M64; v1[0] = (v1[0] + a0) & _M64
        a1, a0 = _zipper(v0[3], v0[2]); v1[3] = (v1[3] + a1) & _M64; v1[2] = (v1[2] + a0) & _M64

    def _remainder(self, rem: bytes) -> None:
        size = len(rem)
        for i in range(4):
            self.v0[i] = (self.v0[i] + ((size << 32) + size)) & _M64
        if size:
            for i in range(4):
                h0, h1 = self.v1[i] & 0xFFFFFFFF, self.v1[i] >> 32
                h0 = ((h0 << size) | (h0 >> (32 - size))) & 0xFFFFFFFF
                h1 = ((h1 << size) | (h1 >> (32 - size))) & 0xFFFFFFFF
                self.v1[i] = h0 | (h1 << 32)
        packet = bytearray(32)
        mod4 = size & 3
        body = size & ~3
        packet[:body] = rem[:body]
        tail = rem[body:]
        if size & 16:
            for i in range(4):
                packet[28 + i] = rem[body + i + mod4 - 4]
        elif mod4:
            packet[16] = tail[0]
            packet[17] = tail[mod4 >> 1]
            packet[18] = tail[mod4 - 1]
        self._update(struct.unpack("<4Q", bytes(packet)))

    def digest64(self, data: bytes) -> int:
        n = len(data) // 32 * 32
        for off in range(0, n, 32):
            self._update(struct.unpack_from("<4Q", data, off))
        if len(data) & 31:
            self._remainder(data[n:])
        for _ in range(4):
            v0 = self.v0
            self._update([((v0[2] >> 32) | (v0[2] << 32)) & _M64,
                          ((v0[3] >> 32) | (v0[3] << 32)) & _M64,
                          ((v0[0] >> 32) | (v0[0] << 32)) & _M64,
                          ((v0[1] >> 32) | (v0[1] << 32)) & _M64])
        return (self.v0[0] + self.v1[0] + self.mul0[0] + self.mul1[0]) & _M64


def hh64(data: bytes, key: tuple[int, int, int, int] = _RIEGELI_KEY) -> int:
    """HighwayHash-64 of ``data`` (Riegeli's key by default)."""
    return _HighwayHash(key).digest64(bytes(data))


# ---------------------------------------------------------------------------
# Minimal Riegeli records (what XLA's GetSplitProtoRiegeliOptions writes).
_BLOCK = 1 << 16
_BLOCK_HEADER = 24
_CHUNK_HEADER = 40


def _u64(buf: bytes, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        shift += 7
        if byte < 128:
            return value, pos


def _add_with_overhead(pos: int, length: int) -> int:
    """Physical position after ``length`` logical bytes starting at ``pos``."""
    while length > 0:
        if pos % _BLOCK == 0:
            pos += _BLOCK_HEADER
        step = min(_BLOCK - pos % _BLOCK, length)
        pos += step
        length -= step
    return pos


def _read_logical(buf: bytes, pos: int, n: int) -> tuple[bytes, int]:
    out = bytearray()
    while n > 0:
        if pos % _BLOCK == 0:
            pos += _BLOCK_HEADER
        step = min(_BLOCK - pos % _BLOCK, n)
        out += buf[pos:pos + step]
        pos += step
        n -= step
    return bytes(out), pos


def _chunk_end(begin: int, data_size: int, num_records: int) -> int:
    end = begin + num_records
    rem = end % _BLOCK
    if 0 < rem < _BLOCK_HEADER:
        end += _BLOCK_HEADER - rem
    return max(_add_with_overhead(begin, _CHUNK_HEADER + data_size), end)


def _chunks(buf: bytes) -> list[tuple[int, dict[str, int], int]]:
    """``(begin, header, data_pos)`` per chunk; header hashes verified."""
    out = []
    pos = 0
    while pos < len(buf):
        header, data_pos = _read_logical(buf, pos, _CHUNK_HEADER)
        if hh64(header[8:40]) != _u64(header, 0):
            raise SlimModuleError(f"riegeli chunk header hash mismatch at {pos}")
        info = {"data_size": _u64(header, 8), "data_hash": _u64(header, 16),
                "type": header[24], "num_records": int.from_bytes(header[25:32], "little"),
                "decoded_size": _u64(header, 32)}
        out.append((pos, info, data_pos))
        pos = _chunk_end(pos, info["data_size"], info["num_records"])
    if pos != len(buf):
        raise SlimModuleError("riegeli file does not end at a chunk boundary")
    return out


def _snappy_decompress(src: bytes) -> bytes:
    size, pos = _read_varint(src, 0)
    out = bytearray()
    end = len(src)
    while pos < end:
        tag = src[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                nbytes = length - 59
                length = int.from_bytes(src[pos:pos + nbytes], "little")
                pos += nbytes
            length += 1
            out += src[pos:pos + length]
            pos += length
            continue
        if kind == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | src[pos]
            pos += 1
        elif kind == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(src[pos:pos + 2], "little")
            pos += 2
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(src[pos:pos + 4], "little")
            pos += 4
        start = len(out) - offset
        if offset <= 0 or start < 0:
            raise SlimModuleError("invalid snappy copy offset")
        if offset >= length:
            out += out[start:start + length]
        else:  # overlapping copy: repeat the period
            period = bytes(out[start:])
            out += (period * (length // offset + 1))[:length]
    if len(out) != size:
        raise SlimModuleError("snappy size mismatch")
    return bytes(out)


def _decompress(kind: int, buf: bytes) -> bytes:
    if kind == 0:
        return buf
    if kind == ord("s"):
        size, pos = _read_varint(buf, 0)
        out = _snappy_decompress(buf[pos:])
        if len(out) != size:
            raise SlimModuleError("snappy decoded size mismatch")
        return out
    raise SlimModuleError(f"unsupported riegeli compression {kind}")


def _decode_simple_chunk(data: bytes, num_records: int) -> list[bytes]:
    kind = data[0]
    sizes_len, pos = _read_varint(data, 1)
    sizes = _decompress(kind, data[pos:pos + sizes_len])
    values = _decompress(kind, data[pos + sizes_len:])
    records, spos, vpos = [], 0, 0
    for _ in range(num_records):
        n, spos = _read_varint(sizes, spos)
        records.append(values[vpos:vpos + n])
        vpos += n
    if vpos != len(values):
        raise SlimModuleError("simple chunk record sizes do not cover the values")
    return records


def _block_header(block_begin: int, chunk_begin: int, chunk_end: int) -> bytes:
    body = struct.pack("<QQ", block_begin - chunk_begin, chunk_end - block_begin)
    return struct.pack("<Q", hh64(body)) + body


def _append_simple_chunk(prefix: bytes, records: list[bytes]) -> bytes:
    """``prefix`` (ending at a chunk boundary) + one uncompressed simple chunk."""
    sizes = b"".join(_varint(len(r)) for r in records)
    data = b"\x00" + _varint(len(sizes)) + sizes + b"".join(records)
    decoded = sum(len(r) for r in records)
    begin = len(prefix)
    body = (struct.pack("<QQ", len(data), hh64(data)) + b"r"
            + len(records).to_bytes(7, "little") + struct.pack("<Q", decoded))
    logical = struct.pack("<Q", hh64(body)) + body + data
    end = _chunk_end(begin, len(data), len(records))
    out = bytearray(prefix)
    pos, i = begin, 0
    while i < len(logical):
        if pos % _BLOCK == 0:
            out += _block_header(pos, begin, end)
            pos += _BLOCK_HEADER
        step = min(_BLOCK - pos % _BLOCK, len(logical) - i)
        out += logical[i:i + step]
        pos += step
        i += step
    while pos < end:  # padding, with any block header inside it
        if pos % _BLOCK == 0:
            out += _block_header(pos, begin, end)
            pos += _BLOCK_HEADER
            continue
        step = min(_BLOCK - pos % _BLOCK, end - pos)
        out += bytes(step)
        pos += step
    return bytes(out)


# ---------------------------------------------------------------------------
# Protobuf helpers (wire format; field numbers from pinned xla/hlo.proto, xla_data.proto).
_S32 = 4
_UNSLIMMABLE_TYPES = {13, 14, 17, 34}  # TUPLE, OPAQUE_TYPE, TOKEN, BUFFER


def _repack(fields: list[tuple[int, int, Any]]) -> bytes:
    return b"".join(_pack(f, w, v) for f, w, v in fields)


def _exact_fields(data: bytes, what: str) -> list[tuple[int, int, Any]]:
    fields = list(_proto_fields(data))
    if _repack(fields) != data:
        raise SlimModuleError(f"{what}: wire repack not exact")
    return fields


def _only(fields: list[tuple[int, int, Any]], field: int, what: str) -> bytes:
    values = [v for f, w, v in fields if f == field and w == 2]
    if len(values) != 1:
        raise SlimModuleError(f"{what}: expected one field {field}, found {len(values)}")
    return values[0]


def _int_field(fields: list[tuple[int, int, Any]], field: int) -> int | None:
    values = [v for f, w, v in fields if f == field and w == 0]
    return values[-1] if values else None


def _scalar_shape(element_type: int) -> bytes:
    return _pack(2, 0, element_type) + _pack(5, 2, b"")  # element_type, empty layout


def _instruction(name: str, opcode: str, shape: bytes, iid: int,
                 operands: tuple[int, ...] = (), extra: bytes = b"") -> bytes:
    body = _pack(1, 2, name.encode()) + _pack(2, 2, opcode.encode()) + _pack(3, 2, shape) + extra
    body += _pack(35, 0, iid)
    if operands:
        body += _pack(36, 2, b"".join(_varint(o) for o in operands))
    return body


def slim_hlo_module(module: bytes) -> tuple[bytes, dict[str, Any]]:
    """Serialized ``HloModuleProto`` -> slim one (see module docstring) + stats."""
    mod_fields = _exact_fields(module, "HloModuleProto")
    entry_id = _int_field(mod_fields, 6)
    computations = [v for f, w, v in mod_fields if f == 3 and w == 2]
    entries = [c for c in computations if _int_field(list(_proto_fields(c)), 5) == entry_id]
    if entry_id is None or len(entries) != 1:
        raise SlimModuleError("entry computation not found")
    comp_fields = _exact_fields(entries[0], "entry HloComputationProto")
    root_id = _int_field(comp_fields, 6)
    instructions = [list(_proto_fields(v)) for f, w, v in comp_fields if f == 2 and w == 2]
    params = [i for i in instructions if _only(i, 2, "opcode") == b"parameter"]
    roots = [i for i in instructions if _int_field(i, 35) == root_id]
    if len(roots) != 1:
        raise SlimModuleError("entry root instruction not found")
    root_shape = _only(roots[0], 3, "root shape")
    root_shape_fields = list(_proto_fields(root_shape))
    is_tuple = _int_field(root_shape_fields, 2) == 13
    leaves = [v for f, w, v in root_shape_fields if f == 4] if is_tuple else [root_shape]
    next_id = max(_int_field(i, 35) or 0 for i in instructions) + 1
    zero = next_id
    next_id += 1
    literal = _pack(1, 2, _scalar_shape(_S32)) + _pack(4, 2, b"\x00")  # s32s = [0]
    new = [_instruction("slim.zero", "constant", _scalar_shape(_S32), zero,
                        extra=_pack(8, 2, literal))]
    outputs = []
    for k, leaf in enumerate(leaves):
        leaf_fields = list(_proto_fields(leaf))
        element_type = _int_field(leaf_fields, 2)
        if element_type is None or element_type in _UNSLIMMABLE_TYPES or any(
                f == 4 for f, _, _ in leaf_fields):
            raise SlimModuleError(f"output {k}: unsupported element type {element_type}")
        convert, broadcast = next_id, next_id + 1
        next_id += 2
        new.append(_instruction(f"slim.convert.{k}", "convert", _scalar_shape(element_type),
                                convert, (zero,)))
        new.append(_instruction(f"slim.out.{k}", "broadcast", leaf, broadcast, (convert,)))
        outputs.append(broadcast)
    if is_tuple:
        new_root = next_id
        new.append(_instruction("slim.root", "tuple", root_shape, new_root, tuple(outputs)))
    else:
        new_root = outputs[0]
    kept = [(f, w, v) for f, w, v in comp_fields if f not in (2, 6)]
    computation = (_repack([x for x in kept if x[0] == 1])  # name, instructions, rest, root id
                   + b"".join(_pack(2, 2, _repack(p)) for p in params)
                   + b"".join(_pack(2, 2, i) for i in new)
                   + _repack([x for x in kept if x[0] != 1])
                   + _pack(6, 0, new_root))
    out, placed = [], False
    for f, w, v in mod_fields:
        if f == 7:  # schedule: references the removed computations
            continue
        if f == 3:
            if not placed:
                out.append((3, 2, computation))
                placed = True
            continue
        out.append((f, w, v))
    slim = _repack(out)
    return slim, {"module_bytes": [len(module), len(slim)],
                  "computations": len(computations),
                  "entry_instructions": len(instructions),
                  "parameters": len(params), "outputs": len(leaves)}


def slim_blob(blob: bytes) -> tuple[bytes, dict[str, Any]]:
    """Compiled-thunk GPU AOT blob -> same executable with the slim retained module.

    Raises :class:`SlimModuleError` on any layout this rewriter does not handle.
    """
    size, pos = _read_varint(blob, 0)
    head = blob[:pos + size]
    env = _exact_fields(blob[len(head):], "IFRT envelope")
    if not env or env[0][:2] != (1, 2) or env[0][2][:64] != _RIEGELI_SIGNATURE:
        raise SlimModuleError("not a compiled-thunk (Riegeli) GPU executable")
    riegeli = env[0][2]
    chunks = _chunks(riegeli)
    begin, info, data_pos = chunks[-1]
    if info["type"] != ord("r") or info["num_records"] != 1:
        raise SlimModuleError("last riegeli chunk is not one simple record")
    # Chunk data hashes are not re-verified here (pure Python: ~14 s per PROD domain);
    # XLA verifies them on load and the caller reloads the slim blob on the device.
    data, _ = _read_logical(riegeli, data_pos, info["data_size"])
    (rest,) = _decode_simple_chunk(data, 1)
    rest_fields = _exact_fields(rest, "GpuExecutableProto rest record")
    with_config = _exact_fields(_only(rest_fields, 1, "hlo_module_with_config"),
                                "HloModuleProtoWithConfig")
    slim, stats = slim_hlo_module(_only(with_config, 1, "hlo_module"))
    new_with_config = _repack([(f, w, slim if f == 1 else v) for f, w, v in with_config])
    new_rest = _repack([(f, w, new_with_config if f == 1 else v) for f, w, v in rest_fields])
    new_riegeli = _append_simple_chunk(riegeli[:begin], [new_rest])
    stats["riegeli_bytes"] = [len(riegeli), len(new_riegeli)]
    return head + _repack([(1, 2, new_riegeli)] + env[1:]), stats
