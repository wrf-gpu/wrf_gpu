"""Reusable AOT (ahead-of-time) PJRT-executable serialize / load primitives.

WHAT
----
The de-fuse compile-wall fix (``proofs/v021/parallel_compile/AOT_PLAN.md``) rests
on AOT serialization of the compiled ``_advance_chunk_fori`` PJRT executable:
serialize the fully-optimised executable ONCE (in the parallel-prewarm worker),
then in a FRESH process DESERIALIZE + execute it WITHOUT any trace / lower /
compile -- the only mechanism that removes the ~30 min re-lowering the persistent
HLO cache still pays. The GPU kill-gate
(``proofs/v021/parallel_compile/aot_killgate/``) PROVED this on the REAL bigswiss
``d01`` module: byte-identical (78 leaves, 0 diffs), fingerprint-match, deserialize
8.54 s + execute 1.70 s, 123/123 runtime buffers, 174 MB blob.

This module factors that proven logic into two primitives:

* :func:`serialize` ``(compiled) -> (blob, meta)`` -- the blob is
  ``compiled._executable.xla_executable.serialize()``; ``meta`` carries the
  ``in_tree`` / ``out_tree`` / ``kept_var_idx`` (the const-drop the executor
  applies) / ``in_avals`` / a target ``fingerprint``.
* :func:`load` ``(blob, meta, dev) -> callable`` -- deserialize the executable,
  return a drop-in advance closure that flattens its args in JAX call order,
  checks runtime buffer avals, applies the ``kept_var_idx`` filter (XLA
  const-folds/DCEs some inputs, so the
  executable wants FEWER buffers than a naive flatten -- ``pxla.py:373``:
  ``args = [x for i,x in enumerate(args) if i in self.kept_var_idx]``), calls
  ``le.execute``, and unflattens the result via ``out_tree``.

NUMERICS / SAFETY
-----------------
**Identity-preserving.** The blob IS the cached optimised executable; loading it
runs byte-identical numerics to the cold compile (kill-gate proved it). Every
entry point is **fail-open**: on any missing-attr / fingerprint-mismatch /
deserialize / call-contract / execute error the caller falls back to the normal
jitted path -- never wrong, only slower. The fingerprint guards target
compatibility (SM/jaxlib/driver-specific blobs MUST NOT be mis-targeted -- a
SIGILL risk).

PRIVATE APIs (pinned jaxlib 0.10.0; guarded + fail-open):
``compiled._executable.xla_executable.serialize()``,
``compiled._executable._kept_var_idx``,
``dev.client.deserialize_executable(blob, jaxlib._jax.DeviceList((dev,)), None)``.

GPU exports omit HLO source coordinates, stack frames and op labels to avoid
rebuilding expensive profiler annotations on every load. Set
``GPUWRF_AOT_KEEP_TRACE_METADATA=1`` before export for source-level profiling;
that request refuses a compact cached blob so the caller can export a rich one.
Unsupported serialized formats retain their original bytes. GPU exports are
then re-exported in XLA's compiled-thunk format so loads skip thunk/Triton
re-emission (``GPUWRF_AOT_COMPILED_THUNKS=0`` keeps the legacy format). A
compiled-thunk export then keeps only the entry signature of its retained
HloModule (``gpuwrf.runtime.aot_slim_module``), so executes skip XLA's per-call
module walk; ``GPUWRF_AOT_SLIM_MODULE=0`` (or a profiler-rich export) keeps it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import jax

__all__ = [
    "AotMeta",
    "AotSerializeError",
    "target_fingerprint",
    "hlo_sha256_from_lowered",
    "blob_sha256",
    "serialize",
    "load",
    "bind_loaded_executable",
    "fingerprint_matches",
    "export_compile_options",
]


def blob_sha256(blob: bytes) -> str:
    """sha256 hex of the serialized executable bytes (blob-integrity check)."""
    return hashlib.sha256(blob).hexdigest()


# The pinned XLA compiled-thunk format is a Riegeli split GpuExecutableProto.
# Version this policy independently of source keys and jaxlib's ABI fingerprint.
GPU_THUNK_FORMAT = "xla-gpu-thunks-riegeli-v1"
GPU_LEGACY_FORMAT = "xla-gpu-legacy-v1"
_RIEGELI_SIGNATURE = bytes.fromhex(
    "83af70d10d884a3f0000000000000000400000000000000091bac23c9287e1a9"
    "0000000000000000e19f13c0e9b1c37273000000000000000000000000000000"
)


def _gpu_blob_format(blob: bytes) -> str | None:
    """Inspect the IFRT envelope without copying the large native payload."""
    def varint(pos):
        value = 0
        for shift in range(0, 70, 7):
            byte = blob[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
        raise ValueError("invalid GPU AOT varint")
    try:
        size, pos = varint(0)
        pos += size
        tag, pos = varint(pos)
        if tag != 10:
            return None
        size, pos = varint(pos)
        if size > len(blob) - pos:
            return None
        return (GPU_THUNK_FORMAT if blob[pos:pos + 64] == _RIEGELI_SIGNATURE
                else GPU_LEGACY_FORMAT)
    except (IndexError, ValueError):
        return None


class AotSerializeError(RuntimeError):
    """Raised by :func:`serialize` when the private AOT API is unavailable.

    Callers MUST treat this as fail-open (skip AOT, fall back to compile).
    """


def _proto_fields(data: bytes):
    """Read the wire fields needed by the pinned JAX 0.10 CPU AOT format."""
    def varint(pos):
        value = 0
        for shift in range(0, 70, 7):
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
        raise ValueError("invalid protobuf varint")

    pos = 0
    while pos < len(data):
        tag, pos = varint(pos)
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            value, pos = varint(pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, pos = varint(pos)
            else:
                size = 8 if wire == 1 else 4
            end = pos + size
            if end > len(data):
                raise ValueError("truncated protobuf field")
            value, pos = data[pos:end], end
        else:
            raise ValueError("unsupported protobuf wire type")
        yield field, wire, value


def _cpu_pjrt_fields(blob: bytes):
    """Decode the length-prefixed IFRT header and its CPU PJRT payload."""
    length = 0
    for pos, byte in enumerate(blob[:10]):
        length |= (byte & 127) << (7 * pos)
        if byte < 128:
            offset = pos + 1 + length
            break
    else:
        raise ValueError("invalid IFRT header length")
    if offset >= len(blob):
        raise ValueError("missing CPU executable payload")
    return list(_proto_fields(blob[offset:]))


def _compact_gpu_trace_metadata(blob: bytes) -> tuple[bytes, dict[str, Any]]:
    """Trim profiler labels only from the pinned legacy GPU AOT wire format.

    Computational HLO fields, PTX, cubin, buffer assignment, module config and
    effective compile options remain unchanged. Unsupported formats keep their
    original bytes. Kernel binary names remain available to Nsight; retain the
    original export when source-level profiler attribution is required.
    """
    import jaxlib

    if jaxlib.__version__ != "0.10.0":
        return blob, {"reason": "unsupported-jaxlib-version"}

    def varint(value):
        result = bytearray()
        while value > 127:
            result.append((value & 127) | 128)
            value >>= 7
        result.append(value)
        return bytes(result)

    def pack(field, wire, value):
        tag = varint((field << 3) | wire)
        if wire == 0:
            return tag + varint(value)
        return tag + (varint(len(value)) if wire == 2 else b"") + value

    children = {
        ("envelope", 1): "gpu", ("gpu", 1): "withconfig",
        ("withconfig", 1): "module", ("module", 3): "computation",
        ("computation", 2): "instruction", ("instruction", 7): "metadata",
    }
    removed = 0

    def rewrite(data, level):
        nonlocal removed
        output = []
        for field, wire, value in _proto_fields(data):
            if (level == "metadata" and field in (1, 2, 3, 4, 15, 17, 18, 19)) or (
                level == "module" and field == 17
            ):
                removed += 1
                continue
            child = children.get((level, field)) if wire == 2 else None
            if child:
                value = rewrite(value, child)
            output.append(pack(field, wire, value))
        return b"".join(output)

    try:
        # IFRT header is length-prefixed; retain the header byte-for-byte.
        length = shift = position = 0
        while True:
            byte = blob[position]
            position += 1
            length |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
            if shift >= 70:
                raise ValueError("invalid IFRT prefix")
        offset = position + length
        fields = list(_proto_fields(blob[offset:]))
        if [(f, w) for f, w, v in fields] != [(1, 2), (2, 2)]:
            return blob, {"reason": "unsupported-envelope"}
        gpu = list(_proto_fields(fields[0][2]))
        if [(f, w) for f, w, v in gpu] != [(1, 2), (2, 2), (3, 2), (4, 2)]:
            return blob, {"reason": "unsupported-gpu-format"}
        if gpu[3][2][:4] != b"\x7fELF":
            return blob, {"reason": "no-native-elf"}
        compact = blob[:offset] + rewrite(blob[offset:], "envelope")
        updated = list(_proto_fields(compact[offset:]))
        updated_gpu = list(_proto_fields(updated[0][2]))
        if fields[1] != updated[1] or gpu[1:] != updated_gpu[1:]:
            raise ValueError("GPU native code, ABI or compile options changed")
        return compact, {"reason": "source-trace-labels", "removed_fields": removed,
                         "original_bytes": len(blob), "compact_bytes": len(compact)}
    except (ValueError, IndexError, TypeError):
        return blob, {"reason": "unsupported-or-invalid-format"}


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _pack(field: int, wire: int, value: Any) -> bytes:
    tag = _varint((field << 3) | wire)
    if wire == 0:
        return tag + _varint(value)
    return tag + (_varint(len(value)) if wire == 2 else b"") + value


def _rewrite_one(data: bytes, field: int, edit: Callable[[bytes], bytes]) -> bytes:
    fields = list(_proto_fields(data))
    if sum(f == field and w == 2 for f, w, _ in fields) != 1:
        raise ValueError(f"expected one length-delimited field {field}")
    return b"".join(_pack(f, w, edit(v) if f == field else v) for f, w, v in fields)


_COMPILED_THUNKS_DEBUG_OPTION = 435  # DebugOptions.xla_gpu_experimental_aot_compiled_thunks


def _with_compiled_thunks_option(blob: bytes) -> bytes:
    """Set Debug435 in a legacy GPU blob's stored module config; keep all else.

    Path: IFRT envelope -> GPU result (1) -> HloModuleProtoWithConfig (1) ->
    HloModuleConfigProto (2) -> DebugOptions (14). The pinned exporter reads the
    loaded module's config to choose the compiled-thunk format.
    """
    def set_flag(debug: bytes) -> bytes:
        kept = [x for x in _proto_fields(debug) if x[0] != _COMPILED_THUNKS_DEBUG_OPTION]
        return b"".join(_pack(*x) for x in kept) + _pack(_COMPILED_THUNKS_DEBUG_OPTION, 0, 1)

    length, pos = 0, 0
    for shift in range(0, 70, 7):
        byte = blob[pos]
        pos += 1
        length |= (byte & 127) << shift
        if byte < 128:
            break
    else:
        raise ValueError("invalid IFRT prefix")
    offset = pos + length
    return blob[:offset] + _rewrite_one(blob[offset:], 1, lambda gpu: _rewrite_one(
        gpu, 1, lambda hwc: _rewrite_one(hwc, 2, lambda cfg: _rewrite_one(cfg, 14, set_flag))))


_MEMORY_STAT_FIELDS = ("argument_size_in_bytes", "output_size_in_bytes",
                       "alias_size_in_bytes", "temp_size_in_bytes")


def _memory_stats(executable: Any) -> dict[str, Any]:
    stats = executable.get_compiled_memory_stats()
    return {name: getattr(stats, name, None)
            for name in _MEMORY_STAT_FIELDS + ("generated_code_size_in_bytes",)}


def _convert_to_compiled_thunks(
    blob: bytes, dev: Any, reference: dict[str, Any] | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Re-export a legacy GPU blob in XLA's compiled-thunk format.

    Cached loads then skip thunk re-emission and Triton re-lowering (B12n/B12o):
    the paid LW7 PROD blobs load in 2 s instead of 20 s per domain, with native
    code, buffer assignment and options exact and all 536 carry leaves
    bit-identical (CP69). Converting the compacted legacy blob keeps it compact.
    Global flag-435 exports keep rich source metadata, which costs about 5.7 s
    per domain on load (CP78). This costs one legacy load per domain at capture.
    Any failure returns the input blob unchanged.
    """
    import time

    started = time.perf_counter()
    status: dict[str, Any] = {"reason": None, "legacy_bytes": len(blob)}
    try:
        if _gpu_blob_format(blob) != GPU_LEGACY_FORMAT:
            status["reason"] = "not-legacy-format"
            return blob, status
        import jaxlib._jax as _jax

        devices = _jax.DeviceList((dev,))
        loaded = dev.client.deserialize_executable(
            _with_compiled_thunks_option(blob), devices, None)
        converted = bytes(loaded.serialize())
        del loaded
        if _gpu_blob_format(converted) != GPU_THUNK_FORMAT:
            status["reason"] = "export-not-compiled-thunks"
            return blob, status
        reloaded = dev.client.deserialize_executable(converted, devices, None)
        stats = _memory_stats(reloaded)
        del reloaded
        status["memory_stats"] = stats
        if reference is not None and any(
            stats[name] != reference.get(name) for name in _MEMORY_STAT_FIELDS
        ):
            status.update(reason="buffer-sizes-changed", reference_stats=reference)
            return blob, status
        status.update(reason="converted", thunk_bytes=len(converted))
        return converted, status
    except Exception as exc:  # noqa: BLE001 - legacy blob remains valid
        status["reason"] = f"error:{type(exc).__name__}: {exc}"
        return blob, status
    finally:
        status["seconds"] = time.perf_counter() - started


_COMPILED_THUNKS_COMPILE_OPTION = "xla_gpu_experimental_aot_compiled_thunks"  # DebugOptions 435


def _env_on(name: str, default: str) -> bool:
    import os

    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def export_compile_options() -> dict[str, Any] | None:
    """Compile options for a GPU program that :func:`serialize` will export.

    At pinned XLA b6f37ab DebugOptions 435 is read only by ``GpuCompiler::Export``
    and ``CompileAheadOfTime``: the compiled program is unchanged and
    ``serialize()`` emits compiled thunks directly. That skips the legacy reload
    in :func:`_convert_to_compiled_thunks` (thunk re-emission and Triton
    re-lowering, about 30 s per PROD domain). A direct export keeps rich source
    metadata in its retained module (E81), so this is only on when the slim
    module removes it. ``GPUWRF_AOT_DIRECT_THUNKS=0`` keeps the conversion route.
    """
    if jax.default_backend() != "gpu":
        return None
    if not (_env_on("GPUWRF_AOT_DIRECT_THUNKS", "1") and _env_on("GPUWRF_AOT_COMPILED_THUNKS", "1")
            and _env_on("GPUWRF_AOT_SLIM_MODULE", "1")
            and not _env_on("GPUWRF_AOT_KEEP_TRACE_METADATA", "0")):
        return None
    return {_COMPILED_THUNKS_COMPILE_OPTION: True}


def _slim_retained_module(
    blob: bytes, dev: Any, reference: dict[str, Any] | None = None
) -> tuple[bytes, dict[str, Any]]:
    """Replace a compiled-thunk blob's retained HloModule by its entry signature.

    Loaded executables then skip the per-execute module walk and most of the
    module parsing at load (see :mod:`gpuwrf.runtime.aot_slim_module`). The slim
    blob must deserialize on ``dev`` with the same buffer sizes. Any failure
    returns the input blob unchanged.
    """
    import time

    started = time.perf_counter()
    status: dict[str, Any] = {"reason": None, "full_bytes": len(blob)}
    try:
        if _gpu_blob_format(blob) != GPU_THUNK_FORMAT:
            status["reason"] = "not-compiled-thunks"
            return blob, status
        import jaxlib._jax as _jax

        from gpuwrf.runtime import aot_slim_module

        slim, stats = aot_slim_module.slim_blob(blob)
        status.update(stats)
        reloaded = dev.client.deserialize_executable(slim, _jax.DeviceList((dev,)), None)
        memory = _memory_stats(reloaded)
        del reloaded
        if reference is not None and any(
            memory[name] != reference.get(name) for name in _MEMORY_STAT_FIELDS
        ):
            status.update(reason="buffer-sizes-changed", memory_stats=memory,
                          reference_stats=reference)
            return blob, status
        status.update(reason="slimmed", slim_bytes=len(slim))
        return slim, status
    except Exception as exc:  # noqa: BLE001 - the full blob remains valid
        status["reason"] = f"error:{type(exc).__name__}: {exc}"
        return blob, status
    finally:
        status["seconds"] = time.perf_counter() - started


def _cpu_compile_options(blob: bytes):
    """Recover the effective options, including compile-call overrides."""
    import jaxlib._jax as _jax

    options = [v for f, w, v in _cpu_pjrt_fields(blob) if f == 2 and w == 2]
    if len(options) != 1:
        raise AotSerializeError("original CPU compile options are unavailable")
    return _jax.CompileOptions.ParseFromString(options[0])


def _check_cpu_native_objects(blob: bytes, platform: str | None) -> None:
    """Reject incomplete CPU exports before publishing or loading them.

    JAX 0.10 CPU cache hits lose obj_files/compiled_symbols on re-export.
    Deserialize succeeds, then execution fails with NOT_FOUND. Read the IFRT
    length-prefixed header, PjRt CPU executable field 1, and the CPU compilation
    result's compiled_symbols (7) and object_files (8). Empty/copy-only thunk
    sequences legitimately need no native objects (e.g. constant outputs).
    GPU payloads use a different format and are left to the GPU loader.
    """
    if platform != "cpu":
        return
    try:
        results = [v for f, w, v in _cpu_pjrt_fields(blob) if f == 1 and w == 2]
        if len(results) != 1:
            raise ValueError("missing CPU compilation result")
        fields = list(_proto_fields(results[0]))
        symbols = any(f == 7 and w == 2 and v for f, w, v in fields)
        objects = any(
            any(of == 1 and ow == 2 and ov for of, ow, ov in _proto_fields(v))
            for f, w, v in fields if f == 8 and w == 2
        )
        if not symbols and not objects:
            thunks = [
                thunk
                for f, w, sequence in fields if f == 6 and w == 2
                for tf, tw, thunk in _proto_fields(sequence) if tf == 1 and tw == 2
            ]
            if all(
                [v for f, w, v in _proto_fields(thunk) if f == 1 and w == 2] == [b"copy"]
                for thunk in thunks
            ):
                return
        if not symbols or not objects:
            raise ValueError("CPU AOT export has no native objects or compiled symbols")
    except (ValueError, IndexError) as exc:
        raise AotSerializeError(f"incomplete CPU AOT executable: {exc}") from exc


def _fresh_cpu_blob(compiled: Any, lowered: Any, blob: bytes, status: dict) -> bytes:
    """Compile the same IR with the exact serialized original compile options.

    Go straight to the backend to bypass both JAX caches without recomputing
    options from today's config. The original PJRT device list and callbacks
    are retained. Refuse publication unless the new effective options serialize
    identically, including overrides, optimization and assignment. The Python
    backend binding adds its in-place-IR marker and can materialize an inferred
    output layout; actual input/output layouts must remain identical too.
    """
    import time
    from jax._src import compiler
    from jax._src.lib.mlir import ir
    from jax._src.layout import Layout

    original_options = _cpu_compile_options(blob)
    expected = original_options.SerializeAsString()
    unloaded = compiled._executable._unloaded_executable
    device_list = unloaded.device_list
    if device_list is None or unloaded.backend.platform != "cpu":
        raise AotSerializeError("original CPU device assignment is unavailable")
    callbacks = lowered._lowering.compile_args["host_callbacks"]
    hlo = lowered.compiler_ir("stablehlo")
    # The backend binding permits in-place MLIR modification. Keep the caller's
    # lowering immutable, including the StableHLO digest used for its AOT key.
    hlo = ir.Module.parse(str(hlo), context=hlo.context)
    started = time.perf_counter()
    try:
        executable = compiler.backend_compile_and_load(
            unloaded.backend, hlo, device_list,
            original_options, callbacks,
        )
    finally:
        status["aot_recovery_compile_seconds"] = time.perf_counter() - started
        jax.monitoring.record_event_duration_secs(
            "/gpuwrf/aot/backend_compile_recovery_duration",
            status["aot_recovery_compile_seconds"],
        )
    fresh_blob = bytes(executable.serialize())
    fresh_options = _cpu_compile_options(fresh_blob)

    def normalized_options(options):
        fields = []
        for f, w, value in _proto_fields(options.SerializeAsString()):
            if f == 9:  # allow_in_place_mlir_modification (Python binding only)
                continue
            if f == 3 and w == 2:
                value = tuple(
                    (bf, bw, bv) for bf, bw, bv in _proto_fields(value)
                    if bf != 2 or original_options.executable_build_options.result_layout is not None
                )
            fields.append((f, w, value))
        return tuple(fields)

    def layouts(exe):
        return tuple(
            tuple(Layout.from_pjrt_layout(x) for x in getter())
            for getter in (exe.get_parameter_layouts, exe.get_output_layouts)
        )

    if (
        normalized_options(fresh_options) != normalized_options(original_options)
        or layouts(executable) != layouts(compiled._executable.xla_executable)
    ):
        raise AotSerializeError("fresh CPU AOT effective compile options changed; refusing export")
    status["aot_recovery_layouts_verified"] = True
    status["aot_recovery_compile_options_sha256"] = blob_sha256(expected)
    _check_cpu_native_objects(fresh_blob, "cpu")
    return fresh_blob


@dataclass(frozen=True)
class AotMeta:
    """Everything a fresh process needs to call a deserialized executable.

    * ``in_tree`` / ``out_tree`` -- the pytree treedefs of the (flattened) call
      args and the result, so the consumer can flatten inputs + unflatten outputs.
    * ``kept_var_idx`` -- sorted indices into the naive ``tree_flatten`` of the
      inputs that the executable actually consumes (the XLA const-drop). ``None``
      means it could not be captured (API drift) -> consumer uses the naive
      flatten and surfaces any buffer-count mismatch.
    * ``in_avals`` -- ``[{"shape","dtype","weak_type"}]`` per naive input leaf
      (a structural cross-check; catches the silent Step-C fallback class where a
      loaded executable sees e.g. an ``s32[]`` runtime scalar but was compiled for
      weak ``s64[]`` under x64).
    * ``hlo_sha256`` -- sha256 of the lowered StableHLO text: the program identity
      (the persistent-cache key is the HLO, so this pins WHICH program the blob is).
    * ``fingerprint`` -- the TARGET fingerprint (jaxlib/device/CC/driver/x64): the
      blob is target-specific, so the consumer must match it or fall back.
    """

    in_tree: Any
    out_tree: Any
    kept_var_idx: tuple[int, ...] | None
    in_avals: tuple[dict[str, Any], ...]
    hlo_sha256: str | None
    fingerprint: dict[str, Any]
    # Cheap-key manifest (vNext): the metadata-only lookup key that locates this
    # blob WITHOUT lowering, the schema tag it was minted under, and the sha256 of
    # the serialized blob bytes (re-verified on load to reject a truncated/edited
    # blob). All optional so older pickled metas + direct callers stay fail-open.
    cheap_key: str | None = None
    key_schema: str | None = None
    blob_sha256: str | None = None
    trace_metadata: str | None = None
    executable_format: str | None = None
    # "slim" = retained HloModule reduced to the entry signature (aot_slim_module),
    # "full" = as exported; None for older metas and non-GPU blobs.
    retained_module: str | None = None


def target_fingerprint(dev: Any | None = None) -> dict[str, Any]:
    """Best-effort TARGET fingerprint for the active backend.

    Captures what makes a serialized PJRT blob target-specific: jaxlib version,
    device kind/platform, compute capability, CUDA driver/runtime string, and the
    x64 flag. Missing pieces degrade to ``None`` rather than raising (fail-open).
    The HLO program identity is carried separately (:attr:`AotMeta.hlo_sha256`).
    """
    if dev is None:
        try:
            dev = jax.devices()[0]
        except Exception:  # noqa: BLE001 - fail-open: empty fingerprint
            dev = None

    def _g(obj: Any, attr: str) -> Any:
        try:
            val = getattr(obj, attr, None)
        except Exception:  # noqa: BLE001
            return None
        return None if val is None else str(val)

    try:
        import jaxlib

        jaxlib_v = getattr(jaxlib, "__version__", None)
    except Exception:  # noqa: BLE001
        jaxlib_v = None

    client = getattr(dev, "client", None) if dev is not None else None
    try:
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:  # noqa: BLE001
        x64 = None

    import os as _os

    return {
        "jaxlib_version": jaxlib_v,
        "jax_version": getattr(jax, "__version__", None),
        "device_kind": _g(dev, "device_kind"),
        "platform": _g(dev, "platform"),
        "compute_capability": _g(dev, "compute_capability"),
        "cuda_driver": _g(client, "platform_version"),
        "x64": x64,
        # XLA_FLAGS change the compiled blob (autotune/codegen) but are set late
        # by the autotune hook, so version_cache_tag (which avoids backend init)
        # cannot capture them. Pin them here so a blob compiled under one flag set
        # is never mis-targeted to another (also folded into cheap_key comp 1).
        "xla_flags": _os.environ.get("XLA_FLAGS", ""),
    }


def _extract_kept_var_idx(compiled: Any) -> tuple[int, ...] | None:
    """Sorted ``kept_var_idx`` for ``compiled`` (the executor's const-drop set).

    Reads ``compiled._executable._kept_var_idx`` (fallback the ``ExecuteReplicated``
    on ``unsafe_call``). Returns ``None`` on any API drift (consumer falls back to
    the naive flatten)."""
    me = getattr(compiled, "_executable", None)
    if me is None:
        return None
    kvi = getattr(me, "_kept_var_idx", None)
    if kvi is None:
        uc = getattr(me, "unsafe_call", None)
        kvi = getattr(uc, "kept_var_idx", None)
    if kvi is None:
        return None
    try:
        return tuple(sorted(int(i) for i in kvi))
    except (TypeError, ValueError):
        return None


def _sha256_text(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def hlo_sha256_from_lowered(lowered: Any) -> str | None:
    """sha256 of a lowered program, or ``None`` on API drift.

    This is the cheap Step-C lookup key: derive the HLO identity from
    ``lowered`` without compiling. Prefer ``Lowered.as_text()`` because it is
    present in JAX 0.10 and stable across CPU/GPU for the same lowered program.
    """
    fn = getattr(lowered, "as_text", None)
    if callable(fn):
        try:
            digest = _sha256_text(fn())
            if digest:
                return digest
        except Exception:  # noqa: BLE001
            pass
    fn = getattr(lowered, "compiler_ir", None)
    if callable(fn):
        for kwargs in ({"dialect": "hlo"}, {}):
            try:
                ir = fn(**kwargs)
                if hasattr(ir, "as_hlo_text"):
                    text = ir.as_hlo_text()
                else:
                    text = str(ir)
                digest = _sha256_text(text)
                if digest:
                    return digest
            except Exception:  # noqa: BLE001
                pass
    return None


def _compiled_hlo_sha256(compiled: Any) -> str | None:
    """Fallback sha256 of compiled HLO text, or ``None``.

    New Step-B/Step-C variant blobs pass a lower-only digest into
    :func:`serialize`. This fallback keeps direct callers/tests and old-style
    one-off blobs fail-open when a lower digest is not available.
    """
    for getter in ("as_text",):
        fn = getattr(compiled, getter, None)
        if callable(fn):
            try:
                digest = _sha256_text(fn())
                if digest:
                    return digest
            except Exception:  # noqa: BLE001
                pass
    return None


def _arg_info_aval(info: Any) -> Any | None:
    """Return the JAX 0.10 ``ArgInfo`` aval across private attribute spellings."""
    aval = getattr(info, "aval", None)
    if aval is None:
        aval = getattr(info, "_aval", None)
    return aval


def _aval_record(aval: Any) -> dict[str, Any]:
    """Small, pickle-safe shape/dtype record for an abstract or concrete leaf."""
    return {
        "shape": tuple(getattr(aval, "shape", ()) or ()),
        "dtype": str(getattr(aval, "dtype", "")),
        "weak_type": bool(getattr(aval, "weak_type", False)),
    }


def _compiled_in_aval_records(compiled: Any) -> tuple[dict[str, Any], ...]:
    """Flatten ``compiled`` input avals in the same leaf order as ``in_tree``.

    JAX 0.10 exposes the reliable tree as ``compiled.in_avals``. Older kill-gate
    code tried ``ArgInfo.aval``; the installed class stores ``_aval`` instead, so
    keep that as a guarded fallback.
    """
    try:
        in_avals = getattr(compiled, "in_avals", None)
        if in_avals is not None:
            return tuple(_aval_record(a) for a in jax.tree_util.tree_leaves(in_avals))
    except Exception:  # noqa: BLE001 - cross-check only
        pass
    try:
        args_info = getattr(compiled, "args_info", None)
        if args_info is not None:
            flat_info = jax.tree_util.tree_leaves(
                args_info, is_leaf=lambda x: _arg_info_aval(x) is not None
            )
            return tuple(_aval_record(_arg_info_aval(i)) for i in flat_info)
    except Exception:  # noqa: BLE001 - cross-check only
        pass
    return ()


def _flatten_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Any]:
    """Flatten a call exactly like ``jax.stages.Compiled.call`` when possible."""
    try:
        from jax._src import tree_util as _src_tree_util

        registry = getattr(_src_tree_util, "tracing_registry", None)
        if registry is not None and hasattr(registry, "flatten"):
            return registry.flatten((args, kwargs))
    except Exception:  # noqa: BLE001 - fall back to public pytree flattening
        pass
    return jax.tree_util.tree_flatten((args, kwargs))


def _leaf_aval_record(value: Any) -> dict[str, Any]:
    """Abstractify a runtime call leaf without materializing device arrays."""
    try:
        from jax._src import api as _api

        return _aval_record(_api.shaped_abstractify(value))
    except Exception:  # noqa: BLE001
        return {
            "shape": tuple(getattr(value, "shape", ()) or ()),
            "dtype": str(getattr(value, "dtype", "")),
            "weak_type": bool(getattr(value, "weak_type", False)),
        }


def _check_selected_avals(
    flat: list[Any],
    expected: tuple[dict[str, Any], ...],
    kept: tuple[int, ...] | None,
) -> None:
    """Fail early with a useful message for runtime shape/dtype mismatches."""
    if not expected or len(expected) != len(flat):
        return
    indices = kept if kept is not None else tuple(range(len(flat)))
    for idx in indices:
        exp = expected[idx]
        exp_dtype = exp.get("dtype")
        # Older metadata may not have real avals; let the executable surface it.
        if not exp_dtype:
            continue
        got = _leaf_aval_record(flat[idx])
        if tuple(got.get("shape", ())) != tuple(exp.get("shape", ())) or str(
            got.get("dtype", "")
        ) != str(exp_dtype):
            raise RuntimeError(
                "AOT input aval mismatch at flattened leaf "
                f"{idx}: expected shape={tuple(exp.get('shape', ()))}, "
                f"dtype={exp_dtype}, weak_type={bool(exp.get('weak_type', False))}; "
                f"got shape={tuple(got.get('shape', ()))}, dtype={got.get('dtype')}, "
                f"weak_type={bool(got.get('weak_type', False))}; fall back to compile"
            )


def _tree_num_leaves(tree: Any) -> int | None:
    """Best-effort ``PyTreeDef.num_leaves`` without comparing static metadata."""
    try:
        return int(getattr(tree, "num_leaves"))
    except Exception:  # noqa: BLE001
        return None


def _check_call_contract(
    flat: list[Any],
    in_tree_now: Any,
    expected_tree: Any,
    expected_avals: tuple[dict[str, Any], ...],
    kept: tuple[int, ...] | None,
    hlo_sha256: str | None,
) -> None:
    """Validate an AOT call without raw ``PyTreeDef`` equality.

    ``OperationalNamelist`` static aux can contain JAX arrays (e.g. ``eta_levels``).
    Raw treedef equality then calls array equality while comparing metadata and can
    raise or return false even for value-identical static config after metadata is
    pickled/unpickled. Static config identity is instead guarded by the serialized
    HLO digest: if the blob is used, it runs the HLO that already baked those
    statics. Runtime safety comes from kept-buffer shape/dtype checks.
    """
    if not hlo_sha256:
        raise RuntimeError(
            "AOT HLO fingerprint missing; refusing relaxed in_tree match and "
            "falling back to compile"
        )

    seen_leaves = len(flat)
    expected_leaves = (
        len(expected_avals) if expected_avals else _tree_num_leaves(expected_tree)
    )
    if expected_leaves is not None and seen_leaves != expected_leaves:
        raise RuntimeError(
            "AOT input leaf-count mismatch (call structure differs from the "
            f"serialized executable): expected {expected_leaves}, got {seen_leaves}; "
            "fall back to compile"
        )

    # ``in_tree_now`` is still useful as a source of the seen leaf count when the
    # flat list comes from a non-standard registry. Avoid ``!=`` on full treedefs:
    # that is the array-bearing-static-aux failure this helper exists to bypass.
    seen_tree_leaves = _tree_num_leaves(in_tree_now)
    if (
        expected_leaves is not None
        and seen_tree_leaves is not None
        and seen_tree_leaves != expected_leaves
    ):
        raise RuntimeError(
            "AOT input treedef leaf-count mismatch (call structure differs from "
            f"the serialized executable): expected {expected_leaves}, got "
            f"{seen_tree_leaves}; fall back to compile"
        )

    _check_selected_avals(flat, expected_avals, kept)


def serialize(
    compiled: Any,
    *,
    dev: Any | None = None,
    hlo_sha256: str | None = None,
    lowered: Any | None = None,
    cheap_key: str | None = None,
    key_schema: str | None = None,
    status: dict[str, Any] | None = None,
) -> tuple[bytes, AotMeta]:
    """Serialize a ``jax.stages.Compiled`` to ``(blob, meta)``.

    ``blob`` = ``compiled._executable.xla_executable.serialize()`` (the fully
    optimised PJRT executable). ``meta`` carries the call structure + target
    fingerprint :func:`load` needs.

    ``meta.hlo_sha256`` MUST be the SAME lower-only StableHLO digest the load /
    verify path recomputes via :func:`hlo_sha256_from_lowered` -- otherwise the
    cross-process cheap-key load rejects the blob (``not meta_hlo`` -> fallback)
    and verify-mode never confirms it. Digest precedence (first non-None wins):

    1. the caller-supplied ``hlo_sha256`` (the lower-only digest captured at the
       lower point -- the authoritative key);
    2. ``hlo_sha256_from_lowered(lowered)`` when the ``lowered`` object is passed
       in (so ``serialize`` can re-derive the SAME digest even if the caller's
       capture returned ``None`` -- e.g. a backend where the lower digest was not
       threaded through);
    3. a best-effort ``_compiled_hlo_sha256(compiled)`` fallback ONLY for direct
       callers / tests with neither (NOTE: the compiled-HLO text is a DIFFERENT
       program representation than the lowered StableHLO, so this digest does NOT
       match the load/verify key -- it is a last-resort non-empty value, never the
       cheap-key contract key).

    Raises :class:`AotSerializeError` if the private API is unavailable (caller
    treats it as fail-open -> skip AOT). When provided, ``status`` records the
    recovery reason, compile/total recovery times and verified options digest.
    Recovery compile time is also emitted to JAX monitoring for benchmark cost
    accounting, including a failed verification after the fresh compile.
    """
    status = {} if status is None else status
    status.update(aot_recovery=None, aot_recovery_reason=None,
                  aot_recovery_compile_seconds=0.0, aot_recovery_seconds=0.0)
    me = getattr(compiled, "_executable", None)
    if me is None:
        raise AotSerializeError("compiled._executable missing (jax API drift)")
    xla_exec = getattr(me, "xla_executable", None)
    if xla_exec is None or not hasattr(xla_exec, "serialize"):
        raise AotSerializeError(
            "compiled._executable.xla_executable.serialize unavailable (jax API drift)"
        )
    try:
        blob = bytes(xla_exec.serialize())
        if dev is None:
            dev = jax.devices()[0]
        try:
            _check_cpu_native_objects(blob, getattr(dev, "platform", None))
        except AotSerializeError as exc:
            status["aot_recovery_reason"] = str(exc)
            if lowered is None:
                status["aot_recovery"] = "refused:no-lowering"
                raise
            import time
            started = time.perf_counter()
            status["aot_recovery"] = "refused:unverified-options"
            try:
                blob = _fresh_cpu_blob(compiled, lowered, blob, status)
                status["aot_recovery"] = "fresh-cpu-compile"
            finally:
                status["aot_recovery_seconds"] = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        raise AotSerializeError(f"xla_executable.serialize() raised: {exc}") from exc

    trace_metadata = "rich"
    retained_module = None
    if getattr(dev, "platform", None) == "gpu":
        import os
        import time
        keep_trace = os.environ.get("GPUWRF_AOT_KEEP_TRACE_METADATA", "0").lower() in {
            "1", "true", "yes", "on"
        }
        if not keep_trace:
            started = time.perf_counter()
            blob, trace_status = _compact_gpu_trace_metadata(blob)
            status["aot_trace_compaction_seconds"] = time.perf_counter() - started
            status["aot_trace_compaction"] = trace_status
            if trace_status.get("reason") == "source-trace-labels":
                trace_metadata = "compact"
        if os.environ.get("GPUWRF_AOT_COMPILED_THUNKS", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }:
            try:
                reference = _memory_stats(xla_exec)
            except Exception:  # noqa: BLE001 - conversion still checks format
                reference = None
            blob, status["aot_compiled_thunks"] = _convert_to_compiled_thunks(
                blob, dev, reference)
            if not keep_trace and os.environ.get(
                "GPUWRF_AOT_SLIM_MODULE", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}:
                blob, slim_status = _slim_retained_module(blob, dev, reference)
                status["aot_slim_module"] = slim_status
                if slim_status.get("reason") == "slimmed":
                    retained_module = "slim"
        if retained_module is None:
            retained_module = "full"
    status["aot_trace_metadata"] = trace_metadata

    in_tree = getattr(compiled, "in_tree", None)
    out_tree = getattr(compiled, "out_tree", None)
    if in_tree is None or out_tree is None:
        raise AotSerializeError("compiled.in_tree/out_tree missing (jax API drift)")

    in_avals = _compiled_in_aval_records(compiled)

    # Derive the persisted HLO digest from the SAME lowered-StableHLO source the
    # load/verify path uses. The compiled-HLO fallback is a DIFFERENT program text
    # (and is None on some backends), so it can never satisfy the cheap-key load
    # contract -- it only keeps neither-arg direct callers/tests non-empty.
    persisted_hlo = hlo_sha256
    if not persisted_hlo and lowered is not None:
        persisted_hlo = hlo_sha256_from_lowered(lowered)
    if not persisted_hlo:
        persisted_hlo = _compiled_hlo_sha256(compiled)

    meta = AotMeta(
        in_tree=in_tree,
        out_tree=out_tree,
        kept_var_idx=_extract_kept_var_idx(compiled),
        in_avals=in_avals,
        hlo_sha256=persisted_hlo,
        fingerprint=target_fingerprint(dev),
        cheap_key=cheap_key,
        key_schema=key_schema,
        blob_sha256=blob_sha256(blob),
        trace_metadata=trace_metadata,
        executable_format=(_gpu_blob_format(blob)
                           if getattr(dev, "platform", None) == "gpu" else None),
        retained_module=retained_module,
    )
    return blob, meta


def fingerprint_matches(
    saved: dict[str, Any], live: dict[str, Any] | None = None, *, dev: Any | None = None
) -> bool:
    """True iff the saved target fingerprint matches the live backend.

    Compares the keys present in ``saved`` against the live fingerprint
    (:func:`target_fingerprint`). A ``None`` on either side for a given key is a
    MISMATCH (we refuse to load when we cannot positively confirm the target),
    EXCEPT we never block on a key absent from ``saved``. Fail-CLOSED: any error
    comparing returns ``False`` (fall back to compile)."""
    try:
        if live is None:
            live = target_fingerprint(dev)
        for key, sval in saved.items():
            if key not in live:
                continue
            if sval != live.get(key):
                return False
        return True
    except Exception:  # noqa: BLE001 - fail-closed
        return False


def _committed_on(x: Any, dev: Any) -> bool:
    """True iff ``x`` is a jax.Array committed to exactly ``dev`` (single device, default memory)."""
    if not isinstance(x, jax.Array) or not getattr(x, "_committed", False):
        return False
    sharding = getattr(x, "sharding", None)
    if not isinstance(sharding, jax.sharding.SingleDeviceSharding) or sharding.device_set != {dev}:
        return False
    return getattr(sharding, "memory_kind", None) in (None, "device")


def load(
    blob: bytes,
    meta: AotMeta,
    dev: Any | None = None,
    *,
    check_fingerprint: bool = True,
) -> Callable[..., Any]:
    """Deserialize ``blob`` and return a drop-in callable ``f(*args) -> pytree``.

    The returned callable mimics the executable-binding part of
    ``jax.stages.Compiled.call``: flatten ``(args, kwargs)`` in JAX call order,
    validate the runtime kept-buffer avals against ``meta.in_avals``, keep only
    ``meta.kept_var_idx`` (the const-drop), ``le.execute`` the kept buffers, then
    ``tree_unflatten`` the outputs via ``meta.out_tree``.

    Raises on any failure (missing API, fingerprint mismatch, call-contract
    mismatch, deserialize error) so the CALLER can fall back to the jitted path.
    The caller is responsible for the fail-open try/except (see
    :func:`gpuwrf.runtime.domain_tree._maybe_aot_advance`)."""
    if dev is None:
        dev = jax.devices()[0]

    _check_cpu_native_objects(blob, getattr(dev, "platform", None))
    if getattr(dev, "platform", None) == "gpu":
        import os
        saved_format = getattr(meta, "executable_format", None)
        if saved_format is not None and saved_format != _gpu_blob_format(blob):
            raise RuntimeError("AOT executable format/version mismatch; fall back to compile")
        rich_requested = os.environ.get("GPUWRF_AOT_KEEP_TRACE_METADATA", "0").lower() in {
            "1", "true", "yes", "on"
        }
        if rich_requested and getattr(meta, "trace_metadata", None) == "compact":
            raise RuntimeError(
                "AOT source metadata was compacted; profiler-rich export requested; "
                "re-export from a fresh compile"
            )
        if rich_requested and getattr(meta, "retained_module", None) == "slim":
            raise RuntimeError(
                "AOT retained HloModule was slimmed; profiler-rich export requested; "
                "re-export from a fresh compile"
            )

    if check_fingerprint and not fingerprint_matches(meta.fingerprint, dev=dev):
        raise RuntimeError(
            "AOT fingerprint mismatch (target differs from the serialized blob); "
            "refusing to load (SIGILL risk) -- fall back to compile"
        )

    client = getattr(dev, "client", None)
    if client is None or not hasattr(client, "deserialize_executable"):
        raise RuntimeError("dev.client.deserialize_executable unavailable (jax API drift)")

    try:
        import jaxlib._jax as _jax
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot import jaxlib._jax ({exc}); API drift") from exc

    device_list = _jax.DeviceList((dev,))
    le = client.deserialize_executable(blob, device_list, None)

    return bind_loaded_executable(le, meta, dev)


def bind_loaded_executable(le: Any, meta: AotMeta, dev: Any) -> Callable:
    """Bind validated native code to its saved call contract without reloading.

    The caller must attest that ``meta`` describes this executable and target.
    Runtime shape/dtype checks remain identical to :func:`load`.
    """

    in_tree = meta.in_tree
    out_tree = meta.out_tree
    kept = meta.kept_var_idx
    expected_avals = meta.in_avals

    # Call signatures that already passed _check_call_contract. The contract depends
    # only on the leaf count and the kept leaves' shape/dtype, so a repeated signature
    # skips the per-leaf aval re-derivation on the per-step dispatch path.
    contract_ok: set[tuple[Any, ...]] = set()
    # Per kept position: the last UNcommitted jax.Array passed there and its committed
    # device_put result. Re-passing the same immutable array object (namelist children,
    # clock base) reuses that buffer instead of a device_put per call. The cheap key is
    # computed before aot_call and never sees the committed copy. Host leaves always go
    # through device_put; a committed leaf at that position drops the entry.
    put_cache: dict[int, tuple[Any, Any]] = {}

    def aot_call(*args: Any, **kwargs: Any) -> Any:
        # JAX 0.10 ``Compiled.call`` uses ``tree_util.tracing_registry.flatten`` on
        # ``(args, kwargs)``. Match that leaf order exactly, including keyword args,
        # so kept_var_idx addresses the same flattened input vector JAX would pass.
        flat, in_tree_now = _flatten_call(args, kwargs)
        indices = kept if kept is not None else range(len(flat))
        try:
            signature = (
                len(flat),
                tuple(
                    (type(flat[i]), getattr(flat[i], "shape", None), getattr(flat[i], "dtype", None))
                    for i in indices
                ),
            )
            hash(signature)
        except Exception:  # noqa: BLE001 - unhashable/out-of-range: full check
            signature = None
        if signature is None or signature not in contract_ok:
            _check_call_contract(
                flat,
                in_tree_now,
                in_tree,
                expected_avals,
                kept,
                meta.hlo_sha256,
            )
            if signature is not None:
                contract_ok.add(signature)
        if kept is not None:
            try:
                selected = [flat[i] for i in kept]
            except IndexError as exc:
                raise RuntimeError(
                    f"AOT kept_var_idx out of range for {len(flat)} input leaves "
                    f"({exc}); fall back to compile"
                ) from exc
        else:
            selected = list(flat)
        # Leaves already COMMITTED to ``dev`` in its default memory are passed as-is
        # (device_put would return the same placement); every other leaf (host,
        # uncommitted, other device/memory kind) goes through device_put as before.
        in_bufs = []
        for pos, x in enumerate(selected):
            if _committed_on(x, dev):
                put_cache.pop(pos, None)
                in_bufs.append(x)
                continue
            if isinstance(x, jax.Array):
                hit = put_cache.get(pos)
                if hit is not None and hit[0] is x:
                    in_bufs.append(hit[1])
                    continue
                y = jax.device_put(x, dev)
                put_cache[pos] = (x, y)
                in_bufs.append(y)
            else:
                in_bufs.append(jax.device_put(x, dev))
        out_bufs = le.execute(in_bufs)
        return jax.tree_util.tree_unflatten(out_tree, out_bufs)

    # Expose the underlying executable + meta for introspection / warm-hit checks.
    aot_call.loaded_executable = le  # type: ignore[attr-defined]
    aot_call.meta = meta  # type: ignore[attr-defined]
    aot_call.put_cache = put_cache  # type: ignore[attr-defined]
    return aot_call
