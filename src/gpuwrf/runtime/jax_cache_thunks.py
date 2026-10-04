"""Write Triton-bearing JAX persistent-cache entries as compiled thunks (B12n).

A legacy GPU cache entry re-lowers every Triton (Pallas) kernel on each load:
LW9's warm k1 loads ``jit__coupled_updates`` twice at 2.1 s each. The nested AOT
blobs are already converted by :func:`aot_executable.serialize`; this module does
the same for JAX's own cache entries: same key, same native code, faster load.

Kept outside the trace import closure on purpose: it is installed when the AOT
machinery loads (``aot_precompile``), so editing it never shifts a cheap key.
``GPUWRF_JAX_CACHE_COMPILED_THUNKS=0`` keeps legacy writes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gpuwrf.runtime import aot_executable as _ax

TRITON_CUSTOM_CALL = b"__gpu$xla.gpu.triton"
# Loaded from the nested AOT blobs, never from their large JAX-cache twins:
# converting those would only add cold-run time.
AOT_MANAGED_MODULES = frozenset({"jit__advance_chunk_fori", "jit_fused_jit"})
STATUS: dict[str, Any] = {"installed": False, "converted": 0, "kept_legacy": 0,
                          "seconds": 0.0, "last": None}


class _CompiledThunkExport:
    """Executable view whose ``serialize()`` yields the compiled-thunk format."""

    def __init__(self, executable: Any, module_name: str) -> None:
        self._executable = executable
        self._module_name = module_name

    def serialize(self) -> bytes:
        blob = self._executable.serialize()
        try:
            return _cache_blob(self._executable, self._module_name, blob)
        except Exception:  # noqa: BLE001 - the legacy entry stays valid
            return blob

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executable, name)


def _cache_blob(executable: Any, module_name: str, blob: bytes) -> bytes:
    """Convert a Triton-bearing legacy GPU entry; anything else is unchanged."""
    if module_name in AOT_MANAGED_MODULES or TRITON_CUSTOM_CALL not in blob:
        return blob
    if _ax._gpu_blob_format(blob) != _ax.GPU_LEGACY_FORMAT:
        return blob
    converted, status = _ax._convert_to_compiled_thunks(
        blob, executable.local_devices()[0], _ax._memory_stats(executable))
    ok = status.get("reason") == "converted"
    STATUS["converted" if ok else "kept_legacy"] += 1
    STATUS["seconds"] += float(status.get("seconds") or 0.0)
    STATUS["last"] = {"module": module_name, "reason": status.get("reason")}
    return converted


def install() -> bool:
    """Wrap JAX's cache write so GPU entries serialize through the converter.

    The cache key is computed before the write, so keys are unchanged; existing
    legacy entries keep loading. Costs one legacy load per converted entry at
    write (cold) time. Idempotent; never raises."""
    if os.environ.get("GPUWRF_JAX_CACHE_COMPILED_THUNKS", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return False
    try:
        from jax._src import compilation_cache as _jcc

        original = _jcc.put_executable_and_time
        if not getattr(original, "_gpuwrf_thunks", False):
            def put_executable_and_time(cache_key, module_name, executable, backend, compile_time):
                if getattr(backend, "platform", None) == "gpu":
                    executable = _CompiledThunkExport(executable, module_name)
                return original(cache_key, module_name, executable, backend, compile_time)

            put_executable_and_time._gpuwrf_thunks = True  # type: ignore[attr-defined]
            _jcc.put_executable_and_time = put_executable_and_time
        STATUS["installed"] = True
        return True
    except Exception as exc:  # noqa: BLE001 - legacy writes remain correct
        STATUS["last"] = {"error": f"{type(exc).__name__}: {exc}"}
        return False


def upgrade_cache_dir(cache_dir: str | Path, device: Any = None) -> dict[str, Any]:
    """Rewrite existing Triton-bearing legacy entries of a JAX cache dir as compiled thunks.

    Migrates caches written before :func:`install` without recompiling: same
    key/filename, same native code, atomic replace. Needs the GPU the entries were
    built for; run it while no other process uses the cache."""
    import jaxlib._jax as _jax
    from jax._src import compilation_cache as _jcc

    if device is None:
        import jax

        device = jax.devices()[0]
    report: dict[str, Any] = {"converted": [], "kept": [], "skipped": 0}
    for path in sorted(Path(cache_dir).glob("*-cache")):
        if path.name.rsplit("-", 2)[0] in AOT_MANAGED_MODULES:
            report["skipped"] += 1
            continue
        blob, compile_time = _jcc.extract_executable_and_time(
            _jcc.decompress_executable(path.read_bytes()))
        if TRITON_CUSTOM_CALL not in blob or _ax._gpu_blob_format(blob) != _ax.GPU_LEGACY_FORMAT:
            report["skipped"] += 1
            continue
        legacy = device.client.deserialize_executable(blob, _jax.DeviceList((device,)), None)
        reference = _ax._memory_stats(legacy)
        del legacy
        converted, status = _ax._convert_to_compiled_thunks(blob, device, reference)
        entry = {"entry": path.name, "reason": status.get("reason"),
                 "seconds": status.get("seconds"), "bytes": [len(blob), len(converted)]}
        if status.get("reason") != "converted":
            report["kept"].append(entry)
            continue
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(_jcc.compress_executable(
            _jcc.combine_executable_and_time(converted, compile_time)))
        os.replace(tmp, path)
        report["converted"].append(entry)
    return report
