"""Content-addressed AOT executable store keyed by the lowered program (v0.5.0 item 0.d).

WHY
---
Two cold compiles of the SAME lowered program give different binaries (autotune /
emitter choices, FINDINGS E99/E282), so two builds whose source fingerprint
differs but whose program is identical run different executables and their
outputs differ by compile noise. The per-cache AOT store already has an
exact-HLO address, but it lives inside one cache root, its digest keeps Pallas
source lines, and its target guard compares the raw ``XLA_FLAGS`` string, which
carries a bundle-specific autotune path. This module adds a store that several
builds can share:

    cheap key hit (unchanged, no lowering)
      -> else lower -> content key -> store hit -> same binary, cheap key aliased
      -> else compile -> publish (first writer wins; a loser runs the winner)

DEFAULT OFF. Nothing here runs unless ``GPUWRF_AOT_HLO_STORE`` (read/write root)
or ``GPUWRF_AOT_HLO_SEEDS`` (``os.pathsep`` list of read-only roots) is set. The
names are key-inert (``aot_cheap_key.HLO_AFFECTING_ENV_DENYLIST``). A root that
cannot be parsed counts as unset (:data:`CONFIG_ERRORS`).

CONTENT KEY: bind every effective compile input or refuse
----------------------------------------------------------
``sha256`` of a canonical JSON record of

* ``program_sha256``: the lowered StableHLO module as MLIR bytecode after
  ``strip-debuginfo``. Bytecode keeps every constant in full (E224). Each Pallas
  custom call embeds Triton IR bytecode with source locations (E58/E109); that IR
  is parsed, stripped the same way and replaced by its digest.
* ``compile_options_sha256``: the serialized ``CompileOptions`` JAX hands to XLA
  for exactly this lowering and these per-call options (rebuilt with JAX's own
  ``create_compile_options``). It holds the options embedded in the lowering, the
  debug options XLA parsed from ``XLA_FLAGS`` (inline or from a flags file), the
  device assignment and the optimisation efforts. Only fields that name a
  location or request a dump are cleared (:data:`LOCATION_DEBUG_FIELDS`).
* explicit autotune inputs: the content digest of a results file given with
  ``--xla_gpu_load_autotune_results_from``.
* target: jax/jaxlib, platform, device kind and id, compute capability, backend
  platform version, x64; the NVIDIA kernel driver version and the digests of the
  CUDA compiler files XLA is pointed at (CUDA), or the host CPU features (CPU).
* the AOT export switches (they select the serialized format).

REFUSED (no key, hence ordinary lower + compile, never a shared binary): constants
hoisted into arguments, more than one device, automatic SPMD partitioning, shape
polymorphism, host callbacks, profile-guided recompilation, autotune cache mode
READ, an unreadable autotune results file, an unresolved driver or toolchain on
CUDA, any backend other than CPU or CUDA, and every error while keying.

STORE ENTRIES
-------------
``<root>/v2/keys/<K>.json`` + ``<root>/v2/blobs/<K>/<blob sha256>.xlaexec``. A
blob lives under the key it implements, so a record cannot point at another
program's blob. The record carries the key's components, the blob digest and
size, the executable's parameter list and an attestation digest over all of
them, so a record whose blob pointer was changed no longer verifies. The
attestation is a consistency seal, not a signature: whoever can rewrite a whole
entry (copy a blob under the key and recompute the digest) can forge it, exactly
as for the local AOT cache, whose metadata are pickles. Entries are immutable in
seed roots. In the read/write root an entry that fails verification is repaired
(same digest) or set aside so a valid one can be published.

SERVING
-------
A blob is read once through one file descriptor, hashed, and installed either
as a hard link whose inode equals the hashed one or as a copy of the hashed
bytes; the installed file is checked again. The consumer builds its call
contract from its own lowering and never unpickles store content. A refusal is
remembered for the process: the refused entry is never adopted or aliased, and
the consumer keeps its own compile. A publisher that finds another valid binary
under its key loads and runs that binary instead of its own.

AUTOTUNE POOL (``GPUWRF_AOT_HLO_AUTOTUNE_SEED=1``, default off, GPU-unqualified)
----------------------------------------------------------------------------
Shares XLA's per-fusion autotune decision files per compile environment: missing
names are copied into the live cache before a compile that follows a store miss
and new names are copied back afterwards.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

__all__ = [
    "KEY_SCHEMA",
    "STORE_ENV",
    "SEEDS_ENV",
    "AUTOTUNE_ENV",
    "CONFIG_ERRORS",
    "LOCATION_DEBUG_FIELDS",
    "NEUTRAL_XLA_FLAGS",
    "ContentKey",
    "Refused",
    "enabled",
    "store_roots",
    "canonical_xla_flags",
    "canonical_program_sha256",
    "effective_compile_options",
    "canonical_compile_options",
    "key_or_refusal",
    "content_key",
    "lookup",
    "publish",
    "materialize",
    "prepare",
    "take_winner",
    "note_local_hit",
    "note_init_kernel",
    "adopt_winner",
    "record_publication",
    "link_local",
    "verify_local",
    "meta_from_lowered",
    "report",
    "autotune_seed_enabled",
    "autotune_import",
    "autotune_export",
]

KEY_SCHEMA = "GPUWRF-AOT-HLOKEY-v2"
STORE_ENV = "GPUWRF_AOT_HLO_STORE"
SEEDS_ENV = "GPUWRF_AOT_HLO_SEEDS"
AUTOTUNE_ENV = "GPUWRF_AOT_HLO_AUTOTUNE_SEED"
_AUTOTUNE_DIR = "xla_gpu_per_fusion_autotune_cache_dir"  # JAX places it below its cache dir (E35)
_AUTOTUNE_FILE = re.compile(r"[A-Za-z0-9_-]{43}\.textproto")
_AUTOTUNE_MAX_BYTES = 64 * 1024
_LAYOUT = "v2"
_OFF_VALUES = {"", "0", "false", "off", "no"}
_TRITON_TARGET = "__gpu$xla.gpu.triton"

# Process-wide bookkeeping (one process = one run).
CONFIG_ERRORS: list[str] = []                # store roots that could not be parsed
# content key -> {"state": "placed" | "refused", "blob_sha256": <entry concerned>, "blocking": bool}.
# "blocking" = that very blob must not be adopted here (its call contract differs, or it was
# placed and the process still had to compile). An integrity refusal does not block: the entry
# is verified again, under the publication lock, when this process publishes.
_KEY_STATE: dict[str, dict[str, Any]] = {}
_INIT_KERNELS: list[dict[str, Any]] = []     # init-kernel calls of this process (store on only)
_LOCK_TIMEOUT_S = 300.0                      # a publication takes seconds; a timeout leaves the run unshared
_LOCKS_HELD: dict[str, list[Any]] = {}       # content key -> [thread ident, depth, fd]
_LOCKS_GUARD = threading.Lock()
_WINNER_CALLS: dict[str, Any] = {}           # content key -> loaded executable of the winning binary
_EVENTS: list[dict[str, Any]] = []           # every lookup / publication with the binary that runs
_MAX_EVENTS = 4096

# DebugOptions fields that only name a location or ask for a dump. They are
# cleared before the options are hashed; every other field stays in the key.
# The CUDA directory and the autotune results file are bound by CONTENT instead
# (toolchain file digests, results-file digest).
LOCATION_DEBUG_FIELDS: tuple[str, ...] = (
    "xla_gpu_per_fusion_autotune_cache_dir",
    "xla_gpu_kernel_cache_file",
    "xla_gpu_dump_autotune_results_to",
    "xla_gpu_dump_autotune_logs_to",
    "xla_gpu_load_autotune_results_from",
    "xla_gpu_cuda_data_dir",
    "xla_dump_to",
    "xla_dump_hlo_module_re",
    "xla_dump_hlo_pass_re",
    "xla_dump_hlo_pipeline_re",
    "xla_dump_hlo_as_text",
    "xla_dump_hlo_as_long_text",
    "xla_dump_hlo_as_proto",
    "xla_dump_hlo_as_dot",
    "xla_dump_hlo_as_url",
    "xla_dump_hlo_as_html",
    "xla_dump_hlo_snapshots",
    "xla_dump_fusion_visualization",
    "xla_dump_max_hlo_modules",
    "xla_dump_module_metadata",
    "xla_dump_compress_protos",
    "xla_dump_disable_metadata",
)

# Flag names ignored when the autotune POOL directory is scoped (not the key).
NEUTRAL_XLA_FLAGS: frozenset[str] = frozenset(
    set(LOCATION_DEBUG_FIELDS)
    | {"xla_gpu_experimental_autotune_cache_mode", "xla_gpu_force_compilation_parallelism",
       "xla_dump_include_timestamp"}
)

# Resolved AOT export switches: they select the serialized format of the blob.
_EXPORT_SWITCHES = (
    ("GPUWRF_AOT_COMPILED_THUNKS", "1"),
    ("GPUWRF_AOT_SLIM_MODULE", "1"),
    ("GPUWRF_AOT_DIRECT_THUNKS", "1"),
    ("GPUWRF_AOT_KEEP_TRACE_METADATA", "0"),
)
_TOOLCHAIN_FILES = ("bin/ptxas", "nvvm/libdevice/libdevice.10.bc")
_TOOLCHAIN_GLOBS = ("nvvm/lib64/libnvvm.so*",)


class Refused(Exception):
    """This case is not keyed; the caller compiles as if the store were off."""


class _Invalid(Exception):
    """A store entry failed verification."""


class _Busy(Exception):
    """The publication lock of a key could not be taken."""


@dataclasses.dataclass(frozen=True)
class ContentKey:
    """A content key with the record it hashes."""

    key: str
    program_sha256: str
    components: dict[str, Any]
    info: dict[str, Any] = dataclasses.field(default_factory=dict, compare=False)


# --------------------------------------------------------------------------- #
# Switch
# --------------------------------------------------------------------------- #
def _parse_root(raw: str, name: str) -> Path | None:
    text = raw.strip()
    if text.lower() in _OFF_VALUES:
        return None
    try:
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise ValueError("not an absolute path")
        return path
    except Exception as exc:  # noqa: BLE001 - a bad root is an unset root
        message = f"{name}={text!r}: {type(exc).__name__}: {exc}"
        if message not in CONFIG_ERRORS:
            CONFIG_ERRORS.append(message)
        return None


def store_roots() -> tuple[Path | None, tuple[Path, ...]]:
    """``(read/write root or None, read-only seed roots)``. Never raises."""
    try:
        rw = _parse_root(os.environ.get(STORE_ENV, ""), STORE_ENV)
        seeds = tuple(
            path for path in (_parse_root(part, SEEDS_ENV)
                              for part in os.environ.get(SEEDS_ENV, "").split(os.pathsep))
            if path is not None
        )
        return rw, seeds
    except Exception as exc:  # noqa: BLE001
        CONFIG_ERRORS.append(f"store roots: {type(exc).__name__}: {exc}")
        return None, ()


def enabled() -> bool:
    rw, seeds = store_roots()
    return rw is not None or bool(seeds)


def _roots() -> list[tuple[Path, bool]]:
    rw, seeds = store_roots()
    return ([(rw, True)] if rw is not None else []) + [(seed, False) for seed in seeds]


# --------------------------------------------------------------------------- #
# Program
# --------------------------------------------------------------------------- #
def _strip_debuginfo(operation: Any) -> None:
    from jax._src.lib.mlir import passmanager as pm

    pm.PassManager.parse("builtin.module(strip-debuginfo)").run(operation)


def _bytecode(operation: Any) -> bytes:
    buf = io.BytesIO()
    operation.write_bytecode(file=buf)
    return buf.getvalue()


def _canonical_triton_ir(raw: bytes) -> bytes:
    """Triton IR bytecode of one Pallas kernel without source locations."""
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir
    from jax._src.lib.triton import dialect as tt_dialect

    ctx = mlir.make_ir_context()
    tt_dialect.register_dialect(ctx)
    with ctx:
        module = ir.Module.parse(raw)
        _strip_debuginfo(module.operation)
        return _bytecode(module.operation)


def _walk_custom_calls(operation: Any, out: list[Any]) -> list[Any]:
    for region in operation.regions:
        for block in region.blocks:
            for op in block.operations:
                inner = op.operation
                if inner.name == "stablehlo.custom_call":
                    out.append(inner)
                _walk_custom_calls(inner, out)
    return out


def _canonicalize_triton_calls(operation: Any, info: dict[str, Any]) -> None:
    """Replace each embedded Triton IR by the digest of its location-free form.

    A kernel that cannot be canonicalized keeps its raw bytes: the key then stays
    sensitive to that kernel's source locations (a miss, never a wrong hit).
    """
    from jax._src.lib.mlir import ir

    memo: dict[bytes, str] = {}
    calls = kernels = raw_kept = 0
    for op in _walk_custom_calls(operation, []):
        attrs = op.attributes
        try:
            if ir.StringAttr(attrs["call_target_name"]).value != _TRITON_TARGET:
                continue
            config = ir.DictAttr(attrs["mhlo.backend_config"])
            raw = ir.StringAttr(config["ir"]).value_bytes
        except Exception:  # noqa: BLE001 - other custom call or other layout
            continue
        calls += 1
        raw_digest = hashlib.sha256(raw).digest()
        digest = memo.get(raw_digest)
        if digest is None:
            try:
                digest = hashlib.sha256(_canonical_triton_ir(raw)).hexdigest()
                kernels += 1
            except Exception:  # noqa: BLE001 - keep the raw IR in the key
                raw_kept += 1
                continue
            memo[raw_digest] = digest
        items = {}
        for index in range(len(config)):
            named = config[index]
            items[named.name] = named.attr
        items["ir"] = ir.StringAttr.get(f"ttir-sha256:{digest}")
        attrs["mhlo.backend_config"] = ir.DictAttr.get(items)
    info.update(triton_calls=calls, triton_kernels=kernels, triton_raw_kept=raw_kept)


def canonical_program_sha256(lowered: Any, info: dict[str, Any] | None = None) -> str:
    """sha256 of the lowered module: all constants, no source locations."""
    info = {} if info is None else info
    module = lowered.compiler_ir("stablehlo")
    with module.context:
        clone = module.operation.clone()
        _strip_debuginfo(clone.operation)
        _canonicalize_triton_calls(clone.operation, info)
        data = _bytecode(clone.operation)
    info["program_bytes"] = len(data)
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Compile inputs
# --------------------------------------------------------------------------- #
def effective_compile_options(lowered: Any, compile_options: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """``(CompileOptions, device)`` that ``lowered.compile(compile_options)`` uses.

    Rebuilt with JAX's own ``create_compile_options`` from the lowering's compile
    arguments, as ``MeshComputation.compile`` -> ``from_hlo`` ->
    ``_cached_compilation`` does. Raises :class:`Refused` for every case whose
    compile inputs this module does not bind.
    """
    import jax
    import numpy as np
    from jax._src.interpreters import pxla

    lowering = lowered._lowering
    args = lowering.compile_args
    if len(getattr(lowering, "const_args", ()) or ()) != 0:
        raise Refused("hoisted-constants")  # values live outside the module and the blob
    devices = lowering._device_list
    if devices is None or len(devices) != 1:
        raise Refused("device-assignment")  # the AOT loader binds exactly one device
    if args.get("auto_spmd_lowering"):
        raise Refused("auto-spmd")
    poly = args.get("shape_poly_state")
    if poly is not None and getattr(poly, "uses_dim_vars", False):
        raise Refused("shape-polymorphism")
    if args.get("host_callbacks"):
        raise Refused("host-callbacks")
    if jax.config.jax_enable_pgle or jax.config.jax_compilation_cache_expect_pgle:
        raise Refused("profile-guided-recompilation")
    kvs = tuple(lowering._compiler_options_kvs) + tuple((compile_options or {}).items())
    in_shardings = tuple(pxla.maybe_concretize_mesh(s, devices) for s in args["in_shardings"])
    out_shardings = tuple(pxla.maybe_concretize_mesh(s, devices) for s in args["out_shardings"])
    prop_in, prop_out = pxla.get_prop_to_input_output(
        in_shardings, out_shardings, len(args["ordered_effects"]))
    np_dev = np.vectorize(lambda i: devices[i], otypes=[object])(np.arange(len(devices)))
    options = pxla.create_compile_options(
        lowering._hlo, None, args["tuple_args"], False, prop_in, prop_out,
        args["backend"], np_dev, dict(kvs))
    return options, devices[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _explicit_autotune(options: Any) -> dict[str, Any]:
    """Autotune inputs whose CONTENT selects code: bound by digest or refused."""
    from jax._src.lib import xla_client as xc

    debug = options.executable_build_options.debug_options
    read_mode = getattr(getattr(xc, "AutotuneCacheMode", None), "READ", None)
    mode = getattr(debug, "xla_gpu_experimental_autotune_cache_mode", None)
    if read_mode is not None and mode is not None and (
            mode == read_mode or str(getattr(mode, "name", mode)).upper() == "READ"):
        raise Refused("autotune-cache-read-mode")  # a pinned decision set selects the binary
    results = str(getattr(debug, "xla_gpu_load_autotune_results_from", "") or "")
    if not results:
        return {"load_results_sha256": None}
    try:
        return {"load_results_sha256": _file_sha256(Path(results))}
    except OSError as exc:
        raise Refused(f"autotune-results-unreadable: {exc}") from exc


def canonical_compile_options(options: Any) -> bytes:
    """Serialized compile options with location and dump fields cleared."""
    clone = copy.deepcopy(options)
    debug = clone.executable_build_options.debug_options
    for name in LOCATION_DEBUG_FIELDS:
        try:
            current = getattr(debug, name)
        except AttributeError:
            continue  # not exposed by this jaxlib: it stays in the key (stricter)
        if isinstance(current, bool):
            setattr(debug, name, False)
        elif isinstance(current, str):
            setattr(debug, name, "")
        else:
            setattr(debug, name, 0)
    try:
        debug.xla_gpu_experimental_autotune_cache_mode = 0  # UPDATE vs unset: where results are cached
    except Exception:  # noqa: BLE001 - not exposed: stays in the key
        pass
    return bytes(clone.SerializeAsString())


_NVIDIA_VERSION_FILES = ("/sys/module/nvidia/version", "/proc/driver/nvidia/version")


def _driver_identity(platform: str) -> dict[str, Any]:
    """What executes or generates target code outside jaxlib."""
    if platform == "cpu":
        return {"cpu_features": _cpu_features()}
    for path in _NVIDIA_VERSION_FILES:
        try:
            text = Path(path).read_text(encoding="ascii", errors="replace").strip()
        except OSError:
            continue
        if text:
            return {"nvidia_kernel_module": text.splitlines()[0].strip()}
    raise Refused("driver-unresolved")


_FILE_DIGESTS: dict[tuple[str, int, int], str] = {}


def _toolchain_identity(platform: str, cuda_data_dir: str) -> dict[str, Any]:
    """Compiler files XLA is pointed at, by content, plus the CUDA wheels."""
    if platform == "cpu":
        return {"kind": "cpu-llvm-in-jaxlib"}
    root = Path(cuda_data_dir) if cuda_data_dir else None
    if root is None or not root.is_dir():
        raise Refused("toolchain-unresolved: no CUDA data directory in the compile options")
    files: dict[str, str] = {}
    candidates = [root / rel for rel in _TOOLCHAIN_FILES]
    for pattern in _TOOLCHAIN_GLOBS:
        candidates += sorted(root.glob(pattern))
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        memo = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if memo not in _FILE_DIGESTS:
            _FILE_DIGESTS[memo] = _file_sha256(path)
        files[str(path.relative_to(root))] = _FILE_DIGESTS[memo]
    if "bin/ptxas" not in files:
        raise Refused("toolchain-unresolved: no ptxas under the CUDA data directory")
    return {"files": files, "wheels": _cuda_wheels()}


def _cuda_wheels() -> list[list[str]]:
    from importlib import metadata

    found = set()
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower().replace("_", "-")
        if name.startswith(("nvidia-", "jax-cuda")):
            found.add((name, dist.version))
    return [list(item) for item in sorted(found)]


def _cpu_features() -> str:
    """Digest of the host CPU model and feature flags (CPU blobs are host code)."""
    try:
        model = flags = ""
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name") and not model:
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    flags = " ".join(sorted(line.split(":", 1)[1].split()))
                if model and flags:
                    break
        return hashlib.sha256(f"{model}|{flags}".encode()).hexdigest()
    except OSError:
        import platform

        return "machine:" + platform.machine()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return f"{type(value).__name__}:{value!r}"


def _digest(components: dict[str, Any]) -> str:
    text = json.dumps(components, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _environment(options: Any, device: Any) -> dict[str, Any]:
    from gpuwrf.backends.platform import device_backend
    from gpuwrf.runtime import aot_executable as aotx

    target = dict(aotx.target_fingerprint(device))
    target.pop("xla_flags", None)  # bound through the parsed debug options instead
    target["device_id"] = int(getattr(device, "id", 0))
    platform = str(target.get("platform"))
    backend = "cpu" if platform == "cpu" else device_backend(device)
    if backend not in ("cpu", "cuda"):
        raise Refused(f"backend-unsupported: {backend}")
    debug = options.executable_build_options.debug_options
    return {
        "target": _plain(target),
        "driver": _driver_identity("cpu" if backend == "cpu" else "cuda"),
        "toolchain": _toolchain_identity(
            "cpu" if backend == "cpu" else "cuda",
            str(getattr(debug, "xla_gpu_cuda_data_dir", "") or "")),
        "explicit_autotune": _explicit_autotune(options),
        "compile_options_sha256": hashlib.sha256(canonical_compile_options(options)).hexdigest(),
        "export": {
            name: os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}
            for name, default in _EXPORT_SWITCHES
        },
    }


def key_or_refusal(
    lowered: Any, *, compile_options: dict[str, Any] | None = None, dev: Any | None = None
) -> tuple[ContentKey | None, str | None]:
    """``(key, None)`` or ``(None, reason)``. ``dev`` is ignored: the device of the
    lowering is the one the compile uses. Never raises."""
    del dev
    try:
        info: dict[str, Any] = {}
        started = time.perf_counter()
        options, device = effective_compile_options(lowered, compile_options)
        environment = _environment(options, device)
        program = canonical_program_sha256(lowered, info)
        components = {"schema": KEY_SCHEMA, "program_sha256": program, **environment}
        info["key_seconds"] = time.perf_counter() - started
        info["backend"] = f"{environment['target'].get('platform')}:{environment['target'].get('device_kind')}"
        return ContentKey(_digest(components), program, components, info), None
    except Refused as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - an unkeyed case is an unshared case
        return None, f"error:{type(exc).__name__}: {exc}"


def content_key(
    lowered: Any, *, compile_options: dict[str, Any] | None = None, dev: Any | None = None
) -> ContentKey | None:
    """Content key of ``lowered.compile(compile_options)``, or ``None`` if refused."""
    return key_or_refusal(lowered, compile_options=compile_options, dev=dev)[0]


# --------------------------------------------------------------------------- #
# Store entries
# --------------------------------------------------------------------------- #
def _record_path(root: Path, key: str) -> Path:
    return root / _LAYOUT / "keys" / key[:2] / f"{key}.json"


def _blob_path(root: Path, key: str, blob_sha256: str) -> Path:
    """A blob lives under the key it implements: a record cannot point elsewhere."""
    return root / _LAYOUT / "blobs" / key[:2] / key / f"{blob_sha256}.xlaexec"


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _attestation(record: dict[str, Any]) -> str:
    """Digest binding the program key to the executable and its call contract."""
    return _digest({
        "schema": record.get("schema"),
        "key": record.get("key"),
        "program_sha256": (record.get("components") or {}).get("program_sha256"),
        "blob_sha256": record.get("blob_sha256"),
        "blob_bytes": record.get("blob_bytes"),
        "kept_in_avals": record.get("kept_in_avals"),
        "executable_format": record.get("executable_format"),
        "trace_metadata": record.get("trace_metadata"),
        "retained_module": record.get("retained_module"),
    })


def _read_record(root: Path, key: str) -> tuple[dict[str, Any], Path] | None:
    """The entry of ``key`` under ``root``; ``None`` if absent; raises ``_Invalid``."""
    path = _record_path(root, key)
    try:
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="ascii"))
    except Exception as exc:  # noqa: BLE001
        raise _Invalid(f"record-unreadable: {type(exc).__name__}") from exc
    if not isinstance(record, dict) or record.get("schema") != KEY_SCHEMA:
        raise _Invalid("record-schema")
    components = record.get("components")
    if record.get("key") != key or not isinstance(components, dict) or _digest(components) != key:
        raise _Invalid("record-key")
    if not _is_hex64(record.get("blob_sha256")) or not isinstance(record.get("blob_bytes"), int):
        raise _Invalid("record-blob-pointer")
    if record.get("attestation") != _attestation(record):
        raise _Invalid("record-attestation")
    return record, _blob_path(root, key, record["blob_sha256"])


def _read_verified(path: Path, record: dict[str, Any]) -> tuple[bytes, tuple[int, int]]:
    """Blob bytes read through ONE descriptor, with the inode they came from."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise _Invalid(f"blob-missing: {type(exc).__name__}") from exc
    try:
        stat = os.fstat(fd)
        with os.fdopen(fd, "rb", closefd=False) as fh:
            data = fh.read()
    finally:
        os.close(fd)
    if len(data) != record["blob_bytes"] or hashlib.sha256(data).hexdigest() != record["blob_sha256"]:
        raise _Invalid("blob-integrity")
    return data, (stat.st_dev, stat.st_ino)


def lookup(key: ContentKey | str) -> tuple[dict[str, Any], Path, Path] | None:
    """``(record, blob_path, root)`` of the first root with a verifying record."""
    text = key.key if isinstance(key, ContentKey) else str(key)
    if not _is_hex64(text):
        return None
    for root, _writable in _roots():
        try:
            found = _read_record(root, text)
        except _Invalid:
            continue
        if found is not None and found[1].is_file():
            return found[0], found[1], root
    return None


def _link_new(source: Path, destination: Path) -> bool:
    """Create ``destination`` from ``source`` unless it exists. True if created."""
    try:
        os.link(source, destination)
        return True
    except FileExistsError:
        return False


def _temp_file(destination: Path, data: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    return Path(name)


def _write_new(destination: Path, data: bytes) -> bool:
    """Write ``data`` at ``destination`` unless it exists (complete or absent)."""
    tmp = _temp_file(destination, data)
    try:
        return _link_new(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _replace(destination: Path, data: bytes) -> None:
    tmp = _temp_file(destination, data)
    try:
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _install(source: Path, record: dict[str, Any], destination: Path, *, link: bool) -> dict[str, Any]:
    """Install the verified blob of ``record`` at ``destination``.

    The bytes are hashed from one descriptor. A hard link is kept only if it is
    the hashed inode; otherwise the hashed bytes are written. The installed file
    is then checked: same inode, or same digest. Raises ``_Invalid``.
    """
    data, inode = _read_verified(source, record)
    placed = "copy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".link", dir=destination.parent)
        os.close(fd)
        tmp = Path(name)
        try:
            tmp.unlink()
            os.link(source, tmp)
            stat = os.stat(tmp)
            if (stat.st_dev, stat.st_ino) == inode and stat.st_size == len(data):
                os.replace(tmp, destination)
                placed = "hardlink"
        except OSError:
            pass
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    if placed == "copy":
        _replace(destination, data)
    stat = os.stat(destination)
    if (stat.st_dev, stat.st_ino) != inode:
        _read_verified(destination, record)  # a copy: hash what is installed
    return {"placed": placed, "bytes": len(data), "data": data}


def link_local(key: ContentKey, pub: dict[str, Any], destination: Path) -> str | None:
    """Install the entry a publication just verified at ``destination`` (read/write root).

    Returns how it was placed, or ``None`` (the caller then writes its own bytes).
    The entry is read and hashed again here: the link is taken from that file.
    """
    try:
        rw, _seeds = store_roots()
        if rw is None or pub.get("root") != str(rw):
            return None
        found = _read_record(rw, key.key)
        if found is None or found[0]["blob_sha256"] != pub.get("blob_sha256"):
            return None
        return _install(found[1], found[0], destination, link=True)["placed"]
    except Exception:  # noqa: BLE001 - the caller writes the bytes it holds
        return None


def _local_blob_ok(path: Path, blob_sha256: str) -> bool:
    try:
        return _file_sha256(path) == blob_sha256
    except OSError:
        return False


def _kept_avals(meta: Any) -> list[dict[str, Any]] | None:
    """Shape/dtype of the executable's parameters, in order."""
    avals = getattr(meta, "in_avals", None)
    kept = getattr(meta, "kept_var_idx", None)
    if not avals or kept is None:
        return None
    try:
        return [{"shape": [int(d) for d in avals[i]["shape"]], "dtype": str(avals[i]["dtype"])}
                for i in kept]
    except Exception:  # noqa: BLE001
        return None


def _lock_path(root: Path, key: str) -> Path:
    return root / _LAYOUT / "locks" / key[:2] / f"{key}.lock"


@contextlib.contextmanager
def _publication_lock(root: Path, key: str):
    """Exclusive lock for publishing or repairing ``key`` in ``root`` (flock; released on exit or death).

    Everything that can create, repair or set aside an entry runs inside it, on a fresh read
    of the entry, so no decision rests on an observation another writer has since invalidated.
    The same thread may re-enter. Raises :class:`_Busy` after ``_LOCK_TIMEOUT_S``.
    """
    me = threading.get_ident()
    with _LOCKS_GUARD:
        held = _LOCKS_HELD.get(key)
        if held is not None and held[0] == me:
            held[1] += 1
            fd = None
        else:
            held = None
    if held is not None:
        try:
            yield
        finally:
            with _LOCKS_GUARD:
                held[1] -= 1
        return
    path = _lock_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise _Busy(f"timeout after {_LOCK_TIMEOUT_S:.0f} s") from None
                time.sleep(0.05)
        with _LOCKS_GUARD:
            _LOCKS_HELD[key] = [me, 1, fd]
        try:
            yield
        finally:
            with _LOCKS_GUARD:
                _LOCKS_HELD.pop(key, None)
    finally:
        os.close(fd)  # closing the descriptor releases the flock


def _still_invalid(root: Path, key: str) -> bool:
    """Fresh check of the entry: True only if it fails verification right now."""
    try:
        found = _read_record(root, key)
        if found is None:
            return False  # nothing there: nothing to set aside
        _read_verified(found[1], found[0])
        return False
    except _Invalid:
        return True


def _set_aside(root: Path, key: str, reason: str) -> bool:
    """Move an entry that fails verification NOW out of the way (read/write root, under the lock).

    The entry is verified again immediately before the rename: a record that another writer
    made valid since the failure was observed is never renamed.
    """
    del reason
    if not _still_invalid(root, key):
        return False
    path = _record_path(root, key)
    try:
        os.replace(path, path.with_name(f"{path.name}.invalid.{os.getpid()}.{time.time_ns()}"))
        return True
    except OSError:
        return False


def _live_backend() -> str | None:
    try:
        import jax

        device = jax.devices()[0]
        return f"{device.platform}:{getattr(device, 'device_kind', None)}"
    except Exception:  # noqa: BLE001
        return None


def _event(name: str, key: ContentKey | None, state: str, **fields: Any) -> None:
    """One log line per lookup, publication, refusal or unshared execution (key may be absent)."""
    if len(_EVENTS) < _MAX_EVENTS:
        _EVENTS.append({"name": str(name), "key": key.key if key else None,
                        "program_sha256": key.program_sha256 if key else None, "state": state,
                        "backend": (key.info.get("backend") if key else None) or _live_backend(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields})


def note_local_hit(name: str, key: ContentKey | None) -> None:
    """A program ran from this cache's own exact-HLO address although the store did not serve it.

    That binary is not known to be the store's: the work is logged as unshared so that a gate
    over the event log cannot overlook it.
    """
    try:
        if key is None or _KEY_STATE.get(key.key, {}).get("state") == "placed":
            return
        _event(name, key, "local:exact-hlo-hit", executed_blob_sha256=None, shared=False)
    except Exception:  # noqa: BLE001
        pass


def note_init_kernel(kernel: str, *, hit: bool, lowered: bool, cheap_key: bool) -> None:
    """Init-kernel call log (store on only): a warm run must show hits without lowering."""
    try:
        if enabled() and len(_INIT_KERNELS) < _MAX_EVENTS:
            _INIT_KERNELS.append({"kernel": str(kernel), "hit": bool(hit), "lowered": bool(lowered),
                                  "cheap_key": bool(cheap_key), "backend": _live_backend()})
    except Exception:  # noqa: BLE001
        pass


def report() -> dict[str, Any]:
    """Every lookup and publication of this process, with the binary that runs.

    One event per (program variant, step): ``executed_blob_sha256`` names the
    executable the call that follows runs (``None`` for a miss, which compiles and
    then adds a publication event).
    """
    return {"events": [dict(event) for event in _EVENTS], "config_errors": list(CONFIG_ERRORS),
            "init_kernels": [dict(event) for event in _INIT_KERNELS]}


def publish(key: ContentKey, blob: bytes, meta: Any, *, name: str = "") -> dict[str, Any]:
    """Offer ``blob`` as the executable of ``key``. First valid writer wins. Never raises.

    With a read/write root the whole decision (scan, repair, set aside, write) runs under the
    key's publication lock on a fresh read of the entry. ``state``:

    * ``published``: this blob is now the entry (written, read back, verified);
    * ``exists-same``: a verified entry already holds these bytes; ``repaired``:
      its blob file was corrupt and has been rewritten from these bytes;
    * ``exists-differs``: a verified entry holds another binary; ``winner`` (not
      part of the status the caller records) carries its bytes and record;
    * ``exists-refused``: a verified entry exists but this process refused that
      very blob or had to compile after placing it; it must not be adopted;
    * ``read-only`` | ``refused:<why>`` | ``error:<why>``.
    """
    out: dict[str, Any] = {"state": None, "key": key.key, "program_sha256": key.program_sha256}
    try:
        # Hash the bytes that are written, never a digest carried beside them.
        own_sha = hashlib.sha256(blob).hexdigest()
        out["own_blob_sha256"] = own_sha
        kept = _kept_avals(meta)
        if kept is None:
            out["state"] = "refused:no-call-contract"
            return out
        rw, _seeds = store_roots()
        if rw is None:
            return _publish_scan(key, blob, meta, name, own_sha, kept, out)
        try:
            with _publication_lock(rw, key.key):
                return _publish_scan(key, blob, meta, name, own_sha, kept, out)
        except _Busy as exc:
            out["state"] = f"refused:publication-lock: {exc}"
            return out
    except Exception as exc:  # noqa: BLE001 - the store is an optimisation
        out["state"] = f"error:{type(exc).__name__}: {exc}"
        return out


def _publish_scan(key: ContentKey, blob: bytes, meta: Any, name: str, own_sha: str,
                  kept: list[dict[str, Any]], out: dict[str, Any]) -> dict[str, Any]:
    """Body of :func:`publish`; the caller holds the publication lock when a root is writable."""
    notes: list[str] = []
    for _attempt in range(2):
        for root, writable in _roots():
            try:
                found = _read_record(root, key.key)
                if found is None:
                    continue
                record, blob_path = found
                try:
                    data, _inode = _read_verified(blob_path, record)
                except _Invalid:
                    if writable and record["blob_sha256"] == own_sha:
                        _replace(blob_path, blob)  # same digest: put the right bytes back
                        _read_verified(blob_path, record)
                        out.update(state="repaired", blob_sha256=own_sha, blob_path=str(blob_path),
                                   root=str(root), notes=notes)
                        return out
                    raise
            except _Invalid as exc:
                notes.append(f"{root}: {exc}")
                if writable and _set_aside(root, key.key, str(exc)):
                    notes.append(f"{root}: set aside")
                continue
            out.update(blob_sha256=record["blob_sha256"], blob_path=str(blob_path), root=str(root),
                       notes=notes)
            seen = _KEY_STATE.get(key.key) or {}
            if record["blob_sha256"] == own_sha:
                out["state"] = "exists-same"
            elif seen.get("blocking") and seen.get("blob_sha256") == record["blob_sha256"]:
                out["state"] = "exists-refused"
            else:
                out["state"] = "exists-differs"
                out["winner"] = {"record": record, "data": data}
            return out
        rw, _seeds = store_roots()
        if rw is None:
            out.update(state="read-only", notes=notes)
            return out
        record = {
            "schema": KEY_SCHEMA,
            "key": key.key,
            "components": key.components,
            "blob_sha256": own_sha,
            "blob_bytes": len(blob),
            "executable_format": getattr(meta, "executable_format", None),
            "trace_metadata": getattr(meta, "trace_metadata", None),
            "retained_module": getattr(meta, "retained_module", None),
            "kept_in_avals": kept,
            "producer": {
                "name": str(name),
                "legacy_hlo_sha256": getattr(meta, "hlo_sha256", None),
                "cheap_key": getattr(meta, "cheap_key", None),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
            },
        }
        record["attestation"] = _attestation(record)
        blob_path = _blob_path(rw, key.key, own_sha)
        if not blob_path.is_file():
            _write_new(blob_path, blob)
        try:
            _read_verified(blob_path, record)
        except _Invalid:
            _replace(blob_path, blob)  # a stale file under this digest: rewrite it
            _read_verified(blob_path, record)
        data = json.dumps(record, sort_keys=True, indent=1).encode("ascii")
        if _write_new(_record_path(rw, key.key), data):
            out.update(state="published", blob_sha256=own_sha, blob_path=str(blob_path),
                       root=str(rw), notes=notes)
            return out
        # A record appeared although the scan saw none or set one aside: scan once more.
    out.update(state="refused:publication-race", notes=notes)
    return out


def meta_from_lowered(
    lowered: Any, *, hlo_sha256: str | None, blob_sha256: str, record: dict[str, Any],
    dev: Any | None = None,
) -> Any:
    """Call contract of a stored executable, derived from the consumer's lowering.

    Pytrees and kept inputs are the consumer's own, so a producer build with other
    class definitions or unused inputs cannot leak into the call.
    """
    from gpuwrf.runtime import aot_executable as aotx

    lowering = lowered._lowering
    if len(getattr(lowering, "const_args", ()) or ()) != 0:
        raise Refused("hoisted-constants")
    kept = tuple(sorted(int(i) for i in lowering.compile_args["kept_var_idx"]))
    in_avals = aotx._compiled_in_aval_records(lowered)
    in_tree = lowered.in_tree
    if not in_avals or len(in_avals) != int(in_tree.num_leaves):
        raise Refused("input-avals-unavailable")
    if dev is None:
        dev = lowering._device_list[0]
    return aotx.AotMeta(
        in_tree=in_tree,
        out_tree=lowered.out_tree,
        kept_var_idx=kept,
        in_avals=in_avals,
        hlo_sha256=hlo_sha256,
        fingerprint=aotx.target_fingerprint(dev),
        blob_sha256=blob_sha256,
        trace_metadata=record.get("trace_metadata"),
        executable_format=record.get("executable_format"),
        retained_module=record.get("retained_module"),
    )


def materialize(
    name: str, key: ContentKey, lowered: Any, *, hlo_sha256: str | None,
    cache_dir: str | Path | None = None, dev: Any | None = None,
) -> dict[str, Any]:
    """Place the stored executable of ``key`` at this cache's exact-HLO address.

    The ordinary loader then finds, verifies and executes it, and the caller
    aliases the cheap key as for any exact-HLO hit. ``state``: ``hit`` |
    ``hit-already-local`` | ``miss`` | ``refused:<why>`` (an entry exists but
    failed a check; remembered for the process) | ``error:<why>``. Never raises.
    """
    out: dict[str, Any] = {"state": "miss", "key": key.key, "program_sha256": key.program_sha256}
    refusals: list[str] = []
    contract_refused: str | None = None
    try:
        if not hlo_sha256:
            out["state"] = "refused:no-legacy-digest"
            return out
        from gpuwrf.runtime import aot_precompile as aotp

        import pickle

        paths = aotp._aot_blob_paths(name, cache_dir, hlo_sha256=hlo_sha256)
        if paths is None:
            out["state"] = "refused:cache-disabled"
            return out
        local_blob, local_meta = paths
        for root, writable in _roots():
            try:
                found = _read_record(root, key.key)
                if found is None:
                    continue
                record, blob_path = found
                meta = meta_from_lowered(lowered, hlo_sha256=hlo_sha256,
                                         blob_sha256=record["blob_sha256"], record=record, dev=dev)
                if _kept_avals(meta) != record.get("kept_in_avals"):
                    raise _Invalid("call-contract")
                state = "hit"
                if (local_blob.is_file() and local_meta.is_file()
                        and _local_blob_ok(local_blob, record["blob_sha256"])):
                    state = "hit-already-local"  # the bytes on disk were hashed just now
                else:
                    out.update({k: v for k, v in _install(
                        blob_path, record, local_blob, link=writable).items() if k != "data"})
                aotp._atomic_write_bytes(
                    local_meta, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
            except (_Invalid, Refused) as exc:
                refusals.append(f"{root}: {exc}")
                if str(exc) == "call-contract":  # that blob is not this program's executable here
                    contract_refused = record["blob_sha256"]
                continue
            out.update(state=state, blob_sha256=record["blob_sha256"], root=str(root),
                       executed_blob_sha256=record["blob_sha256"])
            # Placed: if this process compiles the key after all, that blob did not run here.
            _KEY_STATE[key.key] = {"state": "placed", "blob_sha256": record["blob_sha256"], "blocking": True}
            if refusals:
                out["skipped"] = refusals
            return out
        if refusals:
            out.update(state="refused:" + refusals[0].split(": ", 1)[-1], refusals=refusals)
            _KEY_STATE[key.key] = {"state": "refused", "blob_sha256": contract_refused,
                                   "blocking": contract_refused is not None}
        return out
    except Exception as exc:  # noqa: BLE001 - fall back to compile
        out["state"] = f"error:{type(exc).__name__}: {exc}"
        _KEY_STATE[key.key] = {"state": "refused", "blob_sha256": None, "blocking": False}
        return out


def verify_local(
    name: str, *, blob_sha256: str, cache_dir: str | Path | None = None,
    cheap_key: str | None = None, hlo_sha256: str | None = None,
) -> bool:
    """True iff the local address holds exactly ``blob_sha256`` with its metadata.

    Used before an alias is reported as ready: the alias is a hard link made after
    the bytes were hashed, so the installed file is hashed again. A failing cheap
    key alias is removed so the next process lowers instead of tripping on it.
    """
    try:
        from gpuwrf.runtime import aot_precompile as aotp

        import pickle

        paths = aotp._aot_blob_paths(name, cache_dir, cheap_key=cheap_key, hlo_sha256=hlo_sha256)
        if paths is None:
            return False
        blob_path, meta_path = paths
        ok = _local_blob_ok(blob_path, blob_sha256)
        if ok:
            with open(meta_path, "rb") as fh:
                meta = pickle.load(fh)
            ok = getattr(meta, "blob_sha256", None) == blob_sha256 and (
                cheap_key is None or getattr(meta, "cheap_key", None) == cheap_key)
        if not ok and cheap_key:
            for path in paths:
                try:
                    path.unlink()
                except OSError:
                    pass
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def take_winner(key: ContentKey | None) -> Any | None:
    """The loaded executable of the binary that won ``key``'s publication, once.

    A caller that compiled and then lost the publication must run this instead of
    its own compile, so that every live call of a program runs one binary.
    """
    if key is None:
        return None
    return _WINNER_CALLS.pop(key.key, None)


def adopt_winner(key: ContentKey, pub: dict[str, Any], meta: Any, *, name: str = "") -> tuple[bytes, Any] | None:
    """Validate the winning binary of a lost publication by loading it.

    Returns ``(winner bytes, metadata for it)`` and keeps the loaded executable
    for :func:`take_winner`, or ``None`` (the entry is then refused for this
    process and the caller keeps its own compile).
    """
    winner = pub.pop("winner", None)
    if winner is None:
        return None
    try:
        from gpuwrf.runtime import aot_executable as aotx

        record, data = winner["record"], winner["data"]
        if _kept_avals(meta) != record.get("kept_in_avals"):
            raise _Invalid("call-contract")
        winner_meta = dataclasses.replace(
            meta, blob_sha256=record["blob_sha256"],
            executable_format=record.get("executable_format"),
            trace_metadata=record.get("trace_metadata"),
            retained_module=record.get("retained_module"),
        )
        _WINNER_CALLS[key.key] = aotx.load(data, winner_meta)
        _KEY_STATE[key.key] = {"state": "placed", "blob_sha256": record["blob_sha256"], "blocking": True}
        pub.update(adopted=True, executed_blob_sha256=record["blob_sha256"])
        return data, winner_meta
    except Exception as exc:  # noqa: BLE001 - keep the own compile, loudly
        _KEY_STATE[key.key] = {"state": "refused", "blocking": True,
                               "blob_sha256": (winner.get("record") or {}).get("blob_sha256")}
        pub.update(state="exists-refused", adopted=False,
                   adopt_error=f"{type(exc).__name__}: {exc}")
        return None


def record_publication(name: str, key: ContentKey, pub: dict[str, Any]) -> None:
    """Log which binary the call after this publication runs."""
    executed = pub.get("executed_blob_sha256") or pub.get("own_blob_sha256")
    pub["executed_blob_sha256"] = executed
    pub["shared"] = pub.get("state") in ("published", "exists-same", "repaired", "exists-differs")
    _event(name, key, f"publish:{pub.get('state')}", blob_sha256=pub.get("blob_sha256"),
           executed_blob_sha256=executed, shared=pub["shared"])


def prepare(
    name: str, lowered: Any, hlo_sha256: str | None, *,
    compile_options: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None, dev: Any | None = None,
) -> tuple[ContentKey | None, dict[str, Any] | None]:
    """Miss-path hook: key the lowering and materialize a stored executable.

    Returns ``(None, None)`` when the store is off (no work, no state change) and
    ``(None, status)`` when the case is refused. Pass a returned key to
    ``aot_precompile._serialize_domain_blob`` so a compile that follows a miss is
    published. Never raises.
    """
    try:
        if not enabled() or lowered is None:
            return None, None
        started = time.perf_counter()
        key, refusal = key_or_refusal(lowered, compile_options=compile_options, dev=dev)
        if key is None:
            # A refused case runs unshared: it is logged, so it cannot vanish from a gate.
            _event(name, None, f"lookup:refused:{refusal}", executed_blob_sha256=None, shared=False)
            return None, {"state": f"refused:{refusal}", "key": None, "program_sha256": None}
        status = materialize(name, key, lowered, hlo_sha256=hlo_sha256, cache_dir=cache_dir, dev=dev)
        if not str(status.get("state", "")).startswith("hit") and autotune_seed_enabled():
            status["autotune_import"] = autotune_import(key)  # a compile follows
        status.update(key_seconds=key.info.get("key_seconds"), info=dict(key.info),
                      seconds=time.perf_counter() - started)
        _event(name, key, f"lookup:{status.get('state')}", blob_sha256=status.get("blob_sha256"),
               executed_blob_sha256=status.get("executed_blob_sha256"))
        return key, status
    except Exception as exc:  # noqa: BLE001 - a store problem never stops a run
        try:
            _event(name, None, f"lookup:error:{type(exc).__name__}", executed_blob_sha256=None, shared=False)
        except Exception:  # noqa: BLE001
            pass
        return None, {"state": f"error:{type(exc).__name__}: {exc}", "key": None, "program_sha256": None}


# --------------------------------------------------------------------------- #
# Autotune pool
# --------------------------------------------------------------------------- #
def canonical_xla_flags(raw: str | None = None) -> list[str]:
    """``XLA_FLAGS`` tokens without :data:`NEUTRAL_XLA_FLAGS` (pool scope only).

    XLA reads the flags from a file when the value does not start with ``-``; the
    file's tokens are used then. The content key does not use this function: it
    binds the options XLA parsed.
    """
    if raw is None:
        raw = os.environ.get("XLA_FLAGS", "")
    kept = []
    for token in _xla_flag_tokens(raw):
        name = token.split("=", 1)[0].lstrip("-")
        if name in NEUTRAL_XLA_FLAGS:
            continue
        kept.append(token)
    return kept


def _xla_flag_tokens(raw: str) -> list[str]:
    text = raw.strip()
    if text and not text.startswith("-"):  # XLA's rule: the value names a flags file
        text = Path(text).read_text(encoding="utf-8", errors="replace")
    return text.split()


def autotune_seed_enabled() -> bool:
    raw = os.environ.get(AUTOTUNE_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"} and enabled()


def _live_autotune_dir() -> Path | None:
    """Directory XLA reads and writes per-fusion decisions in, or ``None``."""
    import jax

    base = jax.config.jax_compilation_cache_dir
    flags = jax.config.jax_persistent_cache_enable_xla_caches or ""
    if base and (flags == "all" or _AUTOTUNE_DIR in flags):
        return Path(base) / _AUTOTUNE_DIR
    for token in _xla_flag_tokens(os.environ.get("XLA_FLAGS", "")):
        name, _, value = token.partition("=")
        if name.lstrip("-") == _AUTOTUNE_DIR and value:
            return Path(value)
    return None


def _autotune_pool(root: Path, components: dict[str, Any]) -> Path:
    scope = {name: components.get(name) for name in ("target", "driver", "toolchain", "explicit_autotune")}
    scope["xla_flags"] = canonical_xla_flags()
    return root / _LAYOUT / "autotune" / _digest(scope)


def _copy_missing(source: Path, destination: Path) -> int:
    """Copy decision files absent from ``destination``; never replace one."""
    copied = 0
    if not source.is_dir():
        return 0
    for entry in sorted(source.iterdir()):
        try:
            if not _AUTOTUNE_FILE.fullmatch(entry.name) or (destination / entry.name).exists():
                continue
            if not entry.is_file() or entry.stat().st_size > _AUTOTUNE_MAX_BYTES:
                continue
            copied += bool(_write_new(destination / entry.name, entry.read_bytes()))
        except OSError:
            continue
    return copied


def autotune_import(key: ContentKey) -> dict[str, Any]:
    """Before a compile: bring pooled decisions into the live autotune cache."""
    out: dict[str, Any] = {"imported": 0}
    try:
        live = _live_autotune_dir()
        if live is None:
            return dict(out, reason="no-live-autotune-dir")
        for root, _writable in _roots():
            out["imported"] += _copy_missing(_autotune_pool(root, key.components), live)
        return dict(out, live=str(live))
    except Exception as exc:  # noqa: BLE001 - a missing decision is autotuned again
        return dict(out, reason=f"error:{type(exc).__name__}: {exc}")


def autotune_export(key: ContentKey) -> dict[str, Any]:
    """After a compile: add this cache's new decisions to the pool."""
    out: dict[str, Any] = {"exported": 0}
    try:
        live = _live_autotune_dir()
        rw, _seeds = store_roots()
        if live is None or rw is None:
            return dict(out, reason="no-live-autotune-dir" if live is None else "read-only")
        pool = _autotune_pool(rw, key.components)
        return dict(out, exported=_copy_missing(live, pool), pool=str(pool))
    except Exception as exc:  # noqa: BLE001
        return dict(out, reason=f"error:{type(exc).__name__}: {exc}")
