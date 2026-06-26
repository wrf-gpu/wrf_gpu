"""Cheap-key AOT manifest: locate a serialized executable WITHOUT lowering.

WHY
---
The Step-C eager-loop AOT path (``domain_tree._operational_advance_factory``)
historically computed the lookup address of a serialized ``_advance_chunk_fori``
blob by *lowering* the exact per-chunk call to a StableHLO digest
(``hlo_sha256``). For the 9-nest that lower (trace + lower of the giant fused
body) is ~30-54 min -- so the AOT blob existed but the *load* still paid the wall
cost it was meant to remove (``proofs/v021/parallel_compile``).

This module computes a **cheap_key**: a fast sha256 over the CALL METADATA that
fully determines the lowered HLO of ``_advance_chunk_fori``, with NO lowering.
``cheap_key`` indexes the same on-disk blob the lower-only ``hlo_sha256`` did, so
a warm process locates the blob in microseconds.

THE LOAD-BEARING RISK = KEY COMPLETENESS
----------------------------------------
If ``cheap_key`` misses ANY input baked into the lowered HLO, two distinct
executables can share a key -> wrong blob loads -> SILENT WRONG RESULT. The key
is split into two levels (the GPT-critic SAFETY design, KEY_SCHEMA v2):

* ``program_key`` = the HLO-IDENTITY key, proven 1:1 vs ``hlo_sha256``. The
  lowered HLO of ``_advance_chunk_fori`` (a bare ``@jax.jit``, no static_argnames,
  no donate) is a deterministic function of exactly these classes:

  - (A) function code              -> ``fn_identity_hash``       (component 2)
  - (A') TRANSITIVE traced source  -> ``source_fingerprint_hash`` (component 2b)
         (git HEAD + dirty content digest of the traced ``gpuwrf`` tree, so a
          callee edit invalidates the key even with ``__version__`` unbumped)
  - (B) static namelist aux        -> ``static_config_hash``     (component 3)
        (this IS the jit compile-constant set by construction: it is
         ``OperationalNamelist.tree_flatten()[1]``)
  - (C) traced call avals + MetaTy -> ``carry_aval_hash``        (component 4)
        (shape/dtype/weak_type + sharding/format/_committed/np-vs-jax)
  - (D) HLO-affecting jax config   -> ``program_config_hash``    (component 1a)
  - (E) trace-time GPUWRF_* env    -> ``global_trace_env_hash``  (component 5)
  - (E') RESOLVED import-time env constants -> ``module_const_env_hash`` (comp 5b)

* ``exec_key`` = the ON-DISK BLOB ADDRESS = ``hash(program_key, exec_env)`` where
  ``exec_env`` = target + compile options (XLA flags / PGLE / opt effort). Same
  StableHLO can serialize to a DIFFERENT executable under these, so the blob
  address is MORE specific than the program identity. ``cheap_key`` IS ``exec_key``.

``n_steps`` / ``cadence`` are ``jnp.asarray``'d to TRACED int32 inside the fori
body (``operational_mode.py``), so they affect only avals (component 4), never
the HLO content -- they MUST be excluded from the static key (including them
would silently fragment the cache and re-introduce the 30-min cost as "warm but
slow"). The date scalars (``time_utc`` / ``noahmp_julian`` / ``noahmp_yearlen``)
ride in a date-BLIND ``_DateClockAux`` holder (#114) and flow via the traced
``clock_base`` (#91), so they are dead inside the compiled fn and MUST hash to a
fixed sentinel (hashing them would re-fork the cache per date).

Completeness is PROVEN, not asserted: ``tests/test_aot_cheap_key.py`` lowers the
REAL ``_advance_chunk_fori`` on CPU across a determinant matrix and asserts the
``cheap_key <-> hlo_sha256`` map is INJECTIVE (one key never maps to two HLOs)
and that a deliberately-incomplete key is CAUGHT (collision detected).

CANONICALIZATION
----------------
Every hash is sha256 over a deterministic, tag-prefixed byte walk -- NEVER
Python ``hash()``/``id()`` (PYTHONHASHSEED-salted, process-local). Floats ->
``struct.pack('<d')`` (exact -0.0/nan/precision). Arrays -> shape + dtype + raw
bytes. ``_StaticHolder`` is content-hashed by the WRAPPED value's bytes (its
``__hash__`` is ``id()`` = process-local = useless cross-process; reusing it
would make every fresh run miss). ``_DateClockAux`` -> fixed sentinel.

NUMERICS / SAFETY
-----------------
**Numerically inert.** The key only LOCATES a blob; the blob's bytes are the same
optimised executable a cold compile builds (kill-gate proved byte-identity). Every
helper is fail-open: any error computing a key returns ``None`` so the caller
compiles normally -- never wrong, only slower.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib
import os
import struct
from pathlib import Path
from typing import Any

import jax

__all__ = [
    "KEY_SCHEMA",
    "cheap_key",
    "program_key",
    "exec_key",
    "static_config_hash",
    "carry_aval_hash",
    "global_trace_env_hash",
    "version_fingerprint_hash",
    "fn_identity_hash",
    "source_fingerprint_hash",
    "module_const_env_hash",
    "canonical_digest",
    "HLO_AFFECTING_ENV_DENYLIST",
    "HLO_INERT_ENV_PREFIXES",
    "IMPORT_TIME_ENV_CONSTANTS",
    "TRACE_REACHABLE_ENV_SCAN_ROOTS",
]

# Bump this tag whenever the key COMPOSITION changes (so an old blob keyed under
# the previous scheme is never located by a new-scheme key). It is also folded
# into the key itself, so the bump alone forces a clean miss + recompile.
#
# v2 (2026-06-25, GPT-critic SAFETY closure):
#   * adds a TRANSITIVE source fingerprint (git HEAD + content digest of the
#     traced ``src/gpuwrf`` tree) so a code edit in ANY traced callee invalidates
#     the key even when ``__version__`` is unbumped (was the P0-2 silent-wrong-
#     result hole);
#   * adds RESOLVED import-time env-constant capture (e.g.
#     ``boundary_apply.NORMAL_BDY_RELAX_STRENGTH``) so an env mutation AFTER import
#     cannot desync the key from the baked HLO constant (P1-5);
#   * adds compile-option / JAX-config determinants to the exec key (P1-4);
#   * splits ``program_key`` (HLO identity, proven 1:1 vs hlo_sha256) from
#     ``exec_key`` (the on-disk blob address = program_key + target/compile env).
KEY_SCHEMA = "GPUWRF-AOTKEY-v2"


# --------------------------------------------------------------------------- #
# Canonical, process-stable content digest (NEVER Python hash()/id()).
# --------------------------------------------------------------------------- #
def _upd(h: "hashlib._Hash", tag: bytes, payload: bytes = b"") -> None:
    """Length-prefixed write so concatenations are unambiguous (no aliasing)."""
    h.update(len(tag).to_bytes(4, "little"))
    h.update(tag)
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)


def _walk(h: "hashlib._Hash", obj: Any, depth: int) -> None:
    if depth > 64:
        # Defensive: refuse pathologically deep / cyclic structures rather than
        # recurse forever. The caller treats a raised error as fail-open.
        raise ValueError("aot cheap_key: structure too deep to canonicalize")

    if obj is None:
        _upd(h, b"N")
        return
    # bool BEFORE int (bool is an int subclass).
    if isinstance(obj, bool):
        _upd(h, b"b", b"1" if obj else b"0")
        return
    if isinstance(obj, int):
        _upd(h, b"i", str(obj).encode())
        return
    if isinstance(obj, float):
        _upd(h, b"f", struct.pack("<d", obj))
        return
    if isinstance(obj, complex):
        _upd(h, b"c", struct.pack("<dd", obj.real, obj.imag))
        return
    if isinstance(obj, str):
        _upd(h, b"s", obj.encode("utf-8"))
        return
    if isinstance(obj, (bytes, bytearray)):
        _upd(h, b"y", bytes(obj))
        return

    cls_name = type(obj).__name__

    # Code objects: hash their bytecode + RECURSE into co_consts (which can hold
    # nested code objects). NEVER repr() them -- a code object's repr embeds its
    # ``id()`` memory address (process-salted = R2). This makes fn_identity_hash
    # process-stable even when the traced body has nested closures/comprehensions.
    import types as _types

    if isinstance(obj, _types.CodeType):
        _upd(h, b"code")
        _upd(h, b"co_code", bytes(getattr(obj, "co_code", b"")))
        _upd(h, b"co_names", str(getattr(obj, "co_names", ())).encode())
        _upd(h, b"co_varnames", str(getattr(obj, "co_varnames", ())).encode())
        _walk(h, tuple(getattr(obj, "co_consts", ())), depth + 1)
        return

    # The two project sentinels MUST be handled before the generic paths so the
    # key has the correct (in)variance: date-blind clock aux -> fixed sentinel;
    # _StaticHolder -> CONTENT of the wrapped value (NOT its id()-based hash).
    if cls_name == "_DateClockAux":
        # #114: date-blind. Two namelists differing only in date MUST share a key.
        _upd(h, b"DATECLOCK")
        return
    if cls_name == "_StaticHolder":
        _upd(h, b"H")
        _walk(h, getattr(obj, "value", None), depth + 1)
        return

    # Array-like (jax.Array, numpy ndarray, ShapeDtypeStruct-with-data). Content
    # bytes when materializable, else shape+dtype only (e.g. ShapeDtypeStruct).
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None and not isinstance(obj, type):
        import numpy as np

        _upd(h, b"a")
        _upd(h, b"shape", str(tuple(shape)).encode())
        _upd(h, b"dtype", str(dtype).encode())
        try:
            host = np.ascontiguousarray(np.asarray(obj))
            _upd(h, b"bytes", host.tobytes())
        except Exception:  # noqa: BLE001 - abstract leaf: shape+dtype is enough
            _upd(h, b"abstract")
        return

    if isinstance(obj, dict):
        _upd(h, b"d")
        for k in sorted(obj.keys(), key=lambda x: repr(x)):
            _walk(h, k, depth + 1)
            _walk(h, obj[k], depth + 1)
        return

    # NamedTuple (deterministic field order).
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        _upd(h, b"nt", cls_name.encode())
        for field in obj._fields:  # type: ignore[attr-defined]
            _upd(h, b"k", field.encode())
            _walk(h, getattr(obj, field), depth + 1)
        return

    if isinstance(obj, (tuple, list)):
        _upd(h, b"t" if isinstance(obj, tuple) else b"l", str(len(obj)).encode())
        for item in obj:
            _walk(h, item, depth + 1)
        return

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        _upd(h, b"dc", cls_name.encode())
        for field in dataclasses.fields(obj):
            _upd(h, b"k", field.name.encode())
            _walk(h, getattr(obj, field.name), depth + 1)
        return

    # Registered pytree fallback (GridSpec, VerticalCoord, ... that expose
    # tree_flatten). aux + children walked deterministically; this is what
    # content-hashes GridSpec's eta_levels/terrain/metric arrays by BYTES rather
    # than via its salted __hash__.
    flatten = getattr(obj, "tree_flatten", None)
    if callable(flatten):
        try:
            children, aux = flatten()
        except Exception:  # noqa: BLE001 - fall through to repr
            children, aux = None, None
        if children is not None:
            _upd(h, b"pt", cls_name.encode())
            _walk(h, aux, depth + 1)
            _walk(h, tuple(children), depth + 1)
            return

    # Last resort: stable repr (enums, frozen scalar configs without dataclass).
    _upd(h, b"r", (cls_name + ":" + repr(obj)).encode())


def canonical_digest(obj: Any) -> str:
    """sha256 hex of a deterministic, process-stable content walk of ``obj``."""
    h = hashlib.sha256()
    _walk(h, obj, 0)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Component 2b: source_fingerprint (TRANSITIVE traced-source identity) -- P0-2.
#
# fn_identity (component 2) hashes only ``_advance_chunk_fori``'s own bytecode +
# ``__version__``. Editing ANY TRACED CALLEE (dynamics/physics/coupling) changes
# the lowered HLO with the SAME key when ``__version__`` is unbumped -> verify is
# default-OFF -> silent wrong result. This component closes that hole: it makes a
# source change in the trace-reachable tree produce a DIFFERENT key, so a stale
# blob can never be silently reused after a code edit.
#
# Policy (documented for the manager/CI):
#   * Content digest (path + size + sha256 of bytes) of the STATIC IMPORT CLOSURE
#     of the traced body (``_TRACE_ROOT_MODULE``) -- a provable SUPERSET of the
#     modules whose source can change the lowered HLO. Catches an UNCOMMITTED edit
#     to ANY traced callee; INVARIANT to an orchestration/IO edit (domain_tree /
#     nested_pipeline / cli / aot_*) that cannot change the HLO. NO repo-wide
#     ``git HEAD`` (it shifted on ANY commit anywhere = the v0.21.0 9-nest
#     cold->warm key-churn blocker); the content digest covers committed AND
#     uncommitted edits to the scoped files, which is both necessary and
#     sufficient.
#   * If the closure cannot be computed -> fall back to the whole-tree content
#     digest (over-scoping is SAFE: never a wrong load, only an extra miss).
#   * Fail-OPEN: any error -> a stable "unidentified-source" marker is folded in
#     (the key is still a function of *something*; verify-mode + version bump are
#     the backstops). The marker is constant, so it does NOT fragment the cache.
# --------------------------------------------------------------------------- #
def _gpuwrf_package_root() -> Path | None:
    """The on-disk root dir of the importable ``gpuwrf`` package, or ``None``."""
    try:
        import gpuwrf as _g

        f = getattr(_g, "__file__", None)
        if not f:
            return None
        return Path(f).resolve().parent
    except Exception:  # noqa: BLE001 - fail-open
        return None


# The traced body whose lowered HLO this whole key identifies. The source
# fingerprint scope is the STATIC IMPORT CLOSURE rooted here: a module can only
# contribute ops to the lowered ``_advance_chunk_fori`` HLO if it is reachable
# from this module's transitive imports, so the closure is a provable SUPERSET of
# the trace-reachable set -- it cannot MISS an HLO-affecting source edit (the
# silent-wrong-result risk). Orchestration/IO modules (``domain_tree``,
# ``nested_pipeline``, ``cli``, ``aot_*``) import the traced body but are NOT
# imported BY it, so they are correctly OUTSIDE the closure: an edit to them does
# not change the lowered HLO and MUST NOT shift the cache key (the v0.21.0 9-nest
# cold->warm blocker -- a concurrent ``domain_tree.py`` edit shifted all 9 keys
# while the HLO was invariant).
_TRACE_ROOT_MODULE = "gpuwrf.runtime.operational_mode"


def _module_to_source_path(module: str, src_root: Path) -> Path | None:
    """Resolve a dotted ``gpuwrf...`` module to its ``.py`` file under ``src_root``.

    Tries ``a/b/c.py`` then the package ``a/b/c/__init__.py``. Returns ``None`` if
    neither exists (e.g. a C-extension or a stale import)."""
    parts = module.split(".")
    cand = src_root.joinpath(*parts).with_suffix(".py")
    if cand.is_file():
        return cand
    pkg_init = src_root.joinpath(*parts) / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _gpuwrf_imports_in_file(path: Path, this_module: str) -> set[str]:
    """All ``gpuwrf*`` modules imported by ``path`` (module- AND function-level).

    Walks the FULL AST (``ast.walk``), not just module-level statements, so a
    function-body ``from gpuwrf.x import y`` (lazy import inside a traced callee)
    is included -- those still contribute traced code. Relative imports are
    resolved against ``this_module``'s package. Process-stable (pure AST, no
    execution, no ``id()``/``hash()``)."""
    out: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gpuwrf" or alias.name.startswith("gpuwrf."):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and (
                    node.module == "gpuwrf" or node.module.startswith("gpuwrf.")
                ):
                    out.add(node.module)
                    # `from gpuwrf.pkg import submod` may name a submodule, not an
                    # attribute -> also fold the candidate dotted submodule so the
                    # closure follows package re-exports.
                    for alias in node.names:
                        out.add(f"{node.module}.{alias.name}")
            else:
                # Relative import: drop `level` trailing components of this_module
                # (a module path; `level==1` -> same package).
                base_parts = this_module.split(".")
                trim = node.level
                base = ".".join(base_parts[: max(1, len(base_parts) - trim)])
                if node.module:
                    out.add(f"{base}.{node.module}")
                    for alias in node.names:
                        out.add(f"{base}.{node.module}.{alias.name}")
                else:
                    out.add(base)
                    for alias in node.names:
                        out.add(f"{base}.{alias.name}")
    return {m for m in out if m == "gpuwrf" or m.startswith("gpuwrf.")}


def _trace_reachable_source_files(pkg_root: Path) -> list[Path] | None:
    """The static import closure of :data:`_TRACE_ROOT_MODULE` as ``.py`` paths.

    BFS over ``gpuwrf*`` imports starting from the traced-body module. The result
    is the SUPERSET of modules whose code can reach the lowered HLO, computed by
    AST only (process-stable). Returns ``None`` if the root module cannot be
    resolved (caller fails open to the whole-tree digest -- NEVER under-scopes)."""
    src_root = pkg_root.parent  # .../src   (pkg_root = .../src/gpuwrf)
    root_path = _module_to_source_path(_TRACE_ROOT_MODULE, src_root)
    if root_path is None:
        return None
    seen: set[str] = set()
    files: set[Path] = set()
    frontier: list[str] = [_TRACE_ROOT_MODULE]
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_to_source_path(module, src_root)
        if path is None:
            continue
        files.add(path)
        for dep in _gpuwrf_imports_in_file(path, module):
            if dep not in seen:
                frontier.append(dep)
    if not files:
        return None
    return sorted(files)


def _source_tree_content_digest(pkg_root: Path, files: list[Path] | None = None) -> str:
    """sha256 over (relpath, size, sha256(bytes)) of the given ``.py`` ``files``.

    When ``files`` is ``None`` it falls back to EVERY ``.py`` under ``pkg_root``
    (the original whole-tree scope -- used only as the fail-open path if the
    import closure cannot be computed; over-scoping is SAFE, only slower-to-miss).
    Deterministic (sorted by relpath), content-addressed (catches an UNCOMMITTED
    edit a git-HEAD-only fingerprint would miss), and process-stable (no
    ``id()``/``hash()``). Excludes ``__pycache__``."""
    if files is None:
        files = sorted(
            p for p in pkg_root.rglob("*.py") if "__pycache__" not in p.parts
        )
    h = hashlib.sha256()
    for p in files:
        if "__pycache__" in p.parts:
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        try:
            rel = p.relative_to(pkg_root).as_posix()
        except ValueError:
            rel = p.as_posix()
        _upd(h, b"f_rel", rel.encode("utf-8"))
        _upd(h, b"f_size", str(len(data)).encode())
        _upd(h, b"f_sha", hashlib.sha256(data).hexdigest().encode())
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def source_fingerprint_hash() -> str:
    """TRANSITIVE source fingerprint of the TRACE-REACHABLE ``gpuwrf`` modules (P0-2).

    Content digest of the static import closure of the traced body
    (:data:`_TRACE_ROOT_MODULE`) -- the provable SUPERSET of modules whose source
    can change the lowered ``_advance_chunk_fori`` HLO. This catches an
    UNCOMMITTED edit to ANY traced callee (dynamics/physics/coupling/contracts)
    while staying INVARIANT to an edit of an orchestration/IO module
    (``domain_tree``/``nested_pipeline``/``cli``/``aot_*``) that cannot change the
    HLO -- the v0.21.0 9-nest cold->warm cache-key stability fix.

    The repo-wide ``git HEAD`` is deliberately NOT folded in: it shifts on ANY
    commit anywhere in the repo (docs, tests, orchestration), re-introducing the
    HLO-irrelevant key churn this fix removes. The scoped content digest already
    catches every committed AND uncommitted edit to a trace-reachable module, so
    it is both necessary and sufficient -- and ``GPUWRF_AOT_VERIFY`` (lower-once +
    HLO-digest compare, fail-closed) remains the correctness backstop that catches
    a genuine HLO divergence the scope could ever miss.

    Cached per process (the source cannot change under a running process). Any
    error folds in a constant "unidentified-source" marker (fail-open; does not
    fragment the cache). If the import closure cannot be computed it falls back to
    the whole-tree digest (over-scoping is SAFE: never a wrong load, only an extra
    miss on an HLO-irrelevant edit)."""
    pkg_root = _gpuwrf_package_root()
    if pkg_root is None:
        return canonical_digest(("source-fingerprint", "unidentified-source"))
    try:
        files = _trace_reachable_source_files(pkg_root)
        payload = {
            # The scope marker records WHICH scope produced the digest so a future
            # scope change (closure <-> whole-tree fallback) forces a clean miss.
            "scope": "trace-closure" if files is not None else "whole-tree-fallback",
            "n_files": len(files) if files is not None else None,
            "content_digest": _source_tree_content_digest(pkg_root, files),
            "pkg_root_name": pkg_root.name,
        }
        return canonical_digest(("source-fingerprint", payload))
    except Exception:  # noqa: BLE001 - fail-open to the constant marker
        return canonical_digest(("source-fingerprint", "unidentified-source"))


# --------------------------------------------------------------------------- #
# Component 5b: module_const_env_hash (RESOLVED import-time env constants) -- P1-5.
#
# Some trace-reachable modules resolve a module-level constant FROM the env AT
# IMPORT (e.g. ``boundary_apply.NORMAL_BDY_RELAX_STRENGTH =
# float(os.environ.get("GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "20.0"))``). The HLO
# bakes the RESOLVED constant, but ``global_trace_env_hash`` reads LIVE
# ``os.environ`` -- so if the env is mutated AFTER import, the key desyncs from
# the baked HLO determinant (silent wrong load). Fix: hash the RESOLVED module
# attribute value (what the HLO actually baked), not the live env string.
#
# Each entry is ``(module_path, attribute_name)``. A CI scanner test
# (``tests/test_aot_cheap_key.py::test_no_uncovered_import_time_env_constants``)
# flags any NEW import-time env-derived module constant in trace-reachable code
# not listed here, so this registry cannot silently drift out of date.
IMPORT_TIME_ENV_CONSTANTS: tuple[tuple[str, str], ...] = (
    ("gpuwrf.coupling.boundary_apply", "NORMAL_BDY_RELAX_STRENGTH"),
)

# Package subtrees whose ``.py`` modules are scanned by the CI env-coverage test
# for import-time env-derived module constants. Trace-reachable physics/dynamics/
# coupling/dycore live here.
TRACE_REACHABLE_ENV_SCAN_ROOTS: tuple[str, ...] = (
    "gpuwrf/coupling",
    "gpuwrf/dynamics",
    "gpuwrf/physics",
    "gpuwrf/nesting",
)


def module_const_env_hash() -> str:
    """Hash the RESOLVED values of import-time env-derived module constants (P1-5).

    For each ``(module, attr)`` in :data:`IMPORT_TIME_ENV_CONSTANTS`, import the
    module and hash the *resolved* attribute value (the constant the HLO baked at
    import). This is env-mutation-proof: it captures what the trace actually used,
    not the live ``os.environ`` (which can differ if the env was changed after the
    module was imported). Missing module/attr -> a sentinel (fail-open; constant,
    so it does not fragment the cache)."""
    resolved: dict[str, Any] = {}
    for mod_path, attr in IMPORT_TIME_ENV_CONSTANTS:
        try:
            mod = importlib.import_module(mod_path)
            resolved[f"{mod_path}:{attr}"] = getattr(mod, attr)
        except Exception:  # noqa: BLE001 - fail-open per entry
            resolved[f"{mod_path}:{attr}"] = "<unresolved>"
    return canonical_digest(("module-const-env", resolved))


# --------------------------------------------------------------------------- #
# Component 1: version_fingerprint (target + toolchain + HLO-affecting config).
# --------------------------------------------------------------------------- #
# HLO-affecting JAX config (these change the LOWERED program -> program_key).
_PROGRAM_JAX_CONFIG_FLAGS = (
    "jax_default_matmul_precision",
    "jax_dynamic_shapes",
    "jax_numpy_dtype_promotion",
    "jax_enable_x64",
    # Sharding/partitioner mode can change the lowered StableHLO placement.
    "jax_use_shardy_partitioner",
)

# Compile-option / executable-environment determinants (same HLO can serialize to
# a DIFFERENT executable under these -> they belong in exec_key, NOT program_key).
# Raw values are folded in conservatively (over-fragmenting blobs is safe;
# under-fragmenting is the silent-wrong-blob risk). See the GPT plan section B.2.
_EXEC_JAX_CONFIG_FLAGS = (
    "jax_exec_time_optimization_effort",
    "jax_memory_fitting_effort",
    "jax_optimization_level",
    "jax_memory_fitting_level",
    "jax_compiler_enable_remat_pass",
    "jax_enable_pgle",
    "jax_pgle_profiling_runs",
    "jax_pgle_aggregation_percentile",
    "jax_compilation_cache_expect_pgle",
    "jax_persistent_cache_enable_xla_caches",
)
_EXEC_ENV_FLAGS = (
    "XLA_FLAGS",
    "LIBTPU_INIT_ARGS",
    "JAX_DISABLE_MOST_OPTIMIZATIONS",
)


def _jax_config_values(names: tuple[str, ...]) -> dict[str, Any]:
    """Best-effort resolved jax.config values for ``names`` (missing -> None)."""
    cfg: dict[str, Any] = {}
    for name in names:
        try:
            cfg[name] = getattr(jax.config, name)
        except Exception:  # noqa: BLE001
            cfg[name] = None
    return cfg


def program_config_hash() -> str:
    """HLO-program-shaping config that is NOT in the namelist/carry/env.

    This is the part of the old ``version_fingerprint`` that actually changes the
    LOWERED program (jax dtype/precision/sharding semantics). It belongs in
    ``program_key`` (the HLO-identity key proven 1:1 vs hlo_sha256). Target +
    compile options are deliberately EXCLUDED here (they shape the serialized
    executable, not the StableHLO) and live in :func:`exec_env_hash`."""
    return canonical_digest({"jax_config": _jax_config_values(_PROGRAM_JAX_CONFIG_FLAGS)})


def exec_env_hash(dev: Any | None = None) -> str:
    """TARGET + compile-option fingerprint = the serialized-executable identity.

    Folds :func:`aot_executable.target_fingerprint` (jaxlib/jax/device/CC/driver/
    x64 + XLA_FLAGS) and :func:`compile_cache.version_cache_tag`
    (gpuwrf+jax+jaxlib+backend) with the EXACT raw compile-option env vars and the
    JAX compile-option config subset. Same StableHLO can serialize to a DIFFERENT
    executable under different XLA flags / PGLE / optimization effort, so these are
    exec-key (blob-address) determinants, NOT program-key determinants. Captured
    explicitly because ``version_cache_tag`` avoids backend init (cannot see
    ``XLA_FLAGS`` set late by the autotune hook)."""
    from gpuwrf.runtime import aot_executable as _aotx
    from gpuwrf.runtime.compile_cache import version_cache_tag

    payload: dict[str, Any] = {
        "target": _aotx.target_fingerprint(dev),
        "version_tag": version_cache_tag(),
        "exec_env": {name: os.environ.get(name, "") for name in _EXEC_ENV_FLAGS},
        "exec_jax_config": _jax_config_values(_EXEC_JAX_CONFIG_FLAGS),
    }
    return canonical_digest(payload)


def version_fingerprint_hash(dev: Any | None = None) -> str:
    """BACK-COMPAT folded fingerprint = ``program_config_hash`` + ``exec_env_hash``.

    Retained so existing callers/tests that fold a single "version fingerprint"
    into a key keep working; it is the union of the program-shaping config and the
    executable environment. The assembled :func:`cheap_key` uses the SPLIT
    components (:func:`program_config_hash` for the HLO identity, :func:`exec_env_hash`
    for the blob address) -- see :func:`program_key` / :func:`exec_key`."""
    return canonical_digest(
        {
            "program_config": program_config_hash(),
            "exec_env": exec_env_hash(dev),
        }
    )


# --------------------------------------------------------------------------- #
# Component 2: fn_identity (code-version-sensitive).
# --------------------------------------------------------------------------- #
def fn_identity_hash(fn: Any) -> str:
    """Hash the function identity + a best-effort code-version stamp.

    qualname alone is unsafe: editing any TRACED callee changes the HLO without
    changing the qualname. This fn's own ``co_code``/``co_consts`` digest +
    ``__version__`` covers an edit to ``_advance_chunk_fori`` ITSELF; the TRANSITIVE
    traced-callee coverage (an edit to dynamics/physics/coupling) is provided by
    :func:`source_fingerprint_hash` (component 2b, folded into ``program_key``), so
    a code edit can no longer silently reuse a stale blob even with ``__version__``
    unbumped. Verify-mode (lower-once) remains a backstop. Fail-open.
    """
    target = getattr(fn, "__wrapped__", fn)
    payload: dict[str, Any] = {
        "module": getattr(target, "__module__", None),
        "qualname": getattr(target, "__qualname__", None),
    }
    try:
        from gpuwrf import __version__ as _gv

        payload["gpuwrf_version"] = _gv
    except Exception:  # noqa: BLE001
        payload["gpuwrf_version"] = None
    code = getattr(target, "__code__", None)
    if code is not None:
        # Walk the code object structurally (co_code + recursive co_consts). The
        # walker handles nested code objects WITHOUT their id()-bearing repr, so
        # this digest is process-stable (PYTHONHASHSEED-independent).
        try:
            payload["code"] = canonical_digest(code)
        except Exception:  # noqa: BLE001
            pass
    return canonical_digest(payload)


# --------------------------------------------------------------------------- #
# Component 3: static_config_hash (the authoritative HLO-static set).
# --------------------------------------------------------------------------- #
def static_config_hash(namelist: Any) -> str:
    """Content-hash ``OperationalNamelist.tree_flatten()[1]`` (the static aux).

    This aux IS the jit compile-constant set BY CONSTRUCTION (see
    ``OperationalNamelist.tree_flatten``), so hashing it captures every static
    HLO-determinant: every scalar physics/dynamics option, the ``grid`` (GridSpec
    arrays content-hashed), ``boundary_config``, every ``_StaticHolder`` bundle
    (content, not id()), and the date-blind ``_DateClockAux`` (sentinel).
    """
    _children, aux = namelist.tree_flatten()
    return canonical_digest(aux)


# --------------------------------------------------------------------------- #
# Component 4: carry_aval_hash (traced call avals -> shape/dtype/weak_type).
# --------------------------------------------------------------------------- #
def _structure_fingerprint(obj: Any) -> str:
    """Date-blind, process-stable fingerprint of a pytree's STRUCTURE only.

    Flattens ``obj``, replaces every leaf with a fixed token, and canonical-walks
    the reconstruction. The walker maps ``_DateClockAux`` to a sentinel and
    ``_StaticHolder`` to its content, so the result captures the None-vs-present
    subtrees / nest-shape treedef variants WITHOUT the ``id()``/date noise that
    ``str(treedef)`` would leak. Fail-open to a num-leaves marker."""
    try:
        flat, treedef = jax.tree_util.tree_flatten(obj)
        placeholder = jax.tree_util.tree_unflatten(treedef, [b"\x00LEAF" for _ in flat])
        return canonical_digest(placeholder)
    except Exception:  # noqa: BLE001
        return canonical_digest(("nleaves-fallback",))


def _leaf_metaty_record(leaf: Any) -> dict[str, Any]:
    """Per-leaf (shape, dtype, weak_type) + JAX MetaTy fields the lowering uses.

    JAX 0.10 lowers through ``MetaTy`` values (``pjit.py``): sharding, format/
    layout, ``_committed``, and NumPy-vs-JAX source can all change the lowered
    program / its placement. The pre-existing key hashed only shape/dtype/weak_type
    (``aot_executable._leaf_aval_record``) -- a gap the GPT plan flagged. This
    record ADDS those MetaTy fields so a sharding/commitment/layout change cannot
    silently reuse a blob compiled under a different placement. Every field is
    best-effort (missing -> None); fail-open."""
    from gpuwrf.runtime import aot_executable as _aotx

    rec = dict(_aotx._leaf_aval_record(leaf))

    def _safe(getter) -> Any:
        try:
            return getter()
        except Exception:  # noqa: BLE001 - best-effort metaty capture
            return None

    # Sharding identity (canonical string; OK to be coarse, only must DISTINGUISH).
    sharding = getattr(leaf, "sharding", None)
    rec["sharding"] = _safe(lambda: str(sharding)) if sharding is not None else None
    # format / layout object (placement/layout the executable was compiled for).
    fmt = getattr(leaf, "format", None)
    if fmt is None:
        fmt = getattr(leaf, "layout", None)
    rec["format"] = _safe(lambda: str(fmt)) if fmt is not None else None
    # _committed: a committed array pins device placement (changes lowering).
    rec["committed"] = _safe(lambda: bool(getattr(leaf, "_committed")))
    # NumPy-vs-JAX source: a host np.ndarray and a jax.Array can trace differently.
    rec["is_jax_array"] = _safe(lambda: isinstance(leaf, jax.Array))
    # Placement EQUIVALENCE CLASS (the ACTUAL key determinant -- see _placement_class).
    # The raw MetaTy strings above are RETAINED for introspection/back-compat but are
    # NOT hashed by carry_aval_hash; the canonical class is. This is what makes a
    # prewarm ShapeDtypeStruct carry and the concrete runtime carry it represents
    # agree on the key while still distinguishing a genuinely different placement.
    rec["placement_class"] = _placement_class(rec)
    return rec


def _placement_class(rec: dict[str, Any]) -> str:
    """Coarse, REPRESENTATION-INVARIANT placement equivalence class for a leaf.

    THE BUG this closes: the parallel-prewarm worker keys a domain off a
    ``ShapeDtypeStruct`` carry (``aot_precompile._to_shape_dtype_tree`` -- no
    device buffers, picklable) while the eager warm loop keys off the CONCRETE
    runtime carry. ``_to_shape_dtype_tree`` lowers to the IDENTICAL HLO (proven:
    same ``hlo_sha256`` for shape-dtype vs concrete on GPU), so the cheap_key MUST
    match too. But the raw MetaTy strings DIFFER between the two representations
    (an abstract leaf has ``sharding=None, committed=None, is_jax_array=False``; a
    concrete single-device leaf has ``sharding=SingleDeviceSharding(...),
    committed=False, is_jax_array=True``) -> different key -> the prewarm-written
    blob is unloadable by the eager loop (its whole purpose defeated).

    The fix maps the COMMON runtime case (abstract OR uncommitted single-device
    default placement) to a single ``"default"`` class -- exactly the cases that
    lower to the same StableHLO for this NON-donated, single-device jit. A
    genuinely DIFFERENT placement (an explicitly ``committed`` array, or a
    multi-device / named / positional sharding) maps to a distinct class, so the
    GPT-critic SAFETY intent (never silently reuse a blob compiled under a
    different placement) is preserved. Fail-open: any ambiguity -> ``"default"``
    (the abstract-tree class), so a cross-process key can never be MORE specific
    than the prewarm tree could express.
    """
    committed = rec.get("committed")
    sharding = rec.get("sharding")
    # Genuinely committed array -> distinct class (pins device placement).
    if committed is True:
        return f"committed:{sharding}"
    # Single-device / unspecified sharding == the default the abstract tree and an
    # uncommitted runtime array share. Anything mentioning a real partition spec
    # (mesh/named/positional/GSPMD sharding) is a DIFFERENT placement.
    if sharding is None:
        return "default"
    s = str(sharding)
    if s.startswith("SingleDeviceSharding"):
        return "default"
    # A non-single-device sharding genuinely changes the lowered placement.
    return f"sharded:{s}"


def carry_aval_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Hash the call STRUCTURE + per-leaf (shape, dtype, weak_type, MetaTy) of a call.

    Flattens ``(args, kwargs)`` in JAX call order (reusing
    ``aot_executable._flatten_call``) and abstractifies each leaf without
    materializing device arrays. Captures: every State/scratch array shape+dtype
    (fp32-vs-fp64 distinguished via dtype), the None-vs-present subtrees that
    change the treedef (clock_base, optional carry subtrees), the leading-dim
    nest-shape variants, the s32/s64 weak-type class, AND a REPRESENTATION-INVARIANT
    placement equivalence class (:func:`_placement_class`) the lowering path
    consumes. No XLA, no lowering.

    PLACEMENT INVARIANCE (the cross-process determinant): the raw MetaTy strings
    (``sharding``/``format``/``committed``/``is_jax_array``) are DROPPED from the
    hash and replaced by ``placement_class``. The raw strings differ between a
    prewarm ``ShapeDtypeStruct`` carry and the concrete runtime carry it abstracts
    to (both lower to the IDENTICAL HLO), so hashing them fragmented the key per
    representation -- a prewarm-written blob was unloadable by the eager loop. The
    canonical class collapses the abstract-tree case and the uncommitted
    single-device runtime case to one ``"default"`` value while still
    distinguishing a genuinely committed / multi-device-sharded placement.

    NOTE 1: this DELIBERATELY includes the ``kwargs`` keys (``n_steps``/``cadence``)
    in the flatten order but their VALUES never reach the HLO -- they are traced
    int32 inside the body, so two calls differing only in n_steps/cadence have the
    same avals and thus the same carry_aval_hash. They are correctly NOT a
    fragmenting determinant; the identity-proof asserts this.

    NOTE 2: the namelist argument (``args[1]``) is EXCLUDED from the structure
    fingerprint -- its full static identity is already in
    :func:`static_config_hash` (comp 3), and re-walking its treedef here would
    re-leak the date (the #114 trap) via the date-bearing aux that survives a
    ``tree_unflatten``. Its TRACED children still contribute to ``leaf_records``
    (shapes/dtypes only), so nothing HLO-relevant is dropped.
    """
    from gpuwrf.runtime import aot_executable as _aotx

    flat, _in_tree = _aotx._flatten_call(args, kwargs)
    # Hash a REPRESENTATION-INVARIANT projection of each leaf record: the HLO
    # determinants (shape/dtype/weak_type) + the canonical placement class. The raw
    # process-/representation-variant MetaTy strings (sharding/format/committed/
    # is_jax_array) are deliberately EXCLUDED here -- they differ between a prewarm
    # ShapeDtypeStruct carry and the concrete runtime carry that lowers to the SAME
    # HLO, and hashing them broke the cross-process / prewarm-vs-eager key match.
    def _hashable_leaf(rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "shape": rec.get("shape"),
            "dtype": rec.get("dtype"),
            "weak_type": rec.get("weak_type"),
            "placement_class": rec.get("placement_class"),
        }

    leaf_records = tuple(_hashable_leaf(_leaf_metaty_record(leaf)) for leaf in flat)

    # Structure fingerprint of the NON-namelist args (carry, start, clock_base,
    # ...) + the kwargs keys. args[1] is the namelist (owned by comp 3).
    non_namelist_args = tuple(a for i, a in enumerate(args) if i != 1)
    structure = {
        "args": _structure_fingerprint(non_namelist_args),
        "kwarg_keys": tuple(sorted(kwargs.keys())),
        "n_leaves": len(flat),
    }
    payload = {
        "structure": structure,
        "leaves": leaf_records,
    }
    return canonical_digest(payload)


# --------------------------------------------------------------------------- #
# Component 5: global_trace_env_hash (THE GAP) -- HLO-branching env knobs.
# --------------------------------------------------------------------------- #
# Env vars that are read at module-import / infra / IO time and do NOT branch the
# traced HLO. Everything ELSE matching GPUWRF_* is treated as a potential
# HLO-determinant and folded into the key (fail-SAFE: an unknown new knob is
# captured automatically rather than silently dropped). This is the robust answer
# to "did we enumerate every trace-time env read?": we do not enumerate the
# INCLUDE set (which drifts as physics is edited); we enumerate only the small,
# stable EXCLUDE set of infra knobs.
HLO_AFFECTING_ENV_DENYLIST: frozenset[str] = frozenset(
    {
        # Compile-cache / AOT location + behaviour (location, not HLO content).
        "GPUWRF_JAX_CACHE",
        "GPUWRF_JAX_CACHE_DIR",
        "GPUWRF_JAX_CACHE_LOCK",
        "GPUWRF_NESTED_AOT",
        "GPUWRF_AOT_VERIFY",
        "GPUWRF_NESTED_DEFUSE_COMPILE",
        "GPUWRF_NESTED_PARALLEL_COMPILE",
        "GPUWRF_NESTED_PARALLEL_VERIFY",
        "GPUWRF_NESTED_FUSE",
        # Scratch / IO / runtime-process knobs.
        "GPUWRF_SCRATCH",
        "GPUWRF_TMPDIR",
        "GPUWRF_KEEP_SCRATCH",
        "GPUWRF_ALLOCATOR",
        "GPUWRF_HOST_RAM_GUARD",
        "GPUWRF_PROFILE",
        "GPUWRF_PROOF_WRITE",
        "GPUWRF_OUTPUT_DIR",
        "GPUWRF_TRAINING_OUTPUT_SUBSET",
        "GPUWRF_WRF_ROOT",
        # GPU-lock bookkeeping injected by scripts/with_gpu_lock.sh. These are
        # PROCESS-LOCAL infra (the held flag, the lock fd/file, and a per-invocation
        # UNIQUE token + label) with ZERO effect on the traced HLO. They MUST be
        # excluded or every lock-wrapped process computes a DIFFERENT cheap_key
        # (GPUWRF_GPU_LOCK_TOKEN/LABEL change every invocation) -> the warm run
        # looks under a key the cold run never wrote -> fallback:missing. This is
        # the GPU-only cross-process break the CPU tests missed (the CPU suite is
        # not lock-wrapped, so the env was stable). See the prefix guard below for
        # the durable backstop against future GPUWRF_GPU_LOCK_* vars.
        "GPUWRF_GPU_LOCK_HELD",
        "GPUWRF_GPU_LOCK_FD",
        "GPUWRF_GPU_LOCK_FILE",
        "GPUWRF_GPU_LOCK_HOLDER_FILE",
        "GPUWRF_GPU_LOCK_LABEL",
        "GPUWRF_GPU_LOCK_TOKEN",
        # The loop-mode knob is captured EXPLICITLY below (only 'fori' is
        # AOT-eligible) so it is in the denylist here to avoid double-count.
        "GPUWRF_ADVANCE_CHUNK_LOOP",
    }
)

# Infra/bookkeeping ``GPUWRF_*`` PREFIXES that are NEVER HLO determinants. The
# fail-SAFE auto-discovery in :func:`global_trace_env_hash` captures every unknown
# ``GPUWRF_*`` var (so a NEW physics knob is never silently dropped) -- but a
# process-LOCAL infra var (e.g. a per-invocation unique GPU-lock token) is the
# OPPOSITE failure: it fragments the key per process and breaks the cross-process
# warm load. Any var whose name starts with one of these prefixes is treated as
# inert infra and excluded, so future bookkeeping vars under a known-inert
# namespace cannot silently re-break the warm cache. Keep this to namespaces that
# are PROVABLY trace-irrelevant (lock/IPC bookkeeping), never physics/dycore knobs.
HLO_INERT_ENV_PREFIXES: tuple[str, ...] = (
    "GPUWRF_GPU_LOCK_",
)


def global_trace_env_hash() -> str:
    """Hash the resolved set of HLO-branching ``GPUWRF_*`` trace-time env knobs.

    Many traced physics/dycore functions read ``GPUWRF_*`` env vars at TRACE time
    (e.g. ``GPUWRF_MOIST_CQW``, ``GPUWRF_ACOUSTIC_UNROLL``, ``GPUWRF_W_CORIOLIS``,
    ``GPUWRF_THOMPSON_*``, ``GPUWRF_MYNN_*``, ``GPUWRF_THOMAS_UNROLL``, ...) and
    those branch the HLO without appearing in the namelist aux OR the carry avals.
    Missing one is the silent-wrong-load risk, so we capture EVERY currently-set
    ``GPUWRF_*`` var except the small infra denylist (fail-SAFE auto-discovery),
    plus the ``GPUWRF_ADVANCE_CHUNK_LOOP`` loop mode explicitly. Raw values are
    hashed (so an unset var and its default are NOT distinguished here -- they are
    the same trace) only for the vars actually present in the environment.
    """
    captured: dict[str, str] = {}
    for name, value in os.environ.items():
        if not name.startswith("GPUWRF_"):
            continue
        if name in HLO_AFFECTING_ENV_DENYLIST:
            continue
        # Inert infra namespaces (e.g. per-invocation GPU-lock bookkeeping) are
        # process-LOCAL and never branch the HLO; folding them in would fragment
        # the cheap_key per process and break the cross-process warm load.
        if any(name.startswith(prefix) for prefix in HLO_INERT_ENV_PREFIXES):
            continue
        captured[name] = value
    payload = {
        "gpuwrf_env": captured,
        # Loop mode is AOT-gating (only 'fori' is eligible); resolve explicitly.
        "advance_chunk_loop": os.environ.get("GPUWRF_ADVANCE_CHUNK_LOOP", "fori")
        .strip()
        .lower(),
    }
    return canonical_digest(payload)


# --------------------------------------------------------------------------- #
# The assembled keys: program_key (HLO identity) + exec_key (blob address).
# --------------------------------------------------------------------------- #
def program_key(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    namelist: Any,
) -> str:
    """The HLO-IDENTITY key: the determinants of the LOWERED StableHLO program.

    This is the key the identity-proof asserts is 1:1 with ``hlo_sha256``: within
    the Step-C call contract one ``program_key`` must never map to two HLO digests
    (a collision = a missed determinant = the silent-wrong-load bug). It folds
    EVERYTHING that shapes the lowered program and NOTHING that only shapes the
    serialized executable (target / compile options live in :func:`exec_env_hash`):

    * ``KEY_SCHEMA``               -- composition version;
    * ``program_config_hash``      -- HLO-affecting jax config (precision/x64/...);
    * ``fn_identity_hash``         -- ``_advance_chunk_fori`` own bytecode+version;
    * ``source_fingerprint_hash``  -- TRANSITIVE traced-source identity (P0-2);
    * ``static_config_hash``       -- OperationalNamelist static aux (content);
    * ``carry_aval_hash``          -- traced call avals + MetaTy fields;
    * ``global_trace_env_hash``    -- HLO-branching GPUWRF_* trace-time env;
    * ``module_const_env_hash``    -- RESOLVED import-time env constants (P1-5).
    """
    return canonical_digest(
        (
            KEY_SCHEMA,
            "program",
            program_config_hash(),
            fn_identity_hash(fn),
            source_fingerprint_hash(),
            static_config_hash(namelist),
            carry_aval_hash(args, kwargs),
            global_trace_env_hash(),
            module_const_env_hash(),
        )
    )


def exec_key(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    namelist: Any,
    *,
    dev: Any | None = None,
) -> str:
    """The ON-DISK BLOB ADDRESS = ``hash(schema, program_key, exec_env)``.

    Same StableHLO can serialize to a DIFFERENT executable under a different
    target / XLA flags / PGLE / optimization effort, so the blob address is MORE
    specific than the program identity. :func:`exec_env_hash` supplies the
    target+compile-option fingerprint; ``program_key`` supplies the HLO identity."""
    return canonical_digest(
        (
            KEY_SCHEMA,
            "exec",
            program_key(fn, args, kwargs, namelist),
            exec_env_hash(dev),
        )
    )


def cheap_key(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    namelist: Any,
    *,
    dev: Any | None = None,
) -> str | None:
    """The metadata-only blob-lookup key for a ``_advance_chunk_fori`` call.

    Returns the sha256-hex :func:`exec_key` (the on-disk blob address = program
    identity + target/compile-option environment), or ``None`` on ANY error
    (caller falls back to the lower-only digest / compile path -- never wrong,
    only slower). The HLO-identity component is :func:`program_key`, proven 1:1 vs
    ``hlo_sha256`` by ``tests/test_aot_cheap_key.py``; ``cheap_key`` adds the
    exec-environment determinants so a blob compiled under a different
    target/XLA-flag policy is never mis-loaded under the same key."""
    try:
        return exec_key(fn, args, kwargs, namelist, dev=dev)
    except Exception:  # noqa: BLE001 - fail-open: no key -> caller lowers/compiles
        return None
