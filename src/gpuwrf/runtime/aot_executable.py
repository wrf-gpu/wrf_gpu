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
    "fingerprint_matches",
]


def blob_sha256(blob: bytes) -> str:
    """sha256 hex of the serialized executable bytes (blob-integrity check)."""
    return hashlib.sha256(blob).hexdigest()


class AotSerializeError(RuntimeError):
    """Raised by :func:`serialize` when the private AOT API is unavailable.

    Callers MUST treat this as fail-open (skip AOT, fall back to compile).
    """


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
    treats it as fail-open -> skip AOT)."""
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
    except Exception as exc:  # noqa: BLE001
        raise AotSerializeError(f"xla_executable.serialize() raised: {exc}") from exc

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

    in_tree = meta.in_tree
    out_tree = meta.out_tree
    kept = meta.kept_var_idx
    expected_avals = meta.in_avals

    def aot_call(*args: Any, **kwargs: Any) -> Any:
        # JAX 0.10 ``Compiled.call`` uses ``tree_util.tracing_registry.flatten`` on
        # ``(args, kwargs)``. Match that leaf order exactly, including keyword args,
        # so kept_var_idx addresses the same flattened input vector JAX would pass.
        flat, in_tree_now = _flatten_call(args, kwargs)
        _check_call_contract(
            flat,
            in_tree_now,
            in_tree,
            expected_avals,
            kept,
            meta.hlo_sha256,
        )
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
        in_bufs = [jax.device_put(x, dev) for x in selected]
        out_bufs = le.execute(in_bufs)
        return jax.tree_util.tree_unflatten(out_tree, out_bufs)

    # Expose the underlying executable + meta for introspection / warm-hit checks.
    aot_call.loaded_executable = le  # type: ignore[attr-defined]
    aot_call.meta = meta  # type: ignore[attr-defined]
    return aot_call
