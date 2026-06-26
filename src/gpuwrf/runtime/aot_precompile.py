"""Ahead-of-time (AOT) precompile entrypoint for the fixed production grids
(v0.13 speed roadmap #1).

WHY
---
For SHORT runs (the dev/validation regime) XLA compile dominates wall-clock
(~5x the compute on a 1 h validation run). For the *fixed* production grids
(Canary 9/3/1 km, Switzerland) we know the exact graph ahead of time, so we can
pay the compile ONCE and serve every subsequent run a warm executable -- "near-
zero compile at runtime for known grids".

WHAT THIS MODULE PROVIDES
-------------------------
1. :func:`precompile` -- the generic primitive: ``jax.jit(fn).lower(*args,
   **static).compile()``. The act of compiling writes the fully-optimised XLA
   executable into the persistent on-disk compile cache
   (:mod:`gpuwrf.runtime.compile_cache`), keyed by the lowered program's HLO +
   backend + compile flags. A subsequent process that lowers+compiles the SAME
   program (same grid shape, dom count, key flags) gets a disk-read warm hit
   instead of a cold compile.

2. :func:`config_key` + :class:`GridConfig` -- a stable, human-readable key
   built from ``(grid shape, domain count, key flags)`` so callers can index a
   manifest of which production configs have been pre-warmed.

3. :data:`PRODUCTION_GRIDS` + :func:`prewarm` -- a registry of the standard
   production configs and a one-call warmer that, given a *spec provider*
   (a callable supplied by the operational pipeline that returns the concrete
   ``(fn, args, static_kwargs)`` for a named config), compiles each so the cache
   is warm. The spec provider is injected rather than hard-imported so this
   module stays decoupled from State/IO construction and **numerically inert**.

NUMERICS
--------
**Numerically inert.** The cached/AOT executable is *bit-for-bit the identical*
XLA program a cold compile would build for the same HLO + backend + flags --
JAX guarantees this (it is the basis of the persistent compile cache). This
module only changes *when* compilation happens, never *what* is computed. The
proof in ``proofs/v0130/compile_speed.py`` asserts identical results between a
cold compile and an AOT/warm compile.

WHY NOT serialize the Compiled object directly?
-----------------------------------------------
In jaxlib 0.10 the ``Compiled`` stage object is not directly serialisable, and
``jax.export`` serialisation requires the optional ``flatbuffers`` package
(absent here) and round-trips only the StableHLO, not the autotuned executable.
The persistent compile cache is JAX's supported, dependency-free mechanism for
"compile once on this machine, warm-hit forever", and it stores the FULLY
optimised executable (autotuning included). So AOT here == "lower+compile to
warm the persistent cache", which is exactly what gives near-zero load compile.
A serialised-artifact path (ship a precompiled blob) is tracked as a
v0.13-GPU-followup once ``flatbuffers``/export-executable lands.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import shutil
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax

from gpuwrf.runtime.compile_cache import (
    CACHE_STATUS,
    cache_entry_count,
    configure_compilation_cache,
    resolve_cache_dir,
    version_cache_tag,
)

__all__ = [
    "GridConfig",
    "PRODUCTION_GRIDS",
    "PrecompileResult",
    "config_key",
    "precompile",
    "prewarm",
    "prewarm_one",
    "load_spec_provider",
    "pack_cache",
    "unpack_cache",
    "cache_artifact_info",
    "DomainCompileSpec",
    "build_defused_specs",
    "default_parallel_workers",
    "prewarm_defused_nest",
    "aot_dir",
    "load_domain_blob",
    "cheap_key_is_quarantined",
    "quarantine_cheap_key",
    "main",
]


@dataclass(frozen=True)
class GridConfig:
    """Identity of a fixed production forecast configuration.

    Only the fields that change the compiled HLO belong here: grid shape, domain
    count, and the boolean/integer physics-suite flags that branch the graph at
    trace time. These map 1:1 to the static cache key of the operational jit, so
    two runs with the same :class:`GridConfig` share one compiled executable.
    """

    name: str
    nx: int
    ny: int
    nz: int
    n_domains: int = 1
    dt_s: float = 10.0
    # Key graph-shaping flags (subset of OperationalNamelist statics that branch
    # the traced program). Stored as a sorted tuple of (name, value) for a stable
    # hashable key; callers pass whatever subset is load-bearing for their graph.
    flags: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def with_flags(self, **kwargs: Any) -> "GridConfig":
        """Return a copy with ``flags`` set from keyword args (sorted, stable)."""
        merged = dict(self.flags)
        merged.update(kwargs)
        return GridConfig(
            name=self.name,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            n_domains=self.n_domains,
            dt_s=self.dt_s,
            flags=tuple(sorted(merged.items())),
        )


def config_key(config: GridConfig) -> str:
    """Stable, human-readable key for a :class:`GridConfig`.

    Used to index a pre-warm manifest and to label proof artifacts. NOT the XLA
    cache key (that is the HLO hash, owned by JAX); this is the project-level
    config identity so we can answer "is config X pre-warmed?".
    """
    flag_part = ",".join(f"{k}={v}" for k, v in config.flags)
    return (
        f"{config.name}__{config.nx}x{config.ny}x{config.nz}"
        f"__dom{config.n_domains}__dt{config.dt_s:g}"
        + (f"__{flag_part}" if flag_part else "")
    )


# Standard production grids (shape/dom metadata only; the operational pipeline
# supplies the concrete State/namelist when pre-warming). Canary nest is the
# validated 9/3/1 km chain; Switzerland is the fp64 single-GPU 128^2 ceiling
# case. Extend as new fixed grids are productionised.
PRODUCTION_GRIDS: tuple[GridConfig, ...] = (
    GridConfig(name="canary-d01-9km", nx=100, ny=100, nz=45, n_domains=1, dt_s=54.0),
    GridConfig(name="canary-d02-3km", nx=139, ny=124, nz=45, n_domains=1, dt_s=18.0),
    GridConfig(name="canary-d03-1km", nx=160, ny=139, nz=45, n_domains=1, dt_s=6.0),
    GridConfig(name="switzerland-128", nx=128, ny=128, nz=45, n_domains=1, dt_s=10.0),
)


@dataclass(frozen=True)
class PrecompileResult:
    """Outcome of one :func:`precompile` call."""

    key: str
    compile_seconds: float
    cache_dir: str | None
    cache_enabled: bool
    cache_hit: bool | None = None  # None => not measured (single-shot)
    error: str | None = None
    hlo_sha256: str | None = None


def precompile(
    fn: Callable[..., Any],
    *args: Any,
    static_kwargs: dict[str, Any] | None = None,
    key: str | None = None,
    jit: bool = True,
) -> tuple[Any, PrecompileResult]:
    """Lower + compile ``fn`` for the given concrete ``args``, warming the cache.

    This is the AOT primitive. It runs ``jax.jit(fn).lower(*args,
    **static_kwargs).compile()``; the compile populates the persistent on-disk
    compile cache so subsequent processes that build the identical program get a
    warm disk-read instead of a cold compile.

    Parameters
    ----------
    fn:
        The function to compile. If ``jit`` is True it is wrapped in
        :func:`jax.jit`; pass an already-jitted fn (e.g. the operational entry)
        with ``jit=False``.
    args:
        Concrete example arguments (or :class:`jax.ShapeDtypeStruct`) defining
        the shapes/dtypes of the program to compile.
    static_kwargs:
        Keyword args forwarded to ``.lower`` (e.g. ``hours=`` for the
        operational jit, which marks ``hours`` static).
    key:
        Optional project-level config key for the result label.
    jit:
        Wrap ``fn`` in ``jax.jit`` (True) or treat it as already jitted (False).

    Returns
    -------
    (compiled, result):
        ``compiled`` is the ``jax.stages.Compiled`` executable (callable);
        ``result`` is a :class:`PrecompileResult` with timing + cache status.
    """
    # Make sure the persistent compile cache is configured (idempotent). It is
    # already wired at package import, but a bare script that imports this module
    # directly still gets the cache.
    configure_compilation_cache()

    jfn = jax.jit(fn) if jit else fn
    static_kwargs = static_kwargs or {}

    t0 = time.perf_counter()
    try:
        lowered = jfn.lower(*args, **static_kwargs)
        try:
            from gpuwrf.runtime import aot_executable as _aotx

            hlo_sha256 = _aotx.hlo_sha256_from_lowered(lowered)
        except Exception:  # noqa: BLE001 - digest is a diagnostic/lookup aid
            hlo_sha256 = None
        compiled = lowered.compile()
        # Force the cache write / executable realisation to complete.
        _ = compiled  # compile() already triggered the cache store
        dt = time.perf_counter() - t0
    except Exception as exc:  # surface but don't crash a batch pre-warm
        dt = time.perf_counter() - t0
        return None, PrecompileResult(
            key=key or getattr(fn, "__name__", "fn"),
            compile_seconds=dt,
            cache_dir=CACHE_STATUS.get("dir"),  # type: ignore[arg-type]
            cache_enabled=bool(CACHE_STATUS.get("enabled")),
            error=f"{type(exc).__name__}: {exc}",
        )

    return compiled, PrecompileResult(
        key=key or getattr(fn, "__name__", "fn"),
        compile_seconds=dt,
        cache_dir=CACHE_STATUS.get("dir"),  # type: ignore[arg-type]
        cache_enabled=bool(CACHE_STATUS.get("enabled")),
        hlo_sha256=hlo_sha256,
    )


# A spec provider builds the concrete (fn, args, static_kwargs) for a named
# production config. The operational pipeline supplies this so AOT stays
# decoupled from State/IO construction (and numerically inert).
SpecProvider = Callable[[GridConfig], "tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], bool]"]


def prewarm_one(config: GridConfig, spec_provider: SpecProvider) -> PrecompileResult:
    """Pre-warm one production config.

    ``spec_provider(config)`` must return ``(fn, args, static_kwargs, is_jitted)``
    where ``is_jitted`` indicates whether ``fn`` is already a ``jax.jit`` callable
    (passed straight through) or a plain function to wrap.
    """
    try:
        fn, args, static_kwargs, is_jitted = spec_provider(config)
    except Exception as exc:
        return PrecompileResult(
            key=config_key(config),
            compile_seconds=0.0,
            cache_dir=CACHE_STATUS.get("dir"),  # type: ignore[arg-type]
            cache_enabled=bool(CACHE_STATUS.get("enabled")),
            error=f"spec_provider: {type(exc).__name__}: {exc}",
        )
    _, result = precompile(
        fn,
        *args,
        static_kwargs=static_kwargs,
        key=config_key(config),
        jit=not is_jitted,
    )
    return result


def prewarm(
    spec_provider: SpecProvider,
    configs: tuple[GridConfig, ...] = PRODUCTION_GRIDS,
) -> list[PrecompileResult]:
    """Pre-warm every config in ``configs`` (default: the production grids).

    Intended to run once on install / first boot so the standard grids load
    near-instantly thereafter. Errors on one config do not abort the rest; each
    :class:`PrecompileResult` carries its own ``error`` if any.
    """
    return [prewarm_one(cfg, spec_provider) for cfg in configs]


# --------------------------------------------------------------------------- #
# vNext: CROSS-DOMAIN PARALLEL COMPILE of the de-fuse nest path.
#
# The de-fuse nest path (GPUWRF_NESTED_DEFUSE_COMPILE=1) compiles each domain's
# ``_advance_chunk_fori`` as its OWN executable. Today those N modules are
# compiled SEQUENTIALLY by the eager integration loop (~Sum(9 bodies) ~50 min
# cold). They are INDEPENDENT modules, so we pre-compile them CONCURRENTLY in
# SPAWNED CHILD PROCESSES (not threads -- no GIL ceiling, and each child isolates
# one domain's compile RAM + CUDA context, PRESERVING the de-fuse RAM win), each
# warming exactly one domain's ``<key>-cache`` entry in the ONE shared
# version-keyed JAX cache (made concurrent-write-safe by the FileLock +
# atomic-writer in compile_cache.py). The main run then warm-hits all N and runs
# the UNCHANGED eager de-fuse numerics path. Cold wall ~Sum(N) -> ~max(one body)
# + pool overhead. NUMERICALLY INERT: only WHEN/WHERE XLA compiles, never the
# executable bytes (same fn, same shapes/dtypes, same statics-as-traced, same
# namelist static aux, same clock_base => identical HLO => same cache key).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DomainCompileSpec:
    """Picklable per-domain spec the spawned child rebuilds its compile from.

    Carries ONLY what determines the HLO/cache key, as picklable values:

    * ``name``        -- domain name (label only).
    * ``carry``       -- a ``jax.ShapeDtypeStruct`` pytree (shapes/dtypes only)
      describing the ``OperationalCarry`` the runtime advances. ``_advance_chunk_fori``
      is NOT donated, so a ShapeDtypeStruct lowers to the IDENTICAL HLO as a real
      carry, and it pickles trivially (no live device arrays cross the spawn).
    * ``namelist``    -- the ``OperationalNamelist`` (a registered pytree whose
      grid + scalar physics controls are STATIC aux; it pickles across spawn and
      is the source of the static cache-key aux).
    * ``clock_base``  -- a ``jax.ShapeDtypeStruct`` pytree for the traced
      ``_ClockBase`` (#91 date scalars ride as TRACED args, so only their
      shapes/dtypes matter for the key).
    * ``n_steps`` / ``cadence`` -- the EXACT Python-int values the runtime passes
      for THIS domain. They are ``jnp.asarray``'d inside the jitted body, i.e.
      baked as HLO constants, so they ARE part of the cache key -- the child MUST
      use the runtime value or there is no warm hit. ``n_steps`` is 1 for a
      domain that has children (advanced one own-step at a time) and the parent
      ``parent_grid_ratio`` for a leaf.
    """

    name: str
    carry: Any
    namelist: Any
    clock_base: Any
    n_steps: int
    cadence: int


def _to_shape_dtype_tree(value: Any) -> Any:
    """Map every array-like leaf of ``value`` to a ``jax.ShapeDtypeStruct``.

    Non-array leaves (Python scalars, statics) pass through unchanged. Result is
    picklable (no device buffers) and lowers to the identical HLO as the concrete
    pytree for a non-donated jit (shapes/dtypes are all the cache key needs)."""
    def _leaf(x: Any) -> Any:
        shape = getattr(x, "shape", None)
        dtype = getattr(x, "dtype", None)
        if shape is not None and dtype is not None:
            return jax.ShapeDtypeStruct(tuple(shape), dtype)
        return x

    return jax.tree_util.tree_map(_leaf, value)


def build_defused_specs(
    tree: Any,
    *,
    carries: dict[str, Any] | None = None,
) -> list["DomainCompileSpec"]:
    """Build one picklable :class:`DomainCompileSpec` per domain from ``tree``.

    Mirrors EXACTLY how the eager de-fuse path calls ``_advance_chunk`` /
    ``_advance_chunk_fori`` (see ``domain_tree._operational_advance_factory`` and
    the eager integrate loop) so each child's compiled HLO is byte-identical to
    the runtime call:

    * carry      = caller-supplied runtime carry -> shapes when available; this is
      the production nested path, where the carry may already include post-init
      scheme subtrees such as Noah-MP land/radiation state;
    * clock_base = ``build_clock_base(namelist)`` -> shapes;
    * cadence    = ``int(namelist.radiation_cadence_steps)``;
    * n_steps    = 1 for a domain WITH children, else the parent edge's
      ``parent_grid_ratio`` (the leaf advance count the parent drives).

    Pure / no device work beyond the legacy fallback ``jax.eval_shape`` path.
    """
    from gpuwrf.runtime.operational_mode import (
        _initial_carry_for_run,
        build_clock_base,
    )

    # n_steps per domain: leaves advance parent_grid_ratio at a time; a domain
    # with children advances one own-step at a time (n_steps=1). Default 1.
    n_steps_by_domain: dict[str, int] = {name: 1 for name in tree.domains}
    for parent, edges in tree.edges.items():
        for edge in edges:
            child = edge.child
            if not tree.hierarchy.children(child):  # leaf
                n_steps_by_domain[child] = int(edge.parent_grid_ratio)

    runtime_carries = dict(carries) if carries is not None else None
    specs: list[DomainCompileSpec] = []
    for name, bundle in tree.domains.items():
        namelist = bundle.namelist
        cadence = int(namelist.radiation_cadence_steps)
        if cadence <= 0:
            raise ValueError(f"{name}: radiation_cadence_steps must be positive")
        if runtime_carries is not None:
            if name not in runtime_carries:
                raise ValueError(f"{name}: missing runtime carry for de-fuse prewarm")
            # Use the ACTUAL runtime carry pytree after all post-init seeding
            # (Noah-MP land/radiation, explicit scheme carries, device commit). This
            # is the cache-key-critical shape/treedef the eager loop will compile.
            carry_shapes = _to_shape_dtype_tree(runtime_carries[name])
        else:
            # Legacy fallback for callers that do not have a pre-seeded carry map:
            # eval_shape traces _initial_carry_for_run without running the step.
            carry_shapes = jax.eval_shape(_initial_carry_for_run, bundle.state, namelist)
        clock_shapes = _to_shape_dtype_tree(build_clock_base(namelist))
        specs.append(
            DomainCompileSpec(
                name=str(name),
                carry=carry_shapes,
                namelist=namelist,
                clock_base=clock_shapes,
                n_steps=int(n_steps_by_domain.get(name, 1)),
                cadence=cadence,
            )
        )
    return specs


def _aot_enabled() -> bool:
    """``GPUWRF_NESTED_AOT`` truthy? DEFAULT-ON (vNext cheap-key manifest).

    When ON the parallel-prewarm child ALSO serializes each domain's compiled
    executable (under the cheap_key + hlo address) so a fresh warm process loads
    it without re-lowering. Set ``GPUWRF_NESTED_AOT=0`` to disable. Fail-open:
    serialization is best-effort and never blocks the cache-warming prewarm."""
    return os.environ.get("GPUWRF_NESTED_AOT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def aot_dir(cache_dir: str | Path | None = None) -> Path | None:
    """The version-keyed AOT-blob dir ``<cache_dir>/aot/<version_tag>``.

    AOT blobs are target-specific (SM/jaxlib/driver), so they live in a
    version-tagged subdir of the SAME version-keyed JIT cache (a v0.20 blob never
    leaks into a v0.21 process). Returns ``None`` if caching is disabled."""
    base = Path(cache_dir).expanduser() if cache_dir else resolve_cache_dir()
    if base is None:
        return None
    return Path(base) / "aot" / version_cache_tag()


def _safe_aot_component(value: str) -> str:
    """Filesystem-safe AOT path component."""
    return str(value).replace("/", "_").replace("..", "_")


def _aot_blob_paths(
    name: str,
    cache_dir: str | Path | None = None,
    *,
    cheap_key: str | None = None,
    hlo_sha256: str | None = None,
) -> tuple[Path, Path] | None:
    """``(blob_path, meta_path)`` for a domain variant (``None`` if cache off).

    Addressing precedence (vNext cheap-key manifest):

    * ``cheap_key`` set -> ``<aot>/<tag>/<domain>/k_<cheap_key>.xlaexec`` -- the
      METADATA-only key the warm eager loop computes WITHOUT lowering.
    * else ``hlo_sha256`` set -> legacy ``<aot>/<tag>/<domain>/<hlo>.xlaexec``.
    * else -> legacy one-blob-per-domain ``<aot>/<tag>/<domain>.xlaexec``.

    The ``k_`` prefix keeps cheap-key blobs in a distinct namespace from the
    hlo-addressed ones so the two schemes never collide in the same domain dir.
    """
    d = aot_dir(cache_dir)
    if d is None:
        return None
    safe = _safe_aot_component(name)
    if cheap_key:
        k = _safe_aot_component(str(cheap_key))
        return d / safe / f"k_{k}.xlaexec", d / safe / f"k_{k}.meta"
    if hlo_sha256:
        h = _safe_aot_component(str(hlo_sha256))
        return d / safe / f"{h}.xlaexec", d / safe / f"{h}.meta"
    return d / f"{safe}.xlaexec", d / f"{safe}.meta"


def _cheap_key_quarantine_path(
    name: str,
    cheap_key: str,
    cache_dir: str | Path | None = None,
) -> Path | None:
    """``<aot>/<tag>/<domain>/k_<cheap_key>.quarantine`` marker path (or ``None``).

    A quarantine marker POISONS a ``(domain, cheap_key)`` pair: it is written when
    a REAL collision is detected (two distinct HLOs hash to the same cheap_key, or
    a verify-mode HLO mismatch). Once present, NO cheap-key blob is ever written or
    loaded for that key -- the eager loop fails OPEN to a fresh compile instead of
    serving a silently-wrong sibling blob. This is the durable, cross-process
    fail-CLOSED guard for the collision the GPT critic found (P0-1)."""
    d = aot_dir(cache_dir)
    if d is None:
        return None
    safe = _safe_aot_component(name)
    k = _safe_aot_component(str(cheap_key))
    return d / safe / f"k_{k}.quarantine"


def cheap_key_is_quarantined(
    name: str,
    cheap_key: str,
    cache_dir: str | Path | None = None,
) -> bool:
    """True iff a quarantine marker exists for ``(name, cheap_key)`` (fail-open).

    A quarantined key must NEVER load or write a cheap-key blob -- the load path
    falls open to compile. Any error checking returns ``False`` (so a probe error
    cannot block the cache); the WRITE guard is the authoritative collision check."""
    try:
        qp = _cheap_key_quarantine_path(name, cheap_key, cache_dir)
        return qp is not None and qp.is_file()
    except Exception:  # noqa: BLE001 - fail-open
        return False


def quarantine_cheap_key(
    name: str,
    cheap_key: str,
    cache_dir: str | Path | None = None,
    *,
    reason: str = "",
    detail: dict[str, Any] | None = None,
) -> bool:
    """POISON ``(name, cheap_key)``: write a marker + REMOVE any cheap-key blob.

    After this, :func:`cheap_key_is_quarantined` is true and the cheap-key blob /
    meta for this key are deleted so a stale (and now known-ambiguous) sibling can
    never be loaded. The hlo-addressed fallback blob (if any) is left intact -- it
    is keyed by the exact HLO, so it is never ambiguous. Best-effort / fail-open:
    returns ``True`` on success, ``False`` on any error (caller still fails open to
    compile)."""
    try:
        qp = _cheap_key_quarantine_path(name, cheap_key, cache_dir)
        if qp is None:
            return False
        import json as _json
        import time as _time

        payload = {
            "name": str(name),
            "cheap_key": str(cheap_key),
            "reason": str(reason),
            "detail": detail or {},
            "quarantined_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }
        _atomic_write_bytes(qp, _json.dumps(payload, sort_keys=True).encode("utf-8"))
        # Remove the now-ambiguous cheap-key blob + meta so it is never served.
        paths = _aot_blob_paths(name, cache_dir, cheap_key=cheap_key)
        if paths is not None:
            for p in paths:
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
        return True
    except Exception:  # noqa: BLE001 - fail-open
        return False


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via temp-file + ``os.replace`` (atomic rename).

    Mirrors compile_cache's atomic-writer guarantee: a concurrent reader sees
    either the old file or the complete new file, never a truncated one, and a
    crash mid-write leaves only a stray ``.tmp``. Same-dir temp => same-fs rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _serialize_domain_blob(
    name: str,
    compiled: Any,
    cache_dir: str | Path | None,
    *,
    hlo_sha256: str | None = None,
    lowered: Any | None = None,
    cheap_key: str | None = None,
    key_schema: str | None = None,
) -> dict[str, Any]:
    """Serialize ``compiled`` to the version-keyed AOT dir (Step B). Fail-open.

    When ``cheap_key`` is supplied the blob is ALSO written under the cheap-key
    address (``k_<cheap_key>``) so a fresh warm process can locate it by metadata
    hashing WITHOUT lowering. The legacy hlo-addressed blob is still written when
    no cheap_key is given (back-compat). ``lowered`` (when available) lets
    :func:`aot_executable.serialize` re-derive the lower-only StableHLO digest if
    the caller's ``hlo_sha256`` came back ``None`` -- the persisted
    ``meta.hlo_sha256`` MUST be that lower digest or the cross-process cheap-key
    load rejects the blob (``not meta_hlo``) and verify-mode never confirms it.
    Returns a small status dict
    (``aot_written``, ``aot_blob_bytes``, ``aot_path``, ``aot_error``,
    ``cheap_key``, ``blob_sha256``). Any failure is captured -- the prewarm still
    wrote the HLO cache entry, so the run just cold/warm-compiles.

    COLLISION GUARD (P0-1, GPT-critic SAFETY): a cheap-key write is fail-CLOSED.
    Before writing under ``k_<cheap_key>`` we check (a) the quarantine marker and
    (b) any EXISTING ``k_<cheap_key>.meta`` whose recorded ``hlo_sha256`` differs
    from the one we are about to write. Either case is a REAL collision (two
    distinct HLOs hashing to one cheap_key) -> we POISON the key (quarantine +
    remove the ambiguous cheap-key blob) and write ONLY the hlo-addressed fallback,
    NEVER overwriting the colliding sibling. The next load of that cheap_key
    fails OPEN to a fresh compile (no silent wrong blob)."""
    from gpuwrf.runtime import aot_executable as aotx

    out: dict[str, Any] = {"aot_written": False, "aot_blob_bytes": 0, "aot_error": None}
    try:
        blob, meta = aotx.serialize(
            compiled,
            hlo_sha256=hlo_sha256,
            lowered=lowered,
            cheap_key=cheap_key,
            key_schema=key_schema,
        )
        import pickle as _pickle

        # --- P0-1 collision guard: decide whether the cheap-key address is safe. ---
        write_cheap_key = cheap_key
        if cheap_key:
            collision_reason = None
            collision_detail: dict[str, Any] = {}
            if cheap_key_is_quarantined(name, cheap_key, cache_dir):
                collision_reason = "previously-quarantined cheap_key"
            else:
                ck_paths = _aot_blob_paths(name, cache_dir, cheap_key=cheap_key)
                if ck_paths is not None:
                    _ck_blob, ck_meta_path = ck_paths
                    if ck_meta_path.is_file():
                        try:
                            with open(ck_meta_path, "rb") as _fh:
                                existing = _pickle.load(_fh)
                            existing_hlo = getattr(existing, "hlo_sha256", None)
                        except Exception:  # noqa: BLE001 - unreadable sibling meta
                            existing_hlo = None
                            collision_reason = "unreadable existing cheap_key meta"
                        else:
                            if (
                                existing_hlo
                                and meta.hlo_sha256
                                and existing_hlo != meta.hlo_sha256
                            ):
                                collision_reason = "cheap_key HLO collision"
                                collision_detail = {
                                    "existing_hlo": existing_hlo,
                                    "new_hlo": meta.hlo_sha256,
                                }
            if collision_reason is not None:
                # POISON the key: never write/serve an ambiguous cheap-key blob.
                quarantine_cheap_key(
                    name,
                    cheap_key,
                    cache_dir,
                    reason=collision_reason,
                    detail=collision_detail,
                )
                write_cheap_key = None  # only the hlo-addressed fallback below
                out["cheap_key_quarantined"] = True
                out["cheap_key_quarantine_reason"] = collision_reason

        # Address by cheap_key when SAFE, else fall back to the hlo digest. If the
        # key was quarantined, write_cheap_key is None so the hlo-addressed blob is
        # written instead (keyed by the exact HLO -> never ambiguous).
        paths = _aot_blob_paths(
            name, cache_dir, cheap_key=write_cheap_key, hlo_sha256=meta.hlo_sha256
        )
        if paths is None:
            out["aot_error"] = "cache disabled; AOT blob not written"
            return out
        blob_path, meta_path = paths
        # Pickle the AotMeta (treedefs + kept_var_idx + fingerprint + hlo hash +
        # cheap_key + blob_sha256).
        _atomic_write_bytes(blob_path, blob)
        _atomic_write_bytes(meta_path, _pickle.dumps(meta, protocol=_pickle.HIGHEST_PROTOCOL))
        out["aot_written"] = True
        out["aot_blob_bytes"] = len(blob)
        out["aot_path"] = str(blob_path)
        out["aot_meta_path"] = str(meta_path)
        out["hlo_sha256"] = meta.hlo_sha256
        # Report the cheap_key actually WRITTEN (None when quarantined -> only the
        # hlo-addressed fallback exists), so callers do not believe a cheap-key blob
        # was written when it was poisoned.
        out["cheap_key"] = write_cheap_key
        out["blob_sha256"] = meta.blob_sha256
        out["aot_kept_buffers"] = (
            len(meta.kept_var_idx) if meta.kept_var_idx is not None else None
        )
    except BaseException as exc:  # noqa: BLE001 - serialize is best-effort
        out["aot_error"] = f"{type(exc).__name__}: {exc}"
    return out


def load_domain_blob(
    name: str,
    cache_dir: str | Path | None = None,
    *,
    cheap_key: str | None = None,
    hlo_sha256: str | None = None,
    dev: Any | None = None,
    check_fingerprint: bool = True,
    return_status: bool = False,
) -> Any | None:
    """Load domain ``name``'s AOT blob into a drop-in advance callable (Step C).

    Addressing precedence: ``cheap_key`` (the metadata-only key the warm eager
    loop computes WITHOUT lowering) -> ``hlo_sha256`` -> legacy one-blob path. On
    load the saved target fingerprint must match the live backend, the on-disk
    blob bytes are re-hashed against ``meta.blob_sha256`` (rejects a truncated /
    edited blob), and the saved cheap_key/hlo must match the requested key; then
    the executable is deserialized and wrapped by :func:`aot_executable.load`.
    Returns ``None`` on ANY failure (missing blob, fingerprint mismatch, integrity
    mismatch, corrupt meta, deserialize error) so the caller falls back to compile
    -- never wrong, only slower.

    When ``return_status`` is true, returns ``(call_or_none, status_dict)`` with a
    stable reason string for stderr diagnostics at the eager Step-C call site;
    the status carries the ``meta.hlo_sha256`` so verify-mode can confirm it.
    """
    status: dict[str, Any] = {
        "name": str(name),
        "loaded": False,
        "source": None,
        "error": None,
        "blob_path": None,
        "meta_path": None,
        "cheap_key": cheap_key,
        "hlo_sha256": hlo_sha256,
        "meta_hlo_sha256": None,
    }

    def _done(call: Any | None) -> Any:
        return (call, status) if return_status else call

    try:
        from gpuwrf.runtime import aot_executable as aotx

        # P0-1: a quarantined (poisoned) cheap_key NEVER loads -- fail OPEN to a
        # fresh compile so a known-ambiguous key cannot serve a wrong blob.
        if cheap_key and cheap_key_is_quarantined(name, cheap_key, cache_dir):
            status["source"] = "fallback:cheap-key-quarantined"
            status["error"] = (
                f"cheap_key {cheap_key[:12]} is quarantined (known collision); "
                "refusing to load -- compiling fresh"
            )
            return _done(None)

        paths = _aot_blob_paths(
            name, cache_dir, cheap_key=cheap_key, hlo_sha256=hlo_sha256
        )
        if paths is None:
            status["source"] = "fallback:cache-disabled"
            status["error"] = "cache disabled; AOT blob not loaded"
            return _done(None)
        blob_path, meta_path = paths
        status["blob_path"] = str(blob_path)
        status["meta_path"] = str(meta_path)
        if not blob_path.is_file() or not meta_path.is_file():
            status["source"] = "fallback:missing"
            missing = []
            if not blob_path.is_file():
                missing.append(str(blob_path))
            if not meta_path.is_file():
                missing.append(str(meta_path))
            status["error"] = "missing AOT artifact(s): " + ", ".join(missing)
            return _done(None)
        import pickle as _pickle

        blob = blob_path.read_bytes()
        with open(meta_path, "rb") as fh:
            meta = _pickle.load(fh)
        status["meta_hlo_sha256"] = getattr(meta, "hlo_sha256", None)

        # P1-3: a cheap-key load must POSITIVELY pass the metadata contract; a
        # missing/empty/mismatched field FAILS OPEN (never accept an under-attested
        # blob). For cheap_key loads we REQUIRE: nonempty blob_sha256 (verified
        # below), meta.cheap_key == cheap_key, meta.key_schema == current KEY_SCHEMA,
        # nonempty meta.hlo_sha256. (hlo-addressed loads keep the legacy contract:
        # blob_sha256 verified-if-present, meta.hlo == requested hlo.)
        expected_blob = getattr(meta, "blob_sha256", None)
        if cheap_key:
            from gpuwrf.runtime import aot_cheap_key as _ck

            meta_ck = getattr(meta, "cheap_key", None)
            meta_schema = getattr(meta, "key_schema", None)
            meta_hlo = getattr(meta, "hlo_sha256", None)
            reason = None
            if not expected_blob:
                reason = "missing meta.blob_sha256"
            elif meta_ck != cheap_key:
                reason = (
                    f"meta.cheap_key={str(meta_ck)[:12]} != requested {cheap_key[:12]}"
                )
            elif meta_schema != _ck.KEY_SCHEMA:
                reason = (
                    f"meta.key_schema={meta_schema!r} != current {_ck.KEY_SCHEMA!r}"
                )
            elif not meta_hlo:
                reason = "missing meta.hlo_sha256"
            if reason is not None:
                status["source"] = "fallback:cheap-key-meta-mismatch"
                status["error"] = f"AOT cheap-key metadata contract failed: {reason}"
                return _done(None)
        elif hlo_sha256 and status["meta_hlo_sha256"] != hlo_sha256:
            status["source"] = "fallback:hlo-mismatch"
            status["error"] = (
                "AOT artifact HLO mismatch: "
                f"path key={hlo_sha256}, meta={status['meta_hlo_sha256']}"
            )
            return _done(None)

        # Blob-integrity: the on-disk bytes must match meta.blob_sha256 (rejects a
        # truncated / corrupted / hand-edited blob -- right path, wrong bytes). For
        # cheap-key loads a nonempty blob_sha256 was already REQUIRED above.
        if expected_blob is not None:
            got_blob = aotx.blob_sha256(blob)
            if got_blob != expected_blob:
                status["source"] = "fallback:blob-integrity"
                status["error"] = (
                    "AOT blob integrity mismatch: "
                    f"meta={expected_blob[:12]} disk={got_blob[:12]}"
                )
                return _done(None)

        call = aotx.load(blob, meta, dev, check_fingerprint=check_fingerprint)
        status["loaded"] = True
        status["source"] = "aot_blob"
        return _done(call)
    except BaseException as exc:  # noqa: BLE001 - fail-open: any error -> no AOT, compile
        status["source"] = "fallback:load-error"
        status["error"] = f"{type(exc).__name__}: {exc}"
        return _done(None)


def _compile_one_domain_worker(spec: "DomainCompileSpec") -> dict[str, Any]:
    """TOP-LEVEL (picklable) worker run in a SPAWNED child process.

    Imports ``gpuwrf`` FIRST (configures x64 + the SAME version-keyed locked
    cache dir -- without this there is no warm hit), then compiles exactly ONE
    domain's ``_advance_chunk_fori`` with the runtime arguments so the child
    writes that domain's ``<key>-cache`` entry into the shared cache. When
    ``GPUWRF_NESTED_AOT`` is set it ALSO serializes the compiled executable to the
    version-keyed AOT dir (Step B) so a fresh process can load it without
    re-lowering. Returns a plain-dict result (picklable back to the parent); never
    raises across the process boundary (errors are returned as data so the driver
    fails open)."""
    import time as _time

    t0 = _time.perf_counter()
    try:
        import gpuwrf  # noqa: F401  - triggers x64 + cache config in the child
        import jax.numpy as jnp

        from gpuwrf.runtime.compile_cache import (
            configure_compilation_cache as _cfg,
            cache_entry_count as _count,
        )
        from gpuwrf.runtime.operational_mode import _advance_chunk_fori

        status = _cfg()  # idempotent; ensures the locked cache dir is wired
        before = _count()
        # Match _operational_advance_factory.advance / _advance_chunk EXACTLY:
        #   (carry, namelist, jnp.asarray(start, int32), clock_base,
        #    n_steps=<int>, cadence=<int>).  _advance_chunk_fori is ALREADY
        #   @jax.jit, so jit=False.  start_step value is a TRACED scalar and does
        #   not affect the key; we use 1 for tidiness.
        compiled, result = precompile(
            _advance_chunk_fori,
            spec.carry,
            spec.namelist,
            jnp.asarray(1, dtype=jnp.int32),
            spec.clock_base,
            static_kwargs={"n_steps": int(spec.n_steps), "cadence": int(spec.cadence)},
            key=spec.name,
            jit=False,
        )
        after = _count()
        res: dict[str, Any] = {
            "name": spec.name,
            "compile_seconds": round(result.compile_seconds, 3),
            "entries_before": before,
            "entries_after": after,
            "wrote_entry": after > before,
            "cache_dir": status.get("dir"),
            "cache_locked": bool(status.get("locked")),
            "error": result.error,
            "hlo_sha256": result.hlo_sha256,
        }
        # Step B: AOT-serialize the compiled executable (opt-in, fail-open). Also
        # compute the cheap_key for THIS exact runtime spec so the blob is written
        # under the metadata-only address the warm eager loop will look it up by
        # WITHOUT lowering. cheap_key is best-effort (None -> hlo-addressed only).
        if _aot_enabled() and compiled is not None and result.error is None:
            ck = None
            key_schema = None
            try:
                from gpuwrf.runtime import aot_cheap_key as _ck
                from gpuwrf.runtime.operational_mode import _advance_chunk_fori as _adv

                ck = _ck.cheap_key(
                    _adv,
                    (
                        spec.carry,
                        spec.namelist,
                        jnp.asarray(1, dtype=jnp.int32),
                        spec.clock_base,
                    ),
                    {"n_steps": int(spec.n_steps), "cadence": int(spec.cadence)},
                    spec.namelist,
                )
                if ck:
                    key_schema = _ck.KEY_SCHEMA
            except Exception:  # noqa: BLE001 - cheap_key is best-effort
                ck = None
                key_schema = None
            res.update(
                _serialize_domain_blob(
                    spec.name,
                    compiled,
                    status.get("dir"),
                    hlo_sha256=result.hlo_sha256,
                    cheap_key=ck,
                    key_schema=key_schema,
                )
            )
        return res
    except BaseException as exc:  # noqa: BLE001 - report, never propagate
        return {
            "name": spec.name,
            "compile_seconds": round(_time.perf_counter() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _compile_one_domain_parent_verify(spec: "DomainCompileSpec") -> dict[str, Any]:
    """Compile one spec in the parent and report whether it was a true warm hit.

    This is the load-bearing cache-key check: after child prewarm, the parent lowers
    the EXACT runtime spec it will later use in the eager loop. ``entries_after ==
    entries_before`` means the child wrote the same persistent-cache key; any
    positive delta means the parent had to cold-compile a sibling/missing key. That
    is fail-open (integration will now be warm) but not a parallel-compile speedup.
    """
    import time as _time

    import jax.numpy as jnp

    from gpuwrf.runtime.compile_cache import cache_entry_count as _count
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    t0 = _time.perf_counter()
    before = _count()
    try:
        _, result = precompile(
            _advance_chunk_fori,
            spec.carry,
            spec.namelist,
            jnp.asarray(1, dtype=jnp.int32),
            spec.clock_base,
            static_kwargs={"n_steps": int(spec.n_steps), "cadence": int(spec.cadence)},
            key=spec.name,
            jit=False,
        )
        after = _count()
        error = result.error
    except BaseException as exc:  # noqa: BLE001 - verification must fail open
        after = _count()
        error = f"{type(exc).__name__}: {exc}"
    warm_hit = error is None and after == before and before > 0
    return {
        "name": spec.name,
        "entries_before": before,
        "entries_after": after,
        "entries_delta": after - before,
        "warm_hit": bool(warm_hit),
        "verify_seconds": round(_time.perf_counter() - t0, 3),
        "error": error,
    }


def _verify_parent_warm_hits(specs: list["DomainCompileSpec"]) -> list[dict[str, Any]]:
    """Parent-side exact-key warm-hit verification for all de-fuse specs."""
    return [_compile_one_domain_parent_verify(spec) for spec in specs]


def _host_ram_gib() -> float:
    """Best-effort total host RAM in GiB (0.0 if it cannot be probed)."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / float(1024 ** 3)
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return 0.0


# Per-domain compile working-set budget (GiB) used to RAM-cap the pool. The
# measured de-fuse per-domain peak is ~25 GB (COMPILE_FINDING.md); 28 leaves a
# margin so P concurrent compiles stay well under box RAM.
_PER_DOMAIN_COMPILE_GIB = 28
# Hard default cap: NOT N (N concurrent ~25 GB compiles + N CUDA contexts would
# OOM the box / compile-VRAM-OOM across N autotuners on the single 5090).
_DEFAULT_MAX_WORKERS = 4


def default_parallel_workers(n_domains: int) -> int:
    """Default pool size: ``min(n_domains, cpu_count, RAM//28)`` capped at 4.

    The RAM-derived term keeps P concurrent ~25 GB domain compiles under box RAM
    (the de-fuse RAM win is per-PROCESS, so P processes ~= P*per-domain); the
    hard cap of 4 bounds single-5090 autotune GPU-contention / compile-VRAM.
    ``GPUWRF_NESTED_PARALLEL_COMPILE=N`` overrides this (handled by the caller)."""
    n_domains = max(1, int(n_domains))
    cpu = os.cpu_count() or 1
    ram = _host_ram_gib()
    ram_cap = max(1, int(ram // _PER_DOMAIN_COMPILE_GIB)) if ram > 0 else _DEFAULT_MAX_WORKERS
    return max(1, min(n_domains, cpu, ram_cap, _DEFAULT_MAX_WORKERS))


def prewarm_defused_nest(
    tree: Any,
    *,
    carries: dict[str, Any] | None = None,
    max_workers: int | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Cross-domain PARALLEL pre-compile of the de-fuse nest (the wall win).

    Fans the N independent per-domain ``_advance_chunk_fori`` compiles out across
    spawned child processes, each warming exactly one domain's ``<key>-cache``
    entry in the shared version-keyed (locked) cache. The caller's UNCHANGED
    eager de-fuse loop then warm-hits all N. Returns a report dict
    (``workers``, ``domains`` per-domain results, ``wall_seconds``, ``warm_all``,
    ``error``). FAIL-OPEN: any error is captured in the report and the caller
    proceeds to cold-compile sequentially as today -- never wrong, only slower.

    ``max_workers`` defaults to :func:`default_parallel_workers`.

    ``verify`` (default ``False``) controls the parent-side exact-key warm-hit
    verification. When ``True`` the parent re-lowers+compiles each spec AFTER the
    pool to assert the child wrote the identical persistent-cache key (a domain
    that adds a cache entry there marks ``warm_all=False``). That check is a
    DIAGNOSTIC: it re-lowers (re-traces) all N huge ``_advance_chunk_fori``
    modules SEQUENTIALLY, which is the ~30 min re-lowering cost the de-fuse path
    pays once and which makes the verify pass NET-NEGATIVE for wall-clock. The
    eager integration loop already does its own single warm-hit per domain, so
    with ``verify=False`` the run still ends warm -- the verification only
    detects "silent no-speedup" (the ``entries_delta==0`` per-domain check) and
    is wanted only when diagnosing the parallel path. With ``verify=False`` the
    workers-only signal (``warm_all = not errs``) is reported and the costly
    re-lower is skipped entirely (the free ~1.4x cold win)."""
    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "workers": 0,
        "domains": [],
        "parent_verification": [],
        "wall_seconds": 0.0,
        "warm_all": False,
        "error": None,
    }
    try:
        configure_compilation_cache()
        if not CACHE_STATUS.get("enabled"):
            report["error"] = "compile cache disabled; parallel prewarm skipped"
            return report
        specs = build_defused_specs(tree, carries=carries)
        if not specs:
            report["error"] = "no domains to prewarm"
            return report
        n_workers = int(max_workers) if max_workers else default_parallel_workers(len(specs))
        n_workers = max(1, min(n_workers, len(specs)))
        report["workers"] = n_workers

        ctx = multiprocessing.get_context("spawn")
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_compile_one_domain_worker, s): s.name for s in specs}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results.append(fut.result())
                except BaseException as exc:  # noqa: BLE001 - capture, fail open
                    results.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        report["domains"] = results
        errs = [r for r in results if r.get("error")]
        errors: list[str] = []
        if errs:
            errors.append(
                f"{len(errs)}/{len(results)} domain prewarms reported an error"
            )

        if verify:
            # DIAGNOSTIC opt-in (GPUWRF_NESTED_PARALLEL_VERIFY): re-lower every
            # spec in the parent to assert the child wrote the identical cache
            # key. This is the ~30 min sequential re-lowering pass -- NET-NEGATIVE
            # for wall -- so it is OFF by default.
            parent_verify = _verify_parent_warm_hits(specs)
            report["parent_verification"] = parent_verify
            verify_misses = [r for r in parent_verify if not r.get("warm_hit")]
            if verify_misses:
                misses = ", ".join(
                    f"{r.get('name')} delta={r.get('entries_delta')} error={r.get('error')}"
                    for r in verify_misses
                )
                errors.append(
                    "parent exact-key warm-hit verification failed for "
                    f"{len(verify_misses)}/{len(parent_verify)} domains: {misses}"
                )
            if errors:
                report["error"] = "; ".join(errors) + "; falling back to sequential cold compile"
            report["warm_all"] = not errors and all(r.get("warm_hit") for r in parent_verify)
        else:
            # Default (verify OFF): skip the costly parent re-lower. The eager
            # integration loop will warm-hit each domain itself, so the run still
            # ends warm; the only signal we can give cheaply is the workers-only
            # "all workers reported success" (no parent exact-key proof).
            report["parent_verification"] = []
            if errors:
                report["error"] = "; ".join(errors) + "; falling back to sequential cold compile"
            report["warm_all"] = not errs
    except BaseException as exc:  # noqa: BLE001 - the whole driver is fail-open
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["wall_seconds"] = round(time.perf_counter() - t0, 3)
    return report


# --------------------------------------------------------------------------- #
# B2 deliverable #1: a single clean command to build / ship / reuse a prewarmed,
# version-keyed (B1) cache as a RELEASE ARTIFACT, so a fresh machine warm-starts
# instead of paying the ~40 min cold nest compile. See the module CLI below.
# --------------------------------------------------------------------------- #


def load_spec_provider(spec: str) -> SpecProvider:
    """Import a spec-provider callable from a ``module:callable`` string.

    The operational pipeline supplies the concrete ``(fn, args, static_kwargs,
    is_jitted)`` builder; we import it lazily by dotted path so this AOT module
    stays decoupled from State/IO construction. ``spec`` is e.g.
    ``mypkg.aot_specs:production_spec_provider``. Raises ``ValueError`` /
    ``ImportError`` / ``AttributeError`` with a clear message on a bad path.
    """
    if ":" not in spec:
        raise ValueError(
            f"spec-provider {spec!r} must be 'module.path:callable' "
            "(the operational pipeline's (fn,args,static,is_jitted) builder)"
        )
    mod_name, _, attr = spec.partition(":")
    module = importlib.import_module(mod_name)
    provider = getattr(module, attr)
    if not callable(provider):
        raise ValueError(f"spec-provider {spec!r} is not callable")
    return provider


def cache_artifact_info() -> dict[str, object]:
    """Report the version-keyed cache dir + tag + entry count (no compile).

    Works on a FRESH machine with no GPU / no spec-provider: it just resolves
    where the cache lives (B1's ``version_cache_tag``) and how warm it is, so an
    operator can see up front whether a run will warm-hit or cold-compile, and
    what the shippable artifact name should be.
    """
    configure_compilation_cache()
    cache_dir = resolve_cache_dir()
    tag = version_cache_tag()
    entries = cache_entry_count(cache_dir) if cache_dir else 0
    return {
        "version_tag": tag,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cache_enabled": bool(CACHE_STATUS.get("enabled")),
        "entry_count": entries,
        "warm_capable": entries > 0,
        "suggested_artifact": f"gpuwrf-jitcache-{tag}.tar.gz",
        "source": CACHE_STATUS.get("source"),
    }


def pack_cache(out_path: str | Path | None = None, cache_dir: str | Path | None = None) -> Path:
    """Tar the version-keyed JIT cache dir into a shippable ``.tar.gz`` artifact.

    The archive stores the cache entries under a SINGLE top-level directory named
    by the version tag (``gpuwrf-jitcache-<tag>/...``) so :func:`unpack_cache` can
    verify the artifact matches the running version before extracting. Returns the
    artifact path. Raises ``FileNotFoundError`` if the cache dir is missing/empty
    (nothing to ship -- the operator must :func:`prewarm` first).
    """
    src = Path(cache_dir).expanduser() if cache_dir else resolve_cache_dir()
    if src is None:
        raise FileNotFoundError(
            "compile cache is disabled (GPUWRF_JAX_CACHE=0?); nothing to pack"
        )
    src = Path(src)
    if not src.is_dir() or cache_entry_count(src) == 0:
        raise FileNotFoundError(
            f"cache dir {src} is missing or empty; run a prewarm first "
            "(no warm executables to ship)"
        )
    tag = version_cache_tag()
    top = f"gpuwrf-jitcache-{tag}"
    out = Path(out_path).expanduser() if out_path else Path.cwd() / f"{top}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        # arcname=top puts every entry under one tagged top-level dir.
        tar.add(str(src), arcname=top)
    return out


def unpack_cache(
    artifact: str | Path,
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Extract a shipped cache artifact INTO the version-keyed cache dir ``dest``.

    The artifact MUST contain exactly one ``gpuwrf-jitcache-<tag>`` top-level dir
    (the layout :func:`pack_cache` produces); 0 or >1 such dirs is rejected
    fail-closed. The single tagged dir's CONTENTS are placed strictly inside
    ``dest`` -- never a sibling, never ``dest.parent`` -- so a successful return
    means the active cache really has the entries (the bug this guards against:
    reporting success while ``dest`` stays empty and entries land elsewhere).

    Verifies the artifact tag matches the running version's tag (so a ``v0.20.0``
    artifact is NOT silently used for a ``v0.20.2`` binary -- a guaranteed miss,
    the failure B1 prevents). Pass ``force=True`` to extract a mismatched artifact
    anyway; even then the contents go INTO ``dest`` (the operator owns the dest
    path). Returns ``dest``. Path-traversal guarded (extract into a private temp
    staging dir + ``filter="data"``).
    """
    artifact = Path(artifact).expanduser()
    if not artifact.is_file():
        raise FileNotFoundError(f"cache artifact not found: {artifact}")
    dest = Path(cache_dir).expanduser() if cache_dir else resolve_cache_dir()
    if dest is None:
        raise RuntimeError(
            "compile cache is disabled (GPUWRF_JAX_CACHE=0?); cannot unpack"
        )
    dest = Path(dest)
    running_tag = version_cache_tag()

    with tarfile.open(artifact, "r:gz") as tar:
        members = tar.getmembers()
        # Require EXACTLY ONE gpuwrf-jitcache-<tag> top-level dir. Anything else
        # (a no-tag tarball, multiple tagged dirs, stray top-level files) is a
        # malformed artifact -> fail closed so dest is never silently left empty.
        tagged_tops = {
            m.name.split("/", 1)[0]
            for m in members
            if m.name and m.name.split("/", 1)[0].startswith("gpuwrf-jitcache-")
        }
        if len(tagged_tops) != 1:
            raise ValueError(
                f"artifact must contain exactly ONE 'gpuwrf-jitcache-<tag>' "
                f"top-level dir; found {sorted(tagged_tops) or 'none'}. This is "
                "not a gpuwrf cache artifact (or is corrupt); refusing to unpack."
            )
        top = next(iter(tagged_tops))
        artifact_tag = top[len("gpuwrf-jitcache-"):]
        if artifact_tag != running_tag and not force:
            raise ValueError(
                f"artifact tag {artifact_tag!r} != running tag {running_tag!r}; "
                "extracting it would be a GUARANTEED cache MISS (different HLO). "
                "Pass force=True / --force to override."
            )

        # Extract into a PRIVATE temp staging dir (never beside dest), then move
        # the single tagged dir's contents into dest. Path-traversal guard: every
        # member must resolve inside the temp staging root.
        with tempfile.TemporaryDirectory(prefix="gpuwrf-unpack-") as tmp:
            staging = Path(tmp)
            staging_resolved = staging.resolve()
            safe: list[tarfile.TarInfo] = []
            for m in members:
                target = (staging / m.name).resolve()
                if staging_resolved not in target.parents and target != staging_resolved:
                    raise ValueError(f"unsafe path in artifact: {m.name!r}")
                safe.append(m)
            # filter="data" (py3.12+) is the hardened extractor (no abs paths / no
            # escaping links); our guard above is belt-and-braces. Fall back to a
            # plain extractall on older runtimes that lack the kwarg.
            try:
                tar.extractall(path=str(staging), members=safe, filter="data")
            except TypeError:  # pragma: no cover - py<3.12 has no filter kwarg
                tar.extractall(path=str(staging), members=safe)

            extracted = staging / top
            if not extracted.is_dir():
                raise ValueError(
                    f"artifact top-level {top!r} did not extract to a directory; "
                    "refusing to unpack (corrupt artifact)."
                )
            dest.mkdir(parents=True, exist_ok=True)
            for child in extracted.iterdir():
                target = dest / child.name
                if target.exists():
                    # Replace an existing same-named entry so a re-unpack is
                    # idempotent rather than raising on rename.
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(child), str(target))
    return dest


# --------------------------------------------------------------------------- #
# CLI: python -m gpuwrf.runtime.aot_precompile <info|warm|pack|unpack>
# --------------------------------------------------------------------------- #


def _cmd_info(_args: argparse.Namespace) -> int:
    print(json.dumps(cache_artifact_info(), indent=2))
    return 0


def _cmd_warm(args: argparse.Namespace) -> int:
    if not args.spec_provider:
        sys.stderr.write(
            "warm: --spec-provider module:callable is required (the operational "
            "pipeline's (fn,args,static,is_jitted) builder). Use `info`/`pack` "
            "without a provider to inspect/ship an already-warmed cache.\n"
        )
        return 2
    try:
        provider = load_spec_provider(args.spec_provider)
    except Exception as exc:  # noqa: BLE001 - report cleanly, do not traceback
        sys.stderr.write(f"warm: failed to load spec-provider: {exc}\n")
        return 2
    results = prewarm(provider)
    summary = {
        "version_tag": version_cache_tag(),
        "cache_dir": CACHE_STATUS.get("dir"),
        "configs": [
            {
                "key": r.key,
                "compile_seconds": round(r.compile_seconds, 3),
                "error": r.error,
            }
            for r in results
        ],
        "ok": all(r.error is None for r in results),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def _cmd_pack(args: argparse.Namespace) -> int:
    try:
        out = pack_cache(out_path=args.out)
    except FileNotFoundError as exc:
        sys.stderr.write(f"pack: {exc}\n")
        return 1
    print(json.dumps({"artifact": str(out), "version_tag": version_cache_tag()}, indent=2))
    return 0


def _cmd_unpack(args: argparse.Namespace) -> int:
    try:
        dest = unpack_cache(args.artifact, force=args.force)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"unpack: {exc}\n")
        return 1
    info = cache_artifact_info()
    print(
        json.dumps(
            {"unpacked_into": str(dest), "entry_count": info["entry_count"]},
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m gpuwrf.runtime.aot_precompile`` -- build/ship/reuse the cache.

    Subcommands:

    * ``info``   -- report the version-keyed cache dir, tag, entry count + the
      suggested artifact name (no GPU / no compile; works on a fresh box).
    * ``warm``   -- ``--spec-provider module:callable`` builds the prewarmed cache
      for the PRODUCTION_GRIDS (this is the compile step; needs the operational
      spec-provider + a GPU to warm the GPU cache).
    * ``pack``   -- tar the version-keyed cache dir into a shippable
      ``gpuwrf-jitcache-<tag>.tar.gz`` (the RELEASE ARTIFACT).
    * ``unpack`` -- extract a shipped artifact into the version-keyed cache dir on
      a fresh machine (refuses a tag mismatch unless ``--force``), so the next run
      warm-starts instead of cold-compiling.
    """
    parser = argparse.ArgumentParser(
        prog="python -m gpuwrf.runtime.aot_precompile",
        description="Build / ship / reuse a prewarmed, version-keyed JIT cache.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="report cache dir/tag/entries (no compile)")
    p_info.set_defaults(func=_cmd_info)

    p_warm = sub.add_parser("warm", help="prewarm PRODUCTION_GRIDS (needs --spec-provider + GPU)")
    p_warm.add_argument(
        "--spec-provider",
        help="module.path:callable returning (fn,args,static_kwargs,is_jitted) per GridConfig",
    )
    p_warm.set_defaults(func=_cmd_warm)

    p_pack = sub.add_parser("pack", help="tar the warmed cache into a shippable artifact")
    p_pack.add_argument("--out", help="output .tar.gz path (default: ./gpuwrf-jitcache-<tag>.tar.gz)")
    p_pack.set_defaults(func=_cmd_pack)

    p_unpack = sub.add_parser("unpack", help="extract a shipped artifact into the cache dir")
    p_unpack.add_argument("artifact", help="path to a gpuwrf-jitcache-<tag>.tar.gz")
    p_unpack.add_argument(
        "--force", action="store_true", help="extract even if the artifact tag mismatches the running version"
    )
    p_unpack.set_defaults(func=_cmd_unpack)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI tests
    raise SystemExit(main())
