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

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax

from gpuwrf.runtime.compile_cache import CACHE_STATUS, configure_compilation_cache

__all__ = [
    "GridConfig",
    "PRODUCTION_GRIDS",
    "PrecompileResult",
    "config_key",
    "precompile",
    "prewarm",
    "prewarm_one",
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
