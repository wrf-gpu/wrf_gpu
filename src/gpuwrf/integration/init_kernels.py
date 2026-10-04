"""Compact AOT reuse of existing init kernels, without changing their JITs.

Known default-table RRTMG JITs use the existing guarded program/target key plus
the table asset digest; other calls use the exact lowered StableHLO digest.
The normal AOT loader retains its target, blob-integrity and call-ABI guards.
This helper belongs to eager initialization; it is never used in a timestep.
"""
from __future__ import annotations

import time
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax

from gpuwrf.runtime import aot_executable as aotx
from gpuwrf.runtime import aot_precompile as aotp
from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime.compile_cache import resolve_cache_dir


@dataclass(frozen=True)
class _StaticKwargs:
    values: tuple

    def tree_flatten(self):
        # Reuse the existing static_config_hash contract (static aux only).
        return (), self.values


@lru_cache(maxsize=1)
def _table_digest():
    from gpuwrf.physics.rrtmg_tables import asset_sha256
    return asset_sha256()


def resolved_radiation_constants():
    """Snapshot the values radiation imported, even if live env later changes.

    The general Step-C registry covers boundary import constants. Radiation has
    its own import-time switches in both solvers and McICA. Capture all scalar
    globals in these modules, including future scalar switches; source/table
    guards continue to cover the remaining constants. No device arrays are read.
    """
    from gpuwrf.physics import rrtmg_sw, rrtmg_lw
    from gpuwrf.kernels import rad_mcica
    return {
        module.__name__: {
            name: value for name,value in vars(module).items()
            if name.isupper() and isinstance(value,(bool,int,float,str))
        } for module in (rrtmg_sw,rrtmg_lw,rad_mcica)
    }


def radiation_init_key(fn, args, kwargs):
    """Existing guarded program/target key, restricted to default RRTMG JITs.

    Include static keywords and the table asset CONTENT as well as the existing
    source, resolved import constants, environment, dynamic avals/tree and target
    guards. Numeric column/gas values remain runtime operands. Other functions or
    explicit table overrides use the exact-HLO fallback instead.
    """
    from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column
    from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column
    if fn not in (solve_rrtmg_lw_column, solve_rrtmg_sw_column):
        return None
    if len(args) != 1 or 'tables' in kwargs:
        return None
    statics = {"debug": False, "with_clear_sky": False, "column_tile_cols": None}
    statics.update({k: v for k, v in kwargs.items() if k in statics})
    dynamic = {k: v for k, v in kwargs.items() if k not in statics}
    try:
        key = ck.exec_key(fn, args, dynamic, _StaticKwargs(tuple(sorted(statics.items()))))
        return ck.canonical_digest(("rrtmg-init-v2", key, _table_digest(),
                                    resolved_radiation_constants()))
    except Exception:
        return None  # fail open to exact-HLO lookup, never a weaker key


class InitKernelCache:
    """Load/export existing radiation JITs without changing their arithmetic.

    Static keywords must match the original JIT's static_argnames. They are
    passed to lowering, then omitted from the compiled executable's call ABI.
    A missing/invalid artifact falls back to the original compilation path.
    ``events`` separates setup cost from the warm load measurement.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else resolve_cache_dir()
        self.events: list[dict] = []

    def __call__(self, fn, *args, **kwargs):
        if self.cache_dir is None:
            return fn(*args, **kwargs)
        started = time.perf_counter()
        dynamic_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"debug", "with_clear_sky", "column_tile_cols"}
        }
        name = "init_" + fn.__name__
        ckey = radiation_init_key(fn, args, kwargs)
        call, status = aotp.load_domain_blob(
            name, self.cache_dir, cheap_key=ckey, return_status=True,
        ) if ckey else (None, {})
        metadata_status = status
        if call is not None and os.environ.get('GPUWRF_AOT_VERIFY','0').lower() in {'1','true','yes','on'}:
            actual = aotx.hlo_sha256_from_lowered(fn.lower(*args,**kwargs))
            if actual is None or actual != status.get('hlo_sha256'):
                aotp.quarantine_cheap_key(name, ckey, self.cache_dir,
                    reason='init verify-mode HLO mismatch',
                    detail={'actual':actual,'stored':status.get('hlo_sha256')})
                call = None
        if call is not None:
            try:
                result = call(*args, **dynamic_kwargs)
            except Exception:
                pass  # ABI guard rejected the hit; lower the exact call below
            else:
                self.events.append({
                    "kernel": name, "hlo_sha256": status.get("hlo_sha256"),
                    "cheap_key": ckey, "hit": True, "lower_s": 0.0,
                    "load_s": time.perf_counter() - started,
                    "setup_dispatch_s": time.perf_counter() - started,
                    "status": status,
                })
                return result
        # Lower the ORIGINAL jitted function, with its ORIGINAL static kwargs.
        # No enclosing jit, fusion boundary change, or new arithmetic is added.
        lowered = fn.lower(*args, **kwargs)
        digest = aotx.hlo_sha256_from_lowered(lowered)
        lowered_at = time.perf_counter()
        call, status = aotp.load_domain_blob(
            name, self.cache_dir, hlo_sha256=digest, return_status=True,
        ) if digest else (None, {"reason": "no lowered HLO digest"})
        loaded_at = time.perf_counter()
        hit = call is not None
        if hit:
            try:
                result = call(*args, **dynamic_kwargs)
            except Exception as exc:
                # A rejected call contract cannot silently serve a wrong result.
                status = {**status, "call_error": f"{type(exc).__name__}: {exc}"}
                hit = False
        if not hit:
            call = lowered.compile()
            result = call(*args, **dynamic_kwargs)
        if not hit or ckey:
            # A legacy exact-HLO hit can publish the new metadata address once.
            # The AOT writer quarantines collisions instead of overwriting them.
            compiled = lowered.compile() if hit else call
            status = {**status, **aotp._serialize_domain_blob(
                name, compiled, self.cache_dir, hlo_sha256=digest, lowered=lowered,
                cheap_key=ckey, key_schema=ck.KEY_SCHEMA,
            )}
        self.events.append({
            "kernel": name, "hlo_sha256": digest, "hit": hit,
            "lower_s": lowered_at - started,
            "load_s": loaded_at - lowered_at,
            "setup_dispatch_s": time.perf_counter() - started,
            "status": status,
            "metadata_lookup": metadata_status,
        })
        return result
