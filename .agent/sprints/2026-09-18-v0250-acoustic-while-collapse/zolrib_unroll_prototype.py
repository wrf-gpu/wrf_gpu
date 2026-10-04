"""Acoustic while-collapse sprint — zolrib trace-time unroll prototype (2026-09-18).

STATUS: the sprint contract's premise was FALSIFIED during attribution (see
WORKER_REPORT.md): the profiled ``loop_or_select_fusion`` (20/step @ 60.9 us,
pred[70,120] + f64[70,120]) is NOT the acoustic while-condition.  It is the BODY
fusion of ``gpuwrf.physics.surface_layer._zolrib``'s fixed-point z/L iteration
(MYNN surface layer, module_sf_mynn.F:1984-2048), lowered from
``jax.lax.fori_loop(0, 20, ...)`` to a trip-20 XLA while whose per-iteration
thunks are: divide fusion -> or/select body fusion (the 60.9 us kernel) ->
s64 counter add -> trip-count condition compare.

This module prototypes the collapse for that loop WITHOUT touching
``physics/**`` (hard scope): a verbatim copy of ``_zolrib`` with the 20 fixed
iterations unrolled at trace time.  Arithmetic per element is identical (same
ops, same order, same freeze-select semantics); only the control flow changes:
no while, no s64 counter, no per-iteration condition thunk.

PRE-REGISTERED GATES (declared before first run, per sprint contract):

- G1 numerics (primary): BITWISE fp64 equality of ``zolrib_unrolled`` outputs vs
  ``surface_layer._zolrib`` on both fixtures (cold seed + warm seed), each
  containing converged cells, non-converged cells (|ri| beyond the 20-iteration
  basin -> Li-2010 fallback), and zero-ri cells.
- G1 fallback tolerance (declared now, only if bitwise fails): max |diff| <=
  1e-13 * max(1, max|zol|) AND identical convergence/fallback selection masks;
  any NaN/inf => FAIL.  Rationale: the E1/FMA precedent
  (proofs/v025/econ/compare_e1_applied_anchor_VIOLATION.json) shows moving
  subgraphs between fusion contexts can change FMA contraction, so a nonzero
  diff is possible even for an op-identical unroll; it must then be bounded and
  reported, never silent.
- G2 HLO structure: the reference jit contains exactly one ``while`` op; the
  unrolled jit contains ZERO ``while`` ops; report fusion counts (GPU kernel /
  launch proxy) and total HLO op counts for both.
"""

from __future__ import annotations

import os
import re

import jax
import jax.numpy as jnp

from gpuwrf.physics.surface_layer import (
    _li_etal_2010,
    _psih_stable,
    _psih_unstable,
    _psim_stable,
    _psim_unstable,
    _zolrib,
)

_ZOLRIB_MAX_ITER = 20  # surface_layer._zolrib nmax (module_sf_mynn.F nmax=20)


def zolrib_unrolled(ri, za, z0, zt, logz0, logzt, zol1_seed=None, n_iter=_ZOLRIB_MAX_ITER):
    """``_zolrib`` with the fixed 20 iterations unrolled at trace time.

    ``residual`` is copied verbatim from ``surface_layer._zolrib`` (same psi
    lookups, same expression order); the freeze-select semantics are identical:
    ``nxt = where(frozen, zol_old, zol_new)`` evaluated BEFORE the flag update.
    """

    unstable = ri < 0.0

    def residual(zol_old):
        zol20 = zol_old * z0 / za
        zol3 = zol_old + zol20
        zolt = zol_old * zt / za
        psit2_u = jnp.maximum(logzt - (_psih_unstable(zol3) - _psih_unstable(zolt)), 1.0)
        psix2_u = jnp.maximum(logz0 - (_psim_unstable(zol3) - _psim_unstable(zol20)), 1.0)
        psit2_s = jnp.maximum(logzt - (_psih_stable(zol3) - _psih_stable(zolt)), 1.0)
        psix2_s = jnp.maximum(logz0 - (_psim_stable(zol3) - _psim_stable(zol20)), 1.0)
        psit2 = jnp.where(unstable, psit2_u, psit2_s)
        psix2 = jnp.where(unstable, psix2_u, psix2_s)
        return ri * psix2 * psix2 / psit2

    zol1 = _li_etal_2010(ri, za / z0, z0 / zt) if zol1_seed is None else zol1_seed
    zol1 = jnp.where(zol1 * ri < 0.0, 0.0, zol1)

    zol = zol1
    frozen = jnp.zeros_like(ri, dtype=bool)
    for _ in range(int(n_iter)):
        zol_new = residual(zol)
        converged = jnp.abs(zol_new - zol) <= 0.01
        zol = jnp.where(frozen, zol, zol_new)
        frozen = frozen | converged

    zol_fallback = _li_etal_2010(ri, za / z0, z0 / zt)
    return jnp.where(frozen, zol, zol_fallback), frozen


def fixture(ny=70, nx=120, seed=20260918):
    """[ny, nx] fp64 fields covering stable/unstable/neutral/extreme regimes."""

    rng = jax.random.PRNGKey(seed)
    ri = jax.random.uniform(rng, (ny, nx), minval=-4.0, maxval=4.0, dtype=jnp.float64)
    # clump near neutral (|ri| < 1e-3): exercises the <=0.01 convergence path
    near = jax.random.uniform(rng, (ny, nx), minval=-1.0e-3, maxval=1.0e-3, dtype=jnp.float64)
    mask_near = jax.random.uniform(rng, (ny, nx)) < 0.15
    # extremes far outside the 20-iteration basin -> forces the Li-2010 fallback
    extreme = jnp.where(
        jax.random.uniform(rng, (ny, nx)) < 0.5,
        jnp.full((ny, nx), 50.0, dtype=jnp.float64),
        jnp.full((ny, nx), -50.0, dtype=jnp.float64),
    )
    mask_ext = jax.random.uniform(rng, (ny, nx)) < 0.05
    mask_zero = jax.random.uniform(rng, (ny, nx)) < 0.02
    ri = jnp.where(mask_near, near, ri)
    ri = jnp.where(mask_ext, extreme, ri)
    ri = jnp.where(mask_zero, jnp.zeros_like(ri), ri)

    za = jax.random.uniform(rng, (ny, nx), minval=30.0, maxval=80.0, dtype=jnp.float64)
    z0 = jax.random.uniform(rng, (ny, nx), minval=1.0e-4, maxval=0.2, dtype=jnp.float64)
    zt = z0 / jax.random.uniform(rng, (ny, nx), minval=1.0, maxval=10.0, dtype=jnp.float64)
    logz0 = jnp.log(za / z0)
    logzt = jnp.log(za / zt)
    return dict(ri=ri, za=za, z0=z0, zt=zt, logz0=logz0, logzt=logzt)


def census(compiler_ir_text: str) -> dict:
    """Count while ops, fusions, gathers and total instructions in HLO/MLIR text."""

    n_while = len(re.findall(r"\bwhile\(", compiler_ir_text))
    n_fusion = len(re.findall(r"= fusion\(", compiler_ir_text))
    n_reduce = len(re.findall(r"\breduce\(", compiler_ir_text))
    n_ops = len(re.findall(r"^\s+%[\w.]+ = ", compiler_ir_text, re.M))
    n_s64 = len(re.findall(r"\bs64\[\]", compiler_ir_text))
    return {"while_ops": n_while, "fusion_ops": n_fusion, "reduce_ops": n_reduce,
            "total_ops": n_ops, "s64_refs": n_s64}


def compiled_census(fn, args) -> dict:
    """Census the OPTIMIZED HLO (fusion sites = GPU thunk/launch proxy)."""

    lowered = jax.jit(fn).lower(**args) if isinstance(args, dict) else jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    return census(compiled.as_text())


def bitwise_equal(a, b) -> bool:
    return bool(jnp.array_equal(a, b) and jnp.array_equal(jnp.isnan(a), jnp.isnan(b)))
