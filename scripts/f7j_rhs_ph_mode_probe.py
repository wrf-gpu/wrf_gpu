#!/usr/bin/env python
"""F7J probe: warm-bubble rise + Straka stability under rhs_ph term selection.

Compares two WRF-faithfulness readings of the geopotential-equation RHS:
  (A) FULL rhs_ph: terms 1,2 (horizontal phi advection) + 3 (-omega dphi/dz)
      + 4 (gw).  This is the literal Fortran rhs_ph
      (module_big_step_utilities_em.F:1481-1612).
  (B) terms 1,2 ONLY: matches the advance_w self-comment
      (module_small_step_em.F:1349 "ph_tend contains terms 1 and 2; now adding
      3 and 4 in stages"), with advance_w supplying terms 3,4.

Decides which reading removes the warm-bubble standing mode AND keeps Straka
finite, per F7J AC1/AC2/AC3.
"""

from __future__ import annotations

import argparse
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import gpuwrf.runtime.operational_mode as om
from gpuwrf.dynamics.core import rhs_ph as rhs_ph_mod
from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_density_current_numpy,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


def _patch(include_vertical_gw: bool):
    orig = rhs_ph_mod.rhs_ph_wrf

    def patched(**kw):
        kw["include_vertical_gw"] = include_vertical_gw
        return orig(**kw)

    om.rhs_ph_wrf = patched
    return orig


def run_case(case, nl, carry, stride, marks, label):
    @jax.jit
    def step(c, st):
        return om._physics_boundary_step(c, nl, st, run_radiation=False), None

    print(f"--- {label} ---", flush=True)
    for m in range(marks):
        for s in range(stride):
            carry, _ = step(carry, jnp.asarray(m * stride + s + 1, dtype=jnp.int32))
        w = carry.state.w
        th = carry.state.theta
        fin = bool(jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(th)))
        mw = float(jnp.max(jnp.abs(w)))
        t = (m + 1) * stride * case.dt_s
        # warm-bubble centroid rise proxy: mass-of theta'>0 weighted z
        print(f"  t={t:7.1f}s fin={fin} max|w|={mw:.3e}", flush=True)
        if not fin:
            break
    return carry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["warm", "straka"], default="warm")
    p.add_argument("--gw", choices=["full", "h12"], default="full")
    p.add_argument("--stride", type=int, default=200)
    p.add_argument("--marks", type=int, default=15)
    args = p.parse_args(argv)

    _patch(args.gw == "full")

    if args.case == "warm":
        case = build_warm_bubble_numpy()
    else:
        case = build_density_current_numpy()
    setup = _build_setup(case, require_gpu=True)
    nl = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))
    run_case(case, nl, carry, args.stride, args.marks, f"{args.case} gw={args.gw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
