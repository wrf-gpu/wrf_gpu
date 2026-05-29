#!/usr/bin/env python
"""F7I: which term re-injects the epssm-independent growing vertical mode?

Monkeypatch-ablate one large-step / buoyancy term at a time in the operational
acoustic core and observe whether the center-column vertical-mode growth
survives.  Pure diagnostic harness (no production code changed).

Terms probed:
  baseline   : unmodified
  no_pgbuoy  : zero the once-per-stage pg_buoy_w rw_tend (stage vertical PGF+buoy)
  no_thadv   : zero the large-step coupled theta tendency (theta advection->t_2ave)
  no_wadv    : (already absent) -- N/A, documented
  no_t2ave_b : zero the in-solver buoyancy c2a*alt*t_2ave term contribution

Run:  PYTHONPATH=src taskset -c 0-3 python -u scripts/f7i_term_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

import gpuwrf.runtime.operational_mode as opmod
from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry

ORIG_SUBSTEP = opmod.acoustic_substep_core


def run(nsteps: int, ic: int) -> dict:
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    @jax.jit
    def go(carry):
        steps = jnp.arange(nsteps, dtype=jnp.int32) + 1
        final, _ = jax.lax.scan(
            lambda cc, st: (_physics_boundary_step(cc, namelist, st, run_radiation=False), None), carry, steps
        )
        s = final.state
        wcol = s.w[:, 0, ic]
        curv = wcol[1:-1] - 0.5 * (wcol[:-2] + wcol[2:])
        return {
            "max_abs_w": jnp.max(jnp.abs(s.w)),
            "zz_energy_center": jnp.sum(curv * curv),
            "finite": (jnp.all(jnp.isfinite(s.w))).astype(jnp.float64),
        }

    d = jax.device_get(go(carry0))
    return {k: float(v) for k, v in d.items()}


def main() -> int:
    ic = int(round(10000.0 / 250.0))
    nsteps = 1000
    out = {}

    # baseline
    opmod.acoustic_substep_core = ORIG_SUBSTEP
    out["baseline"] = run(nsteps, ic)

    # no_pgbuoy: wrap to zero state.rw_tend_pg_buoy before the core runs
    def patched_no_pgbuoy(state, **kw):
        return ORIG_SUBSTEP(state.replace(rw_tend_pg_buoy=jnp.zeros_like(state.w)), **kw)

    opmod.acoustic_substep_core = patched_no_pgbuoy
    out["no_pgbuoy"] = run(nsteps, ic)

    # no_thadv: zero the large-step coupled theta tendency
    def patched_no_thadv(state, **kw):
        st = state.replace(theta_tend=jnp.zeros_like(state.theta_tend)) if state.theta_tend is not None else state
        return ORIG_SUBSTEP(st, **kw)

    opmod.acoustic_substep_core = patched_no_thadv
    out["no_thadv"] = run(nsteps, ic)

    opmod.acoustic_substep_core = ORIG_SUBSTEP

    for k, v in out.items():
        print(
            f"{k:14s} fin={v['finite']>0.5}  max|w|={v['max_abs_w']:.4e}  zz_energy={v['zz_energy_center']:.4e}",
            flush=True,
        )
    Path("proofs/f7i").mkdir(parents=True, exist_ok=True)
    Path("proofs/f7i/term_ablation.json").write_text(json.dumps({"nsteps": nsteps, "results": out}, indent=2))
    print("wrote proofs/f7i/term_ablation.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
