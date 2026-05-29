#!/usr/bin/env python
"""F7I: is the growing vertical mode an off-centering deficiency?

WRF's epssm forward-weighting is the designed damper for vertically-propagating
acoustic modes.  If the JAX off-centering is faithful, INCREASING epssm should
monotonically damp the growing center-column vertical mode; if raising epssm
does NOT damp it, the energy source is elsewhere (sign error / wrong term).

Runs the warm bubble to a fixed time for a sweep of epssm and reports
max|w| and the center-column curvature energy at the end.

Run:  PYTHONPATH=src taskset -c 0-3 python -u scripts/f7i_epssm_sensitivity.py
"""

from __future__ import annotations

import json
from dataclasses import replace as dataclass_replace
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def run_one(epssm: float, nsteps: int, ic: int) -> dict:
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = dataclass_replace(setup.namelist, epssm=float(epssm))
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    @jax.jit
    def run(carry):
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

    d = jax.device_get(run(carry0))
    return {k: float(v) for k, v in d.items()}


def main() -> int:
    ic = int(round(10000.0 / 250.0))
    nsteps = 1000  # t=100s, before detonation at default epssm
    results = []
    for eps in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        r = run_one(eps, nsteps, ic)
        r["epssm"] = eps
        results.append(r)
        print(
            f"epssm={eps:.2f}  t={nsteps*0.1:.0f}s  fin={r['finite']>0.5}  "
            f"max|w|={r['max_abs_w']:.4e}  zz_energy={r['zz_energy_center']:.4e}",
            flush=True,
        )
    Path("proofs/f7i").mkdir(parents=True, exist_ok=True)
    Path("proofs/f7i/epssm_sensitivity.json").write_text(json.dumps({"nsteps": nsteps, "results": results}, indent=2))
    print("wrote proofs/f7i/epssm_sensitivity.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
