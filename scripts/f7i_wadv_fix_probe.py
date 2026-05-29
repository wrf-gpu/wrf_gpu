#!/usr/bin/env python
"""F7I: does adding the missing large-step advect_w to rw_tend fix the mode?

WRF rk_tendency builds rw_tend = advect_w(w) + pg_buoy_w(grid%p)
(module_em.F:1011-1059 advect_w writes rw_tend, :1362 pg_buoy_w adds to it).
The JAX operational path sets rw_tend_pg_buoy = pg_buoy_w only, dropping the
advect_w transport.  Probe: monkeypatch acoustic_substep_core to add the
COUPLED large-step w tendency (state.w_tend, carried as the unused leaf) into
rw_tend, and re-run the warm bubble.

Because the operational AcousticCoreState does not currently carry w_tend, we
instead reconstruct the coupled advect_w tendency from the haloed state inside
the patched _augment path is not reachable here; simplest: recompute the coupled
w-advection from the substep-entry state w/u/v is not equal to the stage value.

So this probe instead patches _acoustic_core_state_from_prep to fold the
stage advect_w into rw_tend_stage.

Run: PYTHONPATH=src taskset -c 0-3 python -u scripts/f7i_wadv_fix_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

import gpuwrf.runtime.operational_mode as opmod
from gpuwrf.dynamics.advection import advect_w_face, halo_spec
from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry

ORIG_BUILD = opmod._acoustic_core_state_from_prep


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
        # positive-theta centroid height
        z = jnp.arange(s.theta.shape[0], dtype=jnp.float64) * 250.0 + 125.0
        thp = jnp.maximum(s.theta[:, 0, :] - 300.0, 0.0)
        tot = jnp.sum(thp)
        cz = jnp.where(tot > 1e-9, jnp.sum(thp * z[:, None]) / jnp.maximum(tot, 1e-12), 0.0)
        return {
            "max_abs_w": jnp.max(jnp.abs(s.w)),
            "zz_energy_center": jnp.sum(curv * curv),
            "finite": (jnp.all(jnp.isfinite(s.w))).astype(jnp.float64),
            "centroid_z": cz,
            "thp_max": jnp.max(s.theta[:, 0, :] - 300.0),
        }

    d = jax.device_get(go(carry0))
    return {k: float(v) for k, v in d.items()}


def main() -> int:
    ic = int(round(10000.0 / 250.0))
    out = {}

    # baseline (unpatched)
    opmod._acoustic_core_state_from_prep = ORIG_BUILD
    out["baseline_t100"] = run(1000, ic)

    # patched: add coupled advect_w(stage state) into rw_tend_pg_buoy
    def patched_build(carry, prep, pressure, namelist, tendencies):
        acoustic = ORIG_BUILD(carry, prep, pressure, namelist, tendencies)
        metrics = namelist.metrics
        # tendencies.w already carries the COUPLED large-step w tendency
        # (advect_w * mass_f + diffusion), built in _augment_large_step_tendencies.
        rw = acoustic.rw_tend_pg_buoy
        if tendencies is not None and tendencies.w is not None and rw is not None:
            rw = rw + tendencies.w
        return acoustic.replace(rw_tend_pg_buoy=rw)

    opmod._acoustic_core_state_from_prep = patched_build
    out["wadv_t100"] = run(1000, ic)
    out["wadv_t500"] = run(5000, ic)
    out["wadv_t900"] = run(9000, ic)

    opmod._acoustic_core_state_from_prep = ORIG_BUILD

    for k, v in out.items():
        print(
            f"{k:14s} fin={v['finite']>0.5}  max|w|={v['max_abs_w']:.4e}  "
            f"zz_energy={v['zz_energy_center']:.4e}  centroid_z={v['centroid_z']:.1f}  thp_max={v['thp_max']:.3f}",
            flush=True,
        )
    Path("proofs/f7i").mkdir(parents=True, exist_ok=True)
    Path("proofs/f7i/wadv_fix_probe.json").write_text(json.dumps(out, indent=2))
    print("wrote proofs/f7i/wadv_fix_probe.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
