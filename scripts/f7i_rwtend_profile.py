#!/usr/bin/env python
"""F7I: dump the once-per-stage pg_buoy_w rw_tend vertical profile at center column.

The term-ablation probe proved the growing vertical mode is driven by the
once-per-RK-stage rw_tend_pg_buoy (NOT theta advection, NOT the implicit solve).
This probe steps the warm bubble to a mid-run state and then reconstructs the
RK3-stage rw_tend exactly as the operational code does, dumping the center-column
vertical profile of grid_p_full, rw_tend, w, ph', theta' so the oscillatory
structure of the forcing is directly visible.

Run: PYTHONPATH=src taskset -c 0-3 python -u scripts/f7i_rwtend_profile.py --steps 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.dynamics.acoustic_wrf import diagnose_pressure_al_alt
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, pg_buoy_w_dry
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.contracts.state import BaseState
from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step, _base_mu, apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.runtime.operational_state import initial_operational_carry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("proofs/f7i/rwtend_profile.json"))
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    metrics = namelist.metrics
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))
    ic = int(round(10000.0 / 250.0))

    @jax.jit
    def step_to(carry):
        steps = jnp.arange(args.steps, dtype=jnp.int32) + 1
        final, _ = jax.lax.scan(
            lambda cc, st: (_physics_boundary_step(cc, namelist, st, run_radiation=False), None), carry, steps
        )
        return final

    carry = step_to(carry0)
    state = apply_halo(carry.state, halo_spec(namelist.grid))

    # Reconstruct the RK3-stage pg_buoy_w rw_tend exactly as operational code does.
    mu_base = _base_mu(state)
    theta_base = jnp.full_like(state.theta, 300.0)
    base_state = BaseState(
        pb=(state.p_total - state.p_perturbation),
        phb=(state.ph_total - state.ph_perturbation),
        mub=mu_base,
        t0=jnp.asarray(300.0),
        theta_base=theta_base,
    )
    grid_p_full, al_full, alt_full = diagnose_pressure_al_alt(state, base_state, metrics)
    mu_prime = state.mu_perturbation
    rw_tend = pg_buoy_w_dry(
        grid_p_full, mu_prime,
        c1f=metrics.c1f, rdnw=metrics.rdnw, rdn=metrics.rdn, msfty=metrics.msfty, gravity=GRAVITY_M_S2,
    )

    def col(a, faces=False):
        return np.asarray(a[:, 0, ic]).tolist()

    rec = {
        "step": args.steps,
        "second": args.steps * float(case.dt_s),
        "center_col": ic,
        "grid_p_full_center": col(grid_p_full),          # mass levels
        "rw_tend_center": col(rw_tend),                   # faces
        "w_center": col(state.w),                         # faces
        "ph_pert_center": col(state.ph_perturbation),     # faces
        "theta_prime_center": np.asarray(state.theta[:, 0, ic] - 300.0).tolist(),
        "al_full_center": col(al_full),
        "alt_full_center": col(alt_full),
        "max_abs_w": float(jnp.max(jnp.abs(state.w))),
    }

    print(f"t={rec['second']}s ic={ic} max|w|={rec['max_abs_w']:.3e}", flush=True)
    print("grid_p_full (mass) center:", " ".join(f"{v:+.1f}" for v in rec["grid_p_full_center"]), flush=True)
    print("rw_tend (face) center    :", " ".join(f"{v:+.3f}" for v in rec["rw_tend_center"]), flush=True)
    print("w (face) center          :", " ".join(f"{v:+.2f}" for v in rec["w_center"]), flush=True)
    print("theta' (mass) center     :", " ".join(f"{v:+.3f}" for v in rec["theta_prime_center"]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
