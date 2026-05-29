#!/usr/bin/env python
"""F7I: net vertical forcing decomposition at the bubble-center column.

Captures a mid-run state and decomposes the per-substep vertical-momentum
forcing on w into:
  (1) stage pg_buoy_w  rw_tend  (g*rdn*dp_full - g*c1f*mu')
  (2) in-solver buoyancy term    dts*g*msft_inv*(rdn*d(c2a*alt*t_2ave) - c1f*muave)
  (3) in-solver pressure off-centering term (c2a*(rhs/ph) gradient)

If (1) and (2) are both large and same-sign, that is a double-count of buoyancy.
If they cancel, the residual is the true acoustic increment.  We evaluate at the
stage-entry t_2ave (=0 fresh stage) and after a few substeps.

Run: PYTHONPATH=src taskset -c 0-3 python -u scripts/f7i_net_vert_forcing.py --steps 100
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
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    metrics = namelist.metrics
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))
    ic = int(round(10000.0 / 250.0))
    g = GRAVITY_M_S2

    @jax.jit
    def step_to(carry):
        steps = jnp.arange(args.steps, dtype=jnp.int32) + 1
        final, _ = jax.lax.scan(
            lambda cc, st: (_physics_boundary_step(cc, namelist, st, run_radiation=False), None), carry, steps
        )
        return final

    carry = step_to(carry0)
    state = apply_halo(carry.state, halo_spec(namelist.grid))

    mu_base = _base_mu(state)
    base_state = BaseState(
        pb=(state.p_total - state.p_perturbation),
        phb=(state.ph_total - state.ph_perturbation),
        mub=mu_base,
        t0=jnp.asarray(300.0),
        theta_base=jnp.full_like(state.theta, 300.0),
    )
    grid_p_full, al_full, alt_full = diagnose_pressure_al_alt(state, base_state, metrics)
    mu_prime = state.mu_perturbation

    # (1) stage pg_buoy_w forcing (full grid%p)
    rw_tend_full = pg_buoy_w_dry(
        grid_p_full, mu_prime,
        c1f=metrics.c1f, rdnw=metrics.rdnw, rdn=metrics.rdn, msfty=metrics.msfty, gravity=g,
    )

    # (1b) stage pg_buoy_w forcing from the WORK-DELTA pressure (pre-F7H form)
    rw_tend_work = pg_buoy_w_dry(
        state.p_perturbation, mu_prime,
        c1f=metrics.c1f, rdnw=metrics.rdnw, rdn=metrics.rdn, msfty=metrics.msfty, gravity=g,
    )

    # (2) in-solver buoyancy term at FRESH-stage t_2ave (WRF builds t_2ave from
    # t_2 during substeps; at stage entry the work-theta t_2=0 -> t_2ave_next is a
    # function only of t0/t_1).  Evaluate the WRF normalization at t_2=0, muave=0,
    # muts=mut, t_1=theta':
    nz = int(state.theta.shape[0])
    theta_prime = state.theta - 300.0
    c2a = alt_full  # placeholder; recompute below from EOS-consistent c2a
    # Use the operational c2a = cpovcv*(p_total)/alt
    from gpuwrf.runtime.operational_mode import CPOVCV
    c2a = CPOVCV * state.p_total / jnp.maximum(jnp.abs(alt_full), 1e-12)
    muts = mu_base + mu_prime
    mass_h = metrics.c1h[:, None, None] * muts[None, :, :] + metrics.c2h[:, None, None]
    # t_2ave_next with t_2=0, muave=0:  (0.5*(1-eps)*t_2ave_prev=0) + 0 ) / (mass_h*(t0+t_1))
    # On a fresh stage t_2ave_prev=0 too -> t_2ave_next ~ 0.  So in-solver buoyancy
    # at fresh stage start ~0; it BUILDS from t_2 evolving.  Evaluate instead the
    # buoyancy with t_2ave = theta'/(t0+theta') as a representative scale:
    t2ave_scale = theta_prime / (300.0 + theta_prime)
    buoy = c2a * alt_full * t2ave_scale  # mass-level
    rdn = metrics.rdn[:, None, None]
    inb = jnp.zeros((nz + 1,) + tuple(theta_prime.shape[1:]), dtype=theta_prime.dtype)
    inb = inb.at[1:nz, :, :].set(g * (rdn[1:nz] * (buoy[1:nz] - buoy[: nz - 1])))

    def col(a):
        return np.asarray(a[:, 0, ic]).tolist()

    rec = {
        "step": args.steps, "second": args.steps * float(case.dt_s), "center_col": ic,
        "rw_tend_full_center": col(rw_tend_full),
        "rw_tend_work_center": col(rw_tend_work),
        "insolver_buoy_scale_center": col(inb),
        "grid_p_full_center": col(grid_p_full),
        "p_work_center": col(state.p_perturbation),
        "theta_prime_center": col(theta_prime),
        "max_abs_rw_full": float(jnp.max(jnp.abs(rw_tend_full[1:nz]))),
        "max_abs_rw_work": float(jnp.max(jnp.abs(rw_tend_work[1:nz]))),
    }
    print(f"t={rec['second']}s  max|rw_full(int)|={rec['max_abs_rw_full']:.2f}  max|rw_work(int)|={rec['max_abs_rw_work']:.4f}", flush=True)
    print("rw_tend_full (face) :", " ".join(f"{v:+.1f}" for v in rec["rw_tend_full_center"][:20]), flush=True)
    print("rw_tend_work (face) :", " ".join(f"{v:+.3f}" for v in rec["rw_tend_work_center"][:20]), flush=True)
    print("p_full (mass)       :", " ".join(f"{v:+.1f}" for v in rec["grid_p_full_center"][:20]), flush=True)
    print("p_work (mass)       :", " ".join(f"{v:+.3f}" for v in rec["p_work_center"][:20]), flush=True)

    Path("proofs/f7i").mkdir(parents=True, exist_ok=True)
    Path(f"proofs/f7i/net_vert_forcing_{args.steps}.json").write_text(json.dumps(rec, indent=2))
    print(f"wrote proofs/f7i/net_vert_forcing_{args.steps}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
