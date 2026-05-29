#!/usr/bin/env python
"""F7D: measure the stage-entry frozen buoyancy rw_tend vs the analytic physical
buoyancy b = g*theta'/theta0, to quantify the over-forcing that drives the
warm-bubble runaway and pin its source (p_buoy pressure-gradient term vs the
c1f*mu' term vs an IC discrete-balance residual).

Run:  taskset -c 0-3 python -u scripts/f7d_rwtend_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, pg_buoy_w_dry
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.ic_generators.idealized import _build_setup, _enforce_operational_precision, build_warm_bubble_numpy
from gpuwrf.runtime.operational_mode import (
    _acoustic_core_state_from_prep,
    _augment_large_step_tendencies,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


def main() -> int:
    setup = _build_setup(build_warm_bubble_numpy(), require_gpu=True)
    nl = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))
    haloed = apply_halo(carry.state, halo_spec(nl.grid))
    tend = compute_advection_tendencies(haloed, nl.tendencies, nl.grid)
    tend = _augment_large_step_tendencies(haloed, tend, nl, rk_step=1)
    prep = small_step_prep_wrf(haloed, 1, float(setup.numpy_case.dt_s) / 3.0,
                               metrics=nl.metrics, reference_state=haloed, ww=carry.ww)
    pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
    ac = _acoustic_core_state_from_prep(carry, prep, pressure, nl, tend)

    # rw_tend exactly as acoustic_substep_core builds it at substep 1.
    mu_work = ac.muts - ac.mut
    rw = pg_buoy_w_dry(ac.p_buoy, mu_work, c1f=ac.c1f, rdnw=ac.rdnw, rdn=ac.rdn,
                       msfty=ac.msfty, gravity=GRAVITY_M_S2)
    rw = np.asarray(jax.device_get(rw))  # coupled (nz+1, ny, nx); /mass = physical accel
    c1f = np.asarray(jax.device_get(ac.c1f)); c2f = np.asarray(jax.device_get(ac.c2f))
    mut = np.asarray(jax.device_get(ac.mut))  # (ny,nx)
    nz = rw.shape[0] - 1
    mass_f = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]  # (nz+1,ny,nx)
    rw_phys = rw / np.maximum(np.abs(mass_f), 1e-12)  # physical accel m/s^2 (msfty=1)
    thp = np.asarray(jax.device_get(prep.entry_state.theta[:, 0, :])) - 300.0
    b_analytic = GRAVITY_M_S2 * thp.max() / 300.0
    print(f"theta'max = {thp.max():.3f} K  ->  analytic buoyancy b=g*theta'/theta0 = {b_analytic:.4e} m/s^2", flush=True)
    print(f"max|rw_tend_phys| (coupled/mass) = {np.max(np.abs(rw_phys)):.4e} m/s^2", flush=True)
    print(f"ratio rw_phys / b_analytic = {np.max(np.abs(rw_phys))/b_analytic:.2f}", flush=True)
    print(f"max|p_buoy| = {float(jnp.max(jnp.abs(ac.p_buoy))):.4e} Pa", flush=True)
    # split: pressure-gradient term vs c1f*mu' term
    rdn = np.asarray(jax.device_get(ac.rdn)); rdnw = np.asarray(jax.device_get(ac.rdnw))
    pbuoy = np.asarray(jax.device_get(ac.p_buoy))[:, 0, :]  # (nz,nx)
    pg_term = rdn[1:nz, None] * (pbuoy[1:nz, :] - pbuoy[:nz-1, :])  # interior faces
    muw = np.asarray(jax.device_get(mu_work))[0, :]
    muterm = c1f[1:nz, None] * muw[None, :]
    print(f"max|pg_term (rdn*dp_buoy)| = {np.max(np.abs(pg_term)):.4e}", flush=True)
    print(f"max|c1f*mu' term| = {np.max(np.abs(muterm)):.4e}  (mu'={np.max(np.abs(muw)):.3e})", flush=True)
    out = Path("proofs/f7d/rwtend_check.json")
    out.write_text(json.dumps({
        "theta_prime_max_K": float(thp.max()),
        "analytic_buoyancy_m_s2": float(b_analytic),
        "max_abs_rw_phys_m_s2": float(np.max(np.abs(rw_phys))),
        "ratio_rw_over_analytic": float(np.max(np.abs(rw_phys)) / b_analytic),
        "max_abs_p_buoy_Pa": float(jnp.max(jnp.abs(ac.p_buoy))),
        "max_abs_pg_term": float(np.max(np.abs(pg_term))),
        "max_abs_c1f_mu_term": float(np.max(np.abs(muterm))),
    }, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
