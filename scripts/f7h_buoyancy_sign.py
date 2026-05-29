#!/usr/bin/env python
"""F7H — sign & magnitude of the warm-bubble buoyancy at the IC (full grid%p).

For a warm bubble (theta'>0) the net vertical forcing rw_tend at the bubble
center column should be POSITIVE (upward) in the lower half of the thermal.
Check the SIGN and column structure of:
  pg_buoy_w(full grid%p, mu'=0)  at the IC (rest, mu'=0)
vs the naive parcel buoyancy g*theta'/theta0.

If the IC (mu'=0) full-p pg_buoy_w is upward and ~ g*theta'/theta0 at the
bubble center, the operator is correct and the runaway is a downstream mu'
feedback.  If it is downward or wrong-magnitude, the buoyancy source itself is
mis-signed.
"""
from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.contracts.state import BaseState
from gpuwrf.dynamics.acoustic_wrf import diagnose_pressure_al_alt
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, pg_buoy_w_dry
from gpuwrf.ic_generators.idealized import (
    _build_setup, _enforce_operational_precision, build_warm_bubble_numpy,
)


def main() -> int:
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    metrics = setup.namelist.metrics
    state = _enforce_operational_precision(setup.state, force_fp64=True)
    theta0 = 300.0
    xc = case.nx // 2

    mub = jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)
    pb = jnp.asarray(state.p_total) - jnp.asarray(state.p_perturbation)
    phb = jnp.asarray(state.ph_total) - jnp.asarray(state.ph_perturbation)
    base = BaseState(pb=pb, phb=phb, mub=mub, t0=jnp.asarray(theta0),
                     theta_base=jnp.full_like(state.theta, theta0))
    full_p, al, alt = diagnose_pressure_al_alt(state, base, metrics)
    mu_prime = jnp.asarray(state.mu_perturbation)  # =0 at IC
    rw = pg_buoy_w_dry(full_p, mu_prime, c1f=metrics.c1f, rdnw=metrics.rdnw,
                       rdn=metrics.rdn, msfty=metrics.msfty, gravity=GRAVITY_M_S2)

    th = np.asarray(jax.device_get(state.theta[:, 0, xc])) - theta0
    rwc = np.asarray(jax.device_get(rw[:, 0, xc]))  # faces (nz+1,)
    pc = np.asarray(jax.device_get(full_p[:, 0, xc]))
    alc = np.asarray(jax.device_get(al[:, 0, xc]))
    z = case.z_m
    print(f"mu'(max)={float(jnp.max(jnp.abs(mu_prime))):.3e} (should be 0)")
    print(f"max|full_p|={float(jnp.max(jnp.abs(full_p))):.3e} Pa  max|al|={float(jnp.max(jnp.abs(al))):.3e}")
    print(f"max|rw_tend|={float(jnp.max(jnp.abs(rw))):.3e} m/s2")
    print("\n k   z(m)   theta'   full_p(Pa)   al        rw_tend(face,m/s2)  parcel g*th'/th0")
    for k in range(case.nz):
        parcel = GRAVITY_M_S2 * th[k] / theta0
        print(f"{k:3d} {z[k]:6.0f} {th[k]:+7.4f} {pc[k]:+10.4e} {alc[k]:+9.2e} "
              f"  rw[k]={rwc[k]:+10.4e}  {parcel:+8.4e}")
    # where is theta'>0?
    kpos = np.where(th > 0.01)[0]
    if len(kpos):
        print(f"\ntheta'>0 at k={kpos[0]}..{kpos[-1]} (z={z[kpos[0]]:.0f}..{z[kpos[-1]]:.0f}m)")
        print(f"rw_tend over that band (faces): min={rwc[kpos[0]:kpos[-1]+2].min():+.3e} "
              f"max={rwc[kpos[0]:kpos[-1]+2].max():+.3e}")
        print("Sign check: warm bubble => rw_tend should be POSITIVE in lower half, NEGATIVE upper half (dipole).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
