"""PROTOTYPE (NOT shipped): backward-Euler implicit sedimentation for Thompson.

This is the single large lever for the Thompson kernel's dominant cost (the
sedimentation substep loop = ~85 % of the kernel). It replaces WRF's 64 explicit
sub-stepped upwind passes per species with ONE backward-Euler upwind vertical
sweep (unconditionally stable, no CFL substeps). Measured ~2.4x on the full
Thompson kernel on the d02 grid.

WHY IT IS NOT SHIPPED (it is an ALGORITHM change, not a faithful drop-in):
  * Backward-Euler upwind is more numerically diffusive than WRF's small-Courant
    explicit sub-stepping. On a moist test column the POINT-WISE vertical profile
    differs by O(1) relative (qr max_rel ~2.6 vs the converged explicit NSED=128),
    i.e. it smears the falling fronts and shifts the vertical hydrometeor (and
    latent-heating) distribution.
  * BUT the INTEGRATED quantities are well preserved: total surface precip within
    ~1.0 % and total column hydrometeor mass within ~0.3-1 % of the converged
    explicit scheme.
  * vt is frozen over the full dt here (semi-implicit); WRF recomputes vt each
    substep. A Picard corrector (1-2 vt recomputes) would reduce the fast-species
    arrival bias -- recommended before any adoption.
  * Adopting it therefore requires a precipitating WRF Thompson oracle (the
    current oracle savepoint is a dry/clear column -- all fall speeds 0, so it
    CANNOT discriminate this change) PLUS a 6-24h coupled precip/T2/U10/V10 skill
    comparison vs CPU-WRF. That is a manager-gated scheme change / ADR, not a
    lane-local default flip. GPT-5.5 xhigh independently concurred (see report).

Measured (full Thompson kernel, 20748-col tiled d02 grid, median of 120 reps):
  base explicit (per-species, 64 substeps) : 33.8 ms
  implicit backward-Euler (1 sweep)        : 13.9 ms  -> 2.4x
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import gpuwrf.physics.thompson_column as tc


def sed_implicit_q(q, vt, dz, rho, dt, nsub: int = 1):
    """Backward-Euler upwind sedimentation of one field in ``nsub`` implicit sweeps.

    Axis -1 is vertical, index 0 = surface, last = model top.  Sedimentation
    flux enters a layer only from the layer ABOVE (higher index), so the implicit
    solve is a top->bottom recurrence:
        (1 + dt_s*vt_k/dz_k) q_k' = q_k + (dt_s/(rho_k dz_k)) * rho_{k+1} vt_{k+1} q_{k+1}'
    ``nsub=1`` is the single-sweep full-step backward Euler; nsub>1 reduces the
    implicit diffusion at proportional extra cost.  Returns (q', surface_precip_mm).
    """

    acc = jnp.result_type(q.dtype, vt.dtype, rho.dtype, dz.dtype)
    q = q.astype(acc)
    dts = float(dt) / nsub
    qr0 = jnp.moveaxis(q, -1, 0)[::-1]      # (z, ...) z=0 == model top
    vtr = jnp.moveaxis(vt, -1, 0)[::-1]
    rhor = jnp.moveaxis(rho, -1, 0)[::-1]
    dzr = jnp.moveaxis(dz, -1, 0)[::-1]
    nz = qr0.shape[0]
    diag = 1.0 + dts * vtr / dzr

    def one(qcur):
        def body(inflow_mass, k):
            qk = (qcur[k] + dts / (rhor[k] * dzr[k]) * inflow_mass) / diag[k]
            return rhor[k] * vtr[k] * qk, qk
        _, qsol = jax.lax.scan(body, jnp.zeros(qcur.shape[1:], acc), jnp.arange(nz))
        surf = rhor[nz - 1] * vtr[nz - 1] * qsol[nz - 1]  # bottom (surface) flux
        return jnp.maximum(qsol, 0.0), surf

    def step(carry, _):
        qc, sacc = carry
        qsol, surf = one(qc)
        return (qsol, sacc + surf * dts), None

    (qf, sf), _ = jax.lax.scan(step, (qr0, jnp.zeros(qr0.shape[1:], acc)), None, length=nsub)
    qf = jnp.moveaxis(jnp.maximum(qf, 0.0)[::-1], 0, -1)
    return qf.astype(q.dtype), sf


def sedimentation_implicit(state, dt: float, nsub: int = 1):
    """Drop-in-shaped replacement for ``_sedimentation`` using implicit sweeps."""

    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = tc._fall_speeds(state)
    dz = jnp.maximum(state.dz, 1.0)
    rho = jnp.maximum(state.rho, tc.R1)
    qr, pr = sed_implicit_q(state.qr, vt_r_mass, dz, rho, dt, nsub)
    Nr, _ = sed_implicit_q(state.Nr, vt_r_num, dz, rho, dt, nsub)
    qi, pi = sed_implicit_q(state.qi, vt_i_mass, dz, rho, dt, nsub)
    Ni, _ = sed_implicit_q(state.Ni, vt_i_num, dz, rho, dt, nsub)
    qs, ps = sed_implicit_q(state.qs, vt_s_mass, dz, rho, dt, nsub)
    Ns, _ = sed_implicit_q(state.Ns, vt_s_mass, dz, rho, dt, nsub)
    qg, pg = sed_implicit_q(state.qg, vt_g_mass, dz, rho, dt, nsub)
    Ng, _ = sed_implicit_q(state.Ng, vt_g_num, dz, rho, dt, nsub)
    updated = state.replace(qr=qr, Nr=Nr, qi=qi, Ni=Ni, qs=qs, Ns=Ns, qg=qg, Ng=Ng)
    return updated, {"rain": pr, "ice": pi, "snow": ps, "graupel": pg}
