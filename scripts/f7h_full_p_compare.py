#!/usr/bin/env python
"""F7H Phase 1c — confirm the stage grid%p inconsistency.

Hypothesis (refined): pg_buoy_w in WRF consumes grid%p = the FULL-perturbation
calc_p_rho_phi diagnostic (built from the FULL ph', mu', theta'), so its
rdn*(p[k]-p[k-1]) PGF term hydrostatically balances the -c1f*mu' term for a
near-balanced thermal -> small net rw_tend.

Our operational stage feeds pg_buoy_w the calc_p_rho_wrf(prep) WORK-delta
pressure (built from prep.ph_work ~0, prep.mu_work ~0) which does NOT carry the
ph'/mu' structure -> the two terms do not cancel -> growing net forcing.

This script, per RK stage, compares for the SAME entry state:
  rw_tend(work_p)  -- current operational path (calc_p_rho_wrf prep)
  rw_tend(full_p)  -- WRF-faithful (diagnose_pressure_al_alt full-perturbation p)
and prints the PGF vs mu terms for each.  If full_p makes the terms cancel
(small rw_tend) while work_p does not, the fix is to feed pg_buoy_w the full p.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from gpuwrf.contracts.state import BaseState
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.acoustic_wrf import diagnose_pressure_al_alt
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, dry_cqw, pg_buoy_w_dry
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.ic_generators.idealized import (
    _build_setup, _enforce_operational_precision, build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import (
    _RKStageDescriptor, _augment_large_step_tendencies, _carry_from_finished_stage,
    _acoustic_core_state_from_prep,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


def _m(x):
    return float(jnp.max(jnp.abs(jnp.asarray(x))))


def main() -> int:
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    metrics = namelist.metrics
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))
    grid = namelist.grid
    dt = float(namelist.dt_s)
    configured = int(namelist.acoustic_substeps)
    theta0 = 300.0

    print(f"device={jax.devices()[0]} dt={dt}", flush=True)
    out = []
    for step in range(6):
        origin = apply_halo(carry.state, halo_spec(grid))
        rk1_reference = origin
        carry = carry.replace(state=origin)
        stages = (
            _RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
            _RKStageDescriptor(2, 0.5 * dt, dt / configured, max(1, configured // 2)),
            _RKStageDescriptor(3, dt, dt / configured, configured),
        )
        recs = []
        for stage in stages:
            haloed = apply_halo(carry.state, halo_spec(grid))
            tend = compute_advection_tendencies(haloed, namelist.tendencies, grid)
            tend = _augment_large_step_tendencies(haloed, tend, namelist, rk_step=int(stage.rk_step))
            candidate = apply_halo(carry.state, halo_spec(grid))
            prep = small_step_prep_wrf(candidate, int(stage.rk_step), float(stage.dt_rk),
                                       metrics=metrics, reference_state=rk1_reference, ww=carry.ww)
            pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
            mu_prime = prep.mut - prep.mub

            # FULL-perturbation grid%p via diagnose_pressure_al_alt (WRF calc_p_rho_phi)
            base = BaseState(pb=prep.pb,
                             phb=(candidate.ph_total - candidate.ph_perturbation),
                             mub=prep.mub, t0=jnp.asarray(theta0),
                             theta_base=jnp.full_like(candidate.theta, theta0))
            full_p, full_al, full_alt = diagnose_pressure_al_alt(candidate, base, metrics)

            nz = int(full_p.shape[0])
            def decomp(p):
                pgf = GRAVITY_M_S2 * (metrics.rdn[1:nz, None, None] * (p[1:nz] - p[:nz-1]))
                mu_t = GRAVITY_M_S2 * (metrics.c1f[1:nz, None, None] * mu_prime[None, :, :])
                interior = pgf - mu_t
                rw = pg_buoy_w_dry(p, mu_prime, c1f=metrics.c1f, rdnw=metrics.rdnw,
                                   rdn=metrics.rdn, msfty=metrics.msfty, gravity=GRAVITY_M_S2)
                return _m(pgf), _m(mu_t), _m(interior), _m(rw)

            wp = decomp(pressure.p)
            fp = decomp(full_p)
            recs.append({"rk": int(stage.rk_step), "mu_prime": _m(mu_prime),
                         "work_p_max": _m(pressure.p), "full_p_max": _m(full_p),
                         "work_pgf": wp[0], "work_mu": wp[1], "work_interior_net": wp[2], "work_rw": wp[3],
                         "full_pgf": fp[0], "full_mu": fp[1], "full_interior_net": fp[2], "full_rw": fp[3]})

            # advance stage on the CURRENT (work_p) path to keep evolving exactly as production
            acoustic = _acoustic_core_state_from_prep(carry.replace(state=candidate), prep, pressure, namelist, tend)
            cqw_field = dry_cqw(nz, int(full_p.shape[1]), int(full_p.shape[2]), dtype=full_p.dtype)
            a, alpha, gamma = calc_coef_w_wrf_coefficients(prep.mut, metrics, dt=float(stage.dts_rk),
                                                           epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
                                                           cqw=cqw_field, c2a=prep.c2a)
            cfg = AcousticCoreConfig(dt=float(stage.dts_rk), dx=float(grid.projection.dx_m), dy=float(grid.projection.dy_m),
                                     epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
                                     w_damping=int(namelist.w_damping), damp_opt=int(namelist.damp_opt),
                                     dampcoef=float(namelist.dampcoef), zdamp=float(namelist.zdamp))
            cur = acoustic
            for _ in range(int(stage.number_of_small_timesteps)):
                cur = acoustic_substep_core(cur, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw_field)
            nc = _carry_from_finished_stage(carry, prep, cur)
            carry = nc.replace(state=apply_halo(nc.state, halo_spec(grid)))
        out.append({"step": step + 1, "records": recs})
        print(f"\n=== step {step+1} t={(step+1)*dt:.1f}s ===", flush=True)
        for r in recs:
            print(f"  RK{r['rk']} mu'={r['mu_prime']:.3e} | WORK_p: max={r['work_p_max']:.3e} pgf={r['work_pgf']:.3e} "
                  f"mu={r['work_mu']:.3e} net={r['work_interior_net']:.3e} rw={r['work_rw']:.3e}", flush=True)
            print(f"        {'':6s}| FULL_p: max={r['full_p_max']:.3e} pgf={r['full_pgf']:.3e} "
                  f"mu={r['full_mu']:.3e} net={r['full_interior_net']:.3e} rw={r['full_rw']:.3e}", flush=True)

    Path("proofs/f7h").mkdir(parents=True, exist_ok=True)
    Path("proofs/f7h/full_p_compare.json").write_text(json.dumps({"steps": out}, indent=2))
    print("\nwrote proofs/f7h/full_p_compare.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
