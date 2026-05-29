#!/usr/bin/env python
"""F7H Phase 1b — localize the growing pg_buoy_w stage forcing (rw_tend).

The ph_carry_trace showed rw_tend(stage) growing 2 -> 43 -> 56 -> 77 -> 111 ...
(O(100 m/s2), ~10x g) every RK stage and ratcheting step over step.  This is
the unbounded forcing driving the w runaway.  rw_tend = pg_buoy_w(pressure.p,
mu') built ONCE per stage from the stage diagnostic grid%p (calc_p_rho_wrf on
the prep).  Decompose: where does the growing input come from?

Per RK stage records:
  max|state.p_pert|     (entry physical perturbation pressure)
  max|prep.ph_work|     (RK reference - current ph')  -- 0 at RK1
  max|pressure.p|       (calc_p_rho_wrf stage grid%p; the pg_buoy_w input)
  max|pressure.al|
  max|mu_prime|         (mut - mub; should be ~0 for fixed-mass)
  max|rw_tend|          (pg_buoy_w output)
  -- decompose rw_tend into the rdn*(dp) PGF term vs the c1f*mu' term --
  max|pgf_term|         g*rdn*(p[k]-p[k-1])
  max|mu_term|          g*c1f*mu'
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, pg_buoy_w_dry
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.ic_generators.idealized import (
    _build_setup, _enforce_operational_precision, build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import (
    _RKStageDescriptor, _acoustic_core_state_from_prep,
    _augment_large_step_tendencies, _carry_from_finished_stage,
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

    print(f"device={jax.devices()[0]} dt={dt}", flush=True)
    out_steps = []
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
            tendencies = compute_advection_tendencies(haloed, namelist.tendencies, grid)
            tendencies = _augment_large_step_tendencies(haloed, tendencies, namelist, rk_step=int(stage.rk_step))
            candidate = apply_halo(carry.state, halo_spec(grid))
            prep = small_step_prep_wrf(candidate, int(stage.rk_step), float(stage.dt_rk),
                                       metrics=metrics, reference_state=rk1_reference, ww=carry.ww)
            pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
            mu_prime = prep.mut - prep.mub
            p = pressure.p
            nz = int(p.shape[0])
            pgf = GRAVITY_M_S2 * (metrics.rdn[1:nz, None, None] * (p[1:nz] - p[:nz-1]))
            mu_t = GRAVITY_M_S2 * (metrics.c1f[1:nz, None, None] * mu_prime[None, :, :])
            rw = pg_buoy_w_dry(p, mu_prime, c1f=metrics.c1f, rdnw=metrics.rdnw,
                               rdn=metrics.rdn, msfty=metrics.msfty, gravity=GRAVITY_M_S2)
            recs.append({
                "rk": int(stage.rk_step),
                "state_p_pert": _m(candidate.p_perturbation),
                "state_ph_pert": _m(candidate.ph_perturbation),
                "prep_ph_work": _m(prep.ph_work),
                "pressure_p": _m(p), "pressure_al": _m(pressure.al),
                "mu_prime": _m(mu_prime),
                "pgf_term": _m(pgf), "mu_term": _m(mu_t), "rw_tend": _m(rw),
            })
            acoustic = _acoustic_core_state_from_prep(carry.replace(state=candidate), prep, pressure, namelist, tendencies)
            # advance the stage (use the operational scan helper logic)
            from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core
            from gpuwrf.dynamics.core.advance_w import dry_cqw
            from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
            cqw_field = dry_cqw(nz, int(p.shape[1]), int(p.shape[2]), dtype=p.dtype)
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
        out_steps.append({"step": step + 1, "records": recs})
        print(f"\n=== step {step+1} t={(step+1)*dt:.1f}s ===", flush=True)
        for r in recs:
            print(f"  RK{r['rk']} state_p'={r['state_p_pert']:.4e} state_ph'={r['state_ph_pert']:.3f} "
                  f"prep_ph_work={r['prep_ph_work']:.3e} pressure.p={r['pressure_p']:.4e} al={r['pressure_al']:.3e} "
                  f"mu'={r['mu_prime']:.3e} | pgf={r['pgf_term']:.4e} muT={r['mu_term']:.3e} rw_tend={r['rw_tend']:.4e}", flush=True)

    Path("proofs/f7h").mkdir(parents=True, exist_ok=True)
    Path("proofs/f7h/rwtend_source.json").write_text(json.dumps({"steps": out_steps}, indent=2))
    print("\nwrote proofs/f7h/rwtend_source.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
