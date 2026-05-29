#!/usr/bin/env python
"""F7H Phase 1d — is mu' growth physical or spurious?

Trace per acoustic substep within each RK stage:
  max|mu_work| (acoustic.muts - acoustic.mut)  -- the WRF small-step mass delta
  max|mu|      (acoustic.mu = physical perturbation)
  max|muave|
  max|dmdt|    proxy: change in mu_work this substep / dts
  max|ww|      (coupled omega)
  max|u_work|, max|v_work|

A bounded oscillation = acoustic adjustment (physical). A monotone ramp = the
mass divergence is being driven by a non-cancelling forcing (the w runaway
feeding back into the column-integrated mass flux).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import dry_cqw
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

    print(f"device={jax.devices()[0]} dt={dt}", flush=True)
    for step in range(3):
        origin = apply_halo(carry.state, halo_spec(grid))
        rk1_reference = origin
        carry = carry.replace(state=origin)
        stages = (
            _RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
            _RKStageDescriptor(2, 0.5 * dt, dt / configured, max(1, configured // 2)),
            _RKStageDescriptor(3, dt, dt / configured, configured),
        )
        print(f"\n=== step {step+1} t={(step+1)*dt:.1f}s ===", flush=True)
        for stage in stages:
            haloed = apply_halo(carry.state, halo_spec(grid))
            tend = compute_advection_tendencies(haloed, namelist.tendencies, grid)
            tend = _augment_large_step_tendencies(haloed, tend, namelist, rk_step=int(stage.rk_step))
            candidate = apply_halo(carry.state, halo_spec(grid))
            prep = small_step_prep_wrf(candidate, int(stage.rk_step), float(stage.dt_rk),
                                       metrics=metrics, reference_state=rk1_reference, ww=carry.ww)
            pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
            acoustic = _acoustic_core_state_from_prep(carry.replace(state=candidate), prep, pressure, namelist, tend)
            nz = int(prep.theta_work.shape[0])
            cqw_field = dry_cqw(nz, int(prep.theta_work.shape[1]), int(prep.theta_work.shape[2]), dtype=prep.theta_work.dtype)
            a, alpha, gamma = calc_coef_w_wrf_coefficients(prep.mut, metrics, dt=float(stage.dts_rk),
                                                           epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
                                                           cqw=cqw_field, c2a=prep.c2a)
            cfg = AcousticCoreConfig(dt=float(stage.dts_rk), dx=float(grid.projection.dx_m), dy=float(grid.projection.dy_m),
                                     epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
                                     w_damping=int(namelist.w_damping), damp_opt=int(namelist.damp_opt),
                                     dampcoef=float(namelist.dampcoef), zdamp=float(namelist.zdamp))
            cur = acoustic
            mw0 = _m(cur.muts - cur.mut)
            print(f"  RK{stage.rk_step} dts={stage.dts_rk:.4f} entry mu_work={mw0:.3e} mu={_m(cur.mu):.3e} u_work={_m(cur.u):.3e}", flush=True)
            for sub in range(int(stage.number_of_small_timesteps)):
                cur = acoustic_substep_core(cur, a=a, alpha=alpha, gamma=gamma, cfg=cfg, cqw=cqw_field)
                mw = _m(cur.muts - cur.mut)
                print(f"     sub{sub+1:2d} mu_work={mw:.4e} mu={_m(cur.mu):.4e} muave={_m(cur.muave):.3e} "
                      f"ww={_m(cur.ww):.3e} w={_m(cur.w):.3e} u={_m(cur.u):.3e} v={_m(cur.v):.3e}", flush=True)
            nc = _carry_from_finished_stage(carry, prep, cur)
            carry = nc.replace(state=apply_halo(nc.state, halo_spec(grid)))
            print(f"     -> finished mu'={_m(carry.state.mu_perturbation):.4e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
