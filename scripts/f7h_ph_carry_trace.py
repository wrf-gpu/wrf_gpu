#!/usr/bin/env python
"""F7H Phase 1 — per-acoustic-substep ph-carry trace.

The leading hypothesis: the geopotential perturbation is FROZEN.
``ph_perturbation = ph_work + ph_save`` stays at ~131.83 while w grows
linearly, i.e. the acoustic-scan geopotential WORK array ``ph_work``
(advance_w's ``ph_next``) is ~0 and not responding to w through the scan.

This script reproduces ONE operational warm-bubble step (RK1/RK2/RK3) and
records, per acoustic substep AND per RK stage:

    max|w|        (coupled small-step w work array)
    max|ph_work|  (acoustic.ph -- the advance_w ph_next work array)
    max|t_2ave|   (the buoyancy theta half-average feeding advance_w term B)
    max|p|        (substep calc_p_rho work pressure)
    max|al|       (substep inverse-density perturbation)
    max|rw_tend|  (the once-per-stage pg_buoy_w forcing carried unchanged)

It also records the FINISHED-stage ``ph_perturbation`` after small_step_finish
so we can see whether ph_work + ph_save evolves the physical ph'.

Run:  PYTHONPATH=src taskset -c 0-3 python -u scripts/f7h_ph_carry_trace.py
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
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import dry_cqw
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import (
    _RKStageDescriptor,
    _acoustic_core_state_from_prep,
    _augment_large_step_tendencies,
    _carry_from_finished_stage,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


def _m(x):
    return float(jnp.max(jnp.abs(jnp.asarray(x))))


def trace_one_step(carry, namelist):
    """Run one RK3 step with per-substep ph-work tracing (eager, host-side)."""
    grid = namelist.grid
    origin = apply_halo(carry.state, halo_spec(grid))
    rk1_reference = origin
    carry = carry.replace(state=origin)
    dt = float(namelist.dt_s)
    configured = int(namelist.acoustic_substeps)
    stages = (
        _RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        _RKStageDescriptor(2, 0.5 * dt, dt / float(configured), max(1, configured // 2)),
        _RKStageDescriptor(3, dt, dt / float(configured), configured),
    )

    records = []
    for stage in stages:
        haloed = apply_halo(carry.state, halo_spec(grid))
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, grid)
        tendencies = _augment_large_step_tendencies(haloed, tendencies, namelist, rk_step=int(stage.rk_step))
        candidate = apply_halo(carry.state, halo_spec(grid))
        prep = small_step_prep_wrf(
            candidate, int(stage.rk_step), float(stage.dt_rk),
            metrics=namelist.metrics, reference_state=rk1_reference, ww=carry.ww,
        )
        pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        acoustic = _acoustic_core_state_from_prep(carry.replace(state=candidate), prep, pressure, namelist, tendencies)

        cqw_field = dry_cqw(
            int(prep.theta_work.shape[0]), int(prep.theta_work.shape[1]),
            int(prep.theta_work.shape[2]), dtype=prep.theta_work.dtype,
        )
        a, alpha, gamma = calc_coef_w_wrf_coefficients(
            prep.mut, namelist.metrics, dt=float(stage.dts_rk),
            epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
            cqw=cqw_field, c2a=prep.c2a,
        )
        stage_cfg = AcousticCoreConfig(
            dt=float(stage.dts_rk), dx=float(grid.projection.dx_m), dy=float(grid.projection.dy_m),
            epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
            w_damping=int(namelist.w_damping), damp_opt=int(namelist.damp_opt),
            dampcoef=float(namelist.dampcoef), zdamp=float(namelist.zdamp),
        )

        rw_tend_stage = _m(acoustic.rw_tend_pg_buoy)
        ph_save_max = _m(prep.ph_save)
        ph_work_entry = _m(acoustic.ph)

        # entry record (substep 0 = pre-acoustic)
        records.append({
            "rk_stage": int(stage.rk_step), "substep": 0,
            "max_w": _m(acoustic.w), "max_ph_work": ph_work_entry,
            "max_t_2ave": _m(acoustic.t_2ave), "max_p": _m(acoustic.p),
            "max_al": _m(acoustic.al) if acoustic.al is not None else 0.0,
            "max_rw_tend": rw_tend_stage, "ph_save_max": ph_save_max,
        })

        cur = acoustic
        for sub in range(int(stage.number_of_small_timesteps)):
            cur = acoustic_substep_core(cur, a=a, alpha=alpha, gamma=gamma, cfg=stage_cfg, cqw=cqw_field)
            records.append({
                "rk_stage": int(stage.rk_step), "substep": sub + 1,
                "max_w": _m(cur.w), "max_ph_work": _m(cur.ph),
                "max_t_2ave": _m(cur.t_2ave), "max_p": _m(cur.p),
                "max_al": _m(cur.al) if cur.al is not None else 0.0,
                "max_rw_tend": rw_tend_stage, "ph_save_max": ph_save_max,
            })

        next_carry = _carry_from_finished_stage(carry, prep, cur)
        carry = next_carry.replace(state=apply_halo(next_carry.state, halo_spec(grid)))
        records[-1]["finished_ph_pert_max"] = _m(carry.state.ph_perturbation)
        records[-1]["finished_p_pert_max"] = _m(carry.state.p_perturbation)
        records[-1]["finished_w_max"] = _m(carry.state.w)

    return carry, records


def main() -> int:
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    print(f"device={jax.devices()[0]}  dt={case.dt_s}s  substeps={namelist.acoustic_substeps}", flush=True)

    all_steps = []
    n_steps = 5
    for step in range(n_steps):
        carry, recs = trace_one_step(carry, namelist)
        all_steps.append({"step": step + 1, "records": recs})
        print(f"\n=== operational step {step+1} (t={(step+1)*case.dt_s:.1f}s) ===", flush=True)
        for r in recs:
            tag = f"  RK{r['rk_stage']} sub{r['substep']:2d}"
            extra = ""
            if "finished_ph_pert_max" in r:
                extra = f"  -> FINISH ph'={r['finished_ph_pert_max']:.4f} p'={r['finished_p_pert_max']:.3e} w={r['finished_w_max']:.3e}"
            print(f"{tag} max|w|={r['max_w']:.4e} max|ph_work|={r['max_ph_work']:.4e} "
                  f"max|t2ave|={r['max_t_2ave']:.4e} max|p|={r['max_p']:.3e} max|al|={r['max_al']:.3e} "
                  f"rw_tend(stage)={r['max_rw_tend']:.4e}{extra}", flush=True)

    out = Path("proofs/f7h/ph_carry_trace.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": "warm_bubble", "dt_s": case.dt_s,
                               "substeps": int(namelist.acoustic_substeps),
                               "steps": all_steps}, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
