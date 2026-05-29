"""F7N: JAX per-acoustic-substep touchdown-column dump for the Straka density current.

Mirrors the operational RK3/acoustic cadence in
``gpuwrf.runtime.operational_mode._rk_scan_step`` EXACTLY (same
``_augment_large_step_tendencies`` -> ``small_step_prep_wrf`` -> ``calc_p_rho_wrf``
-> ``_acoustic_core_state_from_prep`` -> ``acoustic_substep_core`` ->
``_carry_from_finished_stage`` chain), but replaces the per-stage ``lax.scan``
over acoustic substeps with a Python ``for`` loop so each substep's
``AcousticCoreState`` can be snapshotted at the cold-pool TOUCHDOWN column
(domain center x=0, the JAX mass index nearest x=0) and its x-neighbours.

The instrumented window is the touchdown band itimestep 170..205 (t = 170..205 s,
dt = 1 s in WRF; JAX dt = 0.1 s with 10 substeps/step is time-equivalent -- we
run the JAX step at dt=1.0 with 10 substeps to match WRF's per-step acoustic count
and make the per-substep diff directly comparable).

We run steps 1..169 with the fast jitted segment, then steps 170..205 with the
instrumented loop, dumping per (step, rk, substep) at columns center-1/center/center+1.

Output: /mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_substeps.json (JAX side)
"""
from __future__ import annotations

import argparse
import json
import os

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.core.acoustic import AcousticCoreState, acoustic_substep_core, AcousticCoreConfig
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.runtime import operational_mode as om
from gpuwrf.ic_generators.idealized import (
    build_density_current_numpy,
    _build_setup,
    _initial_carry,
    _ready_carry,
    _run_segment_jit,
    _snapshot,
)

# WRF dump field set (per column, per level). The acoustic-PGF u-tendency is
# recovered offline as (u_after_substep - u_before_substep)/dts per substep; we
# dump u directly so the diff can compute it.
DUMP_FIELDS = ("w", "ph", "p", "rw_tend", "ph_tend", "ww", "u_iface", "u_ip1face", "v", "muts", "mut", "muave", "t_2ave")


def _stage_descriptors(dt: float, sound_steps: int):
    return (
        om._RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        om._RKStageDescriptor(2, 0.5 * dt, dt / float(sound_steps), max(1, sound_steps // 2)),
        om._RKStageDescriptor(3, dt, dt / float(sound_steps), sound_steps),
    )


def _col_snapshot(acoustic: AcousticCoreState, center: int) -> dict:
    """Extract touchdown-column (center-1, center, center+1) per-level fields.

    Arrays are (nz[,+1], ny=1, nx[+stagger]); y index is 0.  u is x-staggered
    (nx+1 faces): u_iface = u[:, 0, i]; u_ip1face = u[:, 0, i+1].
    """
    cols = [center - 1, center, center + 1]
    out = {}
    w = np.asarray(acoustic.w[:, 0, :])          # (nz+1, nx)
    ph = np.asarray(acoustic.ph[:, 0, :])        # (nz+1, nx)
    p = np.asarray(acoustic.p[:, 0, :])          # (nz, nx)
    rw = np.asarray(acoustic.rw_tend_pg_buoy[:, 0, :]) if acoustic.rw_tend_pg_buoy is not None else None
    pht = np.asarray(acoustic.ph_tend[:, 0, :])  # (nz+1, nx)
    ww = np.asarray(acoustic.ww[:, 0, :])        # (nz+1, nx)
    u = np.asarray(acoustic.u[:, 0, :])          # (nz, nx+1)
    v = np.asarray(acoustic.v[:, 0, :])          # (nz, nx) (ny+1=2 -> use row 0)
    muts = np.asarray(acoustic.muts[0, :])       # (nx,)
    mut = np.asarray(acoustic.mut[0, :])         # (nx,)
    muave = np.asarray(acoustic.muave[0, :])     # (nx,)
    t2a = np.asarray(acoustic.t_2ave[:, 0, :])   # (nz, nx)
    for ci in cols:
        out[str(ci)] = {
            "w": w[:, ci].tolist(),
            "ph": ph[:, ci].tolist(),
            "p": p[:, ci].tolist(),
            "rw_tend": (rw[:, ci].tolist() if rw is not None else None),
            "ph_tend": pht[:, ci].tolist(),
            "ww": ww[:, ci].tolist(),
            "u_iface": u[:, ci].tolist(),
            "u_ip1face": u[:, ci + 1].tolist(),
            "v": v[:, ci].tolist(),
            "muts": float(muts[ci]),
            "mut": float(mut[ci]),
            "muave": float(muave[ci]),
            "t_2ave": t2a[:, ci].tolist(),
        }
    return out


def _instrumented_step(carry, namelist, center, records, itimestep, dt, sound_steps):
    """One operational dt step (RK3) with per-substep touchdown snapshots."""
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin
    carry = carry.replace(state=origin)
    stages = _stage_descriptors(dt, sound_steps)

    for stage in stages:
        haloed = apply_halo(carry.state, halo_spec(namelist.grid))
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        tendencies = om._augment_large_step_tendencies(
            haloed, tendencies, namelist, rk_step=int(stage.rk_step)
        )
        candidate = apply_halo(carry.state, halo_spec(namelist.grid))
        prep = small_step_prep_wrf(
            candidate,
            int(stage.rk_step),
            float(stage.dt_rk),
            metrics=namelist.metrics,
            reference_state=rk1_reference,
            ww=carry.ww,
        )
        pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        acoustic = om._acoustic_core_state_from_prep(
            carry.replace(state=candidate), prep, pressure, namelist, tendencies
        )
        from gpuwrf.dynamics.core.advance_w import dry_cqw
        from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
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
            dt=float(stage.dts_rk),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm), top_lid=bool(namelist.top_lid),
            w_damping=int(namelist.w_damping), damp_opt=int(namelist.damp_opt),
            dampcoef=float(namelist.dampcoef), zdamp=float(namelist.zdamp),
        )
        for sub in range(int(stage.number_of_small_timesteps)):
            acoustic = acoustic_substep_core(
                acoustic, a=a, alpha=alpha, gamma=gamma, cfg=stage_cfg, cqw=cqw_field,
            )
            records.append({
                "itimestep": int(itimestep),
                "rk_step": int(stage.rk_step),
                "iteration": int(sub + 1),
                "nsmall": int(stage.number_of_small_timesteps),
                "cols": _col_snapshot(acoustic, center),
            })
        carry = om._carry_from_finished_stage(carry, prep, acoustic, namelist)
        carry = carry.replace(state=apply_halo(carry.state, halo_spec(namelist.grid)))
    return carry


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=170)
    p.add_argument("--end", type=int, default=205)
    p.add_argument("--out", type=str, default="/mnt/data/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_substeps.json")
    args = p.parse_args()

    case = build_density_current_numpy()
    # Run the dump at dt=1.0 with 10 substeps/step so the per-step acoustic count
    # and per-substep dts match WRF (dt=1, time_step_sound=6 -> RK3 uses 6 sound
    # steps; we use the operational 10).  itimestep here == seconds.
    setup = _build_setup(case, require_gpu=True)
    nl = setup.namelist
    # The operational namelist uses dt_s = case.dt_s (0.1). Re-make at dt=1.0 so a
    # "step" is 1 s, matching WRF's per-step structure and the touchdown timestep
    # window 170..205 s.  Acoustic substeps stay at 10.
    from dataclasses import replace as dataclass_replace
    nl = dataclass_replace(nl, dt_s=1.0)
    sound_steps = int(nl.acoustic_substeps)

    nx = case.nx
    # touchdown column = JAX mass index nearest x=0 (domain center).
    center = int(np.argmin(np.abs(case.x_m)))
    print(f"nx={nx} center_idx={center} x_m[center]={case.x_m[center]:.1f} dt={nl.dt_s} sound={sound_steps}", flush=True)

    carry = _initial_carry(setup.state)
    _ready_carry(carry).block_until_ready()

    # fast-forward to start-1 with the jitted segment (radiation off -> step index
    # irrelevant; reuse start_step=1 so one compile).
    warm = int(args.start) - 1
    if warm > 0:
        carry = _run_segment_jit(carry, nl, start_step=1, steps=warm)
        _ready_carry(carry).block_until_ready()
    snap = _snapshot(case, carry.state, float(warm))
    print(f"warmup t={warm}s finite={snap['finite']} maxw={snap['max_abs_w_m_s']} "
          f"thmin={snap['theta_prime_min_k']} front={snap.get('front_position_m')}", flush=True)

    records = []
    for it in range(int(args.start), int(args.end) + 1):
        carry = _instrumented_step(carry, nl, center, records, it, float(nl.dt_s), sound_steps)
        _ready_carry(carry).block_until_ready()
        snap = _snapshot(case, carry.state, float(it))
        finite = snap["finite"]
        print(f"t={it}s finite={finite} maxw={snap['max_abs_w_m_s']} "
              f"thmin={snap['theta_prime_min_k']} front={snap.get('front_position_m')} "
              f"maxu={snap.get('max_abs_u_m_s')}", flush=True)
        if not finite:
            print("  -> NONFINITE: stopping dump after this step", flush=True)
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "case": "straka_density_current_jax",
        "center_mass_index": center,
        "x_center_m": float(case.x_m[center]),
        "z_m": case.z_m.tolist(),
        "dt_s": float(nl.dt_s),
        "acoustic_substeps": sound_steps,
        "dump_fields": list(DUMP_FIELDS),
        "records": records,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out} ({len(records)} substep records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
