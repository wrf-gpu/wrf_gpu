#!/usr/bin/env python
"""V0.14 Switzerland h36 RK1 advance_w/phi discriminator.

This is a proof-only offline harness.  It rebuilds the first RK1 acoustic
substep from the h36 state already proven identical to WRF call 21601, compares
the resulting stage boundary against WRF call 21602, and tests whether the
known non-WRF surface-w choice in ``advance_w`` explains the first interior
``ph/p`` divergence.

The production path deliberately feeds decoupled physical ``u_1/v_1`` into the
terrain-following lower-boundary ``w`` calculation for stability.  WRF feeds the
coupled acoustic work arrays ``u_2/v_2``.  The variant here changes only that
input to the line-ported ``advance_w_wrf`` call and leaves every other staged
term untouched.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HPG_SPEC = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

_BLOCKER_SPEC = importlib.util.spec_from_file_location(
    "acoustic_substep_blocker", Path(__file__).with_name("switzerland_acoustic_substep_blocker.py")
)
blocker = importlib.util.module_from_spec(_BLOCKER_SPEC)
_BLOCKER_SPEC.loader.exec_module(blocker)  # type: ignore[union-attr]

OUT_JSON = ROOT / "proofs/v014/switzerland_advance_w_phi_discriminator.json"


def _stats(arr: np.ndarray) -> dict[str, float]:
    v = np.asarray(arr, dtype=np.float64)
    if v.size == 0:
        return {"count": 0}
    return {
        "count": int(v.size),
        "mean": float(np.nanmean(v)),
        "rmse": float(np.sqrt(np.nanmean(v * v))),
        "max_abs": float(np.nanmax(np.abs(v))),
    }


def _interior_mask(ny: int, nx: int, depth: int = 8) -> np.ndarray:
    jj, ii = np.mgrid[0:ny, 0:nx]
    return (ii >= depth) & (ii < nx - depth) & (jj >= depth) & (jj < ny - depth)


def _split(diff: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float64)
    if diff.ndim == 2:
        ny, nx = diff.shape
        m2 = _interior_mask(ny, nx)
        return {"full": _stats(diff), "interior": _stats(diff[m2]), "band": _stats(diff[~m2])}
    _, ny, nx = diff.shape
    m2 = _interior_mask(ny, nx)
    m3 = np.broadcast_to(m2[None], diff.shape)
    return {"full": _stats(diff), "interior": _stats(diff[m3]), "band": _stats(diff[~m3])}


def _ring_mean(diff2: np.ndarray, max_ring: int = 12) -> dict[str, float]:
    diff2 = np.asarray(diff2, dtype=np.float64)
    ny, nx = diff2.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    ring = np.minimum(np.minimum(ii, nx - 1 - ii), np.minimum(jj, ny - 1 - jj))
    out: dict[str, float] = {}
    for r in range(max_ring):
        sel = ring == r
        if np.any(sel):
            out[str(r)] = float(np.nanmean(diff2[sel]))
    return out


def _pin_spec_ring(field, target):
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG

    out = field
    for b in range(int(DEFAULT_BOUNDARY_CONFIG.spec_zone)):
        out = out.at[..., b, :].set(target[..., b, :])
        out = out.at[..., field.shape[-2] - 1 - b, :].set(target[..., field.shape[-2] - 1 - b, :])
        out = out.at[..., :, b].set(target[..., :, b])
        out = out.at[..., :, field.shape[-1] - 1 - b].set(target[..., :, field.shape[-1] - 1 - b])
    return out


def _run_one_substep(acoustic_state, prep, pressure, namelist, stage, carry, *, surface_mode: str):
    """Run stage-1's single acoustic substep with a selectable surface-w feed."""

    import jax.numpy as jnp
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG, spec_bdyupdate_ph_inloop
    from gpuwrf.dynamics.acoustic_wrf import GRAVITY_M_S2
    from gpuwrf.dynamics.core import acoustic as ac
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf, dry_cqw, pg_buoy_w_dry
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_step
    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
    from gpuwrf.runtime import operational_mode as om

    cqw_field = acoustic_state.cqw
    if cqw_field is None:
        cqw_field = dry_cqw(
            int(prep.theta_work.shape[0]),
            int(prep.theta_work.shape[1]),
            int(prep.theta_work.shape[2]),
            dtype=prep.theta_work.dtype,
        )
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        prep.mut,
        namelist.metrics,
        dt=float(stage.dts_rk),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        cqw=cqw_field,
        c2a=prep.c2a,
    )
    periodic_x, specified, nested = om._acoustic_lateral_bc_flags(namelist)
    cfg = ac.AcousticCoreConfig(
        dt=float(stage.dts_rk),
        dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        w_damping=int(namelist.w_damping),
        damp_opt=int(namelist.damp_opt),
        dampcoef=float(namelist.dampcoef),
        zdamp=float(namelist.zdamp),
        dt_full=float(namelist.dt_s),
        periodic_x=periodic_x,
        specified=specified,
        nested=nested,
        spec_w_zero_grad=bool(om._specified_bdy_cadence_active(namelist)),
    )

    state = acoustic_state
    uv_state = ac.advance_uv_wrf(
        state,
        dts_rk=float(cfg.dt),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        top_lid=bool(cfg.top_lid),
        emdiv=float(getattr(namelist, "emdiv", 0.01)),
        dt_full=float(namelist.dt_s),
    )
    coupled_state = uv_state.replace(theta=uv_state.theta_coupled_work)
    advanced = ac.advance_mu_t_core(coupled_state, cfg)
    theta_coupled = advanced["theta"]
    ww_new = advanced["ww"]
    muave_new = advanced["muave"]
    muts_new = advanced["muts"]
    mu_new = advanced["mu"]
    mudf_new = advanced["mudf"]

    if state.mu_spec_target is not None:
        mu_new = _pin_spec_ring(mu_new, state.mu_spec_target)
        muts_new = _pin_spec_ring(muts_new, state.muts_spec_target)
        muave_new = _pin_spec_ring(muave_new, state.muave_spec_target)
        theta_coupled = _pin_spec_ring(theta_coupled, state.theta_spec_target)

    state_for_w = uv_state.replace(
        mu=mu_new,
        muts=muts_new,
        muave=muave_new,
        ww=ww_new,
        mudf=mudf_new,
        theta=theta_coupled,
    )

    nz = int(state_for_w.theta.shape[0])
    ny = int(state_for_w.theta.shape[1])
    nx = int(state_for_w.theta.shape[2])
    cqw_w = cqw_field
    c2a = state_for_w.c2a if state_for_w.c2a is not None else jnp.ones_like(state_for_w.theta)
    alt = state_for_w.alt if state_for_w.alt is not None else jnp.ones_like(state_for_w.theta)
    phb = state_for_w.phb if state_for_w.phb is not None else jnp.zeros_like(state_for_w.ph)
    ph_1 = state_for_w.ph_1 if state_for_w.ph_1 is not None else jnp.zeros_like(state_for_w.ph)
    ht = state_for_w.ht if state_for_w.ht is not None else jnp.zeros((ny, nx), dtype=state_for_w.theta.dtype)
    c1f = state_for_w.c1f if state_for_w.c1f is not None else jnp.zeros((nz + 1,), dtype=state_for_w.theta.dtype)
    c2f = state_for_w.c2f if state_for_w.c2f is not None else jnp.zeros((nz + 1,), dtype=state_for_w.theta.dtype)
    rdn = state_for_w.rdn if state_for_w.rdn is not None else state_for_w.rdnw
    cf1 = ac._optional_or(state_for_w.cf1, jnp.asarray(0.0, dtype=state_for_w.theta.dtype))
    cf2 = ac._optional_or(state_for_w.cf2, jnp.asarray(0.0, dtype=state_for_w.theta.dtype))
    cf3 = ac._optional_or(state_for_w.cf3, jnp.asarray(0.0, dtype=state_for_w.theta.dtype))

    mu_work = muts_new - state_for_w.mut
    if state_for_w.rw_tend_pg_buoy is not None:
        rw_tend = state_for_w.rw_tend_pg_buoy
    else:
        p_for_buoy = state_for_w.p_buoy if state_for_w.p_buoy is not None else state_for_w.p
        rw_tend = pg_buoy_w_dry(
            p_for_buoy,
            mu_work,
            c1f=c1f,
            rdnw=state_for_w.rdnw,
            rdn=rdn,
            msfty=state_for_w.msfty,
            gravity=GRAVITY_M_S2,
        )

    if surface_mode == "current_decoupled":
        u_surface = state_for_w.u_1
        v_surface = state_for_w.v_1
    elif surface_mode == "wrf_coupled":
        u_surface = state_for_w.u
        v_surface = state_for_w.v
    else:
        raise ValueError(f"unknown surface_mode={surface_mode}")

    w_solved, ph_next, t_2ave_next = advance_w_wrf(
        w=state_for_w.w,
        rw_tend=rw_tend,
        ww=ww_new,
        u=u_surface,
        v=v_surface,
        mu_work=mu_work,
        mut=state_for_w.mut,
        muave=muave_new,
        muts=muts_new,
        t_2ave=state_for_w.t_2ave,
        t_2=theta_coupled,
        t_1=state_for_w.theta_1,
        ph=state_for_w.ph,
        ph_1=ph_1,
        phb=phb,
        ph_tend=state_for_w.ph_tend,
        ht=ht,
        c2a=c2a,
        cqw=cqw_w,
        alt=alt,
        a=a,
        alpha=alpha,
        gamma=gamma,
        c1h=state_for_w.c1h,
        c2h=state_for_w.c2h,
        c1f=c1f,
        c2f=c2f,
        rdnw=state_for_w.rdnw,
        rdn=rdn,
        fnm=state_for_w.fnm,
        fnp=state_for_w.fnp,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        msftx=state_for_w.msftx,
        msfty=state_for_w.msfty,
        rdx=1.0 / float(cfg.dx),
        rdy=1.0 / float(cfg.dy),
        dts=float(cfg.dt),
        epssm=float(cfg.epssm),
        top_lid=bool(cfg.top_lid),
        gravity=GRAVITY_M_S2,
        w_save=state_for_w.w_save,
        damp_opt=int(cfg.damp_opt),
        dampcoef=float(cfg.dampcoef),
        zdamp=float(cfg.zdamp),
        w_damping=int(cfg.w_damping),
        w_alpha=float(cfg.w_alpha),
        w_crit_cfl=float(cfg.w_crit_cfl),
    )

    if state_for_w.ph_bdy_target is not None and state_for_w.ph_save_for_spec is not None:
        ph_next = spec_bdyupdate_ph_inloop(
            ph_next,
            state_for_w.ph_bdy_target,
            state_for_w.ph_save_for_spec,
            mu_tend=None,
            muts=muts_new,
            c1f=c1f,
            c2f=c2f,
            dts=float(cfg.dt),
            config=DEFAULT_BOUNDARY_CONFIG,
        )

    if bool(cfg.spec_w_zero_grad):
        w_solved = w_solved.at[:, 0, :].set(w_solved[:, 1, :])
        w_solved = w_solved.at[:, -1, :].set(w_solved[:, -2, :])
        w_solved = w_solved.at[:, 1:-1, 0].set(w_solved[:, 1:-1, 1])
        w_solved = w_solved.at[:, 1:-1, -1].set(w_solved[:, 1:-1, -2])

    state_for_pressure = state_for_w.replace(w=w_solved, ph=ph_next, t_2ave=t_2ave_next)
    ru_m = (state_for_pressure.ru_m if state_for_pressure.ru_m is not None else jnp.zeros_like(state_for_pressure.u)) + state_for_pressure.u
    rv_m = (state_for_pressure.rv_m if state_for_pressure.rv_m is not None else jnp.zeros_like(state_for_pressure.v)) + state_for_pressure.v
    ww_m = (state_for_pressure.ww_m if state_for_pressure.ww_m is not None else jnp.zeros_like(state_for_pressure.ww)) + ww_new
    pm1 = state_for_pressure.pm1 if state_for_pressure.pm1 is not None else state_for_pressure.p
    p_rho = calc_p_rho_step(
        mu_work=mu_work,
        muts_total=muts_new,
        ph_work=ph_next,
        theta_work=theta_coupled,
        theta_1=state_for_pressure.theta_1,
        c2a=c2a,
        alt=alt,
        c1h=state_for_pressure.c1h,
        c2h=state_for_pressure.c2h,
        rdnw=state_for_pressure.rdnw,
        pm1=pm1,
        smdiv=float(getattr(namelist, "smdiv", 0.1)),
        t0=300.0,
    )
    theta_phys = ac._decouple_theta_for_finish(state_for_pressure, theta_coupled, muts_new)
    acoustic_out = state_for_pressure.replace(
        mu=mu_new,
        mudf=mudf_new,
        muts=muts_new,
        muave=muave_new,
        ww=ww_new,
        theta=theta_phys,
        theta_coupled_work=theta_coupled,
        theta_ave=theta_phys,
        w=w_solved,
        ph=ph_next,
        p=p_rho.p,
        al=p_rho.al,
        pm1=p_rho.pm1,
        t_2ave=t_2ave_next,
        ru_m=ru_m,
        rv_m=rv_m,
        ww_m=ww_m,
    )
    next_carry = om._carry_from_finished_stage(carry, prep, acoustic_out, namelist)
    return acoustic_out, next_carry, {
        "surface_mode": surface_mode,
        "surface_w_work": np.asarray(w_solved[0], dtype=np.float64),
        "pre_zero_grad_w_work": np.asarray(w_solved, dtype=np.float64),
        "ph_work_after_advance_w": np.asarray(ph_next, dtype=np.float64),
        "p_work_after_calc_p_rho_step": np.asarray(p_rho.p, dtype=np.float64),
        "rw_tend": np.asarray(rw_tend, dtype=np.float64),
        "ph_tend": np.asarray(state_for_w.ph_tend, dtype=np.float64),
        "ww_after_advance_mu_t": np.asarray(ww_new, dtype=np.float64),
        "state_for_w_u_work": np.asarray(state_for_w.u, dtype=np.float64),
        "state_for_w_v_work": np.asarray(state_for_w.v, dtype=np.float64),
        "u_surface_input": np.asarray(u_surface, dtype=np.float64),
        "v_surface_input": np.asarray(v_surface, dtype=np.float64),
    }


def _build_stage1():
    import jax.numpy as jnp
    from gpuwrf.contracts.halo import apply_halo
    from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
    from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
    from gpuwrf.dynamics.flux_advection import stage_omega_specified
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry

    case, state0, run_dir = hpg._build_state(hpg.NATIVE_ROOT)
    namelist = dataclasses.replace(
        case.namelist,
        dt_s=18.0,
        acoustic_substeps=4,
        specified_bdy_cadence=True,
        specified_adv_degrade=True,
    )
    carry0 = initial_operational_carry(state0)
    lead_seconds = jnp.asarray(1, dtype=jnp.int32).astype(jnp.float64) * float(namelist.dt_s)
    forcing = om._physics_step_forcing(
        carry0,
        namelist,
        lead_seconds,
        run_radiation=bool(namelist.run_physics),
        first_timestep=True,
    )
    carry = forcing.carry
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin
    stage = om._RKStageDescriptor(1, float(namelist.dt_s) / 3.0, float(namelist.dt_s) / 3.0, 1)

    carry = carry.replace(state=origin)
    haloed = apply_halo(carry.state, halo_spec(namelist.grid))
    tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    stage_velocities = (
        om._stage_transport_velocities(haloed, namelist)
        if bool(namelist.use_flux_advection)
        else None
    )
    bdy_relax = om._specified_bdy_relax(rk1_reference, namelist, lead_seconds)
    tendencies = om._augment_large_step_tendencies(
        haloed,
        tendencies,
        namelist,
        rk_step=1,
        physics_tendencies=forcing.dry_tendencies,
        step_origin=rk1_reference,
        transport_velocities=stage_velocities,
        bdy_relax=bdy_relax,
    )
    candidate = apply_halo(carry.state, halo_spec(namelist.grid))
    _periodic_x, specified, nested = om._acoustic_lateral_bc_flags(namelist)
    if stage_velocities is not None and (specified or nested):
        ww_stage = stage_omega_specified(
            haloed.u,
            haloed.v,
            haloed.mu_total,
            c1h=namelist.metrics.c1h,
            c2h=namelist.metrics.c2h,
            dnw=namelist.metrics.dnw,
            rdx=1.0 / float(namelist.grid.projection.dx_m),
            rdy=1.0 / float(namelist.grid.projection.dy_m),
            msfuy=namelist.metrics.msfuy,
            msfvx=namelist.metrics.msfvx,
            msftx=namelist.metrics.msftx,
        )
    elif stage_velocities is not None:
        ww_stage = stage_velocities.rom
    else:
        ww_stage = carry.ww
    prep = small_step_prep_wrf(
        candidate,
        1,
        float(stage.dt_rk),
        metrics=namelist.metrics,
        reference_state=rk1_reference,
        ww=ww_stage,
    )
    pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
    acoustic = om._acoustic_core_state_from_prep(
        carry,
        prep,
        pressure,
        namelist,
        tendencies,
        lead_seconds=lead_seconds,
        bdy_relax=bdy_relax,
    )
    return {
        "run_dir": str(run_dir),
        "namelist": namelist,
        "carry": carry,
        "prep": prep,
        "pressure": pressure,
        "stage": stage,
        "acoustic": acoustic,
        "wrf_base": hpg.assemble_wrf_call(21601),
        "wrf_stage1": hpg.assemble_wrf_call(21602),
        "base_capture": blocker.observe_state(state0, namelist),
    }


def _compare_variant(name: str, next_carry, base_capture, wrf_base, wrf_stage1, namelist) -> dict[str, Any]:
    cap = blocker.observe_state(next_carry.state, namelist)
    cmp = blocker.compare_capture_to_wrf(cap, base_capture, wrf_stage1, wrf_base)
    return {
        "name": name,
        "stage1_increment_rmse": {
            fld: {
                "full": cmp[fld]["incr_err"]["full"].get("rmse"),
                "interior": cmp[fld]["incr_err"]["interior"].get("rmse"),
                "band": cmp[fld]["incr_err"]["band"].get("rmse"),
                "max_abs_interior": cmp[fld]["incr_err"]["interior"].get("max_abs"),
                "mean_interior": cmp[fld]["incr_err"]["interior"].get("mean"),
            }
            for fld in ("mu", "p", "ph", "al", "alt", "w")
            if fld in cmp
        },
        "stage1_state_rmse": {
            fld: {
                "full": cmp[fld]["state_err"]["full"].get("rmse"),
                "interior": cmp[fld]["state_err"]["interior"].get("rmse"),
                "band": cmp[fld]["state_err"]["band"].get("rmse"),
            }
            for fld in ("mu", "p", "ph", "al", "alt", "w")
            if fld in cmp
        },
    }


def main() -> int:
    import jax

    ctx = _build_stage1()
    variants: dict[str, Any] = {}
    debug: dict[str, Any] = {}
    for mode in ("current_decoupled", "wrf_coupled"):
        acoustic_out, next_carry, dbg = _run_one_substep(
            ctx["acoustic"],
            ctx["prep"],
            ctx["pressure"],
            ctx["namelist"],
            ctx["stage"],
            ctx["carry"],
            surface_mode=mode,
        )
        jax.block_until_ready(next_carry.state.u)
        variants[mode] = _compare_variant(
            mode,
            next_carry,
            ctx["base_capture"],
            ctx["wrf_base"],
            ctx["wrf_stage1"],
            ctx["namelist"],
        )
        debug[mode] = dbg

    current_dbg = debug["current_decoupled"]
    coupled_dbg = debug["wrf_coupled"]
    surface_delta = coupled_dbg["surface_w_work"] - current_dbg["surface_w_work"]
    ph_work_delta = coupled_dbg["ph_work_after_advance_w"] - current_dbg["ph_work_after_advance_w"]
    p_work_delta = coupled_dbg["p_work_after_calc_p_rho_step"] - current_dbg["p_work_after_calc_p_rho_step"]
    u_input_delta = coupled_dbg["u_surface_input"] - current_dbg["u_surface_input"]
    v_input_delta = coupled_dbg["v_surface_input"] - current_dbg["v_surface_input"]

    cur = variants["current_decoupled"]["stage1_increment_rmse"]
    alt = variants["wrf_coupled"]["stage1_increment_rmse"]
    verdict = {
        "surface_w_deviation_explains_first_ph_p_error": bool(
            alt["ph"]["interior"] < 0.5 * cur["ph"]["interior"]
            and alt["p"]["interior"] < 0.5 * cur["p"]["interior"]
        ),
        "current_reproduces_prior_stage1": {
            "p_interior_rmse": cur["p"]["interior"],
            "ph_interior_rmse": cur["ph"]["interior"],
            "expected_prior_advdeg_p": 1.1261975184532773,
            "expected_prior_advdeg_ph": 0.4352639584631776,
        },
        "wrf_coupled_surface_delta": {
            "p_interior_rmse": alt["p"]["interior"],
            "ph_interior_rmse": alt["ph"]["interior"],
            "p_improvement_fraction": float(1.0 - alt["p"]["interior"] / max(cur["p"]["interior"], 1e-30)),
            "ph_improvement_fraction": float(1.0 - alt["ph"]["interior"] / max(cur["ph"]["interior"], 1e-30)),
        },
    }

    payload = {
        "schema": "v014_switzerland_advance_w_phi_discriminator",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "anchor": "h36 state == WRF call 21601; compare first RK1 stage boundary against WRF call 21602",
        "config": {
            "dt_s": 18.0,
            "acoustic_substeps": 4,
            "stage": 1,
            "stage_dt": 6.0,
            "specified_bdy_cadence": True,
            "specified_adv_degrade": True,
            "top_lid": bool(ctx["namelist"].top_lid),
            "epssm": float(ctx["namelist"].epssm),
        },
        "verdict": verdict,
        "variants": variants,
        "variant_deltas": {
            "wrf_coupled_minus_current_surface_w_work": {
                **_split(surface_delta[None, :, :]),
                "ring_mean": _ring_mean(surface_delta),
            },
            "wrf_coupled_minus_current_ph_work_after_advance_w": _split(ph_work_delta),
            "wrf_coupled_minus_current_p_work_after_calc_p_rho_step": _split(p_work_delta),
            "wrf_coupled_minus_current_u_surface_input": _split(u_input_delta),
            "wrf_coupled_minus_current_v_surface_input": _split(v_input_delta),
        },
        "notes": [
            "This tests only the surface-w feed inside advance_w; all ph_tend, rw_tend, coefficients, theta/mu, pressure refresh, boundary cadence, and advection-degradation settings are unchanged.",
            "If the wrf_coupled variant does not materially reduce both ph and p interior RMSE at call 21602, the known surface-w deviation is not the first h36 phi/p root.",
        ],
    }
    hpg.write_json(OUT_JSON, payload)
    print(json.dumps({
        "backend": payload["backend"],
        "current": verdict["current_reproduces_prior_stage1"],
        "wrf_coupled_surface_delta": verdict["wrf_coupled_surface_delta"],
        "surface_w_explains": verdict["surface_w_deviation_explains_first_ph_p_error"],
        "out": str(OUT_JSON),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
