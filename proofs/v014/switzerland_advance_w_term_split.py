#!/usr/bin/env python
"""V0.14 Switzerland h36 ``advance_w`` term/input split.

Proof-only harness.  It reuses the h36 state already proven identical to WRF
call 21601, constructs the first RK1 acoustic substep up to the
``advance_w_wrf`` call, then runs focused one-term variants and compares the
resulting stage boundary against WRF call 21602.

This is intentionally not a JAX-vs-JAX acceptance test: every variant is scored
against the WRF-native call-21602 state assembled from the HPG dump.
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_SURF_SPEC = importlib.util.spec_from_file_location(
    "advance_w_surface_probe", Path(__file__).with_name("switzerland_advance_w_phi_discriminator.py")
)
surface_probe = importlib.util.module_from_spec(_SURF_SPEC)
_SURF_SPEC.loader.exec_module(surface_probe)  # type: ignore[union-attr]

_BLOCKER_SPEC = importlib.util.spec_from_file_location(
    "acoustic_substep_blocker", Path(__file__).with_name("switzerland_acoustic_substep_blocker.py")
)
blocker = importlib.util.module_from_spec(_BLOCKER_SPEC)
_BLOCKER_SPEC.loader.exec_module(blocker)  # type: ignore[union-attr]

_HPG_SPEC = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

OUT_JSON = ROOT / "proofs/v014/switzerland_advance_w_term_split.json"


def _install_cpu_allocator_shim() -> dict[str, Any]:
    """Proof-only shim: run replay constructors on CPU without editing production."""

    import jax
    from gpuwrf.contracts import state as state_contract

    cpu = jax.devices("cpu")[0]
    original_name = getattr(state_contract._gpu_device, "__name__", repr(state_contract._gpu_device))
    state_contract._gpu_device = lambda: cpu  # type: ignore[assignment]
    return {
        "shim": "gpuwrf.contracts.state._gpu_device -> first CPU device",
        "original_callable": original_name,
        "selected_device": str(cpu),
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _stats(arr: np.ndarray) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.nanmean(values)),
        "rmse": float(np.sqrt(np.nanmean(values * values))),
        "max_abs": float(np.nanmax(np.abs(values))),
    }


def _interior_mask(ny: int, nx: int, depth: int = 8) -> np.ndarray:
    jj, ii = np.mgrid[0:ny, 0:nx]
    return (ii >= depth) & (ii < nx - depth) & (jj >= depth) & (jj < ny - depth)


def _split(diff: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(diff, dtype=np.float64)
    if diff.ndim == 2:
        ny, nx = diff.shape
        mask = _interior_mask(ny, nx)
        return {"full": _stats(diff), "interior": _stats(diff[mask]), "band": _stats(diff[~mask])}
    _, ny, nx = diff.shape
    mask2 = _interior_mask(ny, nx)
    mask = np.broadcast_to(mask2[None], diff.shape)
    return {"full": _stats(diff), "interior": _stats(diff[mask]), "band": _stats(diff[~mask])}


def _scalar_summary(arr: Any) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "rmse": float(np.sqrt(np.nanmean(values * values))),
        "max_abs": float(np.nanmax(np.abs(values))),
    }


def _build_pre_advance_w(ctx: dict[str, Any]) -> dict[str, Any]:
    """Mirror the first half of the existing surface discriminator once."""

    import jax.numpy as jnp
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG
    from gpuwrf.dynamics.acoustic_wrf import GRAVITY_M_S2, calc_coef_w_wrf_coefficients
    from gpuwrf.dynamics.core import acoustic as ac
    from gpuwrf.dynamics.core.advance_w import dry_cqw, pg_buoy_w_dry
    from gpuwrf.runtime import operational_mode as om

    acoustic_state = ctx["acoustic"]
    prep = ctx["prep"]
    namelist = ctx["namelist"]
    stage = ctx["stage"]

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
    dry_cqw_field = dry_cqw(
        int(prep.theta_work.shape[0]),
        int(prep.theta_work.shape[1]),
        int(prep.theta_work.shape[2]),
        dtype=prep.theta_work.dtype,
    )
    a_dry, alpha_dry, gamma_dry = calc_coef_w_wrf_coefficients(
        prep.mut,
        namelist.metrics,
        dt=float(stage.dts_rk),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        cqw=dry_cqw_field,
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

    uv_state = ac.advance_uv_wrf(
        acoustic_state,
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

    if acoustic_state.mu_spec_target is not None:
        mu_new = surface_probe._pin_spec_ring(mu_new, acoustic_state.mu_spec_target)
        muts_new = surface_probe._pin_spec_ring(muts_new, acoustic_state.muts_spec_target)
        muave_new = surface_probe._pin_spec_ring(muave_new, acoustic_state.muave_spec_target)
        theta_coupled = surface_probe._pin_spec_ring(theta_coupled, acoustic_state.theta_spec_target)

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
        rw_tend_stage = state_for_w.rw_tend_pg_buoy
    else:
        p_for_buoy = state_for_w.p_buoy if state_for_w.p_buoy is not None else state_for_w.p
        rw_tend_stage = pg_buoy_w_dry(
            p_for_buoy,
            mu_work,
            c1f=c1f,
            rdnw=state_for_w.rdnw,
            rdn=rdn,
            msfty=state_for_w.msfty,
            gravity=GRAVITY_M_S2,
        )
    rw_tend_dry = pg_buoy_w_dry(
        state_for_w.p_buoy if state_for_w.p_buoy is not None else state_for_w.p,
        mu_work,
        c1f=c1f,
        rdnw=state_for_w.rdnw,
        rdn=rdn,
        msfty=state_for_w.msfty,
        gravity=GRAVITY_M_S2,
    )

    return {
        "cfg": cfg,
        "state_for_w": state_for_w,
        "theta_coupled": theta_coupled,
        "ww_new": ww_new,
        "mu_new": mu_new,
        "mudf_new": mudf_new,
        "muts_new": muts_new,
        "muave_new": muave_new,
        "mu_work": mu_work,
        "rw_tend_stage": rw_tend_stage,
        "rw_tend_dry_recomputed": rw_tend_dry,
        "cqw_field": cqw_field,
        "dry_cqw_field": dry_cqw_field,
        "a": a,
        "alpha": alpha,
        "gamma": gamma,
        "a_dry": a_dry,
        "alpha_dry": alpha_dry,
        "gamma_dry": gamma_dry,
        "c2a": c2a,
        "alt": alt,
        "phb": phb,
        "ph_1": ph_1,
        "ht": ht,
        "c1f": c1f,
        "c2f": c2f,
        "rdn": rdn,
        "cf1": cf1,
        "cf2": cf2,
        "cf3": cf3,
        "boundary_config": DEFAULT_BOUNDARY_CONFIG,
    }


def _run_variant(ctx: dict[str, Any], pre: dict[str, Any], name: str, options: dict[str, Any]):
    import jax.numpy as jnp
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG, spec_bdyupdate_ph_inloop
    from gpuwrf.dynamics.acoustic_wrf import GRAVITY_M_S2
    from gpuwrf.dynamics.core import acoustic as ac
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_step
    from gpuwrf.runtime import operational_mode as om

    namelist = ctx["namelist"]
    carry = ctx["carry"]
    prep = ctx["prep"]
    cfg = pre["cfg"]
    s = pre["state_for_w"]

    ph_tend = s.ph_tend
    rw_tend = pre["rw_tend_stage"]
    ww = pre["ww_new"]
    cqw = pre["cqw_field"]
    a = pre["a"]
    alpha = pre["alpha"]
    gamma = pre["gamma"]
    u_surface = s.u_1
    v_surface = s.v_1
    calc_theta = pre["theta_coupled"]
    calc_muts = pre["muts_new"]
    calc_mu_work = pre["mu_work"]
    ph_finish_muts = pre["muts_new"]
    mu_for_finish = pre["mu_new"]
    muts_for_finish = pre["muts_new"]
    muave_for_finish = pre["muave_new"]
    ww_for_finish = pre["ww_new"]
    theta_for_finish = pre["theta_coupled"]
    advance_w_mu_work = pre["mu_work"]
    advance_w_muave = pre["muave_new"]

    if options.get("zero_ph_tend"):
        ph_tend = jnp.zeros_like(ph_tend)
    if options.get("zero_rw_tend"):
        rw_tend = jnp.zeros_like(rw_tend)
    if options.get("dry_recomputed_rw_tend"):
        rw_tend = pre["rw_tend_dry_recomputed"]
    if options.get("zero_phi_adv"):
        ww = jnp.zeros_like(ww)
    if options.get("wrf_coupled_surface"):
        u_surface = s.u
        v_surface = s.v
    if options.get("dry_cqw_and_coefficients"):
        cqw = pre["dry_cqw_field"]
        a = pre["a_dry"]
        alpha = pre["alpha_dry"]
        gamma = pre["gamma_dry"]
    if options.get("wrf_call21602_mu_inputs"):
        ny, nx = int(s.mu.shape[0]), int(s.mu.shape[1])
        wrf_mu_base = jnp.asarray(ctx["wrf_base"]["fields2"]["mu"][:ny, :nx], dtype=s.mu.dtype)
        wrf_mu_stage1 = jnp.asarray(ctx["wrf_stage1"]["fields2"]["mu"][:ny, :nx], dtype=s.mu.dtype)
        mu_delta = wrf_mu_stage1 - wrf_mu_base
        mu_for_finish = wrf_mu_stage1
        muts_for_finish = s.mut + mu_delta
        muave_for_finish = 0.5 * (1.0 + float(cfg.epssm)) * mu_delta
        advance_w_mu_work = mu_delta
        advance_w_muave = muave_for_finish
        calc_mu_work = mu_delta
        calc_muts = muts_for_finish
        ph_finish_muts = muts_for_finish
    if options.get("calc_p_rho_stage_mut_denominator"):
        calc_muts = s.mut
    if options.get("calc_p_rho_no_smdiv"):
        smdiv = 0.0
    else:
        smdiv = float(getattr(namelist, "smdiv", 0.1))

    w_solved, ph_next, t_2ave_next = advance_w_wrf(
        w=s.w,
        rw_tend=rw_tend,
        ww=ww,
        u=u_surface,
        v=v_surface,
        mu_work=advance_w_mu_work,
        mut=s.mut,
        muave=advance_w_muave,
        muts=ph_finish_muts,
        t_2ave=s.t_2ave,
        t_2=pre["theta_coupled"],
        t_1=s.theta_1,
        ph=s.ph,
        ph_1=pre["ph_1"],
        phb=pre["phb"],
        ph_tend=ph_tend,
        ht=pre["ht"],
        c2a=pre["c2a"],
        cqw=cqw,
        alt=pre["alt"],
        a=a,
        alpha=alpha,
        gamma=gamma,
        c1h=s.c1h,
        c2h=s.c2h,
        c1f=pre["c1f"],
        c2f=pre["c2f"],
        rdnw=s.rdnw,
        rdn=pre["rdn"],
        fnm=s.fnm,
        fnp=s.fnp,
        cf1=pre["cf1"],
        cf2=pre["cf2"],
        cf3=pre["cf3"],
        msftx=s.msftx,
        msfty=s.msfty,
        rdx=1.0 / float(cfg.dx),
        rdy=1.0 / float(cfg.dy),
        dts=float(cfg.dt),
        epssm=float(cfg.epssm),
        top_lid=bool(cfg.top_lid),
        gravity=GRAVITY_M_S2,
        w_save=s.w_save,
        damp_opt=int(cfg.damp_opt),
        dampcoef=float(cfg.dampcoef),
        zdamp=float(cfg.zdamp),
        w_damping=int(cfg.w_damping),
        w_alpha=float(cfg.w_alpha),
        w_crit_cfl=float(cfg.w_crit_cfl),
    )

    if s.ph_bdy_target is not None and s.ph_save_for_spec is not None:
        ph_next = spec_bdyupdate_ph_inloop(
            ph_next,
            s.ph_bdy_target,
            s.ph_save_for_spec,
            mu_tend=None,
            muts=ph_finish_muts,
            c1f=pre["c1f"],
            c2f=pre["c2f"],
            dts=float(cfg.dt),
            config=DEFAULT_BOUNDARY_CONFIG,
        )

    if bool(cfg.spec_w_zero_grad):
        w_solved = w_solved.at[:, 0, :].set(w_solved[:, 1, :])
        w_solved = w_solved.at[:, -1, :].set(w_solved[:, -2, :])
        w_solved = w_solved.at[:, 1:-1, 0].set(w_solved[:, 1:-1, 1])
        w_solved = w_solved.at[:, 1:-1, -1].set(w_solved[:, 1:-1, -2])

    state_for_pressure = s.replace(w=w_solved, ph=ph_next, t_2ave=t_2ave_next)
    p_rho = calc_p_rho_step(
        mu_work=calc_mu_work,
        muts_total=calc_muts,
        ph_work=ph_next,
        theta_work=calc_theta,
        theta_1=state_for_pressure.theta_1,
        c2a=pre["c2a"],
        alt=pre["alt"],
        c1h=state_for_pressure.c1h,
        c2h=state_for_pressure.c2h,
        rdnw=state_for_pressure.rdnw,
        pm1=(state_for_pressure.pm1 if state_for_pressure.pm1 is not None else state_for_pressure.p),
        smdiv=smdiv,
        t0=300.0,
    )
    theta_phys = ac._decouple_theta_for_finish(state_for_pressure, theta_for_finish, muts_for_finish)
    ru_m = (state_for_pressure.ru_m if state_for_pressure.ru_m is not None else jnp.zeros_like(state_for_pressure.u)) + state_for_pressure.u
    rv_m = (state_for_pressure.rv_m if state_for_pressure.rv_m is not None else jnp.zeros_like(state_for_pressure.v)) + state_for_pressure.v
    ww_m = (state_for_pressure.ww_m if state_for_pressure.ww_m is not None else jnp.zeros_like(state_for_pressure.ww)) + ww_for_finish
    acoustic_out = state_for_pressure.replace(
        mu=mu_for_finish,
        mudf=pre["mudf_new"],
        muts=muts_for_finish,
        muave=muave_for_finish,
        ww=ww_for_finish,
        theta=theta_phys,
        theta_coupled_work=theta_for_finish,
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
    cap = blocker.observe_state(next_carry.state, namelist)
    cmp = blocker.compare_capture_to_wrf(cap, ctx["base_capture"], ctx["wrf_stage1"], ctx["wrf_base"])
    return acoustic_out, next_carry, {
        "name": name,
        "options": options,
        "stage1_increment_rmse": {
            fld: {
                "full": cmp[fld]["incr_err"]["full"].get("rmse"),
                "interior": cmp[fld]["incr_err"]["interior"].get("rmse"),
                "band": cmp[fld]["incr_err"]["band"].get("rmse"),
                "mean_interior": cmp[fld]["incr_err"]["interior"].get("mean"),
                "max_abs_interior": cmp[fld]["incr_err"]["interior"].get("max_abs"),
            }
            for fld in ("mu", "p", "ph", "al", "alt", "w")
            if fld in cmp
        },
        "advance_w_outputs": {
            "ph_work_after_advance_w_minus_baseline_placeholder": {},
            "ph_work": {
                "delta_vs_wrf_21602": _split(np.asarray(ph_next, dtype=np.float64) - ctx["wrf_stage1"]["fields3"]["ph"][: ph_next.shape[0], : ph_next.shape[1], : ph_next.shape[2]]),
            },
            "p_work": {
                "delta_vs_wrf_21602": _split(np.asarray(p_rho.p, dtype=np.float64) - ctx["wrf_stage1"]["fields3"]["p"][: p_rho.p.shape[0], : p_rho.p.shape[1], : p_rho.p.shape[2]]),
            },
        },
    }


def _input_ledger(pre: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import jax.numpy as jnp

    s = pre["state_for_w"]
    mu_work = np.asarray(pre["mu_work"], dtype=np.float64)
    ny, nx = int(s.mu.shape[0]), int(s.mu.shape[1])
    wrf_mu_base = jnp.asarray(ctx["wrf_base"]["fields2"]["mu"][:ny, :nx], dtype=s.mu.dtype)
    wrf_mu_stage1 = jnp.asarray(ctx["wrf_stage1"]["fields2"]["mu"][:ny, :nx], dtype=s.mu.dtype)
    wrf_mu_delta = wrf_mu_stage1 - wrf_mu_base
    wrf_muts_stage1 = s.mut + wrf_mu_delta
    wrf_muave_stage1 = 0.5 * (1.0 + float(pre["cfg"].epssm)) * wrf_mu_delta
    return {
        "pre_advance_w_input_scales": {
            "ww_new": _scalar_summary(pre["ww_new"]),
            "mu_work": _scalar_summary(mu_work),
            "muave_new": _scalar_summary(pre["muave_new"]),
            "muts_new": _scalar_summary(pre["muts_new"]),
            "theta_coupled": _scalar_summary(pre["theta_coupled"]),
            "w_work": _scalar_summary(s.w),
            "ph_work": _scalar_summary(s.ph),
            "ph_tend": _scalar_summary(s.ph_tend),
            "rw_tend_stage": _scalar_summary(pre["rw_tend_stage"]),
            "cqw_stage_minus_dry": _scalar_summary(np.asarray(pre["cqw_field"], dtype=np.float64) - np.asarray(pre["dry_cqw_field"], dtype=np.float64)),
        },
        "variant_input_deltas": {
            "jax_mu_new_minus_wrf_call21602_mu": _split(
                np.asarray(pre["mu_new"], dtype=np.float64) - np.asarray(wrf_mu_stage1, dtype=np.float64)
            ),
            "jax_muts_new_minus_wrf_derived_muts": _split(
                np.asarray(pre["muts_new"], dtype=np.float64) - np.asarray(wrf_muts_stage1, dtype=np.float64)
            ),
            "jax_muave_new_minus_wrf_derived_muave": _split(
                np.asarray(pre["muave_new"], dtype=np.float64) - np.asarray(wrf_muave_stage1, dtype=np.float64)
            ),
            "dry_recomputed_rw_minus_stage_rw": _split(
                np.asarray(pre["rw_tend_dry_recomputed"], dtype=np.float64)
                - np.asarray(pre["rw_tend_stage"], dtype=np.float64)
            ),
        },
    }


def main() -> int:
    import jax

    allocator_shim = _install_cpu_allocator_shim()
    ctx = surface_probe._build_stage1()
    pre = _build_pre_advance_w(ctx)
    variants = [
        ("baseline_current", {}),
        ("zero_ph_tend", {"zero_ph_tend": True}),
        ("zero_rw_tend", {"zero_rw_tend": True}),
        ("dry_recomputed_rw_tend", {"dry_recomputed_rw_tend": True}),
        ("dry_cqw_and_coefficients", {"dry_cqw_and_coefficients": True}),
        ("dry_cqw_and_dry_rw", {"dry_cqw_and_coefficients": True, "dry_recomputed_rw_tend": True}),
        ("zero_phi_adv", {"zero_phi_adv": True}),
        ("wrf_coupled_surface", {"wrf_coupled_surface": True}),
        ("wrf_call21602_mu_inputs", {"wrf_call21602_mu_inputs": True}),
        ("calc_p_rho_stage_mut_denominator", {"calc_p_rho_stage_mut_denominator": True}),
        ("calc_p_rho_no_smdiv", {"calc_p_rho_no_smdiv": True}),
    ]
    results: dict[str, Any] = {}
    baseline_debug = None
    for name, options in variants:
        acoustic_out, next_carry, result = _run_variant(ctx, pre, name, options)
        jax.block_until_ready(next_carry.state.u)
        results[name] = result
        if name == "baseline_current":
            baseline_debug = {
                "ph_work": np.asarray(acoustic_out.ph, dtype=np.float64),
                "p_work": np.asarray(acoustic_out.p, dtype=np.float64),
            }

    if baseline_debug is not None:
        base_ph = baseline_debug["ph_work"]
        base_p = baseline_debug["p_work"]
        for name, result in results.items():
            if name == "baseline_current":
                continue
            # Rerun cheaply only for output deltas would duplicate work; the
            # WRF-scored RMSEs above are the primary evidence.  Keep this slot
            # explicit so readers do not mistake it for a missing comparison.
            result["advance_w_outputs"]["variant_minus_baseline_note"] = (
                "variant was scored against WRF call 21602; baseline-relative full arrays are omitted to keep JSON compact"
            )

    base = results["baseline_current"]["stage1_increment_rmse"]
    scored = {}
    for name, result in results.items():
        cur = result["stage1_increment_rmse"]
        scored[name] = {
            "p_interior_rmse": cur["p"]["interior"],
            "ph_interior_rmse": cur["ph"]["interior"],
            "p_improvement_fraction": float(1.0 - cur["p"]["interior"] / max(base["p"]["interior"], 1e-30)),
            "ph_improvement_fraction": float(1.0 - cur["ph"]["interior"] / max(base["ph"]["interior"], 1e-30)),
        }

    payload = {
        "schema": "v014_switzerland_advance_w_term_split",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
            "JAX_ENABLE_COMPILATION_CACHE": os.environ.get("JAX_ENABLE_COMPILATION_CACHE"),
        },
        "allocator_shim": allocator_shim,
        "anchor": "h36 state == WRF call 21601; all variants scored against WRF call 21602",
        "config": {
            "dt_s": float(ctx["namelist"].dt_s),
            "stage_dt": float(ctx["stage"].dts_rk),
            "acoustic_substeps": int(ctx["namelist"].acoustic_substeps),
            "specified_bdy_cadence": bool(ctx["namelist"].specified_bdy_cadence),
            "specified_adv_degrade": bool(ctx["namelist"].specified_adv_degrade),
            "moist_cqw_env": os.environ.get("GPUWRF_MOIST_CQW", "<default-on>"),
            "top_lid": bool(ctx["namelist"].top_lid),
            "epssm": float(ctx["namelist"].epssm),
        },
        "inputs": _input_ledger(pre, ctx),
        "variants": results,
        "scored_improvements_vs_baseline": scored,
        "verdict": {
            "best_ph_variant": min(scored, key=lambda n: scored[n]["ph_interior_rmse"]),
            "best_p_variant": min(scored, key=lambda n: scored[n]["p_interior_rmse"]),
            "material_local_variant_found": any(
                v["ph_improvement_fraction"] > 0.5 and v["p_improvement_fraction"] > 0.5
                for k, v in scored.items()
                if k != "baseline_current"
            ),
            "baseline_reproduces_prior": {
                "p_interior_rmse": base["p"]["interior"],
                "ph_interior_rmse": base["ph"]["interior"],
                "expected_prior_p": 1.1261975184532773,
                "expected_prior_ph": 0.4352639584631776,
            },
        },
    }
    hpg.write_json(OUT_JSON, payload)
    print(json.dumps(payload["verdict"], indent=2, sort_keys=True))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
