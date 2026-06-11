#!/usr/bin/env python
"""V0.14 Switzerland h36 ru/rv_tend + theta_tend contributor decomposition.

Phase 2 of the uv-lane decomposition (see switzerland_uv_lane_decomposition.py).
Phase 1 proved against the WRF-native dump (call 21601 -> 21602, rk1 it1):

* JAX staged ru_tend / rv_tend are 57% / 72% rel off the WRF-native implied
  truth (u_dump/dts), surface-peaked, mean-biased;
* the ENTIRE mu''/ww error collapses (15.5% -> 0.14%, 51% -> 0.065%) when the
  WRF u''/v'' are substituted, exonerating mu_tend/stage parts/operator;
* theta_tend (ft) is independently 53.7% rel off the implied truth.

This phase rebuilds the production stage-tendency pipeline (identical calls to
the proven _build_stage1 context) and scores each contributor against the
WRF-native implied tendencies:

  err_u = ru_tend_jax_total - ru_tend_wrf_implied
  per contributor C: rmse(err_u - C) ("what if C were absent"), corr(err_u, C).

Contributors: flux advection, 6th-order diffusion, Smagorinsky horizontal
diffusion, large-step horizontal PGF, large-step Coriolis, rk_addtend_dry
physics fold, specified relax_bdy fold.
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

OUT_JSON = ROOT / "proofs/v014/switzerland_uv_lane_contributors.json"

_AWD_SPEC = importlib.util.spec_from_file_location(
    "wrf_native_advance_w_dump", Path(__file__).with_name("wrf_native_advance_w_dump.py")
)
awd = importlib.util.module_from_spec(_AWD_SPEC)
_AWD_SPEC.loader.exec_module(awd)  # type: ignore[union-attr]

ts = awd._load_term_split()
hpg = ts.hpg


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _interior_stats(diff: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    _, ny, nx = diff.shape
    m = awd._interior_mask(ny, nx)
    sel = np.broadcast_to(m[None], diff.shape)
    if valid is not None:
        sel = sel & valid
    vals = diff[sel]
    vals = vals[np.isfinite(vals)]
    return {
        "rmse": float(np.sqrt(np.mean(vals**2))),
        "mean": float(np.mean(vals)),
        "max_abs": float(np.max(np.abs(vals))),
    }


def _corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    _, ny, nx = a.shape
    m = np.broadcast_to(awd._interior_mask(ny, nx)[None], a.shape) & valid
    av, bv = a[m], b[m]
    keep = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[keep], bv[keep]
    if av.size < 10 or float(np.std(av)) == 0.0 or float(np.std(bv)) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def main() -> int:
    import jax
    import jax.numpy as jnp

    shim = ts._install_cpu_allocator_shim()

    from gpuwrf.contracts.halo import apply_halo
    from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
    from gpuwrf.dynamics.core.rk_addtend_dry import (
        DryPhysicsTendencies,
        large_step_coriolis,
        large_step_horizontal_pgf,
        rk_addtend_dry,
    )
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry

    # ------------------------------------------------------------------
    # rebuild the EXACT _build_stage1 pipeline with intermediate captures
    # ------------------------------------------------------------------
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
    # GPUWRF_PROBE_FIRST_TS=0 keeps the loaded (spun-up) WRF h36 QKE instead of
    # forcing the genuine-cold-start MYNN seed -- the re-init discriminator.
    probe_first_ts = os.environ.get("GPUWRF_PROBE_FIRST_TS", "1") != "0"
    forcing = om._physics_step_forcing(
        carry0,
        namelist,
        lead_seconds,
        run_radiation=bool(namelist.run_physics),
        first_timestep=probe_first_ts,
    )
    carry = forcing.carry
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin
    carry = carry.replace(state=origin)
    haloed = apply_halo(carry.state, halo_spec(namelist.grid))
    base_tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    stage_velocities = (
        om._stage_transport_velocities(haloed, namelist)
        if bool(namelist.use_flux_advection)
        else None
    )
    bdy_relax = om._specified_bdy_relax(rk1_reference, namelist, lead_seconds)

    def augment(physics, relax):
        return om._augment_large_step_tendencies(
            haloed,
            base_tendencies,
            namelist,
            rk_step=1,
            physics_tendencies=physics,
            step_origin=rk1_reference,
            transport_velocities=stage_velocities,
            bdy_relax=relax,
        )

    t_full = augment(forcing.dry_tendencies, bdy_relax)
    t_nophys = augment(None, bdy_relax)
    t_nophys_norelax = augment(None, None)
    t_phys_norelax = augment(forcing.dry_tendencies, None)

    # ------------------------------------------------------------------
    # WRF-native implied truth
    # ------------------------------------------------------------------
    aw = awd.assemble_awd("awd_aw_s0007201_rk1_it01")
    wrf = awd.wrf_crops(aw, {})
    dts = float(aw["meta"]["scalars"]["dts"])

    ru_implied = wrf["u"] / dts
    rv_implied = wrf["v"] / dts
    valid_u = np.isfinite(ru_implied) & (ru_implied > -9.0e30)
    valid_v = np.isfinite(rv_implied) & (rv_implied > -9.0e30)

    # implied ft from the theta'' closure (phase-1 formula, replica verified 0.0)
    nz, ny, nx = 44, 128, 128
    rdx = 1.0 / float(namelist.grid.projection.dx_m)
    rdy = 1.0 / float(namelist.grid.projection.dy_m)
    u_d, v_d, ww_d, t1_d = wrf["u"], wrf["v"], wrf["ww"], wrf["t_1"]
    fnm, fnp, rdnw = wrf["fnm"], wrf["fnp"], wrf["rdnw"]
    msftx, msfty = wrf["msftx"], wrf["msfty"]
    wdtn = np.zeros((nz + 1, ny, nx))
    for k in range(1, nz):
        wdtn[k] = ww_d[k] * (fnm[k] * t1_d[k] + fnp[k] * t1_d[k - 1])
    adv = np.full((nz, ny, nx), np.nan)
    ys, xs = slice(1, ny - 1), slice(1, nx - 1)
    ys_n, ys_s = slice(2, ny), slice(0, ny - 2)
    xs_e, xs_w = slice(2, nx), slice(0, nx - 2)
    for k in range(nz):
        th = t1_d[k]
        v_flux = v_d[k, ys_n, xs] * (th[ys_n, xs] + th[ys, xs]) - v_d[k, ys, xs] * (th[ys, xs] + th[ys_s, xs])
        u_flux = u_d[k, ys, xs_e] * (th[ys, xs_e] + th[ys, xs]) - u_d[k, ys, xs] * (th[ys, xs] + th[ys, xs_w])
        adv[k, ys, xs] = msftx[ys, xs] * (0.5 * rdy * v_flux + 0.5 * rdx * u_flux) + rdnw[k] * (
            wdtn[k + 1, ys, xs] - wdtn[k, ys, xs]
        )
    ft_implied = (wrf["t_2"] + dts * msfty[None] * adv) / (msfty[None] * dts)
    valid_t = np.isfinite(ft_implied)

    # ------------------------------------------------------------------
    # contributor fields
    # ------------------------------------------------------------------
    metrics = namelist.metrics
    ru_pgf, rv_pgf = large_step_horizontal_pgf(
        haloed,
        metrics,
        dx_m=float(namelist.grid.projection.dx_m),
        dy_m=float(namelist.grid.projection.dy_m),
        non_hydrostatic=True,
        top_lid=bool(namelist.top_lid),
        hypsometric_opt=int(namelist.hypsometric_opt),
    )
    ru_cor, rv_cor = large_step_coriolis(haloed, metrics, specified=bool(namelist.run_boundary))

    contrib_u: dict[str, np.ndarray] = {
        "physics_fold": _np(t_full.u) - _np(t_nophys.u),
        "relax_fold": _np(t_full.u) - _np(t_phys_norelax.u),
        "pgf": _np(ru_pgf),
        "coriolis": _np(ru_cor),
        "adv_plus_diff": _np(t_nophys_norelax.u) - _np(ru_pgf) - _np(ru_cor),
    }
    contrib_v: dict[str, np.ndarray] = {
        "physics_fold": _np(t_full.v) - _np(t_nophys.v),
        "relax_fold": _np(t_full.v) - _np(t_phys_norelax.v),
        "pgf": _np(rv_pgf),
        "coriolis": _np(rv_cor),
        "adv_plus_diff": _np(t_nophys_norelax.v) - _np(rv_pgf) - _np(rv_cor),
    }
    contrib_t: dict[str, np.ndarray] = {
        "physics_fold": _np(t_full.theta) - _np(t_nophys.theta),
        "relax_fold": _np(t_full.theta) - _np(t_phys_norelax.theta),
        "adv_plus_diff": _np(t_nophys_norelax.theta),
    }

    payload: dict[str, Any] = {
        "schema": "v014_switzerland_uv_lane_contributors",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": "WRF itimestep=7201 rk=1 iter=1 (call 21601 -> 21602); h36 JAX context",
        "allocator_shim": shim,
        "dts": dts,
    }

    def section(name, total, implied, valid, contribs):
        err = _np(total) - implied
        out = {
            "total_vs_implied": _interior_stats(err, valid),
            "implied_interior_rms": _interior_stats(implied - 0.0, valid)["rmse"],
            "contributor_scales": {},
            "remove_one": {},
            "corr_err_vs_contrib": {},
        }
        for cname, cfield in contribs.items():
            out["contributor_scales"][cname] = _interior_stats(cfield, valid)
            out["remove_one"][cname] = _interior_stats(err - cfield, valid)
            out["corr_err_vs_contrib"][cname] = _corr(err, cfield, valid)
        payload[name] = out
        return err

    err_u = section("ru_tend", t_full.u, ru_implied, valid_u, contrib_u)
    err_v = section("rv_tend", t_full.v, rv_implied, valid_v, contrib_v)
    err_t = section("theta_tend", t_full.theta, ft_implied, valid_t, contrib_t)

    # phase-1 cross-guard: the rebuilt totals must reproduce the recorded rmse
    payload["phase1_guard"] = {
        "ru_rmse_here": payload["ru_tend"]["total_vs_implied"]["rmse"],
        "ru_rmse_phase1": 27.302311650333404,
        "rv_rmse_here": payload["rv_tend"]["total_vs_implied"]["rmse"],
        "rv_rmse_phase1": 35.83267914551164,
        "ft_rmse_here": payload["theta_tend"]["total_vs_implied"]["rmse"],
        "ft_rmse_phase1": 13.768453880097735,
    }

    # k-profiles of err vs the leading contributor for shape forensics
    def kprof(arr, valid, kmax=14):
        out = {}
        _, nyy, nxx = arr.shape
        m = awd._interior_mask(nyy, nxx)
        for k in range(min(arr.shape[0], 44)):
            sel = m & valid[k] & np.isfinite(arr[k])
            if sel.any():
                out[f"k{k:02d}"] = float(np.sqrt(np.mean(arr[k][sel] ** 2)))
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:kmax])

    payload["kprofiles"] = {
        "err_u": kprof(err_u, valid_u),
        "physics_fold_u": kprof(contrib_u["physics_fold"], valid_u),
        "adv_plus_diff_u": kprof(contrib_u["adv_plus_diff"], valid_u),
        "err_t": kprof(err_t, valid_t),
        "physics_fold_t": kprof(contrib_t["physics_fold"], valid_t),
    }

    # ------------------------------------------------------------------
    # fold diagnosis: per-level regression of the JAX physics fold against the
    # WRF-implied fold  F_wrf := implied - (total - fold) = fold - err.
    # slope ~1 + high corr  => fold matches WRF at that level;
    # slope >>1             => JAX source overscaled at that level.
    # ------------------------------------------------------------------
    def fold_diag(err, fold, valid, kmax=14):
        out = {}
        _, nyy, nxx = err.shape
        m = awd._interior_mask(nyy, nxx)
        for k in range(min(kmax, err.shape[0])):
            sel = m & valid[k] & np.isfinite(err[k]) & np.isfinite(fold[k])
            fw = fold[k][sel] - err[k][sel]
            fj = fold[k][sel]
            denom = float(np.dot(fw, fw))
            slope = float(np.dot(fj, fw) / denom) if denom > 0 else float("nan")
            corr = (
                float(np.corrcoef(fj, fw)[0, 1])
                if fj.size > 10 and np.std(fj) > 0 and np.std(fw) > 0
                else float("nan")
            )
            out[f"k{k:02d}"] = {
                "slope_jax_vs_wrf": slope,
                "corr": corr,
                "rms_jax": float(np.sqrt(np.mean(fj**2))),
                "rms_wrf_implied": float(np.sqrt(np.mean(fw**2))),
            }
        return out

    payload["fold_diagnosis"] = {
        "u": fold_diag(err_u, contrib_u["physics_fold"], valid_u),
        "v": fold_diag(err_v, contrib_v["physics_fold"], valid_v),
        "theta": fold_diag(err_t, contrib_t["physics_fold"], valid_t),
    }

    # k0 input forensics: JAX post-sfclay ustar vs the WRF h36 carried UST.
    try:
        from netCDF4 import Dataset

        wrfout0 = sorted(Path(run_dir).glob("wrfout_d01_*"))[0]
        with Dataset(wrfout0) as ds:
            ust_wrf = np.asarray(ds.variables["UST"][0], dtype=np.float64)
        ust_jax = np.asarray(forcing.state.ustar, dtype=np.float64)
        m2 = awd._interior_mask(*ust_wrf.shape)
        diff = ust_jax - ust_wrf
        payload["ust_parity_vs_wrfout_h36"] = {
            "interior_rmse": float(np.sqrt(np.mean(diff[m2] ** 2))),
            "interior_mean": float(np.mean(diff[m2])),
            "wrf_interior_rms": float(np.sqrt(np.mean(ust_wrf[m2] ** 2))),
            "corr": float(np.corrcoef(ust_jax[m2], ust_wrf[m2])[0, 1]),
        }
    except Exception as exc:  # noqa: BLE001 - forensics only
        payload["ust_parity_vs_wrfout_h36"] = {"error": str(exc)}

    OUT_JSON.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    print(json.dumps(payload, indent=1, default=float)[:6000])
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
