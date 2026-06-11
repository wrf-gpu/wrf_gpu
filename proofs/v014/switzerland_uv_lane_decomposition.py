#!/usr/bin/env python
"""V0.14 Switzerland h36 acoustic-substep u''/v'' (advance_uv) lane decomposition.

Proof-only harness; every score is JAX-vs-WRF-native (dump truth at
``/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/awd_dumps``, WRF call
21601 -> 21602, itimestep=7201 rk=1 iter=1).

Key identity exploited: at rk_step=1, iteration=1 WRF ``small_step_prep`` zeroes
every small-step work array (u''=v''=t''=ph''=mu''=p''=al''=0, mudf=0), so WRF
``advance_uv`` reduces EXACTLY to ``u'' = dts*ru_tend`` (the small-step PGF and
divergence-damping terms vanish identically with zero perturbations,
module_small_step_em.F:805-942).  The dumped ``advance_w``-entry ``u/v``
(= grid%u_2/v_2 post-advance_uv) therefore expose the WRF-native large-step
coupled ``ru_tend``/``rv_tend`` truth: ``ru_tend_wrf = u_dump/dts``.

From the same dump, the ``advance_mu_t`` theta'' and mu''/ww closures yield the
WRF-native implied ``ft`` (t_tend) and ``mu_tend`` fields, because every other
input of those updates (u'', v'', ww'', t_1, map factors, column vectors) is in
the dump and the stage-entry carried fields are already proven bit-exact.

Comparisons produced (interior depth-8, same convention as the awd oracle):
  1. JAX post-advance_uv u''/v''       vs dumped u/v
  2. JAX staged u_tend/v_tend           vs implied WRF ru/rv_tend (u_dump/dts)
  3. JAX staged theta_tend (ft)         vs implied WRF ft (theta'' closure)
  4. JAX staged mu_tend                 vs implied WRF mu_tend (mu'' closure)
  5. production advance_mu_t mixed runs: all-JAX baseline vs WRF-u''v'' swap,
     scored on mu''/ww/t_2 against the dump (isolates the u''-driven share).
"""

from __future__ import annotations

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

OUT_JSON = ROOT / "proofs/v014/switzerland_uv_lane_decomposition.json"

_AWD_SPEC = importlib.util.spec_from_file_location(
    "wrf_native_advance_w_dump", Path(__file__).with_name("wrf_native_advance_w_dump.py")
)
awd = importlib.util.module_from_spec(_AWD_SPEC)
_AWD_SPEC.loader.exec_module(awd)  # type: ignore[union-attr]

ts = awd._load_term_split()


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _summ(cmp: dict[str, Any]) -> dict[str, Any]:
    return {
        "interior_rmse": cmp["interior"].get("rmse"),
        "interior_max_abs": cmp["interior"].get("max_abs"),
        "interior_mean": cmp["interior"].get("mean"),
        "interior_rel_rmse": cmp.get("interior_rel_rmse"),
        "wrf_interior_rms": cmp.get("wrf_interior_rms"),
        "full_rmse": cmp["full"].get("rmse"),
    }


def _kprof(jax_arr: np.ndarray, wrf_arr: np.ndarray, kmax: int = 12) -> dict[str, float]:
    diff = _np(jax_arr) - _np(wrf_arr)
    valid = np.isfinite(_np(wrf_arr)) & (_np(wrf_arr) > -9.0e30)
    _, ny, nx = diff.shape
    m2 = awd._interior_mask(ny, nx)
    out = {}
    for k in range(min(diff.shape[0], 44)):
        sel = valid[k] & m2
        if sel.any():
            out[f"k{k:02d}"] = float(np.sqrt(np.mean(diff[k][sel] ** 2)))
    top = sorted(out.items(), key=lambda kv: -kv[1])[:kmax]
    return dict(top)


def _field_stats(arr) -> dict[str, float]:
    v = _np(arr)
    return {
        "rms": float(np.sqrt(np.nanmean(v * v))),
        "max_abs": float(np.nanmax(np.abs(v))),
    }


def main() -> int:
    import jax
    import jax.numpy as jnp

    shim = ts._install_cpu_allocator_shim()
    ctx = ts.surface_probe._build_stage1()
    pre = ts._build_pre_advance_w(ctx)

    from gpuwrf.dynamics.core import acoustic as ac

    cfg = pre["cfg"]
    s0 = ctx["acoustic"]          # stage-entry acoustic work state
    s = pre["state_for_w"]        # post advance_uv + advance_mu_t

    aw = awd.assemble_awd("awd_aw_s0007201_rk1_it01")
    wrf = awd.wrf_crops(aw, {})
    dts = float(aw["meta"]["scalars"]["dts"])
    assert abs(dts - float(cfg.dt)) < 1e-12, (dts, cfg.dt)

    nz, ny, nx = 44, 128, 128
    rdx = 1.0 / float(cfg.dx)
    rdy = 1.0 / float(cfg.dy)

    payload: dict[str, Any] = {
        "schema": "v014_switzerland_uv_lane_decomposition",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": "WRF itimestep=7201 rk=1 iter=1 (call 21601 -> 21602); h36 JAX context",
        "allocator_shim": shim,
        "dts": dts,
        "identity": "rk1/it1: u''_wrf == dts*ru_tend_wrf (small-step PGF identically 0)",
    }

    # ------------------------------------------------------------------
    # 0. sanity: JAX stage-entry small-step work arrays are zero
    # ------------------------------------------------------------------
    payload["stage_entry_work_arrays"] = {
        "u_work": _field_stats(s0.u),
        "v_work": _field_stats(s0.v),
        "ph_work": _field_stats(s0.ph),
        "p_work": _field_stats(s0.p),
        "al_work": _field_stats(s0.al) if s0.al is not None else None,
        "mu_work": _field_stats(s0.muts - s0.mut),
        "mudf": _field_stats(s0.mudf),
    }

    # ------------------------------------------------------------------
    # 1. post-advance_uv u''/v'' vs dump
    # ------------------------------------------------------------------
    payload["post_advance_uv"] = {
        "u_pp": _summ(awd.compare_field(_np(s.u), wrf["u"])),
        "v_pp": _summ(awd.compare_field(_np(s.v), wrf["v"])),
        "u_pp_kprofile_top": _kprof(_np(s.u), wrf["u"]),
        "v_pp_kprofile_top": _kprof(_np(s.v), wrf["v"]),
    }

    # ------------------------------------------------------------------
    # 2. implied WRF ru/rv_tend vs JAX staged u_tend/v_tend
    # ------------------------------------------------------------------
    ru_implied = wrf["u"] / dts
    rv_implied = wrf["v"] / dts
    jax_ru = _np(s0.u_tend)
    jax_rv = _np(s0.v_tend)
    payload["large_step_uv_tend"] = {
        "ru_tend": _summ(awd.compare_field(jax_ru, ru_implied)),
        "rv_tend": _summ(awd.compare_field(jax_rv, rv_implied)),
        "ru_tend_kprofile_top": _kprof(jax_ru, ru_implied),
        "rv_tend_kprofile_top": _kprof(jax_rv, rv_implied),
    }

    # ------------------------------------------------------------------
    # 3. theta'' closure -> implied WRF ft vs JAX theta_tend
    #    t2_dump = msfty*dts*ft - dts*msfty*adv(u'',v'',ww'',t_1)   (t''_old = 0)
    # ------------------------------------------------------------------
    u_d = wrf["u"]; v_d = wrf["v"]; ww_d = wrf["ww"]; t1_d = wrf["t_1"]
    fnm = wrf["fnm"]; fnp = wrf["fnp"]; rdnw = wrf["rdnw"]; dnw = wrf["dnw"]
    c1h = wrf["c1h"]; c2h = wrf["c2h"]
    msftx = wrf["msftx"]; msfty = wrf["msfty"]

    wdtn = np.zeros((nz + 1, ny, nx))
    for k in range(1, nz):
        wdtn[k] = ww_d[k] * (fnm[k] * t1_d[k] + fnp[k] * t1_d[k - 1])

    adv = np.full((nz, ny, nx), np.nan)
    ys = slice(1, ny - 1); xs = slice(1, nx - 1)
    ys_n = slice(2, ny); ys_s = slice(0, ny - 2)
    xs_e = slice(2, nx); xs_w = slice(0, nx - 2)
    for k in range(nz):
        th = t1_d[k]
        v_flux = v_d[k, ys_n, xs] * (th[ys_n, xs] + th[ys, xs]) - v_d[k, ys, xs] * (th[ys, xs] + th[ys_s, xs])
        u_flux = u_d[k, ys, xs_e] * (th[ys, xs_e] + th[ys, xs]) - u_d[k, ys, xs] * (th[ys, xs] + th[ys, xs_w])
        adv[k, ys, xs] = msftx[ys, xs] * (0.5 * rdy * v_flux + 0.5 * rdx * u_flux) + rdnw[k] * (
            wdtn[k + 1, ys, xs] - wdtn[k, ys, xs]
        )
    # NOTE staggering: u_d index i is the WEST face of mass cell i (u_d has nx+1
    # columns), v_d index j is the SOUTH face of mass cell j.
    ft_implied = (wrf["t_2"] + dts * msfty[None] * adv) / (msfty[None] * dts)
    jax_ft = _np(s0.theta_tend)
    payload["theta_ft_closure"] = {
        "replica_check_t2_jax": None,  # filled below
        "ft": _summ(awd.compare_field(jax_ft, ft_implied)),
        "ft_kprofile_top": _kprof(jax_ft, ft_implied),
        "adv_wrf_interior_rms": float(
            np.sqrt(np.nanmean((adv * msfty[None] * dts)[:, awd._interior_mask(ny, nx)] ** 2))
        ),
    }

    # replica self-check: same closure with all-JAX inputs must reproduce the
    # JAX t_2 (validates the NumPy advection replica against production).
    u_j = _np(s.u); v_j = _np(s.v); ww_j = _np(pre["ww_new"]); t1_j = _np(s.theta_1)
    wdtn_j = np.zeros((nz + 1, ny, nx))
    for k in range(1, nz):
        wdtn_j[k] = ww_j[k] * (fnm[k] * t1_j[k] + fnp[k] * t1_j[k - 1])
    adv_j = np.full((nz, ny, nx), np.nan)
    for k in range(nz):
        th = t1_j[k]
        v_flux = v_j[k, ys_n, xs] * (th[ys_n, xs] + th[ys, xs]) - v_j[k, ys, xs] * (th[ys, xs] + th[ys_s, xs])
        u_flux = u_j[k, ys, xs_e] * (th[ys, xs_e] + th[ys, xs]) - u_j[k, ys, xs] * (th[ys, xs] + th[ys, xs_w])
        adv_j[k, ys, xs] = msftx[ys, xs] * (0.5 * rdy * v_flux + 0.5 * rdx * u_flux) + rdnw[k] * (
            wdtn_j[k + 1, ys, xs] - wdtn_j[k, ys, xs]
        )
    t2_replica = msfty[None] * dts * jax_ft - dts * msfty[None] * adv_j
    payload["theta_ft_closure"]["replica_check_t2_jax"] = _summ(
        awd.compare_field(t2_replica, _np(pre["theta_coupled"]))
    )

    # ------------------------------------------------------------------
    # 4. mu'' closure -> implied WRF mu_tend vs JAX mu_tend
    # ------------------------------------------------------------------
    muu = _np(s0.muu); muv = _np(s0.muv)
    u1 = _np(s0.u_1); v1 = _np(s0.v_1)
    msfuy = _np(s0.msfuy); msfvx_inv = _np(s0.msfvx_inv)
    mass_u = c1h[:nz, None, None] * muu[None] + c2h[:nz, None, None]
    mass_v = c1h[:nz, None, None] * muv[None] + c2h[:nz, None, None]
    U = u_d + mass_u * u1 / msfuy[None]
    V = v_d + mass_v * v1 * msfvx_inv[None]
    dvdxi = np.full((nz, ny, nx), np.nan)
    dvdxi[:, ys, xs] = msftx[ys, xs] * msfty[ys, xs] * (
        rdy * (V[:, ys_n, xs] - V[:, ys, xs]) + rdx * (U[:, ys, xs_e] - U[:, ys, xs])
    )
    dmdt = np.sum(dnw[:nz, None, None] * dvdxi, axis=0)
    mu_tend_implied = wrf["mu1"] / dts - dmdt
    jax_mu_tend = _np(s0.mu_tend)
    payload["mu_tend_closure"] = {
        "mu_tend": _summ(awd.compare_field(jax_mu_tend, mu_tend_implied)),
        "mu1_pred_wrf_uv_jax_mu_tend_vs_dump": _summ(
            awd.compare_field(dts * (dmdt + jax_mu_tend), wrf["mu1"])
        ),
        "mu1_jax_vs_dump_reference": _summ(awd.compare_field(_np(pre["mu_new"]), wrf["mu1"])),
    }

    # ww closure with WRF u'',v'' + JAX mu_tend, vs dumped ww
    ww_pred = np.zeros((nz + 1, ny, nx))
    for kk in range(1, nz):
        k = kk - 1
        inc = dnw[kk - 1] * (c1h[k] * dmdt + dvdxi[kk - 1] + c1h[k] * jax_mu_tend) / msfty
        ww_pred[kk] = ww_pred[kk - 1] - inc
    ww_1 = _np(s0.ww_1)
    ww_pred[:nz] = ww_pred[:nz] - ww_1[:nz]
    payload["mu_tend_closure"]["ww_pred_wrf_uv_jax_mu_tend_vs_dump"] = _summ(
        awd.compare_field(ww_pred, wrf["ww"])
    )
    payload["mu_tend_closure"]["ww_jax_vs_dump_reference"] = _summ(
        awd.compare_field(_np(pre["ww_new"]), wrf["ww"])
    )

    # ------------------------------------------------------------------
    # 5. production advance_mu_t mixed-input runs
    # ------------------------------------------------------------------
    uv_state = ac.advance_uv_wrf(
        s0,
        dts_rk=float(cfg.dt),
        dx=float(cfg.dx),
        dy=float(cfg.dy),
        top_lid=bool(cfg.top_lid),
        emdiv=float(getattr(ctx["namelist"], "emdiv", 0.01)),
        dt_full=float(ctx["namelist"].dt_s),
    )
    coupled_state = uv_state.replace(theta=uv_state.theta_coupled_work)

    def _mu_t_outputs(state) -> dict[str, np.ndarray]:
        adv_out = ac.advance_mu_t_core(state, cfg)
        return {
            # the dumped WRF "mu1" is the small-step work increment mu'' =
            # muts_new - mut (the full advanced mu carries the stage mu_save)
            "mu": _np(adv_out["muts"]) - _np(state.mut),
            "ww": _np(adv_out["ww"]),
            "t_2": _np(adv_out["theta"]),
        }

    runs = {
        "all_jax_baseline": _mu_t_outputs(coupled_state),
        "wrf_uv_swap": _mu_t_outputs(
            coupled_state.replace(
                u=jnp.asarray(u_d, dtype=coupled_state.u.dtype),
                v=jnp.asarray(v_d, dtype=coupled_state.v.dtype),
            )
        ),
    }
    mixed: dict[str, Any] = {}
    for name, out in runs.items():
        mixed[name] = {
            "mu1": _summ(awd.compare_field(out["mu"], wrf["mu1"])),
            "ww": _summ(awd.compare_field(out["ww"], wrf["ww"])),
            "t_2": _summ(awd.compare_field(out["t_2"], wrf["t_2"])),
        }
    # baseline must reproduce pre in the INTERIOR (pre additionally pins the
    # spec ring on mu/theta after advance, so only depth-8 interior is bitwise)
    m2 = awd._interior_mask(ny, nx)
    mixed["baseline_equals_pre_interior_max_abs"] = {
        "mu_work": float(np.max(np.abs(
            (runs["all_jax_baseline"]["mu"] - (_np(pre["muts_new"]) - _np(s.mut)))[m2]
        ))),
        "ww": float(np.max(np.abs((runs["all_jax_baseline"]["ww"] - _np(pre["ww_new"]))[:, m2]))),
        "t_2": float(np.max(np.abs((runs["all_jax_baseline"]["t_2"] - _np(pre["theta_coupled"]))[:, m2]))),
    }
    payload["advance_mu_t_mixed_runs"] = mixed

    OUT_JSON.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    print(json.dumps({k: payload[k] for k in (
        "stage_entry_work_arrays", "post_advance_uv", "large_step_uv_tend",
    )}, indent=1, default=float))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
