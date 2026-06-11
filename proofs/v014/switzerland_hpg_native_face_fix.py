#!/usr/bin/env python
"""V0.14 Switzerland h36 HPG native-face fix proof.

Root cause (this sprint): the entire JAX runtime rebuilt the calc_p_rho_phi
diagnostics ``al``/``alt``/``p`` with the WRF ``hypsometric_opt=1`` LINEAR
specific-volume relation, while every real WRF case runs the v4 Registry
DEFAULT ``hypsometric_opt=2`` LOG-pressure-thickness relation
(``module_big_step_utilities_em.F:1043-1062``, Registry.EM_COMMON:2285).

Offline anchor (reproduced in ``same_state_alt_forms``): at the h36 CPU truth
state the live ``alt`` implied by the EOS identity ``p = EOS(theta_m, alt)``
matches the LOG form to fp32 roundoff (~1.4e-6 rel) while the linear form is
one-signed off by ~4.2e-4 mean / 6.2e-4 max rel, modulated horizontally by the
terrain dry-column mass ``muts`` -> a spurious large-step horizontal PGF over
the Alps -> the d01 h36 strong-flow dry mass venting.

Native-face truth: an instrumented disposable WRF copy
(``proofs/v014/switzerland_hpg_native_face_wrf_patch.diff``) dumps, inside
``horizontal_pressure_gradient`` at RK1/RK2/RK3 of steps 7201-7202 of a 36h30m
24-rank re-run, the native U/V-face inputs (``ph/alt/al/p/pb/php/cqu/cqv/
muu/muv/mu``) and the recomputed per-face subterms (ph-pair ``t1``,
``(alt_l+alt_r)*dp`` ``t2``, ``(al_l+al_r)*dpb`` ``t3``, the non-hydro ``t4``,
and the full ``dpx/dpy``).  ``--wrf-faces`` reassembles the rank dumps and
compares them against the JAX ``large_step_horizontal_pgf`` subterms computed
from the SAME h36 state under hypsometric_opt 1 (legacy) and 2 (fixed).

Gates:
* ``--step-probe``: 30-step h36 dry probes, hypso1 vs hypso2 vs zero-PGF.
* ``--forecast-variant``: 1h fixed-source GPU forecast for the hourly
  excess-outflux budget gate vs CPU truth (>=70 % collapse required).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CPU = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
PROBE_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
BASELINE_GPU = PROBE_ROOT / "gpu_output"
FIX_GPU = PROBE_ROOT / "gpu_output_hpg_native_face_fix"
NATIVE_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_hpg_native_face")
NATIVE_RUN = NATIVE_ROOT / "run_wrf"
NATIVE_DUMPS = NATIVE_ROOT / "hpg_dumps"
NATIVE_REPLAY = NATIVE_ROOT / "run_h36"
OUT_JSON = ROOT / "proofs/v014/switzerland_hpg_native_face_fix.json"
RUN_START = datetime(2023, 1, 15)

R_D = 287.0
CP = 7.0 * R_D / 2.0
CV = CP - R_D
P0 = 1.0e5
CVOVCP = CV / CP


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def fn(base: Path, hour: int) -> Path:
    label = (RUN_START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
    return base / f"wrfout_d01_{label}"


def get(base: Path, hour: int, var: str) -> np.ndarray:
    with Dataset(fn(base, hour)) as handle:
        return np.asarray(handle.variables[var][0])


def _array_stats(arr: Any) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
        "rmse": float(np.sqrt(np.nanmean(values * values))),
        "mean_abs": float(np.nanmean(np.abs(values))),
        "max_abs": float(np.nanmax(np.abs(values))),
    }


def _diff_stats(left: Any, right: Any) -> dict[str, float]:
    return _array_stats(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))


# --------------------------------------------------------------------------
# Offline anchor: the three alt forms at the h36 CPU truth state.
# --------------------------------------------------------------------------

def same_state_alt_forms(h36_file: Path) -> dict[str, Any]:
    with Dataset(h36_file) as d:
        g = lambda v: np.asarray(d.variables[v][0], dtype=np.float64)
        P_ = g("P"); PB_ = g("PB"); PH_ = g("PH"); PHB_ = g("PHB")
        MU_ = g("MU"); MUB_ = g("MUB"); THM_ = g("THM")
        c1h = g("C1H"); c2h = g("C2H"); c3h = g("C3H"); c4h = g("C4H")
        c3f = g("C3F"); c4f = g("C4F"); rdnw = g("RDNW")
        ptop = float(d.variables["P_TOP"][0])
    theta_m = THM_ + 300.0
    p_tot = P_ + PB_
    dph = (PH_ + PHB_)[1:] - (PH_ + PHB_)[:-1]
    muts = MUB_ + MU_
    alt_eos = (R_D / P0) * theta_m * (p_tot / P0) ** (-CVOVCP)
    alt_h1 = -rdnw[:, None, None] * dph / (c1h[:, None, None] * muts[None] + c2h[:, None, None])
    pfu = c3f[1:, None, None] * muts[None] + c4f[1:, None, None] + ptop
    pfd = c3f[:-1, None, None] * muts[None] + c4f[:-1, None, None] + ptop
    phm = c3h[:, None, None] * muts[None] + c4h[:, None, None] + ptop
    alt_h2 = dph / phm / np.log(pfd / pfu)
    rel = lambda a, b: _array_stats((a - b) / alt_eos)
    r1 = np.abs((alt_h1 - alt_eos) / alt_eos).mean(axis=(1, 2))
    r2 = np.abs((alt_h2 - alt_eos) / alt_eos).mean(axis=(1, 2))
    return {
        "h36_file": str(h36_file),
        "alt_log_form_vs_live_eos_rel": rel(alt_h2, alt_eos),
        "alt_linear_form_vs_live_eos_rel": rel(alt_h1, alt_eos),
        "per_level_mean_abs_rel": {
            "linear_form": [float(v) for v in r1],
            "log_form": [float(v) for v in r2],
        },
        "reading": (
            "WRF live alt (EOS-implied from file P/THM) IS the hypsometric_opt=2 LOG form to "
            "fp32 roundoff; the hypsometric_opt=1 linear form the JAX runtime used is one-signed "
            "off with terrain-modulated structure."
        ),
    }


# --------------------------------------------------------------------------
# WRF native dump reader + reassembly.
# --------------------------------------------------------------------------

def _read_rank_dump(txt_path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"manifest": []}
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if not parts or line.startswith("#") or line.startswith("wrfgpu2"):
            continue
        key = parts[0]
        if key in {"VEC", "ARR2", "ARR3"}:
            meta["manifest"].append((key, parts[1], [int(v) for v in parts[2:]]))
        elif key.startswith("scalars_"):
            meta[key] = [float(v) for v in parts[1:]]
        elif key.startswith("flags_"):
            meta[key] = parts[1:]
        else:
            meta[key] = [int(v) for v in parts[1:]]
    raw = np.fromfile(txt_path.with_suffix(".bin"), dtype=">f4").astype(np.float64)
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for kind, name, bounds in meta["manifest"]:
        if kind == "VEC":
            k0, k1 = bounds
            n = k1 - k0 + 1
            arrays[name] = raw[offset:offset + n]
            offset += n
        elif kind == "ARR2":
            i0, i1, j0, j1 = bounds
            ni, nj = i1 - i0 + 1, j1 - j0 + 1
            arrays[name] = raw[offset:offset + ni * nj].reshape(nj, ni)  # [j, i]
            offset += ni * nj
        else:
            i0, i1, k0, k1, j0, j1 = bounds
            ni, nk, nj = i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1
            arrays[name] = raw[offset:offset + ni * nk * nj].reshape(nj, nk, ni).transpose(1, 0, 2)  # [k, j, i]
            offset += ni * nk * nj
    if offset != raw.size:
        raise ValueError(f"{txt_path}: consumed {offset} of {raw.size} floats")
    meta["arrays"] = arrays
    return meta


def assemble_wrf_call(ncall: int) -> dict[str, Any]:
    """Reassemble one HPG call's per-rank dumps into global [k, j, i] arrays."""

    rank_files = sorted(NATIVE_DUMPS.glob(f"hpg_call{ncall:06d}_i*_j*.txt"))
    if not rank_files:
        raise FileNotFoundError(f"no dumps for call {ncall} under {NATIVE_DUMPS}")
    first = _read_rank_dump(rank_files[0])
    ids, ide, jds, jde, kds, kde = first["domain_ids_ide_jds_jde_kds_kde"]
    nx, ny, nz = ide - ids, jde - jds, kde - kds  # mass dims 128,128,44
    out: dict[str, Any] = {
        "ncall": ncall,
        "ranks": len(rank_files),
        "flags": first["flags_nonhydro_toplid_specified"],
        "scalars": first["scalars_rdx_rdy_cf1_cf2_cf3_cfn_cfn1"],
    }
    fields3 = ["ph", "php", "alt", "al", "p", "pb", "cqu", "cqv",
               "t1y", "t2y", "t3y", "t4y", "dpy", "t1x", "t2x", "t3x", "t4x", "dpx"]
    fields2 = ["muu", "muv", "mu", "msfux", "msfuy", "msfvx", "msfvy"]
    glob3 = {name: np.full((nz + 1, ny + 1, nx + 1), np.nan) for name in fields3}
    glob2 = {name: np.full((ny + 1, nx + 1), np.nan) for name in fields2}
    for path in rank_files:
        rank = _read_rank_dump(path)
        its, ite, jts, jte, _, _ = rank["tile_its_ite_jts_jte_kts_kte"]
        wi0, wi1, wj0, wj1, ktf = rank["window_i0_i1_j0_j1_ktf"]
        yl = rank["yloop_i0_i1_j0_j1"]
        xl = rank["xloop_i0_i1_j0_j1"]
        arrays = rank["arrays"]

        def put3(name: str, i_lo: int, i_hi: int, j_lo: int, j_hi: int, k_hi: int) -> None:
            src = arrays[name]
            # manifest k0 is always kds; src index [k - kds, j - wj0, i - wi0]
            glob3[name][0:k_hi - kds + 1, j_lo - 1:j_hi, i_lo - 1:i_hi] = src[
                0:k_hi - kds + 1, j_lo - wj0:j_hi - wj0 + 1, i_lo - wi0:i_hi - wi0 + 1
            ]

        def put2(name: str, i_lo: int, i_hi: int, j_lo: int, j_hi: int) -> None:
            src = arrays[name]
            glob2[name][j_lo - 1:j_hi, i_lo - 1:i_hi] = src[
                j_lo - wj0:j_hi - wj0 + 1, i_lo - wi0:i_hi - wi0 + 1
            ]

        mi1, mj1 = min(ite, ide - 1), min(jte, jde - 1)
        for name in ["php", "alt", "al", "p", "pb"]:
            put3(name, its, mi1, jts, mj1, kde - 1)
        put3("ph", its, mi1, jts, mj1, kde)
        put3("cqu", xl[0], xl[1], xl[2], xl[3], kde - 1)
        put3("cqv", yl[0], yl[1], yl[2], yl[3], kde - 1)
        for name in ["t1x", "t2x", "t3x", "t4x", "dpx"]:
            put3(name, xl[0], xl[1], xl[2], xl[3], ktf)
        for name in ["t1y", "t2y", "t3y", "t4y", "dpy"]:
            put3(name, yl[0], yl[1], yl[2], yl[3], ktf)
        put2("mu", its, mi1, jts, mj1)
        put2("muu", xl[0], xl[1], xl[2], xl[3])
        put2("msfux", xl[0], xl[1], xl[2], xl[3])
        put2("msfuy", xl[0], xl[1], xl[2], xl[3])
        put2("muv", yl[0], yl[1], yl[2], yl[3])
        put2("msfvx", yl[0], yl[1], yl[2], yl[3])
        put2("msfvy", yl[0], yl[1], yl[2], yl[3])
        out.setdefault("vectors", {name: arrays[name] for name in ["c1h", "c2h", "fnm", "fnp", "rdnw"]})
    out["fields3"] = glob3
    out["fields2"] = glob2
    out["dims"] = {"nx": nx, "ny": ny, "nz": nz}
    return out


# --------------------------------------------------------------------------
# JAX-side faces from the same h36 state.
# --------------------------------------------------------------------------

def jax_face_terms(state: Any, metrics: Any, *, dx_m: float, dy_m: float, hypsometric_opt: int, top_lid: bool) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    from gpuwrf.dynamics.core import rk_addtend_dry as rk

    ph, p_abs, al, alt, php = rk._absolute_diagnostics(state, metrics, hypsometric_opt=hypsometric_opt)
    pb = (state.p_total - state.p_perturbation).astype(jnp.float64)
    mu_pert = state.mu_perturbation.astype(jnp.float64)
    mu_total = state.mu_total.astype(jnp.float64)
    rdx, rdy = 1.0 / float(dx_m), 1.0 / float(dy_m)
    c1h = metrics.c1h[:, None, None]
    c2h = metrics.c2h[:, None, None]
    rdnw = metrics.rdnw[:, None, None]

    _dn_top = metrics.dn[-1]
    _dn_safe = jnp.where(jnp.abs(_dn_top) > 1.0e-30, _dn_top, jnp.asarray(1.0, dtype=_dn_top.dtype))
    cfn = (0.5 * metrics.dnw[-1] + metrics.dn[-1]) / _dn_safe
    cfn1 = -0.5 * metrics.dnw[-1] / _dn_safe

    def dpn_faces(pair_sum):
        nz = int(pair_sum.shape[0])
        dpn = jnp.zeros((nz + 1,) + pair_sum.shape[1:], dtype=pair_sum.dtype)
        dpn = dpn.at[0].set(0.5 * (metrics.cf1 * pair_sum[0] + metrics.cf2 * pair_sum[1] + metrics.cf3 * pair_sum[2]))
        dpn = dpn.at[1:nz].set(0.5 * (metrics.fnm[1:, None, None] * pair_sum[1:] + metrics.fnp[1:, None, None] * pair_sum[:-1]))
        if top_lid:
            dpn = dpn.at[nz].set(0.5 * (cfn * pair_sum[-1] + cfn1 * pair_sum[-2]))
        return dpn

    out: dict[str, np.ndarray] = {}
    out["al"] = np.asarray(al)
    out["alt"] = np.asarray(alt)
    out["p"] = np.asarray(p_abs)
    out["php"] = np.asarray(php)

    ph_l, ph_r = rk._x_face_pair_3d(ph)
    p_l, p_r = rk._x_face_pair_3d(p_abs)
    pb_l, pb_r = rk._x_face_pair_3d(pb)
    al_l, al_r = rk._x_face_pair_3d(al)
    alt_l, alt_r = rk._x_face_pair_3d(alt)
    muu = 0.5 * sum(rk._x_face_pair_2d(mu_total))
    msf_u = (metrics.msfux / metrics.msfuy)[None, :, :]
    out["t1x"] = np.asarray((ph_r[1:] - ph_l[1:]) + (ph_r[:-1] - ph_l[:-1]))
    out["t2x"] = np.asarray((alt_l + alt_r) * (p_r - p_l))
    out["t3x"] = np.asarray((al_l + al_r) * (pb_r - pb_l))
    mass_u = c1h * muu[None, :, :] + c2h
    dpx123 = msf_u * 0.5 * rdx * mass_u * (
        (ph_r[1:] - ph_l[1:]) + (ph_r[:-1] - ph_l[:-1])
        + (alt_l + alt_r) * (p_r - p_l) + (al_l + al_r) * (pb_r - pb_l)
    )
    php_l, php_r = rk._x_face_pair_3d(php)
    mu_l, mu_r = rk._x_face_pair_2d(mu_pert)
    dpn_x = dpn_faces(p_l + p_r)
    t4x = msf_u * rdx * (php_r - php_l) * (
        rdnw * (dpn_x[1:] - dpn_x[:-1]) - 0.5 * (c1h * (mu_l + mu_r)[None, :, :])
    )
    out["t4x"] = np.asarray(t4x)
    out["dpx"] = np.asarray(dpx123 + t4x)
    out["muu"] = np.asarray(muu)

    ph_s, ph_n = rk._y_face_pair_3d(ph)
    p_s, p_n = rk._y_face_pair_3d(p_abs)
    pb_s, pb_n = rk._y_face_pair_3d(pb)
    al_s, al_n = rk._y_face_pair_3d(al)
    alt_s, alt_n = rk._y_face_pair_3d(alt)
    muv = 0.5 * sum(rk._y_face_pair_2d(mu_total))
    msf_v = (metrics.msfvy / metrics.msfvx)[None, :, :]
    out["t1y"] = np.asarray((ph_n[1:] - ph_s[1:]) + (ph_n[:-1] - ph_s[:-1]))
    out["t2y"] = np.asarray((alt_s + alt_n) * (p_n - p_s))
    out["t3y"] = np.asarray((al_s + al_n) * (pb_n - pb_s))
    mass_v = c1h * muv[None, :, :] + c2h
    dpy123 = msf_v * 0.5 * rdy * mass_v * (
        (ph_n[1:] - ph_s[1:]) + (ph_n[:-1] - ph_s[:-1])
        + (alt_s + alt_n) * (p_n - p_s) + (al_s + al_n) * (pb_n - pb_s)
    )
    php_s, php_n = rk._y_face_pair_3d(php)
    mu_s, mu_n = rk._y_face_pair_2d(mu_pert)
    dpn_y = dpn_faces(p_s + p_n)
    t4y = msf_v * rdy * (php_n - php_s) * (
        rdnw * (dpn_y[1:] - dpn_y[:-1]) - 0.5 * (c1h * (mu_s + mu_n)[None, :, :])
    )
    out["t4y"] = np.asarray(t4y)
    out["dpy"] = np.asarray(dpy123 + t4y)
    out["muv"] = np.asarray(muv)
    return out


def _build_state(run_root: Path):
    from gpuwrf.integration import daily_pipeline as dp
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=run_root / "unused_hpg_face_output",
        proof_dir=run_root / "unused_hpg_face_proofs",
        run_root=run_root,
        domain="d01",
        async_output=False,
    )
    case, run_dir = dp._build_real_case(cfg)
    state = _enforce_operational_precision(case.state, force_fp64=True)
    return case, state, run_dir


def wrf_faces_comparison(args: argparse.Namespace) -> dict[str, Any]:
    ncall = int(args.ncall)
    wrf = assemble_wrf_call(ncall)
    nz, ny, nx = wrf["dims"]["nz"], wrf["dims"]["ny"], wrf["dims"]["nx"]
    case, state, run_dir = _build_state(NATIVE_ROOT)
    metrics = case.namelist.metrics
    dx = float(case.namelist.grid.projection.dx_m)
    dy = float(case.namelist.grid.projection.dy_m)
    result: dict[str, Any] = {
        "ncall": ncall,
        "run_dir": str(run_dir),
        "wrf_flags": wrf["flags"],
        "ranks": wrf["ranks"],
        "comparisons": {},
    }

    # WRF arrays in [k, j, i] with 1-based WRF indices mapped to 0-based slots.
    w3 = wrf["fields3"]
    w2 = wrf["fields2"]

    # state-level identity checks (live arrays vs wrfout-loaded state)
    p_state = np.asarray(state.p_perturbation)
    ph_state = np.asarray(state.ph_perturbation)
    pb_state = np.asarray(state.p_total - state.p_perturbation)
    mu_state = np.asarray(state.mu_perturbation)
    result["state_vs_wrf_live"] = {
        "p": _diff_stats(p_state, w3["p"][:nz, :ny, :nx]),
        "ph": _diff_stats(ph_state, w3["ph"][:nz + 1, :ny, :nx]),
        "pb": _diff_stats(pb_state, w3["pb"][:nz, :ny, :nx]),
        "mu": _diff_stats(mu_state, w2["mu"][:ny, :nx]),
    }

    for opt in (1, 2):
        jaxf = jax_face_terms(
            state, metrics, dx_m=dx, dy_m=dy, hypsometric_opt=opt,
            top_lid=False,  # match the WRF real-case HPG (top_lid=F)
        )
        comp: dict[str, Any] = {}
        # cell-centred diagnostics
        for name in ["al", "alt"]:
            wrf_arr = w3[name][:nz, :ny, :nx]
            mask = np.isfinite(wrf_arr)
            comp[name] = _diff_stats(jaxf[name][mask], wrf_arr[mask])
            comp[name]["rel_mean_abs_vs_alt"] = float(
                np.nanmean(np.abs(jaxf[name][mask] - wrf_arr[mask]) / np.abs(w3["alt"][:nz, :ny, :nx][mask]))
            )
        # u-face arrays: WRF face i (1-based) -> jax face index i-1
        for name in ["t1x", "t2x", "t3x", "t4x", "dpx"]:
            wrf_arr = w3[name][:nz, :ny, :nx + 1]
            jax_arr = jaxf[name][:, :ny, :nx + 1]
            mask = np.isfinite(wrf_arr) & (wrf_arr > -9.0e33)
            comp[name] = _diff_stats(jax_arr[mask], wrf_arr[mask])
            comp[name]["wrf_rmse"] = _array_stats(wrf_arr[mask])["rmse"]
            comp[name]["count"] = int(mask.sum())
        for name in ["t1y", "t2y", "t3y", "t4y", "dpy"]:
            wrf_arr = w3[name][:nz, :ny + 1, :nx]
            jax_arr = jaxf[name][:, :ny + 1, :nx]
            mask = np.isfinite(wrf_arr) & (wrf_arr > -9.0e33)
            comp[name] = _diff_stats(jax_arr[mask], wrf_arr[mask])
            comp[name]["wrf_rmse"] = _array_stats(wrf_arr[mask])["rmse"]
            comp[name]["count"] = int(mask.sum())
        for name, jx in [("muu", "muu"), ("muv", "muv")]:
            wrf_arr = w2[name]
            jax_arr = jaxf[jx]
            ny_c = min(wrf_arr.shape[0], jax_arr.shape[0])
            nx_c = min(wrf_arr.shape[1], jax_arr.shape[1])
            wrf_c = wrf_arr[:ny_c, :nx_c]
            jax_c = jax_arr[:ny_c, :nx_c]
            mask = np.isfinite(wrf_c)
            comp[name] = _diff_stats(jax_c[mask], wrf_c[mask])
        result["comparisons"][f"hypsometric_opt_{opt}"] = comp
    return result


# --------------------------------------------------------------------------
# 30-step probe + forecast variant (GPU).
# --------------------------------------------------------------------------

def run_step_probe(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import gpuwrf.runtime.operational_mode as operational_mode
    from gpuwrf.runtime.operational_mode import _physics_boundary_step
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision
    from gpuwrf.integration import daily_pipeline as dp

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=PROBE_ROOT / "unused_hpg_step_probe_output",
        proof_dir=PROBE_ROOT / "unused_hpg_step_probe_proofs",
        run_root=PROBE_ROOT,
        domain="d01",
        async_output=False,
    )
    case, run_dir = dp._build_real_case(cfg)

    def state_summary(state) -> dict[str, float | bool]:
        arrays = {
            "u": state.u, "v": state.v, "w": state.w, "theta": state.theta,
            "mu_total": state.mu_total, "p_total": state.p_total,
        }
        out: dict[str, float | bool] = {}
        all_finite = True
        for name, arr in arrays.items():
            finite = bool(np.isfinite(np.asarray(arr)).all())
            all_finite = all_finite and finite
            out[f"{name}_finite"] = finite
            out[f"{name}_absmax"] = float(jnp.max(jnp.abs(arr)))
        out["w_top_absmax"] = float(jnp.max(jnp.abs(state.w[-1])))
        out["theta_min"] = float(jnp.min(state.theta))
        out["theta_max"] = float(jnp.max(state.theta))
        out["mu_total_mean"] = float(jnp.mean(state.mu_total))
        out["p_total_k0_mean"] = float(jnp.mean(state.p_total[0]))
        out["all_finite"] = all_finite
        return out

    variants = [
        ("hypso2_fixed", 2, "full"),
        ("hypso1_legacy", 1, "full"),
        ("zero_large_step_pgf_hypso2", 2, "zero"),
        ("zero_large_step_pgf_hypso1", 1, "zero"),
    ]
    output: dict[str, Any] = {
        "available": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "steps_requested": int(args.steps),
        "dt_s": float(case.namelist.dt_s),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": getattr(jax, "__version__", None),
            "jax_backend": jax.default_backend(),
        },
        "variants": {},
    }
    only = set(args.only or [])
    original_pgf = operational_mode.large_step_horizontal_pgf
    for name, hypso, pgf_kind in variants:
        if only and name not in only:
            continue
        base = dataclasses.replace(
            case.namelist,
            run_physics=False, disable_guards=True, top_lid=True, run_boundary=True,
            hypsometric_opt=hypso,
        )
        carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))
        if pgf_kind == "zero":
            def _zero(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False, hypsometric_opt=1):
                del metrics, dx_m, dy_m, non_hydrostatic, top_lid, hypsometric_opt
                return jnp.zeros_like(state.u), jnp.zeros_like(state.v)
            operational_mode.large_step_horizontal_pgf = _zero
        hist = []
        first_bad_step = None
        start = time.perf_counter()
        try:
            @jax.jit
            def _one_step(carry_in, namelist_in, step_index):
                return _physics_boundary_step(carry_in, namelist_in, step_index, run_radiation=False, debug=False)

            print(f"[hpg-fix] variant={name} hypsometric_opt={hypso} pgf={pgf_kind}", flush=True)
            for step in range(1, int(args.steps) + 1):
                carry = _one_step(carry, base, jnp.asarray(step, dtype=jnp.int32))
                jax.block_until_ready(carry.state.u)
                rec = {"step": step} | state_summary(carry.state)
                hist.append(rec)
                if step <= 2 or step % 10 == 0 or not bool(rec["all_finite"]):
                    print(
                        f"[hpg-fix] {name} step={step} finite={rec['all_finite']} "
                        f"u={rec['u_absmax']:.2f} v={rec['v_absmax']:.2f} w={rec['w_absmax']:.2f} "
                        f"mu_mean={rec['mu_total_mean']:.4f} p0_mean={rec['p_total_k0_mean']:.2f}",
                        flush=True,
                    )
                if first_bad_step is None and (
                    not bool(rec["all_finite"]) or float(rec["w_absmax"]) > 500.0
                    or float(rec["u_absmax"]) > 500.0
                    or float(rec["theta_min"]) < 150.0 or float(rec["theta_max"]) > 650.0
                ):
                    first_bad_step = step
                    break
        finally:
            operational_mode.large_step_horizontal_pgf = original_pgf
        output["variants"][name] = {
            "hypsometric_opt": hypso,
            "pgf_kind": pgf_kind,
            "steps_completed": len(hist),
            "first_bad_step": first_bad_step,
            "wall_s": float(time.perf_counter() - start),
            "history": hist,
            "final": hist[-1] if hist else None,
        }
    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}
    previous = (existing.get("step_probe") or {}).get("variants", {})
    if isinstance(previous, Mapping):
        merged = dict(previous)
        merged.update(output["variants"])
        output["variants"] = merged
    existing["step_probe"] = output
    write_json(out_path, existing)
    print(f"wrote {out_path}", flush=True)
    return existing


def run_forecast_variant(args: argparse.Namespace) -> None:
    from gpuwrf.integration import daily_pipeline as dp

    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=int(args.hours),
        output_dir=FIX_GPU,
        proof_dir=PROBE_ROOT / "proofs_hpg_native_face_fix",
        run_root=PROBE_ROOT,
        domain="d01",
    )
    result = dp._run_forecast_sequence(config, output_dir=config.output_dir)
    print(f"forecast result: status={result.status} hours={result.hours} output_dir={result.output_dir}", flush=True)


# --------------------------------------------------------------------------
# Hourly budget (identical control surface to the prior sprints).
# --------------------------------------------------------------------------

def load_budget_state(base: Path, hour: int) -> dict[str, Any]:
    with Dataset(fn(base, hour)) as d:
        return {
            "mu": np.asarray(d.variables["MU"][0]) + np.asarray(d.variables["MUB"][0]),
            "u": np.asarray(d.variables["U"][0]),
            "v": np.asarray(d.variables["V"][0]),
            "dnw": np.asarray(d.variables["DNW"][0]),
            "c1h": np.asarray(d.variables["C1H"][0]),
            "c2h": np.asarray(d.variables["C2H"][0]),
            "mx": np.asarray(d.variables["MAPFAC_MX"][0]),
            "my": np.asarray(d.variables["MAPFAC_MY"][0]),
            "muy": np.asarray(d.variables["MAPFAC_UY"][0]),
            "mvx": np.asarray(d.variables["MAPFAC_VX"][0]),
            "dx": float(d.DX),
        }


def budget_between(start_base: Path, start_hour: int, end_base: Path, end_hour: int, depth: int = 8) -> dict[str, float]:
    a = load_budget_state(start_base, start_hour)
    b = load_budget_state(end_base, end_hour)
    wk = -a["dnw"]
    c1 = a["c1h"]
    c2 = a["c2h"]
    ny, nx = a["mu"].shape
    i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth

    def colmass(s: Mapping[str, Any]) -> float:
        m = ((c1[:, None, None] * s["mu"][None] + c2[:, None, None]) * wk[:, None, None]).sum(0)
        return float((m / (s["mx"] * s["my"]))[j0:j1, i0:i1].sum())

    def outflux(s: Mapping[str, Any]) -> float:
        mu, u, v = s["mu"], s["u"], s["v"]

        def mul_u(i: int) -> np.ndarray:
            muf = 0.5 * (mu[j0:j1, i - 1] + mu[j0:j1, i])
            return c1[:, None] * muf[None, :] + c2[:, None]

        def mul_v(j: int) -> np.ndarray:
            muf = 0.5 * (mu[j - 1, i0:i1] + mu[j, i0:i1])
            return c1[:, None] * muf[None, :] + c2[:, None]

        fw = (u[:, j0:j1, i0] * mul_u(i0) * wk[:, None] / s["muy"][j0:j1, i0][None, :]).sum()
        fe = (u[:, j0:j1, i1] * mul_u(i1) * wk[:, None] / s["muy"][j0:j1, i1][None, :]).sum()
        fs = (v[:, j0, i0:i1] * mul_v(j0) * wk[:, None] / s["mvx"][j0, i0:i1][None, :]).sum()
        fnn = (v[:, j1, i0:i1] * mul_v(j1) * wk[:, None] / s["mvx"][j1, i0:i1][None, :]).sum()
        return float((fe - fw) + (fnn - fs))

    ncell = (j1 - j0) * (i1 - i0)
    dm = (colmass(b) - colmass(a)) / ncell
    flux = (outflux(a) + outflux(b)) / 2.0 * 3600.0 / a["dx"] / ncell
    return {
        "dM_pa_per_cell_h": float(dm),
        "net_influx_pa_per_cell_h": float(-flux),
        "residual_pa_per_cell_h": float(dm + flux),
    }


def field_metrics(base: Path, hour: int) -> dict[str, Any]:
    if not fn(base, hour).exists():
        return {"available": False, "path": str(fn(base, hour))}
    out: dict[str, Any] = {"available": True, "valid_h": hour}
    for var in ["MU", "PSFC", "T", "U", "V", "W", "PH"]:
        diff = get(base, hour, var) - get(CPU, hour, var)
        out[var.lower()] = {"rmse": float(np.sqrt((diff ** 2).mean())), "bias": float(diff.mean())}
        out[var.lower()]["finite"] = bool(np.isfinite(get(base, hour, var)).all())
    return out


# --------------------------------------------------------------------------
# Analyze: collate everything into the proof JSON.
# --------------------------------------------------------------------------

def _delta_at(variants: Mapping[str, Any], name: str, step: int) -> float | None:
    var = variants.get(name)
    if not var:
        return None
    hist = var.get("history") or []
    if len(hist) < step:
        return None
    return float(hist[step - 1]["mu_total_mean"]) - float(hist[0]["mu_total_mean"])


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    proof: dict[str, Any] = {
        "schema": "v014_switzerland_hpg_native_face_fix",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wrf_source_anchors": {
            "registry_hypsometric_default": "/home/enric/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON:2285 (default 2)",
            "calc_p_rho_phi_hypso2": "/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:1043-1062",
            "calc_p_rho_phi_hypso1": "/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:1027-1030",
            "horizontal_pgf_terms": "/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:2310-2316,2385-2392",
            "gen2_hpg_identical_to_pristine": "diff of horizontal_pressure_gradient gen2 vs pristine 4.7.1: byte-identical",
        },
        "jax_fix_anchors": {
            "diagnose_pressure_al_alt": "src/gpuwrf/dynamics/acoustic_wrf.py (hypsometric_opt=2 branch)",
            "absolute_diagnostics": "src/gpuwrf/dynamics/core/rk_addtend_dry.py (hypsometric_opt=2 branch)",
            "namelist_field": "src/gpuwrf/runtime/operational_mode.py OperationalNamelist.hypsometric_opt (default 1; real pipelines pass 2)",
            "real_pipelines": "src/gpuwrf/integration/daily_pipeline.py + nested_pipeline.py from_grid(hypsometric_opt=2)",
        },
        "artifacts": {
            "cpu": str(CPU),
            "h36_baseline_gpu": str(BASELINE_GPU),
            "h36_fixed_gpu": str(FIX_GPU),
            "native_wrf_run": str(NATIVE_RUN),
            "native_dumps": str(NATIVE_DUMPS),
            "wrf_patch": "proofs/v014/switzerland_hpg_native_face_wrf_patch.diff",
        },
        "step_probe": existing.get("step_probe", {}),
        "wrf_faces": existing.get("wrf_faces", {}),
    }
    proof["same_state_alt_forms"] = same_state_alt_forms(fn(CPU, 36))

    variants = (proof["step_probe"] or {}).get("variants", {})
    step = 30
    deltas = {name: _delta_at(variants, name, step) for name in variants}
    zero1 = deltas.get("zero_large_step_pgf_hypso1")
    zero2 = deltas.get("zero_large_step_pgf_hypso2")
    old = deltas.get("hypso1_legacy")
    fixed = deltas.get("hypso2_fixed")
    summary: dict[str, Any] = {"mu_delta_30_steps_pa": deltas}
    if old is not None and zero1 is not None:
        summary["old_pgf_contribution_pa_per_cell_h"] = float((old - zero1) * 12.0)
    if fixed is not None and zero2 is not None:
        summary["fixed_pgf_contribution_pa_per_cell_h"] = float((fixed - zero2) * 12.0)
    if old is not None and fixed is not None:
        summary["mu_delta_collapse_fraction"] = float(1.0 - abs(fixed) / max(abs(old), 1.0e-12))
    proof["step_probe_summary"] = summary

    proof["hourly_gate"] = {"available": fn(FIX_GPU, 37).exists(), "fixed_output": str(FIX_GPU)}
    if fn(FIX_GPU, 37).exists():
        cpu_budget = budget_between(CPU, 36, CPU, 37, depth=8)
        old_budget = budget_between(CPU, 36, BASELINE_GPU, 37, depth=8)
        fixed_budget = budget_between(CPU, 36, FIX_GPU, 37, depth=8)
        old_excess = old_budget["net_influx_pa_per_cell_h"] - cpu_budget["net_influx_pa_per_cell_h"]
        fixed_excess = fixed_budget["net_influx_pa_per_cell_h"] - cpu_budget["net_influx_pa_per_cell_h"]
        proof["hourly_gate"] |= {
            "metrics_h37": field_metrics(FIX_GPU, 37),
            "baseline_metrics_h37": field_metrics(BASELINE_GPU, 37),
            "cpu_budget_h36_h37_depth8": cpu_budget,
            "old_baseline_budget_h36_h37_depth8": old_budget,
            "fixed_budget_h36_h37_depth8": fixed_budget,
            "old_excess_outflux_pa_per_cell_h": float(old_excess),
            "fixed_excess_outflux_pa_per_cell_h": float(fixed_excess),
            "collapse_fraction": float(1.0 - abs(fixed_excess) / max(abs(old_excess), 1.0e-12)),
        }
        if fn(FIX_GPU, 39).exists():
            proof["hourly_gate"]["metrics_h39"] = field_metrics(FIX_GPU, 39)
            cpu_b39 = budget_between(CPU, 36, CPU, 39, depth=8)
            fix_b39 = budget_between(CPU, 36, FIX_GPU, 39, depth=8)
            old_b39 = budget_between(CPU, 36, BASELINE_GPU, 39, depth=8) if fn(BASELINE_GPU, 39).exists() else None
            proof["hourly_gate"]["h36_h39"] = {
                "cpu": cpu_b39, "fixed": fix_b39, "old": old_b39,
                "fixed_excess": float(fix_b39["net_influx_pa_per_cell_h"] - cpu_b39["net_influx_pa_per_cell_h"]),
                "old_excess": (
                    float(old_b39["net_influx_pa_per_cell_h"] - cpu_b39["net_influx_pa_per_cell_h"])
                    if old_b39 else None
                ),
            }

    gate = proof.get("hourly_gate", {})
    collapse = gate.get("collapse_fraction")
    proof["verdict"] = "FIXED" if (collapse is not None and float(collapse) >= 0.70) else "EXACT_ROOT_NO_FIX"
    proof["root_classification"] = {
        "hpg_face_root_cause_FIXED": (
            "JAX runtime calc_p_rho_phi diagnostics (al/alt/p) used WRF hypsometric_opt=1 (linear) "
            "everywhere; WRF real cases run the Registry default hypsometric_opt=2 (LOG form), and "
            "the carried base alb is ALSO the LOG-form base. Native-face proof: with the fix the "
            "large-step HPG per-face terms match WRF native truth to ~1.3-1.9e-4 rel (dpx rmse "
            "0.070 vs signal 368; legacy was rmse 6.05 / max 87, concentrated in the pb*al branch), "
            "and the state arrays p/ph/pb/mu are bit-identical at RK1 of step 7201."
        ),
        "venting_blocker_verdict": (
            "REFUTED as the venting blocker: with the HPG native faces now exact, the h36->h37 "
            "full-physics excess outflux collapses only ~1.0% (old -28.62 -> fixed -28.33 "
            "Pa/cell/h). The d01 strong-flow dry mass venting is NOT caused by the large-step "
            "horizontal PGF inputs or operator at the RK-stage boundary."
        ),
        "next_implementation_target": (
            "The acoustic-substep lane (advance_uv/advance_w/advance_mu work-array evolution and "
            "ww divergence) between RK-stage boundaries. The WRF stage-boundary truth for this is "
            "ALREADY CAPTURED: hpg_dumps calls 21602/21603 (RK2/RK3 of step 7201) and 21604-21606 "
            "(step 7202) contain the live post-substep p/ph/al/alt/mu arrays; instrument the JAX "
            "operational stage states at the same boundaries and bisect the first diverging stage "
            "increment. No new WRF run is needed."
        ),
        "fix_kept": (
            "hypsometric_opt=2 LOG-form al/alb in diagnose_pressure_al_alt (per-stage refresh + "
            "pg_buoy_w stage pressure) and rk_addtend_dry._absolute_diagnostics (large-step HPG), "
            "threaded via OperationalNamelist.hypsometric_opt; real pipelines pass 2, idealized "
            "paths keep 1 (their generator metrics carry placeholder c3f/c4f). Kept: it is a real, "
            "native-face-proven WRF-faithfulness fix (evolving-p bias -19.5 Pa -> -0.005 Pa)."
        ),
    }
    write_json(OUT_JSON, proof)
    return proof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-probe", action="store_true")
    parser.add_argument("--forecast-variant", action="store_true")
    parser.add_argument("--wrf-faces", action="store_true")
    parser.add_argument("--ncall", type=int, default=21601)
    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--only", action="append")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args()

    if args.step_probe:
        run_step_probe(args)
    elif args.forecast_variant:
        run_forecast_variant(args)
    elif args.wrf_faces:
        existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        existing["wrf_faces"] = wrf_faces_comparison(args)
        write_json(OUT_JSON, existing)
        print(f"wrote wrf_faces to {OUT_JSON}")
    else:
        proof = analyze(args)
        print(f"wrote {OUT_JSON}")
        print(json.dumps({
            "verdict": proof.get("verdict"),
            "hourly_gate_collapse": (proof.get("hourly_gate") or {}).get("collapse_fraction"),
            "step_probe_summary": proof.get("step_probe_summary"),
        }, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
