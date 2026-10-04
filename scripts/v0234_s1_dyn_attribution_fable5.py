"""S1 dynamics-residual attribution (Fable 5, CPU-only, offline).

Tests the preregistered H1 of
`.agent/sprints/2026-07-18-v0234-s1-dyn-attribution-fable5/CONTRACT.md`:
the GPU momentum sixth-order-diffusion lane (periodic roll, no ownership
rings, point mass, no msf) inside the S1 rk_tendency bundle explains the
frozen 8105x S1 residual.  Pure NumPy fp64; imports neither JAX nor gpuwrf;
zero GPU activity.  Inherits loaders, band metrics, and the exact DYN
algebra from the committed Kimi internal-split module so R is reproduced
bit-consistently (F-A1) before any attribution arithmetic (F-A2/F-A3).

Companions: C-3 spec-row exit-value check (GPU L5==SP4 identity; WRF
spec_bdy_final overwrite magnitude; GPU-vs-WRF SP4 bands) and C-1
descriptive structure of the inherited SP2/MYNN-input discrepancy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_internal_split_kimi as split  # noqa: E402
from scripts import v0234_dycore_suboperator_gpt_cpu_analysis as cpu  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-dyn-attribution-fable5"
OUTPUT = SPRINT / "s1-attribution-analysis.json"
FROZEN_SPLIT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-internal-split-kimi/internal-split-analysis.json"
FROZEN_SPLIT_SELF = "65421e6c7ef823c821e43e3d28a460df56656e4b30e8d638dd1ec88895788996"

DT = split.DT_D03
DIFF_6TH_FACTOR = 0.12
DIFF_6TH_COEF = DIFF_6TH_FACTOR * 0.015625 / (2.0 * DT)


# ---------------------------------------------------------------------------
# Shared periodic flux pair (identical stencil algebra on both replicas)
# ---------------------------------------------------------------------------

def _dflux6(field: np.ndarray, axis: int):
    fm3 = np.roll(field, 3, axis=axis)
    fm2 = np.roll(field, 2, axis=axis)
    fm1 = np.roll(field, 1, axis=axis)
    f0 = field
    fp1 = np.roll(field, -1, axis=axis)
    fp2 = np.roll(field, -2, axis=axis)
    fp3 = np.roll(field, -3, axis=axis)
    dflux_p0 = 10.0 * (f0 - fm1) - 5.0 * (fp1 - fm2) + (fp2 - fm3)
    dflux_p1 = 10.0 * (fp1 - f0) - 5.0 * (fp2 - fm1) + (fp3 - fm2)
    return dflux_p0, dflux_p1, f0 - fm1, fp1 - f0


def _limit(df0, df1, g0, g1, monotonic: bool):
    if monotonic:
        df0 = np.where(df0 * g0 <= 0.0, 0.0, df0)
        df1 = np.where(df1 * g1 <= 0.0, 0.0, df1)
    return df0, df1


# ---------------------------------------------------------------------------
# GPU-lane replica: src/gpuwrf/dynamics/explicit_diffusion.py::
# sixth_order_diffusion_tendency consumed at operational_mode.py:3696-3697,
# coupled by mass_u/mass_v (edge-padded face-average mass), fp64 evaluation.
# ---------------------------------------------------------------------------

def gpu_periodic_diff6(field: np.ndarray, monotonic: bool = True) -> np.ndarray:
    dfx0, dfx1, gx0, gx1 = _dflux6(field, axis=2)
    dfx0, dfx1 = _limit(dfx0, dfx1, gx0, gx1, monotonic)
    tend = DIFF_6TH_COEF * (dfx1 - dfx0)
    if field.shape[1] > 1:
        dfy0, dfy1, gy0, gy1 = _dflux6(field, axis=1)
        dfy0, dfy1 = _limit(dfy0, dfy1, gy0, gy1, monotonic)
        tend = tend + DIFF_6TH_COEF * (dfy1 - dfy0)
    return tend


def gpu_face_mass_u(mu_total: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """operational_mode._u_face_average_2d + c1h/c2h coupling (edge copies)."""
    west = mu_total[:, :1]
    east = mu_total[:, -1:]
    interior = 0.5 * (mu_total[:, :-1] + mu_total[:, 1:])
    muu = np.concatenate((west, interior, east), axis=1)
    return c1[:, None, None] * muu[None, :, :] + c2[:, None, None]


def gpu_face_mass_v(mu_total: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    south = mu_total[:1, :]
    north = mu_total[-1:, :]
    interior = 0.5 * (mu_total[:-1, :] + mu_total[1:, :])
    muv = np.concatenate((south, interior, north), axis=0)
    return c1[:, None, None] * muv[None, :, :] + c2[:, None, None]


# ---------------------------------------------------------------------------
# WRF-lane replica: module_big_step_utilities_em.F::sixth_order_diffusion,
# 'u' and 'v' branches, specified/nested ownership, diff_6th_slopeopt=0,
# effective (rk_addtend_dry-folded) form /msfuy resp. *1/msfvx.
# ---------------------------------------------------------------------------

def _shift_m1(arr: np.ndarray, axis: int) -> np.ndarray:
    """arr[i-1] at position i (edge duplicated; unowned rows masked later)."""
    lead = [slice(None)] * arr.ndim
    lead[axis] = slice(0, 1)
    return np.concatenate([arr[tuple(lead)], arr], axis=axis).take(range(arr.shape[axis]), axis=axis)


def _shift_p1(arr: np.ndarray, axis: int) -> np.ndarray:
    """arr[i+1] at position i (edge duplicated; unowned rows masked later)."""
    tail = [slice(None)] * arr.ndim
    tail[axis] = slice(arr.shape[axis] - 1, arr.shape[axis])
    return np.concatenate([arr, arr[tuple(tail)]], axis=axis).take(range(1, arr.shape[axis] + 1), axis=axis)


def wrf_diff6_u_effective(
    u: np.ndarray,
    mut: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    msfux: np.ndarray,
    msfuy: np.ndarray,
    monotonic: bool = True,
) -> np.ndarray:
    """Coupled 'u' tendency / msfuy.  u (nz, ny=93, nxs=112); mut (93, 111)."""
    nz, ny, nxs = u.shape
    mu3 = c1[:, None, None] * mut[None, :, :] + c2[:, None, None]  # (nz, 93, 111)

    dfx0, dfx1, gx0, gx1 = _dflux6(u, axis=2)
    dfx0, dfx1 = _limit(dfx0, dfx1, gx0, gx1, monotonic)
    # x faces: p0 mass = mass cell i-1, p1 mass = mass cell i (0-based u face i).
    mu_x_p0 = np.concatenate([mu3[:, :, :1], mu3], axis=2)[:, :, :nxs]
    mu_x_p1 = np.concatenate([mu3, mu3[:, :, -1:]], axis=2)[:, :, :nxs]
    tend_x = DIFF_6TH_COEF * msfux[None, :, :] * (mu_x_p1 * dfx1 - mu_x_p0 * dfx0)

    dfy0, dfy1, gy0, gy1 = _dflux6(u, axis=1)
    dfy0, dfy1 = _limit(dfy0, dfy1, gy0, gy1, monotonic)
    # y faces: 4-point masses = y-average of the 2-point x-face mass.
    mx = 0.5 * (mu_x_p0 + mu_x_p1)  # (nz, 93, 112) mass at u faces
    mu_y_p0 = 0.5 * (_shift_m1(mx, 1) + mx)
    mu_y_p1 = 0.5 * (mx + _shift_p1(mx, 1))
    tend_y = DIFF_6TH_COEF * msfuy[None, :, :] * (mu_y_p1 * dfy1 - mu_y_p0 * dfy0)

    tend = tend_x + tend_y
    jj = np.arange(ny)[None, :, None]
    ii = np.arange(nxs)[None, None, :]
    owned = (ii >= 3) & (ii <= nxs - 4) & (jj >= 3) & (jj <= ny - 4)
    # 1-based: i in [ids+3, ide-3] on 112 faces -> 0-based [3, 108] == nxs-4;
    #          j in [jds+3, jde-4] on 93 rows  -> 0-based [3, 89]  == ny-4.
    tend = np.where(owned, tend, 0.0)
    return tend / msfuy[None, :, :]


def wrf_diff6_v_effective(
    v: np.ndarray,
    mut: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    msfvx: np.ndarray,
    msfvy: np.ndarray,
    monotonic: bool = True,
) -> np.ndarray:
    """Coupled 'v' tendency * (1/msfvx).  v (nz, nys=94, nx=111); mut (93, 111)."""
    nz, nys, nx = v.shape
    mu3 = c1[:, None, None] * mut[None, :, :] + c2[:, None, None]  # (nz, 93, 111)

    dfy0, dfy1, gy0, gy1 = _dflux6(v, axis=1)
    dfy0, dfy1 = _limit(dfy0, dfy1, gy0, gy1, monotonic)
    # y faces: p0 mass = mass row j-1, p1 = mass row j (0-based v face j).
    mu_y_p0 = np.concatenate([mu3[:, :1, :], mu3], axis=1)[:, :nys, :]
    mu_y_p1 = np.concatenate([mu3, mu3[:, -1:, :]], axis=1)[:, :nys, :]
    tend_y = DIFF_6TH_COEF * msfvy[None, :, :] * (mu_y_p1 * dfy1 - mu_y_p0 * dfy0)

    dfx0, dfx1, gx0, gx1 = _dflux6(v, axis=2)
    dfx0, dfx1 = _limit(dfx0, dfx1, gx0, gx1, monotonic)
    my = 0.5 * (mu_y_p0 + mu_y_p1)  # (nz, 94, 111) mass at v faces
    mu_x_p0 = 0.5 * (_shift_m1(my, 2) + my)
    mu_x_p1 = 0.5 * (my + _shift_p1(my, 2))
    tend_x = DIFF_6TH_COEF * msfvx[None, :, :] * (mu_x_p1 * dfx1 - mu_x_p0 * dfx0)

    tend = tend_x + tend_y
    jj = np.arange(nys)[None, :, None]
    ii = np.arange(nx)[None, None, :]
    owned = (ii >= 3) & (ii <= nx - 4) & (jj >= 3) & (jj <= nys - 4)
    # 1-based: i in [ids+3, ide-4] on 111 cols -> 0-based [3, 107] == nx-4;
    #          j in [jds+3, jde-3] on 94 faces -> 0-based [3, 90]  == nys-4.
    tend = np.where(owned, tend, 0.0)
    return tend / msfvx[None, :, :]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def band_stats(diff: dict[str, np.ndarray]) -> dict[str, float]:
    return split.per_band_rmse(diff)


def nonspec_masks(diff: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {c: cpu.distance_to_edge(a.shape) >= 1 for c, a in diff.items()}


def sse(diff: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> float:
    total = 0.0
    for comp, arr in diff.items():
        total += float(np.sum(np.square(arr[masks[comp]], dtype=np.float64)))
    return total


def corrcoef(a: dict[str, np.ndarray], b: dict[str, np.ndarray], masks) -> float:
    xs = np.concatenate([a[c][masks[c]].ravel() for c in a])
    ys = np.concatenate([b[c][masks[c]].ravel() for c in b])
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def band_masks_for(diff: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    out = {}
    for comp, arr in diff.items():
        dist = cpu.distance_to_edge(arr.shape)
        if name == "relax_rows_1_4":
            out[comp] = (dist >= 1) & (dist <= 4)
        elif name == "interior_ge_5":
            out[comp] = dist >= 5
        elif name == "nonspec":
            out[comp] = dist >= 1
        else:
            raise ValueError(name)
    return out


def attribution_block(
    r: dict[str, np.ndarray], p: dict[str, np.ndarray]
) -> dict[str, Any]:
    remainder = {c: r[c] - p[c] for c in r}
    blocks: dict[str, Any] = {}
    for band in ("nonspec", "relax_rows_1_4", "interior_ge_5"):
        masks = band_masks_for(r, band)
        sse_r = sse(r, masks)
        sse_rem = sse(remainder, masks)
        blocks[band] = {
            "rmse_R": split.combined_rmse(r, masks),
            "rmse_P": split.combined_rmse(p, masks),
            "rmse_R_minus_P": split.combined_rmse(remainder, masks),
            "sse_R": sse_r,
            "sse_R_minus_P": sse_rem,
            "explained_fraction": (1.0 - sse_rem / sse_r) if sse_r > 0 else None,
            "corr_R_P": corrcoef(r, p, masks),
        }
    return blocks


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_step(step: int):
    gpu = {
        "u_sp1": split.load_gpu_savepoint(step, "sp1_entry", "u"),
        "v_sp1": split.load_gpu_savepoint(step, "sp1_entry", "v"),
        "rublten": split.load_gpu_savepoint(step, "sp2_pbl", "rublten"),
        "rvblten": split.load_gpu_savepoint(step, "sp2_pbl", "rvblten"),
        "ru_tendf": split.load_gpu_savepoint(step, "sp3_tendf", "ru_tendf"),
        "rv_tendf": split.load_gpu_savepoint(step, "sp3_tendf", "rv_tendf"),
        "ru_tend": split.load_gpu_savepoint(step, "l1_rk1_tend", "ru_tend"),
        "rv_tend": split.load_gpu_savepoint(step, "l1_rk1_tend", "rv_tend"),
        "u_save": split.load_gpu_savepoint(step, "l1_rk1_relax", "u_save"),
        "v_save": split.load_gpu_savepoint(step, "l1_rk1_relax", "v_save"),
        "u_l5": split.load_gpu_savepoint(step, "l5_prebdry", "u"),
        "v_l5": split.load_gpu_savepoint(step, "l5_prebdry", "v"),
        "u_sp4": split.load_gpu_savepoint(step, "sp4_exit", "u"),
        "v_sp4": split.load_gpu_savepoint(step, "sp4_exit", "v"),
    }
    ranks = reassemble.load_ranks(split.RUNS / "control/momsp_dumps")
    control = {
        "u_sp1": reassemble.reassemble3d("sp1_entry__u", step, ranks),
        "v_sp1": reassemble.reassemble3d("sp1_entry__v", step, ranks),
        "rublten": reassemble.reassemble3d("sp2_pbl__rublten", step, ranks),
        "rvblten": reassemble.reassemble3d("sp2_pbl__rvblten", step, ranks),
        "mut": reassemble.reassemble2d("sp2_pbl__mut", step, ranks),
        "ru_tendf": reassemble.reassemble3d("sp3_tendf__ru_tendf", step, ranks),
        "rv_tendf": reassemble.reassemble3d("sp3_tendf__rv_tendf", step, ranks),
        "ru_tend": reassemble.reassemble3d("l1_rk1_tend__ru_tend", step, ranks),
        "rv_tend": reassemble.reassemble3d("l1_rk1_tend__rv_tend", step, ranks),
        "u_save": reassemble.reassemble3d("l1_rk1_tend__u_save", step, ranks),
        "v_save": reassemble.reassemble3d("l1_rk1_tend__v_save", step, ranks),
        "u_l5": reassemble.reassemble3d("l5_prebdry__u", step, ranks),
        "v_l5": reassemble.reassemble3d("l5_prebdry__v", step, ranks),
        "u_sp4": reassemble.reassemble3d("sp4_exit__u", step, ranks),
        "v_sp4": reassemble.reassemble3d("sp4_exit__v", step, ranks),
    }
    return gpu, control


def s1_residual(gpu, control, const) -> dict[str, np.ndarray]:
    """Exact Kimi DYN algebra (internal-split :533-565)."""
    msfu = const["mapfac_uy"][None, :, :]
    msfv = const["mapfac_vx"][None, :, :]
    gpu_msfu = const["gpu_mapfac_uy"][None, :, :]
    gpu_msfv = const["gpu_mapfac_vx"][None, :, :]
    dyn_ctl = {
        "u": control["ru_tend"] - control["ru_tendf"] / msfu - control["u_save"],
        "v": control["rv_tend"] - control["rv_tendf"] * (1.0 / msfv) - control["v_save"],
    }
    dyn_gpu = {
        "u": gpu["ru_tend"] - gpu["ru_tendf"] / gpu_msfu - gpu["u_save"],
        "v": gpu["rv_tend"] - gpu["rv_tendf"] * (1.0 / gpu_msfv) - gpu["v_save"],
    }
    return {c: dyn_gpu[c] - dyn_ctl[c] for c in ("u", "v")}


def predicted_field(gpu, control, const, gpu_mu_total: np.ndarray) -> tuple[dict, dict, dict]:
    """P = G - W: GPU periodic lane minus WRF effective lane (per side inputs)."""
    g = {
        "u": gpu_face_mass_u(gpu_mu_total, const["gpu_c1h"], const["gpu_c2h"])
        * gpu_periodic_diff6(gpu["u_sp1"]),
        "v": gpu_face_mass_v(gpu_mu_total, const["gpu_c1h"], const["gpu_c2h"])
        * gpu_periodic_diff6(gpu["v_sp1"]),
    }
    w = {
        "u": wrf_diff6_u_effective(
            control["u_sp1"], control["mut"], const["c1h"], const["c2h"],
            const["msfux"], const["mapfac_uy"],
        ),
        "v": wrf_diff6_v_effective(
            control["v_sp1"], control["mut"], const["c1h"], const["c2h"],
            const["mapfac_vx"], const["msfvy"],
        ),
    }
    return {c: g[c] - w[c] for c in ("u", "v")}, g, w


def spec_exit_check(gpu, control, step: int) -> dict[str, Any]:
    """C-3: GPU L5==SP4 identity; WRF L5->SP4 overwrite; GPU-vs-WRF SP4 bands."""
    gpu_l5_sp4 = {
        "u": gpu["u_sp4"] - gpu["u_l5"],
        "v": gpu["v_sp4"] - gpu["v_l5"],
    }
    wrf_l5_sp4 = {
        "u": control["u_sp4"] - control["u_l5"],
        "v": control["v_sp4"] - control["v_l5"],
    }
    sp4_delta = {
        "u": gpu["u_sp4"] - control["u_sp4"],
        "v": gpu["v_sp4"] - control["v_sp4"],
    }
    return {
        "step": step,
        "gpu_l5_to_sp4": band_stats(gpu_l5_sp4)
        | {"max_abs": float(max(np.abs(a).max() for a in gpu_l5_sp4.values()))},
        "wrf_l5_to_sp4": band_stats(wrf_l5_sp4)
        | {"max_abs": float(max(np.abs(a).max() for a in wrf_l5_sp4.values()))},
        "gpu_vs_wrf_sp4": band_stats(sp4_delta),
    }


def sp2_structure(gpu, control) -> dict[str, Any]:
    """C-1: descriptive structure of the inherited MYNN-input discrepancy."""
    hgt = cpu.load_hgt()
    land = hgt > 0.5
    out: dict[str, Any] = {}
    for comp in ("rublten", "rvblten"):
        d = gpu[comp] - control[comp]
        prof = [split.rms(d[k]) for k in range(d.shape[0])]
        k_arg = int(np.argmax(prof))
        flat = np.abs(d).max(axis=0)
        out[comp] = {
            "rmse": split.rms(d),
            "max_abs": float(np.abs(d).max()),
            "argmax_zero_based_kji": [int(x) for x in np.unravel_index(np.argmax(np.abs(d)), d.shape)],
            "vertical_rmse_profile_argmax_level": k_arg,
            "vertical_rmse_profile_top5_levels": sorted(
                range(len(prof)), key=lambda k: -prof[k]
            )[:5],
            "land_rms_columnmax": split.rms(flat[land]),
            "sea_rms_columnmax": split.rms(flat[~land]),
            "reference_rms_control": split.rms(control[comp]),
        }
    return out


def main() -> None:
    frozen = json.loads(FROZEN_SPLIT.read_text())
    actual_self = split.canonical_digest(frozen, omit="proof_sha256")
    if actual_self != FROZEN_SPLIT_SELF or frozen.get("proof_sha256") != FROZEN_SPLIT_SELF:
        raise RuntimeError("frozen internal-split analysis authority breach")
    frozen_nonspec = frozen["s1_rk_tendency_residual"]["nonspec"]["gpu_rmse"]
    frozen_bands = frozen["s1_rk_tendency_residual"]["delta_dyn"]["bands"]
    frozen_relax = frozen_bands["relax_rows_1_4"]["rmse"]
    frozen_interior = frozen_bands["interior_ge_5"]["rmse"]

    const = split.load_constants()
    # msfux / msfvy additions (contract F-A2; same wrfinput authority).
    from netCDF4 import Dataset

    with Dataset(split.CONTROL_WRFINPUT) as ds:
        const["msfux"] = np.asarray(ds.variables["MAPFAC_UX"][0], dtype=np.float64)
        const["msfvy"] = np.asarray(ds.variables["MAPFAC_VY"][0], dtype=np.float64)

    report: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-dyn-attribution.v1",
        "contract": str(SPRINT / "CONTRACT.md"),
        "authority": {
            "frozen_split_self_hash": FROZEN_SPLIT_SELF,
            "frozen_nonspec_rmse": frozen_nonspec,
            "frozen_relax_rmse": frozen_relax,
            "frozen_interior_rmse": frozen_interior,
            "diff_6th": {"opt": 2, "factor": DIFF_6TH_FACTOR, "coef": DIFF_6TH_COEF, "dt": DT},
        },
        "steps": {},
    }

    verdicts: dict[str, Any] = {}
    for step in (1, 2):
        gpu, control = load_step(step)
        r = s1_residual(gpu, control, const)

        if step == 1:
            ns = nonspec_masks(r)
            got = split.combined_rmse(r, ns)
            rel = abs(got - frozen_nonspec) / frozen_nonspec
            bands_now = band_stats(r)
            rel_relax = abs(bands_now["relax_rows_1_4"] - frozen_relax) / frozen_relax
            rel_int = abs(bands_now["interior_ge_5"] - frozen_interior) / frozen_interior
            fa1 = bool(rel <= 1e-6 and rel_relax <= 1e-4 and rel_int <= 1e-4)
            report["F_A1_reproduction"] = {
                "recomputed_nonspec_rmse": got,
                "relative_error": rel,
                "recomputed_relax_rmse": bands_now["relax_rows_1_4"],
                "recomputed_interior_rmse": bands_now["interior_ge_5"],
                "passed": fa1,
            }
            if not fa1:
                write_and_exit(report, "F_A1_FAILED_AUTHORITY_STOP")
                return

        # GPU mu: init frame for step 1; control in-run mut approximation
        # beyond (documented, sensitivity ~coef*dmu*dflux, dmu rmse 3.5e-4 Pa).
        gpu_mu = const["gpu_mu_total"] if step == 1 else control["mut"]
        p, g_lane, w_lane = predicted_field(gpu, control, const, gpu_mu)

        blocks = attribution_block(r, p)
        report["steps"][str(step)] = {
            "attribution": blocks,
            "lane_magnitudes": {
                "gpu_lane": band_stats(g_lane),
                "wrf_lane": band_stats(w_lane),
                "P": band_stats(p),
                "R": band_stats(r),
            },
            "spec_exit_C3": spec_exit_check(gpu, control, step),
        }
        if step == 1:
            report["sp2_structure_C1"] = sp2_structure(gpu, control)
            gpu1, control1, p1, r1 = gpu, control, p, r

        ex = blocks["nonspec"]["explained_fraction"]
        relax_drop = (
            blocks["relax_rows_1_4"]["rmse_R"] / blocks["relax_rows_1_4"]["rmse_R_minus_P"]
            if blocks["relax_rows_1_4"]["rmse_R_minus_P"] > 0
            else float("inf")
        )
        interior_drop = (
            blocks["interior_ge_5"]["rmse_R"] / blocks["interior_ge_5"]["rmse_R_minus_P"]
            if blocks["interior_ge_5"]["rmse_R_minus_P"] > 0
            else float("inf")
        )
        if ex is None:
            cls = "UNDEFINED"
        elif ex >= 0.95 and relax_drop >= 10.0 and interior_drop >= 10.0:
            cls = "CONFIRMED"
        elif ex >= 0.50:
            cls = "PARTIAL"
        else:
            cls = "REJECTED"
        verdicts[str(step)] = {
            "explained_fraction_nonspec": ex,
            "relax_rmse_drop_factor": relax_drop,
            "interior_rmse_drop_factor": interior_drop,
            "class": cls,
        }

    same = verdicts["1"]["class"] == verdicts["2"]["class"]
    final_class = verdicts["1"]["class"] if same else "PARTIAL"
    report["F_A2_F_A3"] = {
        "per_step": verdicts,
        "step_classes_agree": same,
        "final_class": final_class,
    }

    # GPU-side steps 3..9: L5 == SP4 identity only (C-3 robustness).
    l5_sp4 = {}
    for step in range(3, 10):
        u5 = split.load_gpu_savepoint(step, "l5_prebdry", "u")
        u4 = split.load_gpu_savepoint(step, "sp4_exit", "u")
        v5 = split.load_gpu_savepoint(step, "l5_prebdry", "v")
        v4 = split.load_gpu_savepoint(step, "sp4_exit", "v")
        l5_sp4[str(step)] = float(
            max(np.abs(u4 - u5).max(), np.abs(v4 - v5).max())
        )
    report["gpu_l5_sp4_max_abs_steps_3_9"] = l5_sp4

    write_and_exit(report, final_class)


def write_and_exit(report: dict[str, Any], terminal: str) -> None:
    assert "jax" not in sys.modules and not any(
        m == "gpuwrf" or m.startswith("gpuwrf.") for m in sys.modules
    ), "hygiene breach: jax/gpuwrf imported"
    report["terminal_class"] = terminal
    report["gpu_attestation"] = {
        "gpu_commands_queries_locks": 0,
        "jax_imported": False,
        "gpuwrf_imported": False,
    }
    split.write_self_hashed(OUTPUT, report)
    print(f"TERMINAL {terminal} -> {OUTPUT}")


if __name__ == "__main__":
    main()
