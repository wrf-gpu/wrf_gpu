"""RRTMG topo-shading / slope-radiation proof against a real WRF fixture.

CPU-first.  The proof feeds WRF history-file columns into the JAX RRTMG SW/LW
kernels, using real XLAT/XLONG/HGT/map-rotation fields and WRF-held solar time
(history valid time minus radt/2).  It reports gross SWDOWN, topographic
SWNORM-equivalent, GLW, and the topographic correction ratio.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.coupling.physics_couplers import (  # noqa: E402
    _compute_solar_geometry,
    _wrf_topographic_shadow_mask,
    build_radiation_static_from_wrf_fields,
)
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column  # noqa: E402
from gpuwrf.physics.rrtmg_sw import (  # noqa: E402
    RRTMGSWColumnState,
    RRTMGSWTopographyState,
    solve_rrtmg_sw_column,
)


DEFAULT_RUN_DIR = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DEFAULT_DOMAIN = "d02"
DEFAULT_VALID = "2026-05-22_12:00:00"
GRAVITY = 9.80665
RD = 287.0
CP = 1004.0
P00 = 100000.0
RCP = RD / CP
RRSW_SCON = 1368.22


def _valid_time_from_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def _squeeze_var(ds: nc.Dataset, name: str) -> np.ndarray:
    return np.squeeze(np.asarray(np.ma.filled(ds.variables[name][:], np.nan)))


def _to_columns(arr3d: np.ndarray, indices: np.ndarray) -> np.ndarray:
    nz, ny, nx = arr3d.shape
    return np.moveaxis(arr3d, 0, -1).reshape(ny * nx, nz)[indices].astype(np.float64)


def _pair_metrics(gpu: np.ndarray, wrf: np.ndarray) -> dict[str, float]:
    diff = gpu - wrf
    return {
        "wrf_mean": float(np.mean(wrf)),
        "gpu_mean": float(np.mean(gpu)),
        "bias": float(np.mean(diff)),
        "bias_pct": float(100.0 * np.mean(diff) / max(float(np.mean(wrf)), 1.0e-6)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _load_fixture(run_dir: Path, domain: str, valid_stamp: str) -> dict[str, object]:
    path = run_dir / f"wrfout_{domain}_{valid_stamp}"
    with nc.Dataset(path) as ds:
        fields = {name: _squeeze_var(ds, name) for name in (
            "T", "P", "PB", "QVAPOR", "QCLOUD", "QICE", "QSNOW", "QGRAUP",
            "CLDFRA", "PH", "PHB", "COSZEN", "ALBEDO", "EMISS", "TSK",
            "SWDOWN", "SWDNB", "SWNORM", "SWDNTC", "GLW", "XLAT", "XLONG",
            "HGT", "MAPFAC_MX", "MAPFAC_MY", "SINALPHA", "COSALPHA", "XLAND",
        )}
        dx_m = float(getattr(ds, "DX", 3000.0))
        dy_m = float(getattr(ds, "DY", 3000.0))
    p_full = fields["P"] + fields["PB"]
    theta = fields["T"] + 300.0
    temperature = theta * (p_full / P00) ** RCP
    z_w = (fields["PH"] + fields["PHB"]) / GRAVITY
    dz = z_w[1:, :, :] - z_w[:-1, :, :]
    rho = p_full / (RD * temperature * (1.0 + 0.61 * fields["QVAPOR"]))
    fields.update(T_abs=temperature, p_full=p_full, dz=dz, rho=rho, source_file=str(path), dx_m=dx_m, dy_m=dy_m)
    return fields


def _select_columns(fields: dict[str, np.ndarray], max_columns: int) -> np.ndarray:
    flat = np.arange(fields["COSZEN"].size)
    day_land = (
        (fields["XLAND"].reshape(-1) < 1.5)
        & (fields["COSZEN"].reshape(-1) > 0.05)
        & (fields["SWDOWN"].reshape(-1) > 1.0)
    )
    candidates = flat[day_land]
    topo_delta = np.abs(fields["SWNORM"].reshape(-1)[candidates] - fields["SWDOWN"].reshape(-1)[candidates])
    order = np.argsort(topo_delta)[::-1]
    if max_columns > 0:
        order = order[:max_columns]
    return candidates[order].astype(np.int64)


def _run_sw(
    fields: dict[str, np.ndarray],
    indices: np.ndarray,
    topography: RRTMGSWTopographyState,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    coszen = fields["COSZEN"].reshape(-1)[indices].astype(np.float64)
    swdntc = fields["SWDNTC"].reshape(-1)[indices].astype(np.float64)
    solar_source_scale = swdntc / np.maximum(coszen * RRSW_SCON, 1.0e-6)
    arrays = {
        "T": _to_columns(fields["T_abs"], indices),
        "p": _to_columns(fields["p_full"], indices),
        "qv": _to_columns(fields["QVAPOR"], indices),
        "qc": _to_columns(fields["QCLOUD"], indices),
        "qi": _to_columns(fields["QICE"], indices),
        "qs": _to_columns(fields["QSNOW"], indices),
        "qg": _to_columns(fields["QGRAUP"], indices),
        "cldfra": _to_columns(fields["CLDFRA"], indices),
        "dz": _to_columns(fields["dz"], indices),
        "rho": _to_columns(fields["rho"], indices),
        "albedo": fields["ALBEDO"].reshape(-1)[indices].astype(np.float64),
        "coszen": coszen,
        "solar_source_scale": solar_source_scale,
    }
    outputs: dict[str, list[np.ndarray]] = {
        "swdown": [],
        "swnorm": [],
        "correction": [],
        "diffuse_fraction": [],
    }
    for start in range(0, indices.size, batch_size):
        end = min(start + batch_size, indices.size)
        state = RRTMGSWColumnState(
            *(jnp.asarray(arrays[name][start:end]) for name in (
                "T", "p", "qv", "qc", "qi", "qs", "qg", "cldfra", "albedo", "coszen", "dz", "rho"
            )),
            solar_source_scale=jnp.asarray(arrays["solar_source_scale"][start:end]),
        )
        topo_batch = RRTMGSWTopographyState(
            *(jnp.asarray(getattr(topography, name)[start:end]) for name in RRTMGSWTopographyState._fields)
        )
        sw = solve_rrtmg_sw_column(state, debug=False, topography=topo_batch)
        outputs["swdown"].append(np.asarray(jax.device_get(sw.surface_down)))
        outputs["swnorm"].append(np.asarray(jax.device_get(sw.surface_down_topographic)))
        outputs["correction"].append(np.asarray(jax.device_get(sw.topographic_correction_factor)))
        outputs["diffuse_fraction"].append(np.asarray(jax.device_get(sw.surface_diffuse_fraction)))
    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def _run_lw(fields: dict[str, np.ndarray], indices: np.ndarray, *, batch_size: int) -> dict[str, np.ndarray]:
    arrays = {
        "T": _to_columns(fields["T_abs"], indices),
        "p": _to_columns(fields["p_full"], indices),
        "qv": _to_columns(fields["QVAPOR"], indices),
        "qc": _to_columns(fields["QCLOUD"], indices),
        "qi": _to_columns(fields["QICE"], indices),
        "qs": _to_columns(fields["QSNOW"], indices),
        "qg": _to_columns(fields["QGRAUP"], indices),
        "cldfra": _to_columns(fields["CLDFRA"], indices),
        "dz": _to_columns(fields["dz"], indices),
        "rho": _to_columns(fields["rho"], indices),
        "tsk": fields["TSK"].reshape(-1)[indices].astype(np.float64),
        "emiss": fields["EMISS"].reshape(-1)[indices].astype(np.float64),
    }
    glw = []
    for start in range(0, indices.size, batch_size):
        end = min(start + batch_size, indices.size)
        state = RRTMGLWColumnState(
            *(jnp.asarray(arrays[name][start:end]) for name in (
                "T", "p", "qv", "qc", "qi", "qs", "qg", "cldfra", "tsk", "emiss", "dz", "rho"
            ))
        )
        lw = solve_rrtmg_lw_column(state, debug=False)
        glw.append(np.asarray(jax.device_get(lw.surface_down)))
    return {"glw": np.concatenate(glw)}


def run_proof(
    *,
    run_dir: Path,
    domain: str,
    valid_stamp: str,
    max_columns: int,
    batch_size: int,
    out_json: Path,
    out_status: Path,
) -> dict[str, object]:
    fields = _load_fixture(run_dir, domain, valid_stamp)
    indices = _select_columns(fields, max_columns=max_columns)
    valid_time = _valid_time_from_stamp(valid_stamp)
    held_time = valid_time - timedelta(minutes=15)
    dx_m = float(fields["dx_m"])
    dy_m = float(fields["dy_m"])

    static = build_radiation_static_from_wrf_fields(
        fields["XLAT"],
        fields["XLONG"],
        fields["HGT"],
        dx_m=dx_m,
        dy_m=dy_m,
        msftx=fields["MAPFAC_MX"],
        msfty=fields["MAPFAC_MY"],
        sina=fields["SINALPHA"],
        cosa=fields["COSALPHA"],
    )
    geometry = _compute_solar_geometry(fields["XLAT"], fields["XLONG"], held_time, lead_seconds=0.0)
    shadow_mask = _wrf_topographic_shadow_mask(
        static.terrain_height_m,
        latitude_deg=static.xlat_deg,
        coszen=fields["COSZEN"],
        declination_rad=geometry.declination_rad,
        hour_angle_rad=geometry.hour_angle_rad,
        sina=static.sina,
        cosa=static.cosa,
        dx_m=dx_m,
        dy_m=dy_m,
        shadow_length_m=25000.0,
    )
    flat = lambda a: np.asarray(a).reshape(-1)[indices].astype(np.float64)
    topography = RRTMGSWTopographyState(
        latitude_deg=flat(static.xlat_deg),
        declination_rad=np.full(indices.size, float(np.asarray(geometry.declination_rad)), dtype=np.float64),
        hour_angle_rad=flat(geometry.hour_angle_rad),
        slope_rad=flat(static.slope_rad),
        slope_azimuth_rad=flat(static.slope_azimuth_rad),
        shadow_mask=np.asarray(shadow_mask).reshape(-1)[indices].astype(np.int32),
    )

    sw = _run_sw(fields, indices, topography, batch_size=batch_size)
    lw = _run_lw(fields, indices, batch_size=batch_size)
    wrf_swdown = flat(fields["SWDOWN"])
    wrf_swdnb = flat(fields["SWDNB"])
    wrf_swnorm = flat(fields["SWNORM"])
    wrf_glw = flat(fields["GLW"])
    wrf_correction = wrf_swnorm / np.maximum(wrf_swdown, 1.0e-6)

    metrics = {
        "gross_swdown_vs_wrf_SWDNB": _pair_metrics(sw["swdown"], wrf_swdnb),
        "gross_swdown_vs_wrf_SWDOWN": _pair_metrics(sw["swdown"], wrf_swdown),
        "topographic_swnorm_vs_wrf_SWNORM": _pair_metrics(sw["swnorm"], wrf_swnorm),
        "topographic_correction_vs_wrf_SWNORM_over_SWDOWN": _pair_metrics(sw["correction"], wrf_correction),
        "glw_vs_wrf_GLW": _pair_metrics(lw["glw"], wrf_glw),
    }
    tolerances = {
        "gross_swdown_rmse_Wm2": 15.0,
        "topographic_swnorm_rmse_Wm2": 35.0,
        "topographic_correction_rmse": 0.05,
        "glw_rmse_Wm2": 35.0,
    }
    checks = {
        "gross_swdown": metrics["gross_swdown_vs_wrf_SWDNB"]["rmse"] <= tolerances["gross_swdown_rmse_Wm2"],
        "topographic_swnorm": metrics["topographic_swnorm_vs_wrf_SWNORM"]["rmse"] <= tolerances["topographic_swnorm_rmse_Wm2"],
        "topographic_correction": metrics["topographic_correction_vs_wrf_SWNORM_over_SWDOWN"]["rmse"] <= tolerances["topographic_correction_rmse"],
        "glw": metrics["glw_vs_wrf_GLW"]["rmse"] <= tolerances["glw_rmse_Wm2"],
    }
    result = {
        "schema": "v0110_rrtmg_slope_parity",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run_dir": str(run_dir),
        "domain": domain,
        "source_file": fields["source_file"],
        "valid_time_utc": valid_time.isoformat(),
        "held_radiation_time_utc": held_time.isoformat(),
        "selection": {
            "method": "largest |WRF_SWNORM - WRF_SWDOWN| land/daylit columns",
            "selected_columns": int(indices.size),
            "domain_shape": list(fields["COSZEN"].shape),
            "max_columns": int(max_columns),
            "batch_size": int(batch_size),
            "dx_m": dx_m,
            "dy_m": dy_m,
        },
        "wiring": {
            "uses_real_xlat_xlong": True,
            "topo_shading": 1,
            "slope_rad": 1,
            "land_coupled_albedo_emissivity_path": "runtime RRTMG accepts NoahMPLandState.albedo/emiss over XLAND<1.5; this proof feeds WRF ALBEDO/EMISS fixture fields directly",
        },
        "topography": {
            "shadow_mask_selected_count": int(np.asarray(topography.shadow_mask).sum()),
            "slope_rad_mean": float(np.mean(topography.slope_rad)),
            "slope_rad_max": float(np.max(topography.slope_rad)),
            "wrf_topographic_delta_mean_Wm2": float(np.mean(wrf_swnorm - wrf_swdown)),
            "gpu_topographic_delta_mean_Wm2": float(np.mean(sw["swnorm"] - sw["swdown"])),
            "gpu_correction_minmax": [float(np.min(sw["correction"])), float(np.max(sw["correction"]))],
            "wrf_correction_minmax": [float(np.min(wrf_correction)), float(np.max(wrf_correction))],
        },
        "tolerances": tolerances,
        "checks": checks,
        "metrics": metrics,
        "notes": [
            "WRF history COSZEN/SWDOWN are radiation-cadence-held fields; proof uses valid_time - 15 min for solar declination/hour angle.",
            "WRF gross SWDOWN/SWDNB and WRF slope-dependent SWNORM are reported separately. Runtime surface SOLDN uses the SWNORM-equivalent when slope_rad=1.",
            "The proof isolates radiation/topography by feeding WRF history columns to RRTMG; it is not a forecast-state self-compare.",
            "The selected d02 daytime fixture has zero ray-cast shadow cells; the exercised terrain signal is WRF slope/aspect radiation.",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = [
        "# v0.11.0 RRTMG slope-radiation status",
        "",
        f"- status: {result['status']}",
        f"- fixture: `{fields['source_file']}`",
        f"- selected columns: {indices.size} largest terrain-radiation deltas",
        f"- gross SWDOWN RMSE vs WRF SWDNB: {metrics['gross_swdown_vs_wrf_SWDNB']['rmse']:.3f} W m-2",
        f"- topo SWNORM RMSE vs WRF SWNORM: {metrics['topographic_swnorm_vs_wrf_SWNORM']['rmse']:.3f} W m-2",
        f"- GLW RMSE vs WRF GLW: {metrics['glw_vs_wrf_GLW']['rmse']:.3f} W m-2",
        f"- selected shadow-mask count: {result['topography']['shadow_mask_selected_count']}",
        "- shadow note: this d02 daytime fixture exercises slope/aspect; computed WRF-style ray shadows are zero in the selected columns.",
        "",
        "Runtime wiring: real XLAT/XLONG radiation static fields are on the operational namelist; `topo_shading=1` and `slope_rad=1` are read from the Canary WRF namelist; Noah-MP `albedo`/`emiss` are used over land when a Noah-MP land carry is present.",
    ]
    out_status.write_text("\n".join(status) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--max-columns", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out-json", type=Path, default=ROOT / "proofs/v0110/rrtmg_slope_parity.json")
    parser.add_argument("--out-status", type=Path, default=ROOT / "proofs/v0110/rrtmg_slope_status.md")
    args = parser.parse_args()
    result = run_proof(
        run_dir=args.run_dir,
        domain=args.domain,
        valid_stamp=args.valid,
        max_columns=args.max_columns,
        batch_size=args.batch_size,
        out_json=args.out_json,
        out_status=args.out_status,
    )
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, sort_keys=True))


if __name__ == "__main__":
    main()
