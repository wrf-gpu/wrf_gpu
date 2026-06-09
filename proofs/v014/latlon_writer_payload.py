#!/usr/bin/env python3
"""V0.14 lat/lon writer-payload proof.

CPU-only proof for the writer-only XLAT/XLONG fallback fix. It builds a host
state sized from the real L2 d02 wrfinput, prepares/writes wrfout once without
lat/lon diagnostics and once with the real wrfinput lat/lon diagnostics, and
compares the emitted coordinate payloads against CPU and GPU-native wrfinput.

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
    python proofs/v014/latlon_writer_payload.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.dynamics.metrics import load_wrfinput_metrics  # noqa: E402
from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    _load_static_latlon_writer_diagnostics,
)
from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: E402
from gpuwrf.io.wrfout_writer import (  # noqa: E402
    prepare_wrfout_payload,
    write_prepared_wrfout,
)


RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DOMAIN = "d02"
CPU_INIT_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2") / RUN_ID
GPU_RUN_DIR = Path("/tmp/v0120_merged_run_root") / RUN_ID
CPU_WRFINPUT = CPU_INIT_DIR / "wrfinput_d02"
GPU_NATIVE_WRFINPUT = GPU_RUN_DIR / "wrfinput_d02"
TMP_DIR = Path("/tmp/v014_latlon_writer_payload")
FALLBACK_WRFOUT = TMP_DIR / "wrfout_d02_2026-05-01_19:00:00.fallback"
REAL_WRFOUT = TMP_DIR / "wrfout_d02_2026-05-01_19:00:00.real_latlon"
OUT_JSON = ROOT / "proofs/v014/latlon_writer_payload.json"
OUT_MD = ROOT / "proofs/v014/latlon_writer_payload.md"

LATLON_FIELDS = ("XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V")
MASS_LATLON_FIELDS = ("XLAT", "XLONG")
STATIC_UNTOUCHED_FIELDS = (
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "DN",
    "DNW",
    "RDN",
    "RDNW",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_var(path: Path, name: str) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Dataset(path, "r") as dataset:
        if name not in dataset.variables:
            return None
        variable = dataset.variables[name]
        data = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
        return np.asarray(np.ma.filled(data, np.nan))


def nc_dims(path: Path) -> dict[str, int]:
    with Dataset(path, "r") as dataset:
        return {name: int(len(dim)) for name, dim in dataset.dimensions.items()}


def nc_attrs(path: Path) -> dict[str, Any]:
    names = ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON")
    attrs: dict[str, Any] = {}
    with Dataset(path, "r") as dataset:
        for name in names:
            if hasattr(dataset, name):
                value = getattr(dataset, name)
                attrs[name] = value.item() if hasattr(value, "item") else value
    return attrs


def compare_arrays(candidate: np.ndarray | None, truth: np.ndarray | None) -> dict[str, Any]:
    if candidate is None or truth is None:
        return {
            "status": "MISSING",
            "candidate_present": candidate is not None,
            "truth_present": truth is not None,
        }
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(truth, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "candidate_shape": list(cand.shape),
            "truth_shape": list(ref.shape),
        }
    diff = cand - ref
    finite_candidate = np.isfinite(cand)
    finite_truth = np.isfinite(ref)
    finite_pair = finite_candidate & finite_truth
    total = int(diff.size)
    if not np.any(finite_pair):
        return {
            "status": "NO_FINITE_PAIR",
            "shape": list(diff.shape),
            "total": total,
            "finite_pair": 0,
            "finite_pair_fraction": 0.0,
        }
    vals = diff[finite_pair]
    abs_vals = np.abs(vals)
    worst_flat = int(np.argmax(np.where(finite_pair, np.abs(diff), -np.inf)))
    worst_index = tuple(int(i) for i in np.unravel_index(worst_flat, diff.shape))
    max_abs = float(np.max(abs_vals))
    return {
        "status": "EXACT" if max_abs == 0.0 else "DIFF",
        "exact": bool(max_abs == 0.0),
        "shape": list(diff.shape),
        "total": total,
        "finite_candidate": int(finite_candidate.sum()),
        "finite_truth": int(finite_truth.sum()),
        "finite_pair": int(finite_pair.sum()),
        "finite_pair_fraction": float(finite_pair.sum() / max(total, 1)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
        "bias": float(np.mean(vals)),
        "p99_abs": float(np.percentile(abs_vals, 99.0)),
        "max_abs": max_abs,
        "worst_cell": {
            "index": list(worst_index),
            "candidate": float(cand[worst_index]),
            "truth": float(ref[worst_index]),
            "diff": float(diff[worst_index]),
            "abs_diff": float(abs(diff[worst_index])),
        },
    }


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def build_host_case(wrfinput: Path) -> tuple[Any, Any, Mapping[str, Any]]:
    dims = nc_dims(wrfinput)
    attrs = nc_attrs(wrfinput)
    nx = dims["west_east"]
    ny = dims["south_north"]
    nz = dims["bottom_top"]

    def arr(name: str, shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
        value = read_var(wrfinput, name)
        if value is None:
            return np.full(shape, default, dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    shape_xy = (ny, nx)
    shape_xyz = (nz, ny, nx)
    shape_z = (nz + 1, ny, nx)
    p_pert = arr("P", shape_xyz)
    pb = arr("PB", shape_xyz, 90000.0)
    ph_pert = arr("PH", shape_z)
    phb = arr("PHB", shape_z)
    mu_pert = arr("MU", shape_xy)
    mub = arr("MUB", shape_xy, 85000.0)
    hgt = arr("HGT", shape_xy)
    xland = arr("XLAND", shape_xy, 1.0)
    landmask = np.where(xland <= 1.5, 1.0, 0.0).astype(np.float32)
    theta_pert = arr("T", shape_xyz)
    theta = theta_pert + np.float32(300.0)

    state = SimpleNamespace(
        u=arr("U", (nz, ny, nx + 1)),
        v=arr("V", (nz, ny + 1, nx)),
        w=arr("W", (nz + 1, ny, nx)),
        theta=theta,
        qv=arr("QVAPOR", shape_xyz, 0.008),
        qc=arr("QCLOUD", shape_xyz),
        qi=arr("QICE", shape_xyz),
        qr=arr("QRAIN", shape_xyz),
        p_total=pb + p_pert,
        p_perturbation=p_pert,
        ph_total=phb + ph_pert,
        ph_perturbation=ph_pert,
        mu_total=mub + mu_pert,
        mu_perturbation=mu_pert,
        u10=arr("U10", shape_xy),
        v10=arr("V10", shape_xy),
        t2=arr("T2", shape_xy, 290.0),
        q2=arr("Q2", shape_xy, 0.008),
        psfc=arr("PSFC", shape_xy, 90000.0),
        rainc=arr("RAINC", shape_xy),
        rain_acc=arr("RAINNC", shape_xy),
        rainsh=arr("RAINSH", shape_xy),
        swdown=arr("SWDOWN", shape_xy),
        glw=arr("GLW", shape_xy),
        pblh=arr("PBLH", shape_xy),
        ustar=arr("UST", shape_xy, 0.3),
        hfx=arr("HFX", shape_xy),
        lh=arr("LH", shape_xy),
        t_skin=arr("TSK", shape_xy, 290.0),
        cldfra=arr("CLDFRA", shape_xyz),
        landmask=landmask,
        lu_index=arr("LU_INDEX", shape_xy, 2.0),
        xland=xland,
    )
    metrics = load_wrfinput_metrics(wrfinput)
    grid = SimpleNamespace(
        nx=nx,
        ny=ny,
        nz=nz,
        projection=SimpleNamespace(
            kind={1: "lambert", 2: "polar", 3: "mercator"}.get(int(attrs.get("MAP_PROJ", 1)), "lambert"),
            lat_0=float(attrs.get("CEN_LAT", 0.0)),
            lon_0=float(attrs.get("CEN_LON", 0.0)),
            dx_m=float(attrs.get("DX", 3000.0)),
            dy_m=float(attrs.get("DY", attrs.get("DX", 3000.0))),
            nx=nx,
            ny=ny,
        ),
        vertical=SimpleNamespace(nz=nz, top_pressure_pa=float(np.asarray(read_var(wrfinput, "P_TOP")).reshape(-1)[0])),
        terrain_height=hgt,
        eta_levels=read_var(wrfinput, "ZNW"),
        metrics=metrics,
    )
    namelist = {
        "title": " OUTPUT FROM GPUWRF V014 LATLON WRITER PAYLOAD PROOF",
        "dx": float(attrs.get("DX", 3000.0)),
        "dy": float(attrs.get("DY", attrs.get("DX", 3000.0))),
        "cen_lat": float(attrs.get("CEN_LAT", 0.0)),
        "cen_lon": float(attrs.get("CEN_LON", 0.0)),
        "truelat1": float(attrs.get("TRUELAT1", attrs.get("CEN_LAT", 0.0))),
        "truelat2": float(attrs.get("TRUELAT2", attrs.get("CEN_LAT", 0.0))),
        "stand_lon": float(attrs.get("STAND_LON", attrs.get("CEN_LON", 0.0))),
        "moad_cen_lat": float(attrs.get("CEN_LAT", 0.0)),
        "soil_layers_stag": int(dims.get("soil_layers_stag", 4)),
    }
    return state, grid, namelist


def build_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    xlat = payload["latlon_comparisons"]["XLAT"]["emitted_vs_gpu_native_wrfinput"]
    xlong = payload["latlon_comparisons"]["XLONG"]["emitted_vs_gpu_native_wrfinput"]
    return "\n".join(
        [
            "# V0.14 Lat/Lon Writer Payload",
            "",
            f"Generated UTC: `{payload['generated_utc']}`",
            "",
            f"Verdict: `{verdict}`.",
            "",
            "- Writer diagnostics with real `XLAT`/`XLONG` are selected over the projection fallback.",
            "- With diagnostics absent, the writer still uses the projection fallback for synthetic/no-static callers.",
            (
                f"- Emitted `XLAT` vs GPU-native wrfinput: RMSE `{xlat['rmse']}`, "
                f"max_abs `{xlat['max_abs']}`."
            ),
            (
                f"- Emitted `XLONG` vs GPU-native wrfinput: RMSE `{xlong['rmse']}`, "
                f"max_abs `{xlong['max_abs']}`."
            ),
            "- Static metric payloads checked here are unchanged by the lat/lon diagnostics path.",
            "- Model numerics changed: `false` (writer-only host output payload).",
            "",
            f"Full tables and exact file paths are in `{OUT_JSON.relative_to(ROOT)}`.",
            "",
        ]
    )


def main() -> int:
    require_file(CPU_WRFINPUT)
    require_file(GPU_NATIVE_WRFINPUT)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for path in (FALLBACK_WRFOUT, REAL_WRFOUT):
        if path.exists():
            path.unlink()

    state, grid, namelist = build_host_case(GPU_NATIVE_WRFINPUT)
    run = Gen2Run(GPU_RUN_DIR)
    latlon_payload, latlon_meta = _load_static_latlon_writer_diagnostics(run, DOMAIN, grid=grid)
    if not latlon_payload:
        raise RuntimeError("static lat/lon writer diagnostics did not load")

    run_start = datetime(2026, 5, 1, 18)
    valid_time = datetime(2026, 5, 1, 19)
    fallback_prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        FALLBACK_WRFOUT,
        valid_time=valid_time,
        lead_hours=1.0,
        run_start=run_start,
        diagnostics=None,
    )
    real_prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        REAL_WRFOUT,
        valid_time=valid_time,
        lead_hours=1.0,
        run_start=run_start,
        diagnostics=latlon_payload,
    )
    write_prepared_wrfout(fallback_prepared)
    write_prepared_wrfout(real_prepared)

    selection: dict[str, Any] = {}
    for name in LATLON_FIELDS:
        if name not in latlon_payload:
            continue
        selection[name] = {
            "with_payload_prepared_vs_loaded_payload": compare_arrays(
                real_prepared.fields.get(name), latlon_payload.get(name)
            ),
            "without_payload_prepared_vs_loaded_payload": compare_arrays(
                fallback_prepared.fields.get(name), latlon_payload.get(name)
            ),
        }

    latlon_comparisons: dict[str, Any] = {}
    for name in LATLON_FIELDS:
        emitted = read_var(REAL_WRFOUT, name)
        fallback_emitted = read_var(FALLBACK_WRFOUT, name)
        gpu_native = read_var(GPU_NATIVE_WRFINPUT, name)
        cpu_native = read_var(CPU_WRFINPUT, name)
        if emitted is None and gpu_native is None and cpu_native is None:
            continue
        latlon_comparisons[name] = {
            "emitted_vs_gpu_native_wrfinput": compare_arrays(emitted, gpu_native),
            "emitted_vs_cpu_wrfinput": compare_arrays(emitted, cpu_native),
            "fallback_without_payload_vs_gpu_native_wrfinput": compare_arrays(fallback_emitted, gpu_native),
            "cpu_wrfinput_vs_gpu_native_wrfinput": compare_arrays(cpu_native, gpu_native),
        }

    static_untouched: dict[str, Any] = {}
    for name in STATIC_UNTOUCHED_FIELDS:
        if name in fallback_prepared.fields or name in real_prepared.fields:
            static_untouched[name] = compare_arrays(
                real_prepared.fields.get(name), fallback_prepared.fields.get(name)
            )

    mass_exact = all(
        latlon_comparisons[name]["emitted_vs_gpu_native_wrfinput"].get("exact") is True
        and latlon_comparisons[name]["emitted_vs_cpu_wrfinput"].get("exact") is True
        for name in MASS_LATLON_FIELDS
    )
    real_selected = all(
        selection[name]["with_payload_prepared_vs_loaded_payload"].get("exact") is True
        for name in MASS_LATLON_FIELDS
    )
    fallback_differs_when_absent = all(
        selection[name]["without_payload_prepared_vs_loaded_payload"].get("max_abs", 0.0) > 0.0
        for name in MASS_LATLON_FIELDS
    )
    static_unchanged = all(item.get("exact") is True for item in static_untouched.values())
    verdict = "PASS" if (mass_exact and real_selected and fallback_differs_when_absent and static_unchanged) else "FAIL"

    payload: dict[str, Any] = {
        "schema": "V014LatLonWriterPayloadProof",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "verdict": verdict,
        "inputs": {
            "domain": DOMAIN,
            "cpu_wrfinput": str(CPU_WRFINPUT),
            "gpu_native_wrfinput": str(GPU_NATIVE_WRFINPUT),
            "gpu_run_dir": str(GPU_RUN_DIR),
        },
        "outputs": {
            "fallback_wrfout_without_latlon_payload": str(FALLBACK_WRFOUT),
            "real_latlon_wrfout_with_payload": str(REAL_WRFOUT),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
        "latlon_loader": latlon_meta,
        "selection": selection,
        "latlon_comparisons": latlon_comparisons,
        "static_metric_fields_unchanged_by_latlon_path": static_untouched,
        "acceptance": {
            "mass_xlat_xlong_exact_vs_cpu_and_gpu_native_wrfinput": bool(mass_exact),
            "writer_selects_real_payload_when_supplied": bool(real_selected),
            "projection_fallback_used_only_when_payload_absent": bool(fallback_differs_when_absent),
            "static_metric_fields_unchanged": bool(static_unchanged),
        },
        "model_numerics_changed": False,
        "model_numerics_note": (
            "No dycore, physics, State, GridSpec, or OperationalNamelist runtime "
            "arrays are changed; this is a host-only wrfout payload selection."
        ),
        "remaining_static_base_exclusions": ["PHB", "HGT", "PB", "MUB"],
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
