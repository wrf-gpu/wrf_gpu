#!/usr/bin/env python3
"""CPU-only smoke proof for the M7 NetCDF wrfout writer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import xarray as xr
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.io.wrfout_writer import (
    DOWNSTREAM_CRITICAL_VARIABLES,
    MINIMUM_WRFOUT_VARIABLES,
    WRFOUT_VARIABLE_SPECS,
    write_wrfout_netcdf,
)
from gpuwrf.paths import reference_path


SPRINT_DIR = ROOT / "proofs" / "generated" / "2026-05-27-m7-netcdf-writer"
DEFAULT_REFERENCE = reference_path(
    "runs",
    "wrf_l3",
    "20260525_18z_l3_24h_20260526T221207Z",
    "wrfout_d02_2026-05-25_18:00:00",
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/wrf_gpu_ncwriter_m7_netcdf_writer")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def build_synthetic_case(reference: Path) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, Any], datetime, datetime]:
    with Dataset(reference) as dataset:
        ny = len(dataset.dimensions["south_north"])
        nx = len(dataset.dimensions["west_east"])
        nz = len(dataset.dimensions["bottom_top"])
        dx = float(dataset.getncattr("DX"))
        dy = float(dataset.getncattr("DY"))
        cen_lat = float(dataset.getncattr("CEN_LAT"))
        cen_lon = float(dataset.getncattr("CEN_LON"))
        truelat1 = float(dataset.getncattr("TRUELAT1"))
        truelat2 = float(dataset.getncattr("TRUELAT2"))
        stand_lon = float(dataset.getncattr("STAND_LON"))
        moad_cen_lat = float(dataset.getncattr("MOAD_CEN_LAT"))
        run_start = datetime.strptime(str(dataset.getncattr("START_DATE")), "%Y-%m-%d_%H:%M:%S")

    y2, x2 = np.indices((ny, nx), dtype=np.float32)
    z3 = np.arange(nz, dtype=np.float32)[:, None, None]
    zf = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    terrain = 50.0 + 0.3 * y2 + 0.1 * x2
    landmask = np.where((x2 + y2) % 7 == 0, 0.0, 1.0).astype(np.float32)

    pb = (91_000.0 - 900.0 * z3 + 0.04 * terrain[None, :, :]).astype(np.float32)
    p_pert = (120.0 + 0.5 * z3 + 0.02 * x2[None, :, :]).astype(np.float32)
    phb = (9.81 * (terrain[None, :, :] + 500.0 * zf)).astype(np.float32)
    ph_pert = (2.0 + 0.1 * zf + 0.01 * y2[None, :, :]).astype(np.float32)
    mub = (88_000.0 + 0.2 * terrain).astype(np.float32)
    mu_pert = (55.0 + 0.03 * x2 + 0.02 * y2).astype(np.float32)

    state = SimpleNamespace(
        u=(5.0 + np.zeros((nz, ny, nx + 1), dtype=np.float32)),
        v=(-2.0 + np.zeros((nz, ny + 1, nx), dtype=np.float32)),
        w=np.zeros((nz + 1, ny, nx), dtype=np.float32),
        theta=(300.0 + 0.25 * z3 + 0.01 * y2[None, :, :]).astype(np.float32),
        qv=(0.008 + 1.0e-5 * z3 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qc=(1.0e-5 * ((z3 % 4) == 0)).astype(np.float32) + np.zeros((nz, ny, nx), dtype=np.float32),
        qi=(5.0e-6 * ((z3 % 5) == 0)).astype(np.float32) + np.zeros((nz, ny, nx), dtype=np.float32),
        qr=(2.0e-6 * ((z3 % 6) == 0)).astype(np.float32) + np.zeros((nz, ny, nx), dtype=np.float32),
        p_total=pb + p_pert,
        p_perturbation=p_pert,
        ph_total=phb + ph_pert,
        ph_perturbation=ph_pert,
        mu_total=mub + mu_pert,
        mu_perturbation=mu_pert,
        t2=(290.0 + 0.03 * y2).astype(np.float32),
        q2=(0.0075 + np.zeros((ny, nx), dtype=np.float32)),
        u10=(4.0 + 0.01 * x2).astype(np.float32),
        v10=(-1.5 + 0.01 * y2).astype(np.float32),
        psfc=(pb[0] + p_pert[0]).astype(np.float32),
        rainc=np.zeros((ny, nx), dtype=np.float32),
        rain_acc=(0.4 + 0.001 * x2).astype(np.float32),
        rainsh=np.zeros((ny, nx), dtype=np.float32),
        swdown=(620.0 + 0.1 * x2).astype(np.float32),
        glw=(310.0 + 0.1 * y2).astype(np.float32),
        pblh=(700.0 + 0.5 * y2).astype(np.float32),
        ustar=(0.35 + np.zeros((ny, nx), dtype=np.float32)),
        hfx=(25.0 + 0.2 * y2).astype(np.float32),
        lh=(80.0 + 0.15 * x2).astype(np.float32),
        t_skin=(291.0 + 0.04 * y2).astype(np.float32),
        cldfra=np.clip(np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None], 0.0, 1.0)
        + np.zeros((nz, ny, nx), dtype=np.float32),
        landmask=landmask,
        lu_index=np.where(landmask > 0.5, 2.0, 17.0).astype(np.float32),
    )
    grid = SimpleNamespace(
        nx=nx,
        ny=ny,
        nz=nz,
        projection=SimpleNamespace(kind="lambert", lat_0=cen_lat, lon_0=cen_lon, dx_m=dx, dy_m=dy, nx=nx, ny=ny),
        vertical=SimpleNamespace(nz=nz, top_pressure_pa=5_000.0),
        terrain_height=terrain.astype(np.float32),
    )
    namelist = {
        "title": " OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF WRITER",
        "truelat1": truelat1,
        "truelat2": truelat2,
        "stand_lon": stand_lon,
        "moad_cen_lat": moad_cen_lat,
        "cen_lat": cen_lat,
        "cen_lon": cen_lon,
        "dx": dx,
        "dy": dy,
        "soil_layers_stag": 4,
    }
    return state, grid, namelist, run_start, run_start


def schema_compare(reference: Path, candidate: Path) -> dict[str, Any]:
    with Dataset(reference) as ref, Dataset(candidate) as out:
        missing = [name for name in MINIMUM_WRFOUT_VARIABLES if name not in out.variables]
        downstream_missing = [name for name in DOWNSTREAM_CRITICAL_VARIABLES if name not in out.variables]
        dimension_mismatches = []
        dtype_mismatches = []
        attribute_mismatches = []
        for name in MINIMUM_WRFOUT_VARIABLES:
            if name not in ref.variables or name not in out.variables:
                continue
            ref_var = ref.variables[name]
            out_var = out.variables[name]
            if tuple(ref_var.dimensions) != tuple(out_var.dimensions):
                dimension_mismatches.append(
                    {
                        "variable": name,
                        "reference": list(ref_var.dimensions),
                        "candidate": list(out_var.dimensions),
                    }
                )
            if str(np.dtype(ref_var.dtype)) != str(np.dtype(out_var.dtype)):
                dtype_mismatches.append(
                    {
                        "variable": name,
                        "reference": str(np.dtype(ref_var.dtype)),
                        "candidate": str(np.dtype(out_var.dtype)),
                    }
                )
            for attr in ("units", "description", "MemoryOrder", "stagger"):
                if attr not in ref_var.ncattrs() and attr not in out_var.ncattrs():
                    continue
                ref_attr = ref_var.getncattr(attr) if attr in ref_var.ncattrs() else ""
                out_attr = out_var.getncattr(attr) if attr in out_var.ncattrs() else ""
                if str(ref_attr) != str(out_attr):
                    attribute_mismatches.append(
                        {"variable": name, "attribute": attr, "reference": str(ref_attr), "candidate": str(out_attr)}
                    )
        return {
            "missing_minimum_variables": missing,
            "downstream_critical_missing": downstream_missing,
            "dimension_mismatches": dimension_mismatches,
            "dtype_mismatches": dtype_mismatches,
            "attribute_mismatches": attribute_mismatches,
        }


def base_perturbation_checks(path: Path, state: SimpleNamespace) -> dict[str, float]:
    with Dataset(path) as dataset:
        p_error = np.max(np.abs(dataset["P"][0] + dataset["PB"][0] - state.p_total))
        ph_error = np.max(np.abs(dataset["PH"][0] + dataset["PHB"][0] - state.ph_total))
        mu_error = np.max(np.abs(dataset["MU"][0] + dataset["MUB"][0] - state.mu_total))
        return {
            "P_plus_PB_reconstructs_written_total_max_abs_error": float(p_error),
            "PH_plus_PHB_reconstructs_written_total_max_abs_error": float(ph_error),
            "MU_plus_MUB_reconstructs_written_total_max_abs_error": float(mu_error),
            "P_field_present": float("P" in dataset.variables),
            "PB_field_present": float("PB" in dataset.variables),
            "PH_field_present": float("PH" in dataset.variables),
            "PHB_field_present": float("PHB" in dataset.variables),
            "MU_field_present": float("MU" in dataset.variables),
            "MUB_field_present": float("MUB" in dataset.variables),
        }


def xarray_open_check(reference: Path, candidate: Path) -> dict[str, Any]:
    with xr.open_dataset(reference, engine="netcdf4", decode_times=False) as ref_ds:
        ref_vars = sorted(ref_ds.variables)
        ref_dims = {name: int(size) for name, size in ref_ds.sizes.items()}
    with xr.open_dataset(candidate, engine="netcdf4", decode_times=False) as out_ds:
        out_vars = sorted(out_ds.variables)
        out_dims = {name: int(size) for name, size in out_ds.sizes.items()}
    return {
        "reference_opened": True,
        "candidate_opened": True,
        "reference_variable_count": len(ref_vars),
        "candidate_variable_count": len(out_vars),
        "candidate_dimensions": out_dims,
        "reference_dimensions": ref_dims,
    }


def write_compat_matrix(path: Path, proof: dict[str, Any]) -> None:
    schema = proof["schema_comparison"]
    lines = [
        "# M7 NetCDF Writer Compatibility Matrix v2",
        "",
        f"Generated UTC: {proof['generated_utc']}",
        f"CPU reference: `{proof['reference_file']}`",
        f"Candidate writer output: `{proof['candidate_file']}`",
        "",
        "## Summary",
        "",
        f"- Minimum variable count: {proof['minimum_variable_count']}",
        f"- Downstream-critical missing fields: {len(schema['downstream_critical_missing'])}",
        f"- AC1 minimum missing fields: {len(schema['missing_minimum_variables'])}",
        f"- AC1 dimension mismatches: {len(schema['dimension_mismatches'])}",
        f"- AC1 dtype mismatches: {len(schema['dtype_mismatches'])}",
        f"- AC1 metadata mismatches: {len(schema['attribute_mismatches'])}",
        f"- Verdict: {'PASS' if proof['pass'] else 'FAIL'}",
        "",
        "## Minimum Variable Rows",
        "",
        "| Variable | Downstream-critical | Reference dims | Candidate dims | Dtype match | Attr match |",
        "|---|---:|---|---|---:|---:|",
    ]
    dim_mismatch = {item["variable"]: item for item in schema["dimension_mismatches"]}
    dtype_mismatch = {item["variable"]: item for item in schema["dtype_mismatches"]}
    attr_bad = {item["variable"] for item in schema["attribute_mismatches"]}
    with Dataset(proof["reference_file"]) as ref, Dataset(proof["candidate_file"]) as out:
        for name in MINIMUM_WRFOUT_VARIABLES:
            ref_dims = list(ref.variables[name].dimensions) if name in ref.variables else []
            out_dims = list(out.variables[name].dimensions) if name in out.variables else []
            lines.append(
                f"| `{name}` | {'YES' if name in DOWNSTREAM_CRITICAL_VARIABLES else 'NO'} | "
                f"`{ref_dims}` | `{out_dims}` | {'NO' if name in dtype_mismatch else 'YES'} | "
                f"{'NO' if name in attr_bad else 'YES'} |"
            )
    lines.append("")
    if dim_mismatch:
        lines.extend(["## Dimension Mismatches", "", "```json", json.dumps(schema["dimension_mismatches"], indent=2), "```", ""])
    if dtype_mismatch:
        lines.extend(["## Dtype Mismatches", "", "```json", json.dumps(schema["dtype_mismatches"], indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(reference: Path, output_dir: Path, output_root: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    state, grid, namelist, run_start, valid_time = build_synthetic_case(reference)
    candidate = output_root / "wrfout_d02_2026-05-25_18:00:00"
    write_wrfout_netcdf(
        state,
        grid,
        namelist,
        candidate,
        valid_time=valid_time,
        lead_hours=0.0,
        run_start=run_start,
    )
    with Dataset(candidate) as dataset:
        present = sorted(name for name in MINIMUM_WRFOUT_VARIABLES if name in dataset.variables)
        times = b"".join(dataset["Times"][0, :].astype("S1").tolist()).decode("ascii")
        xtime = float(dataset["XTIME"][0])
        required_dims = {
            name: int(len(dataset.dimensions[name]))
            for name in (
                "DateStrLen",
                "west_east",
                "west_east_stag",
                "south_north",
                "south_north_stag",
                "bottom_top",
                "bottom_top_stag",
                "soil_layers_stag",
            )
        }
        required_dims["Time"] = int(len(dataset.dimensions["Time"]))
    proof = {
        "schema": "m7_netcdf_writer_roundtrip_proof_v1",
        "generated_utc": utc_now(),
        "reference_file": str(reference),
        "candidate_file": str(candidate),
        "minimum_variable_count": len(MINIMUM_WRFOUT_VARIABLES),
        "downstream_critical_variable_count": len(DOWNSTREAM_CRITICAL_VARIABLES),
        "present_minimum_variables": present,
        "times_value": times,
        "xtime_minutes": xtime,
        "required_dimensions": required_dims,
        "xarray_open_check": xarray_open_check(reference, candidate),
        "schema_comparison": schema_compare(reference, candidate),
        "base_perturbation_checks": base_perturbation_checks(candidate, state),
    }
    comparison = proof["schema_comparison"]
    proof["pass"] = (
        not comparison["missing_minimum_variables"]
        and not comparison["downstream_critical_missing"]
        and not comparison["dimension_mismatches"]
        and not comparison["dtype_mismatches"]
        and not comparison["attribute_mismatches"]
        and proof["base_perturbation_checks"]["P_plus_PB_reconstructs_written_total_max_abs_error"] <= 1.0e-5
        and proof["base_perturbation_checks"]["PH_plus_PHB_reconstructs_written_total_max_abs_error"] <= 1.0e-5
        and proof["base_perturbation_checks"]["MU_plus_MUB_reconstructs_written_total_max_abs_error"] <= 1.0e-5
        and proof["times_value"] == "2026-05-25_18:00:00"
        and proof["xtime_minutes"] == 0.0
    )
    write_json(output_dir / "roundtrip_proof.json", proof)
    write_compat_matrix(output_dir / "compat_matrix_v2.md", proof)
    return proof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proof = run(args.reference, args.output_dir, args.output_root)
    print("M7 NetCDF writer smoke complete")
    print(f"roundtrip_proof: {args.output_dir / 'roundtrip_proof.json'}")
    print(f"compat_matrix_v2: {args.output_dir / 'compat_matrix_v2.md'}")
    print(f"candidate_file: {proof['candidate_file']}")
    print(f"pass: {proof['pass']}")
    return 0 if proof["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
