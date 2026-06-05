"""Wave-B3 daily-wrapper timing and wrfout parity gate.

Compares the pre-B3 output wrapper semantics against the B3 path:

* old diagnostics: compute M9 diagnostics, then recompute surface/PBL once more
  solely to obtain Q2; writer then runs its fallback surface solve for HFX/LH.
* B3 diagnostics: Q2 comes from the already-computed M9 surface diagnostic,
  HFX/LH/UST are supplied to the writer, and grid-static fields come from a
  per-run cache.

The forecast itself is advanced with the existing segmented operational entry.
No forecast kernel changes are made by this proof.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import time
from typing import Any, Callable

import jax
import numpy as np
from netCDF4 import Dataset

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import (
    DailyPipelineConfig,
    _build_real_case,
    _surface_diagnostics_for_output,
    _wrfout_name,
    finite_summary,
)
from gpuwrf.io.wrfout_writer import (
    build_wrfout_static_field_cache,
    prepare_wrfout_payload,
    write_prepared_wrfout,
)
from gpuwrf.runtime.operational_mode import (
    _enforce_operational_precision,
    compute_m9_diagnostics,
    run_forecast_operational_segmented,
    surface_layer_diagnostics,
)


L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
FIELD_TOLERANCES = {
    # Old wrfout HFX/LH came from the writer fallback after early float32 host
    # coercion. B3 reuses the same surface diagnostic computed in M9 at fp64 and
    # writes float32. The tolerance is sub-milliwatt per m2 scale; all other fields
    # remain strict bit-identical.
    "HFX": {"atol": 2.0e-3, "rtol": 0.0, "units": "W m-2"},
    "LH": {"atol": 2.0e-3, "rtol": 0.0, "units": "W m-2"},
}


def _time_repeated(label: str, fn: Callable[[], Any], reps: int) -> tuple[Any, dict[str, Any]]:
    warm = fn()
    samples: list[float] = []
    result = warm
    for _ in range(int(reps)):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return result, {
        "label": label,
        "samples_s": samples,
        "min_s": min(samples),
        "median_s": float(np.median(samples)),
    }


def _old_surface_diagnostics(state, namelist, lead_seconds: float) -> dict[str, np.ndarray]:
    m9 = compute_m9_diagnostics(state, namelist, lead_seconds)
    q2 = getattr(surface_layer_diagnostics(state, namelist.grid), "q2", None)
    fields = (
        ("T2", "t2"),
        ("U10", "u10"),
        ("V10", "v10"),
        ("Q2", None),
        ("PSFC", "psfc"),
        ("SWDOWN", "swdown"),
        ("GLW", "glw"),
        ("PBLH", "pblh"),
        ("TSK", "tsk"),
    )
    out: dict[str, np.ndarray] = {}
    for wrf_name, attr in fields:
        value = q2 if wrf_name == "Q2" else (getattr(m9, attr, None) if attr else None)
        if value is not None:
            out[wrf_name] = np.asarray(jax.device_get(value))
    return out


def _field_tolerance(name: str) -> tuple[float, float]:
    spec = FIELD_TOLERANCES.get(name, {})
    return float(spec.get("atol", 0.0)), float(spec.get("rtol", 0.0))


def _compare_arrays(name: str, old, new) -> dict[str, Any]:
    a = np.asarray(old)
    b = np.asarray(new)
    record: dict[str, Any] = {
        "name": name,
        "old_present": old is not None,
        "new_present": new is not None,
        "shape_equal": bool(a.shape == b.shape),
        "dtype_old": str(a.dtype),
        "dtype_new": str(b.dtype),
    }
    if a.shape != b.shape:
        record.update({"status": "FAIL_SHAPE", "max_abs": None, "max_rel": None})
        return record
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number):
        equal = bool(np.array_equal(a, b))
        record.update(
            {
                "max_abs": 0.0 if equal else None,
                "max_rel": 0.0 if equal else None,
                "atol": 0.0,
                "rtol": 0.0,
                "bit_identical": equal,
                "status": "PASS" if equal else "FAIL_VALUE",
            }
        )
        return record
    diff = b.astype(np.float64) - a.astype(np.float64)
    max_abs = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
    denom = np.maximum(np.abs(a.astype(np.float64)), 1.0e-30)
    max_rel = float(np.nanmax(np.abs(diff) / denom)) if diff.size else 0.0
    atol, rtol = _field_tolerance(name)
    limit = atol + rtol * float(np.nanmax(np.abs(a.astype(np.float64)))) if a.size else atol
    record.update(
        {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "atol": atol,
            "rtol": rtol,
            "bit_identical": bool(np.array_equal(a, b, equal_nan=True)),
            "status": "PASS" if max_abs <= limit else "FAIL_VALUE",
        }
    )
    return record


def _compare_field_maps(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for name in sorted(set(old) | set(new)):
        if name not in old or name not in new:
            records.append(
                {
                    "name": name,
                    "old_present": name in old,
                    "new_present": name in new,
                    "status": "FAIL_MISSING",
                }
            )
            continue
        records.append(_compare_arrays(name, old[name], new[name]))
    status = "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL"
    return status, records


def _read_netcdf_fields(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        return {name: np.asarray(var[:]) for name, var in ds.variables.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/v0100_wave_b3_daily_wrapper"))
    parser.add_argument("--gains-json", type=Path, default=Path("proofs/v0100/wave_b3_daily_wrapper_gains.json"))
    parser.add_argument("--parity-json", type=Path, default=Path("proofs/v0100/wave_b3_output_parity.json"))
    args = parser.parse_args()

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Wave-B3 daily-wrapper gate requires a visible JAX GPU")

    cfg = DailyPipelineConfig(
        hours=1,
        dt_s=10.0,
        acoustic_substeps=10,
        run_id=L2_RUN_ID,
        run_root=paths.wrf_l2_root(),
        domain="d02",
        radiation_cadence_steps=180,
    )
    case, run_dir = _build_real_case(cfg)
    namelist = replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=False,
        radiation_cadence_steps=180,
        time_utc=case.run_start,
    )
    state0 = _enforce_operational_precision(case.state, force_fp64=bool(namelist.force_fp64))
    static_cache = build_wrfout_static_field_cache(state0, case.grid, namelist)

    # Compile/warm once, then measure a second equal forecast pass for the daily-wall denominator.
    warm_state = run_forecast_operational_segmented(state0, namelist, 1.0, segment_steps=180)
    jax.block_until_ready(warm_state.theta)
    t0 = time.perf_counter()
    state = run_forecast_operational_segmented(state0, namelist, 1.0, segment_steps=180)
    jax.block_until_ready(state.theta)
    forecast_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    finite = finite_summary(state)
    finite_s = time.perf_counter() - t0

    lead_seconds = 3600.0
    valid_time = case.run_start + timedelta(hours=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_path = args.output_dir / (_wrfout_name(valid_time, "d02") + ".old")
    new_path = args.output_dir / (_wrfout_name(valid_time, "d02") + ".b3")

    old_diags, old_diag_t = _time_repeated(
        "old_m9_diagnostics_plus_q2_recompute",
        lambda: _old_surface_diagnostics(state, namelist, lead_seconds),
        args.reps,
    )
    new_diags, new_diag_t = _time_repeated(
        "b3_m9_diagnostics_reuse_q2",
        lambda: _surface_diagnostics_for_output(
            state, namelist, case.run_start, lead_seconds=lead_seconds
        ),
        args.reps,
    )

    old_prepared, old_pack_t = _time_repeated(
        "old_prepare_wrfout_payload",
        lambda: prepare_wrfout_payload(
            state,
            case.grid,
            namelist,
            old_path,
            valid_time=valid_time,
            lead_hours=1.0,
            run_start=case.run_start,
            diagnostics=old_diags,
        ),
        args.reps,
    )
    new_prepared, new_pack_t = _time_repeated(
        "b3_prepare_wrfout_payload_static_cache",
        lambda: prepare_wrfout_payload(
            state,
            case.grid,
            namelist,
            new_path,
            valid_time=valid_time,
            lead_hours=1.0,
            run_start=case.run_start,
            diagnostics=new_diags,
            static_cache=static_cache,
        ),
        args.reps,
    )

    t0 = time.perf_counter()
    write_prepared_wrfout(old_prepared)
    old_write_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    write_prepared_wrfout(new_prepared)
    new_write_s = time.perf_counter() - t0

    prepared_status, prepared_records = _compare_field_maps(old_prepared.fields, new_prepared.fields)
    netcdf_status, netcdf_records = _compare_field_maps(_read_netcdf_fields(old_path), _read_netcdf_fields(new_path))

    old_wrapper_s = finite_s + old_diag_t["min_s"] + old_pack_t["min_s"] + old_write_s
    new_wrapper_s = finite_s + new_diag_t["min_s"] + new_pack_t["min_s"] + new_write_s
    old_daily_hour_s = forecast_s + old_wrapper_s
    new_daily_hour_s = forecast_s + new_wrapper_s
    diag_saved_s = old_diag_t["min_s"] - new_diag_t["min_s"]
    pack_saved_s = old_pack_t["min_s"] - new_pack_t["min_s"]
    total_saved_s = old_daily_hour_s - new_daily_hour_s

    gains = {
        "schema": "V0100WaveB3DailyWrapperGains",
        "schema_version": 1,
        "status": "PASS" if prepared_status == "PASS" and netcdf_status == "PASS" else "FAIL",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "timing_protocol": "segmented 1h L2 d02; one compile/warm forecast, then timed equal forecast; diagnostics/pack min over warmed reps",
        "forecast_s": forecast_s,
        "finite_summary_s": finite_s,
        "finite_summary_all_finite": bool(finite["all_finite"]),
        "old_m9_diagnostics_s": old_diag_t,
        "b3_m9_diagnostics_s": new_diag_t,
        "old_prepare_wrfout_payload_s": old_pack_t,
        "b3_prepare_wrfout_payload_s": new_pack_t,
        "old_netcdf_write_s": old_write_s,
        "b3_netcdf_write_s": new_write_s,
        "old_daily_hour_s": old_daily_hour_s,
        "b3_daily_hour_s": new_daily_hour_s,
        "m9_reuse_saved_s_per_output": diag_saved_s,
        "output_packer_static_cache_saved_s_per_output": pack_saved_s,
        "daily_wall_saved_s_per_hour": total_saved_s,
        "m9_reuse_gain_pct_daily_wall": 100.0 * diag_saved_s / old_daily_hour_s,
        "output_packer_gain_pct_daily_wall": 100.0 * pack_saved_s / old_daily_hour_s,
        "daily_wall_gain_pct": 100.0 * total_saved_s / old_daily_hour_s,
        "static_cache_fields": sorted(static_cache.fields),
        "field_tolerances": FIELD_TOLERANCES,
        "disposition": {
            "m9_q2_duplicate_recompute": "REMOVED",
            "writer_surface_fallback_for_hfx_lh_when_m9_supplies_fluxes": "REMOVED_WITH_TOLERANCE_GATE",
            "static_grid_cache": "REMOVED",
            "full_device_side_single_device_get_packer": "<1%-SKIPPED; warmed output payload path is below 1% of daily wall after low-risk removals",
        },
    }
    parity = {
        "schema": "V0100WaveB3OutputParity",
        "schema_version": 1,
        "status": "PASS" if prepared_status == "PASS" and netcdf_status == "PASS" else "FAIL",
        "prepared_field_status": prepared_status,
        "netcdf_field_status": netcdf_status,
        "prepared_field_records": prepared_records,
        "netcdf_field_records": netcdf_records,
        "old_wrfout": str(old_path),
        "b3_wrfout": str(new_path),
        "field_tolerances": FIELD_TOLERANCES,
    }
    args.gains_json.parent.mkdir(parents=True, exist_ok=True)
    args.parity_json.parent.mkdir(parents=True, exist_ok=True)
    args.gains_json.write_text(json.dumps(gains, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.parity_json.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gains": gains, "parity_status": parity["status"]}, indent=2, sort_keys=True), flush=True)
    return 0 if gains["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
