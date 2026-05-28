#!/usr/bin/env python
"""M13 radiation parity proof helpers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402

from gpuwrf.coupling.physics_couplers import rrtmg_radiation_diagnostics  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: E402
from gpuwrf.profiling.transfer_audit import visible_gpu_name  # noqa: E402
from scripts.m7_gpu_vs_cpu_skill_diff import build_skill_diff_payload  # noqa: E402


DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
DEFAULT_WRF_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3") / DEFAULT_RUN_ID
DEFAULT_GPU_ROOT = Path("/tmp/m13_radiation_diurnal_20260521")
DEFAULT_AEMET_ROOT = Path("/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations")
DEFAULT_PROOF_DIR = ROOT / "proofs/m13"
M9_DIVERGENCE = ROOT / "proofs/m9/divergence_map_v2.json"
M10_SKILL = ROOT / "proofs/m10/post_m10_skill_diff.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_valid_time(path: Path) -> datetime:
    return datetime.strptime(path.name[-19:], "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def _radiation_midpoint(valid_time: datetime) -> datetime:
    return valid_time - timedelta(minutes=15)


def _array(value, dtype=np.float32) -> np.ndarray:
    return np.asarray(np.ma.filled(value[:], np.nan), dtype=dtype)


def _time0(dataset: Dataset, name: str, dtype=np.float32) -> np.ndarray:
    data = _array(dataset.variables[name], dtype=dtype)
    return data[0] if data.ndim > 0 and data.shape[0] == 1 else data


def _minimal_state_from_wrfout(path: Path) -> SimpleNamespace:
    with Dataset(path, "r") as dataset:
        theta = _time0(dataset, "T") + 300.0
        pressure = _time0(dataset, "P") + _time0(dataset, "PB")
        geopotential = _time0(dataset, "PH", dtype=np.float64) + _time0(dataset, "PHB", dtype=np.float64)
        zeros = np.zeros_like(theta)
        return SimpleNamespace(
            theta=jnp.asarray(theta),
            p=jnp.asarray(pressure),
            ph=jnp.asarray(geopotential),
            qv=jnp.asarray(_time0(dataset, "QVAPOR")),
            qc=jnp.asarray(_time0(dataset, "QCLOUD")),
            qi=jnp.asarray(_time0(dataset, "QICE")),
            qs=jnp.asarray(zeros),
            qg=jnp.asarray(zeros),
            t_skin=jnp.asarray(_time0(dataset, "TSK")),
            lu_index=jnp.asarray(_time0(dataset, "LU_INDEX", dtype=np.int32), dtype=jnp.int32),
        )


def _wrfout_files(root: Path, domain: str, hours: int) -> list[Path]:
    return sorted(root.glob(f"wrfout_{domain}_*"))[: int(hours)]


def _rmse(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    if not bool(mask.any()):
        return float("nan"), 0
    delta = left[mask].astype(np.float64) - right[mask].astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta))), int(mask.sum())


def _field_stats(gpu: np.ndarray, wrf: np.ndarray) -> dict[str, Any]:
    rmse, count = _rmse(gpu, wrf)
    delta = gpu.astype(np.float64) - wrf.astype(np.float64)
    finite = np.isfinite(delta)
    return {
        "rmse": rmse,
        "finite_count": count,
        "max_abs_diff": float(np.nanmax(np.abs(delta[finite]))) if bool(finite.any()) else None,
        "gpu_min": float(np.nanmin(gpu)),
        "gpu_max": float(np.nanmax(gpu)),
        "gpu_mean": float(np.nanmean(gpu)),
        "wrf_min": float(np.nanmin(wrf)),
        "wrf_max": float(np.nanmax(wrf)),
        "wrf_mean": float(np.nanmean(wrf)),
    }


def write_hour1_parity(wrf_root: Path, proof_dir: Path, domain: str) -> dict[str, Any]:
    case = build_replay_case(wrf_root, domain=domain)
    valid_time = datetime(2026, 5, 21, 19, tzinfo=timezone.utc)
    diagnostics = rrtmg_radiation_diagnostics(case.state, case.grid, time_utc=_radiation_midpoint(valid_time))
    swdown = np.asarray(diagnostics.swdown)
    coszen = np.asarray(diagnostics.coszen)
    albedo = np.asarray(diagnostics.surface_albedo)
    emissivity = np.asarray(diagnostics.surface_emissivity)

    reference_path = wrf_root / f"wrfout_{domain}_2026-05-21_19:00:00"
    with Dataset(reference_path, "r") as dataset:
        wrf_swdown = _time0(dataset, "SWDOWN")
        wrf_coszen = _time0(dataset, "COSZEN")
        wrf_albedo = _time0(dataset, "ALBEDO")
        wrf_emiss = _time0(dataset, "EMISS")

    daytime = wrf_swdown > 50.0
    rel = np.abs(swdown[daytime] - wrf_swdown[daytime]) / np.maximum(np.abs(wrf_swdown[daytime]), 1.0e-6)
    pass_mask = rel <= 0.10
    payload = {
        "artifact_type": "m13_radiation_parity_hour_1",
        "status": "PASS" if bool(pass_mask.all()) else "FAIL",
        "valid_time_utc": valid_time.isoformat(),
        "radiation_time_utc": _radiation_midpoint(valid_time).isoformat(),
        "domain": domain,
        "source_run": str(wrf_root),
        "reference_wrfout": str(reference_path),
        "device": visible_gpu_name(),
        "method": "GPU RRTMG diagnostics from build_replay_case state, compared with WRF wrfout SWDOWN on WRF SWDOWN > 50 W m-2 cells.",
        "acceptance": {
            "per_cell_relative_tolerance": 0.10,
            "daytime_cell_count": int(daytime.sum()),
            "within_10pct_fraction": float(np.mean(pass_mask)) if pass_mask.size else 0.0,
            "violating_cell_count": int(pass_mask.size - int(pass_mask.sum())),
            "passed": bool(pass_mask.all()),
        },
        "metrics": {
            "SWDOWN": _field_stats(swdown, wrf_swdown),
            "COSZEN": _field_stats(coszen, wrf_coszen),
            "ALBEDO": _field_stats(albedo, wrf_albedo),
            "EMISS": _field_stats(emissivity, wrf_emiss),
            "relative_error": {
                "p50": float(np.quantile(rel, 0.50)) if rel.size else None,
                "p90": float(np.quantile(rel, 0.90)) if rel.size else None,
                "max": float(np.max(rel)) if rel.size else None,
            },
        },
    }
    _write_json(proof_dir / "radiation_parity_hour_1.json", payload)
    return payload


def augment_gpu_wrfouts(gpu_root: Path, wrf_root: Path, proof_dir: Path, domain: str, hours: int) -> dict[str, Any]:
    grid = Gen2Run(wrf_root).grid(domain).as_grid_spec()
    records: list[dict[str, Any]] = []
    for path in _wrfout_files(gpu_root, domain, hours):
        valid_time = _parse_valid_time(path)
        state = _minimal_state_from_wrfout(path)
        diagnostics = rrtmg_radiation_diagnostics(state, grid, time_utc=_radiation_midpoint(valid_time))
        swdown = np.asarray(diagnostics.swdown, dtype=np.float32)
        glw = np.asarray(diagnostics.glw, dtype=np.float32)
        with Dataset(path, "r+") as dataset:
            dataset.variables["SWDOWN"][0, :, :] = swdown
            dataset.variables["GLW"][0, :, :] = glw
            dataset.setncattr("GPUWRF_M13_RADIATION_DIAGNOSTICS", "SWDOWN/GLW recomputed by m13_radiation_proofs.py")
        records.append(
            {
                "path": str(path),
                "valid_time_utc": valid_time.isoformat(),
                "radiation_time_utc": _radiation_midpoint(valid_time).isoformat(),
                "swdown_mean": float(np.nanmean(swdown)),
                "swdown_max": float(np.nanmax(swdown)),
                "glw_mean": float(np.nanmean(glw)),
                "glw_max": float(np.nanmax(glw)),
            }
        )

    payload = {
        "artifact_type": "m13_augmented_wrfout_radiation_diagnostics",
        "status": "PASS" if len(records) == int(hours) else "FAIL",
        "gpu_root": str(gpu_root),
        "wrf_root": str(wrf_root),
        "domain": domain,
        "expected_hours": int(hours),
        "augmented_file_count": int(len(records)),
        "records": records,
    }
    _write_json(proof_dir / "radiation_diagnostic_augmentation.json", payload)
    return payload


def radiation_trace(gpu_root: Path, wrf_root: Path, proof_dir: Path, domain: str, hours: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = {"SWDOWN": {"sum_sq": 0.0, "count": 0}, "GLW": {"sum_sq": 0.0, "count": 0}}
    gpu_files = {path.name: path for path in _wrfout_files(gpu_root, domain, hours)}
    for name, gpu_path in sorted(gpu_files.items()):
        wrf_path = wrf_root / name
        if not wrf_path.is_file():
            continue
        fields: dict[str, Any] = {}
        with Dataset(gpu_path, "r") as gpu_ds, Dataset(wrf_path, "r") as wrf_ds:
            for field in ("SWDOWN", "GLW"):
                gpu = _time0(gpu_ds, field)
                wrf = _time0(wrf_ds, field)
                fields[field] = _field_stats(gpu, wrf)
                mask = np.isfinite(gpu) & np.isfinite(wrf)
                delta = gpu[mask].astype(np.float64) - wrf[mask].astype(np.float64)
                totals[field]["sum_sq"] += float(np.sum(delta * delta))
                totals[field]["count"] += int(mask.sum())
        rows.append({"valid_time_utc": _parse_valid_time(gpu_path).isoformat(), "gpu_path": str(gpu_path), "wrf_path": str(wrf_path), "fields": fields})

    summary = {
        field: {
            "rmse_over_all_hours": float(np.sqrt(item["sum_sq"] / item["count"])) if item["count"] else None,
            "finite_count_over_hours": int(item["count"]),
        }
        for field, item in totals.items()
    }
    payload = {
        "artifact_type": "m13_radiation_trace",
        "status": "PASS" if len(rows) == int(hours) else "FAIL",
        "gpu_root": str(gpu_root),
        "wrf_root": str(wrf_root),
        "domain": domain,
        "matched_hour_count": int(len(rows)),
        "hours": rows,
        "summary": summary,
    }
    _write_json(proof_dir / "radiation_trace_24h.json", payload)
    return payload


def write_post_skill_diff(
    gpu_root: Path,
    wrf_root: Path,
    aemet_root: Path,
    proof_dir: Path,
    domain: str,
    hours: int,
) -> dict[str, Any]:
    skill = build_skill_diff_payload(gpu_root=gpu_root, cpu_run=wrf_root, aemet_root=aemet_root, variables=("T2", "U10", "V10"))
    trace = radiation_trace(gpu_root, wrf_root, proof_dir, domain, hours)
    m9 = _read_json(M9_DIVERGENCE)
    m10 = _read_json(M10_SKILL)

    baseline_swd = float(m9["summary"]["operational_variable_24h_summary"]["SWDOWN"]["mean_rmse"])
    post_swd = float(trace["summary"]["SWDOWN"]["rmse_over_all_hours"])
    swd_reduction = 100.0 * (baseline_swd - post_swd) / baseline_swd
    baseline_t2 = float(m10["aggregate_comparison"]["variables"]["T2"]["metrics"]["rmse"]["gpu"])
    post_t2 = float(skill["aggregate_comparison"]["variables"]["T2"]["metrics"]["rmse"]["gpu"])
    acceptance = {
        "baseline_swd_rmse": baseline_swd,
        "post_m13_swd_rmse": post_swd,
        "swdown_rmse_reduction_pct": swd_reduction,
        "swdown_rmse_drop_ge_50pct": bool(swd_reduction >= 50.0),
        "baseline_t2_gpu_rmse": baseline_t2,
        "post_m13_t2_gpu_rmse": post_t2,
        "t2_non_regression_vs_m10": bool(post_t2 <= baseline_t2),
        "radiation_trace_path": str(proof_dir / "radiation_trace_24h.json"),
    }
    payload = {
        "schema": "M13PostSkillDiff",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if acceptance["swdown_rmse_drop_ge_50pct"] and acceptance["t2_non_regression_vs_m10"] else "FAIL",
        "gpu_root": str(gpu_root),
        "cpu_run": str(wrf_root),
        "aemet_root": str(aemet_root),
        "m13_acceptance": acceptance,
        "station_skill_diff": skill,
    }
    _write_json(proof_dir / "post_m13_skill_diff.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--wrf-root", type=Path, default=DEFAULT_WRF_ROOT)
    parser.add_argument("--aemet-root", type=Path, default=DEFAULT_AEMET_ROOT)
    parser.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--hour1", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--post-skill", action="store_true")
    args = parser.parse_args(argv)

    run_all = not (args.hour1 or args.augment or args.post_skill)
    results = {}
    if run_all or args.hour1:
        results["hour1"] = write_hour1_parity(args.wrf_root, args.proof_dir, args.domain)
    if run_all or args.augment:
        results["augment"] = augment_gpu_wrfouts(args.gpu_root, args.wrf_root, args.proof_dir, args.domain, args.hours)
    if run_all or args.post_skill:
        results["post_skill"] = write_post_skill_diff(
            args.gpu_root,
            args.wrf_root,
            args.aemet_root,
            args.proof_dir,
            args.domain,
            args.hours,
        )
    print(json.dumps(results, indent=2, sort_keys=True, default=_json_default))
    failed = [name for name, payload in results.items() if payload.get("status") != "PASS"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
