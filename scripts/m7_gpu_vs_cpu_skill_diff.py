#!/usr/bin/env python
"""Compare M7 GPU and Gen2 CPU d02 station skill against AEMET observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.io.data_inventory import parse_wrfout_valid_time
from gpuwrf.paths import reference_path
from gpuwrf.validation.forecast_vs_obs import (
    DEFAULT_AEMET_ROOT,
    compute_station_scores,
    interpolate_to_stations,
    load_aemet_observations,
    write_json,
)


SPRINT_DIR = ROOT / "proofs" / "generated" / "2026-05-27-m7-honest-speedup-skill-diff"
DEFAULT_GPU_ROOT = Path("/tmp/m7_pipeline_runs/20260521")
DEFAULT_CPU_RUN_GLOB = str(reference_path("runs", "wrf_l3", "20260521_18z_l3_24h_*")) + "/"
DEFAULT_OUTPUT = SPRINT_DIR / "gpu_vs_cpu_skill_diff.json"
DEFAULT_VARIABLES = ("T2", "U10", "V10")
TOLERANCE_FRACTION = 0.20


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _station_metadata(observations: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in ("station_id", "lat", "lon", "elev_m") if column in observations.columns]
    if not keep:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "elev_m"])
    return observations[keep].dropna(subset=["station_id", "lat", "lon"]).drop_duplicates(subset=["station_id"])


def discover_wrfouts(root: Path) -> list[Path]:
    files = sorted(path for path in root.glob("wrfout_d02_*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no wrfout_d02_* files found under {root}")
    return files


def choose_cpu_run(run_glob: str) -> Path:
    candidates = sorted(Path(path) for path in glob.glob(run_glob) if Path(path).is_dir())
    if not candidates:
        raise FileNotFoundError(f"no CPU run directories matched: {run_glob}")
    with_counts = [(len(discover_wrfouts(path)), path) for path in candidates]
    return max(with_counts, key=lambda item: (item[0], item[1].name))[1]


def _time_map(files: Sequence[Path]) -> dict[pd.Timestamp, Path]:
    mapped: dict[pd.Timestamp, Path] = {}
    for path in files:
        timestamp = pd.Timestamp(parse_wrfout_valid_time(path))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        mapped[timestamp] = path
    return mapped


def _source_forecasts(
    files_by_time: Mapping[pd.Timestamp, Path],
    times: Sequence[pd.Timestamp],
    station_metadata: pd.DataFrame,
    *,
    variables: Sequence[str],
) -> pd.DataFrame:
    frames = [
        interpolate_to_stations(files_by_time[time], station_metadata, variables=variables, valid_time=time.to_pydatetime())
        for time in times
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _prepare_score_table(df: pd.DataFrame, variables: Sequence[str], suffix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["station_id", "time", *[f"{variable}{suffix}" for variable in variables]])
    table = df.copy()
    table["station_id"] = table["station_id"].astype(str)
    table["time"] = pd.to_datetime(table["time"], utc=True).dt.floor("s")
    keep = ["station_id", "time", *[variable for variable in variables if variable in table.columns]]
    return table[keep].rename(columns={variable: f"{variable}{suffix}" for variable in variables if variable in table.columns})


def compute_per_station_scores(
    forecast_at_stations: pd.DataFrame,
    observations_at_stations: pd.DataFrame,
    *,
    variables: Sequence[str],
) -> list[dict[str, Any]]:
    forecast = _prepare_score_table(forecast_at_stations, variables, "_forecast")
    observations = _prepare_score_table(observations_at_stations, variables, "_obs")
    joined = forecast.merge(observations, on=["station_id", "time"], how="inner")
    rows: list[dict[str, Any]] = []
    for station_id, station_table in joined.groupby("station_id", sort=True):
        for variable in variables:
            f_col = f"{variable}_forecast"
            o_col = f"{variable}_obs"
            if f_col not in station_table.columns or o_col not in station_table.columns:
                rows.append({"station_id": station_id, "variable": variable, "status": "MISSING_VARIABLE", "sample_count": 0})
                continue
            pair = station_table[[f_col, o_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if pair.empty:
                rows.append({"station_id": station_id, "variable": variable, "status": "NO_VALID_PAIRS", "sample_count": 0})
                continue
            diff = pair[f_col].to_numpy(dtype=np.float64) - pair[o_col].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "station_id": station_id,
                    "variable": variable,
                    "status": "OK",
                    "sample_count": int(diff.size),
                    "bias": float(np.mean(diff)),
                    "rmse": float(np.sqrt(np.mean(diff * diff))),
                    "mae": float(np.mean(np.abs(diff))),
                }
            )
    return rows


def _relative_delta(candidate: float | None, reference: float | None, *, use_abs: bool = False) -> float | None:
    if candidate is None or reference is None:
        return None
    cand = abs(float(candidate)) if use_abs else float(candidate)
    ref = abs(float(reference)) if use_abs else float(reference)
    if abs(ref) < 1.0e-12:
        return 0.0 if abs(cand) < 1.0e-12 else None
    return float((cand - ref) / abs(ref))


def compare_aggregate_scores(
    gpu_scores: Mapping[str, Any],
    cpu_scores: Mapping[str, Any],
    *,
    variables: Sequence[str],
) -> dict[str, Any]:
    variable_results: dict[str, Any] = {}
    all_within = True
    materially_worse = False
    for variable in variables:
        gpu = gpu_scores.get(variable, {})
        cpu = cpu_scores.get(variable, {})
        metrics: dict[str, Any] = {}
        for metric in ("bias", "rmse", "mae"):
            use_abs = metric == "bias"
            delta = _relative_delta(gpu.get(metric), cpu.get(metric), use_abs=use_abs)
            within = bool(delta is not None and abs(delta) <= TOLERANCE_FRACTION)
            worse = bool(delta is not None and delta > TOLERANCE_FRACTION)
            metrics[metric] = {
                "gpu": gpu.get(metric),
                "cpu": cpu.get(metric),
                "relative_delta": delta,
                "within_20pct": within,
                "materially_worse": worse,
                "comparison_basis": "absolute bias magnitude" if metric == "bias" else "direct error magnitude",
            }
            all_within = all_within and within
            materially_worse = materially_worse or worse
        variable_results[variable] = {
            "gpu_sample_count": gpu.get("sample_count"),
            "cpu_sample_count": cpu.get("sample_count"),
            "metrics": metrics,
            "within_20pct_all_metrics": all(metric["within_20pct"] for metric in metrics.values()),
        }
    return {
        "tolerance_fraction": TOLERANCE_FRACTION,
        "tolerance_note": "BIAS is compared by absolute error magnitude; RMSE and MAE are positive direct magnitudes.",
        "variables": variable_results,
        "all_variables_within_20pct": all_within,
        "gpu_materially_worse_than_cpu": materially_worse,
    }


def _score_source(
    *,
    label: str,
    files_by_time: Mapping[pd.Timestamp, Path],
    common_times: Sequence[pd.Timestamp],
    observations: pd.DataFrame,
    station_metadata: pd.DataFrame,
    variables: Sequence[str],
) -> dict[str, Any]:
    forecasts = _source_forecasts(files_by_time, common_times, station_metadata, variables=variables)
    aggregate = compute_station_scores(forecasts, observations, variables=variables).to_dict()
    per_station = compute_per_station_scores(forecasts, observations, variables=variables)
    return {
        "label": label,
        "wrfout_file_count": int(len(common_times)),
        "wrfout_files": [str(files_by_time[time]) for time in common_times],
        "forecast_rows": int(len(forecasts)),
        "aggregate": aggregate,
        "per_station": per_station,
    }


def build_skill_diff_payload(
    *,
    gpu_root: Path,
    cpu_run: Path,
    aemet_root: Path,
    variables: Sequence[str],
) -> dict[str, Any]:
    gpu_files = discover_wrfouts(gpu_root)
    cpu_files = discover_wrfouts(cpu_run)
    gpu_by_time = _time_map(gpu_files)
    cpu_by_time = _time_map(cpu_files)
    common_times = sorted(set(gpu_by_time) & set(cpu_by_time))
    if not common_times:
        raise ValueError("GPU and CPU wrfout files have no common valid times")

    observations = load_aemet_observations(
        aemet_root,
        variables=variables,
        start_time=common_times[0].to_pydatetime(),
        end_time=common_times[-1].to_pydatetime(),
    )
    station_metadata = _station_metadata(observations)
    gpu = _score_source(
        label="gpu",
        files_by_time=gpu_by_time,
        common_times=common_times,
        observations=observations,
        station_metadata=station_metadata,
        variables=variables,
    )
    cpu = _score_source(
        label="cpu",
        files_by_time=cpu_by_time,
        common_times=common_times,
        observations=observations,
        station_metadata=station_metadata,
        variables=variables,
    )
    comparison = compare_aggregate_scores(gpu["aggregate"]["scores"], cpu["aggregate"]["scores"], variables=variables)
    verdict = "PASS_WITHIN_20PCT" if comparison["all_variables_within_20pct"] else "FAIL_SKILL_DIFF"
    return {
        "schema": "M7GpuVsCpuSkillDiff",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "variables": list(variables),
        "gpu_root": str(gpu_root),
        "cpu_run": str(cpu_run),
        "aemet_root": str(aemet_root),
        "common_valid_time_count": int(len(common_times)),
        "common_valid_time_range": {"start": common_times[0].isoformat(), "end": common_times[-1].isoformat()},
        "station_observation_rows": int(len(observations)),
        "station_count_scored": int(station_metadata["station_id"].nunique()) if not station_metadata.empty else 0,
        "sources": {"gpu": gpu, "cpu": cpu},
        "aggregate_comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--cpu-run", type=Path, default=None, help="CPU run directory. Defaults to the complete 20260521 run.")
    parser.add_argument("--cpu-run-glob", default=DEFAULT_CPU_RUN_GLOB)
    parser.add_argument("--aemet-root", type=Path, default=DEFAULT_AEMET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    args = parser.parse_args()

    cpu_run = args.cpu_run if args.cpu_run is not None else choose_cpu_run(args.cpu_run_glob)
    payload = build_skill_diff_payload(
        gpu_root=args.gpu_root,
        cpu_run=cpu_run,
        aemet_root=args.aemet_root,
        variables=tuple(args.variables),
    )
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["verdict"],
                "common_valid_time_count": payload["common_valid_time_count"],
                "station_count_scored": payload["station_count_scored"],
                "output": str(args.output),
            },
            sort_keys=True,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
