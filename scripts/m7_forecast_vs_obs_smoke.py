#!/usr/bin/env python
"""Run the M7 forecast-vs-observation scaffold on one Gen2 CPU WRF day."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from gpuwrf.io.data_inventory import build_gen2_d02_inventory, parse_wrfout_valid_time
from gpuwrf.validation.forecast_vs_obs import (
    DEFAULT_AEMET_ROOT,
    DEFAULT_FSS_THRESHOLD_MM,
    DEFAULT_FSS_WINDOW_CELLS,
    DEFAULT_SCORE_VARIABLES,
    compute_precip_fss_for_wrfouts,
    compute_station_scores,
    interpolate_to_stations,
    inventory_aemet_observations,
    load_aemet_observations,
    write_json,
)


DEFAULT_WRF_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_SPRINT_DIR = Path(".agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold")


def _choose_run(root: Path, requested_run_id: str | None) -> dict[str, Any]:
    inventory = build_gen2_d02_inventory(root)
    complete = [run for run in inventory["runs"] if run["complete"]]
    if requested_run_id:
        matches = [run for run in inventory["runs"] if run["run_id"] == requested_run_id]
        if not matches:
            raise FileNotFoundError(f"requested run_id not found under {root}: {requested_run_id}")
        run = matches[0]
        if not run["complete"]:
            raise ValueError(f"requested run_id is not complete: {requested_run_id}")
        return run
    if not complete:
        raise FileNotFoundError(f"no complete Gen2 d02 WRF runs found under {root}")
    return sorted(complete, key=lambda run: run["valid_time_range"]["end"] or "")[-1]


def _station_metadata(observations: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in ("station_id", "lat", "lon", "elev_m") if column in observations.columns]
    if not keep:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "elev_m"])
    return observations[keep].dropna(subset=["station_id", "lat", "lon"]).drop_duplicates(subset=["station_id"])


def run_smoke(
    *,
    wrf_root: Path,
    aemet_root: Path,
    sprint_dir: Path,
    run_id: str | None,
    threshold_mm: float,
    window_size: int,
) -> dict[str, Any]:
    sprint_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_aemet_observations(aemet_root)
    inventory_path = sprint_dir / "aemet_observation_inventory.json"
    write_json(inventory_path, inventory)

    run = _choose_run(wrf_root, run_id)
    files = [Path(record["path"]) for record in run["files"]]
    if len(files) < 2:
        raise ValueError(f"chosen run has fewer than two wrfout files: {run['run_id']}")
    start_time = parse_wrfout_valid_time(files[0])
    end_time = parse_wrfout_valid_time(files[-1])

    variables = tuple(DEFAULT_SCORE_VARIABLES)
    observations = load_aemet_observations(aemet_root, variables=variables, start_time=start_time, end_time=end_time)
    station_metadata = _station_metadata(observations)
    forecast_frames = [
        interpolate_to_stations(path, station_metadata, variables=variables, valid_time=parse_wrfout_valid_time(path))
        for path in files
    ]
    forecasts = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    score_report = compute_station_scores(forecasts, observations, variables=variables)

    precip_obs = load_aemet_observations(aemet_root, variables=("PRECIP",), start_time=start_time, end_time=end_time)
    if precip_obs.empty:
        fss = {
            "schema": "M7PrecipFSS",
            "schema_version": 1,
            "status": "BLOCKED_PRECIP",
            "reason": "no AEMET precipitation observations overlapped the selected WRF run",
            "fss": None,
            "threshold_mm": float(threshold_mm),
            "window_size_cells": int(window_size),
        }
    else:
        precip_totals = (
            precip_obs.groupby("station_id", as_index=False)
            .agg({"PRECIP": "sum", "lat": "first", "lon": "first", "elev_m": "first"})
            .dropna(subset=["PRECIP", "lat", "lon"])
        )
        fss = compute_precip_fss_for_wrfouts(
            files[0],
            files[-1],
            precip_totals,
            threshold_mm=threshold_mm,
            window_size=window_size,
        )

    finite_station_scores = any(entry.get("sample_count", 0) > 0 for entry in score_report.scores.values())
    verdict = "SCAFFOLD_READY" if finite_station_scores and fss.get("fss") is not None else "BLOCKED_DATA"
    if fss.get("status") == "BLOCKED_PRECIP":
        verdict = "PARTIAL" if finite_station_scores else "BLOCKED_DATA"

    payload = {
        "schema": "M7ForecastVsObsSmokeRun",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": run["run_id"],
        "run_path": run["run_path"],
        "wrfout_file_count": int(len(files)),
        "valid_time_range": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        "aemet_inventory_path": str(inventory_path),
        "station_observation_rows": int(len(observations)),
        "station_count_scored": int(station_metadata["station_id"].nunique()) if not station_metadata.empty else 0,
        "forecast_rows": int(len(forecasts)),
        "score_report": score_report.to_dict(),
        "precip_fss": fss,
        "caveats": [
            "CPU Gen2 WRF is used only to validate scaffold execution; this is not a GPU forecast claim.",
            "Precipitation FSS uses station observations projected to nearest WRF grid cells.",
        ],
    }
    output_path = sprint_dir / "cpu_baseline_scaffold_run.json"
    write_json(output_path, payload)
    return {"verdict": verdict, "inventory_path": str(inventory_path), "run_path": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", type=Path, default=DEFAULT_WRF_ROOT)
    parser.add_argument("--aemet-root", type=Path, default=DEFAULT_AEMET_ROOT)
    parser.add_argument("--sprint-dir", type=Path, default=DEFAULT_SPRINT_DIR)
    parser.add_argument("--run-id", default=None, help="Optional complete Gen2 run_id to score")
    parser.add_argument("--threshold-mm", type=float, default=DEFAULT_FSS_THRESHOLD_MM)
    parser.add_argument("--window-size", type=int, default=DEFAULT_FSS_WINDOW_CELLS)
    args = parser.parse_args()
    result = run_smoke(
        wrf_root=args.wrf_root,
        aemet_root=args.aemet_root,
        sprint_dir=args.sprint_dir,
        run_id=args.run_id,
        threshold_mm=args.threshold_mm,
        window_size=args.window_size,
    )
    print(result)


if __name__ == "__main__":
    main()
