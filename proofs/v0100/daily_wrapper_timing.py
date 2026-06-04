"""One-hour daily-wrapper timing breakdown for Phase 0.

This deliberately mirrors ``daily_pipeline._run_forecast_sequence`` but records
separate wall times for forecast, full-State finite-summary D2H, M9 diagnostics,
wrfout payload preparation/D2H, NetCDF write, and land refresh.
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

from gpuwrf.integration.daily_pipeline import (
    DailyPipelineConfig,
    _build_real_case,
    _refresh_hourly_land_state,
    _surface_diagnostics_for_output,
    _wrfout_name,
    finite_summary,
)
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.runtime.operational_mode import run_forecast_operational


def _time(label: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, float | str]]:
    t0 = time.perf_counter()
    result = fn()
    return result, {"label": label, "wall_s": time.perf_counter() - t0}


def _forecast_one_hour(state: Any, namelist: Any) -> Any:
    result = run_forecast_operational(state, namelist, 1.0)
    block_until_ready(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--hour", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/v0100_daily_wrapper_timing"))
    parser.add_argument("--out-json", type=Path, default=Path("proofs/v0100/daily_wrapper_timing.json"))
    args = parser.parse_args()

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("No JAX GPU backend visible; daily wrapper timing requires the production GPU forecast")

    config = DailyPipelineConfig(hours=1, domain=args.domain, output_dir=args.output_dir)
    case, run_dir = _build_real_case(config)
    state = case.state
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timings: list[dict[str, float | str]] = []
    state, rec = _time(
        "forecast_1h",
        lambda: _forecast_one_hour(state, case.namelist),
    )
    timings.append(rec)

    summary, rec = _time("finite_summary_full_state_d2h_after_forecast", lambda: finite_summary(state))
    timings.append(rec)

    lead_seconds = float(args.hour) * 3600.0
    diagnostics, rec = _time(
        "m9_diagnostics_for_output",
        lambda: _surface_diagnostics_for_output(state, case.namelist, case.run_start, lead_seconds=lead_seconds),
    )
    timings.append(rec)

    valid_time = case.run_start + timedelta(hours=args.hour)
    wrfout = args.output_dir / _wrfout_name(valid_time, args.domain)
    prepared, rec = _time(
        "output_pack_d2h_prepare_wrfout_payload",
        lambda: prepare_wrfout_payload(
            state,
            case.grid,
            case.namelist,
            wrfout,
            valid_time=valid_time,
            lead_hours=float(args.hour),
            run_start=case.run_start,
            diagnostics=diagnostics,
        ),
    )
    timings.append(rec)

    _, rec = _time("netcdf_write", lambda: write_prepared_wrfout(prepared))
    timings.append(rec)

    if bool(config.refresh_land_state_hourly):
        state, rec = _time(
            "land_refresh",
            lambda: _refresh_hourly_land_state(
                state,
                run_dir,
                args.domain,
                args.hour,
                use_noahmp=bool(config.use_noahmp),
            )[0],
        )
        timings.append(rec)
        _, rec = _time("finite_summary_full_state_d2h_after_land_refresh", lambda: finite_summary(state))
        timings.append(rec)

    total = sum(float(item["wall_s"]) for item in timings)
    host_labels = {
        "finite_summary_full_state_d2h_after_forecast",
        "finite_summary_full_state_d2h_after_land_refresh",
        "m9_diagnostics_for_output",
        "output_pack_d2h_prepare_wrfout_payload",
        "netcdf_write",
        "land_refresh",
    }
    host = sum(float(item["wall_s"]) for item in timings if item["label"] in host_labels)
    payload = {
        "schema": "V0100Phase0DailyWrapperTiming",
        "schema_version": 1,
        "status": "PASS",
        "domain": args.domain,
        "run_dir": str(run_dir),
        "output_dir": str(args.output_dir),
        "timings": timings,
        "total_wall_s": total,
        "host_wrapper_wall_s": host,
        "host_wrapper_share": host / total if total > 0 else None,
        "finite_summary_all_finite": bool(summary["all_finite"]),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
