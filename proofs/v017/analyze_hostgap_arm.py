#!/usr/bin/env python3
"""Analyze one all-7 host-gap A/B arm: GPU busy% + warm min/forecast-hour.

Reads <RR>/gpu_samples.csv (timestamp,util_gpu_pct,mem_used_mib @ 0.5 s) and
<RR>/result.txt (start/end epoch + d01 hourly wrfout mtimes), aligns the sampler
local-time stamps to epoch, then reports utilization + per-hour wallclock over
the FULL run and over the WARM window (after forecast hour 1, which carries the
cold-compile tax).  Pure CPU; pin to cores 0-3.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def _parse_samples(path: Path):
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        if i == 0 or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        ts, util, mem = parts[0], parts[1], parts[2]
        try:
            # nvidia-smi local-time stamp, e.g. "2026/06/15 18:25:13.903"
            dt = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f")
            epoch = dt.timestamp()  # local TZ -> absolute epoch
            rows.append((epoch, float(util), float(mem)))
        except ValueError:
            continue
    rows.sort()
    return rows


def _parse_result(path: Path):
    info = {}
    hourly = []  # (epoch, name)
    in_mtimes = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("--- d01 hourly mtimes"):
            in_mtimes = True
            continue
        if line.startswith("---"):
            in_mtimes = False
            continue
        if in_mtimes:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                hourly.append((int(parts[0]), parts[1]))
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            info[k.strip()] = v.strip()
    hourly.sort()
    return info, hourly


def _window_stats(rows, t0, t1):
    sub = [r for r in rows if t0 <= r[0] <= t1]
    if not sub:
        return None
    utils = [r[1] for r in sub]
    mems = [r[2] for r in sub]
    n = len(utils)
    utils_sorted = sorted(utils)
    median = utils_sorted[n // 2]
    return {
        "n_samples": n,
        "span_s": round(t1 - t0, 1),
        "mean_util": round(sum(utils) / n, 1),
        "median_util": round(median, 1),
        "busy_ge50_pct": round(100.0 * sum(1 for u in utils if u >= 50) / n, 1),
        "busy_ge80_pct": round(100.0 * sum(1 for u in utils if u >= 80) / n, 1),
        "idle_lt10_pct": round(100.0 * sum(1 for u in utils if u < 10) / n, 1),
        "idle_lt20_pct": round(100.0 * sum(1 for u in utils if u < 20) / n, 1),
        "peak_mem_mib": int(max(mems)),
    }


def main():
    rr = Path(sys.argv[1])
    rows = _parse_samples(rr / "gpu_samples.csv")
    info, hourly = _parse_result(rr / "result.txt")
    start_epoch = float(info.get("start_epoch", rows[0][0] if rows else 0))
    end_epoch = float(info.get("end_epoch", rows[-1][0] if rows else 0))

    out = {
        "rr": str(rr),
        "sync_mode": info.get("sync_mode"),
        "hours": info.get("hours"),
        "wall_s": info.get("wall_s"),
        "compile_alarms": info.get("advance_chunk_compile_alarms"),
        "peak_vram_mib_result": info.get("peak_vram_mib"),
        "n_d01_hours": len(hourly),
    }

    # Per-forecast-hour wallclock from consecutive d01 wrfout mtimes.
    hour_deltas = []
    for i in range(1, len(hourly)):
        dt = hourly[i][0] - hourly[i - 1][0]
        hour_deltas.append({"to": hourly[i][1], "delta_s": dt, "min": round(dt / 60.0, 2)})
    out["per_hour_wallclock"] = hour_deltas

    # Full-run util.
    out["full_run_util"] = _window_stats(rows, start_epoch, end_epoch)

    # WARM window: from end of forecast hour 1 (first d01 mtime) to last d01 mtime.
    if len(hourly) >= 2:
        warm_t0 = hourly[0][0]
        warm_t1 = hourly[-1][0]
        out["warm_window_util"] = _window_stats(rows, warm_t0, warm_t1)
        warm_hours = len(hourly) - 1
        out["warm_min_per_forecast_hour"] = round((warm_t1 - warm_t0) / 60.0 / warm_hours, 2)
        out["warm_hours_counted"] = warm_hours
    else:
        out["warm_window_util"] = None
        out["warm_min_per_forecast_hour"] = None

    cpu_min_per_hr = 14.9
    wm = out.get("warm_min_per_forecast_hour")
    if wm:
        out["cpu_min_per_forecast_hour"] = cpu_min_per_hr
        out["wallclock_speedup_vs_cpu"] = round(cpu_min_per_hr / wm, 3)
        out["gpu_win_vs_cpu"] = wm < cpu_min_per_hr

    print(json.dumps(out, indent=2))
    (rr / "analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
