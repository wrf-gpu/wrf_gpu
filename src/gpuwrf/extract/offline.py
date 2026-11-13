"""Offline re-extract: EXTRACT_SPEC V1 from existing wrfout files (CPU-WRF or wrf_gpu RAW).

Same collector, same field code and same NetCDF/manifest as the in-run extract; the inputs are the
history variables as stored in the files. ZNT comes from the file's ZNT variable when present
(CPU-WRF history usually has none -> ZNT MISSING, the extract is then FAIL/never READY).

  python -m gpuwrf.extract.offline --config REQ.json --extract-dir DIR --wrfout-dir RUN --hours H \\
      [--domains d01 d02] [--interval-min 60] [--input-dir CASE] [--nwp-diagnostics 0]

The expected frame schedule never comes from the files found: it is lead 0 plus the history alarms of
the forecast length (same rule as the run), or an explicit step list. Each frame's NetCDF ``Times``
must equal its file name; two files of one step are refused. Missing frames -> PARTIAL.
SST provenance is declared, never inferred from values: ``sst_source="file"`` (CPU-WRF history, or a gpuwrf
run with an SST producer, i.e. sst_update=1) takes the file's SST; ``"none"`` (gpuwrf without a producer: its
history SST is the writer's TSK copy) gives NOT_AVAILABLE. Every scheduled/requested nest must be present,
each file's GRID_ID must be its nest, and fill values (>= 9.9e36) fail the frame QC.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from gpuwrf.extract.collector import DomainPlan, ExtractCollector, engine_identity, history_alarm_steps
from gpuwrf.extract.request import ExtractRequest, load_request
from gpuwrf.extract.v1_fields import SST_SOURCE

_NAME = re.compile(r"wrfout_(d[0-9]{2})_(\d{4}-\d{2}-\d{2})_(\d{2})[:_](\d{2})[:_](\d{2})\Z")


def frame_time(path: Path) -> tuple[str, datetime]:
    m = _NAME.match(Path(path).name)
    if not m:
        raise ValueError(f"not a wrfout frame name: {path}")
    stamp = datetime.strptime(f"{m.group(2)} {m.group(3)}:{m.group(4)}:{m.group(5)}", "%Y-%m-%d %H:%M:%S")
    return m.group(1), stamp.replace(tzinfo=timezone.utc)


def _times(ds, path) -> datetime:
    if "Times" not in ds.variables:
        raise ValueError(f"{path}: no Times variable")
    raw = np.asarray(ds["Times"][:])
    if raw.ndim != 2 or raw.shape[0] != 1:
        raise ValueError(f"{path}: expected exactly one Times record, got shape {raw.shape}")
    text = b"".join(bytes(c) if not isinstance(c, str) else c.encode() for c in raw[0].tolist()).decode().strip()
    return datetime.strptime(text, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def _attr_start(ds) -> datetime:
    text = str(ds.getncattr("SIMULATION_START_DATE")).strip()
    return datetime.strptime(text, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def extract_wrfouts(request: ExtractRequest, files: Mapping[str, list], *, interval_s: Mapping[str, float],
                    input_dir: Path | None = None, nwp_diagnostics: int = 0,
                    znt_source: Callable[[str, int, Path], np.ndarray | None] | None = None,
                    forecast_s: float | None = None, expected_steps: Mapping[str, tuple] | None = None,
                    cu_physics: Mapping[str, int] | None = None, sst_source: str | None = None) -> dict:
    """``forecast_s`` (run length) or ``expected_steps`` (per domain, explicit) fixes the schedule;
    ``sst_source`` ("file" | "none") declares the SST provenance."""
    import netCDF4

    if (forecast_s is None) == (expected_steps is None):
        raise ValueError("give exactly one of forecast_s (run length) or expected_steps (explicit schedule)")
    if sst_source not in ("file", "none"):
        raise ValueError("sst_source must be 'file' (CPU-WRF or a run with an SST producer) or 'none'")
    # Every nest that is requested, explicitly scheduled, or (forecast mode) given a history interval is scheduled
    # and must have frames: a forecast-mode run must not publish READY with a scheduled nest missing.
    required = set(request.domains or ()) | set(expected_steps or ()) | set(interval_s)
    absent = sorted(required - {n for n, p in files.items() if p})
    if absent:
        raise ValueError(f"scheduled/requested nests without wrfout frames: {absent}")
    unscheduled = sorted(set(files) - set(interval_s))
    if unscheduled:
        raise ValueError(f"wrfout nests without a history interval: {unscheduled}")
    if expected_steps is not None and set(expected_steps) != set(files):
        raise ValueError(f"expected_steps nests {sorted(expected_steps)} differ from the wrfout nests {sorted(files)}")
    plans, frames, run_start = {}, {}, None
    for name, paths in sorted(files.items()):
        paths = sorted(Path(p) for p in paths)
        if not paths:
            raise ValueError(f"{name}: no wrfout frames")
        with netCDF4.Dataset(paths[0]) as ds:
            start = _attr_start(ds)
            dt = Fraction(str(float(ds.getncattr("DT"))))
            dx = float(ds.getncattr("DX"))
            mp = int(ds.getncattr("MP_PHYSICS"))
            pbl = int(ds.getncattr("BL_PBL_PHYSICS"))
        if run_start is None:
            run_start = start
        elif start != run_start:
            raise ValueError("domains disagree on SIMULATION_START_DATE")
        steps = []
        for p in paths:
            dom, t = frame_time(p)
            if dom != name:
                raise ValueError(f"{p} is not a {name} frame")
            with netCDF4.Dataset(p) as ds:
                inner = _times(ds, p)
                if "GRID_ID" not in ds.ncattrs() or int(ds.getncattr("GRID_ID")) != int(name[1:]):
                    raise ValueError(f"{p}: GRID_ID {ds.getncattr('GRID_ID') if 'GRID_ID' in ds.ncattrs() else None} "
                                     f"is not nest {name}")
                if _attr_start(ds) != start:
                    raise ValueError(f"{p}: SIMULATION_START_DATE differs within {name}")
            if inner != t:
                raise ValueError(f"{p}: NetCDF Times {inner:%Y-%m-%d_%H:%M:%S} differs from the file name")
            lead = Fraction(int((t - start).total_seconds()))
            if (lead / dt).denominator != 1:
                raise ValueError(f"{p}: lead {lead} s is not a multiple of dt {dt}")
            steps.append(int(lead / dt))
        if len(set(steps)) != len(steps):
            raise ValueError(f"{name}: two files hold the same history step {sorted(steps)}")
        if expected_steps is not None:
            schedule = tuple(sorted({int(x) for x in expected_steps[name]}))
        else:
            total = Fraction(str(float(forecast_s))) / dt
            if total.denominator != 1:
                raise ValueError(f"{name}: forecast length is not a multiple of dt {dt}")
            schedule = (0, *history_alarm_steps(int(total), float(dt), float(interval_s[name])))
        plans[name] = DomainPlan(name=name, dt_s=float(dt), interval_s=float(interval_s[name]),
                                 expected_steps=schedule, bl_pbl_physics=pbl, mp_physics=mp,
                                 nwp_diagnostics=int(nwp_diagnostics), dx_m=dx,
                                 cu_physics=None if cu_physics is None else cu_physics.get(name))
        frames[name] = sorted(zip(steps, paths))
    collector = ExtractCollector(request, run_start=run_start, plans=plans,
                                 input_dir=Path(input_dir) if input_dir is not None else Path(files[min(files)][0]).parent)
    try:
        for name, items in frames.items():
            for step, path in items:
                with netCDF4.Dataset(path) as ds:
                    ds.set_auto_maskandscale(False)
                    fields = {k: np.array(v[:]) for k, v in ds.variables.items() if k != "Times"}
                    dims = {k: (None if k == "Time" else len(d)) for k, d in ds.dimensions.items()}
                znt = znt_source(name, step, path) if znt_source is not None else fields.get("ZNT")
                if znt is not None:
                    znt = np.asarray(znt, np.float32).reshape(fields["HGT"].shape[-2:])
                collector.submit_host(name, step, fields, dims, znt,
                                      unobserved=("SST",) if sst_source == "none" else (),
                                      provenance={"SST": SST_SOURCE}
                                      if sst_source == "file" else None)
        return collector.finalize(engine={**engine_identity(), "mode": "offline re-extract from wrfout files",
                                          "wrfout_files": {n: [str(p) for _, p in v] for n, v in frames.items()}})
    finally:
        collector.close()


def namelist_intervals(input_dir: Path) -> dict[str, float]:
    """History interval (s) of every nest d01..d<max_dom> from the case namelist.input: the independent nest set of
    the schedule (never the wrfout files found). Only the plain minute key history_interval is supported."""
    from gpuwrf.io.namelist_parser import parse_namelist

    nml = parse_namelist(Path(input_dir) / "namelist.input")
    tc, dom = nml.get("time_control", {}), nml.get("domains", {})
    other = sorted(k for k in tc if k.startswith(("history_interval_", "history_begin")))
    if other:
        raise ValueError(f"namelist history keys not supported by the offline schedule: {other}")
    max_dom = dom.get("max_dom")
    max_dom = int(max_dom[0] if isinstance(max_dom, list) else max_dom) if max_dom is not None else None
    hist = tc.get("history_interval")
    hist = hist if isinstance(hist, list) else [hist]
    if not max_dom or len(hist) < max_dom or any(h is None for h in hist[:max_dom]):
        raise ValueError(f"namelist max_dom {max_dom} / history_interval {hist}: cannot fix every nest's interval")
    return {f"d{i:02d}": float(hist[i - 1]) * 60.0 for i in range(1, max_dom + 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--wrfout-dir", type=Path, required=True)
    ap.add_argument("--domains", nargs="*", default=None,
                    help="restrict the schedule to these nests (default: every nest of the namelist)")
    ap.add_argument("--interval-min", type=float, default=None,
                    help="only without a case namelist: history interval of every --domains nest")
    ap.add_argument("--hours", type=float, required=True, help="forecast length of the run (fixes the schedule)")
    ap.add_argument("--sst-source", choices=("file", "none"), required=True,
                    help="file: the wrfout SST is real (CPU-WRF, or sst_update=1); none: no SST producer (gpuwrf "
                         "history SST is then a TSK copy) -> NOT_AVAILABLE")
    ap.add_argument("--input-dir", type=Path, default=None, help="case dir; its namelist.input fixes nests + intervals")
    ap.add_argument("--nwp-diagnostics", type=int, default=0)
    args = ap.parse_args(argv)
    if args.input_dir is not None and (args.input_dir / "namelist.input").is_file():
        if args.interval_min is not None:
            ap.error("--interval-min conflicts with the case namelist (it fixes every nest's interval)")
        intervals = namelist_intervals(args.input_dir)
    elif args.domains and args.interval_min is not None:
        intervals = {d: args.interval_min * 60.0 for d in args.domains}
    else:
        ap.error("the nest schedule needs --input-dir with namelist.input, or explicit --domains + --interval-min")
    if args.domains:
        unknown = sorted(set(args.domains) - set(intervals))
        if unknown:
            ap.error(f"--domains {unknown} not in the namelist nests {sorted(intervals)}")
        intervals = {d: intervals[d] for d in args.domains}
    request = load_request("v1", args.config, args.extract_dir)
    files: dict[str, list] = {}
    for p in sorted(args.wrfout_dir.glob("wrfout_d[0-9][0-9]_*")):
        files.setdefault(frame_time(p)[0], []).append(p)
    if args.domains:
        files = {d: v for d, v in files.items() if d in intervals}
    report = extract_wrfouts(request, files, interval_s=intervals,
                             input_dir=args.input_dir, nwp_diagnostics=args.nwp_diagnostics,
                             forecast_s=args.hours * 3600.0, sst_source=args.sst_source)
    print(json.dumps(report))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
