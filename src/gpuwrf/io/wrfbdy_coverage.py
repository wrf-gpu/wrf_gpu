"""Fail-closed lateral-boundary coverage of the root ``wrfbdy`` (host only, no JAX).

The standalone root's ``*_bdy`` leaves hold one level per wrfbdy record plus the
last record's tendency endpoint (integration/d02_replay.load_wrfbdy_boundary_leaves),
so they cover ``records x interval_seconds`` from the first record;
``interpolate_boundary_leaf`` clips its time index beyond that and would silently
hold the last boundary values. WRF instead stops when the run passes the last
record's next-boundary time (``md___nextbdytime``). A native run (single-domain or
nested; children are forced live by the parent) whose end lies past the coverage is
refused before any model work. A missing wrfbdy is left to the loader's own refusal.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

_NEXT_BDY_TIME = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"


class BoundaryCoverageError(ValueError):
    """The requested forecast end lies past the lateral boundary data."""


def _char_times(dataset, name: str) -> list[str]:
    if name not in dataset.variables:
        return []
    return [b"".join(np.asarray(row).tolist()).decode("ascii").strip() for row in dataset.variables[name][:]]


def _wrf_time(label: str) -> datetime:
    return datetime.strptime(label, "%Y-%m-%d_%H:%M:%S")


def _interval_seconds(input_dir: Path) -> float:
    """Same source and default as the wrfbdy leaf loader (namelist, else 6 h)."""
    raw = parse_namelist(input_dir / "namelist.input").get("time_control", {}).get("interval_seconds", 21600)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else 21600
    return float(raw or 21600)


def wrfbdy_coverage(input_dir: str | Path, domain: str = "d01") -> dict[str, Any] | None:
    """Seconds of lateral forcing available from the run start; ``None`` without a wrfbdy."""
    input_dir = Path(input_dir)
    path = input_dir / f"wrfbdy_{domain}"
    if not path.is_file():
        return None
    with Dataset(path, "r") as dataset:
        times = _char_times(dataset, "Times")
        next_times = _char_times(dataset, _NEXT_BDY_TIME)
    if not times:
        raise BoundaryCoverageError(f"{path.name} has no boundary records")
    interval_s = _interval_seconds(input_dir)
    model_s = len(times) * interval_s
    start = times[0]
    wrfinput = input_dir / f"wrfinput_{domain}"
    if wrfinput.is_file():
        with Dataset(wrfinput, "r") as dataset:
            start = (_char_times(dataset, "Times") or [start])[0]
    wrf_s = (_wrf_time(next_times[-1]) - _wrf_time(start)).total_seconds() if next_times else None
    return {
        "path": str(path), "records": len(times), "interval_seconds": interval_s,
        "run_start": start, "first_record": times[0], "last_boundary_time": next_times[-1] if next_times else None,
        "model_seconds": model_s, "wrf_seconds": wrf_s,
        "coverage_seconds": model_s if wrf_s is None else min(model_s, wrf_s),
    }


def require_wrfbdy_coverage(input_dir: str | Path, hours: float, domain: str = "d01") -> dict[str, Any] | None:
    """Raise :class:`BoundaryCoverageError` when ``hours`` runs past the boundary data."""
    info = wrfbdy_coverage(input_dir, domain)
    if info is None:
        return None
    if float(hours) * 3600.0 > info["coverage_seconds"] + 1.0e-6:
        last = f", last boundary time {info['last_boundary_time']}" if info["last_boundary_time"] else ""
        raise BoundaryCoverageError(
            f"forecast end {float(hours):g} h exceeds the lateral boundary coverage of "
            f"{Path(info['path']).name}: {info['coverage_seconds'] / 3600.0:g} h from {info['run_start']} "
            f"({info['records']} records x {info['interval_seconds']:g} s{last}). WRF stops at the end of its "
            "boundary data; provide a wrfbdy that covers the run or request fewer --hours"
        )
    return info
