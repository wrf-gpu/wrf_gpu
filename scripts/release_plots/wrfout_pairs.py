"""Pair CPU-WRF and GPU wrfout files of one domain by valid time (rounded to the hour; d01 frames carry +18 s)."""
from __future__ import annotations

import datetime as dt
import pathlib
import re

_RX = re.compile(r"wrfout_(d\d\d)_(\d{4}-\d\d-\d\d)_(\d\d):(\d\d):(\d\d)$")


def _valid(name: str):
    m = _RX.search(name)
    if not m:
        return None, None
    dom, day, hh, mm, ss = m.groups()
    t = dt.datetime.fromisoformat(f"{day}T{hh}:{mm}:{ss}")
    t = (t + dt.timedelta(minutes=30)).replace(minute=0, second=0)
    return dom, t


def frames(directory: pathlib.Path, domain: str) -> dict:
    out = {}
    for p in pathlib.Path(directory).iterdir():
        dom, t = _valid(p.name)
        if dom == domain:
            out[t] = p
    return out


def pairs(cpu_dir, gpu_dir, domain: str):
    """-> sorted list of (valid_time, cpu_path, gpu_path) present in both dirs."""
    c, g = frames(cpu_dir, domain), frames(gpu_dir, domain)
    return [(t, c[t], g[t]) for t in sorted(set(c) & set(g))]
