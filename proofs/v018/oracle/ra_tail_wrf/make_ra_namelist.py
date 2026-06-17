#!/usr/bin/env python3
"""Create short real-data WRF namelists for v0.18 RA tail oracle runs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


SCHEMES = {3, 5, 7, 99}


def set_key(section: str, key: str, value: str, text: str) -> str:
    pattern = re.compile(rf"(&{section}\b.*?/)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"missing namelist section &{section}")
    block = match.group(1)
    key_pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=).*$", re.MULTILINE)
    if key_pattern.search(block):
        block = key_pattern.sub(rf"\1 {value}", block)
    else:
        block = block.replace("\n/", f"\n {key:<36} = {value}\n/")
    return text[: match.start(1)] + block + text[match.end(1) :]


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: make_ra_namelist.py BASE_NAMELIST OUT_NAMELIST RA_CODE", file=sys.stderr)
        return 2
    base = Path(sys.argv[1])
    out = Path(sys.argv[2])
    code = int(sys.argv[3])
    if code not in SCHEMES:
        raise SystemExit(f"unsupported RA tail oracle code: {code}")

    text = base.read_text()
    # 18 h window 2026-04-28_18:00 -> 2026-04-29_12:00 (UTC). The wrfbdy covers
    # boundary updates at 18/00/06/12, so 18 h is the maximum re-runnable span.
    # The domain (lon ~-16, Canary/Atlantic) reaches solar noon ~13:00 UTC, so the
    # 09:00 + 12:00 UTC history frames carry strong daytime shortwave -- a 6 h
    # 18:00->00:00 run is night-only (single weak dusk SW frame), too thin to prove
    # the SW path of each scheme actually fired. 3-hourly history, single outfile.
    replacements = [
        ("time_control", "run_days", "0,"),
        ("time_control", "run_hours", "18,"),
        ("time_control", "run_minutes", "0,"),
        ("time_control", "run_seconds", "0,"),
        ("time_control", "end_year", "2026,"),
        ("time_control", "end_month", "4,"),
        ("time_control", "end_day", "29,"),
        ("time_control", "end_hour", "12,"),
        ("time_control", "end_minute", "0,"),
        ("time_control", "end_second", "0,"),
        ("time_control", "history_interval", "180,"),
        ("time_control", "frames_per_outfile", "100,"),
        ("time_control", "iofields_filename", '"ra_tail_iofields.txt",'),
        ("time_control", "ignore_iofields_warning", ".true.,"),
        ("physics", "ra_lw_physics", f"{code},"),
        ("physics", "ra_sw_physics", f"{code},"),
        ("physics", "radt", "10,"),
        ("physics", "topo_shading", "0,"),
        ("physics", "slope_rad", "0,"),
        ("physics", "aer_opt", "0,"),
        ("physics", "levsiz", "59,"),
        ("physics", "paerlev", "29,"),
        ("physics", "cam_abs_dim1", "4,"),
        ("physics", "cam_abs_dim2", "45,"),
    ]
    for section, key, value in replacements:
        text = set_key(section, key, value, text)
    out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
