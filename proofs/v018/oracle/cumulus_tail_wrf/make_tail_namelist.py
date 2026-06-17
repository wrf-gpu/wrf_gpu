#!/usr/bin/env python3
"""Create a short real-data WRF namelist for CU7/10/11 oracle runs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


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
        print("usage: make_tail_namelist.py BASE_NAMELIST OUT_NAMELIST CU_CODE", file=sys.stderr)
        return 2
    base = Path(sys.argv[1])
    out = Path(sys.argv[2])
    cu_code = int(sys.argv[3])
    if cu_code not in {7, 10, 11}:
        raise SystemExit(f"unsupported CU code for tail oracle: {cu_code}")

    text = base.read_text()
    replacements = [
        ("time_control", "run_days", "0,"),
        ("time_control", "run_hours", "6,"),
        ("time_control", "run_minutes", "0,"),
        ("time_control", "run_seconds", "0,"),
        ("time_control", "end_year", "2026,"),
        ("time_control", "end_month", "4,"),
        ("time_control", "end_day", "29,"),
        ("time_control", "end_hour", "0,"),
        ("time_control", "end_minute", "0,"),
        ("time_control", "end_second", "0,"),
        ("time_control", "history_interval", "360,"),
        ("time_control", "frames_per_outfile", "10,"),
        ("time_control", "iofields_filename", '"cu_tail_iofields.txt",'),
        ("time_control", "ignore_iofields_warning", ".true.,"),
        ("physics", "cu_physics", f"{cu_code},"),
        ("physics", "cudt", "1,"),
    ]
    for section, key, value in replacements:
        text = set_key(section, key, value, text)
    if cu_code == 7:
        text = set_key("physics", "bl_pbl_physics", "2,", text)
        text = set_key("physics", "sf_sfclay_physics", "2,", text)
    out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
