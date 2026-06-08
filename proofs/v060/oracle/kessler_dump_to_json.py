#!/usr/bin/env python3
"""Parse the flat Kessler Fortran oracle dump into a structured JSON savepoint."""

from __future__ import annotations

import json
import re
import sys


SCALARS = {"CASE", "KX", "DT", "RAINNC", "RAINNCV", "FULL_WRF_EXE"}
CASE_LABELS = {
    1: "condensation",
    2: "autoconversion",
    3: "accretion",
    4: "evaporation",
    5: "sedimentation_fall",
}
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def _scalar(name: str, value: str):
    if name in {"CASE", "KX"}:
        return int(value)
    if name == "FULL_WRF_EXE":
        return value.strip().upper().startswith("T")
    return float(value)


def main(infile: str, outfile: str) -> None:
    scalars: dict[str, object] = {}
    cols: dict[str, dict[int, float]] = {}
    with open(infile, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = COL_RE.match(line)
            if m:
                name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
                cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if m and m.group(1) in SCALARS:
                scalars[m.group(1)] = _scalar(m.group(1), m.group(2))

    kx = int(scalars["KX"])
    case_id = int(scalars["CASE"])
    out = {
        "metadata": {
            "scheme": "Kessler warm rain (mp_physics=1)",
            "case_label": CASE_LABELS.get(case_id, "unknown"),
            "oracle_source": "$WRF_PRISTINE_ROOT/phys/module_mp_kessler.F",
            "full_wrf_exe": bool(scalars["FULL_WRF_EXE"]),
        },
        "scalars": scalars,
        "columns": {},
    }
    for name, values in cols.items():
        out["columns"][name] = [values[k] for k in range(1, kx + 1)]
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
