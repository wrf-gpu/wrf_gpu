#!/usr/bin/env python3
"""Parse the CAM-UW Fortran oracle flat dump into JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INT_SCALARS = {"CASE", "KX", "FULL_WRF_EXE", "KPBL"}
STR_SCALARS = {"REGIME"}
INTERFACE_COLS = {"P8W", "Z_AT_W", "TKE_PBL", "KVM3D", "KVH3D", "SMAW3D", "TURBTYPE3D"}

COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def parse(infile: Path) -> dict:
    scalars: dict[str, int | float | str] = {}
    cols: dict[str, dict[int, float]] = {}
    for line in infile.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("WRF_FATAL", "FATAL", "STOP")):
            raise RuntimeError(line)
        m = COL_RE.match(line)
        if m:
            name, idx, value = m.group(1), int(m.group(2)), float(m.group(3))
            cols.setdefault(name, {})[idx] = value
            continue
        m = SCAL_RE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2).strip()
        if name in INT_SCALARS:
            scalars[name] = int(value)
        elif name in STR_SCALARS:
            scalars[name] = value
        else:
            scalars[name] = float(value)

    kx = int(scalars["KX"])
    out = {"scalars": scalars, "columns": {}}
    for name, values in sorted(cols.items()):
        n = kx + 1 if name in INTERFACE_COLS else kx
        out["columns"][name] = [values[i] for i in range(1, n + 1)]
    return out


def main() -> None:
    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])
    out = parse(infile)
    outfile.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {outfile}")


if __name__ == "__main__":
    main()
