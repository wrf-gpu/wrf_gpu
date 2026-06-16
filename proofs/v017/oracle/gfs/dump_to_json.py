#!/usr/bin/env python3
"""Parse a GFS PBL oracle text dump into the savepoint JSON schema.

The Fortran driver prints scalars as ``NAME=value`` and column entries as
``NAME[k]=value`` (1-based, bottom-up WRF mass levels). This collects them into
``{"scalars": {...}, "columns": {NAME: [...]}}``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COL_RE = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]=(.+)$")
SCALAR_RE = re.compile(r"^([A-Za-z0-9_]+)=(.+)$")
INT_SCALARS = {"CASE", "KX", "FULL_WRF_EXE", "KPBL"}
STR_SCALARS = {"REGIME"}


def main(in_path: str, out_path: str) -> int:
    columns: dict[str, dict[int, float]] = {}
    scalars: dict[str, object] = {}
    for raw in Path(in_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = COL_RE.match(line)
        if m:
            name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
            columns.setdefault(name, {})[idx] = val
            continue
        m = SCALAR_RE.match(line)
        if m:
            name, val = m.group(1), m.group(2).strip()
            if name in STR_SCALARS:
                scalars[name] = val
            elif name in INT_SCALARS:
                scalars[name] = int(val)
            else:
                scalars[name] = float(val)
    cols_out = {
        name: [vals[k] for k in sorted(vals)] for name, vals in columns.items()
    }
    out = {"scalars": scalars, "columns": cols_out}
    Path(out_path).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
