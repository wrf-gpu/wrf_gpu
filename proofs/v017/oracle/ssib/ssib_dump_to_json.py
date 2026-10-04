#!/usr/bin/env python3
"""Parse the flat SSiB LSM oracle dump into JSON.

Reads ``VAR[i]=value`` per-column lines and ``NAME=value`` scalars emitted by
``ssib_oracle_driver.f90`` and writes ``{scalars, columns}`` JSON, mirroring
``proofs/v017/oracle/pxlsm/pxlsm_dump_to_json.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


INT_SCALARS = {"CASE", "N", "FULL_WRF_EXE"}
STRING_SCALARS = {"REGIME_NAME", "PRECISION_MODE"}
SCALARS = INT_SCALARS | STRING_SCALARS
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")
# gfortran ES23.15 drops the 'E' for 3-digit exponents (e.g. '1.0E-321' -> '1.0-321').
MISSING_E_RE = re.compile(r"^([+-]?\d+\.\d+)([+-]\d{3})$")


def _to_float(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        m = MISSING_E_RE.match(token.strip())
        if m:
            return float(f"{m.group(1)}E{m.group(2)}")
        raise


def main(infile: str, outfile: str) -> None:
    scalars: dict[str, int | str] = {}
    cols: dict[str, dict[int, float]] = {}
    with Path(infile).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            match = COL_RE.match(line)
            if match:
                name, idx, value = match.group(1), int(match.group(2)), _to_float(match.group(3))
                cols.setdefault(name, {})[idx] = value
                continue
            match = SCAL_RE.match(line)
            if match and match.group(1) in SCALARS:
                name, value = match.group(1), match.group(2)
                if name in INT_SCALARS:
                    scalars[name] = int(value)
                else:
                    scalars[name] = value

    ncol = int(scalars["N"])
    out = {"scalars": scalars, "columns": {}}
    for name, values in cols.items():
        out["columns"][name] = [values[i] for i in range(1, ncol + 1)]

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path}: scalars={len(scalars)} cols={len(out['columns'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
