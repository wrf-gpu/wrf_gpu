#!/usr/bin/env python3
"""Parse the flat RUC LSM oracle dump into JSON.

Reads the lines emitted by ``ruclsm_oracle_driver.f90``:

* ``NAME=value``         scalars (CASE, N, NSL, NSTEPS, FULL_WRF_EXE, PRECISION_MODE)
* ``REGIME_NAME_<i>=str`` per-column regime labels
* ``VAR[i]=value``       per-column scalar fields
* ``VAR[i][k]=value``    per-column soil-layer profiles (k = soil level)

and writes ``{scalars, columns, profiles}`` JSON, mirroring the Pleim-Xiu
oracle dump format (``pxlsm_dump_to_json.py``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


INT_SCALARS = {"CASE", "N", "NSL", "NSTEPS", "FULL_WRF_EXE"}
STRING_SCALARS = {"PRECISION_MODE"}
SCALARS = INT_SCALARS | STRING_SCALARS

PROF_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]\[(\d+)\]=\s*(.+)$")
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
REGIME_RE = re.compile(r"^REGIME_NAME_(\d+)=\s*(.+)$")
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
    profs: dict[str, dict[int, dict[int, float]]] = {}
    regimes: dict[int, str] = {}

    with Path(infile).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")

            m = REGIME_RE.match(line)
            if m:
                regimes[int(m.group(1))] = m.group(2)
                continue

            m = PROF_RE.match(line)
            if m:
                name, i, k, val = m.group(1), int(m.group(2)), int(m.group(3)), _to_float(m.group(4))
                profs.setdefault(name, {}).setdefault(i, {})[k] = val
                continue

            m = COL_RE.match(line)
            if m:
                name, i, val = m.group(1), int(m.group(2)), _to_float(m.group(3))
                cols.setdefault(name, {})[i] = val
                continue

            m = SCAL_RE.match(line)
            if m and m.group(1) in SCALARS:
                name, val = m.group(1), m.group(2)
                scalars[name] = int(val) if name in INT_SCALARS else val

    ncol = int(scalars["N"])
    nsl = int(scalars["NSL"])

    out: dict = {"scalars": scalars, "columns": {}, "profiles": {}}
    out["scalars"]["regime_names"] = [regimes.get(i, "") for i in range(1, ncol + 1)]

    for name, values in cols.items():
        out["columns"][name] = [values[i] for i in range(1, ncol + 1)]

    for name, byi in profs.items():
        out["profiles"][name] = [
            [byi[i][k] for k in range(1, nsl + 1)] for i in range(1, ncol + 1)
        ]

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"wrote {path}: scalars={len(scalars)} cols={len(out['columns'])} "
        f"profiles={len(out['profiles'])}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
