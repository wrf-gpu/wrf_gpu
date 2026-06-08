#!/usr/bin/env python3
"""Parse the flat key=value dump from slab_oracle into a structured JSON savepoint.

Per-column scalars (length N): all 2-D fields (inputs + outputs).
2-D per-column soil arrays TSLB_IN[i,k] / TSLB[i,k] -> (N, NSOIL) lists.
Shared (NSOIL,) vectors ZS[k] / DZS[k].
Top scalars: CASE, REGIME, N, NSOIL, IFSNOW, DELTSM, DTMIN.
"""
import json
import re
import sys

INT_SCALARS = {"CASE", "N", "NSOIL", "IFSNOW", "FULL_WRF_EXE"}
STR_SCALARS = {"REGIME", "PRECISION_MODE"}
FLOAT_SCALARS = {"DELTSM", "DTMIN"}

COL_FIELDS = {
    "T", "QV", "P", "FLHC", "FLQC", "PSFC", "XLAND", "TMN", "GSW", "GLW",
    "THC", "SNOWC", "EMISS", "MAVAIL", "TSK_IN", "HFX_IN", "QFX_IN",
    "TSK", "HFX", "QFX", "LH", "QSFC", "CHKLOWQ", "CAPG",
}
SOIL2D_FIELDS = {"TSLB_IN", "TSLB"}
SOIL1D_FIELDS = {"ZS", "DZS"}

COL2D_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+),(\d+)\]=\s*(.+)$")
COL_RE = re.compile(r"^([A-Z0-9_]+)\[(\d+)\]=\s*(.+)$")
SCAL_RE = re.compile(r"^([A-Z0-9_]+)=\s*(.+)$")


def main(infile: str, outfile: str) -> None:
    scalars: dict = {}
    cols: dict = {}
    cols2d: dict = {}
    soil1d: dict = {}
    with open(infile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(("INIT_FATAL", "RUN_FATAL", "FATAL")):
                raise SystemExit(f"oracle reported fatal: {line}")
            m = COL2D_RE.match(line)
            if m:
                name, i, k, val = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
                cols2d.setdefault(name, {})[(i, k)] = val
                continue
            m = COL_RE.match(line)
            if m:
                name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
                if name in SOIL1D_FIELDS:
                    soil1d.setdefault(name, {})[idx] = val
                else:
                    cols.setdefault(name, {})[idx] = val
                continue
            m = SCAL_RE.match(line)
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if name in INT_SCALARS:
                scalars[name] = int(val)
            elif name in STR_SCALARS:
                scalars[name] = val.strip()
            elif name in FLOAT_SCALARS:
                scalars[name] = float(val)

    n = scalars["N"]
    nsoil = scalars["NSOIL"]
    out = {"scalars": scalars, "columns": {}, "soil": {}, "vectors": {}}
    for name, d in cols.items():
        out["columns"][name] = [d[k] for k in range(1, n + 1)]
    for name, d in cols2d.items():
        out["soil"][name] = [[d[(i, k)] for k in range(1, nsoil + 1)] for i in range(1, n + 1)]
    for name, d in soil1d.items():
        out["vectors"][name] = [d[k] for k in range(1, nsoil + 1)]
    with open(outfile, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {outfile}: scalars={len(scalars)} cols={list(out['columns'])} soil={list(out['soil'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
