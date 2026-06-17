#!/usr/bin/env python3
"""Extract compact CU tail savepoints from a pristine WRF wrfout file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np


SCHEME_LABELS = {
    7: "Zhang-McFarlane CAMZM",
    10: "KF-CuP",
    11: "MSKF",
}

CORE_FIELDS = (
    "RAINCV",
    "PRATEC",
    "CUTOP",
    "CUBOT",
    "RTHCUTEN",
    "RQVCUTEN",
    "RQCCUTEN",
    "RQICUTEN",
    "RUCUTEN",
    "RVCUTEN",
    "W0AVG",
    "CLDFRA_CUP",
    "CLDFRATEND_CUP",
    "UPDFRA_CUP",
    "QC_IC_CUP",
    "QC_IU_CUP",
    "QNDROP_IC_CUP",
    "MFUP_CUP",
    "MFDN_CUP",
    "WUP_CUP",
    "WACT_CUP",
    "TCLOUD_CUP",
    "TAUCLOUD",
    "activeFrac",
    "ZMDT",
    "ZMDQ",
    "CMFMC",
    "CMFMCDZM",
    "PRECZ",
    "preccdzm",
    "PCONVB",
    "PCONVT",
)


def finite_max_abs(arr: np.ndarray) -> float:
    values = np.asarray(arr, dtype=np.float64)
    if values.size == 0:
        return 0.0
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(np.max(np.abs(values[finite])))


def choose_anchor(ds: netCDF4.Dataset, time_index: int) -> tuple[int, int, str, float]:
    candidates = []
    for name in CORE_FIELDS:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        if len(var.dimensions) < 3:
            continue
        data = np.asarray(var[time_index], dtype=np.float64)
        if data.size == 0 or not np.isfinite(data).any():
            continue
        value = np.abs(data)
        flat = int(np.nanargmax(value))
        max_value = float(value.reshape(-1)[flat])
        if len(data.shape) == 2:
            j, i = np.unravel_index(flat, data.shape)
        else:
            _k, j, i = np.unravel_index(flat, data.shape)
        candidates.append((max_value, int(j), int(i), name))
    if not candidates:
        return 0, 0, "none", 0.0
    max_value, j, i, name = max(candidates, key=lambda item: item[0])
    return j, i, name, max_value


def extract_value(var: netCDF4.Variable, time_index: int, j: int, i: int) -> Any:
    data = np.asarray(var[time_index], dtype=np.float64)
    if data.ndim == 2:
        return float(data[j, i])
    if data.ndim == 3:
        return [float(x) for x in data[:, j, i]]
    return float(np.asarray(data).reshape(-1)[0])


def parse_wrfout(path: Path, scheme: int) -> dict[str, Any]:
    with netCDF4.Dataset(path) as ds:
        time_index = len(ds.dimensions["Time"]) - 1
        j, i, anchor_field, anchor_max = choose_anchor(ds, time_index)
        fields: dict[str, Any] = {}
        maxima: dict[str, float] = {}
        missing: list[str] = []
        for name in CORE_FIELDS:
            if name not in ds.variables:
                missing.append(name)
                continue
            var = ds.variables[name]
            values = np.asarray(var[time_index], dtype=np.float64)
            max_abs = finite_max_abs(values)
            maxima[name] = max_abs
            if len(var.dimensions) >= 3:
                fields[name] = extract_value(var, time_index, j, i)
        nonzero_fields = [name for name, value in maxima.items() if math.isfinite(value) and value > 1.0e-15]
        return {
            "schema": "wrf-v018-cu-tail-real-wrf-column-savepoint-v1",
            "scheme": scheme,
            "label": SCHEME_LABELS[scheme],
            "wrfout": str(path),
            "time_index": time_index,
            "anchor": {
                "south_north": j,
                "west_east": i,
                "field": anchor_field,
                "max_abs": anchor_max,
            },
            "max_abs": maxima,
            "column_or_scalar": fields,
            "missing_fields": missing,
            "nonzero_fields": nonzero_fields,
            "nontrivial": bool(nonzero_fields),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=int, required=True, choices=sorted(SCHEME_LABELS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wrfouts = sorted(args.run_dir.glob("wrfout_d01_*"))
    if not wrfouts:
        raise SystemExit(f"no wrfout files found in {args.run_dir}")
    wrfout = wrfouts[-1]
    data = parse_wrfout(wrfout, args.scheme)
    try:
        data["wrfout"] = str(wrfout.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        data["wrfout"] = str(wrfout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: nontrivial={data['nontrivial']} anchor={data['anchor']}")
    return 0 if data["nontrivial"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
