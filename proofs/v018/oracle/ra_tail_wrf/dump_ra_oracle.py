#!/usr/bin/env python3
"""Extract compact RA tail savepoints from physics-pristine WRF wrfout files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np


SCHEME_LABELS = {
    3: "CAM radiation",
    5: "Goddard NUWRF radiation",
    7: "FLG/UCLA radiation",
    99: "GFDL-Eta radiation",
}

SOURCE_MODULES = {
    3: ["phys/module_radiation_driver.F:CAMRAD", "phys/module_ra_cam.F:CAMRAD"],
    5: ["phys/module_radiation_driver.F:goddardrad", "phys/module_ra_goddard.F:goddardrad"],
    7: ["phys/module_radiation_driver.F:RAD_FLG", "phys/module_ra_flg.F:RAD_FLG"],
    99: ["phys/module_radiation_driver.F:ETARA", "phys/module_ra_gfdleta.F:ETARA"],
}

LW_FIELDS = (
    "RTHRATLW",
    "RTHRATLWC",
    "GLW",
    "OLR",
    "TLWDN",
    "TLWUP",
    "SLWDN",
    "SLWUP",
    "LWUPT",
    "LWDNT",
    "LWUPB",
    "LWDNB",
    "LWUPTC",
    "LWDNTC",
    "LWUPBC",
    "LWDNBC",
    "ACLWUPT",
    "ACLWDNT",
    "ACLWUPB",
    "ACLWDNB",
    "ACLWUPTC",
    "ACLWDNTC",
    "ACLWUPBC",
    "ACLWDNBC",
)

SW_FIELDS = (
    "RTHRATSW",
    "RTHRATSWC",
    "SWDOWN",
    "TSWDN",
    "TSWUP",
    "SSWDN",
    "SSWUP",
    "SWUPT",
    "SWDNT",
    "SWUPB",
    "SWDNB",
    "SWUPTC",
    "SWDNTC",
    "SWUPBC",
    "SWDNBC",
    "ACSWUPT",
    "ACSWDNT",
    "ACSWUPB",
    "ACSWDNB",
    "ACSWUPTC",
    "ACSWDNTC",
    "ACSWUPBC",
    "ACSWDNBC",
)

CORE_FIELDS = ("RTHRATEN",) + LW_FIELDS + SW_FIELDS


def as_float_array(value: Any) -> np.ndarray:
    return np.asarray(np.ma.filled(value, np.nan), dtype=np.float64)


def finite_max_abs(arr: np.ndarray) -> float:
    values = as_float_array(arr)
    if values.size == 0:
        return 0.0
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(np.max(np.abs(values[finite])))


def time_strings(ds: netCDF4.Dataset) -> list[str]:
    if "Times" not in ds.variables:
        return []
    raw = ds.variables["Times"][:]
    try:
        return [str(x) for x in netCDF4.chartostring(raw)]
    except Exception:
        return []


def choose_anchor(ds: netCDF4.Dataset) -> tuple[int, int, int, str, float]:
    candidates = []
    for name in CORE_FIELDS:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        if len(var.dimensions) < 3 or var.dimensions[0] != "Time":
            continue
        data = as_float_array(var[:])
        if data.size == 0 or not np.isfinite(data).any():
            continue
        abs_data = np.abs(data)
        flat = int(np.nanargmax(abs_data))
        max_value = float(abs_data.reshape(-1)[flat])
        if data.ndim == 3:
            t, j, i = np.unravel_index(flat, data.shape)
            k = 0
        else:
            t, k, j, i = np.unravel_index(flat, data.shape)
        candidates.append((max_value, int(t), int(k), int(j), int(i), name))
    if not candidates:
        return 0, 0, 0, "none", 0.0
    max_value, t, k, j, i, name = max(candidates, key=lambda item: item[0])
    return t, j, i, name, max_value


def extract_value(var: netCDF4.Variable, time_index: int, j: int, i: int) -> Any:
    data = as_float_array(var[time_index])
    if data.ndim == 2:
        return float(data[j, i])
    if data.ndim == 3:
        return [float(x) for x in data[:, j, i]]
    return float(np.asarray(data).reshape(-1)[0])


def parse_wrfout(path: Path, scheme: int) -> dict[str, Any]:
    with netCDF4.Dataset(path) as ds:
        times = time_strings(ds)
        time_index, j, i, anchor_field, anchor_max = choose_anchor(ds)
        fields: dict[str, Any] = {}
        maxima: dict[str, float] = {}
        missing: list[str] = []
        for name in CORE_FIELDS:
            if name not in ds.variables:
                missing.append(name)
                continue
            var = ds.variables[name]
            maxima[name] = finite_max_abs(var[:])
            if len(var.dimensions) >= 3 and var.dimensions[0] == "Time":
                fields[name] = extract_value(var, time_index, j, i)

        nonzero_fields = [name for name, value in maxima.items() if math.isfinite(value) and value > 1.0e-15]
        lw_nonzero = [name for name in LW_FIELDS if name in nonzero_fields]
        sw_nonzero = [name for name in SW_FIELDS if name in nonzero_fields]
        return {
            "schema": "wrf-v018-ra-tail-real-wrf-column-savepoint-v1",
            "scheme": scheme,
            "label": SCHEME_LABELS[scheme],
            "ra_lw_physics": scheme,
            "ra_sw_physics": scheme,
            "wrfout": str(path),
            "time_index": time_index,
            "time": times[time_index] if 0 <= time_index < len(times) else None,
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
            "lw_nonzero_fields": lw_nonzero,
            "sw_nonzero_fields": sw_nonzero,
            "lw_nonzero": bool(lw_nonzero),
            "sw_nonzero": bool(sw_nonzero),
            "nontrivial": bool(lw_nonzero and sw_nonzero),
            "source_modules": SOURCE_MODULES[scheme],
            "exact_module_rule": (
                "Generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe "
                "from a real-data fixture with "
                f"ra_lw_physics={scheme}, ra_sw_physics={scheme}; WRF radiation_driver "
                "dispatches the exact listed upstream-identical radiation module, not a "
                "JAX self-compare."
            ),
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
    data = parse_wrfout(wrfouts[-1], args.scheme)
    try:
        data["wrfout"] = str(wrfouts[-1].resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        data["wrfout"] = str(wrfouts[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: nontrivial={data['nontrivial']} "
        f"lw={data['lw_nonzero']} sw={data['sw_nonzero']} anchor={data['anchor']}"
    )
    return 0 if data["nontrivial"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
