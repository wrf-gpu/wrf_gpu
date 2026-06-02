#!/usr/bin/env python3
"""v0.3.0 S0 recon — machine-readable inventory of the metgrid oracle, static
geog, and raw AIFS forcing.

CPU-only, read-only. Produces ``proofs/v030/recon_inventory.json`` (a structured
dump of every variable / dim / attr in the reference NetCDF files) so the S1-S5
lanes can build against a frozen, audited picture of the inputs and the oracle
rather than re-deriving the format ad hoc.

Run:
    JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 proofs/v030/recon_inventory.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


WPS_CASES_ROOT = Path("/mnt/data/canairy_meteo/runs/wps_cases")
AIFS_SINGLE = Path("/mnt/data/canairy_meteo/data/aifs_single/aifs_single_202405.nc")
AIFS_EXPANDED = Path(
    "/mnt/data/canairy_meteo/data/aifs_single_expanded_fields_v1/"
    "aifs_single_expanded_202405.nc"
)
OUT = Path(__file__).resolve().parent / "recon_inventory.json"


def _attrs(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in obj.ncattrs():
        val = obj.getncattr(name)
        if isinstance(val, (np.generic,)):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        out[name] = val
    return out


def _var_stats(ds: Dataset, vname: str) -> dict[str, Any]:
    """Cheap min/max/mean/has_fill on a variable without holding the whole array."""
    var = ds.variables[vname]
    try:
        arr = np.asarray(var[:])
        if arr.dtype.kind in ("S", "U"):
            sample = arr.flatten()[:1]
            return {"sample": [s.decode() if isinstance(s, bytes) else str(s) for s in sample]}
        finite = arr[np.isfinite(arr)]
        stats = {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "n_nan": int(np.isnan(arr).sum()),
        }
        return stats
    except Exception as exc:  # pragma: no cover - recon best-effort
        return {"error": repr(exc)}


def dump_dataset(path: Path, *, with_stats: bool = True, max_stat_vars: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    with Dataset(path) as ds:
        out["dimensions"] = {name: (len(dim), bool(dim.isunlimited())) for name, dim in ds.dimensions.items()}
        out["global_attrs"] = _attrs(ds)
        variables: dict[str, Any] = {}
        for i, (vname, var) in enumerate(ds.variables.items()):
            entry: dict[str, Any] = {
                "dtype": str(var.dtype),
                "dims": list(var.dimensions),
                "shape": list(var.shape),
                "attrs": _attrs(var),
            }
            if with_stats and (max_stat_vars is None or i < max_stat_vars):
                entry["stats"] = _var_stats(ds, vname)
            variables[vname] = entry
        out["variables"] = variables
    return out


def latest_case() -> Path:
    cases = sorted(p for p in WPS_CASES_ROOT.iterdir() if p.is_dir())
    return cases[-1]


def list_cases() -> list[dict[str, Any]]:
    """Inventory all wps_cases: which domains/timestamps each has met_em + geo_em for."""
    out = []
    for case in sorted(p for p in WPS_CASES_ROOT.iterdir() if p.is_dir()):
        l3 = case / "l3"
        if not l3.is_dir():
            out.append({"case": case.name, "l3": False})
            continue
        met = sorted(l3.glob("met_em.*.nc"))
        geo = sorted(l3.glob("geo_em.*.nc"))
        domains: dict[str, int] = {}
        for m in met:
            dom = m.name.split(".")[1]
            domains[dom] = domains.get(dom, 0) + 1
        out.append(
            {
                "case": case.name,
                "l3": True,
                "n_met_em": len(met),
                "met_em_domains": domains,
                "geo_em": [g.name for g in geo],
                "first_met_em": met[0].name if met else None,
            }
        )
    return out


def main() -> None:
    case = latest_case()
    l3 = case / "l3"
    report: dict[str, Any] = {"reference_case": case.name}

    report["all_cases"] = list_cases()

    # met_em oracle, all three operational domains, first timestamp.
    met_d01 = sorted(l3.glob("met_em.d01.*.nc"))[0]
    met_d02 = sorted(l3.glob("met_em.d02.*.nc"))[0]
    met_d03 = sorted(l3.glob("met_em.d03.*.nc"))[0]
    report["met_em_d01"] = dump_dataset(met_d01)
    report["met_em_d02"] = dump_dataset(met_d02)
    report["met_em_d03"] = dump_dataset(met_d03)

    # static geog.
    report["geo_em_d01"] = dump_dataset(l3 / "geo_em.d01.nc")
    report["geo_em_d02"] = dump_dataset(l3 / "geo_em.d02.nc")
    report["geo_em_d03"] = dump_dataset(l3 / "geo_em.d03.nc")

    # raw AIFS forcing (variable list + grid; stats only on a few).
    report["aifs_single"] = dump_dataset(AIFS_SINGLE, with_stats=False)
    report["aifs_single_expanded"] = dump_dataset(AIFS_EXPANDED, with_stats=False)

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

    # Print a compact human summary to stdout.
    print("\n=== CASES ===")
    for c in report["all_cases"]:
        if c.get("l3"):
            print(f"  {c['case']}: met_em={c['met_em_domains']} geo_em={len(c['geo_em'])}")
        else:
            print(f"  {c['case']}: NO l3")
    for key in ("met_em_d01", "met_em_d02", "met_em_d03"):
        d = report[key]
        print(f"\n=== {key} dims ===")
        print("  ", d["dimensions"])
        print(f"  n_vars={len(d['variables'])}")
    print("\n=== AIFS single vars ===")
    print("  ", list(report["aifs_single"].get("variables", {}).keys()))
    print("\n=== AIFS expanded vars ===")
    print("  ", list(report["aifs_single_expanded"].get("variables", {}).keys()))


if __name__ == "__main__":
    main()
