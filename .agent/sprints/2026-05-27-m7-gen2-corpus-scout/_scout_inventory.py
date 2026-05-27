#!/usr/bin/env python3
"""Scout-only: build full Gen2 wrfout inventory under wrf_l2 + wrf_l3.

Header reads only (netCDF4.Dataset header), no variable loads.
Pinned to cores 0-3 (caller responsibility via taskset).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from netCDF4 import Dataset

GEN2_ROOTS = (
    Path("/mnt/data/canairy_meteo/runs/wrf_l3"),
    Path("/mnt/data/canairy_meteo/runs/wrf_l2"),
)
WRFOUT_RE = re.compile(
    r"^wrfout_(?P<dom>d0[1-5])_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)
RUN_DIR_RE_L3 = re.compile(
    r"^(?P<cycle>\d{8}_\d{2}z)_l3_24h_(?P<created>\d{8}T\d{6}Z)$"
)
# Tier4 probtest defaults (pinned)
PINNED_END_CYCLE = "20260520_18z"
PINNED_HELDOUT = "20260519_18z"
PINNED_LEADS_H = (0, 6, 12, 24)
PINNED_GRID_YX = (66, 159)  # mass (ny, nx) per artifacts/m6/gen2_manifest_v2.json
PINNED_GRID_E_WE_SN = (160, 67)  # staggered/dest in WRF header dims

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_wrfout(name: str):
    m = WRFOUT_RE.match(name)
    if not m:
        return None
    return m.group("dom"), datetime.strptime(m.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)

def header_dims_attrs(path: Path) -> dict:
    try:
        with Dataset(path, "r") as ds:
            dims = {k: int(len(v)) for k, v in ds.dimensions.items()}
            attrs = {}
            for a in ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON",
                     "TRUELAT1", "TRUELAT2", "STAND_LON",
                     "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
                     "WEST-EAST_PATCH_END_UNSTAG", "SOUTH-NORTH_PATCH_END_UNSTAG",
                     "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
                     "PARENT_GRID_RATIO"):
                if hasattr(ds, a):
                    v = getattr(ds, a)
                    attrs[a] = v.item() if hasattr(v, "item") else v
            return {"dimensions": dims, "global_attributes": attrs}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

def inventory_run(run_dir: Path) -> dict:
    name = run_dir.name
    files = sorted(p for p in run_dir.glob("wrfout_d*_*") if WRFOUT_RE.match(p.name))
    by_dom: dict[str, list[Path]] = {}
    for p in files:
        m = WRFOUT_RE.match(p.name)
        if m:
            by_dom.setdefault(m.group("dom"), []).append(p)

    # init time + advertised hours from run name (best-effort, l2 and l3)
    init_time = None
    advertised_h = None
    if name.endswith("Z"):
        toks = name.split("_")
        if len(toks) >= 5:
            try:
                init_time = datetime.strptime(toks[0] + toks[1], "%Y%m%d%Hz").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            # last hours hint: NNh or NNh_
            for t in toks:
                if t.endswith("h") and t[:-1].isdigit():
                    advertised_h = int(t[:-1])
                    break

    domains = {}
    for dom in ("d01", "d02", "d03", "d04", "d05"):
        ds_files = by_dom.get(dom, [])
        if not ds_files:
            continue
        times = []
        for p in ds_files:
            r = parse_wrfout(p.name)
            if r is None:
                continue
            times.append(r[1])
        times.sort()
        first = times[0] if times else None
        last = times[-1] if times else None
        observed_hours = int((last - first).total_seconds() // 3600) if len(times) >= 2 else 0
        # header from first file
        hdr = header_dims_attrs(ds_files[0])
        # grid shape (mass) from header dims if available
        mass_yx = None
        if "dimensions" in hdr:
            d = hdr["dimensions"]
            sn = d.get("south_north")
            we = d.get("west_east")
            if sn is not None and we is not None:
                mass_yx = [int(sn), int(we)]
        # complete: 24h+ observed hours, hourly stride, no gaps
        hourly_set = {t.replace(microsecond=0) for t in times}
        complete = False
        if init_time is not None and len(times) >= 25 and observed_hours >= 24:
            # require all 0..24 hours present
            expected = {init_time.replace(minute=0, second=0, microsecond=0).replace(microsecond=0)
                        .__add__(__import__("datetime").timedelta(hours=h))
                        for h in range(25)}
            complete = expected.issubset(hourly_set)
        total_bytes = 0
        file_records = []
        for p in ds_files:
            st = p.stat()
            total_bytes += int(st.st_size)
            r = parse_wrfout(p.name)
            file_records.append({
                "name": p.name,
                "valid_time_utc": r[1].isoformat() if r else None,
                "size_bytes": int(st.st_size),
            })
        domains[dom] = {
            "file_count": len(ds_files),
            "first_valid_time_utc": first.isoformat() if first else None,
            "last_valid_time_utc": last.isoformat() if last else None,
            "observed_hours_span": observed_hours,
            "total_bytes": total_bytes,
            "grid_mass_shape_yx": mass_yx,
            "header_dims": hdr.get("dimensions"),
            "header_attrs": hdr.get("global_attributes"),
            "header_error": hdr.get("error"),
            "is_complete_24h_hourly": complete,
            "files": file_records,
        }
    return {
        "run_id": name,
        "run_path": str(run_dir),
        "matches_l3_24h_pattern": bool(RUN_DIR_RE_L3.match(name)),
        "advertised_hours_from_name": advertised_h,
        "init_time_utc_from_name": init_time.isoformat() if init_time else None,
        "domains": domains,
    }


def main() -> int:
    inventory = {
        "schema": "M7Gen2CorpusScoutInventoryV1",
        "generated_utc": utc_iso(),
        "roots": [str(r) for r in GEN2_ROOTS],
        "pinned_grid_yx_mass": list(PINNED_GRID_YX),
        "pinned_end_cycle_inclusive": PINNED_END_CYCLE,
        "pinned_heldout_cycle_excluded": PINNED_HELDOUT,
        "required_leads_h": list(PINNED_LEADS_H),
        "runs": [],
    }
    for root in GEN2_ROOTS:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            inventory["runs"].append(inventory_run(child))

    # Aggregate stats
    total_files = 0
    d02_runs_with_files = 0
    d02_complete = 0
    d02_pinned_grid_complete = 0
    eligible_complete_pinned = 0
    for run in inventory["runs"]:
        for dom, rec in run["domains"].items():
            total_files += rec["file_count"]
            if dom == "d02" and rec["file_count"] > 0:
                d02_runs_with_files += 1
                if rec["is_complete_24h_hourly"]:
                    d02_complete += 1
                    if rec["grid_mass_shape_yx"] == list(PINNED_GRID_YX):
                        d02_pinned_grid_complete += 1
                        # tier4 eligibility: cycle ≤ end, ≠ heldout, l3_24h pattern
                        m = RUN_DIR_RE_L3.match(run["run_id"])
                        if m:
                            cy = m.group("cycle")
                            if cy <= PINNED_END_CYCLE and cy != PINNED_HELDOUT:
                                eligible_complete_pinned += 1
    inventory["aggregate"] = {
        "run_dir_count": len(inventory["runs"]),
        "wrfout_file_count_all_domains": total_files,
        "d02_runs_with_any_files": d02_runs_with_files,
        "d02_complete_24h_hourly_runs": d02_complete,
        "d02_pinned_grid_complete_24h_runs": d02_pinned_grid_complete,
        "tier4_eligible_pinned_complete_runs": eligible_complete_pinned,
        "tier4_required_member_count": 10,
        "tier4_corpus_gate": "PASS" if eligible_complete_pinned >= 10 else "BLOCKED_CORPUS",
    }
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("full_gen2_inventory.json")
    out_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(inventory["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
