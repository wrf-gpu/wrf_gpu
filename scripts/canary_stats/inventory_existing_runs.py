#!/usr/bin/env python3
"""Inventory existing Canary WRF/GPUWRF run outputs without launching forecasts."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from netCDF4 import Dataset

from common import (
    DEFAULT_SCORE_VARS,
    SURFACE_VARS,
    THREED_VARS,
    json_default,
    list_wrfout_files,
    load_json_if_present,
    parse_case_id,
    parse_init_from_case_id,
    parse_iso_time,
    parse_wrfout_time,
    write_csv,
    write_json,
)


DEFAULT_ROOTS = [
    "/mnt/data/canairy_meteo/runs/wrf_l2",
    "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output",
    "/mnt/data/canairy_meteo/runs/wrf_l3",
    "/mnt/data/canairy_meteo/runs/campaign_l2",
    "/mnt/data/canairy_meteo/runs/campaign_l3",
    "/mnt/data/canairy_meteo/runs/forcing_cases",
    "/mnt/data/canairy_meteo/runs/wps_cases",
    "/mnt/data/canairy_meteo/runs/surface_geo_v2_1",
    "/mnt/data/canairy_meteo/runs/phys_sweep",
    "/mnt/data/canairy_meteo/runs/terrain_sweep",
    "/mnt/data/canairy_meteo/runs/cu0_confirm",
    "/mnt/data/canairy_meteo/gate_2way_d02_v013",
    "/mnt/data/canairy_meteo/gate_2way_v013",
    "/mnt/data/canairy_meteo/gate_2way_v013c",
    "/mnt/data/canairy_meteo/gate_gwd_nested_v013b",
    "/mnt/data/canairy_meteo/gate_revalidate_gwd8",
    "/mnt/data/canairy_meteo/gen2_archive/teacher_l3",
    "/tmp/v0120_powered_tost_runs",
    "proofs/m20/tost_run/gpu_wrfout",
]

RELEVANT_PREFIXES = (
    "wrfout_d",
    "wrfrst_d",
    "wrfinput_d",
    "wrfbdy_d",
    "met_em.d",
    "geo_em.d",
)
RELEVANT_NAMES = {
    "namelist.input",
    "namelist.output",
    "namelist.wps",
    "rsl.error.0000",
    "rsl.out.0000",
    "payload.json",
    "run.rc",
}
NAMELIST_KEYS = {
    "time_control": ["run_hours", "history_interval", "frames_per_outfile"],
    "domains": ["max_dom", "time_step", "e_we", "e_sn", "e_vert", "dx", "dy", "feedback"],
    "physics": [
        "physics_suite",
        "mp_physics",
        "bl_pbl_physics",
        "sf_sfclay_physics",
        "sf_surface_physics",
        "ra_lw_physics",
        "ra_sw_physics",
        "cu_physics",
        "gwd_opt",
    ],
    "bdy_control": ["specified", "nested"],
}


def _rel_to_cwd(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _is_relevant_file(name: str) -> bool:
    return name in RELEVANT_NAMES or any(name.startswith(prefix) for prefix in RELEVANT_PREFIXES)


def _walk_candidate_dirs(roots: list[Path], max_depth: int) -> list[Path]:
    candidates: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        for cur, dirs, files in os.walk(root):
            cur_path = Path(cur)
            try:
                depth = len(cur_path.relative_to(root).parts)
            except ValueError:
                depth = 0
            if depth >= max_depth:
                dirs[:] = []
            if any(_is_relevant_file(name) for name in files):
                candidates.add(cur_path)
            # Keep explicit empty campaign roots visible in the inventory.
            if depth == 0 and root.name.startswith("campaign_"):
                candidates.add(cur_path)
    return sorted(candidates, key=lambda p: str(p))


def _parse_namelist(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    groups: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            current = line[1:].strip().lower()
            groups.setdefault(current, {})
            continue
        if line.startswith("/"):
            current = None
            continue
        if current and "=" in line:
            key, value = line.split("=", 1)
            groups[current][key.strip().lower()] = value.strip().rstrip(",")
    return groups


def _select_namelist_summary(path: Path) -> dict[str, dict[str, str]]:
    parsed = _parse_namelist(path)
    out: dict[str, dict[str, str]] = {}
    for group, keys in NAMELIST_KEYS.items():
        vals = parsed.get(group, {})
        keep = {key: vals[key] for key in keys if key in vals}
        if keep:
            out[group] = keep
    return out


def _m20_sidecar_metadata(run_dir: Path) -> dict[str, Any]:
    parts = run_dir.parts
    if len(parts) < 3 or parts[-2] != "gpu_wrfout":
        return {}
    unit_id = run_dir.name
    base = run_dir.parent.parent
    for name in ("tost_campaign_result.json", "tost_run_plan.json"):
        payload = load_json_if_present(base / name)
        if not payload:
            continue
        units = payload.get("per_unit_meta") or payload.get("units") or []
        for unit in units:
            if unit.get("unit_id") == unit_id:
                return {
                    "unit_id": unit_id,
                    "init_time_utc": unit.get("init_utc"),
                    "forecast_hours": unit.get("fh") or unit.get("max_lead_h"),
                    "cpu_run_dir": unit.get("cpu_run_dir"),
                    "level": unit.get("level"),
                    "sidecar": _rel_to_cwd(base / name),
                }
    return {"unit_id": unit_id}


def _payload_metadata(run_dir: Path) -> dict[str, Any]:
    payload = load_json_if_present(run_dir / "payload.json")
    if not payload:
        return {}
    keep = {
        "device": payload.get("device"),
        "init_mode": payload.get("init_mode"),
        "feedback": payload.get("feedback"),
        "max_dom": payload.get("max_dom"),
        "hours": payload.get("hours"),
        "input_dir": payload.get("input_dir"),
        "all_domains_finite": payload.get("all_domains_finite"),
        "all_outputs_present": payload.get("all_outputs_present"),
    }
    metadata = payload.get("metadata", {})
    domains = metadata.get("domains", {}) if isinstance(metadata, dict) else {}
    keep["domain_metadata"] = {
        name: {
            "grid": info.get("grid"),
            "namelist": info.get("namelist"),
            "ic_source": info.get("ic_source"),
            "lbc_source": info.get("lbc_source"),
        }
        for name, info in domains.items()
        if isinstance(info, dict)
    }
    return {k: v for k, v in keep.items() if v is not None}


def _domain_from_name(name: str) -> str | None:
    m = re.search(r"d\d{2}", name)
    return m.group(0) if m else None


def _dataset_domain_summary(path: Path, init_time: datetime | None) -> dict[str, Any]:
    with Dataset(path, "r") as ds:
        attrs = {}
        for key in (
            "TITLE",
            "START_DATE",
            "SIMULATION_START_DATE",
            "DX",
            "DY",
            "GRID_ID",
            "PARENT_ID",
            "PARENT_GRID_RATIO",
            "WEST-EAST_GRID_DIMENSION",
            "SOUTH-NORTH_GRID_DIMENSION",
            "BOTTOM-TOP_GRID_DIMENSION",
        ):
            if hasattr(ds, key):
                attrs[key] = json_default(getattr(ds, key))
        dims = {
            key: int(len(dim))
            for key, dim in ds.dimensions.items()
            if key
            in {
                "Time",
                "bottom_top",
                "bottom_top_stag",
                "south_north",
                "west_east",
                "south_north_stag",
                "west_east_stag",
            }
        }
        selected = list(SURFACE_VARS + THREED_VARS)
        present = [name for name in selected if name in ds.variables]
        dtypes = {
            name: str(ds.variables[name].dtype)
            for name in present
            if name in DEFAULT_SCORE_VARS or name in {"PSFC", "RAINNC", "T", "U", "V", "W"}
        }
        nvars = len(ds.variables)
    valid = parse_wrfout_time(path)
    lead = None
    if valid and init_time:
        lead = int(round((valid - init_time).total_seconds() / 3600.0))
    return {
        "sample_file": str(path),
        "sample_file_name": path.name,
        "attrs": attrs,
        "dims": dims,
        "n_variables": int(nvars),
        "selected_variables_present": present,
        "selected_variable_dtypes": dtypes,
        "sample_valid_time_utc": valid.isoformat() if valid else None,
        "sample_lead_hour": lead,
    }


def _summarize_domain(run_dir: Path, domain: str, init_time: datetime | None) -> dict[str, Any]:
    files = list_wrfout_files(run_dir, domain)
    if not files:
        return {"wrfout_frame_count": 0}
    times = [t for t, _ in files]
    leads = []
    if init_time is not None:
        leads = [int(round((t - init_time).total_seconds() / 3600.0)) for t in times]
    summary = _dataset_domain_summary(files[0][1], init_time)
    summary.update(
        {
            "wrfout_frame_count": int(len(files)),
            "first_time_utc": times[0].isoformat(),
            "last_time_utc": times[-1].isoformat(),
            "lead_hours": leads,
            "lead_min": min(leads) if leads else None,
            "lead_max": max(leads) if leads else None,
            "lead_count": len(set(leads)) if leads else None,
        }
    )
    if leads:
        expected = set(range(min(leads), max(leads) + 1))
        missing = sorted(expected - set(leads))
        summary["missing_leads_in_span"] = missing
    return summary


def _infer_provenance(record: dict[str, Any]) -> str:
    titles = []
    for dom in record.get("domains", {}).values():
        attrs = dom.get("attrs", {}) if isinstance(dom, dict) else {}
        if attrs.get("TITLE"):
            titles.append(str(attrs["TITLE"]))
    joined = " ".join(titles).upper()
    if "GPUWRF" in joined:
        return "GPU/JAX GPUWRF"
    if "WRF V" in joined or "OUTPUT FROM WRF" in joined:
        return "CPU-WRF"
    counts = record.get("file_counts", {})
    if counts.get("met_em", 0) or counts.get("wrfinput", 0) or counts.get("wrfbdy", 0):
        return "forcing/input-only"
    if record.get("payload", {}).get("device"):
        return "GPU/JAX proof-output"
    if record.get("sidecar", {}).get("unit_id") and counts.get("wrfout", 0):
        return "GPU/JAX proof-output"
    return "unknown"


def _scan_run_dir(run_dir: Path, root: Path) -> dict[str, Any]:
    files = [p for p in run_dir.iterdir() if p.is_file()]
    counts = Counter()
    for p in files:
        name = p.name
        if name.startswith("wrfout_d"):
            counts["wrfout"] += 1
        elif name.startswith("wrfrst_d"):
            counts["wrfrst"] += 1
        elif name.startswith("wrfinput_d"):
            counts["wrfinput"] += 1
        elif name.startswith("wrfbdy_d"):
            counts["wrfbdy"] += 1
        elif name.startswith("met_em.d"):
            counts["met_em"] += 1
        elif name.startswith("geo_em.d"):
            counts["geo_em"] += 1
        elif name.startswith("rsl."):
            counts["rsl"] += 1
    wrfout_domains = sorted({_domain_from_name(p.name) for p in files if p.name.startswith("wrfout_d")})
    wrfout_domains = [d for d in wrfout_domains if d]
    sidecar = _m20_sidecar_metadata(run_dir)
    payload = _payload_metadata(run_dir)
    case_id = (
        parse_case_id(run_dir.name)
        or parse_case_id(str(run_dir))
        or parse_case_id(str(payload.get("input_dir", "")))
        or parse_case_id(str(sidecar.get("cpu_run_dir", "")))
    )
    init_time = parse_iso_time(sidecar.get("init_time_utc")) or parse_init_from_case_id(case_id)
    if not case_id and init_time:
        case_id = init_time.strftime("%Y%m%d_%Hz").lower()
    if not init_time:
        for p in files:
            if p.name.startswith("wrfout_d"):
                valid = parse_wrfout_time(p)
                if valid:
                    init_time = valid if valid.minute == 0 else None
                    break
    domains = {d: _summarize_domain(run_dir, d, init_time) for d in wrfout_domains}
    namelist = _select_namelist_summary(run_dir / "namelist.input")
    record: dict[str, Any] = {
        "path": str(run_dir),
        "path_rel": _rel_to_cwd(run_dir),
        "root": str(root),
        "run_id": run_dir.name,
        "case_id": case_id,
        "init_time_utc": init_time.isoformat() if init_time else sidecar.get("init_time_utc"),
        "domains_present": wrfout_domains,
        "domains": domains,
        "file_counts": dict(counts),
        "namelist": namelist,
        "payload": payload,
        "sidecar": sidecar,
    }
    record["provenance"] = _infer_provenance(record)
    return record


def _human_domains(record: dict[str, Any]) -> str:
    parts = []
    for dom in sorted(record.get("domains", {})):
        d = record["domains"][dom]
        span = ""
        if d.get("lead_min") is not None:
            span = f" L{d['lead_min']}-{d['lead_max']}"
        parts.append(f"{dom}:{d.get('wrfout_frame_count', 0)}{span}")
    return ", ".join(parts) if parts else "-"


def _human_vars(record: dict[str, Any]) -> str:
    all_present = set()
    for dom in record.get("domains", {}).values():
        all_present.update(dom.get("selected_variables_present", []))
    surf = [v for v in SURFACE_VARS if v in all_present]
    three = [v for v in ("T", "U", "V", "W", "QVAPOR") if v in all_present]
    if not all_present:
        return "-"
    return f"surf={','.join(surf) or '-'}; 3d={','.join(three) or '-'}"


def _human_physics(record: dict[str, Any]) -> str:
    phys = record.get("namelist", {}).get("physics", {})
    if not phys:
        payload_domains = record.get("payload", {}).get("domain_metadata", {})
        vals = []
        for dom, info in sorted(payload_domains.items()):
            nl = info.get("namelist", {}) if isinstance(info, dict) else {}
            if nl:
                vals.append(f"{dom}:cu={nl.get('cu_physics')} gwd={nl.get('gwd_opt')}")
        return "; ".join(vals) if vals else "-"
    keys = ["mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "ra_lw_physics", "ra_sw_physics", "cu_physics", "gwd_opt"]
    return " ".join(f"{k}={phys[k]}" for k in keys if k in phys) or "-"


def _write_human_table(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Canary Existing Run Inventory",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| provenance | case | run_id | domains/leads | vars | physics/proof metadata | path |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        path_s = rec["path_rel"]
        lines.append(
            "| {prov} | {case} | {run} | {domains} | {vars} | {phys} | `{path}` |".format(
                prov=rec.get("provenance", "-"),
                case=rec.get("case_id") or "-",
                run=rec.get("run_id") or "-",
                domains=_human_domains(rec),
                vars=_human_vars(rec),
                phys=_human_physics(rec).replace("|", "/"),
                path=path_s,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        rows.append(
            {
                "path": rec["path"],
                "run_id": rec["run_id"],
                "case_id": rec.get("case_id"),
                "init_time_utc": rec.get("init_time_utc"),
                "provenance": rec.get("provenance"),
                "domains": _human_domains(rec),
                "vars": _human_vars(rec),
                "physics": _human_physics(rec),
                "wrfout_count": rec.get("file_counts", {}).get("wrfout", 0),
                "wrfinput_count": rec.get("file_counts", {}).get("wrfinput", 0),
                "wrfbdy_count": rec.get("file_counts", {}).get("wrfbdy", 0),
                "met_em_count": rec.get("file_counts", {}).get("met_em", 0),
                "sidecar_cpu_run_dir": rec.get("sidecar", {}).get("cpu_run_dir"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", type=Path, help="Root to scan; repeatable")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("proofs/canary_stats/2026-06-08_existing_data"))
    args = ap.parse_args(argv)

    roots = [p for p in (args.root or [Path(p) for p in DEFAULT_ROOTS])]
    candidates = _walk_candidate_dirs(roots, args.max_depth)
    records = [_scan_run_dir(path, next((r for r in roots if str(path).startswith(str(r.resolve()))), roots[0])) for path in candidates]
    records = sorted(records, key=lambda r: (r.get("provenance", ""), r.get("case_id") or "", r["path"]))

    out_dir = args.out_dir
    payload = {
        "schema": "CanaryExistingRunInventory",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "roots": [str(p) for p in roots],
        "record_count": len(records),
        "records": records,
    }
    write_json(out_dir / "inventory.json", payload)
    rows = _csv_rows(records)
    write_csv(
        out_dir / "inventory.csv",
        rows,
        [
            "path",
            "run_id",
            "case_id",
            "init_time_utc",
            "provenance",
            "domains",
            "vars",
            "physics",
            "wrfout_count",
            "wrfinput_count",
            "wrfbdy_count",
            "met_em_count",
            "sidecar_cpu_run_dir",
        ],
    )
    _write_human_table(out_dir / "inventory_table.md", records)
    print(f"inventory records={len(records)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
