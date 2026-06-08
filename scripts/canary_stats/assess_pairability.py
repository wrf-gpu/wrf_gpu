#!/usr/bin/env python3
"""Assess raw CPU/GPU pairability from a canary_stats inventory."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_SCORE_VARS, parse_iso_time, write_csv, write_json


def _domain_key(record: dict[str, Any], domain: str) -> tuple[Any, ...]:
    d = record.get("domains", {}).get(domain, {})
    attrs = d.get("attrs", {})
    dims = d.get("dims", {})
    return (
        attrs.get("DX"),
        attrs.get("DY"),
        dims.get("south_north"),
        dims.get("west_east"),
        dims.get("bottom_top"),
    )


def _times(record: dict[str, Any], domain: str) -> set[str]:
    leads = record.get("domains", {}).get(domain, {}).get("lead_hours")
    if leads:
        return {f"L{int(x):03d}" for x in leads}
    # Fall back to valid times if no init was inferable.
    first = record.get("domains", {}).get(domain, {}).get("first_time_utc")
    last = record.get("domains", {}).get(domain, {}).get("last_time_utc")
    count = record.get("domains", {}).get(domain, {}).get("wrfout_frame_count", 0)
    if first and last:
        return {f"{first}..{last}#{count}"}
    return set()


def _vars(record: dict[str, Any], domain: str) -> set[str]:
    return set(record.get("domains", {}).get(domain, {}).get("selected_variables_present", []))


def _pair_record(cpu: dict[str, Any], gpu: dict[str, Any], domain: str, reason: str) -> dict[str, Any]:
    cpu_times = _times(cpu, domain)
    gpu_times = _times(gpu, domain)
    common = sorted(cpu_times & gpu_times)
    common_vars = sorted(_vars(cpu, domain) & _vars(gpu, domain) & set(DEFAULT_SCORE_VARS))
    same_grid = _domain_key(cpu, domain) == _domain_key(gpu, domain)
    if common and common_vars and same_grid:
        cls = "pairable_raw_same_grid"
    elif common and common_vars:
        cls = "pairable_station_only_grid_mismatch"
    elif common:
        cls = "not_usable_missing_score_vars"
    else:
        cls = "not_pairable_no_common_leads"
    if len(common) < 24 and cls.startswith("pairable"):
        cls = "exploratory_incomplete_leads"
    return {
        "classification": cls,
        "reason": reason,
        "domain": domain,
        "case_id": gpu.get("case_id") or cpu.get("case_id"),
        "cpu_run_id": cpu.get("run_id"),
        "gpu_run_id": gpu.get("run_id"),
        "cpu_path": cpu.get("path"),
        "gpu_path": gpu.get("path"),
        "cpu_provenance": cpu.get("provenance"),
        "gpu_provenance": gpu.get("provenance"),
        "common_lead_count": len(common),
        "common_leads": common,
        "same_grid": same_grid,
        "cpu_grid": _domain_key(cpu, domain),
        "gpu_grid": _domain_key(gpu, domain),
        "common_score_vars": common_vars,
        "cpu_frame_count": cpu.get("domains", {}).get(domain, {}).get("wrfout_frame_count", 0),
        "gpu_frame_count": gpu.get("domains", {}).get(domain, {}).get("wrfout_frame_count", 0),
    }


def _records_by_path(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(Path(r["path"]).resolve()): r for r in records}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("proofs/canary_stats/2026-06-08_existing_data"))
    args = ap.parse_args(argv)

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    records = inv["records"]
    by_path = _records_by_path(records)
    cpu_records = [r for r in records if r.get("provenance") == "CPU-WRF"]
    gpu_records = [r for r in records if str(r.get("provenance", "")).startswith("GPU/JAX")]
    cpu_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in cpu_records:
        for domain in r.get("domains", {}):
            if r.get("case_id"):
                cpu_by_case[(r["case_id"], domain)].append(r)

    pairs: list[dict[str, Any]] = []
    for gpu in gpu_records:
        for domain in gpu.get("domains", {}):
            explicit = gpu.get("sidecar", {}).get("cpu_run_dir")
            if explicit:
                cpu = by_path.get(str(Path(explicit).resolve()))
                if cpu and domain in cpu.get("domains", {}):
                    pairs.append(_pair_record(cpu, gpu, domain, "explicit_sidecar_cpu_run_dir"))
                    continue
            case = gpu.get("case_id")
            if not case:
                continue
            for cpu in cpu_by_case.get((case, domain), []):
                pairs.append(_pair_record(cpu, gpu, domain, "same_case_domain"))

    summary_counts = defaultdict(int)
    by_domain = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        summary_counts[p["classification"]] += 1
        by_domain[p["domain"]][p["classification"]] += 1

    payload = {
        "schema": "CanaryExistingPairability",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(args.inventory),
        "pair_count": len(pairs),
        "summary_counts": dict(summary_counts),
        "by_domain": {k: dict(v) for k, v in by_domain.items()},
        "pairs": pairs,
    }
    out_dir = args.out_dir
    write_json(out_dir / "pairability.json", payload)
    rows = [
        {
            "classification": p["classification"],
            "domain": p["domain"],
            "case_id": p.get("case_id"),
            "common_lead_count": p["common_lead_count"],
            "same_grid": p["same_grid"],
            "common_score_vars": ",".join(p["common_score_vars"]),
            "cpu_path": p["cpu_path"],
            "gpu_path": p["gpu_path"],
            "reason": p["reason"],
        }
        for p in pairs
    ]
    write_csv(
        out_dir / "pairability.csv",
        rows,
        [
            "classification",
            "domain",
            "case_id",
            "common_lead_count",
            "same_grid",
            "common_score_vars",
            "cpu_path",
            "gpu_path",
            "reason",
        ],
    )
    lines = [
        "# Canary Raw Pairability",
        "",
        f"Generated UTC: {payload['generated_utc']}",
        "",
        "| classification | domain | case | common leads | same grid | vars | CPU | GPU |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for p in pairs:
        lines.append(
            f"| {p['classification']} | {p['domain']} | {p.get('case_id') or '-'} | "
            f"{p['common_lead_count']} | {p['same_grid']} | {','.join(p['common_score_vars']) or '-'} | "
            f"`{p['cpu_path']}` | `{p['gpu_path']}` |"
        )
    (out_dir / "pairability_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pairability pairs={len(pairs)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
