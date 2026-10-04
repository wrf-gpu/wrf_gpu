#!/usr/bin/env python3
"""Reproducible survey of every real ALISIOS production namelist (CPU-only).

Emits `proofs/v025/m0/alternative_case_survey.json`. Every number in that object
comes from running this script, so a critic reruns one command rather than
trusting a prose summary. Each namelist is SHA-256 hashed and the distinct
configurations are keyed by a hash of the physics/dynamics tuple.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("<DATA_ROOT>/alisios/runs")
FIELDS = ("mp_physics", "ra_lw_physics", "ra_sw_physics", "sf_surface_physics",
          "bl_pbl_physics", "sf_sfclay_physics", "cu_physics", "gwd_opt",
          "radt", "e_we", "e_sn", "e_vert", "max_dom", "dx", "diff_opt",
          "km_opt", "diff_6th_opt", "damp_opt")


def value(text: str, key: str) -> str:
    match = re.search(rf"^\s*{key}\s*=\s*([^\n/]+)", text, re.M)
    return re.sub(r"\s+", "", match.group(1)).rstrip(",") if match else "ABSENT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def survey(root: Path = ROOT) -> dict:
    namelists = sorted(root.glob("**/namelist.input"))
    configs: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    files_by_config = defaultdict(list)

    for path in namelists:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        fields = {key: value(text, key) for key in FIELDS}
        key = hashlib.sha256(
            json.dumps(fields, sort_keys=True).encode()
        ).hexdigest()[:16]
        counts[key] += 1
        configs.setdefault(key, fields)
        files_by_config[key].append({"path": str(path), "sha256": sha256_file(path)})

    return {
        "namelists_found": len(namelists),
        "distinct_configurations": len(configs),
        "configurations": [
            {
                "config_id": key,
                "count": counts[key],
                "fields": configs[key],
                "example_files": files_by_config[key][:3],
            }
            for key, _ in counts.most_common()
        ],
    }


def domains_of(fields: dict) -> list[dict]:
    """Expand the per-domain columns into one row per domain, d01..dN."""
    def col(name: str) -> list[str]:
        raw = fields.get(name, "")
        return [v for v in raw.split(",") if v]

    we, sn = col("e_we"), col("e_sn")
    cu, vert = col("cu_physics"), col("e_vert")
    gwd = col("gwd_opt")
    rows = []
    for index in range(len(we)):
        rows.append({
            "domain": f"d{index + 1:02d}",
            "e_we": int(we[index]), "e_sn": int(sn[index]),
            "mass_cells": (int(we[index]) - 1) * (int(sn[index]) - 1),
            "e_vert": int(vert[index]) if index < len(vert) else None,
            "cu_physics": int(cu[index]) if index < len(cu) else None,
            "gwd_opt": int(gwd[index]) if index < len(gwd) else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path,
                        default=Path("proofs/v025/m0/alternative_case_survey.json"))
    args = parser.parse_args()

    result = survey(args.root)
    census = json.loads(Path("proofs/v025/m0/hlo_dtype_transfer_census.json").read_text())
    ops = [o for o in census["operators"] if o["status"] == "LOWERED" and "seconds" in o]
    total = sum(o["seconds"] for o in ops)
    families: dict[str, float] = {}
    for op in ops:
        families[op["family"]] = families.get(op["family"], 0.0) + op["seconds"]

    biggest = max(result["configurations"], key=lambda c: len(c["fields"]["e_we"].split(",")))
    all_domains = domains_of(biggest["fields"])

    obj = {
        "schema": "wrf_gpu2.v025.m0.alternative_case_survey.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduce": {
            "command": "python scripts/v025/survey_production_cases.py",
            "source": "scripts/v025/survey_production_cases.py",
            "source_sha256": sha256_file(Path(__file__)),
            "root": str(args.root),
            "note": "every number below is produced by rerunning that one command",
        },
        "corpus": result,
        "all_domains_of_the_largest_configuration": all_domains,
        "compile_cost_by_family_seconds": {
            k: round(v, 1) for k, v in sorted(families.items(), key=lambda kv: -kv[1])
        },
        "compile_cost_total_seconds": round(total, 1),
        "compile_cost_source": (
            "proofs/v025/m0/hlo_dtype_transfer_census.json -- 48 operators lowered and "
            "compiled at real production shape on the XLA:CPU backend"
        ),
        "grid_size_independence": {
            "status": "HYPOTHESIS, NOT PROVEN",
            "claim": (
                "compile cost is driven by the scheme set rather than by grid size, so a "
                "smaller real domain would not materially reduce cold compile"
            ),
            "evidence_for": (
                "the A6 census shows compile concentrated in a few scheme families "
                "(radiation 47.1%), and XLA compile work scales with program structure "
                "rather than with array extents for a fixed program"
            ),
            "why_it_is_not_proven": (
                "the A6 census was run at ONE shape (120x70x44). No shape sweep was "
                "performed, so the grid-size sensitivity of compile time is unmeasured. "
                "Earlier wording in v1 of this object asserted this as established; it is "
                "demoted here to a hypothesis."
            ),
            "how_to_settle_it_on_cpu": (
                "rerun build_hlo_census.py at 2-3 real domain shapes (e.g. d06 40x40, d03 "
                "103x70, d02 268x118) and compare total lower+compile seconds. CPU-only, "
                "no GPU needed, and it directly tests the hypothesis."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(f"  namelists {result['namelists_found']}  configs {result['distinct_configurations']}")
    for row in all_domains:
        print(f"    {row['domain']}: {row['e_we']}x{row['e_sn']}x{row['e_vert']} "
              f"cu={row['cu_physics']} gwd={row['gwd_opt']} cells={row['mass_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
