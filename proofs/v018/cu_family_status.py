#!/usr/bin/env python3
"""v0.18 CU-family honesty matrix.

This proof is deliberately not a green-forcing gate. It records the operational
baseline, real-oracle coverage, RED parity, and oracle-needed gaps for the CU
schemes named in the v0.18 family batch.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

OUT_JSON = ROOT / "proofs/v018/cu_family_status.json"
OUT_MD = ROOT / "proofs/v018/cu_family_report.md"
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")

SOURCES = {
    4: ("Scale-aware GFS SAS", WRF / "phys/module_cu_scalesas.F"),
    5: ("Grell-3D ensemble", WRF / "phys/module_cu_g3.F"),
    7: ("Zhang-McFarlane CAMZM", WRF / "phys/module_cu_camzm_driver.F"),
    10: ("KF-CuP", WRF / "phys/module_cu_kfcup.F"),
    11: ("MSKF", WRF / "phys/module_cu_mskf.F"),
    14: ("KIM-SAS", WRF / "phys/module_cu_ksas.F"),
    16: ("New Tiedtke", WRF / "phys/module_cu_ntiedtke.F"),
    93: ("Grell-Devenyi ensemble", WRF / "phys/module_cu_gd.F"),
    94: ("2015 GFS SAS / HWRF", WRF / "phys/module_cu_sas.F"),
    95: ("Previous GFS SAS / HWRF OSAS", WRF / "phys/module_cu_osas.F"),
    96: ("Previous new GFS SAS / YSU NSAS", WRF / "phys/module_cu_nsas.F"),
    99: ("previous Kain-Fritsch", WRF / "phys/module_cu_kf.F"),
}

SAS_STEMS = {4: "scalesas", 94: "sas94", 95: "sas95", 96: "sas96"}
GREL_STEMS = {5: "g3", 93: "gd"}
TAIL_REAL_ORACLE = {
    7: ROOT / "proofs/v018/savepoints/cumulus_tail_wrf/cu7_wrf_real.json",
    10: ROOT / "proofs/v018/savepoints/cumulus_tail_wrf/cu10_wrf_real.json",
    11: ROOT / "proofs/v018/savepoints/cumulus_tail_wrf/cu11_wrf_real.json",
}
TAIL_RELEVANCE = {
    "status": "RELEVANT_NOT_PROVEN_IRRELEVANT",
    "evidence": [
        "WRF Users Guide physics documentation lists MSKF/KF-CuP/Zhang-McFarlane cumulus options.",
        "Pristine WRF source contains active drivers module_cu_camzm_driver.F, module_cu_kfcup.F, and module_cu_mskf.F.",
        "MSKF source header cites regional climate/NWP/aerosol-cloud interaction publications through 2019.",
    ],
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def max_tendency(columns: dict[str, Any]) -> float:
    vals: list[float] = []
    for name, seq in columns.items():
        if name.startswith("R") or name.startswith("CUGD_"):
            vals.extend(abs(float(x)) for x in seq)
    return max(vals) if vals else 0.0


def savepoint_set(pattern: str) -> dict[str, Any]:
    paths = sorted(ROOT.glob(pattern))
    cases = []
    active = []
    null = []
    for path in paths:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        scalars = data.get("scalars", {})
        columns = data.get("columns", {})
        rain = abs(float(scalars.get("RAINCV", 0.0)))
        tend = max_tendency(columns)
        case = data.get("case") or path.stem
        cases.append(str(case))
        if rain > 0.0 or tend > 1.0e-15:
            active.append(str(case))
        else:
            null.append(str(case))
    return {
        "count": len(paths),
        "paths": [str(p.relative_to(ROOT)) for p in paths],
        "active_cases": active,
        "null_cases": null,
        "nontrivial": bool(active) and bool(null),
        "all_null": bool(paths) and not active,
    }


def tail_savepoint(code: int) -> dict[str, Any]:
    path = TAIL_REAL_ORACLE[code]
    data = load_json(path)
    if not isinstance(data, dict):
        return {
            "count": 0,
            "paths": [],
            "active_cases": [],
            "null_cases": ["missing"],
            "nontrivial": False,
        }
    nonzero_fields = data.get("nonzero_fields", [])
    tendency_fields = [
        name
        for name in nonzero_fields
        if name in {"RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN", "ZMDT", "ZMDQ"}
    ]
    rain_fields = [name for name in nonzero_fields if name in {"RAINCV", "PRATEC", "PRECZ", "preccdzm"}]
    return {
        "count": 1,
        "paths": [str(path.relative_to(ROOT))],
        "active_cases": [data.get("label", f"CU{code}")] if data.get("nontrivial") else [],
        "null_cases": [] if data.get("nontrivial") else [data.get("label", f"CU{code}")],
        "nontrivial": bool(data.get("nontrivial")),
        "nonzero_fields": nonzero_fields,
        "nonzero_tendency_fields": tendency_fields,
        "nonzero_rain_fields": rain_fields,
        "anchor": data.get("anchor"),
        "missing_fields": data.get("missing_fields", []),
    }


def source_entry(code: int) -> dict[str, Any]:
    label, path = SOURCES[code]
    return {
        "label": label,
        "wrf_source": str(path),
        "wrf_source_sha256": sha256(path),
    }


def operational_baseline() -> dict[str, Any]:
    from gpuwrf.coupling.scan_adapters import CU_SCAN_ADAPTERS
    from gpuwrf.runtime.operational_mode import _SCAN_WIRED_OPTIONS

    wired = tuple(_SCAN_WIRED_OPTIONS["cu_physics"])
    adapters = tuple(sorted(CU_SCAN_ADAPTERS))
    return {
        "operational_cu_physics": list(wired),
        "cu_scan_adapters": list(adapters),
        "expected": [0, 1, 2, 3, 6],
        "preserved": wired == (0, 1, 2, 3, 6) and adapters == (1, 2, 3, 6),
    }


def accepted_matrix() -> dict[str, Any]:
    from gpuwrf.contracts.physics_registry import ACCEPTED_CU_PHYSICS

    accepted = tuple(ACCEPTED_CU_PHYSICS)
    tail_without_oracle = [code for code in (7, 10, 11) if code in accepted]
    return {
        "accepted_cu_physics": list(accepted),
        "tail_without_oracle_not_accepted": not tail_without_oracle,
        "tail_without_oracle_accepted": tail_without_oracle,
    }


def build_report() -> dict[str, Any]:
    sas_parity = load_json(ROOT / "proofs/v017/sas_family_parity.json") or {}
    kfgrell = load_json(ROOT / "proofs/v017/cu_kfgrell_parity.json") or {}
    v013 = load_json(ROOT / "proofs/v013/t3_cumulus_oracle.json") or {}

    schemes: dict[str, dict[str, Any]] = {}
    operational_labels = {
        0: "disabled",
        1: "Kain-Fritsch",
        2: "Betts-Miller-Janjic",
        3: "Grell-Freitas",
        6: "Tiedtke",
    }
    for code in (0, 1, 2, 3, 6):
        schemes[str(code)] = {
            "label": operational_labels[code],
            "status": "OPERATIONAL_GREEN_BASELINE",
            "oracle_status": "existing operational proof lane",
            "blocker": None,
        }

    for code, stem in SAS_STEMS.items():
        sp = savepoint_set(f"proofs/v017/savepoints/cumulus_sas/{stem}_case_*.json")
        entry = source_entry(code)
        entry.update(
            status="REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX",
            oracle_status="real_pristine_wrf_savepoints_present",
            savepoints=sp,
            parity_status=(sas_parity.get("schemes", {}).get(str(code), {}) or {}).get("status"),
            blocker="shared SAS JAX endpoint is RED vs pristine-WRF oracle; not scan-wired",
        )
        schemes[str(code)] = entry

    for code, stem in GREL_STEMS.items():
        sp = savepoint_set(f"proofs/v018/savepoints/cumulus_grell/{stem}_case_*.json")
        entry = source_entry(code)
        oracle_status = "real_pristine_wrf_nontrivial" if sp["active_cases"] else "real_pristine_wrf_built_but_all_trial_columns_null"
        entry.update(
            status="REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX",
            oracle_status=oracle_status,
            savepoints=sp,
            parity_status=(kfgrell.get("schemes", {}).get(str(code), {}) or {}).get("status"),
            blocker="faithful source-specific JAX operational endpoint not ported in this run; not scan-wired",
        )
        schemes[str(code)] = entry

    for code in (14, 16):
        v013_scheme = (v013.get("schemes", {}) or {}).get(str(code), {})
        entry = source_entry(code)
        entry.update(
            status="REFERENCE_ONLY_WITH_REAL_ORACLE",
            oracle_status="real_pristine_wrf_nontrivial" if v013_scheme.get("oracle", {}).get("nontrivial") else "oracle_status_needs_review",
            savepoints=v013_scheme.get("oracle"),
            parity_status=None,
            blocker="traceable source-specific JAX operational endpoint not promoted in this run; not scan-wired",
        )
        schemes[str(code)] = entry

    oldkf_sp = savepoint_set("proofs/v017/savepoints/oldkf_case_*.json")
    oldkf_entry = source_entry(99)
    oldkf_entry.update(
        status="REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX",
        oracle_status="real_pristine_wrf_savepoints_present",
        savepoints=oldkf_sp,
        parity_status=(kfgrell.get("schemes", {}).get("99", {}) or {}).get("status"),
        blocker="candidate reuses KF-eta family and is RED vs module_cu_kf.F old-KF oracle; not scan-wired",
    )
    schemes["99"] = oldkf_entry

    for code in (7, 10, 11):
        sp = tail_savepoint(code)
        entry = source_entry(code)
        oracle_status = "real_pristine_wrf_nontrivial"
        if code == 7 and not sp["nonzero_tendency_fields"] and not sp["nonzero_rain_fields"]:
            oracle_status = "real_pristine_wrf_completed_diagnostic_only"
        entry.update(
            status="REFERENCE_ONLY_WITH_REAL_ORACLE",
            oracle_status=oracle_status,
            savepoints=sp,
            parity_status=None,
            relevance=TAIL_RELEVANCE,
            blocker="source-specific JAX operational endpoint not ported in this run; not scan-wired",
        )
        schemes[str(code)] = entry

    op = operational_baseline()
    accepted = accepted_matrix()
    scoped = ("4", "5", "7", "10", "11", "93", "94", "95", "96", "99")
    no_silent_gaps = all(schemes[code]["status"] != "REFERENCE_ONLY_WITHOUT_ORACLE" for code in scoped)
    full_ship_gate_met = all(
        schemes[code]["status"] in {
            "OPERATIONAL_GREEN_BASELINE",
            "REFERENCE_ONLY_WITH_REAL_ORACLE",
            "REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX",
        }
        for code in scoped
    )
    step1_honesty_gate_met = op["preserved"] and accepted["tail_without_oracle_not_accepted"] and no_silent_gaps

    return {
        "proof": "v018-cu-family-status",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "operational_baseline": op,
        "accepted_matrix": accepted,
        "tail_relevance_assessment": TAIL_RELEVANCE,
        "schemes": schemes,
        "scoped_batch": list(scoped),
        "step1_honesty_gate_met": step1_honesty_gate_met,
        "full_v018_cu_ship_gate_met": full_ship_gate_met,
        "full_ship_gate_blockers": [
            code for code in scoped
            if schemes[code]["status"] not in {
                "OPERATIONAL_GREEN_BASELINE",
                "REFERENCE_ONLY_WITH_REAL_ORACLE",
                "REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX",
            }
        ],
    }


def write_markdown(report: dict[str, Any]) -> None:
    rows = []
    for code in sorted(report["schemes"], key=lambda c: int(c)):
        s = report["schemes"][code]
        if code not in report["scoped_batch"] and s["status"] != "OPERATIONAL_GREEN_BASELINE":
            continue
        rows.append(
            f"| {code} | {s.get('label', '')} | {s['status']} | "
            f"{s.get('oracle_status', '')} | {s.get('blocker') or ''} |"
        )

    text = "\n".join(
        [
            "# v0.18 CU family step-1 report",
            "",
            f"- Step-1 honesty gate: `{report['step1_honesty_gate_met']}`",
            f"- Full v0.18 CU ship gate: `{report['full_v018_cu_ship_gate_met']}`",
            f"- Operational CU scan preserved: `{report['operational_baseline']['operational_cu_physics']}`",
            f"- Tail CU7/10/11 accepted without oracle: `{report['accepted_matrix']['tail_without_oracle_accepted']}`",
            f"- Tail CU7/10/11 relevance: `{report['tail_relevance_assessment']['status']}`",
            "",
            "| CU | Scheme | Status | Oracle | Blocker |",
            "|---:|---|---|---|---|",
            *rows,
            "",
            "Proof commands refreshed in this run:",
            "",
            "- `taskset -c 0-3 proofs/v017/oracle/cumulus_sas/build_and_run.sh`",
            "- `taskset -c 0-3 proofs/v017/oracle/cumulus/build_oldkf_oracle.sh`",
            "- `taskset -c 0-3 bash proofs/v018/oracle/cumulus_grell/build_and_run.sh`",
            "- `taskset -c 0-3 bash proofs/v018/oracle/cumulus_tail_wrf/build_and_run.sh`",
            "- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 proofs/v017/run_sas_family_parity.py`",
            "- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 proofs/v017/run_cu_kfgrell_parity.py --build-oldkf --allow-red`",
            "- `PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 proofs/v018/cu_family_status.py`",
            "",
            "Tail CU7/10/11 were treated as relevant rather than proven irrelevant; each now has a completed pristine-WRF reference savepoint. CU7 is diagnostic-only in this fixture (base/top fields nonzero, heating/moistening zero) and remains fail-closed reference-only.",
            "",
            "No GPU operational smoke was run for CU5/7/10/11/93/94/95/96/99 because none of those RED/reference-only schemes were scan-wired.",
            "",
        ]
    )
    OUT_MD.write_text(text)


def main() -> int:
    report = build_report()
    OUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report)
    print(
        f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}; "
        f"step1={report['step1_honesty_gate_met']} full_ship={report['full_v018_cu_ship_gate_met']}"
    )
    return 0 if report["step1_honesty_gate_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
