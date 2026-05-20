#!/usr/bin/env python3
"""Evaluate the M5-S3 stop/go gate for RRTMG."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m5"
SW_MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-rrtmg-sw-column-v1.yaml"
LW_MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-rrtmg-lw-column-v1.yaml"


def _load(path: Path) -> dict:
    """Reads one generated proof-object JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _tolerance_regime() -> str:
    """Classifies whether Tier-1 pass is strict or carry-forward."""

    for path in (SW_MANIFEST, LW_MANIFEST):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        for variable in manifest["variables"]:
            rationale = str(variable.get("tolerance_rationale", "")).lower()
            if str(variable["name"]).startswith("output_") and "carry-forward" in rationale:
                return "carry-forward"
    return "strict"


def _oracle_regime() -> str:
    """Classifies whether the manifest documents the full WRF-driver gap."""

    text = SW_MANIFEST.read_text(encoding="utf-8") + LW_MANIFEST.read_text(encoding="utf-8")
    if "full_rrtmg_driver_call=deferred" in text:
        return "linked-source-derived"
    return "wrf-driver"


def evaluate_gate() -> dict:
    """Builds the RRTMG gate result from validation/profile artifacts."""

    tier1_sw = _load(ART / "tier1_rrtmg_sw_parity.json")
    tier1_lw = _load(ART / "tier1_rrtmg_lw_parity.json")
    tier2 = _load(ART / "tier2_rrtmg_invariants.json")
    profile = _load(ART / "rrtmg_profile.json")
    launches = int(profile.get("kernel_launches_per_step") or profile.get("kernel_launches") or 0)
    raw_launches = int(profile.get("raw_hlo_launch_marker_count") or launches)
    local = profile.get("local_memory_bytes_per_kernel", profile.get("local_memory_bytes"))
    registers = profile.get("registers_per_kernel", profile.get("registers_per_thread"))
    sw_hlo_bytes = int(profile.get("hlo_production_bytes_sw", 0))
    lw_hlo_bytes = int(profile.get("hlo_production_bytes_lw", 0))
    tier1_pass = bool(tier1_sw.get("pass") and tier1_lw.get("pass"))
    tier2_pass = bool(tier2.get("pass"))
    status = "GO"
    reasons = []
    if not tier1_pass or not tier2_pass:
        status = "FALLBACK"
        reasons.append("correctness failed")
    if launches > 50:
        status = "FALLBACK"
        reasons.append(f"{launches} launches exceeds fallback threshold 50")
    elif launches > 5 and status != "FALLBACK":
        status = "GRAY-ZONE"
        reasons.append(f"{launches} launches exceeds M5-S3 acceptable threshold 5")
    if max(sw_hlo_bytes, lw_hlo_bytes) > 500_000 and status != "FALLBACK":
        status = "GRAY-ZONE"
        reasons.append(f"HLO size {max(sw_hlo_bytes, lw_hlo_bytes)} exceeds 500 KB")
    if registers is not None:
        if int(registers) > 200:
            status = "FALLBACK"
            reasons.append(f"{registers} registers exceeds fallback threshold 200")
        elif int(registers) > 128 and status == "GO":
            status = "GRAY-ZONE"
            reasons.append(f"{registers} registers exceeds GO threshold 128")
    if local is not None:
        if int(local) > 512:
            status = "FALLBACK"
            reasons.append(f"{local} B local memory exceeds fallback threshold 512")
        elif int(local) > 256 and status == "GO":
            status = "GRAY-ZONE"
            reasons.append(f"{local} B local memory exceeds GO threshold 256")
    tolerance_regime = _tolerance_regime()
    oracle_regime = _oracle_regime()
    if status == "GO" and (tolerance_regime == "carry-forward" or oracle_regime != "wrf-driver"):
        status = "GO_CARRYFORWARD"
        reasons.append("tier-1/tier-2 pass under compact linked source-derived harness; full WRF RRTMG driver oracle remains deferred")
    elif status == "GO":
        reasons.append("tier-1/tier-2 pass under strict tolerances and launch/HLO limits")
    return {
        "kernel_launches_per_step": launches,
        "raw_hlo_launch_marker_count": raw_launches,
        "local_memory_bytes_per_kernel": local,
        "registers_per_kernel": registers,
        "hlo_production_bytes_sw": sw_hlo_bytes,
        "hlo_production_bytes_lw": lw_hlo_bytes,
        "tier1_sw_pass": bool(tier1_sw.get("pass")),
        "tier1_lw_pass": bool(tier1_lw.get("pass")),
        "tier2_pass": tier2_pass,
        "tolerance_regime": tolerance_regime,
        "oracle_regime": oracle_regime,
        "gate_status": status,
        "rationale": "; ".join(reasons)[:400],
    }


def main() -> int:
    """Writes `rrtmg_gate_result.json` and exits nonzero only on FALLBACK."""

    record = evaluate_gate()
    out = ART / "rrtmg_gate_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["gate_status"] in {"GO", "GO_CARRYFORWARD", "GRAY-ZONE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
