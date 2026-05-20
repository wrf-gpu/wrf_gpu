#!/usr/bin/env python3
"""Evaluate the M5-S2 stop/go gate for MYNN."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m5"
MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-mynn-pbl-column-v1.yaml"


def _load(path: Path) -> dict:
    """Reads one generated proof-object JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _tolerance_regime() -> str:
    """Classifies whether Tier-1 pass is strict or carry-forward."""

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for variable in manifest["variables"]:
        rationale = str(variable.get("tolerance_rationale", "")).lower()
        if str(variable["name"]).startswith("output_") and "carry-forward" in rationale:
            return "carry-forward"
    return "strict"


def evaluate_gate() -> dict:
    """Builds the MYNN gate result from validation/profile artifacts."""

    tier1 = _load(ART / "tier1_mynn_parity.json")
    tier2 = _load(ART / "tier2_mynn_invariants.json")
    profile = _load(ART / "mynn_profile.json")
    launches = int(profile.get("kernel_launches_per_step") or profile.get("kernel_launches") or 0)
    local = profile.get("local_memory_bytes_per_kernel", profile.get("local_memory_bytes"))
    registers = profile.get("registers_per_kernel", profile.get("registers_per_thread"))
    hlo_bytes = int(profile.get("hlo_production_bytes", 0))
    tier1_pass = bool(tier1.get("pass"))
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
        reasons.append(f"{launches} launches exceeds M5-S2 acceptable threshold 5")
    if hlo_bytes > 300_000 and status != "FALLBACK":
        status = "GRAY-ZONE"
        reasons.append(f"HLO size {hlo_bytes} exceeds 300 KB")
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
    if status == "GO" and tolerance_regime == "carry-forward":
        status = "GO_CARRYFORWARD"
        reasons.append("tier-1/tier-2 pass under carry-forward source-derived harness tolerances; exact WRF object path is absent")
    elif status == "GO":
        reasons.append("tier-1/tier-2 pass under strict tolerances and launch/HLO limits")
    return {
        "kernel_launches_per_step": launches,
        "local_memory_bytes_per_kernel": local,
        "registers_per_kernel": registers,
        "hlo_production_bytes": hlo_bytes,
        "tier1_pass": tier1_pass,
        "tier2_pass": tier2_pass,
        "tolerance_regime": tolerance_regime,
        "gate_status": status,
        "rationale": "; ".join(reasons)[:300],
    }


def main() -> int:
    """Writes `mynn_gate_result.json` and exits nonzero only on FALLBACK."""

    record = evaluate_gate()
    out = ART / "mynn_gate_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["gate_status"] in {"GO", "GO_CARRYFORWARD", "GRAY-ZONE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
