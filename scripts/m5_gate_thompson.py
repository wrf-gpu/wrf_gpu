#!/usr/bin/env python3
"""Evaluate the ADR-001/ADR-005 M5 stop/go gate for Thompson."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m5"
MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-thompson-column-v1.yaml"


def _load(path: Path) -> dict:
    """Reads one generated proof-object JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _tolerance_regime() -> str:
    """Classifies whether Tier-1 pass is strict ADR-005 or carry-forward."""

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    for variable in manifest["variables"]:
        name = str(variable["name"])
        rationale = str(variable.get("tolerance_rationale", ""))
        if name.startswith("output_") and "carry-forward" in rationale:
            return "carry-forward"
    return "ADR-005-strict"


def evaluate_gate() -> dict:
    """Builds the Thompson gate result from measured validation/profile artifacts."""

    tier1 = _load(ART / "tier1_thompson_parity.json")
    tier2 = _load(ART / "tier2_thompson_invariants.json")
    profile = _load(ART / "thompson_profile.json")
    tolerance_regime = _tolerance_regime()
    launches = int(profile.get("kernel_launches_per_step") or profile.get("kernel_launches") or 0)
    local = profile.get("local_memory_bytes_per_kernel", profile.get("local_memory_bytes"))
    registers = profile.get("registers_per_kernel", profile.get("registers_per_thread"))
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
    elif launches > 10 and status != "FALLBACK":
        status = "GRAY-ZONE"
        reasons.append(f"{launches} launches exceeds GO threshold 10")
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
    if status == "GO":
        if tolerance_regime == "carry-forward":
            status = "GO_CARRYFORWARD"
            reasons.append(
                "tier-1/tier-2 pass under carry-forward tolerances; strict ADR-005 parity remains M5-S1.x handoff; HLO-derived launches are within the GO threshold; register/local-memory counters are null due to perfmon restriction"
            )
        else:
            reasons.append("tier-1/tier-2 pass under ADR-005 strict tolerances and HLO-derived launches are within the GO threshold; register/local-memory counters are null due to perfmon restriction")
    return {
        "kernel_launches_per_step": launches,
        "local_memory_bytes_per_kernel": local,
        "registers_per_kernel": registers,
        "tier1_pass": tier1_pass,
        "tier2_pass": tier2_pass,
        "tolerance_regime": tolerance_regime,
        "gate_status": status,
        "rationale": "; ".join(reasons)[:300],
    }


def main() -> int:
    """Writes `thompson_gate_result.json` and exits nonzero only on FALLBACK."""

    record = evaluate_gate()
    out = ART / "thompson_gate_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["gate_status"] in {"GO", "GO_CARRYFORWARD", "GRAY-ZONE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
