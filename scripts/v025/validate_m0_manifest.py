#!/usr/bin/env python3
"""Validate PROOF_MANIFEST.json: hashes, producing commands, gate classification (§12).

This is the validator the critic runs first. It re-hashes every listed artifact
and refuses a manifest whose recorded hash no longer matches the file on disk,
which is what stops a proof object from drifting away from the evidence it was
written against.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_lib import FAIL, MISSING, NOT_APPLICABLE, PASS, Report, load_json

VALID_GATE_STATUS = {PASS, FAIL, MISSING, NOT_APPLICABLE}

REQUIRED_OBJECTS = [
    "control_manifest.json", "case_manifest.json", "fast_case_qualification.json",
    "cpu_baseline_reconciliation.json", "efficiency_model.json", "kernel_census.json",
    "hlo_dtype_transfer_census.json", "cancellation_map.json",
    "pallas_sm120_viability.json", "environment_manifest.json",
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    rep = Report("PROOF_MANIFEST", args.path)
    obj = load_json(args.path, rep)
    if obj is None:
        return rep.emit()

    artifacts = obj.get("artifacts") or []
    if not artifacts:
        rep.missing("artifacts", "§12 requires every committed object and raw artifact hashed")
    for art in artifacts:
        name = art.get("path", "?")
        recorded = art.get("sha256")
        if not recorded:
            rep.missing(f"artifact:{name}:sha256", "no hash recorded")
            continue
        target = Path(name)
        if not target.is_absolute():
            target = args.path.resolve().parents[3] / name
        if not target.exists():
            rep.missing(f"artifact:{name}", f"listed but absent at {target}")
            continue
        actual = sha256(target)
        rep.require(f"artifact:{name}:hash_matches", actual == recorded,
                    "" if actual == recorded else f"recorded {recorded[:16]}… actual {actual[:16]}…")
        if art.get("producing_command") is None:
            rep.missing(f"artifact:{name}:producing_command", "§12 requires the exact command")
        if art.get("returncode") is None:
            rep.missing(f"artifact:{name}:returncode", "§12 requires the return code")

    gates = obj.get("gates") or {}
    if not gates:
        rep.missing("gates", "§12 requires each contract gate classified")
    for gate, entry in gates.items():
        status = entry.get("status") if isinstance(entry, dict) else entry
        rep.require(f"gate:{gate}:status_valid", status in VALID_GATE_STATUS,
                    f"{status!r} not in {sorted(VALID_GATE_STATUS)}")

    listed = {Path(a.get("path", "")).name for a in artifacts}
    for name in REQUIRED_OBJECTS:
        if name not in listed:
            rep.missing(f"required_object:{name}", "listed in §12 but absent from the manifest")

    # "Missing evidence is never PASS" -- enforce it on the manifest itself.
    for gate, entry in gates.items():
        if isinstance(entry, dict) and entry.get("status") == PASS and not entry.get("evidence"):
            rep.require(f"gate:{gate}:pass_has_evidence", False,
                        "a gate marked PASS must cite its evidence artifact")
    return rep.emit()

if __name__ == "__main__":
    raise SystemExit(main())
