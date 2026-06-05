#!/usr/bin/env python
"""Aggregate RRTMG finite recheck mode proofs into the endpoint JSON/MD."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ON = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck.json"
DEFAULT_TOPO_OFF = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck_topo_off.json"
DEFAULT_RRTMG_OFF = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck_rrtmg_off.json"
DEFAULT_OUT_JSON = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck.json"
DEFAULT_OUT_MD = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck.md"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _finite_fields(proof: Mapping[str, Any]) -> Mapping[str, Any]:
    return (proof.get("all_finite_check") or {}).get("fields", {})


def _nonfinite_signature(proof: Mapping[str, Any]) -> dict[str, int]:
    fields = _finite_fields(proof)
    out: dict[str, int] = {}
    for name, rec in fields.items():
        if not bool(rec.get("finite", True)):
            out[name] = int(rec.get("nonfinite_count", 0))
    return out


def _field_summary(proof: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    fields = _finite_fields(proof)
    return {
        name: {
            "finite": bool(fields.get(name, {}).get("finite", False)),
            "nonfinite_count": int(fields.get(name, {}).get("nonfinite_count", -1)),
            "min": fields.get(name, {}).get("min"),
            "max": fields.get(name, {}).get("max"),
            "dtype": fields.get(name, {}).get("dtype"),
            "shape": fields.get(name, {}).get("shape"),
        }
        for name in names
        if name in fields
    }


def _mode_record(path: Path, proof: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "mode": proof.get("mode"),
        "mode_note": proof.get("mode_note"),
        "status": proof.get("status"),
        "pipeline_verdict": proof.get("pipeline_verdict"),
        "proper_cadence_finite": bool(proof.get("proper_cadence_finite")),
        "topo_shading": proof.get("topo_shading"),
        "slope_rad": proof.get("slope_rad"),
        "radiation_cadence_steps": proof.get("radiation_cadence_steps"),
        "nonfinite_signature": _nonfinite_signature(proof),
        "key_fields": _field_summary(proof, ("theta", "u", "v", "w", "p", "ph", "mu", "qv", "qke")),
        "reason": (proof.get("raw_pipeline_payload") or {}).get("reason"),
        "wrfout_files": proof.get("wrfout_files", []),
    }


def _same_signature(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _nonfinite_signature(left) == _nonfinite_signature(right)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", type=Path, default=DEFAULT_ON)
    ap.add_argument("--topo-off", type=Path, default=DEFAULT_TOPO_OFF)
    ap.add_argument("--rrtmg-off", type=Path, default=DEFAULT_RRTMG_OFF)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    on = _read(args.on)
    topo_off = _read(args.topo_off)
    rrtmg_off = _read(args.rrtmg_off)
    on_sig = _nonfinite_signature(on)
    topo_same = _same_signature(on, topo_off)
    rrtmg_same = _same_signature(on, rrtmg_off)
    theta_uv_finite = all(
        _finite_fields(on).get(name, {}).get("finite") is True for name in ("theta", "u", "v")
    )

    if bool(on.get("proper_cadence_finite")):
        status = "PASS"
        verdict = "KEEP_RRTMG_ON_TRUNK"
        interpretation = (
            "The operational segmented cadence is all-field finite with RRTMG slope/shading on."
        )
    elif on_sig == {"qke": 2024} and topo_same and rrtmg_same and theta_uv_finite:
        status = "DIAGNOSED_KNOWN_QKE_EDGE"
        verdict = "KEEP_RRTMG_FEATURE_DO_NOT_MASK_QKE_KI2"
        interpretation = (
            "The proper segmented cadence does not reproduce the RRTMG lane's cold one-step "
            "theta/u/v nonfinite: theta/u/v are finite with zero nonfinite values. The only "
            "nonfinite field is qke (2024 cells), and the exact qke signature persists with "
            "topo_shading=0/slope_rad=0 and with RRTMG suppressed entirely. This matches the "
            "documented KI-2 20260521 d02 production-path qke edge, so the RRTMG slope/shading "
            "feature is not the offending term. No masking was applied."
        )
    elif on_sig and rrtmg_same:
        status = "DIAGNOSED_NOT_RRTMG_SPECIFIC"
        verdict = "RRTMG_NOT_OFFENDING_BUT_STATE_NOT_ALL_FIELD_FINITE"
        interpretation = (
            "The RRTMG-on nonfinite signature also appears when RRTMG is suppressed. The feature "
            "is not isolated as the offending term, but the run is not all-field finite."
        )
    else:
        status = "FAIL_RRTMG_REGRESSION_SUSPECTED"
        verdict = "ESCALATE_TO_OPUS_SECOND_LINE_DEBUG"
        interpretation = (
            "The RRTMG-on signature differs from the RRTMG-off control. Treat this as a real "
            "radiation-lane regression and continue term-level localization."
        )

    aggregate = {
        "schema": "V0110RRTMGFiniteRecheckAggregate",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "verdict": verdict,
        "interpretation": interpretation,
        "proper_cadence_all_state_finite": bool(on.get("proper_cadence_finite")),
        "proper_cadence_theta_u_v_finite": bool(theta_uv_finite),
        "rrtmg_feature_offending_term": None if rrtmg_same else "radiation-dependent nonfinite signature",
        "nonfinite_signature_on": on_sig,
        "topo_off_same_signature": bool(topo_same),
        "rrtmg_off_same_signature": bool(rrtmg_same),
        "run_id": on.get("run_id"),
        "run_dir": on.get("run_dir"),
        "domain": on.get("domain"),
        "hours": on.get("hours"),
        "forecast_fn": on.get("forecast_fn"),
        "segment_steps": on.get("segment_steps"),
        "radiation_cadence_steps_on": on.get("radiation_cadence_steps"),
        "mode_results": {
            "on": _mode_record(args.on, on),
            "topo_off": _mode_record(args.topo_off, topo_off),
            "rrtmg_off": _mode_record(args.rrtmg_off, rrtmg_off),
        },
        "cold_one_step_context": on.get("cold_one_step_context"),
        "commands": [
            *(on.get("commands") or []),
            *(topo_off.get("commands") or []),
            *(rrtmg_off.get("commands") or []),
            "python proofs/v0110/rrtmg_finite_recheck_aggregate.py",
        ],
        "known_issue_context": {
            "document": "docs/KNOWN_ISSUES.md#KI-2",
            "matching_signature": "20260521 d02 production-path qke nonfinite after forecast hour 1 with 2024 qke cells",
        },
    }

    _write(args.out_json, json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    lines = [
        "# v0.11.0 RRTMG finite recheck",
        "",
        f"- status: {status}",
        f"- verdict: {verdict}",
        f"- all-state finite with RRTMG on: {aggregate['proper_cadence_all_state_finite']}",
        f"- theta/u/v finite with RRTMG on: {aggregate['proper_cadence_theta_u_v_finite']}",
        f"- RRTMG-on nonfinite signature: {on_sig}",
        f"- topo-off same signature: {topo_same}",
        f"- RRTMG-off same signature: {rrtmg_same}",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Mode Results",
        "",
        "| mode | topo | slope | cadence | pipeline | nonfinite signature |",
        "|---|---:|---:|---:|---|---|",
    ]
    for mode, rec in aggregate["mode_results"].items():
        lines.append(
            f"| {mode} | {rec.get('topo_shading')} | {rec.get('slope_rad')} | "
            f"{rec.get('radiation_cadence_steps')} | {rec.get('pipeline_verdict')} | "
            f"{rec.get('nonfinite_signature')} |"
        )
    lines.extend(["", "## Commands", ""])
    for command in aggregate["commands"]:
        lines.append(f"- `{command}`")
    _write(args.out_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": status, "verdict": verdict, "out_json": str(args.out_json)}, indent=2))
    return 0 if status != "FAIL_RRTMG_REGRESSION_SUSPECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
