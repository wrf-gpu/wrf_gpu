#!/usr/bin/env python3
"""Reduce the v0.20.0 allocator A/B run into ms/fc-h + speedup + fit census.

CPU-ONLY; reads each arm's ``proofs/nested_pipeline_run.json`` (written by the
nested pipeline) and derives, per arm:

  * ``ms_per_fc_h`` = 1000 * wall_clock_forecast_only_s / hours  (warm steady)
  * the OOM-fit verdict (rc + any CUDA-OOM strings the harness flagged)
  * the d01..d09 finite + output-present census (the MUST-NOT-REGRESS gate)

and, when both arms ran, the lever speedup ``A_off / A_on`` (platform / cuda_async).

This is a pure post-processor: no GPU, no JAX. It is exercised in the harness's
CPU dry-run against an existing nested proof JSON so the reduction logic is real.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_proof(arm_dir: Path) -> dict[str, Any] | None:
    """Return the nested proof JSON dict for an arm dir, or None if absent."""
    p = arm_dir / "proofs" / "nested_pipeline_run.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _finite_present_census(proof: dict[str, Any]) -> dict[str, Any]:
    """Per-domain finite + output-present census from a nested proof payload."""
    per_domain = proof.get("per_domain") or {}
    census: dict[str, Any] = {}
    all_ok = True
    for name, info in per_domain.items():
        finite = bool(info.get("all_finite", info.get("finite", False)))
        present = bool(info.get("output_present", info.get("has_output", False)))
        census[name] = {"finite": finite, "output_present": present}
        all_ok = all_ok and finite and present
    return {
        "per_domain": census,
        "all_domains_finite": bool(proof.get("all_domains_finite", all_ok)),
        "all_outputs_present": bool(proof.get("all_outputs_present", all_ok)),
    }


def _arm_summary(run_dir: Path, arm: str, hours: int) -> dict[str, Any]:
    arm_dir = run_dir / arm
    rc = _read_text(arm_dir / "rc.txt")
    wall_s = _read_text(arm_dir / "wall_s.txt")
    oom = _read_text(arm_dir / "oom.txt")
    allocator = _read_text(arm_dir / "allocator.txt")
    out: dict[str, Any] = {
        "arm": arm,
        "allocator": allocator or ("cuda_async" if arm == "on" else "platform"),
        "harness_rc": int(rc) if rc.isdigit() else None,
        "harness_wall_s": float(wall_s) if wall_s else None,
        "oom_flag": oom,
        "ms_per_fc_h": None,
        "forecast_only_s": None,
        "fit_ok": None,
    }
    proof = _load_proof(arm_dir)
    if proof is None:
        out["note"] = "no nested_pipeline_run.json (arm not run, or run failed)"
        return out
    fc_s = proof.get("wall_clock_forecast_only_s")
    proof_hours = proof.get("hours", hours) or hours
    if isinstance(fc_s, (int, float)) and proof_hours:
        out["forecast_only_s"] = float(fc_s)
        out["ms_per_fc_h"] = 1000.0 * float(fc_s) / float(proof_hours)
    census = _finite_present_census(proof)
    out["census"] = census
    # The lever's MUST-NOT-REGRESS gate: ran clean (rc 0), no OOM strings, every
    # domain finite + present.
    out["fit_ok"] = bool(
        out["harness_rc"] == 0
        and oom != "OOM_DETECTED"
        and census["all_domains_finite"]
        and census["all_outputs_present"]
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--arms", default="on off")
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument(
        "--dry-fallback-json",
        type=Path,
        default=None,
        help="In CPU dry-run, an existing nested_pipeline_run.json to exercise the reducer.",
    )
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    arms = args.arms.split()
    result: dict[str, Any] = {
        "schema": "V020AllocatorAB",
        "schema_version": 1,
        "lever": "G_allocator_env: nested allocator default cuda_async vs platform",
        "numerics_free": True,
        "hours": int(args.hours),
        "arms": {},
    }

    # Dry-run path: no per-arm proofs exist; exercise the reducer on a real proof.
    have_any_proof = any(
        (args.run_dir / a / "proofs" / "nested_pipeline_run.json").is_file() for a in arms
    )
    if not have_any_proof and args.dry_fallback_json and args.dry_fallback_json.is_file():
        proof = json.loads(args.dry_fallback_json.read_text())
        fc_s = proof.get("wall_clock_forecast_only_s")
        h = proof.get("hours", args.hours) or args.hours
        result["dry_run_reducer_check"] = {
            "source": str(args.dry_fallback_json),
            "ms_per_fc_h": (1000.0 * float(fc_s) / float(h)) if isinstance(fc_s, (int, float)) and h else None,
            "census": _finite_present_census(proof),
        }
        result["note"] = "DRY-RUN: no per-arm GPU output; reducer validated on an existing proof."
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    for arm in arms:
        result["arms"][arm] = _arm_summary(args.run_dir, arm, args.hours)

    on = result["arms"].get("on", {})
    off = result["arms"].get("off", {})
    on_ms = on.get("ms_per_fc_h")
    off_ms = off.get("ms_per_fc_h")
    if isinstance(on_ms, (int, float)) and isinstance(off_ms, (int, float)) and on_ms > 0:
        result["speedup_lever_on"] = round(off_ms / on_ms, 4)  # platform / cuda_async
        result["interpretation"] = (
            f"cuda_async (ON) {result['speedup_lever_on']:.3f}x vs platform (OFF); "
            ">1 means the lever is faster."
        )
    # The lever ships only if cuda_async FITS (no OOM/finite regression).
    if on.get("fit_ok") is not None:
        result["lever_fit_ok"] = bool(on.get("fit_ok"))
        result["ship_gate"] = (
            "PASS: cuda_async fits + all domains finite/present"
            if on.get("fit_ok")
            else "FAIL: cuda_async OOM'd or a domain went non-finite/absent -- keep platform default"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
