#!/usr/bin/env python
"""Generate M6-S7 Tier-4 probtest prototype artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.proof_schemas import Tier4ProbtestTolerances
from gpuwrf.validation.tier4_probtest import (
    DEFAULT_DOMAIN,
    DEFAULT_ENDING_CYCLE,
    DEFAULT_GEN2_ROOT,
    DEFAULT_HELDOUT_CYCLE,
    DEFAULT_LEADS_H,
    DEFAULT_TOLERANCE_FACTOR,
    DEFAULT_VARIABLES,
    available_complete_historical_members,
    build_cost_model,
    build_ensemble_member_manifest,
    derive_probtest_tolerances,
    select_historical_members,
    utc_now_iso,
    validate_heldout_candidate,
    write_json,
    write_tolerance_freeze_report,
)


ARTIFACT_DIR = ROOT / "artifacts" / "m6" / "tier4"
DEFAULT_CANDIDATE_OUTPUTS = ROOT / "artifacts" / "m6" / "forecast_24h_summary.outputs.json"
DEFAULT_SPACETIME_BUDGET = ROOT / "artifacts" / "m6" / "spacetime_budget_d02.json"


def _rel(path: str | Path) -> str:
    target = Path(path)
    try:
        return str(target.relative_to(ROOT))
    except ValueError:
        return str(target)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    member_manifest_path = out_dir / "ensemble_member_manifest.json"
    tolerance_path = out_dir / "probtest_tolerances.json"
    cost_model_path = out_dir / "cost_model.json"
    freeze_report_path = out_dir / "tolerance_freeze_report.md"
    heldout_path = out_dir / "heldout_candidate_validation.json"

    blockers: list[str] = []
    try:
        member_paths = select_historical_members(
            args.gen2_root,
            ending_cycle=args.ending_cycle,
            count=args.member_count,
            heldout_cycle=args.heldout_cycle,
            domain=args.domain,
            required_leads_h=args.leads_h,
        )
    except ValueError as exc:
        blockers.append(str(exc))
        member_paths = available_complete_historical_members(
            args.gen2_root,
            ending_cycle=args.ending_cycle,
            heldout_cycle=args.heldout_cycle,
            domain=args.domain,
            required_leads_h=args.leads_h,
        )
        if len(member_paths) < 2:
            raise RuntimeError(f"Tier-4 blocked and cannot derive diagnostic spread: {exc}") from exc
    member_paths, shape_blockers = _filter_pinned_grid_shape(member_paths, args.gen2_root, args.ending_cycle, args.domain)
    blockers.extend(shape_blockers)
    if len(member_paths) < 2:
        raise RuntimeError(f"Tier-4 blocked and cannot derive diagnostic spread after grid-shape filtering: {blockers}")
    freeze_time = utc_now_iso()
    member_manifest = build_ensemble_member_manifest(
        member_paths,
        domain=args.domain,
        leads_h=args.leads_h,
        created_utc=freeze_time,
        ending_cycle=args.ending_cycle,
        heldout_cycle=args.heldout_cycle,
    )
    member_manifest["status"] = "BLOCKED" if blockers else "PASS"
    if blockers:
        member_manifest["blockers"] = blockers
    write_json(member_manifest_path, member_manifest)

    artifact_paths = [
        _rel(tolerance_path),
        _rel(member_manifest_path),
        _rel(cost_model_path),
        _rel(freeze_report_path),
        _rel(heldout_path),
    ]
    tolerances = derive_probtest_tolerances(
        member_paths,
        domain=args.domain,
        variables=args.variables,
        leads_h=args.leads_h,
        tolerance_factor=args.tolerance_factor,
        member_manifest_path=_rel(member_manifest_path),
        artifact_paths=artifact_paths,
        freeze_time_utc=freeze_time,
    )
    if blockers:
        tolerances["status"] = "BLOCKED"
        tolerances["sample_size_required"] = int(args.member_count)
        tolerances["blockers"] = blockers
        tolerances["method"]["sample_type"] = (
            f"{len(member_paths)} complete deterministic historical Gen2 wrf_l3 day-members; "
            f"{args.member_count} required for acceptance"
        )
        tolerances["method"]["diagnostic_only_reason"] = (
            "Local Gen2 wrf_l3 history has fewer than 10 complete d02 day-members through +24h; "
            "the table is a pre-candidate diagnostic computed from the complete members only."
        )
    Tier4ProbtestTolerances.validate_dict(tolerances)
    write_json(tolerance_path, tolerances)

    write_tolerance_freeze_report(
        freeze_report_path,
        tolerances=tolerances,
        member_manifest=member_manifest,
        cost_model_path=_rel(cost_model_path),
    )

    cost_model = build_cost_model(
        member_manifest=member_manifest,
        tolerances=tolerances,
        member_manifest_path=member_manifest_path,
        tolerance_path=tolerance_path,
        freeze_report_path=freeze_report_path,
        candidate_output_manifest=args.candidate_outputs,
        spacetime_budget_path=args.spacetime_budget,
        created_utc=utc_now_iso(),
    )
    write_json(cost_model_path, cost_model)

    heldout_run = _heldout_run_path(args.gen2_root, args.heldout_cycle)
    try:
        heldout = validate_heldout_candidate(
            tolerances,
            heldout_run_path=heldout_run,
            candidate_output_manifest=args.candidate_outputs,
            domain=args.domain,
            root=ROOT,
            created_utc=utc_now_iso(),
        )
        if blockers and heldout["status"] == "PASS":
            heldout["status"] = "BLOCKED"
            heldout["blockers"] = blockers
    except Exception as exc:
        heldout = {
            "artifact_type": "tier4_heldout_candidate_validation",
            "created_utc": utc_now_iso(),
            "prototype_label": "M6 prototype; full ensemble at M7",
            "status": "BLOCKED",
            "freeze_time_utc": tolerances["freeze_time_utc"],
            "candidate_evaluation_policy": (
                "candidate validation attempted only after the frozen tolerance artifact and report were written"
            ),
            "heldout_run_path": str(heldout_run),
            "candidate_output_manifest": str(args.candidate_outputs),
            "domain": args.domain,
            "variables": args.variables,
            "leads_h": args.leads_h,
            "blockers": blockers + [f"held-out validation unavailable: {type(exc).__name__}: {exc}"],
            "results": {},
            "failures": [],
        }
    heldout["artifact_paths"] = [_rel(heldout_path), _rel(tolerance_path), _rel(args.candidate_outputs)]
    write_json(heldout_path, heldout)

    return {
        "status": "PASS" if tolerances["status"] == "PASS" and heldout["status"] == "PASS" else "BLOCKED",
        "heldout_status": heldout["status"],
        "tolerances": _rel(tolerance_path),
        "member_manifest": _rel(member_manifest_path),
        "cost_model": _rel(cost_model_path),
        "freeze_report": _rel(freeze_report_path),
        "heldout_candidate_validation": _rel(heldout_path),
    }


def _run_path_for_cycle(gen2_root: str | Path, cycle: str | None) -> Path:
    if cycle is None:
        raise ValueError("cycle is required")
    candidates = []
    for child in Path(gen2_root).iterdir():
        if child.is_dir() and child.name.startswith(f"{cycle}_l3_24h_"):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"no Gen2 run found for {cycle}")
    return sorted(candidates)[-1]


def _heldout_run_path(gen2_root: str | Path, heldout_cycle: str | None) -> Path:
    return _run_path_for_cycle(gen2_root, heldout_cycle)


def _filter_pinned_grid_shape(
    member_paths: list[Path],
    gen2_root: str | Path,
    ending_cycle: str,
    domain: str,
) -> tuple[list[Path], list[str]]:
    from gpuwrf.io.gen2_accessor import Gen2Run

    reference_run = Gen2Run(_run_path_for_cycle(gen2_root, ending_cycle))
    reference_grid = reference_run.grid(domain)
    reference_shape = (int(reference_grid.mass_ny), int(reference_grid.mass_nx))
    kept: list[Path] = []
    dropped: list[str] = []
    for path in member_paths:
        run = Gen2Run(path)
        grid = run.grid(domain)
        shape = (int(grid.mass_ny), int(grid.mass_nx))
        if shape == reference_shape:
            kept.append(Path(path))
        else:
            dropped.append(f"{run.run_id} has d02 shape {shape}, expected pinned {reference_shape}")
    if dropped:
        return kept, ["excluded complete members with non-pinned d02 grid shape: " + "; ".join(dropped)]
    return kept, []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen2-root", default=str(DEFAULT_GEN2_ROOT))
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--ending-cycle", default=DEFAULT_ENDING_CYCLE)
    parser.add_argument("--heldout-cycle", default=DEFAULT_HELDOUT_CYCLE)
    parser.add_argument("--member-count", type=int, default=10)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument("--leads-h", nargs="+", type=int, default=list(DEFAULT_LEADS_H))
    parser.add_argument("--tolerance-factor", type=float, default=DEFAULT_TOLERANCE_FACTOR)
    parser.add_argument("--candidate-outputs", default=str(DEFAULT_CANDIDATE_OUTPUTS))
    parser.add_argument("--spacetime-budget", default=str(DEFAULT_SPACETIME_BUDGET))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
