#!/usr/bin/env python
"""Execute available HIGH publication tests and stamp honest proof objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from pubtest_common import (
        CANARY_RUN_ROOT,
        HIGH_TEST_FILES,
        ROOT,
        SPRINT_DIR,
        discover_canary_cases,
        finite_stats,
        gpu_probe,
        proof_header,
        read_json,
        run_command,
        summarize_m6b6,
        summarize_repeatability,
        summarize_restart,
        summarize_skill,
        threshold_rows,
        write_case_summary,
        write_json,
        write_summary_md,
        wrf_provenance,
    )
except ModuleNotFoundError:  # pragma: no cover - import path used by pytest package collection
    from scripts.pubtest_common import (
        CANARY_RUN_ROOT,
        HIGH_TEST_FILES,
        ROOT,
        SPRINT_DIR,
        discover_canary_cases,
        finite_stats,
        gpu_probe,
        proof_header,
        read_json,
        run_command,
        summarize_m6b6,
        summarize_repeatability,
        summarize_restart,
        summarize_skill,
        threshold_rows,
        write_case_summary,
        write_json,
        write_summary_md,
        wrf_provenance,
    )

from gpuwrf.fixtures.idealized_cases import build_density_current, build_schaer_mountain_wave, build_warmbubble


def _write_blocked_idealized(proof_dir: Path, *, gpu: dict[str, Any], wrf: dict[str, Any]) -> list[Path]:
    inputs = proof_dir / "inputs"
    warm = write_case_summary(inputs / "warmbubble_ic_summary.json", build_warmbubble())
    density = write_case_summary(inputs / "density_current_ic_summary.json", build_density_current())
    mountain = write_case_summary(inputs / "schaer_mountain_wave_ic_summary.json", build_schaer_mountain_wave())
    common_blockers = [
        "GPU preflight did not return a usable device in this execution session.",
        "No idealized GPU integration was run; generated IC summaries are not forecast evidence.",
        "Stock WRF idealized compile/reference run was not attempted in this partial execution.",
    ]
    written: list[Path] = []
    cases = [
        (
            "IDEALIZED-WARMBUBBLE",
            "idealized_warmbubble.json",
            warm,
            threshold_rows(
                {
                    "finite_fields": (1.0 if finite_stats(warm) else 0.0, "all fields finite", finite_stats(warm)),
                    "theta_w_nrmse_ladder": (None, "<=0.05/0.08/0.12/0.18 vs CPU WRF", None),
                    "w_max_lead_time_error": (None, "<=10%", None),
                    "dry_mass_drift": (None, "<=1e-10", None),
                    "horizontal_symmetry": (None, "<=1e-10", None),
                }
            ),
            "Blocked after IC generation; no GPU/WRF time integration was produced.",
        ),
        (
            "IDEALIZED-DENSITY-CURRENT",
            "idealized_density_current.json",
            density,
            threshold_rows(
                {
                    "finite_fields": (1.0 if finite_stats(density) else 0.0, "all fields finite", finite_stats(density)),
                    "front_position_900s": (None, "within 1 horizontal grid cell of Straka 1993", None),
                    "front_speed": (None, "within 5% of about 33 m/s", None),
                    "min_theta_perturbation": (
                        density["array_stats"]["theta_perturbation_k"]["min"],
                        "within 0.5 K of -15 K at IC; forecast threshold not evaluated",
                        abs(float(density["array_stats"]["theta_perturbation_k"]["min"]) + 15.0) <= 0.5,
                    ),
                    "dry_mass_drift": (None, "<=1e-10", None),
                }
            ),
            "Blocked after Straka IC generation; no 900 s GPU density-current integration was produced.",
        ),
        (
            "IDEALIZED-MOUNTAIN-WAVE",
            "idealized_mountain_wave.json",
            mountain,
            threshold_rows(
                {
                    "finite_fields": (1.0 if finite_stats(mountain) else 0.0, "all fields finite", finite_stats(mountain)),
                    "w_peak_5h": (None, "within 10% of Schaer analytic steady-state solution", None),
                    "dominant_wave_phase": (None, "<=1 grid cell horizontal and vertical", None),
                    "pressure_nrmse": (None, "<=0.10 vs analytic", None),
                    "open_boundary_mass_residual": (None, "<=1e-6", None),
                }
            ),
            "Blocked after Schaer IC generation; no 5 h GPU mountain-wave integration was produced.",
        ),
    ]
    for test_id, filename, summary, thresholds, note in cases:
        payload = proof_header(test_id, "BLOCKED", "BLOCKED_NO_GPU_IDEALIZED_RUN")
        payload.update(
            {
                "proof": {"reference": summary["reference"], "wrf_provenance": wrf, "gpu_preflight": gpu},
                "inputs": summary,
                "thresholds": thresholds,
                "blockers": common_blockers,
                "honesty_note": note,
                "gpu_hours_used": 0.0,
            }
        )
        path = proof_dir / filename
        write_json(path, payload)
        written.append(path)
    write_summary_md(
        proof_dir / "idealized_warmbubble_summary.md",
        "IDEALIZED-WARMBUBBLE Summary",
        read_json(proof_dir / "idealized_warmbubble.json") or {},
        ["- IC builder completed.", "- Forecast comparison did not run; proof verdict is BLOCKED."],
    )
    write_summary_md(
        proof_dir / "idealized_mountain_wave_summary.md",
        "IDEALIZED-MOUNTAIN-WAVE Summary",
        read_json(proof_dir / "idealized_mountain_wave.json") or {},
        ["- Schaer terrain/IC builder completed.", "- Forecast comparison did not run; proof verdict is BLOCKED."],
    )
    return written


def _write_conservation(proof_dir: Path) -> list[Path]:
    inputs = proof_dir / "inputs"
    tracker_input = inputs / "conservation_static_smoke_states.json"
    tracker_output = proof_dir / "conservation_static_tracker_smoke.json"
    state_series = {
        "states": [
            {"step": 0, "time_s": 0.0, "totals": {"mass": 1.0, "water": 0.0, "kinetic_energy": 0.0, "dry_static_energy": 1.0}},
            {"step": 8640, "time_s": 86400.0, "totals": {"mass": 1.0, "water": 0.0, "kinetic_energy": 0.0, "dry_static_energy": 1.0}},
        ],
        "note": "Static smoke input for diagnostic_conservation_tracker reuse only; not a model integration.",
    }
    write_json(tracker_input, state_series)
    command = run_command(
        [sys.executable, "scripts/diagnostic_conservation_tracker.py", "--input", str(tracker_input), "--output", str(tracker_output)],
        timeout_s=30,
    )
    tracker = read_json(tracker_output) or {}
    common = {
        "diagnostic_reuse": {
            "script": "scripts/diagnostic_conservation_tracker.py",
            "command": command,
            "tracker_output": str(tracker_output),
            "tracker_status": tracker.get("status"),
            "tracker_max_abs_relative_drift": tracker.get("measurements", {}).get("max_abs_relative_drift"),
        },
        "blockers": [
            "No 24 h GPU warm-bubble/Canary state series was produced because GPU preflight is unavailable.",
            "The tracker run is a wrapper smoke only and is not used as acceptance evidence.",
        ],
        "gpu_hours_used": 0.0,
    }
    mass = proof_header("CONSERVATION-MASS-24H", "BLOCKED", "BLOCKED_NO_24H_GPU_STATE_SERIES")
    mass.update(
        common
        | {
            "thresholds": threshold_rows(
                {
                    "closed_domain_dry_mass_drift": (None, "<=1e-10 over 24 h", None),
                    "canary_flux_corrected_residual": (None, "<=1e-5 over 24 h", None),
                    "nonfinite_mass_fields": (None, "zero", None),
                }
            )
        }
    )
    energy = proof_header("CONSERVATION-ENERGY-24H", "BLOCKED", "BLOCKED_NO_CPU_ENVELOPE_OR_24H_GPU_SERIES")
    energy.update(
        common
        | {
            "thresholds": threshold_rows(
                {
                    "total_energy_drift_vs_cpu": (None, "within +/-20% of CPU WRF drift", None),
                    "component_drift_vs_cpu": (None, "KE/internal/potential within CPU envelope +/-20%", None),
                    "unexplained_step_jump": (None, "<=0.05%", None),
                }
            )
        }
    )
    paths = [proof_dir / "conservation_mass_24h.json", proof_dir / "conservation_energy_24h.json"]
    write_json(paths[0], mass)
    write_json(paths[1], energy)
    return paths


def _write_stability(proof_dir: Path, *, gpu: dict[str, Any]) -> list[Path]:
    warm = read_json(proof_dir / "inputs" / "warmbubble_ic_summary.json")
    density = read_json(proof_dir / "inputs" / "density_current_ic_summary.json")
    cfl = proof_header("STABILITY-CFL-SWEEP", "BLOCKED", "BLOCKED_NO_GPU_STABILITY_RUNNER")
    cfl.update(
        {
            "case": "warmbubble",
            "inputs": warm,
            "gpu_preflight": gpu,
            "planned_dt_multipliers": [0.5, 1.0, 1.25],
            "thresholds": threshold_rows(
                {
                    "dt_0p5_finite_mass": (None, "complete finite, mass drift <=1e-10", None),
                    "dt_1p0_finite_mass": (None, "complete finite, mass drift <=1e-10", None),
                    "dt_1p25_deterministic_outcome": (None, "complete or fail deterministically", None),
                }
            ),
            "blockers": ["No GPU stability runner executed in this session."],
            "gpu_hours_used": 0.0,
        }
    )
    acoustic = proof_header("STABILITY-ACOUSTIC-SUBSTEP-SWEEP", "BLOCKED", "BLOCKED_NO_GPU_ACOUSTIC_SWEEP_RUNNER")
    acoustic.update(
        {
            "case": "density_current",
            "inputs": density,
            "gpu_preflight": gpu,
            "planned_substep_counts": [4, 6, 8],
            "thresholds": threshold_rows(
                {
                    "all_runs_finite": (None, "true for n in {4,6,8}", None),
                    "front_position_variation": (None, "<=1 cell across settings", None),
                    "ke_pairwise_nrmse": (None, "<=0.05", None),
                }
            ),
            "blockers": ["No GPU acoustic-substep sweep executed in this session."],
            "gpu_hours_used": 0.0,
        }
    )
    paths = [proof_dir / "stability_cfl_sweep.json", proof_dir / "stability_acoustic_substep.json"]
    write_json(paths[0], cfl)
    write_json(paths[1], acoustic)
    return paths


def _write_determinism(proof_dir: Path) -> Path:
    repeat_path = ROOT / "proofs" / "generated" / "2026-05-27-m7-daily-pipeline-integration" / "repeatability.json"
    restart_path = ROOT / "proofs" / "2026-05-27-m7-restart-continuity__restart_continuity.json"
    repeat = summarize_repeatability(repeat_path)
    restart = summarize_restart(restart_path)
    run_count = repeat.get("run_count") or 0
    max_delta = repeat.get("max_abs_delta")
    pass_two = repeat.get("status") == "PASS" and max_delta == 0.0
    passed = bool(pass_two and run_count >= 3)
    payload = proof_header("DETERMINISM-REPEAT", "PASS" if passed else "FAIL", "FAIL_REQUIRED_THREE_RUNS_NOT_AVAILABLE")
    payload.update(
        {
            "required_run_count": 3,
            "observed_repeatability": repeat,
            "restart_bitwise_guard": restart,
            "thresholds": threshold_rows(
                {
                    "max_delta_across_observed_runs": (max_delta, "0.0 bitwise for observed repeatability artifact", pass_two),
                    "run_count": (float(run_count), ">=3 identical full-pipeline runs", run_count >= 3),
                }
            ),
            "honesty_note": "Existing two-run pipeline repeatability is bitwise PASS, but this does not satisfy the revised three-run gate.",
            "gpu_hours_used": 0.0,
        }
    )
    path = proof_dir / "determinism_repeat.json"
    write_json(path, payload)
    return path


def _write_savepoint(proof_dir: Path) -> Path:
    m6b6_path = (
        ROOT
        / "<development-history-not-included-in-public-repo>"
        / "2026-05-25-m6b6-coupled-step-parity"
        / "proof_coupled_step_parity.json"
    )
    summary = summarize_m6b6(m6b6_path)
    max_depth = max((int(v or 0) for v in summary.get("tier_savepoint_counts", {}).values()), default=0)
    passed = bool(summary.get("passed") is True and max_depth >= 10000)
    payload = proof_header("SAVEPOINT-PARITY-DEEP", "PASS" if passed else "FAIL", "FAIL_INSUFFICIENT_SAVEPOINT_DEPTH")
    payload.update(
        {
            "current_b6_evidence": summary,
            "required_depths": [100, 1000, 10000],
            "observed_max_depth": max_depth,
            "thresholds": threshold_rows(
                {
                    "step_100_bitwise": (None, "max delta 0.0", None),
                    "step_1000_relative": (None, "<1e-12 relative for all fields", None),
                    "step_10000_relative": (None, "<1e-12 relative for all fields", None),
                    "existing_b6_10_step_parity": (float(max_depth), "10-step B6 remains PASS as guardrail", summary.get("passed") is True),
                }
            ),
            "honesty_note": "The locked B6 10-step parity artifact is PASS, but the revised deep-parity depths were not executed.",
            "gpu_hours_used": 0.0,
        }
    )
    path = proof_dir / "savepoint_parity_deep.json"
    write_json(path, payload)
    return path


def _write_canary(proof_dir: Path) -> list[Path]:
    manifest = discover_canary_cases(CANARY_RUN_ROOT, window_days=14)
    manifest_path = proof_dir / "canary_case_manifest.json"
    write_json(manifest_path, manifest)
    skill_path = ROOT / "proofs" / "2026-05-27-m7-honest-speedup-skill-diff__gpu_vs_cpu_skill_diff.json"
    skill = summarize_skill(skill_path)
    cases_complete = 1 if skill.get("exists") else 0
    per_variable_pass = {
        name: bool(row.get("within_20pct_all_metrics")) for name, row in skill.get("variables", {}).items()
    }
    passed = bool(cases_complete >= 14 and all(per_variable_pass.values()))
    payload = proof_header("CANARY-MULTIDAY-SIDE-BY-SIDE", "PASS" if passed else "FAIL", "FAIL_INSUFFICIENT_GPU_CORPUS_AND_SINGLE_DAY_SKILL")
    payload.update(
        {
            "case_manifest": str(manifest_path),
            "cpu_case_selection": manifest,
            "existing_single_day_skill": skill,
            "completed_gpu_case_count": cases_complete,
            "required_gpu_case_count": 14,
            "per_variable_pass": per_variable_pass,
            "thresholds": threshold_rows(
                {
                    "case_count": (float(cases_complete), ">=14 complete cases", cases_complete >= 14),
                    "T2_within_20pct": (None, "GPU within +/-20% CPU RMSE", per_variable_pass.get("T2")),
                    "U10_within_20pct": (None, "GPU within +/-20% CPU RMSE", per_variable_pass.get("U10")),
                    "V10_within_20pct": (None, "GPU within +/-20% CPU RMSE", per_variable_pass.get("V10")),
                }
            ),
            "honesty_note": "The CPU inventory supports a 14-day window, but available GPU proof is a single 20260521 day and fails the per-variable skill gate.",
            "gpu_hours_used": 0.0,
        }
    )
    path = proof_dir / "canary_multiday_skill.json"
    write_json(path, payload)
    first_growth = {
        "schema": "PublicationFirstErrorGrowth",
        "schema_version": 1,
        "status": "NOT_EVALUATED",
        "reason": "Requires per-hour GPU-vs-CPU curves across the 14-day corpus; only single-day aggregate skill artifact is available.",
        "source_skill_artifact": str(skill_path),
    }
    write_json(proof_dir / "canary_first_error_growth.json", first_growth)
    return [manifest_path, path, proof_dir / "canary_first_error_growth.json"]


def _aggregate(proof_dir: Path) -> Path:
    rows = []
    total_gpu_hours = 0.0
    for test_id, filename in HIGH_TEST_FILES.items():
        payload = read_json(proof_dir / filename) or {}
        total_gpu_hours += float(payload.get("gpu_hours_used") or 0.0)
        rows.append(
            {
                "test_id": test_id,
                "proof_object": str(proof_dir / filename),
                "verdict": payload.get("verdict", "MISSING"),
                "status": payload.get("status", "MISSING"),
            }
        )
    report = proof_dir / "aggregate_report.md"
    lines = [
        "# Aggregate Report - Testing Plan Execution",
        "",
        f"Total GPU hours used: {total_gpu_hours:.1f}",
        "",
        "| Test | Verdict | Status | Proof |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['test_id']} | {row['verdict']} | {row['status']} | `{Path(row['proof_object']).name}` |")
    lines.extend(
        [
            "",
            "## What surprised me",
            "",
            "- The CPU Canary inventory is sufficient for a 14-day window, but the checked-in GPU evidence is only single-day.",
            "- The GPU preflight timed out, so no heavy HIGH-priority GPU execution could be started honestly.",
            "- Existing B6/restart/determinism guardrail artifacts remain useful but do not satisfy the revised deeper gates.",
            "",
            "## Publication claim supported from this sprint",
            "",
            "This sprint supports only a partial-result framing: analytic IC builders and preflight proof objects exist, but the requested community-grade HIGH gates are mostly blocked or failing on available evidence.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        proof_dir / "aggregate_report.json",
        {"schema": "PublicationAggregateReport", "schema_version": 1, "total_gpu_hours_used": total_gpu_hours, "rows": rows},
    )
    return report


def execute(proof_dir: Path, *, skip_gpu_probe: bool = False, gpu_probe_timeout_s: int = 5) -> dict[str, Any]:
    proof_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_probe(skip=skip_gpu_probe, timeout_s=gpu_probe_timeout_s)
    wrf = wrf_provenance()
    paths: list[Path] = []
    paths.extend(_write_blocked_idealized(proof_dir, gpu=gpu, wrf=wrf))
    paths.extend(_write_conservation(proof_dir))
    paths.extend(_write_stability(proof_dir, gpu=gpu))
    paths.append(_write_determinism(proof_dir))
    paths.append(_write_savepoint(proof_dir))
    paths.extend(_write_canary(proof_dir))
    aggregate = _aggregate(proof_dir)
    return {
        "status": "EXECUTION_PARTIAL",
        "proof_dir": str(proof_dir),
        "proof_objects": [str(path) for path in paths],
        "aggregate_report": str(aggregate),
        "gpu_preflight": gpu,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--skip-gpu-probe", action="store_true")
    parser.add_argument("--gpu-probe-timeout-s", type=int, default=5)
    args = parser.parse_args(argv)
    payload = execute(args.proof_dir, skip_gpu_probe=bool(args.skip_gpu_probe), gpu_probe_timeout_s=int(args.gpu_probe_timeout_s))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
