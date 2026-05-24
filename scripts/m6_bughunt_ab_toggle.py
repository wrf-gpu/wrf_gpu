#!/usr/bin/env python3
"""Run one-suspect-at-a-time sanitizer-off A/B toggles for M6.x S3 hunt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnostic_first_bad_step_tracer import (  # noqa: E402
    DiagnosticToggle,
    coefficient_sanity,
    load_default_case,
    now_label,
    replay_config_for_steps,
    run_sanitizer_off_replay,
)
from gpuwrf.integration.d02_replay import DEFAULT_REPLAY_RUN_DIR, ReplayConfig  # noqa: E402


SPRINT_DIR = ROOT / ".agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt"
DEFAULT_OUTPUT = SPRINT_DIR / "proof_ab_toggles.json"
DEFAULT_COEFFICIENTS = SPRINT_DIR / "proof_column_coefficients.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coefficient-output", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_REPLAY_RUN_DIR)
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--n-acoustic", type=int, default=4)
    return parser.parse_args(argv)


def _brief(run: dict) -> dict:
    return {
        "toggle": run["toggle"],
        "steps_completed": run["steps_completed"],
        "first_bad_step": run["first_bad_step"],
        "first_nonfinite_step": run["first_nonfinite_step"],
        "first_guard_limit_step": run["first_guard_limit_step"],
        "first_issue": run["first_issue"],
        "terminal_nonfinite_fields": run["terminal_nonfinite_fields"],
        "terminal_fields_on_cap": run["terminal_fields_on_cap"],
        "ten_step_sanitize_off_acceptance": run["ten_step_sanitize_off_acceptance"],
        "wall_time_s": run["wall_time_s"],
    }


def _score(run: dict) -> tuple[int, int, int]:
    first_bad = run["first_bad_step"] if run["first_bad_step"] is not None else 9999
    first_nonfinite = run["first_nonfinite_step"] if run["first_nonfinite_step"] is not None else 9999
    accepted = 1 if run["ten_step_sanitize_off_acceptance"] else 0
    return accepted, first_nonfinite, first_bad


def _recommendation(baseline: dict, variants: list[dict]) -> str:
    base_score = _score(baseline)
    best = max(variants, key=_score)
    if best["ten_step_sanitize_off_acceptance"] and not baseline["ten_step_sanitize_off_acceptance"]:
        return "dominant: this single suspect removes 10-step sanitizer-off nonfinite/cap failure"
    if _score(best) > base_score:
        return "changes first-bad metric; needs source-cited fix sprint before implementation"
    return "does not localize the first-bad metric versus baseline"


def _run(case, replay_config, toggle: DiagnosticToggle, *, steps: int) -> dict:
    print(f"running toggle={toggle.name}", flush=True)
    run = run_sanitizer_off_replay(case, replay_config, steps=steps, toggle=toggle, abort_on_first_bad=True)
    print(
        "toggle={name} completed={completed} first_bad={bad} "
        "first_nonfinite={nonfinite} first_guard={guard} accepted={accepted}".format(
            name=toggle.name,
            completed=run["steps_completed"],
            bad=run["first_bad_step"],
            nonfinite=run["first_nonfinite_step"],
            guard=run["first_guard_limit_step"],
            accepted=run["ten_step_sanitize_off_acceptance"],
        ),
        flush=True,
    )
    return run


def _entry(name: str, description: str, baseline: dict, variants: list[dict], *, extra: dict | None = None) -> dict:
    payload = {
        "name": name,
        "change_description": description,
        "before": _brief(baseline),
        "variants": [_brief(run) for run in variants],
        "recommendation": _recommendation(baseline, variants),
    }
    if extra:
        payload.update(extra)
    if len(variants) == 1:
        payload["after"] = payload["variants"][0]
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "loading real d02 replay case "
        f"run_dir={args.run_dir} domain={args.domain} steps={args.steps} n_acoustic={args.n_acoustic}",
        flush=True,
    )
    case = load_default_case(args.run_dir, domain=args.domain)
    template = ReplayConfig()
    replay_config = replay_config_for_steps(
        int(args.steps),
        dt_s=float(args.dt_s),
        n_acoustic=int(args.n_acoustic),
        template=template,
    )

    coefficient_payload = coefficient_sanity(case, dt=float(args.dt_s) / float(args.n_acoustic), epssm=0.1)
    coefficient_target = args.coefficient_output if args.coefficient_output.is_absolute() else ROOT / args.coefficient_output
    coefficient_target.parent.mkdir(parents=True, exist_ok=True)
    coefficient_target.write_text(
        json.dumps(coefficient_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote coefficient sanity proof: {coefficient_target}", flush=True)

    baseline = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="baseline_current",
            description="Current operator, sanitizer bypassed.",
            n_acoustic=int(args.n_acoustic),
        ),
        steps=int(args.steps),
    )

    rows = []

    sign_flip = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="mpas_recurrence_cofwr_sign_flip",
            description="Flip only the MPAS recurrence density-coupling cofwr sign.",
            n_acoustic=int(args.n_acoustic),
            recurrence_mode="cofwr_sign_flip",
        ),
        steps=int(args.steps),
    )
    rows.append(
        _entry(
            "mpas_recurrence_sign_check",
            "Source check plus one-sign A/B: flip only cofwr density coupling in rhs_interior.",
            baseline,
            [sign_flip],
            extra={
                "source_sign_check": {
                    "current": "src/gpuwrf/dynamics/acoustic_wrf.py:777-793",
                    "mpas": "mpas_atm_time_integration.F:2146-2169",
                    "assessment": [
                        "rs subtracts cofrz*resm*(rw_p[k+1]-rw_p[k]) as in MPAS 2147-2148.",
                        "ts subtracts resm*rdzw*(coftz[k+1]*rw_p[k+1]-coftz[k]*rw_p[k]) as in MPAS 2149-2151.",
                        "rhs subtracts cofwz and cofwr terms and adds upper/lower cofwt terms as in MPAS 2160-2169.",
                    ],
                }
            },
        )
    )

    mu_zero = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="mu_continuity_dmu_zero",
            description="Set dmu update to zero by disabling mu_continuity only.",
            n_acoustic=int(args.n_acoustic),
            mu_mode="zero",
        ),
        steps=int(args.steps),
    )
    mu_raw = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="mu_continuity_raw_unbounded",
            description="Use raw dmu = dt*dmu_dt with no tanh bound.",
            n_acoustic=int(args.n_acoustic),
            mu_mode="raw",
        ),
        steps=int(args.steps),
    )
    rows.append(
        _entry(
            "mu_continuity_increment",
            "Toggle only the mass-continuity increment: dmu=0 and raw unbounded dmu.",
            baseline,
            [mu_zero, mu_raw],
            extra={"source": "WRF module_small_step_em.F:1094-1119; current src/gpuwrf/dynamics/acoustic_wrf.py:473-495,926-930"},
        )
    )

    metric_ref = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="mpas_w_metric_reference_column",
            description="Replace the per-level metric with one fixed center-column reference profile.",
            n_acoustic=int(args.n_acoustic),
            metric_mode="reference_column",
        ),
        steps=int(args.steps),
    )
    rows.append(
        _entry(
            "mpas_w_metric_faces",
            "Toggle only _mpas_w_metric_faces to a fixed reference column metric.",
            baseline,
            [metric_ref],
            extra={"source": "MPAS mpas_atm_time_integration.F:2491-2495; current src/gpuwrf/dynamics/acoustic_wrf.py:512-534"},
        )
    )

    acoustic_runs = []
    for n_acoustic in [1, 4, 8, 16]:
        if n_acoustic == int(args.n_acoustic):
            acoustic_runs.append(baseline)
            continue
        acoustic_config = replay_config_for_steps(
            int(args.steps),
            dt_s=float(args.dt_s),
            n_acoustic=n_acoustic,
            template=template,
        )
        acoustic_runs.append(
            _run(
                case,
                acoustic_config,
                DiagnosticToggle(
                    name=f"n_acoustic_{n_acoustic}",
                    description=f"Set only n_acoustic={n_acoustic}.",
                    n_acoustic=n_acoustic,
                ),
                steps=int(args.steps),
            )
        )
    rows.append(
        _entry(
            "n_acoustic_sweep",
            "Vary only acoustic substep count across {1,4,8,16}.",
            baseline,
            acoustic_runs,
            extra={"source": "src/gpuwrf/integration/d02_replay.py:489-520; src/gpuwrf/dynamics/acoustic_wrf.py:954-987"},
        )
    )

    physics_off = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="physics_disabled",
            description="Skip Thompson, MYNN, surface, and radiation adapters only.",
            n_acoustic=int(args.n_acoustic),
            physics_enabled=False,
        ),
        steps=int(args.steps),
    )
    rows.append(_entry("physics_disable", "Disable only physics tendencies/adapters.", baseline, [physics_off]))

    boundary_off = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="boundary_disabled",
            description="Skip lateral boundary application only.",
            n_acoustic=int(args.n_acoustic),
            boundary_enabled=False,
        ),
        steps=int(args.steps),
    )
    rows.append(_entry("boundary_disable", "Disable only lateral boundary application.", baseline, [boundary_off]))

    branch_forced = _run(
        case,
        replay_config,
        DiagnosticToggle(
            name="force_positive_pressure_branch",
            description="Force vertical_acoustic_update pressure_scale=+1 while leaving nonhydrostatic PGF enabled.",
            n_acoustic=int(args.n_acoustic),
            pressure_scale_override=1.0,
        ),
        steps=int(args.steps),
    )
    rows.append(
        _entry(
            "branch_verification",
            "Verify real d02 baseline takes pressure_scale <= 0 MPAS recurrence; A/B forces positive branch.",
            baseline,
            [branch_forced],
            extra={
                "baseline_branch": "pressure_scale <= 0.0 -> _mpas_recurrence_vertical_update",
                "source": "src/gpuwrf/dynamics/acoustic_wrf.py:686-695,742-752",
            },
        )
    )

    payload = {
        "created_utc": now_label(),
        "objective": "Stage 2 one-suspect-at-a-time sanitizer-off A/B toggles for real Gen2 d02 replay.",
        "run_dir": str(args.run_dir),
        "domain": args.domain,
        "steps": int(args.steps),
        "dt_s": float(args.dt_s),
        "baseline": _brief(baseline),
        "toggles": rows,
        "coefficient_sanity_proof": str(coefficient_target),
        "verdict_hint": "BUG-FOUND-NEEDS-DESIGN" if any("dominant" in row["recommendation"] for row in rows) else "NO-BUG-LOCALIZED",
    }
    target = args.output if args.output.is_absolute() else ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
