#!/usr/bin/env python3
"""Run short real-d02 replay diagnostics with the sanitizer bypassed."""

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
    load_default_case,
    replay_config_for_steps,
    run_sanitizer_off_replay,
    now_label,
)
from gpuwrf.integration.d02_replay import DEFAULT_REPLAY_RUN_DIR, ReplayConfig  # noqa: E402


DEFAULT_OUTPUT = ROOT / ".agent/sprints/2026-05-24-m6x-s3hunt-operator-bug-hunt/proof_first_bad_trace.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_REPLAY_RUN_DIR)
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--n-acoustic", type=int, default=4)
    parser.add_argument("--continue-after-first-bad", action="store_true")
    return parser.parse_args(argv)


def _target_steps(max_steps: int) -> list[int]:
    targets = [1, 2, 5, 10]
    return [step for step in targets if step <= int(max_steps)] or [int(max_steps)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "loading real d02 replay case "
        f"run_dir={args.run_dir} domain={args.domain} steps={args.steps} n_acoustic={args.n_acoustic}",
        flush=True,
    )
    case = load_default_case(args.run_dir, domain=args.domain)
    template = ReplayConfig()
    toggle = DiagnosticToggle(
        name="baseline_sanitizer_off",
        description="Current real d02 ADR-023 replay with _sanitize_replay_candidate bypassed.",
        n_acoustic=int(args.n_acoustic),
    )
    runs = []
    for target in _target_steps(args.steps):
        print(f"running sanitizer-off target_steps={target}", flush=True)
        replay_config = replay_config_for_steps(
            target,
            dt_s=float(args.dt_s),
            n_acoustic=int(args.n_acoustic),
            template=template,
        )
        run = run_sanitizer_off_replay(
            case,
            replay_config,
            steps=target,
            toggle=toggle,
            abort_on_first_bad=not args.continue_after_first_bad,
            localize_stage=True,
        )
        run["target_steps"] = int(target)
        runs.append(run)
        print(
            "target_steps={target} completed={completed} first_bad={bad} "
            "first_nonfinite={nonfinite} first_guard={guard} accepted={accepted}".format(
                target=target,
                completed=run["steps_completed"],
                bad=run["first_bad_step"],
                nonfinite=run["first_nonfinite_step"],
                guard=run["first_guard_limit_step"],
                accepted=run["ten_step_sanitize_off_acceptance"],
            ),
            flush=True,
        )

    primary = next((item for item in runs if item["target_steps"] == min(args.steps, 10)), runs[-1])
    payload = {
        "created_utc": now_label(),
        "objective": "Stage 1 sanitizer-bypass first-bad trace for real Gen2 d02 replay.",
        "mode": "abort_on_first_bad" if not args.continue_after_first_bad else "continue_after_first_bad",
        "run_dir": str(args.run_dir),
        "domain": args.domain,
        "requested_steps": int(args.steps),
        "dt_s": float(args.dt_s),
        "n_acoustic": int(args.n_acoustic),
        "first_bad_trace": primary["first_issue"],
        "first_bad_step": primary["first_bad_step"],
        "first_nonfinite_step": primary["first_nonfinite_step"],
        "first_guard_limit_step": primary["first_guard_limit_step"],
        "acceptance": {
            "first_nonfinite_step_null": primary["first_nonfinite_step"] is None,
            "no_fields_on_caps": not primary["terminal_fields_on_cap"],
            "ten_step_sanitize_off_acceptance": primary["ten_step_sanitize_off_acceptance"],
        },
        "runs": runs,
    }
    target = args.output if args.output.is_absolute() else ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
