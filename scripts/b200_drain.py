#!/usr/bin/env python3
"""Drain verified B200 hourly/block outputs with resume and stop/pull support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b200_io_lib import (  # noqa: E402
    DEFAULT_DRAIN_PATTERNS,
    DrainConfig,
    drain_human_report,
    drain_once,
    parse_bytes,
    parse_expected_dims,
    stop_pull,
    synthetic_dry_run,
    write_json_atomic,
)


def _add_drain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True, help="running output directory")
    parser.add_argument("--target", required=True, help="local/file:// target directory or s3:// prefix")
    parser.add_argument("--state-dir", type=Path, required=True, help="local drain state directory")
    parser.add_argument("--pattern", action="append", default=[], help="glob pattern under output-dir")
    parser.add_argument("--expected-var", action="append", default=[], help="required NetCDF variable")
    parser.add_argument("--expected-dim", action="append", default=[], help="required NetCDF dimension as name=size")
    parser.add_argument("--expected-domain", help="shortcut for west_east x south_north, e.g. 898x898")
    parser.add_argument("--expected-time-steps", type=int, help="exact required NetCDF Time record count")
    parser.add_argument(
        "--min-time-steps",
        type=int,
        default=1,
        help="minimum NetCDF Time record count; default rejects empty/truncated in-progress files",
    )
    parser.add_argument("--min-age-seconds", type=float, default=30.0, help="minimum age before a block is complete")
    parser.add_argument("--local-cap-bytes", type=parse_bytes, help="maximum allowed output-dir bytes")
    parser.add_argument("--target-cap-bytes", type=parse_bytes, help="maximum allowed target bytes / S3 byte budget")
    parser.add_argument(
        "--target-min-free-bytes",
        type=parse_bytes,
        default=0,
        help="minimum free bytes to preserve on local target",
    )
    parser.add_argument("--delete-after-copy", action="store_true", help="delete source block after verified copy")
    parser.add_argument("--dry-run", action="store_true", help="verify and plan copies without copying/deleting")
    parser.add_argument(
        "--training-policy",
        choices=("contiguous", "allow-partial"),
        default="allow-partial",
        help="allow-partial accepts every verified block as training data; contiguous requires --expected-block-count",
    )
    parser.add_argument(
        "--expected-block-count",
        type=int,
        help="required verified block count before contiguous policy marks a case training-ready",
    )
    parser.add_argument("--json-out", type=Path, help="write machine-readable report")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the human report")


def _config_from_args(args: argparse.Namespace) -> DrainConfig:
    return DrainConfig(
        output_dir=args.output_dir,
        target=args.target,
        state_dir=args.state_dir,
        patterns=tuple(args.pattern or DEFAULT_DRAIN_PATTERNS),
        expected_vars=tuple(args.expected_var),
        expected_dims=parse_expected_dims(args.expected_dim, args.expected_domain),
        min_age_seconds=args.min_age_seconds,
        local_cap_bytes=args.local_cap_bytes,
        target_cap_bytes=args.target_cap_bytes,
        target_min_free_bytes=args.target_min_free_bytes or 0,
        delete_after_copy=args.delete_after_copy,
        dry_run=args.dry_run,
        training_policy=args.training_policy,
        expected_block_count=args.expected_block_count,
        expected_time_steps=args.expected_time_steps,
        min_time_steps=args.min_time_steps,
    )


def _emit(report: dict, args: argparse.Namespace) -> int:
    if getattr(args, "json_out", None):
        write_json_atomic(args.json_out, report)
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if report.get("schema") == "gpuwrf.b200_synthetic_dry_run.v1":
            print(f"B200 synthetic dry-run: {report['status']}")
            print(f"work_dir: {report['work_dir']}")
            print(f"manifest_validation: {report['manifest_validation']['status']}")
            print(f"first_drain: {report['first_drain']['status']}")
            print(f"stop_pull: {report['stop_pull']['status']}")
            print(f"resume: {report['resume']['status']}")
        else:
            print(drain_human_report(report))
    return 0 if report.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B200 output block drain, backpressure, resume, and stop/pull tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    drain_parser = sub.add_parser("drain", help="verify and copy/upload completed blocks")
    _add_drain_args(drain_parser)

    resume_parser = sub.add_parser("resume", help="resume drain from verified+done markers")
    _add_drain_args(resume_parser)

    stop_parser = sub.add_parser("stop-pull", help="request a clean stop and drain all completed not-yet-drained blocks")
    _add_drain_args(stop_parser)
    stop_parser.add_argument("--pid-file", type=Path, help="optional run PID file to SIGTERM")
    stop_parser.add_argument("--stop-command", help="optional command to request graceful shutdown")
    stop_parser.add_argument("--stop-marker", type=Path, help="path for the stop-request marker JSON")
    stop_parser.add_argument("--grace-seconds", type=float, default=10.0, help="wait after SIGTERM")
    stop_parser.add_argument(
        "--force-kill-after-grace",
        action="store_true",
        help="send SIGKILL if the PID is still alive after the grace period",
    )

    dry_parser = sub.add_parser("synthetic-dry-run", help="exercise validator, drain, stop-pull, and resume in temp data")
    dry_parser.add_argument("--work-dir", type=Path, help="reuse/write a specific dry-run directory")
    dry_parser.add_argument("--json-out", type=Path, help="write machine-readable report")
    dry_parser.add_argument("--json", action="store_true", help="print JSON instead of the concise report")

    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic-dry-run":
            report = synthetic_dry_run(args.work_dir)
        else:
            config = _config_from_args(args)
            if args.command in {"drain", "resume"}:
                report = drain_once(config)
            elif args.command == "stop-pull":
                stop_command = shlex.split(args.stop_command) if args.stop_command else None
                report = stop_pull(
                    config,
                    pid_file=args.pid_file,
                    stop_command=stop_command,
                    stop_marker=args.stop_marker,
                    grace_seconds=args.grace_seconds,
                    force_kill_after_grace=args.force_kill_after_grace,
                )
            else:
                parser.error(f"unknown command {args.command}")
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema": "gpuwrf.b200_drain.v1",
            "status": "FAIL",
            "ok": False,
            "output_dir": str(getattr(args, "output_dir", "")),
            "target": str(getattr(args, "target", "")),
            "state_dir": str(getattr(args, "state_dir", "")),
            "issues": [{"severity": "error", "code": "drainer_crashed", "message": str(exc)}],
            "blocks": [],
        }
    return _emit(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
