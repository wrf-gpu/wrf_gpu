#!/usr/bin/env python3
"""Validate a B200 run manifest and staged input directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b200_io_lib import manifest_human_report, validate_b200_manifest, write_json_atomic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed validator for B200 manifests, input checksums, and WRF nest dimensions."
    )
    parser.add_argument("manifest", type=Path, help="run manifest JSON")
    parser.add_argument("staged_input_dir", type=Path, help="local staged input directory")
    parser.add_argument("--json-out", type=Path, help="write machine-readable validation report")
    parser.add_argument(
        "--allow-missing-checksums",
        action="store_true",
        help="do not fail when a required input lacks sha256/checksum_sha256 (default: fail closed)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of the human report")
    args = parser.parse_args(argv)

    try:
        report = validate_b200_manifest(
            args.manifest,
            args.staged_input_dir,
            require_checksums=not args.allow_missing_checksums,
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema": "gpuwrf.b200_manifest_validation.v1",
            "status": "FAIL",
            "ok": False,
            "manifest_path": str(args.manifest),
            "staged_input_dir": str(args.staged_input_dir),
            "required_inputs": [],
            "domains": [],
            "issues": [{"severity": "error", "code": "validator_crashed", "message": str(exc)}],
        }
    if args.json_out:
        write_json_atomic(args.json_out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(manifest_human_report(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
