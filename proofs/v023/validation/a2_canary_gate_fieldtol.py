#!/usr/bin/env python3
"""A2 canary gate: field-tolerance vs fresh paired baseline.

This replaces the broken stored-digest canary policy with a CPU-only wrfout
field comparison. It can be used in two modes:

1. ``--self-test``: run positive and negative controls on tiny local NetCDF
   wrfout fixtures. This validates the gate logic without a GPU.
2. paired compare: compare fresh baseline and candidate wrfout directories.

The script never runs a forecast and never reads a stored digest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FIELD_COMPARE = HERE / "paired_baseline_field_compare.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def run_self_test(args: argparse.Namespace) -> int:
    tool_out = args.out_json.with_name(args.out_json.stem + "_field_compare_selftest.json")
    cmd = [sys.executable, str(FIELD_COMPARE), "--self-test", "--out-json", str(tool_out)]
    rc, stdout, stderr = _run(cmd)
    payload = {
        "schema": "v023-a2-canary-fieldtol-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "A2",
        "mode": "self-test",
        "method": "field-tolerance-vs-fresh-paired-baseline; no stored digest",
        "verdict": "PASS" if rc == 0 else "FAIL",
        "field_compare_selftest": str(tool_out),
        "command": cmd,
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
        "acceptance": "positive control passes and perturbed negative control fails",
    }
    _write_json(args.out_json, payload)
    print(json.dumps({"gate": "A2", "verdict": payload["verdict"], "output": str(args.out_json)}, sort_keys=True))
    return 0 if rc == 0 else 1


def run_paired_compare(args: argparse.Namespace) -> int:
    compare_out = args.out_json.with_name(args.out_json.stem + "_field_compare.json")
    cmd = [
        sys.executable,
        str(FIELD_COMPARE),
        "--baseline-dir",
        str(args.baseline_dir),
        "--candidate-dir",
        str(args.candidate_dir),
        "--domains",
        *args.domains,
        "--mode",
        args.mode,
        "--autotune-floor",
        str(args.autotune_floor),
        "--out-json",
        str(compare_out),
    ]
    if args.init:
        cmd.extend(["--init", args.init])
    if args.min_lead is not None:
        cmd.extend(["--min-lead", str(args.min_lead)])
    if args.max_lead is not None:
        cmd.extend(["--max-lead", str(args.max_lead)])
    if args.no_spatial_splits:
        cmd.append("--no-spatial-splits")
    rc, stdout, stderr = _run(cmd)
    compare_payload: dict[str, Any] | None = None
    if compare_out.is_file():
        compare_payload = json.loads(compare_out.read_text(encoding="utf-8"))
    passed = rc == 0 and compare_payload is not None and compare_payload.get("verdict") == "PASS"
    payload = {
        "schema": "v023-a2-canary-fieldtol-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "A2",
        "mode": "paired-compare",
        "method": "field-tolerance-vs-fresh-paired-baseline; no stored digest",
        "verdict": "PASS" if passed else "FAIL",
        "field_compare": str(compare_out),
        "field_compare_payload": compare_payload,
        "command": cmd,
        "returncode": rc,
        "stdout": stdout,
        "stderr": stderr,
        "acceptance": "all paired wrfout variables pass predeclared field tolerances and coverage is complete",
    }
    _write_json(args.out_json, payload)
    print(json.dumps({"gate": "A2", "verdict": payload["verdict"], "output": str(args.out_json)}, sort_keys=True))
    return 0 if passed else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--domains", nargs="+", default=["d01", "d02", "d03"])
    parser.add_argument("--mode", choices=["exact", "autotune-floor", "tolerance"], default="autotune-floor")
    parser.add_argument("--autotune-floor", type=float, default=1.0e-8)
    parser.add_argument("--init")
    parser.add_argument("--min-lead", type=int)
    parser.add_argument("--max-lead", type=int)
    parser.add_argument("--no-spatial-splits", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out-json", type=Path, default=HERE / "results" / "a2_canary_gate_fieldtol.json")
    args = parser.parse_args(argv)
    if not args.self_test and (args.baseline_dir is None or args.candidate_dir is None):
        parser.error("--baseline-dir and --candidate-dir are required unless --self-test is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    return run_paired_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
