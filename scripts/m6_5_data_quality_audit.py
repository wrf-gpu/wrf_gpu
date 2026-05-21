#!/usr/bin/env python
"""Run the M6.5 Gen2 d02 data-quality audit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.data_inventory import DEFAULT_GEN2_WRF_L3_ROOT, build_gen2_d02_inventory, load_inventory, write_json
from gpuwrf.validation.data_quality import (
    QUALITY_FIELDS,
    build_quality_audit,
    compare_boundary_replay_to_wrfout,
    validate_quality_audit,
)


DEFAULT_INVENTORY = ROOT / "artifacts" / "m6_5" / "gen2_d02_inventory.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "m6_5" / "gen2_d02_quality_audit.json"
DEFAULT_BOUNDARY_OUTPUT = ROOT / "artifacts" / "m6_5" / "gen2_boundary_replay_cross_check.json"


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else ROOT / target


def run(args: argparse.Namespace) -> dict[str, object]:
    inventory_path = _resolve(args.inventory)
    if inventory_path.exists():
        inventory = load_inventory(inventory_path)
    else:
        inventory = build_gen2_d02_inventory(args.root)
        write_json(inventory_path, inventory)

    fields = tuple(args.fields.split(",")) if args.fields else QUALITY_FIELDS
    audit = build_quality_audit(inventory, fields=fields)
    if args.boundary_replay and args.boundary_run:
        cross_check = compare_boundary_replay_to_wrfout(
            args.boundary_replay,
            args.boundary_run,
            valid_time=args.boundary_valid_time,
        )
        audit["boundary_replay_cross_check"] = cross_check
        write_json(_resolve(args.boundary_output), cross_check)
    validate_quality_audit(audit)
    output = _resolve(args.output)
    write_json(output, audit)
    return {
        "quality_audit": str(output),
        "run_count": audit["run_count"],
        "sampled_run_count": audit["sampled_run_count"],
        "status_counts": audit["status_counts"],
        "boundary_replay_cross_check": audit.get("boundary_replay_cross_check", {}).get("status"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_GEN2_WRF_L3_ROOT), help="Gen2 wrf_l3 root used if inventory is absent")
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fields", help="Comma-separated fields to audit")
    parser.add_argument("--boundary-replay", help="Optional d02 boundary replay zarr for AC4")
    parser.add_argument("--boundary-run", help="Gen2 run directory matching the replay zarr")
    parser.add_argument("--boundary-valid-time", help="Valid time to compare; defaults to replay lead 0")
    parser.add_argument("--boundary-output", default=str(DEFAULT_BOUNDARY_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    summary = run(parser.parse_args(argv))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
