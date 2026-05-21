#!/usr/bin/env python
"""Build the M6.5 Gen2 d02 wrfout inventory and optional subset manifest."""

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

from gpuwrf.io.data_inventory import (
    DEFAULT_COMPLETE_MIN_HOURS,
    DEFAULT_GEN2_WRF_L3_ROOT,
    build_gen2_d02_inventory,
    build_subset_manifest,
    validate_gen2_d02_inventory,
    write_json,
)


DEFAULT_OUTPUT = ROOT / "artifacts" / "m6_5" / "gen2_d02_inventory.json"


def _resolve_output(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else ROOT / target


def run(args: argparse.Namespace) -> dict[str, object]:
    inventory = build_gen2_d02_inventory(
        args.root,
        domain=args.domain,
        complete_min_hours=args.complete_min_hours,
    )
    validate_gen2_d02_inventory(inventory)
    output = _resolve_output(args.output)
    write_json(output, inventory)

    summary: dict[str, object] = {
        "inventory": str(output),
        "run_count": inventory["run_count"],
        "complete_run_count": inventory["complete_run_count"],
        "wrfout_d02_file_count": inventory["wrfout_d02_file_count"],
    }
    if args.start or args.end or args.min_hours is not None:
        min_hours = args.min_hours if args.min_hours is not None else DEFAULT_COMPLETE_MIN_HOURS
        tag = args.tag or f"{args.start or 'all'}_{args.end or 'all'}_min{min_hours}h"
        subset = build_subset_manifest(
            inventory,
            start=args.start,
            end=args.end,
            min_hours=min_hours,
            tag=tag,
            require_complete=not args.include_partial,
        )
        subset_path = _resolve_output(args.subset_output or Path("artifacts") / "m6_5" / f"gen2_d02_subset_{tag}.json")
        write_json(subset_path, subset)
        summary["subset"] = str(subset_path)
        summary["subset_run_count"] = subset["run_count"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_GEN2_WRF_L3_ROOT), help="Gen2 wrf_l3 root to scan read-only")
    parser.add_argument("--domain", default="d02", help="WRF domain to inventory")
    parser.add_argument("--complete-min-hours", type=int, default=DEFAULT_COMPLETE_MIN_HOURS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--start", help="Inclusive run start date filter as YYYYMMDD")
    parser.add_argument("--end", help="Inclusive run end date filter as YYYYMMDD")
    parser.add_argument("--min-hours", type=int, help="Minimum observed hours for subset selection")
    parser.add_argument("--tag", help="Subset tag used in the output filename")
    parser.add_argument("--subset-output", help="Explicit subset manifest output path")
    parser.add_argument("--include-partial", action="store_true", help="Allow partial runs in subset output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    summary = run(parser.parse_args(argv))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
