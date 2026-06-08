#!/usr/bin/env python3
"""Run ADR-029 station paired scoring on existing CPU/GPU wrfout directories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import parse_iso_time, write_json  # noqa: E402
from proofs.m20.paired_tost_scorer import aggregate_tost, paired_score  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", action="append", help="JSON object or path with case_id,cpu_dir,gpu_dir,domain,init,fh")
    ap.add_argument("--pairs-json", type=Path, help="JSON list of pair objects")
    ap.add_argument("--aemet-root", type=Path, default=Path("/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    pair_specs: list[dict[str, Any]] = []
    if args.pairs_json:
        payload = json.loads(args.pairs_json.read_text(encoding="utf-8"))
        pair_specs.extend(payload if isinstance(payload, list) else payload.get("pairs", []))
    for raw in args.pair or []:
        maybe_path = Path(raw)
        if maybe_path.is_file():
            pair_specs.append(json.loads(maybe_path.read_text(encoding="utf-8")))
        else:
            pair_specs.append(json.loads(raw))
    if not pair_specs:
        raise SystemExit("no pairs provided")

    case_scores = []
    excluded = []
    for spec in pair_specs:
        init = parse_iso_time(spec.get("init") or spec.get("init_time_utc"))
        if init is None:
            excluded.append({"spec": spec, "reason": "invalid init"})
            continue
        try:
            score = paired_score(
                case_id=spec["case_id"],
                cpu_dir=Path(spec["cpu_dir"]),
                gpu_dir=Path(spec["gpu_dir"]),
                domain=spec.get("domain", "d02"),
                init=init,
                fh=int(spec["fh"]),
                aemet_root=args.aemet_root,
            )
        except Exception as exc:
            excluded.append({"case_id": spec.get("case_id"), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        case_scores.append(score)

    aggregate = aggregate_tost(case_scores) if len(case_scores) >= 2 else None
    payload = {
        "schema": "CanaryStationTostExistingPairs",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "aemet_root": str(args.aemet_root),
        "n_scored": len(case_scores),
        "n_excluded": len(excluded),
        "case_scores": case_scores,
        "aggregate_tost": aggregate,
        "excluded": excluded,
        "caveat": "Uses ADR-029 scorer on existing wrfouts only; it does not launch forecasts.",
    }
    write_json(args.out, payload)
    print({"n_scored": len(case_scores), "n_excluded": len(excluded), "aggregate": aggregate is not None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
