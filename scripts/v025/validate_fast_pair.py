#!/usr/bin/env python3
"""Validate FAST-v025 qualification: economy gates, fresh-CPU rule, provenance."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_lib import FAIL, MISSING, PASS, Report, load_json

# Contract §5.3, frozen before measurement.
GATES = {
    "cold_compile_seconds": 600.0,
    "cached_load_seconds": 60.0,
    "gpu_arm_seconds": 300.0,
    "cpu_arm_seconds": 300.0,
    "pair_seconds_excl_lock_wait": 600.0,
}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--require-fresh-cpu", action="store_true")
    args = ap.parse_args()
    rep = Report("fast_case_qualification", args.path)
    obj = load_json(args.path, rep)
    if obj is None:
        return rep.emit()

    for name, limit in GATES.items():
        measured = (obj.get("economy_gates") or {}).get(name)
        if measured is None or measured.get("value") is None:
            rep.missing(f"economy:{name}", f"no measured value (threshold <= {limit}s)")
            continue
        value = measured["value"]
        rep.require(f"economy:{name}", value <= limit,
                    f"{value:.3f}s vs <= {limit}s", value=value, threshold=limit)

    diag = obj.get("diagnostic_gates") or {}
    for name, limit, cmp_ in (
        ("kernel_family_coverage", 0.95, "ge"),
        ("top_family_rank_spearman_rho", 0.80, "ge"),
    ):
        entry = diag.get(name)
        if entry is None or entry.get("value") is None:
            rep.missing(f"diagnostic:{name}", f"no measured value (threshold {cmp_} {limit})")
            continue
        rep.require(f"diagnostic:{name}", entry["value"] >= limit,
                    f"{entry['value']} vs >= {limit}", value=entry["value"], threshold=limit)

    fin = obj.get("finiteness")
    if fin is None:
        rep.missing("finiteness", "no finiteness audit recorded")
    else:
        rep.require("finiteness:zero_non_finite", fin.get("non_finite_total") == 0,
                    f"non_finite_total={fin.get('non_finite_total')}")

    cad = obj.get("cadence_coverage")
    if cad is None:
        rep.missing("cadence_coverage", "no scheme/event coverage recorded")
    else:
        missing_schemes = cad.get("missing_schemes") or []
        missing_events = cad.get("missing_events") or []
        rep.require("cadence:all_d01_schemes_present", not missing_schemes, f"missing={missing_schemes}")
        rep.require("cadence:all_events_present", not missing_events, f"missing={missing_events}")

    if args.require_fresh_cpu:
        pair = obj.get("pair_completeness") or {}
        for key, why in (
            ("fresh_cpu_arm_this_invocation", "a cached CPU result can never satisfy §5.2"),
            ("fresh_gpu_arm_this_invocation", "the pair needs both arms from one invocation"),
            ("comparator_result_present", "a pair without a parity result is invalid"),
            ("provenance_present", "run ids + hashes must be recorded"),
            ("arms_non_overlapping", "CPU and GPU timed arms must not overlap"),
        ):
            value = pair.get(key)
            if value is None:
                rep.missing(f"pair:{key}", why)
            else:
                rep.require(f"pair:{key}", bool(value), why)
        pct = pair.get("completeness_percent")
        if pct is None:
            rep.missing("pair:completeness_percent", "exactly 100% required")
        else:
            rep.require("pair:completeness_percent", pct == 100, f"{pct}% vs exactly 100%")
    return rep.emit()

if __name__ == "__main__":
    raise SystemExit(main())
