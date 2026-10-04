#!/usr/bin/env python3
"""Assemble ``fast_case_qualification.json`` from the measured CPU arms (§5.3).

Only the CPU-side gates can be settled without the GPU. Everything that needs a
coordinated GPU window -- cold compile, cached load, the GPU arm, the complete
pair, and both representativeness gates, which are computed against a matched
short *production* trace -- is emitted with ``value: null`` and an explicit
``pending`` reason. The validator treats those as MISSING, which is the correct
state: contract §12 says missing evidence is never PASS.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_case import case_descriptor  # noqa: E402

GPU_PENDING = (
    "requires a coordinated GPU window (contract §13: dual-manager coordination "
    "with 0:2 and 0:3, then scripts/with_gpu_lock.sh); not measured on CPU"
)
TRACE_PENDING = (
    "computed against a matched short two-domain PRODUCTION GPU trace (§5.3); "
    "needs the baseline-census GPU window"
)


def measured(value, threshold, comparison="le", **extra):
    return {"value": value, "threshold": threshold, "comparison": comparison, **extra}


def pending(threshold, reason, comparison="le"):
    return {"value": None, "threshold": threshold, "comparison": comparison,
            "status": "PENDING", "reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cpu-glob", default="<DATA_ROOT>/wrf_gpu2/v025/m0/raw/cpu_arm_fastbind_r*.json"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = [
        json.loads(Path(p).read_text())
        for p in sorted(glob.glob(args.cpu_glob))
    ]
    arms = [a for a in arms if a.get("status") == "OK"]
    if not arms:
        raise SystemExit(f"no successful CPU arms matched {args.cpu_glob}")

    wall = [a["launcher_wallclock_seconds"] for a in arms]
    digests = sorted({a["content_digest"] for a in arms})
    nonfinite = sum(a["finiteness"]["non_finite_total"] for a in arms)

    obj = {
        "schema": "wrf_gpu2.v025.m0.fast_case_qualification.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_descriptor(),
        "status": "PARTIAL_CPU_ONLY",
        "note": (
            "CPU-side gates are measured; every GPU-side gate is PENDING and must "
            "not be read as satisfied. FAST-v025 is NOT yet qualified."
        ),
        "economy_gates": {
            "cold_compile_seconds": pending(600.0, GPU_PENDING),
            "cached_load_seconds": pending(60.0, GPU_PENDING),
            "gpu_arm_seconds": pending(300.0, GPU_PENDING),
            "cpu_arm_seconds": measured(
                st.median(wall), 300.0,
                n_repeats=len(wall), min=min(wall), max=max(wall),
                envelope="12 ranks, cores 16-27, --bind-to core",
            ),
            "pair_seconds_excl_lock_wait": pending(600.0, GPU_PENDING),
        },
        "diagnostic_gates": {
            "kernel_family_coverage": pending(0.95, TRACE_PENDING, comparison="ge"),
            "top_family_rank_spearman_rho": pending(0.80, TRACE_PENDING, comparison="ge"),
        },
        "finiteness": {
            "non_finite_total": nonfinite,
            "float_variables_checked": arms[0]["finiteness"]["float_variables_checked"],
            "arms_audited": len(arms),
        },
        "cadence_coverage": {
            "status": "PENDING",
            "reason": (
                "the d01 scheme/event inventory is enumerable from the namelist, but "
                "'every event triggered in the hour present' must be confirmed against "
                "an executed trace, not asserted from configuration"
            ),
            "d01_schemes_configured": {
                "mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4,
                "sf_sfclay_physics": 5, "sf_surface_physics": 4,
                "bl_pbl_physics": 5, "cu_physics": 1, "gwd_opt": 1,
            },
            "cadence_seconds": {"radt": 1800, "cudt": 300, "bldt": 0,
                                "history_interval": 3600},
            "missing_schemes": None,
            "missing_events": None,
        },
        "pair_completeness": {
            "fresh_cpu_arm_this_invocation": True,
            "fresh_gpu_arm_this_invocation": None,
            "comparator_result_present": None,
            "provenance_present": True,
            "arms_non_overlapping": None,
            "completeness_percent": None,
            "reason": GPU_PENDING,
        },
        "cpu_determinism": {
            "repeat_content_digests": digests,
            "deterministic": len(digests) == 1,
            "n_repeats": len(arms),
        },
        "cpu_arm_evidence": [
            {
                "label": a["label"],
                "run_dir": a["run_dir"],
                "wallclock_seconds": a["launcher_wallclock_seconds"],
                "timing_classes": a["timing_classes"],
                "contention": a.get("contention", {}).get("loadavg"),
            }
            for a in arms
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": str(args.out), "status": obj["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
