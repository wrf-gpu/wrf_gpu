#!/usr/bin/env python3
"""Validate the Pallas-on-sm_120 verdict against the frozen table (§11)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_lib import Report, load_json

VERDICTS = {"PALLAS_GREEN", "PALLAS_CAPABLE_NOT_YET_FAST", "PALLAS_BLOCKED"}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    rep = Report("pallas_sm120_viability", args.path)
    obj = load_json(args.path, rep)
    if obj is None:
        return rep.emit()

    verdict = obj.get("verdict")
    if verdict not in VERDICTS:
        rep.require("verdict:valid", False, f"{verdict!r} not in {sorted(VERDICTS)}")
        return rep.emit()
    rep.require("verdict:valid", True, verdict)

    rep.require("backend:native_mosaic_not_interpreter",
                obj.get("backend") == "native_mosaic_gpu",
                f"backend={obj.get('backend')!r}; interpreter results are diagnostic only")
    rep.require("operator:is_real_production_shaped",
                bool(obj.get("operator")) and obj.get("state_source_is_real") is True,
                f"operator={obj.get('operator')!r}")

    if obj.get("operator") == "mynn_condensation_edmf":
        rep.require("operator:substitution_justified",
                    bool(obj.get("substitution_proof_document")),
                    "§11 allows the MYNN substitute only with a written isolation proof")

    correctness = obj.get("correctness") or {}
    for key in ("cpu_reference_pass", "native_output_finite",
                "within_wrf_fp32_vs_fp64_envelope", "no_hidden_transfer_in_timed_loop"):
        value = correctness.get(key)
        if value is None:
            rep.missing(f"correctness:{key}", "§11 puts correctness before profiling")
        else:
            rep.require(f"correctness:{key}", bool(value), str(value))

    if verdict == "PALLAS_GREEN":
        perf = obj.get("performance") or {}
        speedup = perf.get("paired_warm_median_speedup")
        low95 = perf.get("bootstrap_lower_95")
        if speedup is None or low95 is None:
            rep.missing("green:speed_evidence", ">=1.30x median and >1.00x lower-95 required")
        else:
            rep.require("green:speedup_ge_1_30", speedup >= 1.30, f"{speedup:.4f} vs >= 1.30")
            rep.require("green:lower95_gt_1_00", low95 > 1.00, f"{low95:.4f} vs > 1.00")
        for key, minimum in (("warm_iterations_per_arm", 100), ("alternating_pairs", 5)):
            value = perf.get(key)
            if value is None:
                rep.missing(f"green:{key}", f">= {minimum} required")
            else:
                rep.require(f"green:{key}", value >= minimum, f"{value} vs >= {minimum}")
        rep.require("green:raw_profiler_artifacts", bool(perf.get("raw_artifacts")),
                    "no performance claim survives without profiler artifacts")

    if verdict == "PALLAS_BLOCKED":
        smoke = obj.get("typed_ffi_fallback_smoke")
        if smoke is None:
            rep.missing("blocked:fallback_smoke", "§11 requires one typed-FFI capability smoke")
        else:
            rep.require("blocked:fallback_smoke_ran", smoke.get("ran") is True, str(smoke))
        rep.require("blocked:toolchain_boundary_stated", bool(obj.get("toolchain_boundary")),
                    "report the exact failing boundary")
    return rep.emit()

if __name__ == "__main__":
    raise SystemExit(main())
