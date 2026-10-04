#!/usr/bin/env python3
"""Recompute T_ceiling and E_baseline from the raw fields (contract §8).

The point of --check is that the critic does not have to trust the numbers in
the file: it recomputes them from F_measured and R_achievable and fails if the
stored values disagree.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_lib import Report, close_enough, load_json

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    rep = Report("efficiency_model", args.path)
    obj = load_json(args.path, rep)
    if obj is None:
        return rep.emit()

    need = ["T_control_ms_per_step", "F_measured_flop_per_step",
            "R_achievable_flop_per_second", "T_ceiling_ms_per_step", "E_baseline"]
    values = {}
    for key in need:
        entry = obj.get(key)
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is None:
            rep.missing(f"field:{key}", "required by §8; an estimate is not a measurement")
        values[key] = value

    if obj.get("R_achievable_source") == "vendor_peak_fraction":
        rep.require("R_achievable_is_measured", False,
                    "§8 requires R_achievable measured on THIS card, not a % of vendor peak")
    elif obj.get("R_achievable_source") is None:
        rep.missing("R_achievable_is_measured", "R_achievable_source not stated")
    else:
        rep.require("R_achievable_is_measured",
                    obj["R_achievable_source"] == "measured_calibration_kernel",
                    f"source={obj['R_achievable_source']}")

    if args.check and all(values.get(k) is not None for k in need):
        ceiling = 1000.0 * values["F_measured_flop_per_step"] / values["R_achievable_flop_per_second"]
        rep.require("recompute:T_ceiling", close_enough(ceiling, values["T_ceiling_ms_per_step"], rel=1e-6),
                    f"recomputed {ceiling:.6g} ms vs stored {values['T_ceiling_ms_per_step']:.6g} ms")
        eff = values["T_ceiling_ms_per_step"] / values["T_control_ms_per_step"]
        rep.require("recompute:E_baseline", close_enough(eff, values["E_baseline"], rel=1e-6),
                    f"recomputed {eff:.6g} vs stored {values['E_baseline']:.6g}")

    reps = obj.get("calibration_repeats")
    if reps is None:
        rep.missing("calibration_repeats_ge_5", "§8 requires >= 5 uncontaminated repeats")
    else:
        rep.require("calibration_repeats_ge_5", reps >= 5, f"n={reps} vs >= 5")

    for key in ("ordinary", "event"):
        if (obj.get("step_classes") or {}).get(key) is None:
            rep.missing(f"step_classes:{key}", "§8 requires ordinary AND event steps reported")
        else:
            rep.require(f"step_classes:{key}", True, "present")

    if obj.get("uncertainty") is None:
        rep.missing("uncertainty_propagation", "§8 requires bootstrap 95% interval + propagation")
    else:
        rep.require("uncertainty_propagation", True, "present")
    return rep.emit()

if __name__ == "__main__":
    raise SystemExit(main())
