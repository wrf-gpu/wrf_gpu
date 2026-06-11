#!/usr/bin/env python
"""V0.14 venting-residual fix: 2h short-forecast venting budget (binding gate).

Compares the depth-8 interior hourly dry-mass budget of the physics-tendf-fold
forecast (`gpu_output_phys_tendf`) against the CPU truth and the prior GPU
baselines, using the established `switzerland_hpg_native_face_fix` budget tool
(identical control surface as every prior venting number).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_HPG_SPEC = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

OUT_JSON = ROOT / "proofs/v014/switzerland_venting_budget_phys_tendf.json"

CPU = hpg.CPU
NEW = hpg.PROBE_ROOT / "gpu_output_phys_tendf"
BASELINES = {
    "old_ec4d6769": hpg.BASELINE_GPU,
    "hypso_3d0b439c": hpg.FIX_GPU,
    "acoustic_fix": hpg.PROBE_ROOT / "gpu_output_acoustic_substep_fix",
    "awd_fix_open_b14b5f17": hpg.PROBE_ROOT / "gpu_output_awd_fix_open",
    "phys_tendf_fold_THIS": NEW,
    # same fix + the already-landed (flag-gated) WRF specified band cadence
    # (GPUWRF_SPECIFIED_BDY_CADENCE=1 GPUWRF_SPECIFIED_ADV_DEGRADE=1): tests
    # whether the band lane sets the depth-8 control-surface winds (the
    # invariant -26.5 excess surviving every interior fix).
    "phys_tendf_speccad_THIS": hpg.PROBE_ROOT / "gpu_output_phys_tendf_speccad",
}


def main() -> int:
    out = {
        "schema": "v014_switzerland_venting_budget_phys_tendf",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_reference": str(CPU),
        "depth": 8,
    }
    cpu37 = hpg.budget_between(CPU, 36, CPU, 37, depth=8)
    out["cpu_budget_h36_h37"] = cpu37
    excesses = {}
    for name, base in BASELINES.items():
        if not hpg.fn(base, 37).exists():
            out[f"{name}_h37"] = {"available": False, "path": str(base)}
            continue
        b = hpg.budget_between(CPU, 36, base, 37, depth=8)
        out[f"{name}_h37"] = b
        excesses[name] = float(b["net_influx_pa_per_cell_h"] - cpu37["net_influx_pa_per_cell_h"])
    out["excess_outflux_pa_per_cell_h_h37"] = excesses

    if hpg.fn(NEW, 38).exists():
        cpu38 = hpg.budget_between(CPU, 36, CPU, 38, depth=8)
        new38 = hpg.budget_between(CPU, 36, NEW, 38, depth=8)
        out["h36_h38"] = {
            "cpu": cpu38,
            "phys_tendf_fold": new38,
            "excess_per_h": float(
                (new38["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"])
            ),
        }
        awd38_base = hpg.PROBE_ROOT / "gpu_output_awd_fix_open"
        if hpg.fn(awd38_base, 38).exists():
            awd38 = hpg.budget_between(CPU, 36, awd38_base, 38, depth=8)
            out["h36_h38"]["awd_fix_open_excess_per_h"] = float(
                awd38["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"]
            )
        spc_base = hpg.PROBE_ROOT / "gpu_output_phys_tendf_speccad"
        if hpg.fn(spc_base, 38).exists():
            spc38 = hpg.budget_between(CPU, 36, spc_base, 38, depth=8)
            out["h36_h38"]["phys_tendf_speccad_excess_per_h"] = float(
                spc38["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"]
            )
            out["metrics_h37_speccad"] = hpg.field_metrics(spc_base, 37)
            out["metrics_h38_speccad"] = hpg.field_metrics(spc_base, 38)

    out["metrics_h37"] = hpg.field_metrics(NEW, 37)
    if hpg.fn(NEW, 38).exists():
        out["metrics_h38"] = hpg.field_metrics(NEW, 38)
    out["metrics_h37_awd_baseline"] = hpg.field_metrics(hpg.PROBE_ROOT / "gpu_output_awd_fix_open", 37)

    OUT_JSON.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(json.dumps({
        "excess_h37": excesses,
        "h38": out.get("h36_h38", {}).get("excess_per_h"),
        "h38_awd": out.get("h36_h38", {}).get("awd_fix_open_excess_per_h"),
        "metrics_h37": out["metrics_h37"],
    }, indent=1, default=float))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
