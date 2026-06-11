#!/usr/bin/env python
"""V0.14 bdy-auditor: NORMAL_BDY_RELAX_STRENGTH sweep vs the depth-8 venting excess.

Tests the hypothesis that the perimeter relaxation strength governs the
Switzerland h36 strong-flow venting. Compares the depth-8 interior hourly excess
outflux (binding metric, CPU ref -74.5) for forecasts run at strength 20
(production) vs 60, plus the per-face h37 instantaneous-flux excess and the
domain-mean U/V bias.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
_HPG = importlib.util.spec_from_file_location("hpg", Path(__file__).with_name("switzerland_hpg_native_face_fix.py"))
hpg = importlib.util.module_from_spec(_HPG)
_HPG.loader.exec_module(hpg)  # type: ignore[union-attr]

CPU = hpg.CPU
PROBE = hpg.PROBE_ROOT
OUT_JSON = ROOT / "proofs/v014/switzerland_bdy_strength_sweep.json"

VARIANTS = {
    "prod_phys_tendf_s20": PROBE / "gpu_output_phys_tendf",
    "my_src_s20": PROBE / "gpu_output_bdyaud_s20",
    "my_src_s60": PROBE / "gpu_output_bdyaud_s60",
}


def _bias(base: Path, hour: int):
    lbl = ["2023-01-16_12:00:00", "2023-01-16_13:00:00", "2023-01-16_14:00:00"][hour - 36]
    def g(p, v):
        with Dataset(p / f"wrfout_d01_{lbl}") as d:
            return np.asarray(d.variables[v][0])
    out = {}
    for var in ("U", "V", "W", "MU"):
        diff = g(base, var) - g(CPU, var)
        out[var] = {"mean_bias": float(diff.mean()), "rmse": float(np.sqrt((diff ** 2).mean()))}
    return out


def main() -> int:
    out = {
        "schema": "v014_bdy_strength_sweep",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "depth": 8,
        "cpu_ref_outflux": -74.515,
    }
    c37 = hpg.budget_between(CPU, 36, CPU, 37, depth=8)
    c38 = hpg.budget_between(CPU, 36, CPU, 38, depth=8)
    for name, base in VARIANTS.items():
        rec = {"path": str(base)}
        if hpg.fn(base, 37).exists():
            g37 = hpg.budget_between(CPU, 36, base, 37, depth=8)
            rec["h37_excess"] = float(g37["net_influx_pa_per_cell_h"] - c37["net_influx_pa_per_cell_h"])
            rec["h37_dM"] = float(g37["dM_pa_per_cell_h"])
            rec["h37_bias"] = _bias(base, 37)
        if hpg.fn(base, 38).exists():
            g38 = hpg.budget_between(CPU, 36, base, 38, depth=8)
            rec["h38_excess"] = float(g38["net_influx_pa_per_cell_h"] - c38["net_influx_pa_per_cell_h"])
            rec["h38_bias"] = _bias(base, 38)
        out[name] = rec
    OUT_JSON.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"{'variant':24s} {'h37_exc':>9s} {'h38_exc':>9s} {'U_bias':>8s} {'V_bias':>8s} {'maxW_rmse':>9s}")
    for name in VARIANTS:
        r = out[name]
        if "h37_excess" not in r:
            print(f"{name:24s}  (no h37)")
            continue
        ub = r["h37_bias"]["U"]["mean_bias"]; vb = r["h37_bias"]["V"]["mean_bias"]; wr = r["h37_bias"]["W"]["rmse"]
        h38 = r.get("h38_excess", float("nan"))
        print(f"{name:24s} {r['h37_excess']:+9.2f} {h38:+9.2f} {ub:+8.4f} {vb:+8.4f} {wr:9.4f}")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
