#!/usr/bin/env python
"""V0.14 Switzerland venting CLOSE proof — theta-limiter ceiling fix (500->1000 K).

Binding metric: depth-8 interior control-surface hourly excess outflux vs the
CPU truth (target ~0; baseline phys_tendf -26.5 Pa/cell/h at h37).  Reuses the
established budget_between/field_metrics oracle from
switzerland_hpg_native_face_fix.py on the gpu_output_guardfix production run
(physics ON, guards ON, only _THETA_LIMITER_MAX_K 500->1000).

Also reports: interior warm-bias profile collapse (the root-cause tracer), the
vertical u-dipole band means, k43 theta>500K fraction restoration, and h39
stability metrics.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
_HPG_SPEC = importlib.util.spec_from_file_location(
    "switzerland_hpg_native_face_fix", HERE / "switzerland_hpg_native_face_fix.py"
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

CPU = hpg.CPU
ROOT = hpg.PROBE_ROOT
FIX = ROOT / "gpu_output_guardfix"
BASE = ROOT / "gpu_output_phys_tendf"
OUT = HERE / "switzerland_guardfix_venting_budget.json"

H = {36: "2023-01-16_12:00:00", 37: "2023-01-16_13:00:00", 38: "2023-01-16_14:00:00", 39: "2023-01-16_15:00:00"}


def _th_u(base: Path, hour: int):
    ds = Dataset(base / f"wrfout_d01_{H[hour]}")
    t = np.asarray(ds.variables["T"][0], dtype=np.float64)
    u = np.asarray(ds.variables["U"][0], dtype=np.float64)
    ds.close()
    return t, 0.5 * (u[:, :, :-1] + u[:, :, 1:])


def main() -> int:
    res: dict = {
        "schema": "v014_switzerland_guardfix_venting_budget",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fix": "_THETA_LIMITER_MAX_K 500.0 -> 1000.0 (operational_mode.py); production config otherwise identical to phys_tendf baseline",
    }
    # --- binding depth-8 budget ---
    for h0, h1, tag in [(36, 37, "h37"), (37, 38, "h37_h38_window"), (36, 38, "h38_cum")]:
        cpu = hpg.budget_between(CPU, h0, CPU, h1)
        start = CPU if h0 == 36 else FIX
        fix = hpg.budget_between(start, h0, FIX, h1)
        res[f"cpu_{tag}"] = cpu
        res[f"guardfix_{tag}"] = fix
        res[f"excess_outflux_pa_cell_h_{tag}"] = fix["net_influx_pa_per_cell_h"] - cpu["net_influx_pa_per_cell_h"]
    base37 = hpg.budget_between(CPU, 36, BASE, 37)
    cpu37 = hpg.budget_between(CPU, 36, CPU, 37)
    res["baseline_phys_tendf_excess_h37"] = base37["net_influx_pa_per_cell_h"] - cpu37["net_influx_pa_per_cell_h"]

    # --- field metrics + stability ---
    for h in (37, 38, 39):
        try:
            res[f"metrics_h{h}"] = hpg.field_metrics(FIX, h)
        except Exception as exc:  # noqa: BLE001
            res[f"metrics_h{h}"] = {"available": False, "error": str(exc)}

    # --- root-cause tracers ---
    tc37, uc37 = _th_u(CPU, 37)
    tg37, ug37 = _th_u(FIX, 37)
    tb37, ub37 = _th_u(BASE, 37)
    ny, nx = tc37.shape[1:]
    m8 = np.zeros((ny, nx), dtype=bool)
    m8[8:-8, 8:-8] = True
    res["dtheta_profile_h37_guardfix"] = [float(np.mean((tg37 - tc37)[k][m8])) for k in range(tc37.shape[0])]
    res["dtheta_profile_h37_baseline"] = [float(np.mean((tb37 - tc37)[k][m8])) for k in range(tc37.shape[0])]
    bands = {"k00": slice(0, 1), "k01_07": slice(1, 8), "k10_24": slice(10, 25), "k27_33": slice(27, 34)}
    res["du_bands_h37"] = {
        name: {
            "guardfix": float(np.mean(np.mean((ug37 - uc37)[sl], axis=0)[m8])),
            "baseline": float(np.mean(np.mean((ub37 - uc37)[sl], axis=0)[m8])),
        }
        for name, sl in bands.items()
    }
    th43_c = tc37[-1] + 300.0
    th43_g = tg37[-1] + 300.0
    th43_b = tb37[-1] + 300.0
    res["k43_frac_theta_gt_500_h37"] = {
        "cpu": float(np.mean(th43_c > 500.0)),
        "guardfix": float(np.mean(th43_g > 500.0)),
        "baseline": float(np.mean(th43_b > 500.0)),
    }

    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("dtheta_profile")}, indent=1))
    print("dtheta guardfix (k0,3,6,10,15,20,25,30,40):",
          [round(res["dtheta_profile_h37_guardfix"][k], 3) for k in (0, 3, 6, 10, 15, 20, 25, 30, 40)])
    print("dtheta baseline (k0,3,6,10,15,20,25,30,40):",
          [round(res["dtheta_profile_h37_baseline"][k], 3) for k in (0, 3, 6, 10, 15, 20, 25, 30, 40)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
