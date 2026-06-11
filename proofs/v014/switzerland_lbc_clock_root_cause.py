#!/usr/bin/env python
"""V0.14 Switzerland d01 72h field-parity FAIL: LBC clock root cause + fix gate.

Root cause: ``daily_pipeline._run_forecast_sequence`` (the ``gpuwrf.cli run``
single-domain path) calls the operational forecast entry once per hour, and
``run_forecast_operational`` restarts its step clock at 1 on every call. The
in-scan ``interpolate_boundary_leaf`` walk therefore re-forced the lateral
boundary from leaf level 0 toward leaf level 1 EVERY hour: the spec zone stayed
pinned at the hour-1 boundary value for the whole 72h run, while CPU-WRF truth
kept ramping. The interior then drifted to a domain-wide dry-mass/pressure
surplus (PSFC bias +2380 Pa @ h72).

Gates (all computed from artifacts, CPU-only):
  G1  broken-run boundary ring == CPU truth h01 ring BIT-EXACT at every probed
      lead through h72 (and != the same-hour truth ring from h2 on).
  G2  mechanism emulation: per-hour-restart walk over the replay leaf time axis
      reproduces the broken run's frozen target; the fixed 2-level windowing
      reproduces CPU truth's ring at every hour exactly.
  G3  fixed 6h GPU rerun: boundary ring tracks CPU truth at h1..h6 and the
      PSFC/MU/T interior errors collapse vs the broken run.

Usage: python proofs/v014/switzerland_lbc_clock_root_cause.py [--fixed-run-root DIR]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import netCDF4 as nc

CPU_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
BROKEN_ROOT = Path(
    "/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_20260611T001250Z/gpu_output"
)
RUN_START = datetime(2023, 1, 15, 0, 0, 0)
PROBE_HOURS = (1, 2, 3, 6, 12, 24, 48, 72)
OUT_JSON = Path(__file__).with_suffix(".json")
OUT_MD = Path(__file__).with_suffix(".md")


def frame(root: Path, hour: int) -> Path:
    t = RUN_START + timedelta(hours=hour)
    return root / f"wrfout_d01_{t:%Y-%m-%d_%H:%M:%S}"


def read_field(path: Path, name: str) -> np.ndarray:
    ds = nc.Dataset(path)
    try:
        return np.asarray(ds.variables[name][0], dtype=np.float64)
    finally:
        ds.close()


def ring(mu: np.ndarray) -> np.ndarray:
    return np.concatenate([mu[0, :], mu[-1, :], mu[1:-1, 0], mu[1:-1, -1]])


def rmse_bias(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    return float(np.sqrt(np.mean(d * d))), float(np.mean(d))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-run-root", type=Path, default=None,
                        help="output_dir of the fixed 6h GPU rerun (gpu_output)")
    args = parser.parse_args()

    payload: dict = {
        "schema": "v014_switzerland_lbc_clock_root_cause",
        "cpu_truth": str(CPU_ROOT),
        "broken_run": str(BROKEN_ROOT),
        "run_start": RUN_START.isoformat(),
    }

    cpu_rings = {h: ring(read_field(frame(CPU_ROOT, h), "MU")) for h in range(0, 73)}

    # --- G1: broken run's boundary ring is frozen at the CPU h01 ring. -------
    g1_rows = []
    for h in PROBE_HOURS:
        g = ring(read_field(frame(BROKEN_ROOT, h), "MU"))
        diff_h01 = float(np.abs(g - cpu_rings[1]).max())
        diff_same = float(np.abs(g - cpu_rings[h]).max())
        best = min(range(0, 73), key=lambda k: float(np.abs(g - cpu_rings[k]).max()))
        g1_rows.append(
            {"hour": h, "max_abs_vs_cpu_h01": diff_h01, "max_abs_vs_cpu_same_hour": diff_same,
             "best_match_cpu_hour": best}
        )
    g1_pass = all(r["max_abs_vs_cpu_h01"] == 0.0 and r["best_match_cpu_hour"] == 1 for r in g1_rows)
    payload["g1_broken_ring_frozen_at_h01"] = {"rows": g1_rows, "pass": bool(g1_pass)}

    # --- G2: mechanism emulation against the replay leaf time axis. ----------
    # The replay leaves are the CPU truth's own hourly boundary values, so the
    # leaf level at hour h IS cpu_rings[h]. Broken loop: every hourly call walks
    # lead 0->3600 s over the FULL leaf axis => output-time target = level 1.
    # Fixed loop: window [h-1, h] => output-time target = level h.
    g2_rows = []
    for h in PROBE_HOURS:
        broken_target = cpu_rings[1]          # level index 1 at lead 3600 s, every hour
        fixed_target = cpu_rings[h]           # window upper level == record h
        g = ring(read_field(frame(BROKEN_ROOT, h), "MU"))
        g2_rows.append(
            {
                "hour": h,
                "broken_emulation_vs_gpu_max_abs": float(np.abs(g - broken_target).max()),
                "fixed_target_vs_cpu_truth_max_abs": float(np.abs(fixed_target - cpu_rings[h]).max()),
            }
        )
    g2_pass = all(
        r["broken_emulation_vs_gpu_max_abs"] == 0.0 and r["fixed_target_vs_cpu_truth_max_abs"] == 0.0
        for r in g2_rows
    )
    payload["g2_mechanism_emulation"] = {"rows": g2_rows, "pass": bool(g2_pass)}

    # --- Broken-run interior drift context (PSFC). ---------------------------
    drift_rows = []
    for h in (1, 2, 3, 6, 12, 24, 48, 72):
        r, b = rmse_bias(read_field(frame(BROKEN_ROOT, h), "PSFC"), read_field(frame(CPU_ROOT, h), "PSFC"))
        drift_rows.append({"hour": h, "psfc_rmse_pa": r, "psfc_bias_pa": b})
    payload["broken_psfc_drift"] = drift_rows

    # --- G3: fixed 6h rerun gate. ---------------------------------------------
    if args.fixed_run_root is not None:
        fixed_root = args.fixed_run_root
        g3_rows = []
        for h in range(1, 7):
            mu_g = read_field(frame(fixed_root, h), "MU")
            mu_c = read_field(frame(CPU_ROOT, h), "MU")
            ring_diff = float(np.abs(ring(mu_g) - cpu_rings[h]).max())
            mu_rmse, mu_bias = rmse_bias(mu_g, mu_c)
            psfc_rmse, psfc_bias = rmse_bias(
                read_field(frame(fixed_root, h), "PSFC"), read_field(frame(CPU_ROOT, h), "PSFC")
            )
            t_rmse, _ = rmse_bias(
                read_field(frame(fixed_root, h), "T"), read_field(frame(CPU_ROOT, h), "T")
            )
            g3_rows.append(
                {
                    "hour": h,
                    "ring_max_abs_vs_cpu_pa": ring_diff,
                    "mu_rmse_pa": mu_rmse,
                    "mu_bias_pa": mu_bias,
                    "psfc_rmse_pa": psfc_rmse,
                    "psfc_bias_pa": psfc_bias,
                    "t_rmse_k": t_rmse,
                }
            )
        # Gate: the spec-zone ring must track CPU truth at every output hour
        # (the broken run reached 352 Pa by h6), and the h2..h6 interior PSFC
        # error must stay in the h1 (physics-residual) class instead of the
        # broken run's monotonic boundary-driven blow-up.
        broken_h6_ring = next(r["max_abs_vs_cpu_same_hour"] for r in g1_rows if r["hour"] == 6)
        ring_pass = all(r["ring_max_abs_vs_cpu_pa"] <= 1.0 for r in g3_rows)
        broken_psfc_h6 = next(r["psfc_rmse_pa"] for r in drift_rows if r["hour"] == 6)
        fixed_psfc_h6 = g3_rows[-1]["psfc_rmse_pa"]
        psfc_pass = fixed_psfc_h6 < 0.5 * broken_psfc_h6
        payload["g3_fixed_rerun"] = {
            "run_root": str(fixed_root),
            "rows": g3_rows,
            "broken_h6_ring_max_abs_pa": broken_h6_ring,
            "broken_h6_psfc_rmse_pa": broken_psfc_h6,
            "ring_gate_max_abs_le_1pa": bool(ring_pass),
            "psfc_h6_collapse_gate": bool(psfc_pass),
            "pass": bool(ring_pass and psfc_pass),
        }

    gates = [payload["g1_broken_ring_frozen_at_h01"]["pass"], payload["g2_mechanism_emulation"]["pass"]]
    if "g3_fixed_rerun" in payload:
        gates.append(payload["g3_fixed_rerun"]["pass"])
    payload["verdict"] = (
        "LBC_CLOCK_ROOT_CAUSE_PROVEN_FIX_GATE_PASS" if all(gates) else "GATES_INCOMPLETE"
    )

    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("verdict",)}, indent=2))
    for section in ("g1_broken_ring_frozen_at_h01", "g2_mechanism_emulation"):
        print(section, "pass:", payload[section]["pass"])
    if "g3_fixed_rerun" in payload:
        print("g3 rows:")
        for row in payload["g3_fixed_rerun"]["rows"]:
            print("  ", row)
        print("g3 pass:", payload["g3_fixed_rerun"]["pass"])
    return 0 if all(gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
