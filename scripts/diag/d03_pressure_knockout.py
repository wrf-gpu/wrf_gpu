#!/usr/bin/env python
"""DIAGNOSIS ONLY (2026-06-01 opus t2-bias bisection): short d03 GPU run that
dumps the PROGNOSTIC surface pressure + perturbation-geopotential profile each
forecast hour, to confirm the +2.7 kPa surface-pressure offset is generated
in-flight by the dycore (perturbation-pressure diagnosis after the
force_geopotential=False nested-boundary fix), not by the wrfout writer.

Does NOT touch production src/. Pins CPU 0-3. Short (default 3h) to save GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

ROOT = Path("/home/enric/src/wrf_gpu2")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from netCDF4 import Dataset

from d03_replay import build_l3_d03_daily_case
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational, _psfc_from_state

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CORB = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")


def corpus_psfc(tag: str) -> float:
    c = Dataset(CORB / f"wrfout_d03_{tag}")
    P = np.asarray(c.variables["P"][0], dtype=np.float64)
    PB = np.asarray(c.variables["PB"][0], dtype=np.float64)
    return float(np.nanmean((P + PB)[0]))


def corpus_phpert_profile(tag: str):
    c = Dataset(CORB / f"wrfout_d03_{tag}")
    PH = np.asarray(c.variables["PH"][0], dtype=np.float64)
    return PH.mean(axis=(1, 2))


def main() -> int:
    cfg = DailyPipelineConfig(
        run_id="20260521_18z_l3_24h_20260522T133443Z",
        hours=HOURS,
        output_dir=Path("/tmp/v010_d03_runs/d03_pknockout"),
        proof_dir=Path("/tmp/v010_d03_runs/d03_pknockout"),
        run_root=Path("/mnt/data/canairy_meteo/runs/wrf_l3"),
        score=False,
        domain="d03",
        dt_s=3.0,
        acoustic_substeps=10,
        radiation_cadence_steps=600,
    )
    case, _ = build_l3_d03_daily_case(cfg)
    state = case.state
    nl = case.namelist

    tags = [
        "2026-05-21_18:00:00",
        "2026-05-21_19:00:00",
        "2026-05-21_20:00:00",
        "2026-05-21_21:00:00",
    ]
    print("=== PROGNOSTIC surface-pressure knockout (d03, in-flight) ===")
    print(f"{'hour':>4} {'gpu_psfc':>11} {'corpus_psfc':>12} {'psfc_bias':>10}")
    psfc0 = float(np.mean(np.asarray(_psfc_from_state(state, nl.metrics))))
    print(f"{0:>4} {psfc0:>11.1f} {corpus_psfc(tags[0]):>12.1f} {psfc0 - corpus_psfc(tags[0]):>+10.1f}")

    for hour in range(1, HOURS + 1):
        state = run_forecast_operational(state, nl, 1.0)
        psfc = float(np.mean(np.asarray(_psfc_from_state(state, nl.metrics))))
        cps = corpus_psfc(tags[hour])
        print(f"{hour:>4} {psfc:>11.1f} {cps:>12.1f} {psfc - cps:>+10.1f}")
        # perturbation geopotential column profile bias
        php = np.asarray(state.ph_perturbation)
        php_prof = php.mean(axis=(1, 2))
        cphp = corpus_phpert_profile(tags[hour])
        sel = [0, 1, 5, 10, 20, len(php_prof) - 1]
        print("     ph_pert bias by level:",
              {k: round(float(php_prof[k] - cphp[k]), 1) for k in sel})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
