#!/usr/bin/env python3
"""How far does the ice-balance fix move a REAL domain, not just the failing column?

The fix is WRF-faithful and correct, but it is emphatically NOT a no-op on real
trajectories: any ice-bearing cell whose (qi, Ni) pair sits outside WRF's
5-300 um band had an inflated ice fall speed before the fix.  This quantifies
that over the full retained d01 step-1147 state so the manager can size the
accepted-trajectory re-baseline before the terminal gate.

Run (CPU only):
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    taskset -c 13,14,15,29,30,31 nice -n 15 ionice -c 3 \
    python scripts/v0234_opus_ice_balance_baseline_impact.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

EXACT_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_gpt_late_ni_df242a850d6ebab1_exact"
)
INPUT_CARRY = EXACT_DIR / "last-green-advance-input-d01-step-1147.pkl"

R1, R2 = 1.0e-12, 1.0e-6
AM_I = np.pi * 890.0 / 6.0
CIE2, CIG2, OIG1, OIG2, OBMI = 4.0, 6.0, 1.0, 1.0 / 6.0, 1.0 / 3.0
NI_CEIL = 999.0e3


def balance(ri: np.ndarray, ni: np.ndarray) -> np.ndarray:
    xri = np.maximum(ri, R1)
    xni = np.maximum(ni, R2)
    lami = (AM_I * CIG2 * OIG1 * xni / xri) ** OBMI
    xdi = CIE2 / lami
    small = np.minimum(NI_CEIL, OIG2 * xri / AM_I * (CIE2 / 5.0e-6) ** 3.0)
    large = OIG2 * xri / AM_I * (CIE2 / 300.0e-6) ** 3.0
    out = np.where(xdi < 5.0e-6, small, np.where(xdi > 300.0e-6, large, xni))
    return np.minimum(np.maximum(np.where(xri > R1, out, 0.0), 0.0), NI_CEIL)


def main() -> int:
    from gpuwrf.coupling.physics_couplers import _thompson_column_from_state
    from gpuwrf.physics.thompson_column import AV_I, BV_I, CIG3, RHO_NOT

    with INPUT_CARRY.open("rb") as stream:
        carry = pickle.load(stream)
    state = _thompson_column_from_state(carry.state)

    qi = np.asarray(state.qi, dtype=np.float64)
    Ni = np.asarray(state.Ni, dtype=np.float64)
    rho = np.asarray(state.rho, dtype=np.float64)

    active = qi > R1
    ri = np.maximum(qi * rho, R1)
    ni_raw = np.maximum(Ni * rho, R2)
    ni_bal = np.maximum(balance(ri, ni_raw), R2)

    rhof = np.sqrt(float(RHO_NOT) / np.maximum(rho, R1))

    def vti(ni):
        lami = (AM_I * CIG2 * OIG1 * ni / ri) ** OBMI
        return rhof * float(AV_I) * float(CIG3) * float(OIG2) * (1.0 / lami) ** float(BV_I)

    vt_before = np.where(active, vti(ni_raw), 0.0)
    vt_after = np.where(active, vti(ni_bal), 0.0)

    a = vt_before[active]
    b = vt_after[active]
    changed = np.abs(a - b) > 1e-12 * np.maximum(np.abs(a), 1.0)

    report: dict[str, Any] = {
        "schema": "gpuwrf.v0234.opus-ice-balance-baseline-impact.v1",
        "source_state": {"path": str(INPUT_CARRY), "domain": "d01", "native_step": 1147},
        "ice_bearing_cells": int(active.sum()),
        "cells_with_changed_ice_fall_speed": int(changed.sum()),
        "fraction_of_ice_cells_changed": float(changed.mean()) if a.size else 0.0,
        "ice_fall_speed_m_s": {
            "before_max": float(a.max()) if a.size else None,
            "after_max": float(b.max()) if b.size else None,
            "before_mean": float(a.mean()) if a.size else None,
            "after_mean": float(b.mean()) if b.size else None,
            "before_median": float(np.median(a)) if a.size else None,
            "after_median": float(np.median(b)) if b.size else None,
        },
        "speed_reduction_factor": {
            "max": float((a / np.maximum(b, 1e-30)).max()) if a.size else None,
            "median_over_changed": float(np.median((a / np.maximum(b, 1e-30))[changed]))
            if changed.any()
            else None,
        },
        "cells_exceeding_cap_before": int(
            (np.floor(54.0 * vt_before / np.maximum(np.asarray(state.dz, dtype=np.float64), 1.0) + 1.0) > 16).sum()
        ),
        "interpretation": (
            "The fix is correct and WRF-faithful, but it is NOT a no-op on real "
            "trajectories: it lowers the ice fall speed in every out-of-band ice cell. "
            "The accepted-trajectory baseline must therefore be re-established after "
            "this fix; a corrected trajectory will not reproduce the prior one bitwise."
        ),
    }
    report["canonical_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

    out = Path(__file__).resolve().parents[1] / (
        ".agent/sprints/2026-07-21-v0234-opus-late-ni-fix/BASELINE_IMPACT.json"
    )
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
