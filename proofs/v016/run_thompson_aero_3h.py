#!/usr/bin/env python
"""v0.16 coupled pre-validation — 3h Switzerland d01 run with mp_physics=28.

Runs the production daily-pipeline forecast path (the v0.15 kernel-probe case:
Switzerland d01 reinit-h36 replay, 128x128x44, dt=18 s) for N hours with a
selectable microphysics option, mirroring proofs/perf/v015/probe_ab_identity.py:

  --mp 8   : baseline standard Thompson (reference for the field gate)
  --mp 28  : aerosol-aware Thompson (candidate), with the WRF thompson_init
             climatological nwfa/nifa cold start applied to the initial state
             (use_aero_icbc=.false. self-init path, module_mp_thompson.F:493-558)

Artifacts (under proofs/v016/):
  aero3h_<tag>.json        — per-hour walls + finite summary + aerosol ranges
  aero3h_<tag>_state.npz   — every numeric state leaf (tiered field-gate input)

Field gate: proofs/perf/v015/compare_tiered_identity.py BASE CAND --hours 3.
GPU job — wrap with scripts/with_gpu_lock.sh.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent

# WRF thompson_init climatological aerosol constants (module_mp_thompson.F:94-97).
NA_CCN0, NA_CCN1 = 300.0e6, 50.0e6
NA_IN0, NA_IN1 = 1.5e6, 0.5e6
GRAVITY = 9.81


def _coldstart_aerosols(state):
    """Apply the WRF self-init aerosol profiles to a zero-aerosol state.

    Heights: mass-level z from the interface geopotential (state.ph/g), the
    same geometry the microphysics adapters use.  Vertical axis FIRST in the
    State layout.  Returns the state with nwfa/nifa filled (per kg).
    """

    z_if = np.asarray(state.ph, dtype=np.float64) / GRAVITY  # (nz+1, ny, nx)
    z = 0.5 * (z_if[:-1] + z_if[1:])  # (nz, ny, nx) mass levels
    h0 = z[0]  # lowest mass level (terrain-following)
    h_01 = np.where(h0 <= 1000.0, 0.8, np.where(h0 >= 2500.0, 0.01, 0.8 * np.cos(h0 * 0.001 - 1.0)))
    ni_ccn3 = -np.log(NA_CCN1 / NA_CCN0) / h_01
    ni_in3 = -np.log(NA_IN1 / NA_IN0) / h_01
    dz_eff = z - z[0]
    dz_eff[0] = z[1] - z[0]  # level 1 uses the level-2 offset (WRF lines 508/546)
    nwfa = NA_CCN1 + NA_CCN0 * np.exp(-(dz_eff / 1000.0) * ni_ccn3)
    nifa = NA_IN1 + NA_IN0 * np.exp(-(dz_eff / 1000.0) * ni_in3)
    return state.replace(
        nwfa=nwfa.astype(np.asarray(state.nwfa).dtype),
        nifa=nifa.astype(np.asarray(state.nifa).dtype),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mp", type=int, required=True, choices=(8, 28))
    ap.add_argument("--hours", type=int, default=3)
    ap.add_argument("--moist-adv", type=int, default=None,
                    help="override moist_adv_opt (e.g. 2 to ALSO exercise the "
                         "flux-form scalar advection of qv..qg + nwfa/nifa)")
    args = ap.parse_args()

    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=args.hours,
        output_dir=Path(f"/tmp/v016_aero/{args.tag}"),
        proof_dir=Path(f"/tmp/v016_aero/{args.tag}/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, _run_dir = dp._build_real_case(config)
    overrides = {"mp_physics": args.mp}
    if args.moist_adv is not None:
        overrides["moist_adv_opt"] = int(args.moist_adv)
    namelist = dataclasses.replace(case.namelist, **overrides)
    state = case.state
    if args.mp == 28:
        # Canonical init-time cold start (WRF thompson_init self-init path).
        from gpuwrf.coupling.physics_couplers import thompson_aero_coldstart_init

        state = thompson_aero_coldstart_init(state)
        cs = _coldstart_aerosols(case.state)  # closed-form cross-check
        assert np.allclose(np.asarray(state.nwfa), np.asarray(cs.nwfa), rtol=1e-10), \
            "coldstart nwfa mismatch between coupler and probe closed forms"

    boundary_leaves = dp._capture_boundary_leaves(state, namelist)
    window_s = dp._boundary_window_cadence_s(namelist)
    record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)

    walls = []
    for hour in range(1, args.hours + 1):
        st_in = (
            dp._rewindow_boundary_leaves(
                state, boundary_leaves, segment_start_s=(hour - 1) * 3600.0,
                record_cadence_s=record_s, window_s=window_s,
            )
            if boundary_leaves
            else state
        )
        t0 = time.perf_counter()
        state = dp._default_forecast_fn(st_in, namelist, 1.0)
        walls.append(round(time.perf_counter() - t0, 3))
        print(f"{args.tag} hour{hour}: {walls[-1]}s", flush=True)

    summary = dp.finite_summary(state)
    aero = {}
    for leaf in ("nwfa", "nifa", "Nc"):
        arr = np.asarray(getattr(state, leaf), dtype=np.float64)
        aero[leaf] = {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}

    leaves = {}
    for name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            leaves[name] = arr
    np.savez_compressed(HERE / f"aero3h_{args.tag}_state.npz", **leaves)

    payload = {
        "schema": "V016ThompsonAero3h",
        "tag": args.tag,
        "mp_physics": args.mp,
        "moist_adv_opt": int(namelist.moist_adv_opt),
        "case": "Switzerland d01 reinit h36 replay, 128x128x44, dt=18s",
        "per_hour_wall_s": walls,
        "steady_ms_per_step": round(walls[-1] / 200.0 * 1000.0, 2),
        "all_finite": bool(summary["all_finite"]),
        "aerosol_ranges": aero,
        "n_leaves_dumped": len(leaves),
    }
    (HERE / f"aero3h_{args.tag}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
