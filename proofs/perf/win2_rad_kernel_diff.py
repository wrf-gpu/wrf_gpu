#!/usr/bin/env python
"""Win #2 isolation: does the RADIATION KERNEL output differ between the
dynamic-scalar clock and the legacy-static clock, on the real d02 GPU state?

Computes coszen, rrtmg_theta_tendency, and rrtmg_radiation_diagnostics (SWDOWN/GLW)
once on the real d02 initial state, BOTH ways, and reports max abs diff. This
isolates the radiation kernel from the 360-step acoustic amplification, so a
nonzero diff here is the true source of the integrated-forecast divergence.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.coupling import physics_couplers as pc


def _md(a, b):
    a = np.asarray(jax.device_get(a)).astype(np.float64)
    b = np.asarray(jax.device_get(b)).astype(np.float64)
    return float(np.nanmax(np.abs(a - b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default="proofs/perf/win2_rad_kernel_diff.json")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(
        run_id=args.run_id, run_root=Path(args.run_root),
        output_dir=Path("/tmp/win2_rk_out"), proof_dir=Path("/tmp/win2_rk_proof"),
        hours=1, domain="d02",
    )
    case, _ = _build_real_case(cfg)
    state, grid = case.state, case.grid
    jul, minute = pc._time_utc_parts(case.run_start)
    jul_s = jnp.asarray(jul, dtype=jnp.float64)
    min_s = jnp.asarray(minute, dtype=jnp.float64)
    lead = 0.0

    # coszen: legacy (host const) vs dynamic (scalar leaf)
    lat, lon = pc._grid_lat_lon(state.t_skin.shape, grid, state.t_skin.dtype)
    cz_leg = pc._compute_coszen(lat, lon, case.run_start, lead)
    cz_dyn = pc._compute_coszen(lat, lon, None, lead, julian_override=jul_s, utc_minute_override=min_s)
    d_cz = _md(cz_leg, cz_dyn)

    # rthraten (the actual radiative theta tendency applied every step)
    rt_leg = pc.rrtmg_theta_tendency(state, grid, time_utc=case.run_start, lead_seconds=lead)
    rt_dyn = pc.rrtmg_theta_tendency(state, grid, time_utc=None, lead_seconds=lead,
                                     julian_override=jul_s, utc_minute_override=min_s)
    jax.block_until_ready(rt_leg); jax.block_until_ready(rt_dyn)
    d_rt = _md(rt_leg, rt_dyn)

    # SWDOWN/GLW diagnostics
    dg_leg = pc.rrtmg_radiation_diagnostics(state, grid, time_utc=case.run_start, lead_seconds=lead)
    dg_dyn = pc.rrtmg_radiation_diagnostics(state, grid, time_utc=None, lead_seconds=lead,
                                            julian_override=jul_s, utc_minute_override=min_s)
    d_sw = _md(dg_leg.swdown, dg_dyn.swdown)
    d_glw = _md(dg_leg.glw, dg_dyn.glw)
    d_czd = _md(dg_leg.coszen, dg_dyn.coszen)

    payload = {
        "win": "2-rad-kernel-isolation",
        "coszen_max_abs_diff": d_cz,
        "rthraten_max_abs_diff": d_rt,
        "swdown_max_abs_diff": d_sw,
        "glw_max_abs_diff": d_glw,
        "diag_coszen_max_abs_diff": d_czd,
        "julian": jul, "utc_minute": minute,
        "all_zero": max(d_cz, d_rt, d_sw, d_glw, d_czd) == 0.0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
