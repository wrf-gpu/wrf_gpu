#!/usr/bin/env python3
"""Sprint S6b ACTIVATION + SYSTEM gate — prognostic Noah-MP in the operational path.

Runs ONE segmented GPU forecast on a real corpus case with prognostic Noah-MP
ACTIVATED over land (``use_noahmp=True``), and the SAME case with the legacy bulk
surface path (``use_noahmp=False``), then scores both against the nightly CPU-WRF
wrfout. Single GPU consumer, ONE forecast carry at a time, OOM-safe segmented
(block_until_ready between segments). The land carry is warm-started from the
corpus wrfinput via a faithful WRF NOAHMP_INIT replica (io.noahmp_land_init).

GATES (the standalone-replacement equivalence milestone):
  * STABILITY: prognostic land activates with no drift/blow-up (all-finite, T2 in
    a physical band) over the run.
  * HFX COLLAPSE: midday LAND HFX moves toward the corpus (the bulk path
    over-fluxes daytime-land HFX 1.4-1.6x; Noah-MP canopy balance should collapse
    that excess).
  * NO-REGRESSION: ocean/water T2 byte-equal to the bulk path (Noah-MP is
    land-masked); U10/V10 winds no-regression.
  * SKILL: T2 RMSE/bias vs corpus; beats persistence.

Usage (manager, sequenced on GPU):
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
    python proofs/noahmp/s6b_activate_validate.py \
      --run-dir /mnt/data/.../<L3 run> --domain d03 --hours 12 \
      --out proofs/noahmp/s6b_d03_short.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _read_wrf(path, fields):
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    return read_wrfout_file(path, fields=fields)["fields"]


def _rmse_bias(g, t, m=None):
    g = np.asarray(g, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    if m is not None:
        g = g[m]; t = t[m]
    d = g - t
    finite = np.isfinite(d)
    d = d[finite]
    if d.size == 0:
        return {"rmse": None, "bias": None, "mae": None, "n": 0}
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "bias": float(np.mean(d)),
            "mae": float(np.mean(np.abs(d))), "n": int(d.size)}


def run_forecast(run_dir, domain, *, use_noahmp, leads, segment_steps, dt_s,
                 acoustic_substeps, radiation_cadence_steps):
    """Advance ONE segmented GPU carry to the max lead; snapshot M9 diagnostics +
    HFX at each lead. Returns per-lead GPU surface fields and meta."""
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk, _enforce_operational_precision, compute_m9_diagnostics,
    )
    from gpuwrf.runtime.operational_state import initial_operational_carry

    run_dir = Path(run_dir)
    cfg = DailyPipelineConfig(
        run_id=run_dir.name, run_root=run_dir.parent, domain=domain, dt_s=dt_s,
        acoustic_substeps=acoustic_substeps, radiation_cadence_steps=radiation_cadence_steps,
    )
    case, rdir = _build_real_case(cfg)
    time_utc = case.run_start
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=radiation_cadence_steps, time_utc=time_utc,
    )

    noahmp_land = None
    noahmp_rad = None
    init_meta = None
    if use_noahmp:
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
        from gpuwrf.runtime.operational_mode import noahmp_initial_rad
        julian = float(time_utc.timetuple().tm_yday)
        noahmp_land, static, init_meta = build_noahmp_land_state(run_dir, domain)
        energy_params, rad_params, nroot = build_noahmp_params(static)
        noahmp_rad = noahmp_initial_rad(
            _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64)))
        nl = dataclasses.replace(
            nl, use_noahmp=True, noahmp_static=static,
            noahmp_energy_params=energy_params, noahmp_rad_params=rad_params,
            noahmp_nroot=nroot, noahmp_julian=julian, noahmp_yearlen=365.0,
        )

    cadence = int(nl.radiation_cadence_steps)
    seg = int(segment_steps) if segment_steps else cadence
    dt = float(nl.dt_s)
    lead_steps = {h: int(round(h * 3600.0 / dt)) for h in leads}

    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64)),
        noahmp_land=noahmp_land, noahmp_rad=noahmp_rad,
    )

    out = {}
    timings = {}
    t0 = time.time()
    start = 1
    for lead_h in leads:
        target = lead_steps[lead_h]
        while start <= target:
            n = min(seg, target - start + 1)
            carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                                   n_steps=int(n), cadence=cadence)
            jax.block_until_ready(carry.state.theta)
            start += n
        lead_seconds = float(lead_h) * 3600.0
        diags = compute_m9_diagnostics(
            carry.state, nl, lead_seconds,
            noahmp_land=carry.noahmp_land, noahmp_rad=carry.noahmp_rad,
        )
        out[lead_h] = {
            "T2": np.asarray(jax.device_get(diags.t2), dtype=np.float64),
            "U10": np.asarray(jax.device_get(diags.u10), dtype=np.float64),
            "V10": np.asarray(jax.device_get(diags.v10), dtype=np.float64),
            "HFX": np.asarray(jax.device_get(diags.hfx), dtype=np.float64),
            "LH": np.asarray(jax.device_get(diags.lh), dtype=np.float64),
            "TSK": np.asarray(jax.device_get(diags.tsk), dtype=np.float64),
            "all_finite": bool(np.all(np.isfinite(
                np.asarray(jax.device_get(carry.state.theta))))),
        }
        timings[lead_h] = round(time.time() - t0, 1)
    return {"valid_base": time_utc, "run_dir": str(rdir), "fields": out,
            "timings_cumulative_s": timings, "init_meta": init_meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--domain", default="d03")
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--leads", type=int, nargs="+", default=None)
    ap.add_argument("--segment-steps", type=int, default=200)
    ap.add_argument("--dt-s", type=float, default=None)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=600)
    ap.add_argument("--bulk-also", action="store_true",
                    help="also run the legacy bulk path for no-regression + collapse delta")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # d03 1km uses a shorter dt; default per domain (matches v0.1.0 d03 runs).
    dt_s = a.dt_s if a.dt_s is not None else (3.0 if a.domain == "d03" else 10.0)
    leads = a.leads if a.leads is not None else list(range(1, a.hours + 1))
    run_dir = Path(a.run_dir)
    domain = a.domain

    kw = dict(leads=leads, segment_steps=a.segment_steps, dt_s=dt_s,
              acoustic_substeps=a.acoustic_substeps,
              radiation_cadence_steps=a.radiation_cadence_steps)

    t_start = time.time()
    nm = run_forecast(run_dir, domain, use_noahmp=True, **kw)
    bulk = run_forecast(run_dir, domain, use_noahmp=False, **kw) if a.bulk_also else None

    # corpus truth + masks
    from gpuwrf.io.gen2_accessor import Gen2Run
    run = Gen2Run(run_dir)
    wi = {n: np.squeeze(np.asarray(run.load_wrfinput(domain, n, lazy=False)))
          for n in ("XLAND",)}
    xland = wi["XLAND"]
    is_land = xland < 1.5
    is_water = ~is_land
    base = nm["valid_base"]
    rdir = Path(nm["run_dir"])

    per_lead = []
    for h in leads:
        valid = base + timedelta(hours=h)
        wrfout = rdir / f"wrfout_{domain}_{valid:%Y-%m-%d_%H:%M:%S}"
        if not wrfout.is_file():
            per_lead.append({"lead_h": h, "note": "no CPU truth", "wrfout": str(wrfout)})
            continue
        w = _read_wrf(wrfout, ("T2", "U10", "V10", "HFX", "LH"))
        wf = {k: np.squeeze(np.asarray(v, dtype=np.float64)) for k, v in w.items()}
        g = nm["fields"][h]
        row = {
            "lead_h": h,
            "valid": valid.isoformat(),
            "all_finite": g["all_finite"],
            "noahmp": {
                "T2_full": _rmse_bias(g["T2"], wf["T2"]),
                "T2_land": _rmse_bias(g["T2"], wf["T2"], is_land),
                "T2_water": _rmse_bias(g["T2"], wf["T2"], is_water),
                "U10_full": _rmse_bias(g["U10"], wf["U10"]),
                "V10_full": _rmse_bias(g["V10"], wf["V10"]),
                "HFX_land": _rmse_bias(g["HFX"], wf["HFX"], is_land),
                "HFX_water": _rmse_bias(g["HFX"], wf["HFX"], is_water),
                "LH_land": _rmse_bias(g["LH"], wf["LH"], is_land),
                "gpu_HFX_land_mean": float(np.mean(g["HFX"][is_land])),
                "wrf_HFX_land_mean": float(np.mean(wf["HFX"][is_land])),
            },
        }
        if bulk is not None:
            b = bulk["fields"][h]
            row["bulk"] = {
                "T2_land": _rmse_bias(b["T2"], wf["T2"], is_land),
                "T2_water": _rmse_bias(b["T2"], wf["T2"], is_water),
                "HFX_land": _rmse_bias(b["HFX"], wf["HFX"], is_land),
                "gpu_HFX_land_mean": float(np.mean(b["HFX"][is_land])),
            }
            # no-regression: ocean T2/U10/V10 should be byte-equal (land-masked switch)
            row["ocean_T2_max_abs_diff"] = float(np.max(np.abs(
                (g["T2"] - b["T2"])[is_water]))) if is_water.any() else 0.0
            row["ocean_U10_max_abs_diff"] = float(np.max(np.abs(
                (g["U10"] - b["U10"])[is_water]))) if is_water.any() else 0.0
            row["ocean_V10_max_abs_diff"] = float(np.max(np.abs(
                (g["V10"] - b["V10"])[is_water]))) if is_water.any() else 0.0
            row["hfx_land_collapse_W"] = (
                row["bulk"]["gpu_HFX_land_mean"] - row["noahmp"]["gpu_HFX_land_mean"])
        per_lead.append(row)

    proof = {
        "proof": "S6b prognostic Noah-MP ACTIVATION + system validation",
        "kind": ("real corpus operational forecast with prognostic Noah-MP over land "
                 "vs nightly CPU-WRF wrfout; land warm-started from corpus wrfinput via "
                 "faithful NOAHMP_INIT; NOT a self-compare"),
        "run_dir": str(rdir), "domain": domain, "dt_s": dt_s,
        "hours": a.hours, "leads": leads,
        "n_land_cells": int(np.sum(is_land)), "n_water_cells": int(np.sum(is_water)),
        "init_meta": nm["init_meta"],
        "per_lead": per_lead,
        "timings_noahmp_s": nm["timings_cumulative_s"],
        "timings_bulk_s": bulk["timings_cumulative_s"] if bulk else None,
        "wall_total_s": round(time.time() - t_start, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(proof, indent=2, default=str) + "\n")
    print(f"proof -> {a.out}")
    # console summary
    for row in per_lead:
        if "noahmp" not in row:
            print(f"  lead {row['lead_h']:>3}h: {row.get('note')}")
            continue
        nm_l = row["noahmp"]
        line = (f"  lead {row['lead_h']:>3}h finite={row['all_finite']} "
                f"T2_land rmse={nm_l['T2_land']['rmse']:.3f} bias={nm_l['T2_land']['bias']:+.3f} "
                f"HFX_land gpu={nm_l['gpu_HFX_land_mean']:.1f} wrf={nm_l['wrf_HFX_land_mean']:.1f}")
        if "bulk" in row:
            line += (f" | bulk HFX_land={row['bulk']['gpu_HFX_land_mean']:.1f} "
                     f"collapse={row['hfx_land_collapse_W']:+.1f}W "
                     f"oceanT2dmax={row['ocean_T2_max_abs_diff']:.2e}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
