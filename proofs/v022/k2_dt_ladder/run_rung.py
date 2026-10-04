"""K2 dt/n_sound CFL-ladder -- single-rung runner (TEST INFRA ONLY).

Runs the Switzerland 128x128 single-domain case through the PRODUCTION
daily pipeline with an overridden model timestep (dt_s) and acoustic substep
count (n_sound), writes wrfout, then computes:

  * max-Courant (Cx/Cy/Cz) per output frame  -> CFL-headroom evidence
  * finite + bounded checks on the final wrfout
  * column dry-mass conservation drift across frames (MU+MUB integral)
  * s/step wall  (from the pipeline's per-hour wall + steps/hour)

Knobs (the lossless K2 levers, config-level only, NO core dycore edit):
  K2_DT_S            -> DailyPipelineConfig.dt_s          (root model dt)
  K2_ACOUSTIC_SUBSTEPS -> DailyPipelineConfig.acoustic_substeps (n_sound)

Everything else (fp64 default, physics, boundaries) is the shipped real-case
config from _build_real_case. This does NOT flip any default and does NOT
expect bit-identity (different dt => different result); acceptance is
CFL-safe + finite + bounded + conserving + within strict operational
tolerance vs the R1 baseline.

Usage:
  K2_INPUT_DIR=... K2_HOURS=3 K2_DT_S=10 K2_ACOUSTIC_SUBSTEPS=10 \
    K2_TAG=R1 python proofs/v022/k2_dt_ladder/run_rung.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))

from courant_diag import courant_for_run  # noqa: E402


def _finite_bounded(wrfout_paths) -> dict:
    """Finite + physical-bound check on the final wrfout frame."""
    from netCDF4 import Dataset

    last = sorted(wrfout_paths)[-1]
    report = {"file": str(last), "all_finite": True, "violations": []}
    bounds = {
        "U": 200.0,
        "V": 200.0,
        "W": 100.0,
        "T": 1000.0,  # perturbation potential temperature (K), generous
        "QVAPOR": 0.1,  # kg/kg
    }
    with Dataset(str(last)) as ds:
        for name in ("U", "V", "W", "T", "QVAPOR", "PH", "MU"):
            if name not in ds.variables:
                continue
            arr = np.asarray(ds.variables[name][:], dtype=np.float64)
            finite = bool(np.all(np.isfinite(arr)))
            report.setdefault("max_abs", {})[name] = float(np.nanmax(np.abs(arr)))
            if not finite:
                report["all_finite"] = False
                report["violations"].append(f"{name}: non-finite")
            if name in bounds and finite:
                amax = float(np.nanmax(np.abs(arr)))
                if amax > bounds[name]:
                    report["violations"].append(
                        f"{name}: |max|={amax:.3g} > bound {bounds[name]:g}"
                    )
        if "QVAPOR" in ds.variables:
            qv = np.asarray(ds.variables["QVAPOR"][:], dtype=np.float64)
            if float(np.nanmin(qv)) < -1.0e-9:
                report["violations"].append(
                    f"QVAPOR: min={float(np.nanmin(qv)):.3g} < 0"
                )
    report["bounded"] = len(report["violations"]) == 0
    return report


def _mass_conservation(wrfout_paths) -> dict:
    """Column dry-mass (MU+MUB) integral drift across frames -- conservation proxy.

    The Switzerland case has specified lateral boundaries (open domain), so exact
    conservation is NOT expected; we report the relative drift of the domain-summed
    dry-air column mass to catch a runaway mass pump (the v0.21 boundary pathology).
    """
    from netCDF4 import Dataset

    totals = []
    for p in sorted(wrfout_paths):
        with Dataset(str(p)) as ds:
            mu = np.asarray(ds.variables["MU"][:], dtype=np.float64)
            mub = np.asarray(ds.variables["MUB"][:], dtype=np.float64)
            col = mu + mub  # (Time, y, x) Pa column dry mass
            for t in range(col.shape[0]):
                totals.append(float(np.nansum(col[t])))
    if not totals:
        return {"frames": 0}
    t0 = totals[0]
    rel = [(v - t0) / t0 if t0 != 0 else 0.0 for v in totals]
    return {
        "frames": len(totals),
        "total_first": t0,
        "total_last": totals[-1],
        "rel_drift_last": rel[-1],
        "rel_drift_max_abs": max(abs(r) for r in rel),
    }


def main() -> int:
    input_dir = os.environ["K2_INPUT_DIR"]
    hours = int(os.environ.get("K2_HOURS", "3"))
    dt_s = float(os.environ.get("K2_DT_S", "10"))
    n_sound = int(os.environ.get("K2_ACOUSTIC_SUBSTEPS", "10"))
    tag = os.environ.get("K2_TAG", "rung")
    domain = os.environ.get("K2_DOMAIN", "d01")

    out_root = HERE / "runs" / tag
    out_root.mkdir(parents=True, exist_ok=True)
    proof_dir = out_root / "proof"
    output_dir = out_root / "out"
    scratch_dir = out_root / "scratch"
    for d in (proof_dir, output_dir, scratch_dir):
        d.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GPUWRF_SCRATCH", str(scratch_dir))
    os.environ["GPUWRF_TMPDIR"] = str(scratch_dir)

    in_path = Path(input_dir)
    from gpuwrf.integration.daily_pipeline import (
        DailyPipelineConfig,
        execute_daily_pipeline,
    )

    # Hold the RADIATION cadence in SECONDS constant across rungs (the R1 baseline
    # is dt=10s x 180 steps = 1800s radt). Otherwise a coarser dt would silently
    # also coarsen radiation, confounding the dt/n_sound CFL comparison. Round to
    # >=1 step. This is the only derived knob; dt_s and n_sound are the K2 levers.
    radt_target_s = float(os.environ.get("K2_RADT_TARGET_S", "1800"))
    radiation_cadence_steps = max(1, int(round(radt_target_s / dt_s)))

    config = DailyPipelineConfig(
        run_id=str(in_path.resolve()),
        run_root=in_path.parent,
        hours=hours,
        output_dir=output_dir,
        proof_dir=proof_dir,
        domain=domain,
        dt_s=dt_s,
        acoustic_substeps=n_sound,
        radiation_cadence_steps=radiation_cadence_steps,
        score=False,
    )
    result_radt = {
        "radt_target_s": radt_target_s,
        "radiation_cadence_steps": radiation_cadence_steps,
    }

    result = {
        "tag": tag,
        "dt_s": dt_s,
        "n_sound": n_sound,
        "hours": hours,
        "domain": domain,
        "input_dir": str(in_path),
        "radiation": result_radt,
    }
    t_wall0 = time.perf_counter()
    try:
        payload = execute_daily_pipeline(config)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "RUN_FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print("DRIVER RAISED:", result["error"], file=sys.stderr)
        (out_root / "result.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        return 1
    result["wall_s_total"] = time.perf_counter() - t_wall0
    result["verdict"] = payload.get("verdict")
    wrfout_files = payload.get("wrfout_files") or payload.get("output_files") or []
    # Some payloads nest the file list; resolve from output_dir as a fallback.
    if not wrfout_files:
        wrfout_files = [str(p) for p in sorted(output_dir.rglob(f"wrfout_{domain}_*"))]
    result["wrfout_files"] = [str(p) for p in wrfout_files]

    steps_per_hour = 3600.0 / dt_s
    per_hour_wall = (
        payload.get("wall_clock_per_hour_s")
        or payload.get("per_hour_wall_s")
        or []
    )
    result["per_hour_wall_s"] = per_hour_wall
    if per_hour_wall:
        # steady per-step = median per-hour wall / steps-per-hour (skip hour 0 = compile)
        steady_hours = per_hour_wall[1:] if len(per_hour_wall) > 1 else per_hour_wall
        med_hour = float(np.median(steady_hours))
        result["steady_per_hour_wall_s"] = med_hour
        result["s_per_step"] = med_hour / steps_per_hour
        result["steps_per_hour"] = steps_per_hour

    try:
        result["courant"] = courant_for_run(wrfout_files, dt_s)
    except Exception as exc:  # noqa: BLE001
        result["courant_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["finite_bounded"] = _finite_bounded(wrfout_files)
    except Exception as exc:  # noqa: BLE001
        result["finite_bounded_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result["mass_conservation"] = _mass_conservation(wrfout_files)
    except Exception as exc:  # noqa: BLE001
        result["mass_conservation_error"] = f"{type(exc).__name__}: {exc}"

    result["status"] = "OK"
    (out_root / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
