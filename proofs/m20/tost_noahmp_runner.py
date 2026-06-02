#!/usr/bin/env python3
"""S6b EQUIVALENCE GATE — n=3 MAM paired-TOST with prognostic Noah-MP ACTIVATED.

Re-runs the ADR-029 paired GPU-vs-CPU-WRF TOST on T2/U10/V10, but with the LAND
surface advanced by prognostic Noah-MP (``use_noahmp=True``) instead of the bulk
prescribed-land path. This makes the T2 equivalence NON-tautological: the GPU
computes its OWN land sensible/latent fluxes (canopy energy balance), so passing
the ADR-029 T2 margin (+/-0.2149 K) means the standalone Noah-MP land model
reproduces CPU-WRF's near-surface temperature to within the predeclared margin.

REUSE-ONLY (launches NO new WRF runs). CPU-WRF truth = corpus wrfout. GPU init =
each run's t=0 wrfout snapshot (the replay case) + a corpus-wrfinput Noah-MP land
warm-start (faithful NOAHMP_INIT). Reuses proofs/m20/paired_tost_scorer.py and the
_emit_gpu_wrfout / score_case_level / aggregate_tost helpers from the frozen
ensemble runner. Writes a NEW file tost_run/tost_noahmp.json -- does NOT touch the
frozen tost_aggregate.json.

HONEST CAVEAT: n=3 single-season (MAM) -- UNDERPOWERED vs the ADR-029 n>=15 target;
the verdict is a single-season point estimate, NOT a seasonal equivalence claim.

Usage (manager, sequenced on GPU, ONE consumer):
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
    python proofs/m20/tost_noahmp_runner.py --execute \
      --out-dir proofs/m20/tost_run --levels L2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # THIS worktree's repo root (NOT the hardcoded shared path)
WT_SRC = str(ROOT / "src")
sys.path.insert(0, WT_SRC)
sys.path.insert(0, str(HERE))  # local scorer + ensemble-runner helpers

from tost_ensemble_runner import (  # noqa: E402
    DEFAULT_MANIFEST, OBS_START_UTC, _emit_gpu_wrfout, expand_units, load_cases,
    score_case_level,
)

# tost_ensemble_runner hardcodes ROOT=/home/enric/src/wrf_gpu2 and inserts the
# SHARED src at sys.path[0] on import -- which would shadow THIS worktree's gpuwrf.
# Re-prepend the worktree src so the S6b-activated gpuwrf wins (gpuwrf is not yet
# imported at this point; the ensemble runner only imports it inside functions).
while WT_SRC in sys.path:
    sys.path.remove(WT_SRC)
sys.path.insert(0, WT_SRC)


def run_gpu_case_level_noahmp(level_spec: dict, out_dir: Path, *, dt_s: float,
                              acoustic_substeps: int, radiation_cadence_steps: int,
                              segment_steps: int) -> dict:
    """Advance ONE GPU forecast carry with prognostic Noah-MP over land, emitting a
    GPU wrfout per obs-covered lead. Mirrors tost_ensemble_runner.run_gpu_case_level
    but ACTIVATES Noah-MP (warm-started from the corpus wrfinput)."""
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk, _enforce_operational_precision, compute_m9_diagnostics,
        noahmp_initial_rad,
    )
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file

    cfg = DailyPipelineConfig(
        run_id=level_spec["run_id"], run_root=Path(level_spec["run_root"]),
        domain=level_spec["domain"], dt_s=dt_s, acoustic_substeps=acoustic_substeps,
        radiation_cadence_steps=radiation_cadence_steps,
    )
    case, run_dir = _build_real_case(cfg)
    time_utc = case.run_start
    domain = level_spec["domain"]

    land, static, init_meta = build_noahmp_land_state(Path(level_spec["run_root"])
                                                      / level_spec["run_id"], domain)
    energy_params, rad_params, nroot = build_noahmp_params(static)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=radiation_cadence_steps, time_utc=time_utc,
        use_noahmp=True, noahmp_static=static, noahmp_energy_params=energy_params,
        noahmp_rad_params=rad_params, noahmp_nroot=nroot,
        noahmp_julian=float(time_utc.timetuple().tm_yday), noahmp_yearlen=365.0,
    )

    init_path = Path(level_spec["gpu_init_source"])
    ll = read_wrfout_file(init_path, fields=("XLAT", "XLONG"))["fields"]
    lat = np.asarray(ll["XLAT"], dtype=np.float64)
    lon = np.asarray(ll["XLONG"], dtype=np.float64)

    max_lead = int(level_spec["max_lead_h"])
    leads = []
    for h in range(1, max_lead + 1):
        valid = time_utc + timedelta(hours=h)
        if valid < OBS_START_UTC:
            continue
        cpu_frame = run_dir / f"wrfout_{domain}_{valid:%Y-%m-%d_%H:%M:%S}"
        if cpu_frame.is_file():
            leads.append(h)
    if not leads:
        return {"emitted": 0, "leads": [], "note": "no obs-covered lead with CPU truth",
                "out_dir": str(out_dir)}

    dt = float(nl.dt_s)
    cadence = int(nl.radiation_cadence_steps)
    seg = int(segment_steps) if segment_steps else cadence
    lead_steps = {h: int(round(h * 3600.0 / dt)) for h in leads}

    st0 = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    carry = initial_operational_carry(
        st0, noahmp_land=land, noahmp_rad=noahmp_initial_rad(st0, nl),
    )
    timings = {}
    t0 = time.time()
    start = 1
    for h in leads:
        target = lead_steps[h]
        while start <= target:
            n = min(seg, target - start + 1)
            carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                                   n_steps=int(n), cadence=cadence)
            jax.block_until_ready(carry.state.theta)
            start += n
        diags = compute_m9_diagnostics(carry.state, nl, float(h) * 3600.0,
                                       noahmp_land=carry.noahmp_land,
                                       noahmp_rad=carry.noahmp_rad)
        valid = time_utc + timedelta(hours=h)
        _emit_gpu_wrfout(
            out_dir, domain, valid, lat, lon,
            np.asarray(jax.device_get(diags.t2), dtype=np.float64),
            np.asarray(jax.device_get(diags.u10), dtype=np.float64),
            np.asarray(jax.device_get(diags.v10), dtype=np.float64),
        )
        timings[h] = round(time.time() - t0, 1)
    return {"emitted": len(leads), "leads": leads, "cpu_run_dir": str(run_dir),
            "out_dir": str(out_dir), "init_utc": time_utc.isoformat(),
            "max_lead_h": max_lead, "timings_cumulative_s": timings,
            "noahmp_init": {k: init_meta[k] for k in
                            ("n_land_cells", "source", "init_note")}}


def main(argv=None) -> int:
    from paired_tost_scorer import aggregate_tost  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=HERE / "tost_run")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--levels", nargs="+", default=["L2"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--segment-steps", type=int, default=180)
    a = ap.parse_args(argv)

    cases = load_cases(a.manifest)
    if a.cases:
        cases = [c for c in cases if c["case_id"] in a.cases]
    units = expand_units(cases)
    if a.levels:
        units = [u for u in units if u["level"] in a.levels]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    if not a.execute:
        plan = {"mode": "DRY-PLAN (no GPU)", "manifest": str(a.manifest),
                "n_units": len(units),
                "units": [{"unit_id": u["unit_id"], "init": u["init_utc"], "fh": u["fh"],
                           "cpu_run_dir": u["spec"]["run_dir"],
                           "init_exists": Path(u["spec"]["gpu_init_source"]).is_file()}
                          for u in units]}
        print(json.dumps(plan, indent=2))
        return 0

    score_kwargs = dict(dt_s=a.dt_s, acoustic_substeps=a.acoustic_substeps,
                        radiation_cadence_steps=a.radiation_cadence_steps,
                        segment_steps=a.segment_steps)
    t0 = time.time()
    case_scores = []
    per_unit_meta = []
    for u in units:
        print(f"=== NOAHMP GPU forecast {u['unit_id']} ({u['init_utc']}, fh={u['fh']}) ===",
              flush=True)
        gpu_out = a.out_dir / "gpu_wrfout_noahmp" / u["unit_id"]
        meta = run_gpu_case_level_noahmp(u["spec"], gpu_out, **score_kwargs)
        per_unit_meta.append({"unit_id": u["unit_id"],
                              **{k: v for k, v in meta.items() if k != "out_dir"}})
        if meta["emitted"] == 0:
            print(f"    {u['unit_id']}: no scoreable lead — skipped", flush=True)
            continue
        init = datetime.fromisoformat(u["init_utc"])
        cs = score_case_level(u["unit_id"], Path(meta["cpu_run_dir"]), gpu_out,
                              u["spec"]["domain"], init, u["fh"])
        (a.out_dir / f"paired_score_noahmp_{u['unit_id']}.json").write_text(
            json.dumps(cs, indent=2, default=str) + "\n")
        case_scores.append(cs)
        agg = aggregate_tost(case_scores)
        # write to the NEW noahmp file (NEVER touch the frozen tost_aggregate.json).
        (a.out_dir / "tost_noahmp.json").write_text(
            json.dumps({"schema": "S6bNoahMPTost", "generated_utc":
                        datetime.now(timezone.utc).isoformat(),
                        "manifest": str(a.manifest), "land_model": "prognostic Noah-MP",
                        "per_unit_meta": per_unit_meta, "tost_aggregate": agg,
                        "honest_caveat": ("n=3 single-season MAM — UNDERPOWERED vs "
                                          "ADR-029 n>=15; single-season point estimate, "
                                          "NOT a seasonal equivalence claim"),
                        "wall_s": round(time.time() - t0, 1)},
                       indent=2, default=str) + "\n")
        print(f"    {u['unit_id']}: pairs={cs['total_complete_pairs']} "
              f"n_so_far={agg['n_cases']}", flush=True)

    print(f"DONE -> {a.out_dir / 'tost_noahmp.json'}  wall={round(time.time()-t0,1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
