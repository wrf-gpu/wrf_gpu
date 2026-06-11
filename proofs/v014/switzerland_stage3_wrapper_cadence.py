"""v0.14 stage-3 / wrapper-cadence sprint proof assembly.

Assembles the three evidence layers into
``proofs/v014/switzerland_stage3_wrapper_cadence.json``:

1. ``diagnosis`` -- extracted from the PRE-FIX stage compare (tag
   ``sub4_dt18_rhsph2`` in switzerland_acoustic_substep_blocker.json): the
   interior stage-3 jump is pre-wrapper (stage3_raw == final on mu/p/ph), the
   spec-zone ring-0 drifts ~200 Pa p / 126-276 m2/s2 ph within single steps
   (WRF own band increment scale 0.53), the once-per-step end-of-step nudge
   resets ring 0 and shocks ring 1 (74 Pa p at step end).

2. ``stage_compare_fix`` -- PRE vs POST (tag ``sub4_dt18_speccad``, the WRF
   specified-cadence flag ON) band/ring evidence against the SAME WRF-native
   stage dumps (calls 21601-21606).

3. ``hourly_gate`` -- the h36->h37 (and h36->h38) depth-8 dry-mass budget for
   the new cadence run vs old/hypso/rhsph baselines and the CPU truth.

Run AFTER the stage compare + forecast variant have populated their outputs:
    python proofs/v014/switzerland_stage3_wrapper_cadence.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "switzerland_stage3_wrapper_cadence.json"
BLOCKER_JSON = HERE / "switzerland_acoustic_substep_blocker.json"

_spec = importlib.util.spec_from_file_location("hpg", HERE / "switzerland_hpg_native_face_fix.py")
hpg = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("hpg", hpg)
_spec.loader.exec_module(hpg)

PRE_TAG = "sub4_dt18_rhsph2"
POST_TAG = "sub4_dt18_speccad"
ADVDEG_TAG = "sub4_dt18_advdeg"  # cadence + WRF specified advection degradation
STAGE_LABELS = (
    "step1_stage1_vs_21602",
    "step1_stage2_vs_21603",
    "step1_stage3_raw_vs_21604_prewrapper",
    "step1_final_vs_21604",
    "step2_stage1_vs_21605",
    "step2_stage2_vs_21606",
)
FIELDS = ("mu", "p", "ph")


def _stage_rows(comparisons: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for label in STAGE_LABELS:
        entry = comparisons.get(label)
        if entry is None:
            continue
        row: dict[str, Any] = {}
        for field in FIELDS:
            err = entry[field]["incr_err"]
            row[field] = {
                "interior_rmse": err["interior"]["rmse"],
                "band_rmse": err["band"]["rmse"],
                "band_max": err["band"]["max_abs"],
                "ring_rmse": entry[field]["incr_err_ring_rmse"],
            }
        rows[label] = row
    return rows


def assemble() -> dict[str, Any]:
    blocker = json.loads(BLOCKER_JSON.read_text())
    pre = blocker["stage_compare"][PRE_TAG]
    post = blocker["stage_compare"][POST_TAG]
    pre_rows = _stage_rows(pre["comparisons"])
    post_rows = _stage_rows(post["comparisons"])

    # 1. diagnosis from the PRE captures.
    s3 = pre_rows["step1_stage3_raw_vs_21604_prewrapper"]
    fin = pre_rows["step1_final_vs_21604"]
    diagnosis = {
        "interior_jump_is_inside_stage3": {
            "stage3_raw_interior_p": s3["p"]["interior_rmse"],
            "final_interior_p": fin["p"]["interior_rmse"],
            "identical": abs(s3["p"]["interior_rmse"] - fin["p"]["interior_rmse"]) < 1e-9,
            "note": "the wrapper writes do not touch the interior mu/p/ph -- the "
            "p 1.84->4.42 growth happens inside RK stage 3",
        },
        "spec_zone_free_drift": {
            "stage1_ring0_p": pre_rows["step1_stage1_vs_21602"]["p"]["ring_rmse"]["0"],
            "stage1_ring0_ph": pre_rows["step1_stage1_vs_21602"]["ph"]["ring_rmse"]["0"],
            "step2_stage2_ring0_p": pre_rows["step2_stage2_vs_21606"]["p"]["ring_rmse"]["0"],
            "wrf_own_band_increment_p_rmse": 0.5336,
            "note": "JAX advanced the outermost specified ring with full dynamics; "
            "WRF excludes it from the small step and walks it along the wrfbdy "
            "trajectory every acoustic substep (spec_bdyupdate/_ph, zero_grad w)",
        },
        "end_of_step_nudge_shock": {
            "final_ring0_p": fin["p"]["ring_rmse"]["0"],
            "final_ring1_p": fin["p"]["ring_rmse"]["1"],
            "note": "the once-per-step nudge resets ring 0 (~0.4) but leaves ring 1 "
            "at 74 Pa and discontinuously rewrites rings 0-3 after the last "
            "calc_p_rho -- WRF has NO end-of-step dry write",
        },
        "step2_interior_explosion_not_band_shock": {
            "pre_step2_stage1_interior_p": pre_rows["step2_stage1_vs_21605"]["p"]["interior_rmse"],
            "post_step2_stage1_interior_p": post_rows["step2_stage1_vs_21605"]["p"]["interior_rmse"],
            "note": "unchanged under the cadence fix -> the step-2 stage-1 interior "
            "p increment error (12.7, flat profile) is the INTERIOR dynamics lane, "
            "not the boundary band",
        },
    }

    # 2. PRE vs POST stage compare.
    fix_table: dict[str, Any] = {"pre_tag": PRE_TAG, "post_tag": POST_TAG, "stages": {}}
    for label in STAGE_LABELS:
        if label not in pre_rows or label not in post_rows:
            continue
        entry: dict[str, Any] = {}
        for field in FIELDS:
            pre_f = pre_rows[label][field]
            post_f = post_rows[label][field]
            entry[field] = {
                "band_rmse_pre": pre_f["band_rmse"],
                "band_rmse_post": post_f["band_rmse"],
                "band_collapse_x": (pre_f["band_rmse"] / post_f["band_rmse"]) if post_f["band_rmse"] > 0 else None,
                "ring0_pre": pre_f["ring_rmse"]["0"],
                "ring0_post": post_f["ring_rmse"]["0"],
                "ring1_pre": pre_f["ring_rmse"]["1"],
                "ring1_post": post_f["ring_rmse"]["1"],
                "interior_rmse_pre": pre_f["interior_rmse"],
                "interior_rmse_post": post_f["interior_rmse"],
            }
        fix_table["stages"][label] = entry
    fix_table["replica_vs_jit_max_diffs"] = post.get("replica_vs_jit_max_diffs")

    # 2b. candidate A on top: WRF specified advection-order degradation.
    advdeg_entry = blocker["stage_compare"].get(ADVDEG_TAG)
    if advdeg_entry is not None:
        advdeg_rows = _stage_rows(advdeg_entry["comparisons"])
        adv_table: dict[str, Any] = {"tag": ADVDEG_TAG, "stages": {}}
        for label in STAGE_LABELS:
            if label not in advdeg_rows or label not in post_rows:
                continue
            entry = {}
            for field in FIELDS:
                a = advdeg_rows[label][field]
                b = post_rows[label][field]
                entry[field] = {
                    "band_rmse_cadence": b["band_rmse"],
                    "band_rmse_advdeg": a["band_rmse"],
                    "ring1_cadence": b["ring_rmse"]["1"],
                    "ring1_advdeg": a["ring_rmse"]["1"],
                    "ring2_cadence": b["ring_rmse"]["2"],
                    "ring2_advdeg": a["ring_rmse"]["2"],
                    "interior_rmse_cadence": b["interior_rmse"],
                    "interior_rmse_advdeg": a["interior_rmse"],
                }
            adv_table["stages"][label] = entry
        adv_table["replica_vs_jit_max_diffs"] = advdeg_entry.get("replica_vs_jit_max_diffs")
        fix_table["advdeg"] = adv_table

    # 3. hourly gate.
    CPU = hpg.CPU
    SPECCAD = hpg.PROBE_ROOT / "gpu_output_speccad"
    ADVDEG = hpg.PROBE_ROOT / "gpu_output_advdeg"
    runs = {
        "old_ec4d6769": hpg.BASELINE_GPU,
        "hypso_3d0b439c": hpg.FIX_GPU,
        "rhsph_79b0c22e": hpg.PROBE_ROOT / "gpu_output_acoustic_rhsph2",
        "speccad_fix": SPECCAD,
        "advdeg_fix": ADVDEG,
    }
    cpu = hpg.budget_between(CPU, 36, CPU, 37, depth=8)
    gate: dict[str, Any] = {"cpu_budget": cpu}
    excess: dict[str, float] = {}
    residual: dict[str, float] = {}
    for name, base in runs.items():
        if not hpg.fn(base, 37).exists():
            gate[name] = {"available": False, "path": str(base)}
            continue
        budget = hpg.budget_between(CPU, 36, base, 37, depth=8)
        gate[name] = budget
        excess[name] = float(budget["net_influx_pa_per_cell_h"] - cpu["net_influx_pa_per_cell_h"])
        residual[name] = float(budget["residual_pa_per_cell_h"])
    gate["excess_outflux_pa_per_cell_h"] = excess
    gate["residual_pa_per_cell_h"] = residual
    if "speccad_fix" in excess:
        gate["excess_collapse_vs_old"] = float(1.0 - abs(excess["speccad_fix"]) / max(abs(excess["old_ec4d6769"]), 1e-12))
        gate["excess_collapse_vs_rhsph"] = float(1.0 - abs(excess["speccad_fix"]) / max(abs(excess["rhsph_79b0c22e"]), 1e-12))
        gate["metrics_h37"] = hpg.field_metrics(SPECCAD, 37)
        if hpg.fn(SPECCAD, 38).exists():
            cpu38 = hpg.budget_between(CPU, 36, CPU, 38, depth=8)
            fix38 = hpg.budget_between(CPU, 36, SPECCAD, 38, depth=8)
            rhsph38 = None
            if hpg.fn(hpg.PROBE_ROOT / "gpu_output_acoustic_rhsph2", 38).exists():
                rb = hpg.budget_between(CPU, 36, hpg.PROBE_ROOT / "gpu_output_acoustic_rhsph2", 38, depth=8)
                rhsph38 = {
                    "excess": float(rb["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"]),
                    "residual": float(rb["residual_pa_per_cell_h"]),
                }
            gate["h36_h38"] = {
                "cpu": cpu38,
                "speccad_fix": fix38,
                "excess": float(fix38["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"]),
                "residual": float(fix38["residual_pa_per_cell_h"]),
                "rhsph_79b0c22e": rhsph38,
            }

    # 4. interior-sink localization (the no-fix handoff evidence): the venting
    # survives BOTH WRF-proven band mechanisms; localize it.
    interior_sink: dict[str, Any] = {}
    try:
        import numpy as _np
        from netCDF4 import Dataset as _DS

        def _mu_hgt(base, hour):
            with _DS(hpg.fn(base, hour)) as dd:
                mu = _np.asarray(dd.variables["MU"][0]) + _np.asarray(dd.variables["MUB"][0])
                hgt = _np.asarray(dd.variables["HGT"][0])
            return mu, hgt

        cpu37_mu, hgt = _mu_hgt(CPU, 37)
        cpu36_mu, _ = _mu_hgt(CPU, 36)
        ny, nx = cpu37_mu.shape
        jj, ii = _np.mgrid[0:ny, 0:nx]
        ring = _np.minimum(_np.minimum(ii, nx - 1 - ii), _np.minimum(jj, ny - 1 - jj))
        inter = ring >= 8
        interior_sink["cpu_own_dmu_interior_mean"] = float((cpu37_mu - cpu36_mu)[inter].mean())
        for name, base in runs.items():
            if not hpg.fn(base, 37).exists():
                continue
            g37, _ = _mu_hgt(base, 37)
            dmu = g37 - cpu37_mu
            hi = hgt > 1500.0
            interior_sink[name] = {
                "interior_mean": float(dmu[inter].mean()),
                "interior_rmse": float(_np.sqrt((dmu[inter] ** 2).mean())),
                "interior_high_terrain_mean": float(dmu[inter & hi].mean()),
                "interior_low_terrain_mean": float(dmu[inter & ~hi].mean()),
                "ring_mean_profile": {str(r): float(dmu[ring == r].mean()) for r in range(0, 12, 2)},
            }
    except Exception as exc:  # pragma: no cover - diagnostic best-effort
        interior_sink["error"] = str(exc)

    # 5. the per-step hydrostatic phi-sink/p-rise pair (interior MEAN increment
    # biases vs the WRF dumps; IDENTICAL across band variants -> band-independent).
    phi_p_pair: dict[str, Any] = {}
    for tag in (PRE_TAG, POST_TAG, ADVDEG_TAG):
        entry = blocker["stage_compare"].get(tag)
        if entry is None:
            continue
        rows = {}
        for label in STAGE_LABELS:
            c = entry["comparisons"].get(label)
            if c is None:
                continue
            rows[label] = {
                "p_err_mean": c["p"]["incr_err"]["interior"]["mean"],
                "ph_err_mean": c["ph"]["incr_err"]["interior"]["mean"],
                "mu_err_mean": c["mu"]["incr_err"]["interior"]["mean"],
                "al_err_mean": c["al"]["incr_err"]["interior"]["mean"],
                "p_wrf_own_mean": c["p"]["wrf_incr"]["interior"]["mean"],
                "ph_wrf_own_mean": c["ph"]["wrf_incr"]["interior"]["mean"],
            }
        phi_p_pair[tag] = rows

    result = {
        "sprint": "2026-06-11-v014-fable-stage3-wrapper-cadence",
        "diagnosis": diagnosis,
        "stage_compare_fix": fix_table,
        "hourly_gate": gate,
        "interior_sink": interior_sink,
        "phi_p_hydrostatic_pair": phi_p_pair,
    }
    hpg.write_json(OUT_JSON, result)
    print(json.dumps({
        "excess": excess,
        "residual": residual,
        "band_p_stage1": [
            fix_table["stages"]["step1_stage1_vs_21602"]["p"]["band_rmse_pre"],
            fix_table["stages"]["step1_stage1_vs_21602"]["p"]["band_rmse_post"],
        ],
    }, indent=2))
    return result


if __name__ == "__main__":
    assemble()
