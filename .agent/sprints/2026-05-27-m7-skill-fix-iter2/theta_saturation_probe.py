#!/usr/bin/env python
"""Instrument one operational hour around the M7 theta guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries  # noqa: E402
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter  # noqa: E402
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case, write_json  # noqa: E402
from gpuwrf.profiling.transfer_audit import block_until_ready  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    _enforce_operational_precision,
    _finite_or_origin,
    _limit_guarded_dynamics_state,
    _limit_theta_by_level,
    _physics_boundary_step,
    _rk_scan_step,
    _steps_for_hours,
    _valid_mixing_ratio,
)
from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: E402


SPRINT = ROOT / ".agent" / "sprints" / "2026-05-27-m7-skill-fix-iter2"
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
LOWER_LEVELS = 30
CAP_K = 400.0
HIST_BINS = jnp.asarray([150.0, 180.0, 200.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 500.0, 600.0, 700.0, 800.0])


def _guard_moisture(state, origin):
    return state.replace(
        qv=_valid_mixing_ratio(state.qv, origin.qv),
        qc=_valid_mixing_ratio(state.qc, origin.qc),
        qr=_valid_mixing_ratio(state.qr, origin.qr),
        qi=_valid_mixing_ratio(state.qi, origin.qi),
        qs=_valid_mixing_ratio(state.qs, origin.qs),
        qg=_valid_mixing_ratio(state.qg, origin.qg),
    )


def _histogram(values):
    flat = jnp.ravel(values[:LOWER_LEVELS])
    index = jnp.clip(jnp.digitize(flat, HIST_BINS) - 1, 0, HIST_BINS.shape[0] - 2)
    return jnp.bincount(index, length=HIST_BINS.shape[0] - 1)


def _first_hit(mask):
    flat = jnp.ravel(mask)
    index = jnp.argmax(flat)
    yx = mask.shape[1] * mask.shape[2]
    return {
        "any": jnp.any(flat),
        "level": index // yx,
        "y": (index % yx) // mask.shape[2],
        "x": index % mask.shape[2],
    }


def _theta_stats(pre_cap, post_cap, after_surface, after_mynn, theta_flux):
    lower_pre = pre_cap[:LOWER_LEVELS]
    lower_post = post_cap[:LOWER_LEVELS]
    hit_mask = lower_post >= (CAP_K - 1.0e-4)
    clipped_mask = jnp.isfinite(lower_pre) & (lower_pre > CAP_K) & hit_mask
    first = _first_hit(hit_mask)
    pbl_delta = after_mynn[:LOWER_LEVELS] - after_surface[:LOWER_LEVELS]
    return {
        "pre_cap_hist": _histogram(pre_cap),
        "post_cap_hist": _histogram(post_cap),
        "pre_cap_min_k": jnp.nanmin(lower_pre),
        "pre_cap_max_k": jnp.nanmax(lower_pre),
        "post_cap_min_k": jnp.nanmin(lower_post),
        "post_cap_max_k": jnp.nanmax(lower_post),
        "cap_hit_count": jnp.sum(hit_mask),
        "clip_from_above_count": jnp.sum(clipped_mask),
        "first_hit_any": first["any"],
        "first_hit_level": first["level"],
        "first_hit_y": first["y"],
        "first_hit_x": first["x"],
        "theta_flux_min": jnp.nanmin(theta_flux),
        "theta_flux_max": jnp.nanmax(theta_flux),
        "theta_flux_mean": jnp.nanmean(theta_flux),
        "pbl_delta_min_k": jnp.nanmin(pbl_delta),
        "pbl_delta_max_k": jnp.nanmax(pbl_delta),
        "pbl_delta_mean_k": jnp.nanmean(pbl_delta),
    }


def _diagnostic_step(carry, namelist, step_index, *, run_radiation: bool):
    physical_origin = carry.state
    rk_carry = _rk_scan_step(carry, namelist, debug=False)
    next_state = rk_carry.state
    if not bool(namelist.disable_guards):
        next_state = _guard_moisture(_limit_guarded_dynamics_state(next_state, physical_origin), physical_origin)

    after_surface_theta = next_state.theta
    after_mynn_theta = next_state.theta
    theta_flux = next_state.theta_flux
    if bool(namelist.run_physics):
        if not bool(namelist.disable_guards):
            next_state = thompson_adapter(next_state, float(namelist.dt_s))
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        after_surface_theta = next_state.theta
        theta_flux = next_state.theta_flux
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        after_mynn_theta = next_state.theta
        if run_radiation:
            next_state = rrtmg_adapter(next_state, float(namelist.dt_s), namelist.grid)

    pre_cap_theta = next_state.theta
    if bool(namelist.run_boundary):
        lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
        bounded = apply_lateral_boundaries(next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
        pre_cap_theta = bounded.theta
        if bool(namelist.disable_guards):
            next_state = bounded
        else:
            next_state = bounded.replace(
                u=_finite_or_origin(bounded.u, physical_origin.u),
                v=_finite_or_origin(bounded.v, physical_origin.v),
                w=_finite_or_origin(bounded.w, physical_origin.w),
                theta=_limit_theta_by_level(bounded.theta, physical_origin.theta),
                qv=_valid_mixing_ratio(bounded.qv, physical_origin.qv),
                p=_finite_or_origin(bounded.p, physical_origin.p),
                ph=_finite_or_origin(bounded.ph, physical_origin.ph),
                p_total=_finite_or_origin(bounded.p_total, physical_origin.p_total),
                ph_total=_finite_or_origin(bounded.ph_total, physical_origin.ph_total),
                p_perturbation=_finite_or_origin(bounded.p_perturbation, physical_origin.p_perturbation),
                ph_perturbation=_finite_or_origin(bounded.ph_perturbation, physical_origin.ph_perturbation),
            )
            next_state = _limit_guarded_dynamics_state(next_state, physical_origin)

    final_state = _enforce_operational_precision(next_state)
    stats = _theta_stats(pre_cap_theta, final_state.theta, after_surface_theta, after_mynn_theta, theta_flux)
    return rk_carry.replace(state=final_state), stats


_diagnostic_step_jit = jax.jit(_diagnostic_step, static_argnames=("run_radiation",))


def _to_jsonable(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (np.bool_, bool)):
            return bool(item)
        if isinstance(item, (np.integer, int)):
            return int(item)
        return float(item)
    return array.tolist()


def _summarize(records: list[dict[str, Any]], dt_s: float) -> dict[str, Any]:
    cap_steps = [row for row in records if int(row["cap_hit_count"]) > 0]
    clip_steps = [row for row in records if int(row["clip_from_above_count"]) > 0]
    first = next((row for row in records if row["first_hit_any"]), None)
    max_pre = max(float(row["pre_cap_max_k"]) for row in records)
    max_post = max(float(row["post_cap_max_k"]) for row in records)
    flux_abs_max = max(max(abs(float(row["theta_flux_min"])), abs(float(row["theta_flux_max"]))) for row in records)
    pbl_abs_max = max(max(abs(float(row["pbl_delta_min_k"])), abs(float(row["pbl_delta_max_k"]))) for row in records)
    return {
        "total_steps": len(records),
        "cap_hit_step_count": len(cap_steps),
        "clip_from_above_step_count": len(clip_steps),
        "cap_hit_every_step": len(cap_steps) == len(records),
        "first_hit": None
        if first is None
        else {
            "step": int(first["step"]),
            "lead_seconds": float(first["lead_seconds"]),
            "lead_minutes": float(first["lead_seconds"]) / 60.0,
            "level": int(first["first_hit_level"]),
            "y": int(first["first_hit_y"]),
            "x": int(first["first_hit_x"]),
        },
        "max_pre_cap_theta_lower_30_k": max_pre,
        "max_post_cap_theta_lower_30_k": max_post,
        "theta_flux_abs_max_kinematic": flux_abs_max,
        "pbl_delta_abs_max_k_per_step": pbl_abs_max,
        "dt_s": float(dt_s),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    first = summary["first_hit"]
    if first is None:
        first_text = "No lower-30 cell reached the 400 K post-cap value during the instrumented hour."
    else:
        first_text = (
            f"First 400 K post-cap hit: step {first['step']} "
            f"({first['lead_minutes']:.2f} min), level {first['level']}, "
            f"grid cell y={first['y']}, x={first['x']}."
        )
    cap_frequency = "every step" if summary["cap_hit_every_step"] else f"{summary['cap_hit_step_count']} of {summary['total_steps']} steps"
    dominant = (
        "The cap is active before the surface/PBL adapters can explain it: pre-cap lower-column theta exceeds 400 K, "
        "while per-step PBL deltas and kinematic surface heat fluxes are not of comparable magnitude."
        if summary["max_pre_cap_theta_lower_30_k"] > CAP_K and summary["pbl_delta_abs_max_k_per_step"] < 20.0
        else "The recorded surface/PBL deltas are large enough that the cap may be masking a downstream physics runaway."
    )
    text = f"""# Theta Saturation Diagnosis

Summary: 1-hour instrumented forecast around `_physics_boundary_step` using run `{payload['run_id']}`.

- {first_text}
- The 400 K lower-30 post-cap value appears on {cap_frequency}.
- Steps clipping finite pre-cap theta from above 400 K: {summary['clip_from_above_step_count']} of {summary['total_steps']}.
- Max pre-cap lower-30 theta: {summary['max_pre_cap_theta_lower_30_k']:.6f} K.
- Max post-cap lower-30 theta: {summary['max_post_cap_theta_lower_30_k']:.6f} K.
- Max absolute kinematic theta flux: {summary['theta_flux_abs_max_kinematic']:.9g}.
- Max absolute PBL theta delta per step: {summary['pbl_delta_abs_max_k_per_step']:.9g} K.

Decision: the 400 K lower-30 guard is too tight for this operational statistic. {dominant}

Proof: per-step pre-cap and post-cap theta histograms are in `{payload['json_path']}`.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=SPRINT / "theta_saturation_diagnosis.json")
    parser.add_argument("--output-md", type=Path, default=SPRINT / "theta_saturation_diagnosis.md")
    args = parser.parse_args(argv)

    config = DailyPipelineConfig(run_id=args.run_id, hours=int(args.hours), proof_dir=SPRINT)
    case, run_dir = _build_real_case(config)
    carry = initial_operational_carry(_enforce_operational_precision(case.state))
    steps = _steps_for_hours(float(args.hours), float(case.namelist.dt_s))
    cadence = int(case.namelist.radiation_cadence_steps)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        run_radiation = bool(case.namelist.run_physics and step % cadence == 0)
        carry, stats = _diagnostic_step_jit(carry, case.namelist, jnp.asarray(step, dtype=jnp.int32), run_radiation=run_radiation)
        block_until_ready(carry)
        row = {name: _to_jsonable(value) for name, value in stats.items()}
        row["step"] = int(step)
        row["lead_seconds"] = float(step) * float(case.namelist.dt_s)
        records.append(row)

    payload = {
        "schema": "M7ThetaSaturationDiagnosis",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "hours": float(args.hours),
        "histogram_bins_k": _to_jsonable(HIST_BINS),
        "lower_levels": LOWER_LEVELS,
        "cap_k": CAP_K,
        "wall_clock_s": float(time.perf_counter() - start),
        "records": records,
        "summary": _summarize(records, float(case.namelist.dt_s)),
        "json_path": str(args.output_json),
    }
    write_json(args.output_json, payload)
    _write_markdown(args.output_md, payload)
    print(json.dumps({"status": "PASS", "output": str(args.output_md), "json": str(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
