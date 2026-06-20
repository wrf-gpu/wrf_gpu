#!/usr/bin/env python
"""M6b Tier-4 RMSE dry-run on 20260429 — verify comparator pipeline end-to-end.

Stages
------
1. Run operational 1h on a known-passing IC; record bounds + finiteness + wall.
2. Compute Tier-4 RMSE (T2/U10/V10) against the +1h Gen2 wrfout, plus a
   per-cell heterogeneity ratio (max(|err|) / mean(|err|)).
3. Sanity-check operational RMSE against the Gen2 noise floor (rmse_summary.csv).

This is a comparator dry-run; it produces no production claim. The acceptance
gate here is that the *infrastructure* is sound on a known-passing IC, so the
M6 close gate can lean on it later.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import (
    _reference_fields,
    _surface_fields,
    build_replay_case,
    forecast_comparison,
)
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)


DEFAULT_RUN_ID = "20260429_18z_l3_24h_20260524T204451Z"
DEFAULT_OUTPUT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-tier4-rmse-dryrun"
RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
BASELINE_RMSE_CSV = ROOT / "data" / "fixtures" / "gen2_baseline" / "rmse_summary.csv"

TIER4_THRESHOLDS = {"T2": 3.0, "U10": 7.5, "V10": 7.5}
TIER4_UNITS = {"T2": "K", "U10": "m s-1", "V10": "m s-1"}
SPATIAL_RATIO_THRESHOLD = 1.5
LOWER_LEVELS = 30
UPPER_LEVELS = 14
WIND_LIMITS = {"u_abs_max_m_s": 100.0, "v_abs_max_m_s": 100.0, "w_abs_max_m_s": 50.0}
NOISE_FLOOR_LEAD_HOURS = 24
NOISE_FLOOR_BAND_MULTIPLIER = 5.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case_state_and_namelist(run_id: str) -> tuple[Any, OperationalNamelist, Any, dict[str, Any]]:
    run_dir = RUN_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Gen2 run dir not found: {run_dir}")
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    meta = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "grid": case.metadata["grid"],
        "namelist": {
            "dt_s": namelist.dt_s,
            "acoustic_substeps": namelist.acoustic_substeps,
            "run_physics": namelist.run_physics,
            "run_boundary": namelist.run_boundary,
            "use_vertical_solver": namelist.use_vertical_solver,
            "radiation_cadence_steps": namelist.radiation_cadence_steps,
        },
    }
    return state, namelist, case.run, meta


def _all_leaves_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _bounds_for_state(state: Any, *, step: int, lead_seconds: float) -> dict[str, Any]:
    theta = state.theta
    if int(theta.shape[0]) != LOWER_LEVELS + UPPER_LEVELS:
        raise ValueError(f"expected {LOWER_LEVELS + UPPER_LEVELS} theta levels, got {theta.shape[0]}")

    lower_theta = theta[:LOWER_LEVELS, :, :]
    upper_theta = theta[LOWER_LEVELS:, :, :]
    values = {
        "step": int(step),
        "lead_seconds": float(lead_seconds),
        "all_leaves_finite": _all_leaves_finite(state),
        "theta_lower_30_min_k": float(np.asarray(jnp.min(lower_theta))),
        "theta_lower_30_max_k": float(np.asarray(jnp.max(lower_theta))),
        "theta_upper_14_min_k": float(np.asarray(jnp.min(upper_theta))),
        "theta_upper_14_max_k": float(np.asarray(jnp.max(upper_theta))),
        "u_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.u)))),
        "v_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.v)))),
        "w_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.w)))),
    }
    lower_ok = 200.0 <= values["theta_lower_30_min_k"] and values["theta_lower_30_max_k"] <= 400.0
    upper_ok = 250.0 <= values["theta_upper_14_min_k"] and values["theta_upper_14_max_k"] <= 700.0
    wind_ok = (
        values["u_abs_max_m_s"] <= WIND_LIMITS["u_abs_max_m_s"]
        and values["v_abs_max_m_s"] <= WIND_LIMITS["v_abs_max_m_s"]
        and values["w_abs_max_m_s"] <= WIND_LIMITS["w_abs_max_m_s"]
    )
    values["theta_lower_30_bounded"] = bool(lower_ok)
    values["theta_upper_14_bounded"] = bool(upper_ok)
    values["wind_bounded"] = bool(wind_ok)
    values["passed"] = bool(values["all_leaves_finite"] and lower_ok and upper_ok and wind_ok)
    return values


def stage1_run_1h(run_id: str, hours: float) -> tuple[Any, Any, dict[str, Any]]:
    """Run the operational 1h forecast and produce proof_1h_run.json."""

    state, namelist, run, meta = _case_state_and_namelist(run_id)
    wall_start = time.perf_counter()
    result = run_forecast_operational(state, namelist, hours)
    block_until_ready(result)
    wall_s = time.perf_counter() - wall_start
    bounds = _bounds_for_state(
        result,
        step=int(round(hours * 3600.0 / float(namelist.dt_s))),
        lead_seconds=hours * 3600.0,
    )
    payload = {
        "artifact_type": "m6b_tier4_rmse_dryrun_stage1",
        "stage": 1,
        "status": "PASS" if bounds["passed"] else "FAIL",
        "device": visible_gpu_name(),
        "run_id": run_id,
        "hours": float(hours),
        "wall_time_s_full_hour_including_compile": wall_s,
        "operational_entrypoint": "gpuwrf.runtime.operational_mode.run_forecast_operational",
        "final_bounds": bounds,
        "bounds_policy": {
            "lower_30_levels_k": [200.0, 400.0],
            "upper_14_levels_k": [250.0, 700.0],
            "full_column_finite_required": True,
            "wind_abs_limits_m_s": WIND_LIMITS,
        },
        "meta": meta,
    }
    return result, run, payload


def _local_error_stats(predicted: Any, reference: Any) -> dict[str, Any]:
    """Per-cell |error| stats — heterogeneity diagnostic for the dry-run."""

    err = np.asarray(predicted, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    abs_err = np.abs(err)
    finite = np.isfinite(abs_err)
    if not bool(np.all(finite)):
        return {
            "all_finite": False,
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
            "spatial_ratio_max_over_mean": float("nan"),
            "shape": list(err.shape),
        }
    mean_abs = float(np.mean(abs_err))
    max_abs = float(np.max(abs_err))
    ratio = float(max_abs / mean_abs) if mean_abs > 0.0 else float("nan")
    return {
        "all_finite": True,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "spatial_ratio_max_over_mean": ratio,
        "shape": list(err.shape),
    }


def stage2_compute_tier4_rmse(forecast_state: Any, run: Any, hours: float) -> dict[str, Any]:
    """Compute Tier-4 RMSE + heterogeneity vs Gen2 wrfout at t=+hours."""

    surface = _surface_fields(forecast_state)
    reference, reference_path, valid_time = _reference_fields(run, "d02", hours)

    fields: dict[str, Any] = {}
    all_pass = True
    for name, threshold in TIER4_THRESHOLDS.items():
        pred = surface[name]
        ref = reference[name]
        if pred.shape != ref.shape:
            raise ValueError(f"shape mismatch for {name}: forecast {pred.shape} vs reference {ref.shape}")
        err = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        rmse = float(np.sqrt(np.nanmean(err * err)))
        local = _local_error_stats(pred, ref)
        rmse_pass = bool(np.isfinite(rmse) and rmse <= threshold)
        # heterogeneity bound: contract calls for spatial-ratio ≤ 1.5 (typo
        # acknowledged — "local rmse" is interpreted as per-cell |error|).
        ratio = local["spatial_ratio_max_over_mean"]
        ratio_pass = bool(np.isfinite(ratio) and ratio <= SPATIAL_RATIO_THRESHOLD)
        ok = bool(rmse_pass and ratio_pass and local["all_finite"])
        fields[name] = {
            "rmse_spatial_mean": rmse,
            "rmse_threshold": float(threshold),
            "rmse_pass": rmse_pass,
            "local_max_abs_error": local["max_abs"],
            "local_mean_abs_error": local["mean_abs"],
            "spatial_ratio_max_over_mean": ratio,
            "spatial_ratio_threshold": SPATIAL_RATIO_THRESHOLD,
            "spatial_ratio_pass": ratio_pass,
            "all_finite": local["all_finite"],
            "shape": local["shape"],
            "units": TIER4_UNITS[name],
            "passed": ok,
        }
        all_pass = all_pass and ok

    payload = {
        "artifact_type": "m6b_tier4_rmse_dryrun_stage2",
        "stage": 2,
        "status": "PASS" if all_pass else "FAIL",
        "lead_hours": float(hours),
        "gen2_reference_path": str(reference_path),
        "valid_time_utc": valid_time,
        "thresholds": TIER4_THRESHOLDS,
        "spatial_ratio_threshold": SPATIAL_RATIO_THRESHOLD,
        "fields": fields,
        "rationale": "spatial-mean RMSE per field + heterogeneity (max|err|/mean|err|) for 8 published numbers (4 per metric x cells inferred), per sprint contract Stage 2.",
    }
    return payload


def _load_noise_floor() -> dict[str, dict[str, Any]]:
    """Load Gen2 internal noise floor at 24h lead from rmse_summary.csv."""

    if not BASELINE_RMSE_CSV.exists():
        raise FileNotFoundError(f"missing baseline noise-floor csv: {BASELINE_RMSE_CSV}")
    rows: dict[str, dict[str, Any]] = {}
    with BASELINE_RMSE_CSV.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            field = row["field"]
            try:
                lead = int(row["lead_hours"])
            except (KeyError, ValueError):
                continue
            if lead != NOISE_FLOOR_LEAD_HOURS:
                continue
            rows[field] = {
                "lead_hours": lead,
                "spatial_mean_rmse": float(row["spatial_mean_rmse"]),
                "p95_rmse": float(row["p95_rmse"]),
                "sample_pairs": int(row["sample_pairs"]),
                "units": row["units"],
                "notes": row.get("notes", ""),
            }
    missing = [name for name in TIER4_THRESHOLDS if name not in rows]
    if missing:
        raise KeyError(f"noise floor csv missing rows for {missing}")
    return rows


def stage3_noise_floor_compare(stage2: dict[str, Any]) -> dict[str, Any]:
    """Compare operational RMSE to Gen2 internal noise floor (24h overlap)."""

    floor = _load_noise_floor()
    fields: dict[str, Any] = {}
    classifications: list[str] = []
    for name, stats in stage2["fields"].items():
        rmse = float(stats["rmse_spatial_mean"])
        noise = float(floor[name]["spatial_mean_rmse"])
        envelope = float(TIER4_THRESHOLDS[name])
        band_top = noise * NOISE_FLOOR_BAND_MULTIPLIER
        if rmse < noise:
            cls = "SUSPICIOUS_BELOW_NOISE_FLOOR"
        elif rmse <= band_top:
            cls = "HEALTHY_IN_NOISE_BAND"
        elif rmse <= envelope:
            cls = "ABOVE_NOISE_WITHIN_ENVELOPE"
        else:
            cls = "OUTSIDE_TIER4_ENVELOPE"
        classifications.append(cls)
        fields[name] = {
            "operational_rmse_t1h": rmse,
            "noise_floor_rmse_24h": noise,
            "noise_floor_band_top_5x": band_top,
            "tier4_envelope": envelope,
            "classification": cls,
            "ratio_to_noise_floor": float(rmse / noise) if noise > 0 else float("nan"),
            "units": TIER4_UNITS[name],
            "noise_floor_source_notes": floor[name]["notes"],
        }
    suspicious = any(c == "SUSPICIOUS_BELOW_NOISE_FLOOR" for c in classifications)
    outside = any(c == "OUTSIDE_TIER4_ENVELOPE" for c in classifications)
    if suspicious:
        status = "COMPARATOR_SUSPICIOUS"
    elif outside:
        status = "OPERATIONAL_OUTSIDE_ENVELOPE"
    else:
        status = "PASS"
    return {
        "artifact_type": "m6b_tier4_rmse_dryrun_stage3",
        "stage": 3,
        "status": status,
        "noise_floor_csv": str(BASELINE_RMSE_CSV.relative_to(ROOT)),
        "noise_floor_lead_hours_used": NOISE_FLOOR_LEAD_HOURS,
        "noise_floor_band_multiplier": NOISE_FLOOR_BAND_MULTIPLIER,
        "fields": fields,
        "interpretation": {
            "SUSPICIOUS_BELOW_NOISE_FLOOR": "operational RMSE < Gen2 internal noise floor — comparator likely broken (too-good).",
            "HEALTHY_IN_NOISE_BAND": "operational RMSE in [noise_floor, 5*noise_floor] — expected healthy band.",
            "ABOVE_NOISE_WITHIN_ENVELOPE": "operational RMSE > 5*noise_floor but still under Tier-4 envelope.",
            "OUTSIDE_TIER4_ENVELOPE": "operational RMSE exceeds the 5x-noise Tier-4 envelope.",
        },
    }


def run_dryrun(run_id: str, hours: float, output_dir: Path) -> dict[str, Any]:
    forecast_state, run, stage1 = stage1_run_1h(run_id, hours)
    _write_json(output_dir / "proof_1h_run.json", stage1)

    if stage1["status"] != "PASS":
        stage2 = {
            "artifact_type": "m6b_tier4_rmse_dryrun_stage2",
            "stage": 2,
            "status": "NOT_RUN",
            "reason": f"stage1 failed: bounds/finiteness blocker on {run_id}",
        }
        _write_json(output_dir / "proof_tier4_rmse.json", stage2)
        stage3 = {
            "artifact_type": "m6b_tier4_rmse_dryrun_stage3",
            "stage": 3,
            "status": "NOT_RUN",
            "reason": "stage2 not run; stage1 blocker",
        }
        _write_json(output_dir / "proof_noise_floor_compare.json", stage3)
    else:
        stage2 = stage2_compute_tier4_rmse(forecast_state, run, hours)
        _write_json(output_dir / "proof_tier4_rmse.json", stage2)
        stage3 = stage3_noise_floor_compare(stage2)
        _write_json(output_dir / "proof_noise_floor_compare.json", stage3)

    summary = {
        "artifact_type": "m6b_tier4_rmse_dryrun_summary",
        "run_id": run_id,
        "hours": float(hours),
        "stage1_status": stage1["status"],
        "stage2_status": stage2["status"],
        "stage3_status": stage3["status"],
        "device": visible_gpu_name(),
        "thresholds": TIER4_THRESHOLDS,
        "spatial_ratio_threshold": SPATIAL_RATIO_THRESHOLD,
        "noise_floor_band_multiplier": NOISE_FLOOR_BAND_MULTIPLIER,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "proof_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Gen2 wrf_l3 run directory id (default: 20260429_18z).")
    parser.add_argument("--hours", type=float, default=1.0, help="Forecast lead in hours (default 1.0).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for proof JSONs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = run_dryrun(args.run_id, args.hours, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["stage1_status"] != "PASS":
        return 2
    if summary["stage3_status"] == "COMPARATOR_SUSPICIOUS":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
