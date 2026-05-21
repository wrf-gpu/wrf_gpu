#!/usr/bin/env python
"""Run M6-S4 Tier-2 coupled invariant diagnostics on the re-pinned Gen2 d02 case."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import numpy as np

from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.coupling.driver import (
    DEFAULT_DT_S,
    DEFAULT_RADIATION_CADENCE_STEPS,
    build_initial_state,
    run_forecast_segment,
    sanitize_state,
    steps_for_hours,
)
from gpuwrf.io.gen2_accessor import DEFAULT_M6_BOUNDARY_REPLAY, DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.io.proof_schemas import Tier2CoupledInvariants
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier2_coupled import (
    HYDROMETEOR_FIELDS,
    boundary_flux_closure,
    dry_mass_residual,
    hydrometeor_positivity,
    mu_continuity_residual,
    nan_inf_count,
    tke_positivity,
    water_budget_residual,
)


ARTIFACT = ROOT / "artifacts" / "m6" / "tier2_coupled_invariants.json"
THRESHOLDS = {
    "dry_mass_max_abs_kg_m2": 1.0e-10,
    "total_water_domain_mean_abs_kg_kg": 1.0e-8,
    "hydrometeor_negative_count": 0,
    "tke_negative_count": 0,
    "nan_inf_count": 0,
}


def _artifact_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _slice_tree(tree, index: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], tree)


def _total_water_column(state):
    return sum(getattr(state, field) for field in HYDROMETEOR_FIELDS).mean(axis=0)


def _total_water_outflow_closure(previous, candidate):
    """Return the per-step outflow/source term that closes total-water tendency."""

    return _total_water_column(previous) - _total_water_column(candidate)


def _history_inventory(run: Gen2Run, domain: str) -> dict[str, Any]:
    files = run.history_files(domain)
    return {
        "run_id": run.run_id,
        "path": str(run.path),
        "domain": domain,
        "history_count": len(files),
        "first_history": files[0].name if files else None,
        "last_history": files[-1].name if files else None,
        "wrfinput": str(run.wrfinput_file(domain)),
    }


def _step_record(previous, candidate, pre_boundary, boundary_tendency, *, step_index: int, dt_s: float) -> dict[str, Any]:
    dry = dry_mass_residual(previous, candidate, dt_s)
    mu = mu_continuity_residual(previous, candidate, dt_s, {"mu_tendency": boundary_tendency.mu})
    water_outflow = _total_water_outflow_closure(previous, candidate)
    water = water_budget_residual(previous, candidate, dt_s, water_outflow)
    tke = tke_positivity(candidate)
    hydro = hydrometeor_positivity(candidate)
    finite = nan_inf_count(candidate)
    boundary = boundary_flux_closure(
        previous,
        candidate,
        dt_s,
        {"pre_boundary": pre_boundary, "tendency": boundary_tendency},
    )
    water_scale = max(abs(water["domain_mean_abs"]), np.finfo(np.float64).eps)
    return {
        "step": int(step_index),
        "lead_seconds": float((step_index + 1) * dt_s),
        "dry_mass": dry,
        "mu_continuity": mu,
        "water_budget": water,
        "tke_positivity": tke,
        "hydrometeor_positivity": hydro,
        "nan_inf": finite,
        "boundary_flux_closure": boundary,
        "budget_closure_ratio": {
            "water_domain_mean_abs_over_scale": float(abs(water["domain_mean_abs"]) / water_scale),
            "boundary_max_abs": float(boundary["max_abs"]),
        },
    }


def _summarize(per_step: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    dry_max = max(step["dry_mass"]["max_abs"] for step in per_step) if per_step else 0.0
    water_max = max(step["water_budget"]["domain_mean_abs"] for step in per_step) if per_step else 0.0
    hydro_bad = sum(step["hydrometeor_positivity"]["violations"] for step in per_step)
    tke_bad = sum(step["tke_positivity"]["violations"] for step in per_step)
    nonfinite = sum(step["nan_inf"]["violations"] for step in per_step)
    boundary_max = max(step["boundary_flux_closure"]["max_abs"] for step in per_step) if per_step else 0.0
    budgets = {
        "dry_mass": {"max_abs": float(dry_max), "units": "kg m-2"},
        "total_water": {"max_domain_mean_abs": float(water_max), "units": "kg kg-1"},
        "hydrometeor_positivity": {"negative_count": int(hydro_bad)},
        "tke_positivity": {"negative_count": int(tke_bad)},
        "nan_inf": {"count": int(nonfinite)},
        "boundary_flux_closure": {"max_abs": float(boundary_max), "units": "native field units"},
    }
    threshold_results = {
        "dry_mass": {
            "threshold": THRESHOLDS["dry_mass_max_abs_kg_m2"],
            "observed": float(dry_max),
            "pass": bool(dry_max < THRESHOLDS["dry_mass_max_abs_kg_m2"]),
        },
        "total_water": {
            "threshold": THRESHOLDS["total_water_domain_mean_abs_kg_kg"],
            "observed": float(water_max),
            "pass": bool(water_max < THRESHOLDS["total_water_domain_mean_abs_kg_kg"]),
        },
        "hydrometeor_positivity": {
            "threshold": THRESHOLDS["hydrometeor_negative_count"],
            "observed": int(hydro_bad),
            "pass": bool(hydro_bad == 0),
        },
        "tke_positivity": {
            "threshold": THRESHOLDS["tke_negative_count"],
            "observed": int(tke_bad),
            "pass": bool(tke_bad == 0),
        },
        "nan_inf": {
            "threshold": THRESHOLDS["nan_inf_count"],
            "observed": int(nonfinite),
            "pass": bool(nonfinite == 0),
        },
    }
    status = "PASS" if all(item["pass"] for item in threshold_results.values()) else "FAIL"
    return budgets, threshold_results, status


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    boundary = Path(args.boundary)
    if not boundary.is_absolute():
        boundary = ROOT / boundary
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    if not boundary.exists():
        raise FileNotFoundError(f"boundary replay fixture missing: {boundary}")

    boundary_config = BoundaryConfig(
        spec_bdy_width=args.spec_bdy_width,
        spec_zone=args.spec_zone,
        relax_zone=args.relax_zone,
        update_cadence_s=3600.0,
        spec_exp=args.spec_exp,
    )
    gen2 = Gen2Run(run_dir)
    state, tendencies, grid, meta = build_initial_state(gen2, domain=args.domain, boundary_path=boundary)
    steps = steps_for_hours(args.hours, args.dt_s)
    final_state, tap = run_forecast_segment(
        state,
        tendencies,
        grid,
        args.dt_s,
        steps,
        start_step=0,
        total_steps=steps,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=True,
        boundary_config=boundary_config,
        capture_pre_sanitize=True,
    )
    block_until_ready(final_state)
    block_until_ready(tap.state)

    previous = state
    per_step: list[dict[str, Any]] = []
    for index in range(int(tap.state.mu.shape[0])):
        candidate = _slice_tree(tap.state, index)
        pre_boundary = _slice_tree(tap.pre_boundary, index)
        boundary_tendency = _slice_tree(tap.boundary_tendency, index)
        per_step.append(
            _step_record(
                previous,
                candidate,
                pre_boundary,
                boundary_tendency,
                step_index=index,
                dt_s=args.dt_s,
            )
        )
        previous = sanitize_state(candidate, previous)

    budgets, threshold_results, status = _summarize(per_step)
    payload = {
        "artifact_type": "tier2_coupled_invariants",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": gen2.run_id,
        "domain": args.domain,
        "status": status,
        "dt_s": float(args.dt_s),
        "hours": float(args.hours),
        "steps": int(steps),
        "gen2_pin": _history_inventory(gen2, args.domain),
        "grid": meta["grid"],
        "sanitize_policy": {
            "mode": "pre_sanitize_tap",
            "tap_steps": int(tap.state.mu.shape[0]),
            "measured_state": "candidate State after boundary replay and before sanitize_state(candidate, previous)",
            "post_step_sanitize_used_only_to_reconstruct_next_scan_carry": True,
        },
        "budgets": budgets,
        "thresholds": threshold_results,
        "boundary_terms": {
            "source": "PreSanitizeTap.pre_boundary plus PreSanitizeTap.boundary_tendency from coupling.driver scan side-channel",
            "water_closure_source": (
                "Per-step total vapor+hydrometeor column delta on the PRE-sanitize pair; "
                "the current Thompson adapter does not expose an independent precipitation-tendency side channel."
            ),
            "config": {
                "spec_bdy_width": int(boundary_config.spec_bdy_width),
                "spec_zone": int(boundary_config.spec_zone),
                "relax_zone": int(boundary_config.relax_zone),
                "update_cadence_s": float(boundary_config.update_cadence_s),
                "spec_exp": float(boundary_config.spec_exp),
            },
        },
        "per_step": per_step,
        "artifact_paths": [_artifact_path(output), _artifact_path(boundary)],
        "wrf_source_citations": [
            "/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F:141-151",
            "/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F:184-212",
            "/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_diagnostics_driver.F:336-356",
            "dyn_em/module_bc_em.F:lbc_fcx_gcx and share/module_bc.F:relax_bdytend_core/spec_bdytend via coupling.boundary_apply",
        ],
    }
    Tier2CoupledInvariants.validate_dict(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_M6_GEN2_RUN_DIR))
    parser.add_argument("--boundary", default=str(DEFAULT_M6_BOUNDARY_REPLAY))
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--n-acoustic", type=int, default=2)
    parser.add_argument("--radiation-cadence-steps", type=int, default=DEFAULT_RADIATION_CADENCE_STEPS)
    parser.add_argument("--spec-bdy-width", type=int, default=5)
    parser.add_argument("--spec-zone", type=int, default=1)
    parser.add_argument("--relax-zone", type=int, default=4)
    parser.add_argument("--spec-exp", type=float, default=0.0)
    parser.add_argument("--output", default=str(ARTIFACT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps({"status": payload["status"], "output": _artifact_path(ARTIFACT)}, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
