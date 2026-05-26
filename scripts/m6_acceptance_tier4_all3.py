#!/usr/bin/env python
"""M6 acceptance gate: Tier-4 RMSE at +1h for all three V3 ICs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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

from gpuwrf.integration.d02_replay import _reference_fields, _surface_fields, build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)

RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_OUTPUT = ROOT / ".agent" / "sprints" / "2026-05-26-m6-acceptance-tier4-all3-ics"
CLOSEOUT = ROOT / ".agent" / "decisions" / "MILESTONE-M6-CLOSEOUT.md"

RUNS = (
    {
        "run_id": "20260429_18z_l3_24h_20260524T204451Z",
        "ic_time": "2026-04-29_18:00:00",
    },
    {
        "run_id": "20260509_18z_l3_24h_20260511T190519Z",
        "ic_time": "2026-05-09_18:00:00",
    },
    {
        "run_id": "20260521_18z_l3_24h_20260522T072630Z",
        "ic_time": "2026-05-21_18:00:00",
    },
)

TIER4_THRESHOLDS = {"T2": 3.0, "U10": 7.5, "V10": 7.5}
TIER4_UNITS = {"T2": "K", "U10": "m s-1", "V10": "m s-1"}
WIND_LIMITS = {"u_abs_max_m_s": 100.0, "v_abs_max_m_s": 100.0, "w_abs_max_m_s": 50.0}
DT_S = 10.0
LEAD_HOURS = 1.0
STEPS_1H = int(round(LEAD_HOURS * 3600.0 / DT_S))

FIX_REPORTS = (
    ".agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/worker-report.md",
    ".agent/sprints/2026-05-26-m6b-coftz-theta-fix/worker-report.md",
    ".agent/sprints/2026-05-26-m6b-operational-theta-fix/worker-report.md",
    ".agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/worker-report.md",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else str(scalar)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        dt_s=DT_S,
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


def _bounds_for_state(state: Any, *, step: int) -> dict[str, Any]:
    theta = state.theta
    values = {
        "step": int(step),
        "lead_seconds": float(step) * DT_S,
        "all_leaves_finite": _all_leaves_finite(state),
        "theta_min_k": float(np.asarray(jnp.min(theta))),
        "theta_max_k": float(np.asarray(jnp.max(theta))),
        "u_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.u)))),
        "v_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.v)))),
        "w_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.w)))),
    }
    theta_ok = 200.0 <= values["theta_min_k"] and values["theta_max_k"] <= 700.0
    wind_ok = (
        values["u_abs_max_m_s"] <= WIND_LIMITS["u_abs_max_m_s"]
        and values["v_abs_max_m_s"] <= WIND_LIMITS["v_abs_max_m_s"]
        and values["w_abs_max_m_s"] <= WIND_LIMITS["w_abs_max_m_s"]
    )
    values["theta_bounded"] = bool(theta_ok)
    values["wind_bounded"] = bool(wind_ok)
    values["passed"] = bool(values["all_leaves_finite"] and theta_ok and wind_ok)
    return values


def _run_bounds_and_forecast(run_id: str) -> tuple[Any, Any, dict[str, Any]]:
    state, namelist, run, meta = _case_state_and_namelist(run_id)
    current = state
    timeline: list[dict[str, Any]] = []
    first_failure: dict[str, Any] | None = None
    step_hours = float(namelist.dt_s) / 3600.0
    start = time.perf_counter()
    for step in range(1, STEPS_1H + 1):
        current = run_forecast_operational(current, namelist, step_hours)
        block_until_ready(current)
        row = _bounds_for_state(current, step=step)
        timeline.append(row)
        if first_failure is None and not row["passed"]:
            first_failure = row
    wall_s = time.perf_counter() - start
    final_bounds = timeline[-1]
    extrema = {
        "theta_min_k": min(float(row["theta_min_k"]) for row in timeline),
        "theta_max_k": max(float(row["theta_max_k"]) for row in timeline),
        "u_abs_max_m_s": max(float(row["u_abs_max_m_s"]) for row in timeline),
        "v_abs_max_m_s": max(float(row["v_abs_max_m_s"]) for row in timeline),
        "w_abs_max_m_s": max(float(row["w_abs_max_m_s"]) for row in timeline),
    }
    passed = all(bool(row["passed"]) for row in timeline)
    proof = {
        "artifact_type": "m6_acceptance_bounds_1h",
        "status": "PASS" if passed else "FAIL",
        "run_id": run_id,
        "lead_hours": LEAD_HOURS,
        "steps": STEPS_1H,
        "dt_s": float(namelist.dt_s),
        "device": visible_gpu_name(),
        "wall_time_s_including_compile": wall_s,
        "bounds_policy": {
            "theta_k": [200.0, 700.0],
            "wind_abs_limits_m_s": WIND_LIMITS,
            "finite_all_state_leaves_required": True,
        },
        "extrema_over_360_steps": extrema,
        "final_bounds": final_bounds,
        "first_failure": first_failure,
        "timeline": timeline,
        "meta": meta,
    }
    return current, run, proof


def _local_error_stats(predicted: Any, reference: Any) -> dict[str, Any]:
    err = np.asarray(predicted, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    abs_err = np.abs(err)
    if not bool(np.all(np.isfinite(abs_err))):
        return {
            "all_finite": False,
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
            "spatial_ratio_max_over_mean": float("nan"),
            "shape": list(err.shape),
        }
    mean_abs = float(np.mean(abs_err))
    max_abs = float(np.max(abs_err))
    return {
        "all_finite": True,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "spatial_ratio_max_over_mean": float(max_abs / mean_abs) if mean_abs > 0.0 else float("nan"),
        "shape": list(err.shape),
    }


def _tier4_rmse(forecast_state: Any, run: Any, run_id: str) -> dict[str, Any]:
    surface = _surface_fields(forecast_state)
    reference, reference_path, valid_time = _reference_fields(run, "d02", LEAD_HOURS)
    fields: dict[str, Any] = {}
    all_pass = True
    for name, threshold in TIER4_THRESHOLDS.items():
        pred = surface[name]
        ref = reference[name]
        if pred.shape != ref.shape:
            raise ValueError(f"{run_id} {name} shape mismatch: forecast {pred.shape} vs reference {ref.shape}")
        err = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        rmse = float(np.sqrt(np.nanmean(err * err)))
        local = _local_error_stats(pred, ref)
        passed = bool(np.isfinite(rmse) and rmse <= threshold and local["all_finite"])
        fields[name] = {
            "rmse_spatial_mean": rmse,
            "rmse_threshold": float(threshold),
            "rmse_pass": bool(np.isfinite(rmse) and rmse <= threshold),
            "local_max_abs_error": local["max_abs"],
            "local_mean_abs_error": local["mean_abs"],
            "spatial_heterogeneity_ratio_max_abs_over_mean_abs": local["spatial_ratio_max_over_mean"],
            "spatial_heterogeneity_ratio_policy": "informational_only",
            "all_finite": local["all_finite"],
            "shape": local["shape"],
            "units": TIER4_UNITS[name],
            "passed": passed,
        }
        all_pass = all_pass and passed
    return {
        "artifact_type": "m6_acceptance_tier4_rmse_single_ic",
        "status": "PASS" if all_pass else "FAIL",
        "run_id": run_id,
        "lead_hours": LEAD_HOURS,
        "gen2_reference_path": str(reference_path),
        "valid_time_utc": valid_time,
        "thresholds": TIER4_THRESHOLDS,
        "fields": fields,
    }


def _run_command(command: list[str], output_base: Path, *, env: dict[str, str]) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    wall_s = time.perf_counter() - start
    output_base.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output_base.with_suffix(".stdout.txt")
    stderr_path = output_base.with_suffix(".stderr.txt")
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "wall_time_s": wall_s,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_parity_commands(run_id: str, ic_time: str, output: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    run_dir = RUN_ROOT / run_id
    ic_path = run_dir / f"wrfout_d02_{ic_time}"
    slug = run_id.split("_18z_", 1)[0]
    b6 = _run_command(
        [
            sys.executable,
            "scripts/m6b6_coupled_step_compare.py",
            "--tier",
            "all",
            "--source-wrfout",
            str(ic_path),
            "--output",
            str(output / "parity" / f"{slug}_b6_coupled_step_parity.json"),
        ],
        output / "command_logs" / f"{slug}_b6_coupled_step_compare",
        env=env,
    )
    multi = _run_command(
        [
            sys.executable,
            "scripts/m6b_real_ic_operational_compare.py",
            "--gen2-run-id",
            run_id,
            "--gen2-ic-time",
            ic_time,
            "--steps",
            "10",
        ],
        output / "command_logs" / f"{slug}_m6b_real_ic_operational_compare_steps10",
        env=env,
    )
    return {"b6_coupled_step": b6, "multi_step_parity": multi, "passed": bool(b6["passed"] and multi["passed"])}


def _skipped_parity(run_id: str) -> dict[str, Any]:
    skipped = {
        "run_id": run_id,
        "status": "NOT_RUN",
        "passed": False,
        "reason": "Tier-4 acceptance was already blocked by bounds/RMSE; standalone contract validation commands remain authoritative for parity.",
    }
    return {"b6_coupled_step": skipped, "multi_step_parity": skipped, "passed": False}


def _aggregate_rmse(per_ic: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    all_pass = True
    for name, threshold in TIER4_THRESHOLDS.items():
        values = [float(item["rmse"]["fields"][name]["rmse_spatial_mean"]) for item in per_ic.values()]
        mean_rmse = float(np.mean(values))
        passed = bool(np.isfinite(mean_rmse) and mean_rmse <= threshold)
        fields[name] = {
            "per_ic_rmse": values,
            "mean_across_3_ics": mean_rmse,
            "threshold": float(threshold),
            "units": TIER4_UNITS[name],
            "passed": passed,
        }
        all_pass = all_pass and passed
    return {"status": "PASS" if all_pass else "FAIL", "fields": fields}


def _write_closeout(output: Path, bounds_parity: dict[str, Any], rmse: dict[str, Any], summary: dict[str, Any]) -> None:
    caveats = [
        "V workaround is an acceptance-preserving suppression, not a completed root-cause proof.",
        "Microphysics coupling guards prevent invalid feedback, but deeper boundary/dynamics audits remain M7 risks.",
        "This closeout proves M6 acceptance thresholds, not GPU performance readiness.",
    ]
    lines = [
        "# MILESTONE M6 CLOSEOUT",
        "",
        "Status: M6-CLOSED",
        "",
        "## Acceptance Evidence",
        "",
    ]
    for report in FIX_REPORTS:
        lines.append(f"- Fix sprint report: `{report}`")
    lines.extend(
        [
            f"- Bounds + parity proof: `{output.relative_to(ROOT) / 'proof_bounds_parity.json'}`",
            f"- Tier-4 RMSE proof: `{output.relative_to(ROOT) / 'proof_tier4_rmse_all3.json'}`",
            f"- Acceptance summary: `{output.relative_to(ROOT) / 'proof_acceptance_summary.json'}`",
            "",
            "## Gate Result",
            "",
            f"- Stage 1 bounds/parity: {bounds_parity['status']}",
            f"- Stage 2 Tier-4 RMSE: {rmse['status']}",
            f"- Final status: {summary['status']}",
            "",
            "## Outstanding Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in caveats)
    lines.extend(
        [
            "",
            "## Recommended Next Steps For M7",
            "",
            "- Run the profiling gate before any speed claim.",
            "- Start GPU optimization only after transfer audit remains clean on the coupled path.",
            "- Compare wall-clock against the 28-rank CPU WRF operational baseline.",
            "- Keep boundary/dynamics and guarded-physics audits explicit in M7 risk tracking.",
            "",
        ]
    )
    _write_text(CLOSEOUT, "\n".join(lines))


def run_acceptance(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    per_ic: dict[str, Any] = {}
    for item in RUNS:
        run_id = str(item["run_id"])
        ic_time = str(item["ic_time"])
        forecast, run, bounds = _run_bounds_and_forecast(run_id)
        _write_json(output / f"proof_bounds_{run_id}.json", bounds)
        rmse = _tier4_rmse(forecast, run, run_id)
        _write_json(output / f"proof_tier4_rmse_{run_id}.json", rmse)
        per_ic[run_id] = {"ic_time": ic_time, "bounds": bounds, "rmse": rmse, "parity": None}

    aggregate = _aggregate_rmse(per_ic)
    per_ic_rmse_pass = all(item["rmse"]["status"] == "PASS" for item in per_ic.values())
    bounds_pass = all(item["bounds"]["status"] == "PASS" for item in per_ic.values())
    if bounds_pass and per_ic_rmse_pass and aggregate["status"] == "PASS":
        for run_id, item in per_ic.items():
            item["parity"] = _run_parity_commands(run_id, str(item["ic_time"]), output)
    else:
        for run_id, item in per_ic.items():
            item["parity"] = _skipped_parity(run_id)

    stage1_pass = all(
        item["bounds"]["status"] == "PASS" and bool(item["parity"]["passed"]) for item in per_ic.values()
    )
    bounds_parity = {
        "artifact_type": "m6_acceptance_bounds_parity_all3",
        "status": "PASS" if stage1_pass else "FAIL",
        "device": visible_gpu_name(),
        "runs": {
            run_id: {
                "ic_time": item["ic_time"],
                "bounds_status": item["bounds"]["status"],
                "bounds_extrema_over_360_steps": item["bounds"]["extrema_over_360_steps"],
                "bounds_first_failure": item["bounds"]["first_failure"],
                "b6_coupled_step": item["parity"]["b6_coupled_step"],
                "multi_step_parity": item["parity"]["multi_step_parity"],
                "passed": bool(item["bounds"]["status"] == "PASS" and item["parity"]["passed"]),
            }
            for run_id, item in per_ic.items()
        },
    }
    _write_json(output / "proof_bounds_parity.json", bounds_parity)

    rmse_all = {
        "artifact_type": "m6_acceptance_tier4_rmse_all3",
        "status": "PASS" if per_ic_rmse_pass and aggregate["status"] == "PASS" else "FAIL",
        "thresholds": TIER4_THRESHOLDS,
        "per_ic": {run_id: item["rmse"] for run_id, item in per_ic.items()},
        "aggregate_mean_across_3_ics": aggregate,
        "spatial_heterogeneity_ratio_policy": "informational_only",
    }
    _write_json(output / "proof_tier4_rmse_all3.json", rmse_all)

    final_pass = bounds_parity["status"] == "PASS" and rmse_all["status"] == "PASS"
    summary = {
        "artifact_type": "m6_acceptance_summary",
        "status": "M6-CLOSED" if final_pass else "M6-BLOCKED-ACCEPTANCE-GATE",
        "stage1_bounds_parity": bounds_parity["status"],
        "stage2_tier4_rmse": rmse_all["status"],
        "closeout_written": bool(final_pass),
        "closeout_path": str(CLOSEOUT) if final_pass else None,
        "proofs": [
            str(output / "proof_bounds_parity.json"),
            str(output / "proof_tier4_rmse_all3.json"),
        ],
    }
    _write_json(output / "proof_acceptance_summary.json", summary)
    if final_pass:
        _write_closeout(output, bounds_parity, rmse_all, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_acceptance(args.output)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "M6-CLOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
