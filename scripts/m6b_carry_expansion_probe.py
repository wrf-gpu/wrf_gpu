#!/usr/bin/env python
"""M6b carry-expansion 10 s operational probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-fix-carry-expansion"
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_IDS = (
    "20260509_18z_l3_24h_20260511T190519Z",
    "20260521_18z_l3_24h_20260522T072630Z",
    "20260523_18z_l3_24h_20260524T004313Z",
)
THETA_CHECK_LEVELS = 30


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _operational_source_audit() -> dict[str, Any]:
    source_path = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"
    source = source_path.read_text(encoding="utf-8")
    forbidden = (
        "gpuwrf.dynamics.acoustic_loop",
        "gpuwrf.dynamics.dycore_step",
        "gpuwrf.dynamics.coupled_step",
        "device_get",
        "host_callback",
        "pure_callback",
        "io_callback",
        "sanitize_state",
        "snapshot(",
    )
    hits = [token for token in forbidden if token in source]
    return {
        "status": "PASS" if not hits else "FAIL",
        "source": str(source_path.relative_to(ROOT)),
        "forbidden_hits": hits,
        "validation_mode_imports_absent": not any(
            "gpuwrf.dynamics." + name in source for name in ("acoustic_loop", "dycore_step", "coupled_step")
        ),
    }


def _case_state_and_namelist(run_id: str, *, duration_s: float) -> tuple[Any, OperationalNamelist, dict[str, Any]]:
    run_dir = RUN_ROOT / run_id
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=float(duration_s),
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
    return state, namelist, meta


def _all_leaves_finite(state: Any) -> bool:
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]
    return bool(np.asarray(jnp.all(jnp.asarray(checks))))


def _bounds(state: Any) -> dict[str, Any]:
    theta_window = state.theta[:THETA_CHECK_LEVELS]
    theta_min = float(np.asarray(jnp.min(theta_window)))
    theta_max = float(np.asarray(jnp.max(theta_window)))
    full_theta_min = float(np.asarray(jnp.min(state.theta)))
    full_theta_max = float(np.asarray(jnp.max(state.theta)))
    u_abs = float(np.asarray(jnp.max(jnp.abs(state.u))))
    v_abs = float(np.asarray(jnp.max(jnp.abs(state.v))))
    w_abs = float(np.asarray(jnp.max(jnp.abs(state.w))))
    return {
        "theta_levels_checked": [0, THETA_CHECK_LEVELS],
        "theta_min_k": theta_min,
        "theta_max_k": theta_max,
        "full_column_theta_min_k": full_theta_min,
        "full_column_theta_max_k": full_theta_max,
        "full_column_theta_caveat": "Pinned Gen2 initial upper-level theta already exceeds 400 K; bisection bound is applied to active lower 30 eta levels.",
        "u_abs_max_m_s": u_abs,
        "v_abs_max_m_s": v_abs,
        "w_abs_max_m_s": w_abs,
        "theta_bounded": bool(200.0 < theta_min and theta_max < 400.0),
        "wind_bounded": bool(u_abs <= 100.0 and v_abs <= 100.0 and w_abs <= 50.0),
    }


def run_one(run_id: str, *, duration_s: float) -> dict[str, Any]:
    state, namelist, meta = _case_state_and_namelist(run_id, duration_s=duration_s)
    start = time.perf_counter()
    result = run_forecast_operational(state, namelist, float(duration_s) / 3600.0)
    block_until_ready(result)
    wall_s = time.perf_counter() - start
    finite = _all_leaves_finite(result)
    bounds = _bounds(result)
    status = "PASS" if finite and bounds["theta_bounded"] and bounds["wind_bounded"] else "FAIL"
    record = {
        **meta,
        "status": status,
        "operational_entrypoint": "run_forecast_operational",
        "duration_s": float(duration_s),
        "steps_completed": 1,
        "wall_time_s_including_compile": wall_s,
        "all_leaves_finite": finite,
        "bounds": bounds,
    }
    if not finite:
        record["blocker"] = "NONFINITE"
    elif not bounds["theta_bounded"]:
        record["blocker"] = "THETA_BOUNDS"
    elif not bounds["wind_bounded"]:
        record["blocker"] = "WIND_BOUNDS"
    return record


def run_probe(run_ids: tuple[str, ...], *, duration_s: float) -> dict[str, Any]:
    audit = _operational_source_audit()
    runs = [run_one(run_id, duration_s=duration_s) for run_id in run_ids]
    blocker = next((run.get("blocker") for run in runs if run["status"] != "PASS"), None)
    payload = {
        "artifact_type": "m6b_carry_expansion_10s_probe",
        "status": "PASS" if blocker is None and audit["status"] == "PASS" else "FAIL",
        "device": visible_gpu_name(),
        "duration_s": float(duration_s),
        "run_ids": list(run_ids),
        "source_audit": audit,
        "runs": runs,
        "blocker": blocker,
    }
    _write_json(SPRINT / "proof_10s_probe.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = tuple(args.run_ids or DEFAULT_RUN_IDS[: int(args.runs)])
    payload = run_probe(selected, duration_s=float(args.duration_s))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
