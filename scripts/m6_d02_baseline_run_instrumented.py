#!/usr/bin/env python3
"""Run the M6.x S2 d02 1h baseline with S1 diagnostic sidecars.

This script is intentionally an orchestration layer. It does not import or
modify operator code in-process unless the preserved replay script completes and
produces proof files. The real replay is attempted first through
``scripts/m6_d02_boundary_replay_1h.py``; if the local GPU/JAX stack or Gen2 d02
data is unavailable, a synthetic Gen2-shaped fallback proof is produced and
classified loudly as a blocker for real S3 dispatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts/m6_d02_boundary_replay_1h.py"
SPRINT_DIR = ROOT / ".agent/sprints/2026-05-23-m6x-s2-d02-baseline-instrumented"
NOISE_FLOOR_CSV = ROOT / "data/fixtures/gen2_baseline/rmse_summary.csv"
SOURCE_TABLE = ROOT / ".agent/decisions/source_mining_operator_table.md"

SIDEcars = [
    ("bound_violation_tracer", "diagnostic_bound_violation_tracer.py"),
    ("sanitizer_audit", "diagnostic_sanitizer_audit.py"),
    ("limiter_activation_tracker", "diagnostic_limiter_activation_tracker.py"),
    ("field_rmse_timeline", "diagnostic_field_rmse_timeline.py"),
    ("spatial_divergence_map", "diagnostic_spatial_divergence_map.py"),
    ("conservation_tracker", "diagnostic_conservation_tracker.py"),
    ("boundary_ring_error_profiler", "diagnostic_boundary_ring_error_profiler.py"),
    ("vertical_column_phase_space", "diagnostic_vertical_column_phase_space.py"),
    ("operator_term_budget_tracer", "diagnostic_operator_term_budget_tracer.py"),
    ("transfer_launch_timeline", "diagnostic_transfer_launch_timeline.py"),
    ("timestep_convergence_dashboard", "diagnostic_timestep_convergence_dashboard.py"),
    ("stabilizer_provenance_scanner", "diagnostic_stabilizer_provenance_scanner.py"),
]

NOISE_ANCHORS = {
    "T2": {"lead_hours": 24, "spatial_mean_rmse": 0.6284061313, "units": "K"},
    "U10": {"lead_hours": 24, "spatial_mean_rmse": 1.456482265, "units": "m/s"},
    "V10": {"lead_hours": 24, "spatial_mean_rmse": 1.590974439, "units": "m/s"},
}
COMMAND_LOG_DIR = SPRINT_DIR / "artifacts/command-logs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--n-acoustic", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=SPRINT_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--radiation-cadence-steps", type=int, default=60)
    parser.add_argument("--probe-timeout-s", type=float, default=1800.0)
    parser.add_argument("--replay-timeout-s", type=float, default=36000.0)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--force-synthetic", action="store_true")
    return parser.parse_args(argv)


def rel(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(candidate)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_noise_floor(path: Path = NOISE_FLOOR_CSV) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return dict(NOISE_ANCHORS)
    anchors: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("lead_hours") != "24":
                continue
            field = str(row.get("field", ""))
            if field in {"T2", "U10", "V10"}:
                anchors[field] = {
                    "lead_hours": int(float(row["lead_hours"])),
                    "spatial_mean_rmse": float(row["spatial_mean_rmse"]),
                    "units": row.get("units", ""),
                    "source": rel(path),
                }
    return anchors or dict(NOISE_ANCHORS)


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("JAX_ENABLE_X64", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def run_command(cmd: list[str], *, cwd: Path = ROOT, timeout_s: float | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    COMMAND_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.time_ns()}_{os.getpid()}"
    stdout_path = COMMAND_LOG_DIR / f"{stamp}.stdout"
    stderr_path = COMMAND_LOG_DIR / f"{stamp}.stderr"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=command_env(),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        proc.wait(timeout=timeout_s if timeout_s and timeout_s > 0 else None)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass
    finally:
        stdout_handle.close()
        stderr_handle.close()
    elapsed = time.perf_counter() - started
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "elapsed_s": elapsed,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_path": rel(stdout_path),
        "stderr_path": rel(stderr_path),
    }


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_replay(args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    proof_path = out / "proof_d02_replay.json"
    fields_path = out / "artifacts/proof_d02_replay_fields.npz"
    trace_dir = out / "artifacts/trace_d02_replay"
    commands: list[dict[str, Any]] = []

    base_cmd = [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--duration-s",
        str(args.duration_s),
        "--dt-s",
        str(args.dt_s),
        "--n-acoustic",
        str(args.n_acoustic),
        "--radiation-cadence-steps",
        str(args.radiation_cadence_steps),
        "--output",
        str(proof_path),
        "--output-fields",
        str(fields_path),
        "--trace-dir",
        str(trace_dir),
    ]
    if args.run_dir is not None:
        base_cmd.extend(["--run-dir", str(args.run_dir)])

    if args.force_synthetic:
        fallback = synthetic_replay_payload(out, args, "forced synthetic mode requested")
        return fallback, {"commands": commands, "mode": "synthetic", "reason": "forced synthetic mode requested"}

    if not args.skip_probe:
        probe_cmd = [
            sys.executable,
            str(REPLAY_SCRIPT),
            "--duration-s",
            "1",
            "--dt-s",
            str(args.dt_s),
            "--n-acoustic",
            str(args.n_acoustic),
            "--radiation-cadence-steps",
            str(args.radiation_cadence_steps),
            "--skip-trace-audit",
            "--skip-static-audit",
            "--output",
            str(out / "artifacts/replay_probe.json"),
            "--output-fields",
            str(out / "artifacts/replay_probe_fields.npz"),
        ]
        if args.run_dir is not None:
            probe_cmd.extend(["--run-dir", str(args.run_dir)])
        print(f"[probe] {' '.join(probe_cmd)}", flush=True)
        probe = run_command(probe_cmd, timeout_s=args.probe_timeout_s)
        commands.append(probe)
        print(format_command_result("probe", probe), flush=True)
        if probe["timed_out"]:
            reason = f"real replay probe timed out after {args.probe_timeout_s:g}s"
            fallback = synthetic_replay_payload(out, args, reason)
            return fallback, {"commands": commands, "mode": "synthetic", "reason": reason}
        probe_json = out / "artifacts/replay_probe.json"
        if not probe_json.exists():
            reason = "real replay probe did not produce replay_probe.json"
            fallback = synthetic_replay_payload(out, args, reason)
            return fallback, {"commands": commands, "mode": "synthetic", "reason": reason}

    print(f"[replay] {' '.join(base_cmd)}", flush=True)
    replay = run_command(base_cmd, timeout_s=args.replay_timeout_s)
    commands.append(replay)
    print(format_command_result("replay", replay), flush=True)
    if proof_path.exists():
        payload = load_json(proof_path)
        payload.setdefault("output_fields_npz", str(fields_path))
        return payload, {"commands": commands, "mode": "real", "reason": None}

    reason = "real replay did not produce proof_d02_replay.json"
    if replay["timed_out"]:
        reason = f"real replay timed out after {args.replay_timeout_s:g}s"
    fallback = synthetic_replay_payload(out, args, reason)
    return fallback, {"commands": commands, "mode": "synthetic", "reason": reason}


def format_command_result(label: str, result: dict[str, Any]) -> str:
    return (
        f"[{label}] exit={result['returncode']} timeout={result['timed_out']} "
        f"elapsed_s={result['elapsed_s']:.3f} stdout_bytes={len(result['stdout'])} "
        f"stderr_bytes={len(result['stderr'])}"
    )


def synthetic_replay_payload(out: Path, args: argparse.Namespace, reason: str) -> dict[str, Any]:
    ny, nx = 24, 24
    y, x = np.indices((ny, nx), dtype=np.float64)
    reference = {
        "T2": 292.0 + 0.01 * y + 0.02 * x,
        "U10": 4.0 + 0.03 * y,
        "V10": -2.0 + 0.02 * x,
        "w_k20": 0.02 * np.sin(x / 3.0),
        "theta_k20": 302.0 + 0.02 * y,
    }
    forecast = {
        "T2": reference["T2"] + 0.4,
        "U10": reference["U10"] + 0.5,
        "V10": reference["V10"] - 0.6,
        "w_k20": reference["w_k20"] + 0.02,
        "theta_k20": reference["theta_k20"] + 0.3,
    }
    fields_path = out / "artifacts/proof_d02_replay_fields.npz"
    fields_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fields_path, **forecast, gen2_reference_path=np.asarray("synthetic://gen2-shaped-reference"))

    comparison = comparison_from_arrays(forecast, reference, args.duration_s / 3600.0, "synthetic://gen2-shaped-reference")
    payload = {
        "status": "SYNTHETIC_FALLBACK",
        "objective": "M6.x S2 synthetic fallback for ADR-023 1h Gen2 d02 boundary replay",
        "synthetic_fallback": True,
        "fallback_reason": reason,
        "run": {
            "run_dir": "synthetic://gen2-shaped",
            "domain": "d02",
            "boundary": {"schema": "synthetic-side-pack-v1", "source": "fallback only, not a Gen2 baseline"},
            "grid": {"mass_shape": [40, ny, nx], "dx_m": 3000.0, "dy_m": 3000.0},
        },
        "duration_s": float(args.duration_s),
        "dt_s": float(args.dt_s),
        "steps": int(round(float(args.duration_s) / float(args.dt_s))),
        "n_acoustic": int(args.n_acoustic),
        "wall_time_s": 0.0,
        "forecast_throughput_x_realtime": None,
        "first_nonfinite_step": None,
        "diagnostics": {
            "all_state_leaves_finite": True,
            "theta_min_k": float(np.nanmin(forecast["theta_k20"])),
            "theta_max_k": float(np.nanmax(forecast["theta_k20"])),
            "qv_min_kg_kg": 0.004,
            "qv_max_kg_kg": 0.016,
            "u_abs_max_m_s": float(np.nanmax(np.abs(forecast["U10"]))),
            "v_abs_max_m_s": float(np.nanmax(np.abs(forecast["V10"]))),
            "w_abs_max_m_s": float(np.nanmax(np.abs(forecast["w_k20"]))),
            "p_min_pa": 70000.0,
            "p_max_pa": 101000.0,
            "first_nonfinite_step": None,
            "first_candidate_nonfinite_step": None,
            "candidate_nonfinite_steps": 0,
            "candidate_nonfinite_count_total": 0,
            "candidate_clip_count_total": 0,
            "candidate_changed_count_total": 0,
            "peak_w_abs_m_s": float(np.nanmax(np.abs(forecast["w_k20"]))),
            "theta_min_over_run_k": float(np.nanmin(forecast["theta_k20"])),
            "theta_max_over_run_k": float(np.nanmax(forecast["theta_k20"])),
        },
        "comparison": comparison,
        "transfer_audit": {
            "static": {
                "method": "not run - synthetic fallback",
                "host_callback_free": False,
                "forbidden_tokens": ["host_callback", "io_callback", "pure_callback"],
                "jaxpr_bytes": None,
            },
            "trace": {
                "method": "not run - synthetic fallback",
                "host_to_device_bytes_post_init": None,
                "device_to_host_bytes_post_init": None,
                "post_init_total_bytes": None,
                "trace_dir": None,
                "trace_transfer_event_files": [],
            },
            "host_device_transfer_bytes_post_init": None,
        },
        "peak_gpu_memory": {"device": "unavailable", "bytes_in_use": None, "peak_bytes_in_use": None, "raw_keys": []},
        "invoked_schemes": [],
        "output_fields_npz": str(fields_path),
        "proof_notes": [
            "Synthetic fallback is not a Gen2 d02 baseline and cannot support physics or GPU-performance claims.",
            "Fallback exists only to exercise the S1 diagnostic sidecars when infrastructure blocks the real replay.",
        ],
        "synthetic_reference": {name: value.tolist() for name, value in reference.items()},
    }
    write_json(out / "proof_d02_replay.json", payload)
    return payload


def comparison_from_arrays(
    forecast: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    lead_hours: float,
    reference_path: str,
) -> dict[str, Any]:
    units = {"T2": "K", "U10": "m s-1", "V10": "m s-1", "w_k20": "m s-1", "theta_k20": "K"}
    rmse: dict[str, Any] = {}
    drift: dict[str, Any] = {}
    max_abs: dict[str, Any] = {}
    shapes: dict[str, Any] = {}
    for name, pred in forecast.items():
        ref = reference[name]
        error = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        rmse[name] = {"value": float(np.sqrt(np.nanmean(error * error))), "units": units.get(name, "")}
        max_abs[name] = {"value": float(np.nanmax(np.abs(error))), "units": units.get(name, "")}
        shapes[name] = {"forecast": list(pred.shape), "reference": list(ref.shape)}
        if name in {"T2", "U10", "V10"}:
            drift[name] = {"value": float(np.nanmean(error)), "units": units.get(name, "")}
    return {
        "lead_hours": float(lead_hours),
        "valid_time_utc": "synthetic",
        "gen2_reference_path": reference_path,
        "rmse": rmse,
        "spatial_mean_drift": drift,
        "max_abs_error": max_abs,
        "shapes": shapes,
    }


def load_output_arrays(proof: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    forecast: dict[str, np.ndarray] = {}
    reference: dict[str, np.ndarray] = {}
    aux: dict[str, np.ndarray] = {}
    fields_path = proof.get("output_fields_npz")
    if fields_path and Path(fields_path).exists():
        with np.load(fields_path, allow_pickle=False) as data:
            for name in ("T2", "U10", "V10", "w_k20", "theta_k20"):
                if name in data:
                    forecast[name] = np.asarray(data[name], dtype=np.float64)

    synthetic_ref = proof.get("synthetic_reference")
    if isinstance(synthetic_ref, dict):
        for name, values in synthetic_ref.items():
            reference[name] = np.asarray(values, dtype=np.float64)
        y, x = np.indices(reference["T2"].shape, dtype=np.float64)
        aux["elevation_m"] = 50.0 + 3600.0 * np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / 120.0)
        aux["landmask"] = np.where(x < x.mean(), 1.0, 0.0)
        return forecast, reference, aux

    ref_path = proof.get("comparison", {}).get("gen2_reference_path")
    if not ref_path:
        return forecast, reference, aux
    path = Path(ref_path)
    if not path.exists():
        return forecast, reference, aux

    try:
        import xarray as xr  # type: ignore
    except Exception:
        return forecast, reference, aux

    try:
        with xr.open_dataset(path) as ds:
            for name in ("T2", "U10", "V10"):
                if name in ds:
                    reference[name] = np.asarray(ds[name].isel(Time=0).values, dtype=np.float64)
            if "W" in ds:
                reference["w_k20"] = np.asarray(ds["W"].isel(Time=0, bottom_top_stag=20).values, dtype=np.float64)
            if "T" in ds:
                reference["theta_k20"] = np.asarray(ds["T"].isel(Time=0, bottom_top=20).values, dtype=np.float64) + 300.0
            if "HGT" in ds:
                aux["elevation_m"] = np.asarray(ds["HGT"].isel(Time=0).values, dtype=np.float64)
            if "XLAND" in ds:
                aux["landmask"] = np.asarray(ds["XLAND"].isel(Time=0).values, dtype=np.float64)
    except Exception:
        return forecast, reference, aux
    return forecast, reference, aux


def sidecar_inputs(
    out: Path,
    proof: dict[str, Any],
    forecast: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    aux: dict[str, np.ndarray],
) -> dict[str, Path]:
    inputs = out / "artifacts/sidecar-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    diagnostics = proof.get("diagnostics", {})
    duration_s = float(proof.get("duration_s", 0.0) or 0.0)
    steps = int(proof.get("steps", 0) or 0)

    bound_rows = [
        {"step": steps, "time_s": duration_s, "theta_K": diagnostics.get("theta_min_k"), "label": "theta_min"},
        {"step": steps, "time_s": duration_s, "theta_K": diagnostics.get("theta_max_k"), "label": "theta_max"},
        {"step": steps, "time_s": duration_s, "qv": diagnostics.get("qv_min_kg_kg"), "label": "qv_min"},
        {"step": steps, "time_s": duration_s, "qv": diagnostics.get("qv_max_kg_kg"), "label": "qv_max"},
        {"step": steps, "time_s": duration_s, "w_abs_m_s": diagnostics.get("w_abs_max_m_s"), "label": "w_abs_max"},
        {"step": steps, "time_s": duration_s, "p_min_pa": diagnostics.get("p_min_pa"), "label": "p_min"},
        {"step": steps, "time_s": duration_s, "p_max_pa": diagnostics.get("p_max_pa"), "label": "p_max"},
    ]
    input_paths = {
        "bound_violation_tracer": write_json(
            inputs / "bound_violation_tracer.input.json",
            {
                "series": bound_rows,
                "bounds": {
                    "w_abs_m_s": {"max": 50.0, "units": "m s-1"},
                    "p_min_pa": {"min": 1000.0, "units": "Pa"},
                    "p_max_pa": {"max": 120000.0, "units": "Pa"},
                },
            },
        ),
        "sanitizer_audit": write_json(inputs / "sanitizer_audit.input.json", proof),
        "limiter_activation_tracker": write_json(
            inputs / "limiter_activation_tracker.input.json",
            {
                "limiter_steps": [],
                "note": "Current preserved replay proof exposes candidate sanitize counts, not raw_dmu/bounded_dmu arrays.",
            },
        ),
        "field_rmse_timeline": write_json(inputs / "field_rmse_timeline.input.json", proof),
        "operator_term_budget_tracer": write_json(
            inputs / "operator_term_budget_tracer.input.json",
            {
                "terms": {},
                "note": "Current preserved replay proof does not expose per-RHS operator terms without operator-side hooks.",
            },
        ),
        "transfer_launch_timeline": write_json(inputs / "transfer_launch_timeline.input.json", proof),
        "timestep_convergence_dashboard": write_json(
            inputs / "timestep_convergence_dashboard.input.json",
            {
                "dt_pairs": [],
                "note": "S2 is a baseline replay; S4 owns controlled dt-pair convergence.",
            },
        ),
        "stabilizer_provenance_scanner": write_json(
            inputs / "stabilizer_provenance_scanner.input.json",
            {
                "source_files": [
                    "src/gpuwrf/dynamics/acoustic_wrf.py",
                    "src/gpuwrf/dynamics/damping.py",
                    "src/gpuwrf/integration/d02_replay.py",
                ]
            },
        ),
    }

    comparable = sorted(set(forecast) & set(reference))
    field = "T2" if "T2" in comparable else (comparable[0] if comparable else None)
    spatial_payload: dict[str, Any] = {"field": field}
    if field is not None:
        spatial_payload["forecast"] = {field: forecast[field].tolist()}
        spatial_payload["reference"] = {field: reference[field].tolist()}
        spatial_payload["elevation_m"] = aux.get("elevation_m", np.zeros_like(forecast[field])).tolist()
        spatial_payload["landmask"] = aux.get("landmask", np.ones_like(forecast[field])).tolist()
    input_paths["spatial_divergence_map"] = write_json(inputs / "spatial_divergence_map.input.json", spatial_payload)
    input_paths["boundary_ring_error_profiler"] = write_json(inputs / "boundary_ring_error_profiler.input.json", spatial_payload)

    input_paths["vertical_column_phase_space"] = write_json(
        inputs / "vertical_column_phase_space.input.json",
        build_column_payload(forecast, reference, aux),
    )
    input_paths["conservation_tracker"] = write_json(
        inputs / "conservation_tracker.input.json",
        build_conservation_payload(forecast, reference),
    )
    return input_paths


def build_column_payload(
    forecast: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    aux: dict[str, np.ndarray],
) -> dict[str, Any]:
    field = forecast["theta_k20"] if "theta_k20" in forecast else forecast.get("T2")
    if field is None:
        return {"columns": [], "note": "No forecast fields available for selected-column diagnostics."}
    shape = field.shape
    elevation = aux.get("elevation_m", np.zeros(shape, dtype=np.float64))
    landmask = aux.get("landmask", np.ones(shape, dtype=np.float64))
    teide_j, teide_i = np.unravel_index(int(np.nanargmax(elevation)), elevation.shape)
    boundary_j, boundary_i = min(2, shape[0] - 1), min(2, shape[1] - 1)
    ocean_candidates = np.argwhere(landmask > 1.5)
    if ocean_candidates.size == 0:
        ocean_candidates = np.argwhere(landmask < 0.5)
    if ocean_candidates.size == 0:
        ocean_j, ocean_i = np.unravel_index(int(np.nanargmin(elevation)), elevation.shape)
    else:
        ocean_j, ocean_i = map(int, ocean_candidates[0])
    columns = []
    for name, j, i in (
        ("boundary_zone", int(boundary_j), int(boundary_i)),
        ("mount_teide_proxy", int(teide_j), int(teide_i)),
        ("ocean_proxy", int(ocean_j), int(ocean_i)),
    ):
        w_forecast = value_at(forecast, "w_k20", j, i)
        theta_forecast = value_at(forecast, "theta_k20", j, i, fallback=value_at(forecast, "T2", j, i))
        w_ref = value_at(reference, "w_k20", j, i, fallback=0.0)
        theta_ref = value_at(reference, "theta_k20", j, i, fallback=value_at(reference, "T2", j, i, fallback=0.0))
        columns.append(
            {
                "name": name,
                "i": i,
                "j": j,
                "profiles": {"w": [w_forecast], "theta": [theta_forecast]},
                "time_series": {
                    "w": [w_ref, w_forecast],
                    "theta": [theta_ref, theta_forecast],
                    "p": [0.0, 0.0],
                    "mu": [0.0, 0.0],
                },
                "note": "Only k20 forecast slices are available from the preserved replay field NPZ.",
            }
        )
    return {"columns": columns}


def value_at(fields: dict[str, np.ndarray], name: str, j: int, i: int, fallback: float | None = None) -> float:
    if name not in fields:
        return float(0.0 if fallback is None else fallback)
    arr = np.asarray(fields[name], dtype=np.float64)
    return float(arr[min(j, arr.shape[0] - 1), min(i, arr.shape[1] - 1)])


def build_conservation_payload(forecast: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    def totals(fields: dict[str, np.ndarray]) -> dict[str, float]:
        t2 = np.asarray(fields.get("T2", np.zeros((1, 1))), dtype=np.float64)
        u10 = np.asarray(fields.get("U10", np.zeros_like(t2)), dtype=np.float64)
        v10 = np.asarray(fields.get("V10", np.zeros_like(t2)), dtype=np.float64)
        theta = np.asarray(fields.get("theta_k20", t2), dtype=np.float64)
        return {
            "surface_temperature_sum": float(np.nansum(t2)),
            "surface_wind_energy_proxy": float(0.5 * np.nansum(u10 * u10 + v10 * v10)),
            "theta_k20_sum": float(np.nansum(theta)),
        }

    states = []
    if reference:
        states.append({"step": 0, "time_s": 0.0, "totals": totals(reference), "source_terms": {}, "boundary_terms": {}})
    if forecast:
        states.append({"step": 1, "time_s": None, "totals": totals(forecast), "source_terms": {}, "boundary_terms": {}})
    return {
        "states": states,
        "note": "Proxy totals use fields written by the preserved replay NPZ; full mass/water/energy leaves are not serialized.",
    }


def run_sidecars(out: Path, input_paths: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    proofs: dict[str, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    for name, script in SIDEcars:
        output = out / f"proof_{name}.json"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--input",
            str(input_paths[name]),
            "--output",
            str(output),
        ]
        print(f"[sidecar:{name}] {' '.join(cmd)}", flush=True)
        result = run_command(cmd, timeout_s=300.0)
        commands.append(result)
        print(format_command_result(f"sidecar:{name}", result), flush=True)
        if output.exists():
            proof = load_json(output)
        else:
            proof = {
                "schema_version": "m6x-s1-diagnostic-sidecar-v1",
                "diagnostic": {"name": name},
                "status": "SIDEcar_FAILED",
                "measurements": {},
                "command_result": result,
            }
            write_json(output, proof)
        proofs[name] = {"path": rel(output), "payload": proof, "command": result}
    return proofs, commands


def summarize(
    out: Path,
    proof: dict[str, Any],
    replay_meta: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    noise: dict[str, dict[str, Any]],
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    findings = classify_findings(proof, replay_meta, sidecars, noise)
    top_numeric = top_numeric_findings(proof, noise)
    s3 = s3_priorities(proof, sidecars, replay_meta)
    summary = {
        "schema_version": "m6x-s2-baseline-instrumented-v1",
        "objective": "Current ADR-023 1h Gen2 d02 replay baseline with S1 sidecars, no operator changes.",
        "status": overall_status(proof, replay_meta, findings),
        "replay_mode": replay_meta.get("mode"),
        "fallback_reason": replay_meta.get("reason"),
        "duration_s": proof.get("duration_s"),
        "dt_s": proof.get("dt_s"),
        "steps": proof.get("steps"),
        "n_acoustic": proof.get("n_acoustic"),
        "wall_time_s": proof.get("wall_time_s"),
        "forecast_completed": replay_meta.get("mode") == "real" and proof.get("first_nonfinite_step") is None,
        "sanitizer_masking": sanitizer_summary(sidecars),
        "transfer_audit": proof.get("transfer_audit", {}),
        "noise_floor": noise,
        "top_numeric_findings": top_numeric,
        "findings": findings,
        "s3_priorities": s3,
        "proof_paths": {
            "replay": rel(out / "proof_d02_replay.json"),
            "summary": rel(out / "proof_s2_baseline_summary.json"),
            **{name: item["path"] for name, item in sidecars.items()},
        },
        "commands": [command_record(item) for item in commands],
        "known_gaps": known_gaps(proof, sidecars),
    }
    write_json(out / "proof_s2_baseline_summary.json", summary)
    return summary


def command_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": " ".join(result.get("cmd", [])),
        "returncode": result.get("returncode"),
        "timed_out": result.get("timed_out"),
        "elapsed_s": result.get("elapsed_s"),
        "stdout_tail": tail(result.get("stdout", "")),
        "stderr_tail": tail(result.get("stderr", "")),
        "stdout_path": result.get("stdout_path"),
        "stderr_path": result.get("stderr_path"),
    }


def tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def overall_status(proof: dict[str, Any], replay_meta: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    if replay_meta.get("mode") != "real":
        return "BLOCKER_SYNTHETIC_FALLBACK"
    if any(item["classification"] == "BLOCKER" for item in findings):
        return "BLOCKER"
    if any(item["classification"] == "NEW-FINDING-NEEDS-S3-FIX" for item in findings):
        return "NEEDS_S3_FIX"
    return "BASELINE_MEASURED"


def sanitizer_summary(sidecars: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = sidecars.get("sanitizer_audit", {}).get("payload", {})
    measurements = payload.get("measurements", {})
    return {
        "status": payload.get("status"),
        "first_bad_candidate_step": measurements.get("first_bad_candidate_step"),
        "totals": measurements.get("totals"),
        "post_sanitize_only_pass_risk": measurements.get("post_sanitize_only_pass_risk"),
    }


def classify_findings(
    proof: dict[str, Any],
    replay_meta: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    noise: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if replay_meta.get("mode") != "real":
        findings.append(
            {
                "id": "F01",
                "classification": "BLOCKER",
                "title": "Real Gen2 d02 baseline was not produced",
                "detail": replay_meta.get("reason") or "real replay unavailable",
                "proof": rel(SPRINT_DIR / "proof_d02_replay.json"),
            }
        )
    elif proof.get("first_nonfinite_step") is not None:
        findings.append(
            {
                "id": "F01",
                "classification": "BLOCKER",
                "title": "Replay became nonfinite",
                "detail": f"first_nonfinite_step={proof.get('first_nonfinite_step')}",
                "proof": rel(SPRINT_DIR / "proof_d02_replay.json"),
            }
        )

    sanitizer = sidecars.get("sanitizer_audit", {}).get("payload", {})
    if sanitizer.get("status") == "FAIL_SANITIZER_MASKING":
        findings.append(
            {
                "id": "F02",
                "classification": "EXPECTED-BAD",
                "title": "Sanitizer changed pre-sanitize candidates",
                "detail": json.dumps(sanitizer.get("measurements", {}).get("totals", {}), sort_keys=True),
                "proof": sidecars["sanitizer_audit"]["path"],
            }
        )
    else:
        findings.append(
            {
                "id": "F02",
                "classification": "OK-WITHIN-NOISE",
                "title": "No sanitizer masking detected by available counts",
                "detail": f"status={sanitizer.get('status')}",
                "proof": sidecars.get("sanitizer_audit", {}).get("path"),
            }
        )

    transfer = sidecars.get("transfer_launch_timeline", {}).get("payload", {})
    transfer_status = transfer.get("status")
    if transfer_status == "OK":
        classification = "OK-WITHIN-NOISE"
    else:
        classification = "BLOCKER" if replay_meta.get("mode") == "real" else "BLOCKER"
    findings.append(
        {
            "id": "F03",
            "classification": classification,
            "title": "Transfer audit status",
            "detail": f"status={transfer_status}; post_init_total_transfer_bytes="
            f"{transfer.get('measurements', {}).get('post_init_total_transfer_bytes')}",
            "proof": sidecars.get("transfer_launch_timeline", {}).get("path"),
        }
    )

    for field in ("T2", "U10", "V10"):
        rmse_item = proof.get("comparison", {}).get("rmse", {}).get(field)
        if not isinstance(rmse_item, dict):
            continue
        rmse = float(rmse_item.get("value", math.inf))
        anchor = float(noise.get(field, {}).get("spatial_mean_rmse", math.inf))
        findings.append(
            {
                "id": f"RMSE-{field}",
                "classification": "OK-WITHIN-NOISE" if rmse <= anchor else "NEW-FINDING-NEEDS-S3-FIX",
                "title": f"{field} RMSE against Gen2 reference",
                "detail": f"1h_rmse={rmse:.6g}; 24h_noise_floor={anchor:.6g}; "
                "24h floor is an anchor, not a binding 1h threshold",
                "proof": sidecars.get("field_rmse_timeline", {}).get("path"),
            }
        )

    limiter = sidecars.get("limiter_activation_tracker", {}).get("payload", {})
    if limiter.get("input", {}).get("step_count", 0) == 0:
        findings.append(
            {
                "id": "F04",
                "classification": "NEW-FINDING-NEEDS-S3-FIX",
                "title": "Limiter raw_dmu telemetry is not exposed by the preserved replay proof",
                "detail": "The sidecar ran, but the current replay proof only exposes sanitizer counts, not raw/bounded dmu arrays.",
                "proof": sidecars.get("limiter_activation_tracker", {}).get("path"),
            }
        )

    terms = sidecars.get("operator_term_budget_tracer", {}).get("payload", {})
    if terms.get("status") == "NO_TERMS":
        findings.append(
            {
                "id": "F05",
                "classification": "NEW-FINDING-NEEDS-S3-FIX",
                "title": "Operator RHS term budget is not exposed by the preserved replay proof",
                "detail": "No per-term replay arrays are serialized without operator-side hooks.",
                "proof": sidecars.get("operator_term_budget_tracer", {}).get("path"),
            }
        )

    stabilizer = sidecars.get("stabilizer_provenance_scanner", {}).get("payload", {})
    counts = stabilizer.get("measurements", {}).get("classification_counts", {})
    if counts:
        findings.append(
            {
                "id": "F06",
                "classification": "EXPECTED-BAD" if counts.get("experiment-backed", 0) else "OK-WITHIN-NOISE",
                "title": "Stabilizer provenance scan found non-source-backed stabilizer-like code",
                "detail": json.dumps(counts, sort_keys=True),
                "proof": sidecars.get("stabilizer_provenance_scanner", {}).get("path"),
            }
        )
    return findings


def top_numeric_findings(proof: dict[str, Any], noise: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    diagnostics = proof.get("diagnostics", {})
    for key in (
        "candidate_nonfinite_count_total",
        "candidate_clip_count_total",
        "candidate_changed_count_total",
        "peak_w_abs_m_s",
        "theta_min_over_run_k",
        "theta_max_over_run_k",
    ):
        if key in diagnostics:
            out.append({"metric": key, "value": diagnostics[key], "proof": rel(SPRINT_DIR / "proof_d02_replay.json")})
    for field, item in proof.get("comparison", {}).get("rmse", {}).items():
        value = item.get("value") if isinstance(item, dict) else None
        row = {"metric": f"rmse_{field}", "value": value, "proof": rel(SPRINT_DIR / "proof_d02_replay.json")}
        if field in noise:
            row["noise_floor_24h"] = noise[field]["spatial_mean_rmse"]
        out.append(row)
    return out[:10]


def s3_priorities(
    proof: dict[str, Any],
    sidecars: dict[str, dict[str, Any]],
    replay_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "concern": "_mu_continuity_increment temporary limiter / hidden mass cap",
            "source_table_reference": f"{rel(SOURCE_TABLE)} row `_mu_continuity_increment` limiter concern",
            "proof_cites": [
                sidecars.get("sanitizer_audit", {}).get("path"),
                sidecars.get("limiter_activation_tracker", {}).get("path"),
                sidecars.get("stabilizer_provenance_scanner", {}).get("path"),
            ],
            "recommended_fix": "Replace or explicitly ratify mass update against WRF MUAVE/MUTS/ww or MPAS perturbation-state lines; do not use Rayleigh damping as a mass limiter replacement.",
            "expected_effect": "Reduce sanitizer/limiter masking and improve U10/V10/T2 drift only if mass continuity is the dominant error source.",
        },
        {
            "priority": 2,
            "concern": "MPAS_OMEGA_TO_W_METRIC = 1.35 constant metric",
            "source_table_reference": f"{rel(SOURCE_TABLE)} row `MPAS_OMEGA_TO_W_METRIC = 1.35` concern",
            "proof_cites": [
                sidecars.get("vertical_column_phase_space", {}).get("path"),
                sidecars.get("operator_term_budget_tracer", {}).get("path"),
            ],
            "recommended_fix": "Replace with per-column/per-level mass-flux metric from MPAS zz geometry, or keep the constant only in the synthetic slice oracle.",
            "expected_effect": "Should improve w_k20/theta_k20 phase behavior and any terrain-amplified spatial divergence.",
        },
        {
            "priority": 3,
            "concern": "MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38 and missing WRF time averaging",
            "source_table_reference": f"{rel(SOURCE_TABLE)} rows `0.38` buoyancy scale and time averaging",
            "proof_cites": [
                sidecars.get("field_rmse_timeline", {}).get("path"),
                sidecars.get("operator_term_budget_tracer", {}).get("path"),
            ],
            "recommended_fix": "Demote 0.38 to slice-only unless pinned by fixture evidence; derive production buoyancy from WRF t_2ave/muave or MPAS coupled coefficient terms.",
            "expected_effect": "Should primarily improve W/theta error growth before surface RMSE if buoyancy forcing is currently mis-scaled.",
        },
    ]


def known_gaps(proof: dict[str, Any], sidecars: dict[str, dict[str, Any]]) -> list[str]:
    gaps = []
    if proof.get("synthetic_fallback"):
        gaps.append("Real Gen2 d02 replay did not complete; synthetic fallback cannot support physics or performance claims.")
    limiter = sidecars.get("limiter_activation_tracker", {}).get("payload", {})
    if limiter.get("input", {}).get("step_count", 0) == 0:
        gaps.append("raw_dmu/bounded_dmu arrays are not serialized by the preserved replay scaffold.")
    terms = sidecars.get("operator_term_budget_tracer", {}).get("payload", {})
    if terms.get("status") == "NO_TERMS":
        gaps.append("Per-substep RHS term arrays are not serialized by the preserved replay scaffold.")
    timestep = sidecars.get("timestep_convergence_dashboard", {}).get("payload", {})
    if timestep.get("status") == "STRUCTURE_ONLY":
        gaps.append("Tier-3 timestep convergence remains S4 scope; S2 does not claim convergence.")
    return gaps


def write_findings(out: Path, summary: dict[str, Any]) -> Path:
    lines = [
        "# Findings - M6.x S2 Instrumented Baseline",
        "",
        f"Replay mode: `{summary['replay_mode']}`",
        f"Overall status: `{summary['status']}`",
        "",
        "Gen2 24h noise-floor anchors used for context only: T2 0.628 K, U10 1.46 m/s, V10 1.59 m/s.",
        "",
        "## Classified Findings",
        "",
    ]
    for item in summary["findings"]:
        lines.extend(
            [
                f"### {item['id']} - {item['classification']}",
                "",
                f"{item['title']}.",
                "",
                f"Detail: {item['detail']}",
                "",
                f"Proof: `{item.get('proof')}`",
                "",
            ]
        )
    lines.extend(["## Known Gaps", ""])
    for gap in summary["known_gaps"]:
        lines.append(f"- {gap}")
    lines.append("")
    return write_text(out / "findings.md", "\n".join(lines))


def write_s3_memo(out: Path, summary: dict[str, Any]) -> Path:
    lines = [
        "# S3 Input Memo - M6.x S2 Baseline",
        "",
        "## Top Operator Concerns",
        "",
    ]
    for item in summary["s3_priorities"]:
        lines.extend(
            [
                f"### {item['priority']}. {item['concern']}",
                "",
                f"Source table: {item['source_table_reference']}",
                "",
                "Proof cites:",
            ]
        )
        for proof in item["proof_cites"]:
            lines.append(f"- `{proof}`")
        lines.extend(
            [
                "",
                f"Recommended source-cited fix: {item['recommended_fix']}",
                "",
                f"Expected baseline effect: {item['expected_effect']}",
                "",
            ]
        )
    blocker = summary["status"].startswith("BLOCKER")
    exit_status = (
        "BLOCKER memo warranted before a real S3 fix sprint; the real d02 baseline is missing."
        if blocker
        else "S3 plus one bounded fix sprint is plausible, but only if it removes or ratifies limiter/sanitizer masking before Tier-3 claims."
    )
    lines.extend(["## Exit-Rule Status", "", exit_status, ""])
    return write_text(out / "s3_input_memo.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    print("[m6x-s2] starting measurement-only d02 baseline orchestration", flush=True)
    print(f"[m6x-s2] output_dir={out}", flush=True)
    proof, replay_meta = run_replay(args, out)
    noise = load_noise_floor()
    forecast, reference, aux = load_output_arrays(proof)
    inputs = sidecar_inputs(out, proof, forecast, reference, aux)
    sidecars, sidecar_commands = run_sidecars(out, inputs)
    all_commands = list(replay_meta.get("commands", [])) + sidecar_commands
    summary = summarize(out, proof, replay_meta, sidecars, noise, all_commands)
    write_findings(out, summary)
    write_s3_memo(out, summary)

    print("[m6x-s2] wrote proof_s2_baseline_summary.json", flush=True)
    print(json.dumps({"status": summary["status"], "replay_mode": summary["replay_mode"], "fallback_reason": summary["fallback_reason"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
