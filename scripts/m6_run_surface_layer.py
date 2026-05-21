#!/usr/bin/env python
"""Generate M6-S3 surface-layer proof artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_state import build_land_state_manifest, load_prescribed_land_state
from gpuwrf.io.proof_schemas import SurfaceLayerArtifact, validate_artifact
from gpuwrf.io.validation import load_gen2_var
from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics


DEFAULT_RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z")
ARTIFACT_DIR = ROOT / "artifacts" / "m6"


class _SurfaceState:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


def _rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        validate_artifact(path)
    except KeyError:
        SurfaceLayerArtifact.validate_dict(payload)


def radiation_feasibility(run: Gen2Run, domain: str) -> dict[str, Any]:
    required = ("RTHRATEN", "RTHRATSW", "RTHRATLW")
    history = run.history_files(domain)
    real_history = [path for path in history if path.name.startswith(f"wrfout_{domain}_")]
    inventory: dict[str, Any] = {}
    if real_history:
        with Dataset(real_history[0], "r") as dataset:
            for name in required:
                if name in dataset.variables:
                    var = dataset.variables[name]
                    inventory[name] = {
                        "available": True,
                        "dimensions": list(var.dimensions),
                        "shape": [int(item) for item in var.shape],
                        "units": getattr(var, "units", ""),
                    }
                else:
                    inventory[name] = {"available": False}
    else:
        inventory = {name: {"available": False, "reason": "no real wrfout_d02_* history files"} for name in required}
    available = bool(real_history) and all(inventory[name]["available"] for name in required)
    return {
        "artifact_type": "radiation_conditioning_feasibility",
        "status": "PASS" if available else "BLOCKED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "domain": domain,
        "history_file_count": len(real_history),
        "history_files_sample": [str(path) for path in real_history[:3]],
        "variables": inventory,
        "decision": (
            "M6-S3 can prescribe Gen2 radiation tendencies" if available else "M6-S3 deviation: RRTMG online remains required"
        ),
        "artifact_paths": [_rel(ARTIFACT_DIR / "radiation_conditioning_feasibility.json")],
    }


def _mass_winds(run: Gen2Run, domain: str):
    u = np.asarray(load_gen2_var(run, domain, "U", 0), dtype=np.float64)
    v = np.asarray(load_gen2_var(run, domain, "V", 0), dtype=np.float64)
    return 0.5 * (u[:, :, :-1] + u[:, :, 1:]), 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def _surface_input_state(run: Gen2Run, domain: str, *, prescribed_roughness: bool):
    land = load_prescribed_land_state(run, domain, 0)
    u, v = _mass_winds(run, domain)
    theta = np.asarray(load_gen2_var(run, domain, "T", 0), dtype=np.float64) + 300.0
    qv = np.asarray(load_gen2_var(run, domain, "QVAPOR", 0), dtype=np.float64)
    p = np.asarray(load_gen2_var(run, domain, "P", 0), dtype=np.float64) + np.asarray(
        load_gen2_var(run, domain, "PB", 0), dtype=np.float64
    )
    ph = np.asarray(load_gen2_var(run, domain, "PH", 0), dtype=np.float64) + np.asarray(
        load_gen2_var(run, domain, "PHB", 0), dtype=np.float64
    )
    dz = np.maximum((ph[1:, :, :] - ph[:-1, :, :]) / 9.80665, 1.0)
    kwargs = {
        "u": jnp.moveaxis(jnp.asarray(u), 0, -1),
        "v": jnp.moveaxis(jnp.asarray(v), 0, -1),
        "theta": jnp.moveaxis(jnp.asarray(theta), 0, -1),
        "qv": jnp.moveaxis(jnp.asarray(qv), 0, -1),
        "p": jnp.moveaxis(jnp.asarray(p), 0, -1),
        "dz": jnp.moveaxis(jnp.asarray(dz), 0, -1),
        "t_skin": land.t_skin,
        "soil_moisture": land.soil_moisture[0],
        "xland": land.xland,
        "lakemask": land.lakemask,
        "mavail": land.mavail,
        "ustar": jnp.zeros_like(land.t_skin),
    }
    if prescribed_roughness:
        kwargs["roughness_m"] = land.roughness_m
    return _SurfaceState(**kwargs)


def _neutral_diagnostics(state: _SurfaceState):
    u0 = np.asarray(state.u)[..., 0]
    v0 = np.asarray(state.v)[..., 0]
    theta0 = np.asarray(state.theta)[..., 0]
    qv0 = np.asarray(state.qv)[..., 0]
    p0 = np.asarray(state.p)[..., 0]
    dz = np.asarray(state.dz)[..., 0]
    t_skin = np.asarray(state.t_skin)
    z0 = np.asarray(getattr(state, "roughness_m", np.ones_like(u0) * 0.10))
    za = np.maximum(0.5 * dz, 2.1)
    denom = np.maximum(np.log(np.maximum(za / np.maximum(z0, 1.0e-7), 1.000001)), 1.0e-6)
    u10 = u0 * np.log(np.maximum(10.0 / np.maximum(z0, 1.0e-7), 1.000001)) / denom
    v10 = v0 * np.log(np.maximum(10.0 / np.maximum(z0, 1.0e-7), 1.000001)) / denom
    theta_skin = t_skin * (P0_PA / p0) ** R_D_OVER_CP
    theta2 = theta_skin + (theta0 - theta_skin) * np.log(np.maximum(2.0 / np.maximum(z0, 1.0e-7), 1.000001)) / denom
    t2 = theta2 * (p0 / P0_PA) ** R_D_OVER_CP
    q2 = qv0
    return {"U10": u10, "V10": v10, "T2": t2, "Q2": q2}


def _rmse(candidate, truth) -> float:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.nanmean(diff * diff)))


def operational_delta(run: Gen2Run, domain: str) -> dict[str, Any]:
    state = _surface_input_state(run, domain, prescribed_roughness=True)
    before = _neutral_diagnostics(state)
    after_diag = surface_layer_with_diagnostics(state)
    after = {
        "U10": np.asarray(after_diag.u10),
        "V10": np.asarray(after_diag.v10),
        "T2": np.asarray(after_diag.t2),
        "Q2": np.asarray(after_diag.q2),
    }
    truth = {name: np.asarray(load_gen2_var(run, domain, name, 0), dtype=np.float64) for name in ("U10", "V10", "T2", "Q2")}
    variables: dict[str, Any] = {}
    for name in ("U10", "V10", "T2", "Q2"):
        before_rmse = _rmse(before[name], truth[name])
        after_rmse = _rmse(after[name], truth[name])
        variables[name] = {
            "0h": {
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
                "delta_after_minus_before": after_rmse - before_rmse,
                "units": "m s-1" if name in {"U10", "V10"} else ("K" if name == "T2" else "kg kg-1"),
            }
        }
    return {
        "artifact_type": "surface_operational_delta",
        "status": "PARTIAL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.run_id,
        "domain": domain,
        "variables": variables,
        "operational_delta": {
            "method": "lead-0 wrfinput_d02 surface diagnostics; no d02 wrfout history exists for 1h/6h/12h/24h truth",
            "before": "neutral log-profile diagnostic surrogate matching the M5 bulk/no-stability surface placeholder",
            "after": "M6-S3 MM5 sfclay surface_layer_with_diagnostics with prescribed land state",
            "binding_limitation": "full M6-S2 before/after forecast RMSE delta is blocked by missing wrfout_d02_* history",
        },
        "prerequisites": {
            "F-S3-1": {
                "status": "PARTIAL",
                "evidence": "lead-0 diagnostic path has no sanitize_state call; full 1h/6h/12h ON/OFF attribution remains blocked pending a no-sanitize forecast run.",
            },
            "F-S3-2": {
                "status": "WAIVED_TO_INTERIOR_ONLY",
                "evidence": "Gen2Run.history_files('d02') falls back to wrfinput_d02; no wrfout_d02_* exists locally.",
            },
            "F-S3-3": {
                "status": "PASS_BY_CODE_AUDIT",
                "evidence": "surface_adapter still casts ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv through DEFAULT_DTYPES FP64 registry.",
            },
        },
        "artifact_paths": [_rel(ARTIFACT_DIR / "surface_operational_delta.json")],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    gen2 = Gen2Run(args.run_dir)
    radiation = radiation_feasibility(gen2, args.domain)
    radiation_path = ARTIFACT_DIR / "radiation_conditioning_feasibility.json"
    _write_json(radiation_path, radiation)

    land_manifest = build_land_state_manifest(gen2, args.domain, 0)
    land_manifest["artifact_paths"] = [_rel(ARTIFACT_DIR / "land_state_manifest.json")]
    land_path = ARTIFACT_DIR / "land_state_manifest.json"
    _write_json(land_path, land_manifest)

    delta = operational_delta(gen2, args.domain)
    delta["artifact_paths"].extend([_rel(radiation_path), _rel(land_path)])
    delta_path = ARTIFACT_DIR / "surface_operational_delta.json"
    _write_json(delta_path, delta)
    return {
        "radiation_conditioning_feasibility": _rel(radiation_path),
        "land_state_manifest": _rel(land_path),
        "surface_operational_delta": _rel(delta_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--domain", default="d02")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
