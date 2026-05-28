"""Tier-1 advection parity against the M1 analytic stencil fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import yaml

from gpuwrf.dynamics.advection import advect_mass_scalar
from gpuwrf.validation.tier2 import make_ideal_grid


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ID = "analytic-stencil-3d-upwind5-v1"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
ARTIFACT = ROOT / "artifacts" / "m4" / "tier1_advection_parity.json"


def _phi_tolerances(manifest: dict[str, Any]) -> tuple[float, float]:
    """Extracts phi_next tolerances from the pinned fixture manifest."""

    for variable in manifest["variables"]:
        if variable["name"] == "phi_next_upwind5":
            return float(variable["tolerance_abs"]), float(variable["tolerance_rel"])
    raise KeyError("phi_next_upwind5 tolerance missing from manifest")


def run_tier1(out: Path = ARTIFACT) -> dict[str, Any]:
    """Runs the fixture operator wrapper and writes the tier-1 proof JSON."""

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    tolerance_abs, tolerance_rel = _phi_tolerances(manifest)
    with np.load(SAMPLE, allow_pickle=False) as loaded:
        phi = jnp.asarray(loaded["phi_initial"], dtype=jnp.float64)
        u = jnp.asarray(loaded["u_face"], dtype=jnp.float64)
        v = jnp.asarray(loaded["v_face"], dtype=jnp.float64)
        w = jnp.asarray(loaded["w_face"], dtype=jnp.float64)
        reference = np.asarray(loaded["phi_next_upwind5"], dtype=np.float64)

    grid = make_ideal_grid(8, 16, 32, dx_m=900.0, dy_m=900.0, top_m=960.0)
    u_mass = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_mass = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    w_mass = 0.5 * (w[:-1, :, :] + w[1:, :, :])
    candidate = np.asarray(phi + 3.0 * advect_mass_scalar(phi, u_mass, v_mass, w_mass, grid))
    diff = np.abs(candidate - reference)
    allowed = tolerance_abs + tolerance_rel * np.abs(reference)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / (np.abs(reference) + np.finfo(np.float64).eps)))
    record = {
        "fixture_id": FIXTURE_ID,
        "operator": "dycore 5th-order horizontal periodic and 3rd-order vertical no-wrap upwind mass-scalar advection",
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "tolerance_abs": tolerance_abs,
        "tolerance_rel": tolerance_rel,
        "pass": bool(np.all(diff <= allowed)),
        "jax_version": jax.__version__,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
