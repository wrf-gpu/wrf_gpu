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

from gpuwrf.dynamics.advection import fixture_reference_update


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-stencil-3d-advdiff-v1.yaml"
SAMPLE = ROOT / "fixtures" / "samples" / "analytic-stencil-3d-advdiff-v1.npz"
ARTIFACT = ROOT / "artifacts" / "m4" / "tier1_advection_parity.json"


def _phi_tolerances(manifest: dict[str, Any]) -> tuple[float, float]:
    """Extracts phi_next tolerances from the pinned fixture manifest."""

    for variable in manifest["variables"]:
        if variable["name"] == "phi_next":
            return float(variable["tolerance_abs"]), float(variable["tolerance_rel"])
    raise KeyError("phi_next tolerance missing from manifest")


def run_tier1(out: Path = ARTIFACT) -> dict[str, Any]:
    """Runs the fixture operator wrapper and writes the tier-1 proof JSON."""

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    tolerance_abs, tolerance_rel = _phi_tolerances(manifest)
    with np.load(SAMPLE, allow_pickle=False) as loaded:
        phi = jnp.asarray(loaded["phi_initial"], dtype=jnp.float64)
        u = jnp.asarray(loaded["u_face"], dtype=jnp.float64)
        v = jnp.asarray(loaded["v_face"], dtype=jnp.float64)
        w = jnp.asarray(loaded["w_face"], dtype=jnp.float64)
        reference = np.asarray(loaded["phi_next"], dtype=np.float64)

    candidate = np.asarray(fixture_reference_update(phi, u, v, w, 3.0))
    diff = np.abs(candidate - reference)
    allowed = tolerance_abs + tolerance_rel * np.abs(reference)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / (np.abs(reference) + np.finfo(np.float64).eps)))
    record = {
        "fixture_id": "analytic-stencil-3d-advdiff-v1",
        "operator": "M1 centered 4H/2V advection-diffusion reference wrapper; dycore uses 5H/3V upwind",
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
