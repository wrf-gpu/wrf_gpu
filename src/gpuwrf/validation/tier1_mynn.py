"""Tier-1 parity engine for the M5-S2 MYNN PBL fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml

from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, step_mynn_pbl_column


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ID = "analytic-mynn-pbl-column-v1"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
ARTIFACT = ROOT / "artifacts" / "m5" / "tier1_mynn_parity.json"
OUTPUT_FIELDS = ("u", "v", "w", "theta", "qv", "tke")


def _load_manifest() -> dict[str, Any]:
    """Reads the pinned MYNN manifest."""

    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _tolerance_map(manifest: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Extracts output tolerances keyed by physical field name."""

    out: dict[str, tuple[float, float]] = {}
    for variable in manifest["variables"]:
        name = str(variable["name"])
        if name.startswith("output_"):
            out[name.removeprefix("output_")] = (float(variable["tolerance_abs"]), float(variable["tolerance_rel"]))
    return out


def load_fixture_state(sample: Path = SAMPLE) -> tuple[MynnPBLColumnState, float, dict[str, np.ndarray]]:
    """Builds the JAX state and NumPy expected outputs from the fixture."""

    with np.load(sample, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    zeros = jnp.zeros_like(jnp.asarray(arrays["input_u"], dtype=jnp.float64))
    state = MynnPBLColumnState(
        u=jnp.asarray(arrays["input_u"], dtype=jnp.float64),
        v=jnp.asarray(arrays["input_v"], dtype=jnp.float64),
        w=jnp.asarray(arrays["input_w"], dtype=jnp.float64),
        theta=jnp.asarray(arrays["input_theta"], dtype=jnp.float64),
        qv=jnp.asarray(arrays["input_qv"], dtype=jnp.float64),
        tke=jnp.asarray(arrays["input_tke"], dtype=jnp.float64),
        p=jnp.asarray(arrays["input_p"], dtype=jnp.float64),
        rho=jnp.asarray(arrays["input_rho"], dtype=jnp.float64),
        dz=jnp.asarray(arrays["input_dz"], dtype=jnp.float64),
        km=zeros,
        kh=zeros,
        el=zeros,
    )
    expected = {field: np.asarray(arrays[f"output_{field}"], dtype=np.float64) for field in OUTPUT_FIELDS}
    return state, float(np.asarray(arrays["input_dt"])[0]), expected


def compare_against_fixture(state: MynnPBLColumnState, dt: float, expected: dict[str, np.ndarray], tolerances: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Runs the kernel once and compares every output field to the fixture."""

    candidate = step_mynn_pbl_column(state, dt, debug=False)
    per_abs: dict[str, float] = {}
    per_rel: dict[str, float] = {}
    pass_fields: dict[str, bool] = {}
    for field in OUTPUT_FIELDS:
        cand = np.asarray(getattr(candidate, field), dtype=np.float64)
        ref = expected[field]
        diff = np.abs(cand - ref)
        abs_tol, rel_tol = tolerances[field]
        allowed = abs_tol + rel_tol * np.abs(ref)
        per_abs[field] = float(np.max(diff))
        per_rel[field] = float(np.max(diff / (np.abs(ref) + np.finfo(np.float64).eps)))
        pass_fields[field] = bool(np.all(diff <= allowed))
    return {
        "fixture_id": FIXTURE_ID,
        "scenarios_tested": int(expected["u"].shape[0]),
        "per_field_max_abs_err": per_abs,
        "per_field_max_rel_err": per_rel,
        "tolerances_met": bool(all(pass_fields.values())),
        "field_pass": pass_fields,
        "pass": bool(all(pass_fields.values())),
    }


def run_tier1(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the required tier-1 MYNN parity proof JSON."""

    manifest = _load_manifest()
    state, dt, expected = load_fixture_state()
    record = compare_against_fixture(state, dt, expected, _tolerance_map(manifest))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
