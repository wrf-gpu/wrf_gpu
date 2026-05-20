"""Tier-1 parity engine for the M5 Thompson column fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml

from gpuwrf.physics.thompson_column import ThompsonColumnState, density_from_pressure_temperature, step_thompson_column


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ID = "analytic-thompson-column-v1"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
ARTIFACT = ROOT / "artifacts" / "m5" / "tier1_thompson_parity.json"
OUTPUT_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T")


def _load_manifest() -> dict[str, Any]:
    """Reads the pinned Thompson manifest for tolerances and fixture metadata."""

    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _tolerance_map(manifest: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Extracts output tolerances keyed by physical output field name."""

    out: dict[str, tuple[float, float]] = {}
    for variable in manifest["variables"]:
        name = str(variable["name"])
        if name.startswith("output_"):
            out[name.removeprefix("output_")] = (float(variable["tolerance_abs"]), float(variable["tolerance_rel"]))
    return out


def load_fixture_state(sample: Path = SAMPLE) -> tuple[ThompsonColumnState, float, dict[str, np.ndarray]]:
    """Builds the JAX column state and NumPy expected outputs from the fixture."""

    with np.load(sample, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    qv = jnp.asarray(arrays["input_qv"], dtype=jnp.float64)
    T = jnp.asarray(arrays["input_T"], dtype=jnp.float64)
    p = jnp.asarray(arrays["input_p"], dtype=jnp.float64)
    rho = density_from_pressure_temperature(p, T, qv)
    state = ThompsonColumnState(
        qv=qv,
        qc=jnp.asarray(arrays["input_qc"], dtype=jnp.float64),
        qr=jnp.asarray(arrays["input_qr"], dtype=jnp.float64),
        qi=jnp.asarray(arrays["input_qi"], dtype=jnp.float64),
        qs=jnp.asarray(arrays["input_qs"], dtype=jnp.float64),
        qg=jnp.asarray(arrays["input_qg"], dtype=jnp.float64),
        Ni=jnp.asarray(arrays["input_Ni"], dtype=jnp.float64),
        Nr=jnp.asarray(arrays["input_Nr"], dtype=jnp.float64),
        T=T,
        p=p,
        rho=rho,
    )
    expected = {field: np.asarray(arrays[f"output_{field}"], dtype=np.float64) for field in OUTPUT_FIELDS}
    return state, float(np.asarray(arrays["input_dt"])[0]), expected


def compare_against_fixture(state: ThompsonColumnState, dt: float, expected: dict[str, np.ndarray], tolerances: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Runs the kernel once and compares every output field to the fixture."""

    candidate = step_thompson_column(state, dt, debug=False)
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
        "scenarios_tested": int(expected["T"].shape[0]),
        "per_field_max_abs_err": per_abs,
        "per_field_max_rel_err": per_rel,
        "tolerances_met": bool(all(pass_fields.values())),
        "field_pass": pass_fields,
        "pass": bool(all(pass_fields.values())),
    }


def run_tier1(out: Path = ARTIFACT) -> dict[str, Any]:
    """Writes the required tier-1 Thompson parity proof JSON."""

    manifest = _load_manifest()
    state, dt, expected = load_fixture_state()
    record = compare_against_fixture(state, dt, expected, _tolerance_map(manifest))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
