"""Tier-1 parity engines for the M5-S3 RRTMG fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml

from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.validation.proof_write import should_write_proof as _should_write_proof


ROOT = Path(__file__).resolve().parents[3]
SW_FIXTURE_ID = "analytic-rrtmg-sw-column-v1"
LW_FIXTURE_ID = "analytic-rrtmg-lw-column-v1"
SW_MANIFEST = ROOT / "fixtures" / "manifests" / f"{SW_FIXTURE_ID}.yaml"
LW_MANIFEST = ROOT / "fixtures" / "manifests" / f"{LW_FIXTURE_ID}.yaml"
SW_SAMPLE = ROOT / "fixtures" / "samples" / f"{SW_FIXTURE_ID}.npz"
LW_SAMPLE = ROOT / "fixtures" / "samples" / f"{LW_FIXTURE_ID}.npz"
SW_ARTIFACT = ROOT / "artifacts" / "m5" / "tier1_rrtmg_sw_parity.json"
LW_ARTIFACT = ROOT / "artifacts" / "m5" / "tier1_rrtmg_lw_parity.json"
OUTPUT_FIELDS = (
    "heating_rate",
    "flux_down",
    "flux_up",
    "toa_down",
    "toa_up",
    "surface_down",
    "surface_up",
)
SW_EXTRA_FIELDS = ("column_absorbed", "surface_absorbed")
LW_EXTRA_FIELDS = ("column_net_heating", "surface_emission")


def _load_manifest(path: Path) -> dict[str, Any]:
    """Reads one pinned RRTMG manifest."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tolerance_map(manifest: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Extracts output tolerances keyed by physical field name."""

    out: dict[str, tuple[float, float]] = {}
    for variable in manifest["variables"]:
        name = str(variable["name"])
        if name.startswith("output_"):
            out[name.removeprefix("output_")] = (float(variable["tolerance_abs"]), float(variable["tolerance_rel"]))
    return out


def _arrays(path: Path) -> dict[str, np.ndarray]:
    """Loads an RRTMG fixture sample."""

    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name], dtype=np.float64) for name in loaded.files}


def load_sw_fixture_state(sample: Path = SW_SAMPLE) -> tuple[RRTMGSWColumnState, dict[str, np.ndarray]]:
    """Builds the JAX SW state and NumPy expected outputs."""

    arrays = _arrays(sample)
    state = RRTMGSWColumnState(
        T=jnp.asarray(arrays["input_T"], dtype=jnp.float64),
        p=jnp.asarray(arrays["input_p"], dtype=jnp.float64),
        qv=jnp.asarray(arrays["input_qv"], dtype=jnp.float64),
        qc=jnp.asarray(arrays["input_qc"], dtype=jnp.float64),
        qi=jnp.asarray(arrays["input_qi"], dtype=jnp.float64),
        qs=jnp.asarray(arrays["input_qs"], dtype=jnp.float64),
        qg=jnp.asarray(arrays["input_qg"], dtype=jnp.float64),
        cloud_fraction=jnp.asarray(arrays["input_cloud_fraction"], dtype=jnp.float64),
        surface_albedo=jnp.asarray(arrays["input_surface_albedo"], dtype=jnp.float64),
        coszen=jnp.asarray(arrays["input_coszen"], dtype=jnp.float64),
        dz=jnp.asarray(arrays["input_dz"], dtype=jnp.float64),
        rho=jnp.asarray(arrays["input_rho"], dtype=jnp.float64),
    )
    expected = {field: arrays[f"output_{field}"] for field in OUTPUT_FIELDS + SW_EXTRA_FIELDS}
    return state, expected


def load_lw_fixture_state(sample: Path = LW_SAMPLE) -> tuple[RRTMGLWColumnState, dict[str, np.ndarray]]:
    """Builds the JAX LW state and NumPy expected outputs."""

    arrays = _arrays(sample)
    state = RRTMGLWColumnState(
        T=jnp.asarray(arrays["input_T"], dtype=jnp.float64),
        p=jnp.asarray(arrays["input_p"], dtype=jnp.float64),
        qv=jnp.asarray(arrays["input_qv"], dtype=jnp.float64),
        qc=jnp.asarray(arrays["input_qc"], dtype=jnp.float64),
        qi=jnp.asarray(arrays["input_qi"], dtype=jnp.float64),
        qs=jnp.asarray(arrays["input_qs"], dtype=jnp.float64),
        qg=jnp.asarray(arrays["input_qg"], dtype=jnp.float64),
        cloud_fraction=jnp.asarray(arrays["input_cloud_fraction"], dtype=jnp.float64),
        surface_temperature=jnp.asarray(arrays["input_surface_temperature"], dtype=jnp.float64),
        surface_emissivity=jnp.asarray(arrays["input_surface_emissivity"], dtype=jnp.float64),
        dz=jnp.asarray(arrays["input_dz"], dtype=jnp.float64),
        rho=jnp.asarray(arrays["input_rho"], dtype=jnp.float64),
    )
    expected = {field: arrays[f"output_{field}"] for field in OUTPUT_FIELDS + LW_EXTRA_FIELDS}
    return state, expected


def _compare(candidate: Any, expected: dict[str, np.ndarray], tolerances: dict[str, tuple[float, float]], fields: tuple[str, ...], fixture_id: str) -> dict[str, Any]:
    """Compares one candidate result against the fixture outputs."""

    per_abs: dict[str, float] = {}
    per_rel: dict[str, float] = {}
    pass_fields: dict[str, bool] = {}
    for field in fields:
        cand = np.asarray(getattr(candidate, field), dtype=np.float64)
        ref = expected[field]
        diff = np.abs(cand - ref)
        abs_tol, rel_tol = tolerances[field]
        allowed = abs_tol + rel_tol * np.abs(ref)
        per_abs[field] = float(np.max(diff))
        per_rel[field] = float(np.max(diff / (np.abs(ref) + np.finfo(np.float64).eps)))
        pass_fields[field] = bool(np.all(diff <= allowed))
    first = next(iter(expected.values()))
    return {
        "fixture_id": fixture_id,
        "scenarios_tested": int(first.shape[0]),
        "per_field_max_abs_err": per_abs,
        "per_field_max_rel_err": per_rel,
        "tolerances_met": bool(all(pass_fields.values())),
        "field_pass": pass_fields,
        "pass": bool(all(pass_fields.values())),
    }


def run_tier1_sw(out: Path = SW_ARTIFACT) -> dict[str, Any]:
    """Writes the required RRTMG-SW tier-1 parity proof JSON."""

    state, expected = load_sw_fixture_state()
    result = solve_rrtmg_sw_column(state, debug=False)
    record = _compare(result, expected, _tolerance_map(_load_manifest(SW_MANIFEST)), OUTPUT_FIELDS + SW_EXTRA_FIELDS, SW_FIXTURE_ID)
    if _should_write_proof(out, SW_ARTIFACT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def run_tier1_lw(out: Path = LW_ARTIFACT) -> dict[str, Any]:
    """Writes the required RRTMG-LW tier-1 parity proof JSON."""

    state, expected = load_lw_fixture_state()
    result = solve_rrtmg_lw_column(state, debug=False)
    record = _compare(result, expected, _tolerance_map(_load_manifest(LW_MANIFEST)), OUTPUT_FIELDS + LW_EXTRA_FIELDS, LW_FIXTURE_ID)
    if _should_write_proof(out, LW_ARTIFACT):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
