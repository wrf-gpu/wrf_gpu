"""Tier-1 parity engine for the M5 Thompson column fixture.

Two paths:
  * the historical analytic-column fixture (``run_tier1`` / ``compare_against_fixture``),
    retained as a fast regression of the source/sink subset; and
  * the REAL WRF-oracle parity harness (``run_oracle_parity`` /
    ``compare_against_oracle``), which validates the full ``mp_gt_driver`` column
    (sedimentation + precip) against the frozen Phase-B savepoint schema
    (``mp_gt_driver_pre`` -> ``mp_gt_driver_post``).  The oracle path is the B1
    gate; it runs the instant the WRF-oracle factory populates
    ``/mnt/data/wrf_gpu2/physics_oracle/microphysics/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import yaml

from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    density_from_pressure_temperature,
    step_thompson_column,
    step_thompson_column_with_precip,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ID = "analytic-thompson-column-v1"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
ARTIFACT = ROOT / "artifacts" / "m5" / "tier1_thompson_parity.json"
OUTPUT_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T")

# --- Real WRF oracle harness (Phase-B B1 gate) ---------------------------------
ORACLE_DIR = Path("/mnt/data/wrf_gpu2/physics_oracle/microphysics")
ORACLE_ARTIFACT = ROOT / "proofs" / "b1" / "oracle_parity.json"
P0_PA = 100000.0
RD_CP = 287.0 / 1004.0
# WRF oracle dumps water-vapour mixing ratio plus the six hydrometeors and two
# number concentrations (module_microphysics_driver.F:1334-1452).  These are the
# fields we validate the JAX kernel against at the mp_gt_driver boundary.
ORACLE_MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
ORACLE_NUMBER_FIELDS = ("ni", "nr")
# Map WRF oracle var-name -> our ThompsonColumnState/State attribute.
_ORACLE_TO_KERNEL = {
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
    "ni": "Ni", "nr": "Nr",
}


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


# ---------------------------------------------------------------------------
# Real WRF-oracle parity harness (B1 gate)
# ---------------------------------------------------------------------------


def _temperature_from_oracle(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Recover temperature (K) from the oracle's theta + Exner (or theta + p)."""

    theta = np.asarray(arrays["th"], dtype=np.float64)
    if "pii" in arrays:
        exner = np.asarray(arrays["pii"], dtype=np.float64)
    else:
        exner = (np.asarray(arrays["p"], dtype=np.float64) / P0_PA) ** RD_CP
    return theta * exner


def _columns_from_oracle(arrays: dict[str, np.ndarray]) -> ThompsonColumnState:
    """Build a JAX ThompsonColumnState from a WRF ``mp_gt_driver_pre`` savepoint.

    The oracle dumps WRF (k, j, i) or (i, k, j) arrays; we flatten to
    (n_columns, n_levels) with vertical last, which is the kernel convention.
    """

    def to_cols(name: str) -> jnp.ndarray:
        a = np.asarray(arrays[name], dtype=np.float64)
        # Heuristic: vertical is the axis matching the dz8w/p level count.  WRF
        # oracle 3-D arrays are (k, j, i); move k to last and flatten (j*i, k).
        a = np.moveaxis(a, 0, -1)
        return jnp.asarray(a.reshape(-1, a.shape[-1]))

    T = jnp.asarray(_temperature_from_oracle(arrays).__array__())
    T = jnp.moveaxis(T, 0, -1).reshape(-1, T.shape[0])
    p = to_cols("p")
    qv = to_cols("qv")
    rho = density_from_pressure_temperature(p, T, qv)
    return ThompsonColumnState(
        qv=qv,
        qc=to_cols("qc"),
        qr=to_cols("qr"),
        qi=to_cols("qi"),
        qs=to_cols("qs"),
        qg=to_cols("qg"),
        Ni=to_cols("ni"),
        Nr=to_cols("nr"),
        T=T,
        p=p,
        rho=rho,
        dz=to_cols("dz8w") if "dz8w" in arrays else jnp.full_like(qv, 250.0),
        w=to_cols("w") if "w" in arrays else jnp.zeros_like(qv),
    )


def compare_against_oracle(pre_arrays: dict[str, np.ndarray], post_arrays: dict[str, np.ndarray], dt: float) -> dict[str, Any]:
    """Run the full Thompson kernel on the WRF pre-state, compare to WRF post.

    Applies the frozen Phase-B transcription tolerance ladder per field and the
    inactive-physical rule (a zero scheme delta is acceptable where the pre-state
    condensate < 1e-8 kg/kg AND qv < 1e-6 kg/kg).  Reports per-field max abs/rel
    error on MOIST columns and the step water-mass + precip closure.
    """

    from gpuwrf.validation.phase_b_savepoint import phase_b_tolerance, activation_floor_for

    column = _columns_from_oracle(pre_arrays)
    out, precip = step_thompson_column_with_precip(column, float(dt))

    cond_floor = activation_floor_for("microphysics_condensate_kg_kg")
    vap_floor = activation_floor_for("microphysics_vapour_kg_kg")
    pre_cond = sum(np.asarray(getattr(column, _ORACLE_TO_KERNEL[f]), dtype=np.float64) for f in ("qc", "qr", "qi", "qs", "qg"))
    pre_qv = np.asarray(column.qv, dtype=np.float64)
    moist_mask = (pre_cond >= cond_floor) | (pre_qv >= vap_floor)

    per_field: dict[str, Any] = {}
    overall_pass = True
    for oracle_name, kernel_name in _ORACLE_TO_KERNEL.items():
        cand = np.asarray(getattr(out, kernel_name), dtype=np.float64)
        ref = np.moveaxis(np.asarray(post_arrays[oracle_name], dtype=np.float64), 0, -1).reshape(cand.shape)
        band = phase_b_tolerance({"qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg", "ni": "Ni", "nr": "Nr"}[oracle_name])
        diff = np.abs(cand - ref)
        allowed = band.transcription_abs + band.transcription_rel * np.abs(ref)
        # Only enforce on moist columns; inactive (dry) columns excluded.
        mask = np.broadcast_to(moist_mask, diff.shape)
        viol = diff[mask] > allowed[mask]
        field_pass = bool(not np.any(viol))
        overall_pass = overall_pass and field_pass
        per_field[oracle_name] = {
            "max_abs_err": float(np.max(diff[mask])) if np.any(mask) else 0.0,
            "max_rel_err": float(np.max((diff / (np.abs(ref) + 1e-30))[mask])) if np.any(mask) else 0.0,
            "abs_tol": band.transcription_abs,
            "rel_tol": band.transcription_rel,
            "enforced_cells": int(np.count_nonzero(mask)),
            "pass": field_pass,
        }

    # Water conservation: column water mass change + surface precip vs WRF.
    rho = np.asarray(column.rho, dtype=np.float64)
    dz = np.asarray(column.dz, dtype=np.float64)
    qtot_in = sum(np.asarray(getattr(column, _ORACLE_TO_KERNEL[f]), dtype=np.float64) for f in ORACLE_MASS_FIELDS)
    qtot_out = sum(np.asarray(getattr(out, _ORACLE_TO_KERNEL[f]), dtype=np.float64) for f in ORACLE_MASS_FIELDS)
    mass_in = np.sum(qtot_in * rho * dz, axis=-1)
    mass_out = np.sum(qtot_out * rho * dz, axis=-1)
    precip_total = sum(np.asarray(v, dtype=np.float64) for v in precip.values())
    closure = np.abs((mass_out - mass_in) + precip_total)
    closure_rel = float(np.max(closure / np.maximum(mass_in, 1e-30)))

    return {
        "boundary": "mp_gt_driver_pre -> mp_gt_driver_post (full driver incl. sedimentation)",
        "dt_s": float(dt),
        "n_columns": int(column.qv.shape[0]),
        "n_cells": int(moist_mask.size),
        "moist_cells": int(np.count_nonzero(moist_mask)),
        "per_field": per_field,
        "water_closure_max_rel_residual": closure_rel,
        "water_closure_pass": bool(closure_rel < 1e-3),
        "pass": bool(overall_pass and closure_rel < 1e-3),
    }


def _find_oracle_pair(oracle_dir: Path) -> tuple[Path, Path] | None:
    """Find a matching (pre, post) Thompson savepoint pair, or None if absent."""

    if not oracle_dir.exists():
        return None
    pres = sorted(oracle_dir.glob("*mp_gt_driver_pre*.h5")) + sorted(oracle_dir.glob("*thompson*in*.h5"))
    posts = sorted(oracle_dir.glob("*mp_gt_driver_post*.h5")) + sorted(oracle_dir.glob("*thompson*out*.h5"))
    if pres and posts:
        return pres[0], posts[0]
    return None


def run_oracle_parity(oracle_dir: Path = ORACLE_DIR, out: Path = ORACLE_ARTIFACT) -> dict[str, Any]:
    """Validate the full Thompson kernel against the WRF oracle, if available.

    Writes a proof JSON.  When no savepoints are present yet (factory still
    populating), records ``status='PENDING-ORACLE'`` so the gate is honest about
    missing evidence rather than silently passing.
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    pair = _find_oracle_pair(oracle_dir)
    if pair is None:
        record = {
            "status": "PENDING-ORACLE",
            "oracle_dir": str(oracle_dir),
            "reason": "no mp_gt_driver_pre/post Thompson savepoints found yet (WRF-oracle factory populating)",
            "pass": None,
        }
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    from gpuwrf.validation.phase_b_savepoint import load_phase_b_savepoint

    pre_path, post_path = pair
    pre = load_phase_b_savepoint(pre_path)
    post = load_phase_b_savepoint(post_path)
    dt = float(getattr(pre.metadata, "dt_seconds", 0.0)) or float(getattr(post.metadata, "dt_seconds", 0.0))
    record = compare_against_oracle(pre.arrays, post.arrays, dt)
    record["status"] = "ORACLE-VALIDATED"
    record["pre_savepoint"] = str(pre_path)
    record["post_savepoint"] = str(post_path)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
