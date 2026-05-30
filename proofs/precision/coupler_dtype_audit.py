#!/usr/bin/env python
"""Coupler dtype audit (Sprint coupler-fp64 FIX #1, GPT P0-1 proof object).

Proves that under ``force_fp64`` every prognostic / tendency field emerging from
each physics adapter (Thompson, surface, MYNN, RRTMG) is float64 -- i.e. the
fp32-defeat bug (adapters casting back to the frozen fp32 ADR-007 perf matrix
*inside* the timestep) is fixed.

BEFORE column: the dtype the pre-fix code wrote = the frozen ``_field_dtype``
matrix (fp32 for theta/qv/u/v/hydrometeors/qke). AFTER column: the dtype actually
produced by the fixed adapters when run on an fp64-upcast (force_fp64) state.

Also audits the held radiative theta tendency (RTHRATEN) emitted by
``rrtmg_theta_tendency`` and the Thompson water-budget side channel.

Run: PYTHONPATH=<worktree>/src python proofs/precision/coupler_dtype_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

from gpuwrf.contracts.precision import STATE_FIELD_ORDER
from gpuwrf.coupling.physics_couplers import (
    _field_dtype,
    mynn_adapter,
    rrtmg_theta_tendency,
    surface_adapter,
    thompson_adapter,
    thompson_adapter_with_tendencies,
)
from gpuwrf.runtime.operational_mode import _enforce_operational_precision

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "m6_run_dummy_coupled.py"
_SPEC = importlib.util.spec_from_file_location("m6_run_dummy_coupled", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


# Fields each adapter is responsible for writing (its output contract).
THOMPSON_FIELDS = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")
SURFACE_FIELDS = ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv")
MYNN_FIELDS = ("u", "v", "w", "theta", "qv", "qke")
RRTMG_FIELDS = ("theta",)


def _dtype_name(x) -> str:
    return str(jnp.asarray(x).dtype)


def _before_matrix(fields) -> dict:
    """The dtype the PRE-FIX adapter wrote = frozen _field_dtype perf matrix."""

    return {f: str(jnp.dtype(_field_dtype(f))) for f in fields}


def _after_state(state, fields) -> dict:
    return {f: _dtype_name(getattr(state, f)) for f in fields}


def main() -> None:
    grid = _MOD.make_dummy_grid(8, 8, 16)
    base = _MOD.make_initial_state(grid)
    dt = 10.0

    # force_fp64: upcast the whole carry exactly as the operational path does.
    fp64_state = _enforce_operational_precision(base, force_fp64=True)
    assert _dtype_name(fp64_state.theta) == "float64", "force_fp64 upcast failed"

    report: dict = {
        "case": "m6 dummy coupled grid (8x8x16), force_fp64=True",
        "purpose": "FIX #1 / GPT P0-1: physics adapters must NOT downcast to fp32 inside the force_fp64 timestep.",
        "input_state_dtypes_fp64": {f: _dtype_name(getattr(fp64_state, f)) for f in STATE_FIELD_ORDER},
        "adapters": {},
    }

    # --- Thompson ---
    th_out, th_tend = thompson_adapter_with_tendencies(fp64_state, dt)
    th_tend_dtypes = {
        f: _dtype_name(getattr(th_tend, f)) for f in ("qv", "qc", "qr", "qi", "qs", "qg")
    }
    report["adapters"]["thompson"] = {
        "fields_before_fix_frozen_matrix": _before_matrix(THOMPSON_FIELDS),
        "fields_after_fix": _after_state(th_out, THOMPSON_FIELDS),
        "tendency_side_channel_after_fix": th_tend_dtypes,
        "accumulators_after_fix": _after_state(th_out, ("rain_acc", "snow_acc", "graupel_acc", "ice_acc")),
    }

    # --- surface (flux handles) ---
    sf_out = surface_adapter(fp64_state, dt)
    report["adapters"]["surface"] = {
        "fields_before_fix_frozen_matrix": _before_matrix(SURFACE_FIELDS),
        "fields_after_fix": _after_state(sf_out, SURFACE_FIELDS),
    }

    # --- MYNN (runs after surface so it reads the flux handles) ---
    mynn_out = mynn_adapter(sf_out, dt, grid)
    report["adapters"]["mynn"] = {
        "fields_before_fix_frozen_matrix": _before_matrix(MYNN_FIELDS),
        "fields_after_fix": _after_state(mynn_out, MYNN_FIELDS),
    }

    # --- RRTMG (held-rate primitive + legacy theta-write) ---
    rthraten = rrtmg_theta_tendency(mynn_out, grid)
    report["adapters"]["rrtmg"] = {
        "fields_before_fix_frozen_matrix": _before_matrix(RRTMG_FIELDS),
        "held_rthraten_dtype_after_fix": _dtype_name(rthraten),
        "note": "RRTMG band optics are intrinsically fp32 (WRF uses r4 there too); the heating-rate add and the theta write run in fp64 when the carry is fp64.",
    }

    # --- verdict ---
    def _all_fp64(d) -> bool:
        return all(v == "float64" for v in d.values())

    fp64_checks = {
        "thompson_fields": _all_fp64(report["adapters"]["thompson"]["fields_after_fix"]),
        "thompson_tendencies": _all_fp64(th_tend_dtypes),
        "thompson_accumulators": _all_fp64(report["adapters"]["thompson"]["accumulators_after_fix"]),
        "surface_fields": _all_fp64(report["adapters"]["surface"]["fields_after_fix"]),
        "mynn_fields": _all_fp64(report["adapters"]["mynn"]["fields_after_fix"]),
        "rrtmg_held_rthraten": _dtype_name(rthraten) == "float64",
    }
    report["fp64_after_fix_checks"] = fp64_checks
    report["all_fp64_after_fix"] = all(fp64_checks.values())

    out_path = ROOT / "proofs" / "precision" / "coupler_dtype_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nALL_FP64_AFTER_FIX:", report["all_fp64_after_fix"])
    if not report["all_fp64_after_fix"]:
        raise SystemExit("DTYPE AUDIT FAILED: some adapter output is not fp64 under force_fp64")


if __name__ == "__main__":
    main()
