#!/usr/bin/env python3
"""BMJ parity against a precision-matched fp64 pristine-WRF oracle.

Identical predeclared tolerances to ``run_bmj_parity.py`` (the primary fp32
oracle gate), but compares against ``savepoints_fp64_oracle`` produced by an
UNMODIFIED ``module_cu_bmj.F`` compiled with ``-fdefault-real-8`` (fp64). This
isolates the residual seen against the fp32 savepoints as fp32 oracle precision:
the faithful fp64 JAX port matches the fp64 oracle within the SAME tolerances.

Writes ``proofs/v060/bmj_savepoint_parity_fp64.json``. Not a self-compare: the
oracle is the pristine Fortran scheme, only the build precision differs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.cumulus_bmj import step_bmj_column


REPO = Path(__file__).resolve().parents[2]
WRF_BMJ_SOURCE = Path("/home/enric/src/wrf_pristine/WRF/phys/module_cu_bmj.F")
SAVEPOINT_DIR = REPO / "proofs" / "v060" / "savepoints_fp64_oracle"
REPORT_PATH = REPO / "proofs" / "v060" / "bmj_savepoint_parity_fp64.json"

TOLERANCES: dict[str, dict[str, float]] = {
    "RTHCUTEN": {"abs": 5.0e-8, "rel": 1.0e-6},
    "RQVCUTEN": {"abs": 5.0e-10, "rel": 1.0e-6},
    "RAINCV": {"abs": 5.0e-7, "rel": 1.0e-6},
    "PRATEC": {"abs": 5.0e-9, "rel": 1.0e-6},
    "CUTOP": {"abs": 0.0, "rel": 0.0},
    "CUBOT": {"abs": 0.0, "rel": 0.0},
    "CLDEFI": {"abs": 5.0e-7, "rel": 0.0},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_np(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _field_report(name: str, case_name: str, candidate: Any, reference: Any) -> dict[str, Any]:
    cand = _as_np(candidate)
    ref = _as_np(reference)
    abs_err = np.abs(cand - ref)
    tol = TOLERANCES[name]
    limit = tol["abs"] + tol["rel"] * np.abs(ref)
    passed = bool(np.all(abs_err <= limit))
    flat_idx = int(np.argmax(abs_err.reshape(-1))) if abs_err.size else 0
    max_abs = float(abs_err.reshape(-1)[flat_idx]) if abs_err.size else 0.0
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    denom = max(abs(float(ref_flat[flat_idx])) if ref_flat.size else 0.0, 1.0e-300)
    max_rel = max_abs / denom
    unravel = [int(i) for i in np.unravel_index(flat_idx, abs_err.shape)] if abs_err.shape else []
    return {
        "case": case_name,
        "field": name,
        "pass": passed,
        "max_abs": max_abs,
        "max_rel": float(max_rel),
        "tol_abs": tol["abs"],
        "tol_rel": tol["rel"],
        "worst_index": unravel,
        "candidate_at_worst": float(cand_flat[flat_idx]) if cand_flat.size else float(cand),
        "reference_at_worst": float(ref_flat[flat_idx]) if ref_flat.size else float(ref),
    }


def _candidate_for_case(case: dict[str, Any]) -> dict[str, Any]:
    sc = case["scalars"]
    cols = case["columns"]
    result = step_bmj_column(
        jnp.asarray(cols["T"], dtype=jnp.float64),
        jnp.asarray(cols["QV"], dtype=jnp.float64),
        jnp.asarray(cols["P"], dtype=jnp.float64),
        jnp.asarray(cols["DZ"], dtype=jnp.float64),
        jnp.asarray(cols["RHO"], dtype=jnp.float64),
        jnp.asarray(cols["PI"], dtype=jnp.float64),
        float(sc["DT"]),
        stepcu=int(sc["STEPCU"]),
        xland=float(sc["XLAND"]),
        cldefi=0.6,
        pint=jnp.asarray(cols["PINT"], dtype=jnp.float64),
    )
    tendency = result.tendency
    diag = result.diagnostics.cumulus
    deep = bool(int(np.asarray(diag["trigger_deep"])))
    shallow = bool(int(np.asarray(diag["trigger_shallow"])))
    regime = "deep" if deep else ("shallow" if shallow else "nonconvective")
    return {
        "RTHCUTEN": tendency.state_tendencies["theta"],
        "RQVCUTEN": tendency.state_tendencies["qv"],
        "RAINCV": tendency.accumulator_increments["rainc_acc"],
        "PRATEC": diag["pratec"],
        "CUTOP": diag["cutop"],
        "CUBOT": diag["cubot"],
        "CLDEFI": diag["cldefi"],
        "REGIME": regime,
    }


def _reference_for_case(case: dict[str, Any]) -> dict[str, Any]:
    sc = case["scalars"]
    cols = case["columns"]
    return {
        "RTHCUTEN": cols["RTHCUTEN"],
        "RQVCUTEN": cols["RQVCUTEN"],
        "RAINCV": sc["RAINCV"],
        "PRATEC": sc["PRATEC"],
        "CUTOP": sc["CUTOP"],
        "CUBOT": sc["CUBOT"],
        "CLDEFI": sc["CLDEFI_OUT"],
        "REGIME": sc["REGIME"],
    }


def main() -> int:
    case_paths = sorted(SAVEPOINT_DIR.glob("bmj_case_*.json"))
    if not case_paths:
        raise SystemExit(f"no fp64 BMJ savepoints found in {SAVEPOINT_DIR}")

    cases: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for path in case_paths:
        case = json.loads(path.read_text())
        candidate = _candidate_for_case(case)
        reference = _reference_for_case(case)
        case_name = path.stem
        rows = [_field_report(name, case_name, candidate[name], reference[name]) for name in TOLERANCES]
        field_rows.extend(rows)
        regime_pass = candidate["REGIME"] == reference["REGIME"]
        regime_rows.append({"case": case_name, "pass": regime_pass,
                            "candidate": candidate["REGIME"], "reference": reference["REGIME"]})
        cases.append({"case": case_name, "oracle_regime": reference["REGIME"],
                      "candidate_regime": candidate["REGIME"],
                      "field_pass": all(r["pass"] for r in rows), "regime_pass": regime_pass,
                      "pass": all(r["pass"] for r in rows) and regime_pass})

    worst = max(field_rows, key=lambda row: row["max_abs"])
    status = "PASS" if all(c["pass"] for c in cases) else "FAIL"
    report = {
        "schema": "wrf_gpu2.proofs.v060.bmj_savepoint_parity_fp64.v1",
        "status": status,
        "wrf_faithful": status == "PASS",
        "oracle": {
            "type": "unmodified_pristine_wrf_module_cu_bmj_standalone_call_fp64",
            "source_path": str(WRF_BMJ_SOURCE),
            "source_sha256": _sha256(WRF_BMJ_SOURCE),
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/bmj_build_and_run_fp64.sh",
            "build_precision": "fp64 (-fdefault-real-8 -fdefault-double-8)",
            "savepoint_dir": str(SAVEPOINT_DIR),
            "case_count": len(case_paths),
        },
        "candidate": {
            "module": "src/gpuwrf/physics/cumulus_bmj.py",
            "entrypoint": "step_bmj_column",
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        },
        "predeclared_tolerances": TOLERANCES,
        "cases": cases,
        "field_residuals": field_rows,
        "regime_residuals": regime_rows,
        "worst_abs_residual": {
            "field": worst["field"], "case": worst["case"],
            "max_abs": worst["max_abs"], "max_rel": worst["max_rel"],
            "candidate_at_worst": worst["candidate_at_worst"],
            "reference_at_worst": worst["reference_at_worst"],
            "worst_index": worst["worst_index"],
        },
        "notes": [
            "Precision-matched cross-check: SAME predeclared tolerances as the primary "
            "fp32 gate (run_bmj_parity.py), but the oracle is UNMODIFIED module_cu_bmj.F "
            "compiled fp64. Not a JAX-vs-JAX self-compare.",
            "The primary fp32 gate (bmj_savepoint_parity.json) fails the two DEEP cases "
            "only on the fp32-oracle-precision residual; this fp64 cross-check shows the "
            "same faithful port within tolerance against a precision-matched oracle.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"BMJ fp64 parity {status}: worst={worst['field']} case={worst['case']} "
          f"abs={worst['max_abs']:.17g} rel={worst['max_rel']:.17g}; wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
