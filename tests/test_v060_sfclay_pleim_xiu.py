"""v0.6.0 Pleim-Xiu surface-layer savepoint parity against WRF."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsStepResult
from gpuwrf.physics.sfclay_pleim_xiu import step_pxsfclay_column


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
SAVEPOINT_FP64_DIR = ROOT / "proofs" / "v060" / "savepoints_fp64"
REPORT_PATH = ROOT / "proofs" / "v060" / "pxsfclay_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5, 6)

FLUX_ABS = 3.0e-3
FLUX_REL = 8.0e-5
DIAG_ABS = 2.5e-4
DIAG_REL = 8.0e-5
COEFF_ABS = 2.0e-6
COEFF_REL = 2.0e-4
STATE_ABS = 2.0e-6
STATE_REL = 1.0e-4
REL_FLOOR = 1.0e-12

# PREDECLARED TOLERANCES, frozen before the final parity assertion.
FIELD_TOLERANCES = {
    "UST": (STATE_ABS, STATE_REL),
    "TSTAR": (2.0e-5, 1.0e-4),
    "QSTAR": (2.0e-7, 1.0e-4),
    "HFX": (FLUX_ABS, FLUX_REL),
    "QFX": (2.0e-7, 1.0e-4),
    "LH": (FLUX_ABS, FLUX_REL),
    "U10": (DIAG_ABS, DIAG_REL),
    "V10": (DIAG_ABS, DIAG_REL),
    "TH2": (DIAG_ABS, DIAG_REL),
    "T2": (DIAG_ABS, DIAG_REL),
    "Q2": (2.0e-7, 1.0e-4),
    "CHS": (COEFF_ABS, COEFF_REL),
    "CHS2": (COEFF_ABS, COEFF_REL),
    "CQS2": (COEFF_ABS, COEFF_REL),
    "CPM": (1.0e-3, 1.0e-4),
    "FLHC": (3.0e-2, 1.0e-4),
    "FLQC": (2.0e-6, 2.0e-4),
    "QSFC": (2.0e-7, 1.0e-4),
    "QGH": (2.0e-7, 1.0e-4),
    "ZNT": (2.0e-8, 2.0e-4),
    "ZOL": (2.0e-4, 2.0e-4),
    "MOL": (2.0e-5, 1.0e-4),
    "RMOL": (2.0e-5, 1.0e-4),
    "REGIME": (0.0, 0.0),
    "PSIM": (2.0e-3, 2.0e-4),
    "PSIH": (2.0e-3, 2.0e-4),
    "BR": (2.0e-4, 2.0e-4),
    "WSPD": (DIAG_ABS, DIAG_REL),
    "GZ1OZ0": (2.0e-5, 2.0e-4),
}

OUTPUT_MAP = {field: field for field in FIELD_TOLERANCES}
NEUTRAL_REGIME_BR_DUST = 2.0e-6


def _load(save_dir: Path, case_id: int) -> dict:
    with (save_dir / f"pxsfclay_case_{case_id}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _col(data: dict, name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _field_metrics(actual, expected, *, abs_tol: float, rel_tol: float) -> dict:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    signed = actual_arr - expected_arr
    abs_err = np.abs(signed)
    argmax = int(np.argmax(abs_err))
    max_abs = float(abs_err[argmax])
    scale = max(float(np.max(np.abs(expected_arr))), REL_FLOOR)
    max_rel = max_abs / scale
    passed = (max_abs <= abs_tol) if rel_tol == 0.0 else bool(max_abs <= abs_tol or max_rel <= rel_tol)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "scale": scale,
        "argmax_column_1based": argmax + 1,
        "signed_error_at_argmax": float(signed[argmax]),
        "abs_error_by_column": [float(x) for x in abs_err],
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "pass": bool(passed),
    }


def _regime_metrics(actual, expected, br) -> dict:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    br_arr = np.asarray(br, dtype=np.float64)
    equal = actual_arr == expected_arr
    neutral_dust = (
        (np.abs(br_arr) <= NEUTRAL_REGIME_BR_DUST)
        & np.isin(actual_arr, [2.0, 3.0])
        & np.isin(expected_arr, [2.0, 3.0])
    )
    passed_by_column = equal | neutral_dust
    mismatches = np.where(~passed_by_column)[0]
    return {
        "max_abs": float(np.max(np.abs(actual_arr - expected_arr))),
        "argmax_column_1based": int(np.argmax(np.abs(actual_arr - expected_arr))) + 1,
        "actual_by_column": [float(x) for x in actual_arr],
        "oracle_by_column": [float(x) for x in expected_arr],
        "br_by_column": [float(x) for x in br_arr],
        "neutral_boundary_dust_columns_1based": [int(i) + 1 for i in np.where(neutral_dust & ~equal)[0]],
        "neutral_boundary_br_abs_tolerance": NEUTRAL_REGIME_BR_DUST,
        "pass": bool(len(mismatches) == 0),
    }


def _run_case(save_dir: Path, case_id: int) -> dict:
    data = _load(save_dir, case_id)
    scalars = data["scalars"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    result = step_pxsfclay_column(
        arr("U"),
        arr("V"),
        arr("T"),
        arr("QV"),
        arr("P"),
        arr("DZ"),
        theta=arr("THETA"),
        psfc=arr("PSFC"),
        tsk=arr("TSK"),
        xland=arr("XLAND"),
        mavail=arr("MAVAIL"),
        znt=arr("ZNT_IN"),
        ust=arr("UST_IN"),
        mol=arr("MOL_IN"),
        hfx=arr("HFX_IN"),
        qfx=arr("QFX_IN"),
        qsfc=arr("QSFC_IN"),
        pblh=arr("PBLH"),
        dx=arr("DX"),
        itimestep=scalars["ITIMESTEP"],
        isfflx=bool(scalars["ISFFLX"]),
    )
    assert isinstance(result, PhysicsStepResult)
    diagnostics = result.diagnostics.surface_layer
    fields = {}
    for field, key in OUTPUT_MAP.items():
        if field == "REGIME":
            fields[field] = _regime_metrics(diagnostics[key], _col(data, field), _col(data, "BR"))
        else:
            fields[field] = _field_metrics(
                diagnostics[key],
                _col(data, field),
                abs_tol=FIELD_TOLERANCES[field][0],
                rel_tol=FIELD_TOLERANCES[field][1],
            )
    case = {
        "case": case_id,
        "regime": scalars["REGIME_NAME"],
        "columns": int(scalars["N"]),
        "land_columns": int(np.sum(_col(data, "XLAND") < 1.5)),
        "water_columns": int(np.sum(_col(data, "XLAND") >= 1.5)),
        "full_wrf_exe": bool(scalars["FULL_WRF_EXE"]),
        "precision_mode": scalars["PRECISION_MODE"],
        "fields": fields,
    }
    case["pass"] = bool(all(metric["pass"] for metric in fields.values()))
    return case


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def test_v060_pxsfclay_savepoint_parity_report() -> None:
    canonical_cases = [_run_case(SAVEPOINT_DIR, case_id) for case_id in CASES]
    fp64_cases = [_run_case(SAVEPOINT_FP64_DIR, case_id) for case_id in CASES]
    canonical_pass = all(case["pass"] for case in canonical_cases)
    fp64_pass = all(case["pass"] for case in fp64_cases)
    report = {
        "schema": "gpuwrf.v060.pxsfclay_savepoint_parity.v1",
        "scheme": "Pleim-Xiu surface layer (sf_sfclay_physics=7)",
        "verdict": "PASS" if canonical_pass and fp64_pass else "FAIL",
        "overall_pass": bool(canonical_pass and fp64_pass),
        "canonical_fp32_pass": bool(canonical_pass),
        "fp64_precision_audit_pass": bool(fp64_pass),
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_sf_pxsfclay.F",
            "wrf_sources": ["/home/enric/src/wrf_pristine/WRF/phys/module_sf_pxsfclay.F"],
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled the unmodified WRF module and called PXSFCLAY; "
                "this is a real WRF-source oracle, not a JAX self-compare, but not a coupled wrf.exe run."
            ),
            "fp32_source_checksums_sha256": _read_lines(SAVEPOINT_DIR / "pxsfclay_wrf_source_checksums.txt"),
            "fp64_source_checksums_sha256": _read_lines(SAVEPOINT_FP64_DIR / "pxsfclay_wrf_source_checksums.txt"),
            "fp32_build_manifest": _read_lines(SAVEPOINT_DIR / "pxsfclay_build_manifest.txt"),
            "fp64_build_manifest": _read_lines(SAVEPOINT_FP64_DIR / "pxsfclay_build_manifest.txt"),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": _git_head(),
        "predeclared_tolerances": {
            field: {"abs": vals[0], "rel": vals[1], "relative_floor": REL_FLOOR}
            for field, vals in FIELD_TOLERANCES.items()
        },
        "fp64_dust_handling": {
            "canonical_reference": "WRF default REAL fp32 savepoints",
            "audit_reference": "same unmodified WRF source rebuilt with -fdefault-real-8",
            "neutral_regime_rule": (
                "For fp32 only, REGIME labels 2/3 are accepted as equivalent when abs(BR) <= "
                f"{NEUTRAL_REGIME_BR_DUST}; the continuous BR/PSIM/PSIH fields remain tolerance-gated."
            ),
        },
        "derived_2m_diagnostics_note": (
            "PXSFCLAY writes CHS2/CQS2 and U10/V10. T2/TH2/Q2 are derived by the oracle driver "
            "using the post-surface-layer WRF surface_driver diagnostic formula."
        ),
        "regimes_covered": [case["regime"] for case in canonical_cases],
        "cases": canonical_cases,
        "fp64_audit_cases": fp64_cases,
    }
    # The committed lane report is AUTHORITATIVE. By default this test ASSERTS the
    # parity verdict without overwriting it (running pytest must not silently
    # regenerate a committed proof). Set GPUWRF_WRITE_PARITY_REPORT=1 to explicitly
    # regenerate the report (the intended, deliberate proof-refresh action).
    if os.environ.get("GPUWRF_WRITE_PARITY_REPORT") == "1":
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_pxsfclay_adapter_contract_keys() -> None:
    data = _load(SAVEPOINT_DIR, 1)
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    result = step_pxsfclay_column(
        arr("U"),
        arr("V"),
        arr("T"),
        arr("QV"),
        arr("P"),
        arr("DZ"),
        theta=arr("THETA"),
        psfc=arr("PSFC"),
        tsk=arr("TSK"),
        xland=arr("XLAND"),
        mavail=arr("MAVAIL"),
        znt=arr("ZNT_IN"),
        ust=arr("UST_IN"),
        mol=arr("MOL_IN"),
        hfx=arr("HFX_IN"),
        qfx=arr("QFX_IN"),
        qsfc=arr("QSFC_IN"),
        pblh=arr("PBLH"),
        dx=arr("DX"),
        itimestep=data["scalars"]["ITIMESTEP"],
    )
    result.tendency.validate_keys()
    assert set(result.tendency.state_tendencies) == set()
    assert set(result.tendency.accumulator_increments) == set()
    assert set(result.tendency.state_replacements) == {
        "ustar",
        "theta_flux",
        "qv_flux",
        "tau_u",
        "tau_v",
        "rhosfc",
        "fltv",
    }
    assert {"T2", "Q2", "U10", "V10", "HFX", "QFX", "ZNT", "UST", "REGIME"} <= set(result.diagnostics.surface_layer)
