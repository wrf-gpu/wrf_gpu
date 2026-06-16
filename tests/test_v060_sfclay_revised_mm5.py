"""v0.6.0 revised-MM5 surface-layer savepoint parity against WRF."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsStepResult
from gpuwrf.physics.sfclay_revised_mm5 import step_sfclay_revised_mm5_column


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "sfclayrev1_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5, 6)

FLUX_ABS = 3.0e-3
FLUX_REL = 8.0e-5
DIAG_ABS = 2.5e-4
DIAG_REL = 8.0e-5
COEFF_ABS = 2.0e-6
COEFF_REL = 2.0e-4
STATE_ABS = 2.0e-7
STATE_REL = 1.0e-4
REL_FLOOR = 1.0e-12

FIELD_TOLERANCES = {
    "UST": (STATE_ABS, STATE_REL),
    "TSTAR": (DIAG_ABS, DIAG_REL),
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
    "FLHC": (3.0e-2, 1.0e-4),
    "FLQC": (2.0e-6, 2.0e-4),
    "CK": (COEFF_ABS, COEFF_REL),
    "CKA": (COEFF_ABS, COEFF_REL),
    "CD": (COEFF_ABS, COEFF_REL),
    "CDA": (COEFF_ABS, COEFF_REL),
    "QSFC": (2.0e-7, 1.0e-4),
    "QGH": (2.0e-7, 1.0e-4),
    "ZNT": (2.0e-8, 2.0e-4),
    "ZOL": (2.0e-5, 2.0e-4),
    "MOL": (DIAG_ABS, DIAG_REL),
    "RMOL": (2.0e-7, 1.0e-4),
    "REGIME": (0.0, 0.0),
    "PSIM": (2.0e-5, 2.0e-4),
    "PSIH": (2.0e-5, 2.0e-4),
    "FM": (2.0e-5, 2.0e-4),
    "FH": (2.0e-5, 2.0e-4),
    "BR": (2.0e-6, 2.0e-4),
    "WSPD": (DIAG_ABS, DIAG_REL),
    "GZ1OZ0": (2.0e-5, 2.0e-4),
}

OUTPUT_MAP = {
    "UST": ("diagnostics", "UST"),
    "TSTAR": ("diagnostics", "TSTAR"),
    "QSTAR": ("diagnostics", "QSTAR"),
    "HFX": ("diagnostics", "HFX"),
    "QFX": ("diagnostics", "QFX"),
    "LH": ("diagnostics", "LH"),
    "U10": ("diagnostics", "U10"),
    "V10": ("diagnostics", "V10"),
    "TH2": ("diagnostics", "TH2"),
    "T2": ("diagnostics", "T2"),
    "Q2": ("diagnostics", "Q2"),
    "CHS": ("diagnostics", "CHS"),
    "CHS2": ("diagnostics", "CHS2"),
    "CQS2": ("diagnostics", "CQS2"),
    "FLHC": ("diagnostics", "FLHC"),
    "FLQC": ("diagnostics", "FLQC"),
    "CK": ("diagnostics", "CK"),
    "CKA": ("diagnostics", "CKA"),
    "CD": ("diagnostics", "CD"),
    "CDA": ("diagnostics", "CDA"),
    "QSFC": ("diagnostics", "QSFC"),
    "QGH": ("diagnostics", "QGH"),
    "ZNT": ("diagnostics", "ZNT"),
    "ZOL": ("diagnostics", "ZOL"),
    "MOL": ("diagnostics", "MOL"),
    "RMOL": ("diagnostics", "RMOL"),
    "REGIME": ("diagnostics", "REGIME"),
    "PSIM": ("diagnostics", "PSIM"),
    "PSIH": ("diagnostics", "PSIH"),
    "FM": ("diagnostics", "FM"),
    "FH": ("diagnostics", "FH"),
    "BR": ("diagnostics", "BR"),
    "WSPD": ("diagnostics", "WSPD"),
    "GZ1OZ0": ("diagnostics", "GZ1OZ0"),
}


def _load(case_id: int) -> dict:
    with (SAVEPOINT_DIR / f"sfclayrev1_case_{case_id}.json").open() as fh:
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
        "pass": passed,
    }


def _run_case(case_id: int) -> dict:
    data = _load(case_id)
    c = data["columns"]
    scalars = data["scalars"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    result = step_sfclay_revised_mm5_column(
        arr("U"),
        arr("V"),
        arr("T"),
        arr("QV"),
        arr("P"),
        arr("DZ"),
        psfc=arr("PSFC"),
        tsk=arr("TSK"),
        xland=arr("XLAND"),
        lakemask=arr("LAKEMASK"),
        mavail=arr("MAVAIL"),
        znt=arr("ZNT_IN"),
        ust=arr("UST_IN"),
        mol=arr("MOL_IN"),
        hfx=arr("HFX_IN"),
        qfx=arr("QFX_IN"),
        qsfc=arr("QSFC_IN"),
        pblh=arr("PBLH"),
        dx=arr("DX"),
        water_depth=arr("WATER_DEPTH"),
        isfflx=bool(scalars["ISFFLX"]),
        shalwater_z0=bool(scalars["SHALWATER_Z0"]),
    )
    assert isinstance(result, PhysicsStepResult)
    diagnostics = result.diagnostics.surface_layer
    fields = {}
    for field, (_kind, key) in OUTPUT_MAP.items():
        fields[field] = _field_metrics(
            diagnostics[key],
            np.asarray(c[field], dtype=np.float64),
            abs_tol=FIELD_TOLERANCES[field][0],
            rel_tol=FIELD_TOLERANCES[field][1],
        )
    case = {
        "case": case_id,
        "regime": scalars["REGIME_NAME"],
        "columns": int(scalars["N"]),
        "land_columns": int(np.sum(_col(data, "XLAND") < 1.5)),
        "water_columns": int(np.sum(_col(data, "XLAND") >= 1.5)),
        "fields": fields,
    }
    case["pass"] = bool(all(metric["pass"] for metric in fields.values()))
    return case


def test_v060_sfclay_revised_mm5_savepoint_parity_report() -> None:
    cases = [_run_case(case_id) for case_id in CASES]
    checksum_path = SAVEPOINT_DIR / "sfclayrev1_wrf_source_checksums.txt"
    source_checksums = checksum_path.read_text(encoding="utf-8").splitlines()
    report = {
        "schema": "gpuwrf.v060.sfclayrev1_savepoint_parity.v1",
        "scheme": "revised-MM5 / Jimenez surface layer (sf_sfclay_physics=1)",
        "verdict": "PASS" if all(case["pass"] for case in cases) else "FAIL",
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_sf_sfclayrev.F and sf_sfclayrev.F90",
            "wrf_sources": [
                "/home/user/src/wrf_pristine/WRF/phys/module_sf_sfclayrev.F",
                "/home/user/src/wrf_pristine/WRF/phys/physics_mmm/sf_sfclayrev.F90",
                "/home/user/src/wrf_pristine/WRF/phys/ccpp_kind_types.f90",
            ],
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
            "full_wrf_exe": False,
            "source_checksums_sha256": source_checksums,
            "note": "Real WRF-module oracle, not a JAX self-compare; not a coupled wrf.exe run.",
        },
        "predeclared_tolerances": {
            field: {"abs": vals[0], "rel": vals[1], "relative_floor": REL_FLOOR}
            for field, vals in FIELD_TOLERANCES.items()
        },
        "regimes_covered": [case["regime"] for case in cases],
        "cases": cases,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_sfclay_revised_mm5_adapter_contract_keys() -> None:
    data = _load(1)
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    result = step_sfclay_revised_mm5_column(
        arr("U"),
        arr("V"),
        arr("T"),
        arr("QV"),
        arr("P"),
        arr("DZ"),
        psfc=arr("PSFC"),
        tsk=arr("TSK"),
        xland=arr("XLAND"),
        lakemask=arr("LAKEMASK"),
        mavail=arr("MAVAIL"),
        znt=arr("ZNT_IN"),
        ust=arr("UST_IN"),
        mol=arr("MOL_IN"),
        hfx=arr("HFX_IN"),
        qfx=arr("QFX_IN"),
        qsfc=arr("QSFC_IN"),
        pblh=arr("PBLH"),
        dx=arr("DX"),
        water_depth=arr("WATER_DEPTH"),
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
    assert {"T2", "Q2", "U10", "V10", "HFX", "QFX", "ZNT", "UST"} <= set(result.diagnostics.surface_layer)
