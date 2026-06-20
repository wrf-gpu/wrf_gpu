"""v0.6.0 Janjic Eta surface-layer savepoint parity against WRF MYJSFC."""

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
from gpuwrf.physics.sfclay_janjic import myjsfc_column, step_janjic_sfclay_column


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_FP64_DIR = ROOT / "proofs" / "v060" / "savepoints_fp64"
REPORT_PATH = ROOT / "proofs" / "v060" / "janjic_sfclay_savepoint_parity.json"
CASES = (1, 2, 3, 4, 5, 6)

# PREDECLARED fp64 savepoint tolerances. These are absolute roundoff-level
# gates with a relative backstop for large flux/pressure diagnostics.
ABS_TOL = 5.0e-10
REL_TOL = 5.0e-12
REL_FLOOR = 1.0e-30

FIELD_TOLERANCES = {
    "USTAR": (ABS_TOL, REL_TOL),
    "ZNT": (ABS_TOL, REL_TOL),
    "AKHS": (ABS_TOL, REL_TOL),
    "AKMS": (ABS_TOL, REL_TOL),
    "RMOL": (ABS_TOL, REL_TOL),
    "RIB": (ABS_TOL, REL_TOL),
    "CHS": (ABS_TOL, REL_TOL),
    "CHS2": (ABS_TOL, REL_TOL),
    "CQS2": (ABS_TOL, REL_TOL),
    "HFX": (ABS_TOL, REL_TOL),
    "QFX": (ABS_TOL, REL_TOL),
    "FLX_LH": (ABS_TOL, REL_TOL),
    "FLHC": (ABS_TOL, REL_TOL),
    "FLQC": (ABS_TOL, REL_TOL),
    "QGH": (ABS_TOL, REL_TOL),
    "CPM": (ABS_TOL, REL_TOL),
    "CT": (ABS_TOL, REL_TOL),
    "QSFC": (ABS_TOL, REL_TOL),
    "THZ0": (ABS_TOL, REL_TOL),
    "QZ0": (ABS_TOL, REL_TOL),
    "UZ0": (ABS_TOL, REL_TOL),
    "VZ0": (ABS_TOL, REL_TOL),
    "PBLH": (ABS_TOL, REL_TOL),
    "U10": (ABS_TOL, REL_TOL),
    "V10": (ABS_TOL, REL_TOL),
    "T02": (ABS_TOL, REL_TOL),
    "TH02": (ABS_TOL, REL_TOL),
    "TSHLTR": (ABS_TOL, REL_TOL),
    "TH10": (ABS_TOL, REL_TOL),
    "Q02": (ABS_TOL, REL_TOL),
    "QSHLTR": (ABS_TOL, REL_TOL),
    "Q10": (ABS_TOL, REL_TOL),
    "PSHLTR": (ABS_TOL, REL_TOL),
    "U10E": (ABS_TOL, REL_TOL),
    "V10E": (ABS_TOL, REL_TOL),
}

OUTPUT_MAP = {
    "USTAR": "ustar",
    "ZNT": "znt",
    "AKHS": "akhs",
    "AKMS": "akms",
    "RMOL": "rmol",
    "RIB": "rib",
    "CHS": "chs",
    "CHS2": "chs2",
    "CQS2": "cqs2",
    "HFX": "hfx",
    "QFX": "qfx",
    "FLX_LH": "flx_lh",
    "FLHC": "flhc",
    "FLQC": "flqc",
    "QGH": "qgh",
    "CPM": "cpm",
    "CT": "ct",
    "QSFC": "qsfc",
    "THZ0": "thz0",
    "QZ0": "qz0",
    "UZ0": "uz0",
    "VZ0": "vz0",
    "PBLH": "pblh",
    "U10": "u10",
    "V10": "v10",
    "T02": "t02",
    "TH02": "th02",
    "TSHLTR": "tshltr",
    "TH10": "th10",
    "Q02": "q02",
    "QSHLTR": "qshltr",
    "Q10": "q10",
    "PSHLTR": "pshltr",
    "U10E": "u10e",
    "V10E": "v10e",
}

# Initial roughness values from the Fortran driver. MYJSFCINIT non-restart
# resets USTAR to 0.1 before MYJSFC, but the driver roughness seed remains.
INITIAL_ZNT = {
    1: 0.10,
    2: 0.08,
    3: 0.05,
    4: 0.001,
    5: 0.08,
    6: 0.0015,
}
INITIAL_USTAR = 0.1


def _load(case_id: int) -> dict:
    with (SAVEPOINT_FP64_DIR / f"myjsfc_case_{case_id}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _col(data: dict, name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _arr(data: dict, name: str):
    return jnp.asarray(_col(data, name), dtype=jnp.float64)


def _metric(actual, expected, *, abs_tol: float, rel_tol: float) -> dict:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    signed = actual_arr - expected_arr
    abs_err = np.abs(signed)
    max_abs = float(np.max(abs_err))
    scale = max(float(np.max(np.abs(expected_arr))), REL_FLOOR)
    max_rel = max_abs / scale
    argmax = int(np.argmax(abs_err))
    return {
        "jax": float(np.ravel(actual_arr)[argmax]),
        "oracle": float(np.ravel(expected_arr)[argmax]),
        "max_abs": max_abs,
        "max_rel": float(max_rel),
        "scale": scale,
        "argmax_flat_index": argmax,
        "signed_error_at_argmax": float(np.ravel(signed)[argmax]),
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "pass": bool(max_abs <= abs_tol or max_rel <= rel_tol),
    }


def _run_case(case_id: int) -> dict:
    data = _load(case_id)
    scalars = data["scalars"]
    qv0 = float(_col(data, "QV")[0])
    out = myjsfc_column(
        u=_arr(data, "U"),
        v=_arr(data, "V"),
        temperature=_arr(data, "T"),
        theta=_arr(data, "TH"),
        qv=_arr(data, "QV"),
        qc=_arr(data, "QC"),
        p_mid=_arr(data, "PMID"),
        dz=_arr(data, "DZ"),
        q2=_arr(data, "Q2"),
        tsk=scalars["TSK"],
        xland=scalars["XLAND"],
        z0base=scalars["Z0BASE"],
        psfc=float(_col(data, "PINT")[0]),
        znt=INITIAL_ZNT[case_id],
        ustar=INITIAL_USTAR,
        mavail=scalars["MAVAIL"],
        qsfc=qv0,
        thz0=float(_col(data, "TH")[0]),
        qz0=qv0 / (1.0 + qv0),
        uz0=0.0,
        vz0=0.0,
        pblh=scalars["PBLH"],
    )

    fields = {
        name: _metric(
            out[OUTPUT_MAP[name]],
            scalars[name],
            abs_tol=FIELD_TOLERANCES[name][0],
            rel_tol=FIELD_TOLERANCES[name][1],
        )
        for name in FIELD_TOLERANCES
    }
    case = {
        "case": case_id,
        "regime": scalars["REGIME"],
        "precision_mode": "fp64",
        "full_wrf_exe": False,
        "initial_conditions": {
            "ustar": INITIAL_USTAR,
            "znt": INITIAL_ZNT[case_id],
            "qsfc": qv0,
            "thz0": float(_col(data, "TH")[0]),
            "qz0": qv0 / (1.0 + qv0),
            "uz0": 0.0,
            "vz0": 0.0,
        },
        "fields": fields,
    }
    case["pass"] = bool(all(metric["pass"] for metric in fields.values()))
    return case


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def test_v060_janjic_sfclay_savepoint_parity_report() -> None:
    cases = [_run_case(case_id) for case_id in CASES]
    overall_pass = all(case["pass"] for case in cases)
    report = {
        "schema": "gpuwrf.v060.janjic_sfclay_savepoint_parity.v1",
        "scheme": "Janjic Eta surface layer (sf_sfclay_physics=2)",
        "paired_scheme": "MYJ PBL (bl_pbl_physics=2)",
        "verdict": "PASS" if overall_pass else "FAIL",
        "overall_pass": bool(overall_pass),
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_sf_myjsfc.F",
            "wrf_sources": [
                "<USER_HOME>/src/wrf_pristine/WRF/share/module_model_constants.F",
                "<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_myjsfc.F",
            ],
            "generation_command": "taskset -c 0-3 env CUDA_VISIBLE_DEVICES= bash proofs/v060/oracle/myjsfc_build_and_run.sh",
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled unmodified WRF MYJSFC source and called MYJSFC/MYJSFCINIT; "
                "this is a real WRF-source oracle, not a JAX self-compare, but not a coupled wrf.exe run."
            ),
            "source_checksums_sha256": _read_lines(SAVEPOINT_FP64_DIR / "myjsfc_wrf_source_checksums.txt"),
            "build_manifest": _read_lines(SAVEPOINT_FP64_DIR / "myjsfc_build_manifest.txt"),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": _git_head(),
        "predeclared_tolerances": {
            field: {"abs": vals[0], "rel": vals[1], "relative_floor": REL_FLOOR}
            for field, vals in FIELD_TOLERANCES.items()
        },
        "regimes_covered": [case["regime"] for case in cases],
        "cases": cases,
    }
    if os.environ.get("GPUWRF_WRITE_PARITY_REPORT") == "1":
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_janjic_sfclay_adapter_contract_keys() -> None:
    data = _load(1)
    scalars = data["scalars"]
    result = step_janjic_sfclay_column(
        _arr(data, "U"),
        _arr(data, "V"),
        _arr(data, "T"),
        _arr(data, "QV"),
        _arr(data, "PMID"),
        _arr(data, "T") / _arr(data, "TH"),
        _arr(data, "DZ"),
        _arr(data, "Q2"),
        tsk=scalars["TSK"],
        xland=scalars["XLAND"],
        z0base=scalars["Z0BASE"],
        psfc=float(_col(data, "PINT")[0]),
        znt=INITIAL_ZNT[1],
        ustar=INITIAL_USTAR,
        mavail=scalars["MAVAIL"],
        theta=_arr(data, "TH"),
        qc=_arr(data, "QC"),
        qsfc=float(_col(data, "QV")[0]),
        thz0=float(_col(data, "TH")[0]),
        qz0=float(_col(data, "QV")[0]) / (1.0 + float(_col(data, "QV")[0])),
        uz0=0.0,
        vz0=0.0,
        pblh=scalars["PBLH"],
    )
    assert isinstance(result, PhysicsStepResult)
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
    assert {
        "UST", "ZNT", "AKHS", "AKMS", "HFX", "QFX", "FLHC", "FLQC",
        "U10", "V10", "T02", "PBLH",
    } <= set(result.diagnostics.surface_layer)
