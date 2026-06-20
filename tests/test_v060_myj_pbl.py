"""v0.6.0 MYJ PBL savepoint parity against WRF MYJPBL."""

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
from gpuwrf.physics.pbl_myj import myjpbl_column, step_myj_pbl_column


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_FP64_DIR = ROOT / "proofs" / "v060" / "savepoints_fp64"
REPORT_PATH = ROOT / "proofs" / "v060" / "myj_pbl_savepoint_parity.json"
CASES = (1, 2, 3, 4, 5, 6)

# PREDECLARED fp64 savepoint tolerances. MYJPBL is a mostly arithmetic column
# port; the largest residuals are roundoff in exchange coefficients.
ARRAY_ABS_TOL = 5.0e-10
ARRAY_REL_TOL = 5.0e-11
SCALAR_ABS_TOL = 5.0e-10
SCALAR_REL_TOL = 5.0e-12
REL_FLOOR = 1.0e-30

ARRAY_FIELDS = (
    "TKE_MYJ",
    "EXCH_H",
    "EL_MYJ",
    "RUBLTEN",
    "RVBLTEN",
    "RTHBLTEN",
    "RQVBLTEN",
)
SCALAR_FIELDS = ("PBLH", "MIXHT")


def _load_pbl(case_id: int) -> dict:
    with (SAVEPOINT_FP64_DIR / f"myjpbl_case_{case_id}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_sfclay(case_id: int) -> dict:
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
    data = _load_pbl(case_id)
    sfclay = _load_sfclay(case_id)
    scalars = data["scalars"]
    initial_tke = 0.5 * _col(sfclay, "Q2")
    out = myjpbl_column(
        u=_col(data, "U"),
        v=_col(data, "V"),
        temperature=_col(data, "T"),
        theta=_col(data, "TH"),
        qv=_col(data, "QV"),
        qc=_col(data, "QC"),
        p_mid=_col(data, "PMID"),
        p_int=_col(data, "PINT"),
        exner=_col(data, "EXNER"),
        dz=_col(data, "DZ"),
        tke=initial_tke,
        tsk=scalars["TSK"],
        xland=scalars["XLAND"],
        ustar=scalars["USTAR"],
        znt=scalars["ZNT"],
        akhs=scalars["AKHS"],
        akms=scalars["AKMS"],
        chklowq=scalars["CHKLOWQ"],
        elflx=scalars["ELFLX"],
        thz0=scalars["THZ0"],
        qz0=scalars["QZ0"],
        uz0=scalars["UZ0"],
        vz0=scalars["VZ0"],
        qsfc=scalars["QSFC"],
        ct=scalars["CT"],
        dt=scalars["DT"],
        stepbl=scalars["STEPBL"],
        ht=scalars["HT"],
    )

    fields = {
        name: _metric(
            out[name],
            _col(data, name),
            abs_tol=ARRAY_ABS_TOL,
            rel_tol=ARRAY_REL_TOL,
        )
        for name in ARRAY_FIELDS
    }
    scalars_out = {
        name: _metric(
            out[name],
            scalars[name],
            abs_tol=SCALAR_ABS_TOL,
            rel_tol=SCALAR_REL_TOL,
        )
        for name in SCALAR_FIELDS
    }
    kpbl = {
        "jax": int(out["KPBL"]),
        "oracle": int(scalars["KPBL"]),
        "pass": int(out["KPBL"]) == int(scalars["KPBL"]),
    }
    ungated = {
        "EXCH_M": {
            "max_abs_jax": float(np.max(np.abs(np.asarray(out["EXCH_M"], dtype=np.float64)))),
            "reason": "unmodified WRF MYJPBL does not expose EXCH_M/Km as an output argument in this oracle driver",
        },
        "AKM": {
            "max_abs_jax": float(np.max(np.abs(np.asarray(out["AKM"], dtype=np.float64)))),
            "reason": "AKM is a local MYJPBL work array, not a dumped unmodified-WRF output field",
        },
        "AKH": {
            "max_abs_jax": float(np.max(np.abs(np.asarray(out["AKH"], dtype=np.float64)))),
            "reason": "AKH is a local MYJPBL work array; exported WRF EXCH_H is the gated heat-exchange diagnostic",
        },
    }
    case = {
        "case": case_id,
        "regime": scalars["REGIME"],
        "precision_mode": "fp64",
        "full_wrf_exe": False,
        "initial_tke_source": "0.5 * matching MYJSFC Q2 savepoint field, matching myjpbl_oracle_driver.f90",
        "fields": fields,
        "scalars": scalars_out,
        "kpbl": kpbl,
        "ungated_jax_diagnostics": ungated,
    }
    case["pass"] = bool(
        all(metric["pass"] for metric in fields.values())
        and all(metric["pass"] for metric in scalars_out.values())
        and kpbl["pass"]
    )
    return case


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def test_v060_myj_pbl_savepoint_parity_report() -> None:
    cases = [_run_case(case_id) for case_id in CASES]
    overall_pass = all(case["pass"] for case in cases)
    report = {
        "schema": "gpuwrf.v060.myj_pbl_savepoint_parity.v1",
        "scheme": "MYJ PBL (bl_pbl_physics=2)",
        "paired_scheme": "Janjic Eta surface layer (sf_sfclay_physics=2)",
        "verdict": "PASS" if overall_pass else "FAIL",
        "overall_pass": bool(overall_pass),
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_bl_myjpbl.F and module_sf_myjsfc.F",
            "wrf_sources": [
                "<USER_HOME>/src/wrf_pristine/WRF/share/module_model_constants.F",
                "<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_myjsfc.F",
                "<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_myjpbl.F",
            ],
            "generation_command": "taskset -c 0-3 env CUDA_VISIBLE_DEVICES= bash proofs/v060/oracle/myjpbl_build_and_run.sh",
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled unmodified WRF MYJSFC/MYJPBL source and called the mandated pair; "
                "this is a real WRF-source oracle, not a JAX self-compare, but not a coupled wrf.exe run."
            ),
            "source_checksums_sha256": _read_lines(SAVEPOINT_FP64_DIR / "myjpbl_wrf_source_checksums.txt"),
            "build_manifest": _read_lines(SAVEPOINT_FP64_DIR / "myjpbl_build_manifest.txt"),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": _git_head(),
        "predeclared_tolerances": {
            "array_fields": {"abs": ARRAY_ABS_TOL, "rel": ARRAY_REL_TOL, "relative_floor": REL_FLOOR},
            "scalar_fields": {"abs": SCALAR_ABS_TOL, "rel": SCALAR_REL_TOL, "relative_floor": REL_FLOOR},
            "kpbl": "exact integer match",
        },
        "gated_fields": {
            "tke": "TKE_MYJ",
            "kh": "EXCH_H",
            "mixing_length": "EL_MYJ",
            "tendencies": ["RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN"],
            "pblh": "PBLH",
            "kpbl": "KPBL",
        },
        "km_limitation": (
            "WRF MYJPBL computes AKM/Km internally but does not expose it as an output argument; "
            "the JAX port computes EXCH_M/AKM, recorded per case as ungated diagnostics, but Km is not "
            "claimed as WRF-parity-gated by this proof object."
        ),
        "regimes_covered": [case["regime"] for case in cases],
        "cases": cases,
    }
    if os.environ.get("GPUWRF_WRITE_PARITY_REPORT") == "1":
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_myj_pbl_adapter_contract_keys() -> None:
    data = _load_pbl(1)
    sfclay = _load_sfclay(1)
    scalars = data["scalars"]
    result = step_myj_pbl_column(
        _arr(data, "U"),
        _arr(data, "V"),
        _arr(data, "T"),
        _arr(data, "TH"),
        _arr(data, "QV"),
        _arr(data, "QC"),
        _arr(data, "PMID"),
        _arr(data, "PINT"),
        _arr(data, "EXNER"),
        _arr(data, "DZ"),
        0.5 * _arr(sfclay, "Q2"),
        tsk=scalars["TSK"],
        xland=scalars["XLAND"],
        ustar=scalars["USTAR"],
        znt=scalars["ZNT"],
        akhs=scalars["AKHS"],
        akms=scalars["AKMS"],
        chklowq=scalars["CHKLOWQ"],
        elflx=scalars["ELFLX"],
        thz0=scalars["THZ0"],
        qz0=scalars["QZ0"],
        uz0=scalars["UZ0"],
        vz0=scalars["VZ0"],
        qsfc=scalars["QSFC"],
        ct=scalars["CT"],
        dt=scalars["DT"],
        stepbl=scalars["STEPBL"],
        ht=scalars["HT"],
    )
    assert isinstance(result, PhysicsStepResult)
    result.tendency.validate_keys()
    assert set(result.tendency.state_tendencies) == {"u", "v", "theta", "qv"}
    assert set(result.tendency.state_replacements) == set()
    assert set(result.carry.pbl) == {"tke_pbl", "el_pbl"}
    assert {
        "pblh", "kpbl", "mixht", "tke_pbl", "exch_h", "exch_m", "el_pbl",
    } <= set(result.diagnostics.pbl)
