"""v0.17 Pleim-Xiu LSM (sf_surface_physics=7) fp64 savepoint parity vs WRF.

Loads the 5 fp64 savepoints produced by the standalone Fortran oracle that
links the UNMODIFIED WRF ``module_sf_pxlsm.F`` (subroutines ``SURFPX`` +
``QFLUX``) and calls ``SURFPX`` directly per column, then compares every output
field of the JAX ``pxlsm_columns`` kernel to the oracle outputs at predeclared
tolerances.  The oracle is compiled WRF Fortran (fp64): it is ground truth, not
a JAX-vs-JAX self-compare.

Reproduce the savepoints with::

    taskset -c 0-3 bash proofs/v017/oracle/pxlsm/pxlsm_build_and_run.sh
"""

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

from gpuwrf.physics.lsm_pleim_xiu import ntsps_substeps, pxlsm_columns


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_FP64_DIR = ROOT / "proofs" / "v017" / "savepoints" / "pxlsm" / "fp64"
ORACLE_DIR = ROOT / "proofs" / "v017" / "oracle" / "pxlsm"
REPORT_PATH = ROOT / "proofs" / "v017" / "pxlsm_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5)

# PREDECLARED TOLERANCES, frozen before the parity assertion. The measured fp64
# residual against the compiled-Fortran oracle is at machine precision: worst
# abs ~4e-11 (CAPG, magnitude ~1e5 -> rel ~3e-16); worst real rel ~3e-13
# (GRDFLX). These bounds sit comfortably above the measured residual. A field
# passes when max_abs <= ABS_TOL OR max_rel <= REL_TOL; the abs floor catches
# water-case HFX (~ -0.0, abs ~6e-12) where the relative scale is degenerate.
ABS_TOL = 1.0e-9
REL_TOL = 1.0e-12
REL_FLOOR = 1.0e-12

# Output field -> oracle savepoint column name.
OUTPUT_MAP = {
    "radnet": "RADNET",
    "grdflx": "GRDFLX",
    "hfx": "HFX",
    "qfx": "QFX",
    "lh": "LH",
    "eg": "EG",
    "er": "ER",
    "etr": "ETR",
    "qst": "QST",
    "capg": "CAPG",
    "rs": "RS",
    "ra": "RA",
    "tg": "TG",
    "t2": "T2",
    "wg": "WG",
    "w2": "W2",
    "wr": "WR",
    "tsk": "TSK",
    "canwat": "CANWAT",
    "qsfc": "QSFC",
    "ta2": "TA2",
    "qa2": "QA2",
}


def _load(save_dir: Path, case_id: int) -> dict:
    with (save_dir / f"pxlsm_case_{case_id}.json").open(encoding="utf-8") as fh:
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
    passed = bool(max_abs <= abs_tol or max_rel <= rel_tol)
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


def _run_kernel(data: dict) -> dict:
    c = data["columns"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    dt = float(c["DT"][0])
    ntsps = ntsps_substeps(dt)
    fn = jax.jit(pxlsm_columns, static_argnames=("dt", "ntsps"))
    out = fn(
        arr("SOLDN"), arr("GSW"), arr("LWDN"), arr("Z1"), arr("RMOL"),
        arr("UST"), arr("PSURF"), arr("DENS1"), arr("QV1"), arr("TA1"),
        arr("THETA1"), arr("PRECIP"), arr("CPAIR"), arr("QST12"),
        arr("IFLAND"), arr("ISNOW"),
        arr("TG_IN"), arr("T2_IN"), arr("WG_IN"), arr("W2_IN"), arr("WR_IN"),
        vegfrc=arr("VEGFRC"), lai=arr("LAI"), imperv=arr("IMPERV"),
        canfra=arr("CANFRA"), rstmin=arr("RSTMIN"), emissi=arr("EMISSI"),
        znt=arr("ZNT"), wetfra=arr("WETFRA"), hc_snow=arr("HC_SNOW"),
        snow_fra=arr("SNOW_FRA"), wwlt=arr("WWLT"), wfc=arr("WFC"),
        wres=arr("WRES"), cgsat=arr("CGSAT"), wsat=arr("WSAT"), b=arr("BCH"),
        c1sat=arr("C1SAT"), c2r=arr("C2R"), asoil=arr("ASOIL"), jp=arr("JP"),
        c3=arr("C3"), ds1=arr("DS1"), ds2=arr("DS2"),
        dt=dt, ntsps=ntsps,
    )
    return out


def _run_case(case_id: int) -> dict:
    data = _load(SAVEPOINT_FP64_DIR, case_id)
    scalars = data["scalars"]
    out = _run_kernel(data)
    fields = {}
    for field, key in OUTPUT_MAP.items():
        fields[key] = _field_metrics(
            np.asarray(out[field]), _col(data, key),
            abs_tol=ABS_TOL, rel_tol=REL_TOL,
        )
    case = {
        "case": case_id,
        "regime": scalars["REGIME_NAME"],
        "columns": int(scalars["N"]),
        "land_columns": int(np.sum(_col(data, "IFLAND") < 1.5)),
        "water_columns": int(np.sum(_col(data, "IFLAND") >= 1.5)),
        "full_wrf_exe": bool(scalars["FULL_WRF_EXE"]),
        "precision_mode": scalars["PRECISION_MODE"],
        "ntsps": ntsps_substeps(float(data["columns"]["DT"][0])),
        "fields": fields,
    }
    case["pass"] = bool(all(metric["pass"] for metric in fields.values()))
    return case


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def test_v017_pxlsm_savepoint_parity_report() -> None:
    fp64_cases = [_run_case(case_id) for case_id in CASES]
    fp64_pass = all(case["pass"] for case in fp64_cases)

    worst_abs = max(
        m["max_abs"] for case in fp64_cases for m in case["fields"].values()
    )
    # Worst real relative error excludes degenerate near-zero scales (where the
    # abs floor governs the pass): only count fields that fail the abs floor.
    real_rels = [
        m["max_rel"]
        for case in fp64_cases
        for m in case["fields"].values()
        if m["max_abs"] > ABS_TOL
    ]
    worst_real_rel = max(real_rels) if real_rels else 0.0

    report = {
        "schema": "gpuwrf.v017.pxlsm_savepoint_parity.v1",
        "scheme": "Pleim-Xiu land-surface model (sf_surface_physics=7)",
        "verdict": "PASS" if fp64_pass else "FAIL",
        "overall_pass": bool(fp64_pass),
        "fp64_precision_audit_pass": bool(fp64_pass),
        "worst_max_abs_all_fields": worst_abs,
        "worst_real_max_rel_all_fields": worst_real_rel,
        "oracle": {
            "type": (
                "single-column Fortran driver linked against unmodified WRF "
                "module_sf_pxlsm.F (SURFPX + QFLUX), called per column"
            ),
            "wrf_sources": [
                "/home/user/src/wrf_pristine/WRF/phys/module_sf_pxlsm.F",
                "/home/user/src/wrf_pristine/WRF/phys/module_sf_pxlsm_data.F",
                "/home/user/src/wrf_pristine/WRF/share/module_model_constants.F",
            ],
            "generation_command": (
                "taskset -c 0-3 bash proofs/v017/oracle/pxlsm/pxlsm_build_and_run.sh"
            ),
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled the unmodified WRF modules with "
                "-fdefault-real-8 and called SURFPX directly (NUDGEX=0, XICE=0); "
                "this is a real fp64 WRF-source oracle, not a coupled wrf.exe run."
            ),
            "configuration": (
                "NUDGEX=0 (no soil/temperature nudging; SMASS path not exercised), "
                "XICE=0 (no sea-ice), snow off; ISBA soil constants derived from the "
                "WRF SOILPROP Noilhan-Mahfouf analytic formulas per case."
            ),
            "fp64_source_checksums_sha256": _read_lines(
                SAVEPOINT_FP64_DIR / "pxlsm_wrf_source_checksums.txt"
            ),
            "fp64_build_manifest": _read_lines(
                SAVEPOINT_FP64_DIR / "pxlsm_build_manifest.txt"
            ),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": _git_head(),
        "predeclared_tolerances": {
            "abs": ABS_TOL,
            "rel": REL_TOL,
            "relative_floor": REL_FLOOR,
            "pass_rule": "per field: max_abs <= abs OR max_rel <= rel",
            "note": (
                "Water-case HFX (~ -0.0) passes on the abs floor; its relative "
                "scale is degenerate. All other fields pass on both abs and rel."
            ),
        },
        "regimes_covered": [case["regime"] for case in fp64_cases],
        "cases": fp64_cases,
    }
    # The committed lane report is AUTHORITATIVE. By default this test ASSERTS the
    # parity verdict without overwriting the committed proof. Set
    # GPUWRF_WRITE_PARITY_REPORT=1 to deliberately regenerate the report.
    if os.environ.get("GPUWRF_WRITE_PARITY_REPORT") == "1":
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    assert report["verdict"] == "PASS", report


def test_v017_pxlsm_water_passthrough() -> None:
    """Water columns (IFLAND>=1.5) keep their soil carry; GRDFLX/CAPG/RS = 0."""

    data = _load(SAVEPOINT_FP64_DIR, 5)
    out = _run_kernel(data)
    is_water = _col(data, "IFLAND") >= 1.5
    assert bool(np.all(is_water))
    # SURFPX leaves the soil carry unchanged over water -> outputs equal inputs.
    for carry in ("tg", "t2", "wg", "w2", "wr"):
        np.testing.assert_allclose(
            np.asarray(out[carry])[is_water],
            _col(data, f"{carry.upper()}_IN")[is_water],
            atol=ABS_TOL, rtol=REL_TOL,
        )
    for zero_field in ("grdflx", "capg", "rs"):
        assert float(np.max(np.abs(np.asarray(out[zero_field])[is_water]))) <= ABS_TOL
