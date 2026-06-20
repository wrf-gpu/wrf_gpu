"""v0.6.0 ACM2 PBL savepoint parity against the WRF module oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsStepResult
from gpuwrf.physics.pbl_acm2 import step_acm2_column


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "acm2_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5, 6)

TEND_ABS = 3.0e-6
TEND_REL = 2.0e-3
TEND_REL_FLOOR = 1.0e-12
PBLH_ABS = 2.0e-2
PBLH_REL = 2.0e-5
RMOL_ABS = 1.0e-7
RMOL_REL = 2.0e-5
EXCH_ABS = 5.0e-4
EXCH_REL = 2.0e-3
EXCH_REL_FLOOR = 1.0e-10

TENDENCY_FIELDS = (
    ("RUBLTEN", "u"),
    ("RVBLTEN", "v"),
    ("RTHBLTEN", "theta"),
    ("RQVBLTEN", "qv"),
)
EXCHANGE_FIELDS = (
    ("EXCH_H", "exch_h"),
    ("EXCH_M", "exch_m"),
)


def _load(case_id: int) -> dict:
    with (SAVEPOINT_DIR / f"acm2_case_{case_id}.json").open() as fh:
        return json.load(fh)


def _col(data: dict, name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _scalar_metrics(actual, expected, *, abs_tol: float, rel_tol: float, rel_floor: float = TEND_REL_FLOOR) -> dict:
    actual_f = float(np.asarray(actual, dtype=np.float64).reshape(()))
    expected_f = float(expected)
    max_abs = abs(actual_f - expected_f)
    scale = max(abs(expected_f), rel_floor)
    max_rel = max_abs / scale
    return {
        "jax": actual_f,
        "oracle": expected_f,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "pass": bool(max_abs <= abs_tol or max_rel <= rel_tol),
    }


def _field_metrics(actual, expected, *, abs_tol: float, rel_tol: float, rel_floor: float) -> dict:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    signed = actual_arr - expected_arr
    abs_err = np.abs(signed)
    argmax = int(np.argmax(abs_err))
    max_abs = float(abs_err[argmax])
    scale = max(float(np.max(np.abs(expected_arr))), rel_floor)
    max_rel = max_abs / scale
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "scale": scale,
        "argmax_level_1based": argmax + 1,
        "signed_error_at_argmax": float(signed[argmax]),
        "abs_error_by_level": [float(x) for x in abs_err],
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "pass": bool(max_abs <= abs_tol or max_rel <= rel_tol),
    }


def _run_case(case_id: int) -> dict:
    data = _load(case_id)
    scalars = data["scalars"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731

    result = step_acm2_column(
        arr("U"),
        arr("V"),
        arr("THETA"),
        arr("T"),
        arr("QV"),
        arr("RR"),
        arr("DZ"),
        pblh_initial=scalars["PBLH_INITIAL"],
        ust=scalars["UST"],
        hfx=scalars["HFX"],
        qfx=scalars["QFX"],
        wspd=scalars["WSPD"],
        mut=scalars["MUT"],
        dt=scalars["DT"],
        xtime=scalars["XTIME"],
        qc=arr("QC"),
        qi=arr("QI"),
    )
    assert isinstance(result, PhysicsStepResult)

    tendencies = result.tendency.state_tendencies
    diagnostics = result.diagnostics.pbl
    case = {
        "case": case_id,
        "regime": scalars["REGIME_NAME"],
        "surface_inputs": {
            "ust": scalars["UST"],
            "hfx": scalars["HFX"],
            "qfx": scalars["QFX"],
            "pblh_initial": scalars["PBLH_INITIAL"],
            "wspd": scalars["WSPD"],
        },
        "branch": {
            "noconv_jax": int(np.asarray(diagnostics["noconv"]).reshape(())),
            "noconv_oracle": int(scalars["NOCONV"]),
            "regime_jax": float(np.asarray(diagnostics["regime"]).reshape(())),
            "regime_oracle": float(scalars["REGIME"]),
        },
        "pblh": _scalar_metrics(diagnostics["pblh"], scalars["PBLH"], abs_tol=PBLH_ABS, rel_tol=PBLH_REL),
        "kpbl": {
            "jax": int(np.asarray(diagnostics["kpbl"]).reshape(())),
            "oracle": int(scalars["KPBL"]),
        },
        "rmol": _scalar_metrics(
            diagnostics["rmol"],
            scalars["RMOL"],
            abs_tol=RMOL_ABS,
            rel_tol=RMOL_REL,
            rel_floor=1.0e-12,
        ),
        "fields": {},
    }
    case["branch"]["pass"] = (
        case["branch"]["noconv_jax"] == case["branch"]["noconv_oracle"]
        and case["branch"]["regime_jax"] == case["branch"]["regime_oracle"]
    )
    case["kpbl"]["pass"] = case["kpbl"]["jax"] == case["kpbl"]["oracle"]

    for oracle_name, state_key in TENDENCY_FIELDS:
        case["fields"][oracle_name] = _field_metrics(
            tendencies[state_key],
            _col(data, oracle_name),
            abs_tol=TEND_ABS,
            rel_tol=TEND_REL,
            rel_floor=TEND_REL_FLOOR,
        )
    for oracle_name, diag_key in EXCHANGE_FIELDS:
        case["fields"][oracle_name] = _field_metrics(
            diagnostics[diag_key],
            _col(data, oracle_name),
            abs_tol=EXCH_ABS,
            rel_tol=EXCH_REL,
            rel_floor=EXCH_REL_FLOOR,
        )

    checks = [
        case["branch"]["pass"],
        case["pblh"]["pass"],
        case["kpbl"]["pass"],
        case["rmol"]["pass"],
        *(metric["pass"] for metric in case["fields"].values()),
    ]
    case["pass"] = bool(all(checks))
    return case


def test_v060_acm2_savepoint_parity_report() -> None:
    cases = [_run_case(case_id) for case_id in CASES]
    checksum_path = SAVEPOINT_DIR / "acm2_wrf_source_checksums.txt"
    source_checksums = checksum_path.read_text(encoding="utf-8").splitlines()
    report = {
        "schema": "gpuwrf.v060.acm2_savepoint_parity.v1",
        "scheme": "ACM2 PBL (bl_pbl_physics=7)",
        "verdict": "PASS" if all(case["pass"] for case in cases) else "FAIL",
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_bl_acm.F",
            "wrf_sources": ["<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_acm.F"],
            "requested_source_path_absent": "<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_acm2.F",
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
            "full_wrf_exe": False,
            "source_checksums_sha256": source_checksums,
            "note": "Real WRF-module oracle, not a JAX self-compare; not a coupled wrf.exe run.",
        },
        "predeclared_tolerances": {
            "tendencies": {"abs": TEND_ABS, "rel": TEND_REL, "relative_floor": TEND_REL_FLOOR},
            "pblh_m": {"abs": PBLH_ABS, "rel": PBLH_REL},
            "kpbl": "exact integer match",
            "branch_regime": "exact ACM2 noconv/regime match",
            "rmol": {"abs": RMOL_ABS, "rel": RMOL_REL},
            "exchange_coefficients": {"abs": EXCH_ABS, "rel": EXCH_REL, "relative_floor": EXCH_REL_FLOOR},
        },
        "regimes_covered": [case["regime"] for case in cases],
        "cases": cases,
    }
    # Regenerate the committed canonical report ONLY on explicit request
    # (GPUWRF_WRITE_PROOFS=1); the verdict assertion below is the correctness
    # signal, so the default suite run does not re-dirty the proof with fp noise.
    if os.environ.get("GPUWRF_WRITE_PROOFS", "").strip().lower() in {"1", "true", "yes", "on"}:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_acm2_adapter_contract_keys() -> None:
    data = _load(1)
    scalars = data["scalars"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    result = step_acm2_column(
        arr("U"),
        arr("V"),
        arr("THETA"),
        arr("T"),
        arr("QV"),
        arr("RR"),
        arr("DZ"),
        pblh_initial=scalars["PBLH_INITIAL"],
        ust=scalars["UST"],
        hfx=scalars["HFX"],
        qfx=scalars["QFX"],
        wspd=scalars["WSPD"],
        mut=scalars["MUT"],
        dt=scalars["DT"],
        xtime=scalars["XTIME"],
        qc=arr("QC"),
        qi=arr("QI"),
    )
    assert set(result.tendency.state_tendencies) == {"u", "v", "theta", "qv"}
    assert set(result.tendency.state_replacements) == set()
    assert set(result.tendency.accumulator_increments) == set()
    assert {"pblh", "kpbl", "regime", "noconv", "rmol", "exch_h", "exch_m"} <= set(result.diagnostics.pbl)
