"""v0.6.0 Kain-Fritsch savepoint parity against the WRF module oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsStepResult
from gpuwrf.physics import cumulus_kf as kf


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "kf_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5)

TEND_REL = 2.0e-3
TEND_ABS = 1.0e-7
TEND_REL_FLOOR = 1.0e-9
RAIN_REL = 3.0e-3
RAIN_ABS = 1.0e-4
NCA_ABS = 1.0e-6
W0AVG_ABS = 1.0e-6
ZERO_ABS = 1.0e-12
TENDENCY_FIELDS = (
    ("RTHCUTEN", "theta"),
    ("RQVCUTEN", "qv"),
    ("RQCCUTEN", "qc"),
    ("RQRCUTEN", "qr"),
    ("RQICUTEN", "qi"),
    ("RQSCUTEN", "qs"),
)


def _load(case_id: int) -> dict:
    with (SAVEPOINT_DIR / f"kf_case_{case_id}.json").open() as fh:
        return json.load(fh)


def _as_float(value) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(()))


def _field_metrics(actual, expected) -> dict[str, float | bool]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    max_abs = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), TEND_REL_FLOOR)
    max_rel = max_abs / scale
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "pass": bool(max_abs <= TEND_ABS or max_rel <= TEND_REL),
    }


def _rain_metrics(actual, expected) -> dict[str, float | bool]:
    max_abs = abs(float(actual) - float(expected))
    tolerance = max(RAIN_ABS, RAIN_REL * abs(float(expected)))
    return {"max_abs": max_abs, "tolerance": tolerance, "pass": bool(max_abs <= tolerance)}


def _run_case(case_id: int) -> dict:
    data = _load(case_id)
    scalars = data["scalars"]
    columns = data["columns"]

    arr = lambda name: jnp.asarray(columns[name], dtype=jnp.float64)  # noqa: E731
    result = kf.step_kf_column(
        arr("T"),
        arr("QV"),
        arr("P"),
        arr("DZ"),
        arr("RHO"),
        arr("W0AVG"),
        arr("U"),
        arr("V"),
        scalars["DT"],
        scalars["DX"],
        w=None,
        nca=-100.0,
        stepcu=scalars["STEPCU"],
        warm_rain=False,
        f_qi=True,
        f_qs=True,
    )
    assert isinstance(result, PhysicsStepResult)
    tendencies = result.tendency.state_tendencies
    diagnostics = result.diagnostics.cumulus
    carry = result.carry.cumulus

    case = {
        "case": case_id,
        "regime": "none" if int(round(scalars["SHALL"])) == 2 else ("shallow" if int(round(scalars["SHALL"])) == 1 else "deep"),
        "trigger": {
            "oracle_ishall": int(round(scalars["SHALL"])),
            "jax_ishall": int(_as_float(diagnostics["ishall"])),
        },
        "cutop": {"oracle": float(scalars["CUTOP"]), "jax": _as_float(diagnostics["cutop"])},
        "cubot": {"oracle": float(scalars["CUBOT"]), "jax": _as_float(diagnostics["cubot"])},
        "rainc_acc": _rain_metrics(result.tendency.accumulator_increments["rainc_acc"], scalars["RAINCV"]),
        "nca": {
            "oracle": float(scalars["NCA"]),
            "jax": _as_float(carry["nca"]),
        },
        "w0avg": {
            "max_abs": float(np.max(np.abs(np.asarray(carry["w0avg"]) - np.asarray(columns["W0AVG"], dtype=np.float64)))),
            "tolerance": W0AVG_ABS,
        },
        "fields": {},
    }
    case["trigger"]["pass"] = case["trigger"]["oracle_ishall"] == case["trigger"]["jax_ishall"]
    case["cutop"]["pass"] = abs(case["cutop"]["oracle"] - case["cutop"]["jax"]) < 0.5
    case["cubot"]["pass"] = abs(case["cubot"]["oracle"] - case["cubot"]["jax"]) < 0.5
    case["nca"]["max_abs"] = abs(case["nca"]["oracle"] - case["nca"]["jax"])
    case["nca"]["pass"] = case["nca"]["max_abs"] <= NCA_ABS
    case["w0avg"]["pass"] = case["w0avg"]["max_abs"] <= W0AVG_ABS

    for oracle_name, state_key in TENDENCY_FIELDS:
        case["fields"][oracle_name] = _field_metrics(tendencies[state_key], columns[oracle_name])

    if case["regime"] == "none":
        no_tendency = all(float(np.max(np.abs(np.asarray(tendencies[state_key])))) <= ZERO_ABS for _, state_key in TENDENCY_FIELDS)
        no_rain = abs(_as_float(result.tendency.accumulator_increments["rainc_acc"])) <= ZERO_ABS
        case["nontrigger_zero_outputs"] = {"pass": bool(no_tendency and no_rain), "tolerance": ZERO_ABS}

    checks = [
        case["trigger"]["pass"],
        case["cutop"]["pass"],
        case["cubot"]["pass"],
        case["rainc_acc"]["pass"],
        case["nca"]["pass"],
        case["w0avg"]["pass"],
        *(field["pass"] for field in case["fields"].values()),
    ]
    if case["regime"] == "none":
        checks.append(case["nontrigger_zero_outputs"]["pass"])
    case["pass"] = bool(all(checks))
    return case


def test_v060_kf_savepoint_parity_report() -> None:
    cases = [_run_case(case_id) for case_id in CASES]
    report = {
        "schema": "gpuwrf.v060.kf_savepoint_parity.v1",
        "verdict": "PASS" if all(case["pass"] for case in cases) else "FAIL",
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_cu_kfeta.F",
            "wrf_source": "/home/enric/src/wrf_pristine/WRF/phys/module_cu_kfeta.F",
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/kf_build_and_run.sh",
            "full_wrf_exe_run": False,
            "note": "M20/current repo search did not expose a cumulus savepoint factory. This is a real WRF-module oracle, not a JAX self-compare, but not a full coupled wrf.exe case.",
            "source_sha256": "e6376c2d85c45470f49d545b25d513b5ec111bf36b87beebc740bf42825c6e5f",
            "source_checksum_sidecar": "proofs/v060/savepoints/kf_wrf_source_checksums.txt",
        },
        "predeclared_tolerances": {
            "trigger": "exact ISHALL",
            "cutop_cubot": "abs < 0.5 model level",
            "triggered_tendencies": {"abs": TEND_ABS, "rel": TEND_REL, "relative_floor": TEND_REL_FLOOR},
            "rainc_acc_or_raincv_mm": {"abs": RAIN_ABS, "rel": RAIN_REL},
            "nca_seconds_abs": NCA_ABS,
            "w0avg_m_s_abs": W0AVG_ABS,
            "nontrigger_zero_abs": ZERO_ABS,
        },
        "cases": cases,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_kf_adapter_contract_keys() -> None:
    data = _load(1)
    scalars = data["scalars"]
    columns = data["columns"]
    arr = lambda name: jnp.asarray(columns[name], dtype=jnp.float64)  # noqa: E731
    result = kf.step_kf_column(
        arr("T"),
        arr("QV"),
        arr("P"),
        arr("DZ"),
        arr("RHO"),
        arr("W0AVG"),
        arr("U"),
        arr("V"),
        scalars["DT"],
        scalars["DX"],
        w=None,
        nca=-100.0,
        stepcu=scalars["STEPCU"],
    )
    assert set(result.tendency.state_tendencies) == {"u", "v", "theta", "qv", "qc", "qr", "qi", "qs"}
    assert set(result.tendency.accumulator_increments) == {"rainc_acc"}
    assert set(result.carry.cumulus) == {"w0avg", "nca"}
    assert {"raincv", "rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten", "rqscuten"} <= set(
        result.diagnostics.cumulus
    )
