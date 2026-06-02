"""v0.6.0 YSU PBL savepoint parity against the WRF module oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsStepResult
from gpuwrf.physics.pbl_ysu import step_ysu_column


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "ysu_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5, 6)

TEND_ABS = 2.0e-6
TEND_REL = 2.0e-3
TEND_REL_FLOOR = 1.0e-12
PBLH_ABS = 2.0e-2
PBLH_REL = 2.0e-5
EXCH_ABS = 2.0e-4
EXCH_REL = 2.0e-3
EXCH_REL_FLOOR = 1.0e-10
DIAG_ABS = 2.0e-4
DIAG_REL = 2.0e-3

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
    with (SAVEPOINT_DIR / f"ysu_case_{case_id}.json").open() as fh:
        return json.load(fh)


def _col(data: dict, name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _scalar_metrics(actual, expected, *, abs_tol: float, rel_tol: float) -> dict[str, float | bool]:
    actual_f = float(np.asarray(actual, dtype=np.float64).reshape(()))
    expected_f = float(expected)
    max_abs = abs(actual_f - expected_f)
    scale = max(abs(expected_f), TEND_REL_FLOOR)
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


def _field_metrics(actual, expected, *, abs_tol: float, rel_tol: float, rel_floor: float) -> dict[str, float | bool]:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    max_abs = float(np.max(np.abs(actual_arr - expected_arr)))
    scale = max(float(np.max(np.abs(expected_arr))), rel_floor)
    max_rel = max_abs / scale
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "scale": scale,
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "pass": bool(max_abs <= abs_tol or max_rel <= rel_tol),
    }


def _run_case(case_id: int) -> dict:
    data = _load(case_id)
    scalars = data["scalars"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731

    result = step_ysu_column(
        arr("U"),
        arr("V"),
        arr("T"),
        arr("QV"),
        arr("P"),
        jnp.asarray(data["columns"]["PDI"], dtype=jnp.float64),
        arr("PI"),
        arr("DZ"),
        psfc=scalars["PSFC"],
        znt=scalars["ZNT"],
        ust=scalars["UST"],
        hfx=scalars["HFX"],
        qfx=scalars["QFX"],
        wspd=scalars["WSPD"],
        br=scalars["BR"],
        psim=scalars["PSIM"],
        psih=scalars["PSIH"],
        dt=scalars["DT"],
        xland=scalars["XLAND"],
        u10=scalars["U10"],
        v10=scalars["V10"],
    )
    assert isinstance(result, PhysicsStepResult)

    tendencies = result.tendency.state_tendencies
    diagnostics = result.diagnostics.pbl
    case = {
        "case": case_id,
        "regime": scalars["REGIME"],
        "surface_inputs": {
            "ust": scalars["UST"],
            "hfx": scalars["HFX"],
            "qfx": scalars["QFX"],
            "br": scalars["BR"],
            "xland": scalars["XLAND"],
        },
        "pblh": _scalar_metrics(diagnostics["pblh"], scalars["PBLH"], abs_tol=PBLH_ABS, rel_tol=PBLH_REL),
        "kpbl": {
            "jax": int(np.asarray(diagnostics["kpbl"]).reshape(())),
            "oracle": int(scalars["KPBL"]),
        },
        "diagnostics": {
            "wstar": _scalar_metrics(diagnostics["wstar"], scalars["WSTAR"], abs_tol=DIAG_ABS, rel_tol=DIAG_REL),
            "delta": _scalar_metrics(diagnostics["delta"], scalars["DELTA"], abs_tol=DIAG_ABS, rel_tol=DIAG_REL),
        },
        "fields": {},
    }
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
        case["pblh"]["pass"],
        case["kpbl"]["pass"],
        *(metric["pass"] for metric in case["diagnostics"].values()),
        *(metric["pass"] for metric in case["fields"].values()),
    ]
    case["pass"] = bool(all(checks))
    return case


def test_v060_ysu_savepoint_parity_report() -> None:
    cases = [_run_case(case_id) for case_id in CASES]
    checksum_path = SAVEPOINT_DIR / "ysu_wrf_source_checksums.txt"
    source_checksums = checksum_path.read_text(encoding="utf-8").splitlines()
    report = {
        "schema": "gpuwrf.v060.ysu_savepoint_parity.v1",
        "scheme": "YSU PBL (bl_pbl_physics=1)",
        "verdict": "PASS" if all(case["pass"] for case in cases) else "FAIL",
        "oracle": {
            "type": "single-column Fortran driver linked against unmodified WRF module_bl_ysu.F and bl_ysu.F90",
            "wrf_sources": [
                "/home/enric/src/wrf_pristine/WRF/phys/module_bl_ysu.F",
                "/home/enric/src/wrf_pristine/WRF/phys/physics_mmm/bl_ysu.F90",
            ],
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
            "full_wrf_exe": False,
            "source_checksums_sha256": source_checksums,
            "note": "Real WRF-module oracle, not a JAX self-compare; not a coupled wrf.exe run.",
        },
        "predeclared_tolerances": {
            "tendencies": {"abs": TEND_ABS, "rel": TEND_REL, "relative_floor": TEND_REL_FLOOR},
            "pblh_m": {"abs": PBLH_ABS, "rel": PBLH_REL},
            "kpbl": "exact integer match",
            "exchange_coefficients": {"abs": EXCH_ABS, "rel": EXCH_REL, "relative_floor": EXCH_REL_FLOOR},
            "wstar_delta": {"abs": DIAG_ABS, "rel": DIAG_REL},
        },
        "regimes_covered": [case["regime"] for case in cases],
        "cases": cases,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    assert report["verdict"] == "PASS"


def test_v060_ysu_adapter_contract_keys() -> None:
    data = _load(1)
    scalars = data["scalars"]
    arr = lambda name: jnp.asarray(_col(data, name), dtype=jnp.float64)  # noqa: E731
    result = step_ysu_column(
        arr("U"),
        arr("V"),
        arr("T"),
        arr("QV"),
        arr("P"),
        jnp.asarray(data["columns"]["PDI"], dtype=jnp.float64),
        arr("PI"),
        arr("DZ"),
        psfc=scalars["PSFC"],
        znt=scalars["ZNT"],
        ust=scalars["UST"],
        hfx=scalars["HFX"],
        qfx=scalars["QFX"],
        wspd=scalars["WSPD"],
        br=scalars["BR"],
        psim=scalars["PSIM"],
        psih=scalars["PSIH"],
        dt=scalars["DT"],
        xland=scalars["XLAND"],
        u10=scalars["U10"],
        v10=scalars["V10"],
    )
    assert set(result.tendency.state_tendencies) == {"u", "v", "theta", "qv"}
    assert set(result.tendency.state_replacements) == set()
    assert set(result.tendency.accumulator_increments) == set()
    assert {"pblh", "kpbl", "exch_h", "exch_m", "wstar", "delta"} <= set(result.diagnostics.pbl)
