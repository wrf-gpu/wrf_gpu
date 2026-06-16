"""v0.6.0 modified-Tiedtke cumulus parity against WRF savepoints.

Oracle: single-column driver linked against unmodified WRF
``phys/module_cu_tiedtke.F`` plus WRF ``share/module_model_constants.F``.

Run CPU-only:
  PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 pytest -q tests/test_tiedtke_cumulus_oracle.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")


ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "proofs" / "v060" / "savepoints"
REPORT = ROOT / "proofs" / "v060" / "tiedtke_savepoint_parity_report.json"
CASES = (1, 2, 3, 4, 5)
TENDENCY_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
MOMENTUM_FIELDS = ("RUCUTEN", "RVCUTEN")

# Predeclared before comparison: fp64 Python/JAX transcription vs WRF REAL*4.
TEND_REL = 5.0e-3
TEND_ABS_FLOOR = 1.0e-10
MOM_REL = 5.0e-3
MOM_ABS_FLOOR = 1.0e-10
RAINCV_REL = 1.0e-3
RAINCV_ABS = 2.0e-4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(case: int) -> dict:
    with (SAVE / f"tiedtke_case_{case}.json").open() as fh:
        return json.load(fh)


def _arr(columns: dict, name: str) -> np.ndarray:
    return np.asarray(columns[name], dtype=np.float64)


def _metrics(actual, oracle, rel_tol: float, abs_floor: float) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    max_abs = float(np.max(np.abs(actual - oracle)))
    scale = max(float(np.max(np.abs(oracle))), abs_floor)
    max_rel = max_abs / scale
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "pass": bool(max_rel <= rel_tol or max_abs <= abs_floor),
    }


def test_tiedtke_savepoint_parity_report():
    from gpuwrf.physics.cumulus_tiedtke import step_tiedtke_column, tiedtke_column

    started = time.perf_counter()
    failures: list[str] = []
    case_records = []

    for case in CASES:
        d = _load(case)
        s = d["scalars"]
        c = d["columns"]
        out = tiedtke_column(
            _arr(c, "T"),
            _arr(c, "QV"),
            _arr(c, "QC"),
            _arr(c, "QI"),
            _arr(c, "P"),
            _arr(c, "P8W"),
            _arr(c, "DZ"),
            _arr(c, "RHO"),
            _arr(c, "PI"),
            _arr(c, "U"),
            _arr(c, "V"),
            _arr(c, "W"),
            _arr(c, "QVFTEN"),
            _arr(c, "QVPBLTEN"),
            s["QFX"],
            s["XLAND"],
            _arr(c, "ZNU"),
            s["DT"],
            stepcu=s["STEPCU"],
        )
        step = step_tiedtke_column(
            _arr(c, "T"),
            _arr(c, "QV"),
            _arr(c, "QC"),
            _arr(c, "QI"),
            _arr(c, "P"),
            _arr(c, "P8W"),
            _arr(c, "DZ"),
            _arr(c, "RHO"),
            _arr(c, "PI"),
            _arr(c, "U"),
            _arr(c, "V"),
            _arr(c, "W"),
            _arr(c, "QVFTEN"),
            _arr(c, "QVPBLTEN"),
            s["QFX"],
            s["XLAND"],
            _arr(c, "ZNU"),
            s["DT"],
            stepcu=s["STEPCU"],
        )
        step.tendency.validate_keys()

        rec = {
            "case": case,
            "regime": {0: "non_triggering", 1: "deep", 2: "shallow", 3: "midlevel"}.get(int(s["KTYPE"]), "unknown"),
            "ktype": {
                "oracle": int(s["KTYPE"]),
                "jax": int(out["KTYPE"]),
                "pass": bool(int(s["KTYPE"]) == int(out["KTYPE"])),
            },
            "fields": {},
            "rainc_acc": {},
        }
        if not rec["ktype"]["pass"]:
            failures.append(f"case {case} KTYPE oracle={s['KTYPE']} jax={int(out['KTYPE'])}")

        for field in TENDENCY_FIELDS:
            m = _metrics(out[field], c[field], TEND_REL, TEND_ABS_FLOOR)
            rec["fields"][field] = m
            if not m["pass"]:
                failures.append(f"case {case} {field}: max_abs={m['max_abs']:.3e} max_rel={m['max_rel']:.3e}")

        for field in MOMENTUM_FIELDS:
            m = _metrics(out[field], c[field], MOM_REL, MOM_ABS_FLOOR)
            rec["fields"][field] = m
            if not m["pass"]:
                failures.append(f"case {case} {field}: max_abs={m['max_abs']:.3e} max_rel={m['max_rel']:.3e}")

        rain_abs = abs(float(out["RAINCV"]) - float(s["RAINCV"]))
        rain_tol = max(RAINCV_REL * abs(float(s["RAINCV"])), RAINCV_ABS)
        rec["rainc_acc"] = {
            "oracle": float(s["RAINCV"]),
            "jax": float(out["RAINCV"]),
            "max_abs": float(rain_abs),
            "tolerance": float(rain_tol),
            "pass": bool(rain_abs <= rain_tol),
        }
        if rain_abs > rain_tol:
            failures.append(f"case {case} RAINCV: max_abs={rain_abs:.3e} tol={rain_tol:.3e}")
        rec["pass"] = not any(f"case {case} " in f for f in failures)
        case_records.append(rec)

    wrf_tiedtke = Path("/home/user/src/wrf_pristine/WRF/phys/module_cu_tiedtke.F")
    wrf_constants = Path("/home/user/src/wrf_pristine/WRF/share/module_model_constants.F")
    savepoint_files = sorted(SAVE.glob("tiedtke_case_*.json"))
    report = {
        "schema": "wrf-v060-tiedtke-savepoint-parity-report-v1",
        "verdict": "PASS" if not failures else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": os.environ.get("JAX_PLATFORMS", ""),
        "elapsed_seconds": time.perf_counter() - started,
        "oracle": {
            "source": "single-column Fortran driver linked against pristine WRF sources",
            "wrf_tiedtke_source": str(wrf_tiedtke),
            "wrf_tiedtke_sha256": _sha256(wrf_tiedtke),
            "wrf_constants_source": str(wrf_constants),
            "wrf_constants_sha256": _sha256(wrf_constants),
            "unmodified_wrf_module": True,
            "full_wrf_exe": False,
            "full_wrf_exe_note": "module-level WRF oracle, not a full real.exe/wrf.exe integration run",
            "generation_command": "taskset -c 0-3 proofs/v060/oracle/build_and_run.sh",
        },
        "implementation": {
            "module": "src/gpuwrf/physics/cumulus_tiedtke.py",
            "path": "CPU Python/JAX single-column transcription returning PhysicsStepResult",
            "jit_vmap_native_kernel": False,
            "gpu_performance_claim": False,
        },
        "predeclared_tolerances": {
            "tendency_max_relative": TEND_REL,
            "tendency_abs_floor": TEND_ABS_FLOOR,
            "momentum_max_relative": MOM_REL,
            "momentum_abs_floor": MOM_ABS_FLOOR,
            "raincv_max_relative": RAINCV_REL,
            "raincv_abs": RAINCV_ABS,
            "categorical": "exact KTYPE",
        },
        "variables": [
            {"name": name, "units": "K s-1 or kg kg-1 s-1", "shape": "column_k", "dtype": "float64", "absolute_tolerance_floor": TEND_ABS_FLOOR, "relative_tolerance": TEND_REL}
            for name in TENDENCY_FIELDS
        ] + [
            {"name": name, "units": "m s-2", "shape": "column_k", "dtype": "float64", "absolute_tolerance_floor": MOM_ABS_FLOOR, "relative_tolerance": MOM_REL}
            for name in MOMENTUM_FIELDS
        ] + [
            {"name": "RAINCV", "units": "mm per cumulus call", "shape": "scalar", "dtype": "float64", "absolute_tolerance": RAINCV_ABS, "relative_tolerance": RAINCV_REL},
            {"name": "KTYPE", "units": "category", "shape": "scalar", "dtype": "int", "absolute_tolerance": 0, "relative_tolerance": 0},
        ],
        "files": {str(path.relative_to(ROOT)): _sha256(path) for path in savepoint_files},
        "cases": case_records,
        "failures": failures,
        "known_limitations": [
            "The oracle is real WRF source at module level, not full wrf.exe.",
            "The implementation is not yet a jit/vmap GPU-resident production kernel.",
            "cu_physics=16 New Tiedtke is interface-compatible but not separately gated by a distinct WRF source path in this report.",
        ],
    }
    # The committed lane report is AUTHORITATIVE. By default this test ASSERTS the
    # parity verdict without overwriting it (running pytest must not silently
    # regenerate a committed proof). Set GPUWRF_WRITE_PARITY_REPORT=1 to explicitly
    # regenerate the report (the intended, deliberate proof-refresh action).
    if os.environ.get("GPUWRF_WRITE_PARITY_REPORT") == "1":
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")

    assert not failures, "; ".join(failures)
