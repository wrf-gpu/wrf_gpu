#!/usr/bin/env python3
"""Generate the v0.6.0 Grell-Freitas savepoint parity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "grellfreitas_savepoint_parity_report.json"

TENDENCY_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
CASES = (1, 2, 3, 4, 5)
PREDECLARED_TOLERANCES = {
    "trigger": "exact deep/shallow/nontrigger categorical match",
    "tendency_max_relative": 5.0e-2,
    "tendency_abs_floor": 1.0e-8,
    "raincv_max_relative": 5.0e-2,
    "raincv_abs": 1.0e-4,
    "ktop_deep_levels": 1,
    "shallow_level_indices": 1,
    "scale_pair": "fine-grid case must produce less RAINCV than coarse-grid case; ratio tolerance 0.20",
}

WRF_SOURCE_PATHS = (
    "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_deep.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_sh.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_wrfdrv.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_gfs_physcons.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_gfs_machine.F",
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_case(case_id: int) -> dict:
    path = SAVEPOINT_DIR / f"gf_case_{case_id}.json"
    with path.open() as fh:
        return json.load(fh)


def field_error(candidate, oracle):
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    diff = np.abs(candidate - oracle)
    max_abs = float(np.max(diff))
    scale = max(float(np.max(np.abs(oracle))), PREDECLARED_TOLERANCES["tendency_abs_floor"])
    max_rel = max_abs / scale
    passed = (max_rel <= PREDECLARED_TOLERANCES["tendency_max_relative"]) or (
        max_abs <= PREDECLARED_TOLERANCES["tendency_abs_floor"]
    )
    return {"max_abs": max_abs, "max_rel": float(max_rel), "pass": bool(passed)}


def scalar_error(candidate, oracle, rel_tol, abs_tol):
    candidate = float(candidate)
    oracle = float(oracle)
    max_abs = abs(candidate - oracle)
    scale = max(abs(oracle), abs_tol)
    max_rel = max_abs / scale
    return {
        "oracle": oracle,
        "jax": candidate,
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
        "tolerance_abs": float(abs_tol),
        "tolerance_rel": float(rel_tol),
        "pass": bool(max_abs <= max(abs_tol, rel_tol * abs(oracle))),
    }


def oracle_triggers(scalars, columns):
    max_tendency = max(max(abs(x) for x in columns.get(field, [0.0])) for field in TENDENCY_FIELDS)
    deep = int(scalars.get("KTOP_DEEP", 0)) > 0 and float(scalars.get("RAINCV", 0.0)) > 0.0
    shallow = float(scalars.get("XMB_SHALLOW", 0.0)) > 0.0 or int(scalars.get("KTOP_SHALLOW", 0)) > 0
    any_trigger = deep or shallow or max_tendency > PREDECLARED_TOLERANCES["tendency_abs_floor"]
    return {"deep": bool(deep), "shallow": bool(shallow), "any": bool(any_trigger)}


def build_report() -> dict:
    from gpuwrf.physics import cumulus_grell_freitas as gf

    started = time.perf_counter()
    case_reports = []
    failures: list[str] = []
    scale_pair = {}

    for case_id in CASES:
        data = load_case(case_id)
        scalars = data["scalars"]
        cols = data["columns"]
        out = gf.grell_freitas_column(
            np.asarray(cols["T"], dtype=np.float64),
            np.asarray(cols["QV"], dtype=np.float64),
            np.asarray(cols["P"], dtype=np.float64),
            np.asarray(cols["DZ"], dtype=np.float64),
            np.asarray(cols["RHO"], dtype=np.float64),
            np.asarray(cols["W"], dtype=np.float64),
            dt=float(scalars["DT"]),
            dx=float(scalars["DX"]),
            pi_exner=np.asarray(cols["PI"], dtype=np.float64),
            u=np.asarray(cols["U"], dtype=np.float64),
            v=np.asarray(cols["V"], dtype=np.float64),
            rthblten=np.asarray(cols["RTHBLTEN"], dtype=np.float64),
            rqvblten=np.asarray(cols["RQVBLTEN"], dtype=np.float64),
            kpbl=int(scalars["KPBL"]),
            hfx=float(scalars["HFX"]),
            qfx=float(scalars["QFX"]),
            xland=float(scalars["XLAND"]),
        )
        out_np = {k: np.asarray(v) for k, v in out.items()}
        oracle_cat = oracle_triggers(scalars, cols)
        jax_cat = {
            "deep": bool(out_np["TRIGGER_DEEP"]),
            "shallow": bool(out_np["TRIGGER_SHALLOW"]),
            "any": bool(out_np["TRIGGER_DEEP"] or out_np["TRIGGER_SHALLOW"]),
        }
        categorical = {
            "oracle": oracle_cat,
            "jax": jax_cat,
            "pass": oracle_cat == jax_cat,
        }
        if not categorical["pass"]:
            failures.append(f"case {case_id} trigger mismatch oracle={oracle_cat} jax={jax_cat}")

        fields = {}
        compare_tendencies = oracle_cat["any"] and jax_cat["any"]
        for field in TENDENCY_FIELDS:
            if compare_tendencies or oracle_cat["any"] or jax_cat["any"]:
                fields[field] = field_error(out_np[field], cols[field])
                if not fields[field]["pass"]:
                    failures.append(
                        f"case {case_id} {field}: max_abs={fields[field]['max_abs']:.3e} "
                        f"max_rel={fields[field]['max_rel']:.3e}"
                    )
            else:
                fields[field] = {"max_abs": 0.0, "max_rel": 0.0, "pass": True}

        rain = scalar_error(
            out_np["RAINCV"],
            scalars["RAINCV"],
            PREDECLARED_TOLERANCES["raincv_max_relative"],
            PREDECLARED_TOLERANCES["raincv_abs"],
        )
        if not rain["pass"]:
            failures.append(f"case {case_id} RAINCV: max_abs={rain['max_abs']:.3e} max_rel={rain['max_rel']:.3e}")

        ktop = {
            "oracle": int(scalars["KTOP_DEEP"]),
            "jax": int(out_np["KTOP_DEEP"]),
            "pass": bool(abs(int(scalars["KTOP_DEEP"]) - int(out_np["KTOP_DEEP"])) <= PREDECLARED_TOLERANCES["ktop_deep_levels"]),
        }
        if oracle_cat["deep"] and not ktop["pass"]:
            failures.append(f"case {case_id} KTOP_DEEP oracle={ktop['oracle']} jax={ktop['jax']}")

        shallow_levels = {}
        for key in ("K22_SHALLOW", "KBCON_SHALLOW", "KTOP_SHALLOW"):
            shallow_levels[key] = {
                "oracle": int(scalars[key]),
                "jax": int(out_np[key]),
                "pass": bool(abs(int(scalars[key]) - int(out_np[key])) <= PREDECLARED_TOLERANCES["shallow_level_indices"]),
            }
            if oracle_cat["shallow"] and not shallow_levels[key]["pass"]:
                failures.append(
                    f"case {case_id} {key} oracle={shallow_levels[key]['oracle']} jax={shallow_levels[key]['jax']}"
                )

        rec = {
            "case": case_id,
            "regime": scalars["REGIME"],
            "categorical": categorical,
            "fields": fields,
            "raincv": rain,
            "ktop_deep": ktop,
            "shallow": {
                "xmb": scalar_error(out_np["XMB_SHALLOW"], scalars["XMB_SHALLOW"], 5.0e-2, 1.0e-5),
                "levels": shallow_levels,
            },
            "scale_factor": float(out_np["SCALE_FACTOR"]),
            "scale_normalized": float(out_np["SCALE_NORMALIZED"]),
            "qv_sink_mm": float(out_np["QVSINK_MM"]),
        }
        case_reports.append(rec)

    by_case = {rec["case"]: rec for rec in case_reports}
    coarse_rain = by_case[4]["raincv"]["oracle"]
    fine_rain = by_case[5]["raincv"]["oracle"]
    jax_coarse_rain = by_case[4]["raincv"]["jax"]
    jax_fine_rain = by_case[5]["raincv"]["jax"]
    oracle_ratio = fine_rain / coarse_rain if coarse_rain else 0.0
    jax_ratio = jax_fine_rain / jax_coarse_rain if jax_coarse_rain else 0.0
    ratio_pass = (fine_rain < coarse_rain) and (jax_fine_rain < jax_coarse_rain) and abs(jax_ratio - oracle_ratio) <= 0.20
    scale_pair = {
        "coarse_case": 4,
        "fine_case": 5,
        "oracle_fine_to_coarse_raincv": float(oracle_ratio),
        "jax_fine_to_coarse_raincv": float(jax_ratio),
        "pass": bool(ratio_pass),
    }
    if not ratio_pass:
        failures.append(
            f"scale-aware pair ratio oracle={oracle_ratio:.3e} jax={jax_ratio:.3e}"
        )

    source_checksums = {str(Path(path)): sha256(path) for path in WRF_SOURCE_PATHS}
    source_checksums[str(ROOT / "proofs" / "v060" / "oracle" / "gf_oracle_driver.f90")] = sha256(
        ROOT / "proofs" / "v060" / "oracle" / "gf_oracle_driver.f90"
    )
    source_checksums[str(ROOT / "src" / "gpuwrf" / "physics" / "cumulus_grell_freitas.py")] = sha256(
        ROOT / "src" / "gpuwrf" / "physics" / "cumulus_grell_freitas.py"
    )

    report = {
        "schema": "gpuwrf.v060.grellfreitas_savepoint_parity.v1",
        "scheme": "Grell-Freitas scale-aware cumulus",
        "cu_physics": 3,
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "predeclared_tolerances": PREDECLARED_TOLERANCES,
        "cases": case_reports,
        "scale_aware_pair": scale_pair,
        "oracle": {
            "full_wrf_exe_run": False,
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
            "source": "unmodified WRF GF modules from /home/enric/src/wrf_pristine/WRF/phys compiled into a standalone single-column driver",
            "note": (
                "This is a real WRF-module oracle, not a JAX self-compare. It is not a full coupled wrf.exe run. "
                "Pristine WRF GFDRV does not take cugd_* arrays; those S0 carry members are reported as adapter carry only."
            ),
            "source_checksums_sha256": source_checksums,
        },
        "commands": {
            "oracle": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
            "report": "JAX_PLATFORMS=cpu taskset -c 0-3 python proofs/v060/run_grellfreitas_parity.py --fail-on-parity-fail",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-parity-fail", action="store_true")
    args = parser.parse_args()
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['verdict']}: wrote {REPORT_PATH}")
    if report["failures"]:
        for failure in report["failures"][:20]:
            print(f"  {failure}")
    if args.fail_on_parity_fail and report["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
