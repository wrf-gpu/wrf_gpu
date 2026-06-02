"""Generate proofs/v060/noahclassic_savepoint_parity_report.json.

Runs the JAX Noah-classic port over the WRF SFLX savepoints and records per-field
/ per-column error, regimes, PASS/FAIL vs the predeclared tolerances, oracle
provenance, and honest residuals. Run on CPU:

    JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 XLA_FLAGS=--xla_cpu_use_thunk_runtime=false \
        taskset -c 0-3 python3 proofs/v060/gen_parity_report.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_use_thunk_runtime=false")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.v060.test_noahclassic_parity import _load, _build_inputs, TOLS  # noqa: E402
from gpuwrf.physics.lsm_noah_classic import sflx_step  # noqa: E402

OUT = ROOT / "proofs" / "v060" / "noahclassic_savepoint_parity_report.json"


def main():
    data = _load()
    cols = data["columns"]
    names = [c["name"] for c in cols]
    forcing, params, state, dt, zsoil, sldpth = _build_inputs(cols)
    out = sflx_step(forcing, params, state, dt, zsoil, sldpth)

    scalar = {
        "t1_out": (np.asarray(out.state.t1), [c["wrf"]["t1_out"] for c in cols]),
        "hfx": (np.asarray(out.hfx), [c["wrf"]["flux"]["hfx"] for c in cols]),
        "qfx": (np.asarray(out.qfx), [c["wrf"]["flux"]["qfx"] for c in cols]),
        "lh": (np.asarray(out.lh), [c["wrf"]["flux"]["lh"] for c in cols]),
        "grdflx": (np.asarray(out.grdflx), [c["wrf"]["flux"]["grdflx"] for c in cols]),
        "sneqv": (np.asarray(out.state.sneqv), [c["wrf"]["snow_out"]["sneqv"] for c in cols]),
        "snowh": (np.asarray(out.state.snowh), [c["wrf"]["snow_out"]["snowh"] for c in cols]),
        "sncovr": (np.asarray(out.state.sncovr), [c["wrf"]["snow_out"]["sncovr"] for c in cols]),
        "albedo": (np.asarray(out.albedo), [c["wrf"]["diag"]["albedo"] for c in cols]),
    }
    vec = {
        "stc": (np.asarray(out.state.stc), np.asarray([c["wrf"]["stc_out"] for c in cols])),
        "smc": (np.asarray(out.state.smc), np.asarray([c["wrf"]["smc_out"] for c in cols])),
        "sh2o": (np.asarray(out.state.sh2o), np.asarray([c["wrf"]["sh2o_out"] for c in cols])),
    }

    fields = {}
    overall_pass = True
    for field, (jv, wv) in scalar.items():
        jv = np.asarray(jv); wv = np.asarray(wv, dtype=float)
        tol = TOLS[field]
        thresh = tol["atol"] + tol["rel"] * np.abs(wv)
        abserr = np.abs(jv - wv)
        passed = bool(np.all(abserr <= thresh))
        overall_pass &= passed
        fields[field] = {
            "tol": tol, "max_abs_err": float(np.max(abserr)),
            "pass": passed,
            "per_column": {names[k]: {"jax": float(jv[k]), "wrf": float(wv[k]),
                                      "abs_err": float(abserr[k])} for k in range(len(names))},
        }
    for field, (jv, wv) in vec.items():
        jv = np.asarray(jv); wv = np.asarray(wv, dtype=float)
        tol = TOLS[field]
        thresh = tol["atol"] + tol["rel"] * np.abs(wv)
        abserr = np.abs(jv - wv)
        passed = bool(np.all(abserr <= thresh))
        overall_pass &= passed
        fields[field] = {
            "tol": tol, "max_abs_err": float(np.max(abserr)),
            "pass": passed,
            "per_column_per_layer": {
                names[k]: {"jax": [float(x) for x in jv[k]],
                           "wrf": [float(x) for x in wv[k]],
                           "abs_err": [float(x) for x in abserr[k]]}
                for k in range(len(names))},
        }

    report = {
        "proof": "noahclassic-savepoint-parity (v0.6.0 lane 14, sf_surface_physics=2)",
        "verdict": "PASS" if overall_pass else "FAIL",
        "oracle": {
            "kind": ("external WRF oracle (NOT self-compare): compiled pristine WRF "
                     "module_sf_noahlsm.o SFLX over real Canary d03 land columns"),
            "driver": "proofs/v060/oracle/noahclassic_offline_driver.F90",
            "wrf_source": "/home/enric/src/wrf_pristine/WRF/phys/module_sf_noahlsm.F",
            "savepoints": "proofs/v060/savepoints_noahclassic.json",
            "scope_options": data["scope_options"],
            "regimes": sorted({c["case"] for c in cols}),
            "ncolumns": len(cols),
            "precision_note": ("WRF SFLX is single precision; JAX port is fp64. "
                               "Residual is fp32-vs-fp64 oracle dust for the flux/temperature "
                               "fields, plus a localized top-layer soil-water distribution "
                               "residual (see honest_residuals)."),
        },
        "predeclared_tolerances": TOLS,
        "fields": fields,
        "honest_residuals": _residual_notes(fields),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": report["verdict"],
                      "field_pass": {k: v["pass"] for k, v in fields.items()},
                      "max_abs_err": {k: v["max_abs_err"] for k, v in fields.items()}}, indent=2))


def _residual_notes(fields):
    notes = []
    for fld, info in fields.items():
        if not info["pass"]:
            notes.append(f"{fld}: FAIL, max|err|={info['max_abs_err']:.3e} > tol "
                         f"(atol={info['tol']['atol']}, rel={info['tol']['rel']})")
    if not notes:
        notes.append("all fields within predeclared tolerances")
    return notes


if __name__ == "__main__":
    main()
