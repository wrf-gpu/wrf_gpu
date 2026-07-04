"""v0.18 RUC LSM CPU/fp64 savepoint parity.

The oracle is the staged pristine-WRF single-column RUC driver under
``proofs/v017/oracle/ruclsm``.  This test validates the JAX no-snow land-column
port against the committed fp64 oracle savepoint without touching the shared
scheme registry/catalog wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("GPUWRF_JAX_CACHE", "0")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from gpuwrf.physics.lsm_ruc import RUC_NUM_SOIL_LAYERS, ruc_columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT = ROOT / "proofs" / "v017" / "savepoints" / "ruclsm" / "fp64" / "ruclsm_fp64.json"

# Tight fp64 cross-language tolerances.  The small nonzero residuals are from
# JAX/libm vs gfortran/libm nonlinear solves in VILKA and the coupled surface
# budget, not from relaxed physics logic.
REL_TOL = 1.0e-4
ABS_TOL = {
    "soilt": 2.0e-5,
    "hfx": 2.0e-4,
    "qfx": 5.0e-10,
    "lh": 5.0e-4,
    "grdflx": 3.0e-4,
    "sfcrunoff": 1.0e-8,
    "qsfc": 5.0e-8,
    "qsg": 5.0e-8,
    "qvg": 5.0e-8,
    "sfcevp": 5.0e-7,
    "tso": 2.0e-5,
    "soilmois": 3.0e-5,
    "sh2o": 3.0e-5,
}
DEFAULT_ABS_TOL = 1.0e-10

SCALAR_MAP = {
    "soilt": "SOILT",
    "hfx": "HFX",
    "qfx": "QFX",
    "lh": "LH",
    "grdflx": "GRDFLX",
    "smavail": "SMAVAIL",
    "smmax": "SMMAX",
    "sfcrunoff": "SFCRUNOFF",
    "udrunoff": "UDRUNOFF",
    "qsfc": "QSFC",
    "qsg": "QSG",
    "qvg": "QVG",
    "qcg": "QCG",
    "dew": "DEW",
    "snow": "SNOW",
    "snowh": "SNOWH",
    "alb": "ALB",
    "emiss": "EMISS",
    "znt": "ZNT",
    "lai": "LAI",
    "mavail": "MAVAIL",
    "sfcevp": "SFCEVP",
    "sfcexc": "SFCEXC",
    "tsnav": "TSNAV",
    "soilt1": "SOILT1",
}
PROFILE_MAP = {"tso": "TSO", "soilmois": "SOILMOIS", "sh2o": "SH2O"}


def test_ruc_lsm_fp64_oracle_parity_green() -> None:
    data = json.loads(SAVEPOINT.read_text())
    out = _run_port(data)

    failures = []
    metrics = {}
    for actual_name, oracle_name in SCALAR_MAP.items():
        metrics[actual_name] = _compare(out[actual_name], data["columns"][oracle_name], actual_name)
    for actual_name, oracle_name in PROFILE_MAP.items():
        metrics[actual_name] = _compare(out[actual_name], data["profiles"][oracle_name], actual_name)

    for name, metric in metrics.items():
        if not metric["pass"]:
            failures.append(
                f"{name}: max_abs={metric['max_abs']:.6e} > {metric['abs_tol']:.6e} "
                f"and max_rel={metric['max_rel']:.6e} > {REL_TOL:.6e}"
            )
    assert not failures, "RUC LSM parity failed:\n" + "\n".join(failures)


def _run_port(data: dict):
    cols = data["columns"]
    zsoil = np.asarray([0.0, 0.05, 0.20, 0.40, 1.0, 2.0], dtype=np.float64)
    assert zsoil.size == RUC_NUM_SOIL_LAYERS

    soilt0 = np.asarray(cols["SOILT_IN"], dtype=np.float64)
    tbot = np.asarray(cols["TBOT"], dtype=np.float64)
    sm1 = np.asarray(cols["SOILMOIS_IN_1"], dtype=np.float64)
    # Deep soil moisture values come from ruclsm_oracle_driver.f90::config_regime.
    sm2 = np.asarray([0.25, 0.25, 0.18, 0.36, 0.20], dtype=np.float64)

    tso = soilt0[:, None] + (tbot - soilt0)[:, None] * (zsoil[None, :] / zsoil[-1])
    soilmois = sm1[:, None] + (sm2 - sm1)[:, None] * (zsoil[None, :] / zsoil[-1])

    return ruc_columns(
        gsw=cols["GSW"],
        glw=cols["GLW"],
        emiss=cols["EMISS_IN"],
        tabs=cols["TABS"],
        qv=cols["QV"],
        qc=cols["QC"],
        rho=cols["RHO"],
        p8w=cols["P8W"],
        z3d=cols["Z3D"],
        rainbl=cols["RAINBL"],
        vegfra=cols["VEGFRA"],
        flhc=cols["FLHC"],
        flqc=cols["FLQC"],
        tbot=cols["TBOT"],
        xland=cols["XLAND"],
        mavail=cols["MAVAIL_IN"],
        ivgtyp=cols["IVGTYP"],
        isltyp=cols["ISLTYP"],
        soilt=soilt0,
        tso=tso,
        soilmois=soilmois,
        sh2o=soilmois,
        dt=float(cols["DT"][0]),
        nsteps=int(data["scalars"]["NSTEPS"]),
    )


def _compare(actual, oracle, name: str) -> dict[str, float | bool]:
    actual_arr = np.asarray(actual, dtype=np.float64)
    oracle_arr = np.asarray(oracle, dtype=np.float64)
    max_abs = float(np.max(np.abs(actual_arr - oracle_arr)))
    scale = max(float(np.max(np.abs(oracle_arr))), 1.0e-12)
    max_rel = max_abs / scale
    abs_tol = ABS_TOL.get(name, DEFAULT_ABS_TOL)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "abs_tol": abs_tol,
        "pass": bool(max_abs <= abs_tol or max_rel <= REL_TOL),
    }
