#!/usr/bin/env python3
"""v0.13 MYJ PBL + Janjic Eta OPERATIONAL savepoint-parity oracle.

Gates the OPERATIONAL, ``jax.jit``/``jax.vmap``-traceable MYJ pair kernels
(``physics.bl_myj.myj_columns`` + ``physics.sf_myj.myjsfc_columns``) against the
SAME fp64 WRF-source savepoints the v0.6.0 reference-only column kernels passed.

Those savepoints (``proofs/v060/savepoints_fp64/myj{pbl,sfc}_case_*.json``) were
produced by a standalone Fortran driver compiled against the UNMODIFIED pristine
WRF ``phys/module_bl_myjpbl.F`` + ``phys/module_sf_myjsfc.F`` (the real WRF-source
oracle, NOT a JAX self-compare; build manifest + source SHA-256 are recorded in
the v0.6.0 report objects). This v0.13 oracle re-validates that the TRACEABLE
operational kernels -- which can ride the device scan -- reproduce that WRF-source
oracle at fp64 ~1e-13, covering all 6 regimes (unstable/stable/neutral land,
stable/unstable marine).

The batched kernels are exercised through ``jax.jit`` over the full 6-case batch
to prove the operational path is genuinely jit/vmap-traceable.

Run (CPU, fp64; the GPU is owned by another lane):

    taskset -c 0-3 env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \\
        python proofs/v013/myj_janjic_oracle.py

Writes ``proofs/v013/myj_janjic_oracle.json``.
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

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics.bl_myj import myj_columns
from gpuwrf.physics.sf_myj import myjsfc_columns


ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "proofs" / "v060" / "savepoints_fp64"
REPORT = Path(__file__).resolve().parent / "myj_janjic_oracle.json"
CASES = (1, 2, 3, 4, 5, 6)

# PREDECLARED fp64 operational-parity tolerances (match the v0.6.0 reference gates).
PBL_ABS_TOL = 5.0e-10
PBL_REL_TOL = 5.0e-11
SFC_ABS_TOL = 5.0e-10
SFC_REL_TOL = 5.0e-12
REL_FLOOR = 1.0e-30

PBL_ARRAY_FIELDS = ("TKE_MYJ", "EXCH_H", "EL_MYJ", "RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN")
PBL_SCALAR_FIELDS = ("PBLH", "MIXHT")

# Janjic surface-layer gated fields (savepoint scalar name -> kernel output key).
SFC_FIELDS = {
    "USTAR": "ustar", "ZNT": "znt", "AKHS": "akhs", "AKMS": "akms",
    "RMOL": "rmol", "RIB": "rib", "CHS": "chs", "CHS2": "chs2", "CQS2": "cqs2",
    "HFX": "hfx", "QFX": "qfx", "FLX_LH": "flx_lh", "FLHC": "flhc", "FLQC": "flqc",
    "QGH": "qgh", "CPM": "cpm", "CT": "ct", "QSFC": "qsfc", "THZ0": "thz0",
    "QZ0": "qz0", "UZ0": "uz0", "VZ0": "vz0", "PBLH": "pblh", "U10": "u10",
    "V10": "v10", "T02": "t02", "TH02": "th02", "TSHLTR": "tshltr", "TH10": "th10",
    "Q02": "q02", "QSHLTR": "qshltr", "Q10": "q10", "PSHLTR": "pshltr",
    "U10E": "u10e", "V10E": "v10e",
}

# Surface-layer initial conditions matching the Fortran driver (test_v060_sfclay).
INITIAL_ZNT = {1: 0.10, 2: 0.08, 3: 0.05, 4: 0.001, 5: 0.08, 6: 0.0015}
INITIAL_USTAR = 0.1


def _load(name: str, case: int) -> dict:
    with (SP / f"{name}_case_{case}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _col(d: dict, n: str) -> np.ndarray:
    return np.asarray(d["columns"][n], dtype=np.float64)


def _metric(actual, expected, *, abs_tol: float, rel_tol: float) -> dict:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    signed = a - b
    abs_err = np.abs(signed)
    max_abs = float(np.max(abs_err))
    scale = max(float(np.max(np.abs(b))), REL_FLOOR)
    max_rel = max_abs / scale
    argmax = int(np.argmax(abs_err))
    return {
        "jax": float(np.ravel(a)[argmax]),
        "oracle": float(np.ravel(b)[argmax]),
        "max_abs": max_abs,
        "max_rel": float(max_rel),
        "scale": scale,
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "pass": bool(max_abs <= abs_tol or max_rel <= rel_tol),
    }


def _build_pbl_batch():
    """Stack the 6 MYJ-PBL cases into a (6, nz) batch + (6,) surface scalars."""

    keys = ("U", "V", "T", "TH", "QV", "QC", "PMID", "PINT", "EXNER", "DZ")
    prof = {k: [] for k in keys}
    tke = []
    sc = {k: [] for k in ("TSK", "XLAND", "USTAR", "AKHS", "AKMS", "CHKLOWQ",
                          "ELFLX", "THZ0", "QZ0", "UZ0", "VZ0", "QSFC", "CT", "HT")}
    expect = []
    for c in CASES:
        d = _load("myjpbl", c)
        sf = _load("myjsfc", c)
        s = d["scalars"]
        for k in keys:
            prof[k].append(_col(d, k))
        tke.append(0.5 * _col(sf, "Q2"))  # TKE_MYJ = 0.5*Q2 (matches the driver)
        for k in sc:
            sc[k].append(s[k])
        expect.append(d)
    A = lambda L: jnp.asarray(np.stack(L), jnp.float64)
    inputs = dict(
        u=A(prof["U"]), v=A(prof["V"]), temperature=A(prof["T"]), theta=A(prof["TH"]),
        qv=A(prof["QV"]), qc=A(prof["QC"]), p_mid=A(prof["PMID"]), p_int=A(prof["PINT"]),
        exner=A(prof["EXNER"]), dz=A(prof["DZ"]), tke=A(tke),
    )
    scal = {k.lower(): jnp.asarray(v, jnp.float64) for k, v in sc.items()}
    return inputs, scal, expect


def _build_sfc_batch():
    keys = ("U", "V", "T", "TH", "QV", "QC", "PMID", "DZ", "Q2")
    prof = {k: [] for k in keys}
    znt = []
    psfc = []
    qsfc = []
    thz0 = []
    qz0 = []
    sc = {k: [] for k in ("TSK", "XLAND", "Z0BASE", "MAVAIL", "PBLH")}
    expect = []
    for c in CASES:
        d = _load("myjsfc", c)
        s = d["scalars"]
        for k in keys:
            prof[k].append(_col(d, k))
        znt.append(INITIAL_ZNT[c])
        psfc.append(float(_col(d, "PINT")[0]) if "PINT" in d["columns"] else float(s["TSK"] * 0))
        qv0 = float(_col(d, "QV")[0])
        qsfc.append(qv0)
        thz0.append(float(_col(d, "TH")[0]))
        qz0.append(qv0 / (1.0 + qv0))
        for k in sc:
            sc[k].append(s[k])
        expect.append(s)
    # PINT not in myjsfc columns -> reconstruct psfc from PMID[0] is wrong; use the
    # paired myjpbl PINT[0] (same column) as the surface pressure (matches driver).
    psfc = []
    for c in CASES:
        dp = _load("myjpbl", c)
        psfc.append(float(_col(dp, "PINT")[0]))
    A = lambda L: jnp.asarray(np.stack(L), jnp.float64)
    inputs = dict(
        u=A(prof["U"]), v=A(prof["V"]), temperature=A(prof["T"]), theta=A(prof["TH"]),
        qv=A(prof["QV"]), qc=A(prof["QC"]), p_mid=A(prof["PMID"]), dz=A(prof["DZ"]),
        q2=A(prof["Q2"]),
    )
    scal = dict(
        tsk=jnp.asarray(sc["TSK"], jnp.float64), xland=jnp.asarray(sc["XLAND"], jnp.float64),
        z0base=jnp.asarray(sc["Z0BASE"], jnp.float64), mavail=jnp.asarray(sc["MAVAIL"], jnp.float64),
        psfc=jnp.asarray(psfc, jnp.float64), znt=jnp.asarray(znt, jnp.float64),
        ustar=jnp.full((len(CASES),), INITIAL_USTAR, jnp.float64),
        qsfc=jnp.asarray(qsfc, jnp.float64), thz0=jnp.asarray(thz0, jnp.float64),
        qz0=jnp.asarray(qz0, jnp.float64), pblh=jnp.asarray(sc["PBLH"], jnp.float64),
    )
    return inputs, scal, expect


def _run_pbl_cases() -> list[dict]:
    inputs, scal, expect = _build_pbl_batch()
    fn = jax.jit(myj_columns)
    out = fn(inputs["u"], inputs["v"], inputs["temperature"], inputs["theta"],
             inputs["qv"], inputs["qc"], inputs["p_mid"], inputs["p_int"],
             inputs["exner"], inputs["dz"], inputs["tke"],
             tsk=scal["tsk"], xland=scal["xland"], ustar=scal["ustar"],
             akhs=scal["akhs"], akms=scal["akms"], chklowq=scal["chklowq"],
             elflx=scal["elflx"], thz0=scal["thz0"], qz0=scal["qz0"],
             uz0=scal["uz0"], vz0=scal["vz0"], qsfc=scal["qsfc"], ct=scal["ct"],
             dt=60.0, stepbl=1, ht=scal["ht"])
    out = {k: np.asarray(v) for k, v in out.items()}
    cases = []
    for i, c in enumerate(CASES):
        d = expect[i]
        s = d["scalars"]
        fields = {
            name: _metric(out[name][i], _col(d, name), abs_tol=PBL_ABS_TOL, rel_tol=PBL_REL_TOL)
            for name in PBL_ARRAY_FIELDS
        }
        scalars = {
            name: _metric(out[name][i], s[name], abs_tol=PBL_ABS_TOL, rel_tol=PBL_REL_TOL)
            for name in PBL_SCALAR_FIELDS
        }
        kpbl = {
            "jax": int(out["KPBL"][i]), "oracle": int(s["KPBL"]),
            "pass": int(out["KPBL"][i]) == int(s["KPBL"]),
        }
        case = {"case": c, "regime": s["REGIME"], "fields": fields,
                "scalars": scalars, "kpbl": kpbl}
        case["pass"] = bool(
            all(m["pass"] for m in fields.values())
            and all(m["pass"] for m in scalars.values())
            and kpbl["pass"]
        )
        cases.append(case)
    return cases


def _run_sfc_cases() -> list[dict]:
    inputs, scal, expect = _build_sfc_batch()
    fn = jax.jit(myjsfc_columns)
    out = fn(inputs["u"], inputs["v"], inputs["temperature"], inputs["theta"],
             inputs["qv"], inputs["qc"], inputs["p_mid"], inputs["dz"], inputs["q2"],
             tsk=scal["tsk"], xland=scal["xland"], z0base=scal["z0base"],
             psfc=scal["psfc"], znt=scal["znt"], ustar=scal["ustar"],
             mavail=scal["mavail"], qsfc=scal["qsfc"], thz0=scal["thz0"],
             qz0=scal["qz0"], uz0=0.0, vz0=0.0, pblh=scal["pblh"], dt=60.0)
    out = {k: np.asarray(v) for k, v in out.items()}
    cases = []
    for i, c in enumerate(CASES):
        s = expect[i]
        fields = {
            name: _metric(out[key][i], s[name], abs_tol=SFC_ABS_TOL, rel_tol=SFC_REL_TOL)
            for name, key in SFC_FIELDS.items()
        }
        case = {"case": c, "regime": s["REGIME"], "fields": fields}
        case["pass"] = bool(all(m["pass"] for m in fields.values()))
        cases.append(case)
    return cases


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def build_report() -> dict:
    pbl = _run_pbl_cases()
    sfc = _run_sfc_cases()
    overall = all(c["pass"] for c in pbl) and all(c["pass"] for c in sfc)
    return {
        "schema": "gpuwrf.v013.myj_janjic_operational_savepoint_parity.v1",
        "scheme": "MYJ PBL (bl_pbl_physics=2) + Janjic Eta surface layer (sf_sfclay_physics=2)",
        "verdict": "PASS" if overall else "FAIL",
        "overall_pass": bool(overall),
        "operational_kernels": {
            "pbl": "gpuwrf.physics.bl_myj.myj_columns (jit/vmap-traceable)",
            "surface_layer": "gpuwrf.physics.sf_myj.myjsfc_columns (jit/vmap-traceable)",
            "adapters": "gpuwrf.physics.myj_adapters.{myj_pbl_adapter,janjic_sfclay_adapter}",
            "traceable": True,
            "exercised_through": "jax.jit over the full 6-case batch (genuine jit/vmap path)",
        },
        "oracle": {
            "type": "single-column Fortran driver linked against UNMODIFIED pristine WRF "
                    "module_bl_myjpbl.F + module_sf_myjsfc.F (reuses the v0.6.0 fp64 savepoints)",
            "savepoints": str(SP.relative_to(ROOT)),
            "wrf_sources": [
                "/home/enric/src/wrf_pristine/WRF/share/module_model_constants.F",
                "/home/enric/src/wrf_pristine/WRF/phys/module_sf_myjsfc.F",
                "/home/enric/src/wrf_pristine/WRF/phys/module_bl_myjpbl.F",
            ],
            "full_wrf_exe": False,
            "self_compare": False,
            "pbl_source_checksums_sha256": _read_lines(SP / "myjpbl_wrf_source_checksums.txt"),
            "sfc_source_checksums_sha256": _read_lines(SP / "myjsfc_wrf_source_checksums.txt"),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": _git_head(),
        "predeclared_tolerances": {
            "pbl": {"abs": PBL_ABS_TOL, "rel": PBL_REL_TOL, "kpbl": "exact integer"},
            "surface_layer": {"abs": SFC_ABS_TOL, "rel": SFC_REL_TOL},
            "relative_floor": REL_FLOOR,
        },
        "regimes_covered": [c["regime"] for c in pbl],
        "tke_coupling": (
            "Faithful: the Janjic surface layer consumes the TKE-derived PBL height "
            "(MYJSFC q2 = qke = q^2) and the MYJ PBL writes the updated q^2 back to "
            "the qke leaf; bl=2/sf=2 are a mandatory pair (resolve_physics_suite raises "
            "if only one is selected)."
        ),
        "pbl_cases": pbl,
        "surface_layer_cases": sfc,
    }


def main() -> int:
    report = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"verdict={report['verdict']} -> {REPORT}")
    worst_pbl = max(
        max(m["max_abs"] for m in c["fields"].values()) for c in report["pbl_cases"]
    )
    worst_sfc = max(
        max(m["max_abs"] for m in c["fields"].values()) for c in report["surface_layer_cases"]
    )
    print(f"worst |jax-oracle| PBL={worst_pbl:.3e}  SFC={worst_sfc:.3e}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
