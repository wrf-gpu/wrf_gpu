#!/usr/bin/env python3
"""Validate the JAX Dudhia shortwave port against independent WRF savepoints.

The oracle is the real WRF ``phys/module_ra_sw.F:SWRAD`` source compiled by
``proofs/v060/oracle/dudhia_build_and_run.sh``.  The JAX port is never used to
create the reference.  Canonical gating is against WRF default REAL*4
savepoints; an additional fp64 build of the same unmodified source is recorded
as a precision-dust audit.

The kernel emits the per-layer TEMPERATURE heating rate (K/s); WRF SWRAD divides
that by the Exner factor pi to form the theta tendency RTHRATEN it writes back.
We compare in RTHRATEN space (jax_rate/pi vs oracle RTHRATEN) AND report the
surface net SW flux GSW.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics.ra_sw_dudhia import (  # noqa: E402
    DudhiaSWColumnState,
    solve_dudhia_sw_column,
)

SAVE_FP32 = HERE / "savepoints"
SAVE_FP64 = HERE / "savepoints_fp64"
REPORT = HERE / "dudhia_sw_savepoint_parity_report.json"

# PREDECLARED TOLERANCES, frozen before any comparison was run.
# RTHRATEN heating rates are O(1e-4 K/s); GSW is O(1e2 W/m^2). The Dudhia
# SWPARA kernel runs in REAL*4 in WRF, so the canonical fp32 oracle carries
# single-precision dust through the long sequential per-layer accumulation.
PREDECLARED_TOL = {
    "rthraten_abs": 5.0e-9,     # K/s, absolute floor (near-zero layers)
    "rthraten_rel": 5.0e-4,     # relative on the column max heating rate
    "gsw_rel": 5.0e-4,          # relative on surface net SW flux
    "gsw_abs": 5.0e-3,          # W/m^2 absolute floor (night/near-zero)
}

CASE_IDS = (1, 2, 3, 4, 5, 6, 7)


def col(d: dict, name: str) -> np.ndarray:
    return np.asarray(d["columns"][name], dtype=np.float64)


def scalar(d: dict, name: str) -> float:
    return float(d["scalars"][name])


def run_jax_for(d: dict) -> tuple[np.ndarray, float]:
    n = len(col(d, "T"))

    def c(name):
        return jnp.asarray(col(d, name)[None, :])

    state = DudhiaSWColumnState(
        T=c("T"), p=c("P"), qv=c("QV"), qc=c("QC"), qr=c("QR"),
        qi=c("QI"), qs=c("QS"), qg=c("QG"), dz=c("DZ"),
        coszen=jnp.asarray([scalar(d, "COSZEN")]),
        albedo=jnp.asarray([scalar(d, "ALBEDO")]),
        solcon=jnp.asarray([scalar(d, "SOLCON")]),
        icloud=int(scalar(d, "ICLOUD")),
    )
    out = solve_dudhia_sw_column(state)
    pi = col(d, "PI")
    # JAX returns dT/dt; WRF RTHRATEN = (dT/dt)/pi.
    rthraten = np.asarray(out.heating_rate)[0] / pi
    return rthraten, float(np.asarray(out.gsw)[0])


def compare_case(save_dir: Path, cid: int) -> tuple[bool, dict]:
    with open(save_dir / f"dudhia_case_{cid}.json", encoding="utf-8") as fh:
        d = json.load(fh)
    jax_rth, jax_gsw = run_jax_for(d)
    oracle_rth = col(d, "RTHRATEN")
    oracle_gsw = scalar(d, "GSW")

    scale = max(float(np.max(np.abs(oracle_rth))), PREDECLARED_TOL["rthraten_abs"])
    absdiff = np.abs(jax_rth - oracle_rth)
    max_abs = float(np.max(absdiff))
    max_rel = float(max_abs / scale)
    rth_ok = (max_rel <= PREDECLARED_TOL["rthraten_rel"]) or (
        max_abs <= PREDECLARED_TOL["rthraten_abs"]
    )

    gsw_tol = max(PREDECLARED_TOL["gsw_rel"] * abs(oracle_gsw), PREDECLARED_TOL["gsw_abs"])
    gsw_err = abs(jax_gsw - oracle_gsw)
    gsw_ok = gsw_err <= gsw_tol

    passed = bool(rth_ok and gsw_ok)
    return passed, {
        "label": d["scalars"]["REGIME"],
        "coszen": oracle_gsw if False else scalar(d, "COSZEN"),
        "full_wrf_exe": int(scalar(d, "FULL_WRF_EXE")),
        "fields": {
            "RTHRATEN": {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "scale": scale,
                "tol_abs": PREDECLARED_TOL["rthraten_abs"],
                "tol_rel": PREDECLARED_TOL["rthraten_rel"],
                "pass": bool(rth_ok),
            },
            "GSW": {
                "jax": jax_gsw,
                "oracle": oracle_gsw,
                "abs_err": gsw_err,
                "tol": gsw_tol,
                "pass": bool(gsw_ok),
            },
        },
        "pass": passed,
    }


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    canonical_cases = {}
    fp64_cases = {}
    canonical_pass = True
    fp64_pass = True

    for cid in CASE_IDS:
        ok, res = compare_case(SAVE_FP32, cid)
        canonical_cases[str(cid)] = res
        canonical_pass = canonical_pass and ok
        status = "PASS" if ok else "FAIL"
        f = res["fields"]
        print(
            f"FP32 CASE {cid} {res['label']:32s} coszen={res['coszen']:.2f} -> {status} | "
            f"RTH max_abs={f['RTHRATEN']['max_abs']:.3e} rel={f['RTHRATEN']['max_rel']:.3e} | "
            f"GSW jax={f['GSW']['jax']:.4f} oracle={f['GSW']['oracle']:.4f} err={f['GSW']['abs_err']:.3e}"
        )

    print()
    for cid in CASE_IDS:
        ok, res = compare_case(SAVE_FP64, cid)
        fp64_cases[str(cid)] = res
        fp64_pass = fp64_pass and ok
        f = res["fields"]
        status = "PASS" if ok else "FAIL"
        print(
            f"FP64 CASE {cid} {res['label']:32s} -> {status} | "
            f"RTH rel={f['RTHRATEN']['max_rel']:.3e} | GSW err={f['GSW']['abs_err']:.3e}"
        )

    report = {
        "scheme": "Dudhia shortwave (ra_sw_physics=1)",
        "verdict": "PASS" if (canonical_pass and fp64_pass) else "FAIL",
        "overall_pass": bool(canonical_pass and fp64_pass),
        "canonical_fp32_pass": bool(canonical_pass),
        "fp64_precision_audit_pass": bool(fp64_pass),
        "oracle": {
            "source": "$WRF_PRISTINE_ROOT/phys/module_ra_sw.F",
            "entry": "SWRAD -> SWPARA (Stephens 1984 broadband shortwave)",
            "source_unmodified": True,
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled the unmodified WRF module_ra_sw.F "
                "and called the public SWRAD entry; not a full wrf.exe savepoint."
            ),
            "fp32_savepoints": "proofs/v060/savepoints (dudhia_case_*.json)",
            "fp64_savepoints": "proofs/v060/savepoints_fp64 (dudhia_case_*.json)",
            "fp32_source_checksums": read_text_if_present(SAVE_FP32 / "dudhia_wrf_source_checksums.txt"),
            "fp64_source_checksums": read_text_if_present(SAVE_FP64 / "dudhia_wrf_source_checksums.txt"),
            "fp32_build_manifest": read_text_if_present(SAVE_FP32 / "dudhia_build_manifest.txt"),
            "fp64_build_manifest": read_text_if_present(SAVE_FP64 / "dudhia_build_manifest.txt"),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": git_head(),
        "predeclared_tolerances": PREDECLARED_TOL,
        "comparison_space": (
            "RTHRATEN (theta tendency) = jax_heating_rate / pi vs oracle RTHRATEN; "
            "GSW surface net downward SW flux compared directly."
        ),
        "edge_cases_covered": [
            "clear-sky high sun (land albedo)",
            "clear-sky low sun (high zenith)",
            "nighttime / zero coszen (must give GSW=0, RTHRATEN=0)",
            "thick warm liquid cloud high sun",
            "ice+snow cloud mid sun marine (low albedo)",
            "mixed snow/graupel/cloud high sun",
            "terminator low sun humid",
        ],
        "cases": canonical_cases,
        "fp64_audit_cases": fp64_cases,
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nOVERALL:", report["verdict"])
    print("wrote", REPORT)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
