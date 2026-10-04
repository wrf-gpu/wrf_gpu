#!/usr/bin/env python3
"""Validate the JAX Kessler port against independent WRF Fortran savepoints.

The oracle is the real WRF ``phys/module_mp_kessler.F`` source compiled by
``proofs/v060/oracle/build_and_run.sh``. The JAX port is never used to create
the reference. Canonical gating is against WRF default REAL*4 savepoints; an
additional fp64 build of the same unmodified source is recorded as a precision
dust audit.
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

from gpuwrf.physics.microphysics_kessler import kessler_run  # noqa: E402


SAVE_FP32 = HERE / "savepoints"
SAVE_FP64 = HERE / "savepoints_fp64"
REPORT = HERE / "kessler_savepoint_parity_report.json"

# PREDECLARED TOLERANCES, frozen before comparisons.
PREDECLARED_TOL = {
    "theta_abs": 2.0e-3,
    "q_rel": 5.0e-3,
    "q_abs_floor": 1.0e-7,
    "precip_rel": 5.0e-3,
    "precip_abs": 2.0e-4,
}

Q_FIELDS = [("qv", "QV_OUT"), ("qc", "QC_OUT"), ("qr", "QR_OUT")]
CASE_IDS = (1, 2, 3, 4, 5)


def col(d: dict, name: str) -> np.ndarray:
    return np.asarray(d["columns"][name], dtype=np.float64)


def scalar(d: dict, name: str) -> float:
    return float(d["scalars"][name])


def field_metrics(jax_arr, oracle_arr, scale_floor: float) -> tuple[float, float, float]:
    a = np.asarray(jax_arr, dtype=np.float64)
    b = np.asarray(oracle_arr, dtype=np.float64)
    scale = max(float(np.max(np.abs(b))), scale_floor)
    absdiff = np.abs(a - b)
    return float(np.max(absdiff)), float(np.max(absdiff) / scale), scale


def run_jax_for(d: dict) -> dict[str, np.ndarray]:
    out = kessler_run(
        jnp.asarray(col(d, "THETA_IN")[None, :]),
        jnp.asarray(col(d, "QV_IN")[None, :]),
        jnp.asarray(col(d, "QC_IN")[None, :]),
        jnp.asarray(col(d, "QR_IN")[None, :]),
        jnp.asarray(col(d, "RHO")[None, :]),
        jnp.asarray(col(d, "PII")[None, :]),
        jnp.asarray(col(d, "Z")[None, :]),
        jnp.asarray(col(d, "DZ8W")[None, :]),
        scalar(d, "DT"),
    )
    return {key: np.asarray(value)[0] for key, value in out.items()}


def compare_case(save_dir: Path, cid: int) -> tuple[bool, dict]:
    with open(save_dir / f"kessler_case_{cid}.json", encoding="utf-8") as fh:
        d = json.load(fh)
    out = run_jax_for(d)
    results = {}
    passed = True

    mad, mrd, _scale = field_metrics(out["theta"], col(d, "THETA_OUT"), 1.0)
    ok = mad <= PREDECLARED_TOL["theta_abs"]
    passed = passed and ok
    results["theta"] = {
        "max_abs": mad,
        "max_rel": mrd,
        "tol_abs": PREDECLARED_TOL["theta_abs"],
        "pass": bool(ok),
    }

    for leaf, oracle_name in Q_FIELDS:
        mad, mrd, scale = field_metrics(out[leaf], col(d, oracle_name), PREDECLARED_TOL["q_abs_floor"])
        ok = (mrd <= PREDECLARED_TOL["q_rel"]) or (mad <= PREDECLARED_TOL["q_abs_floor"])
        passed = passed and ok
        results[leaf] = {
            "max_abs": mad,
            "max_rel": mrd,
            "scale": scale,
            "tol_rel": PREDECLARED_TOL["q_rel"],
            "tol_abs_floor": PREDECLARED_TOL["q_abs_floor"],
            "pass": bool(ok),
        }

    for leaf, oracle_name in (("rainnc", "RAINNC"), ("rainncv", "RAINNCV")):
        jv = float(out[leaf])
        ov = scalar(d, oracle_name)
        tol = max(PREDECLARED_TOL["precip_rel"] * abs(ov), PREDECLARED_TOL["precip_abs"])
        ok = abs(jv - ov) <= tol
        passed = passed and ok
        results[leaf] = {
            "jax": jv,
            "oracle": ov,
            "abs_err": abs(jv - ov),
            "tol": tol,
            "pass": bool(ok),
        }

    return passed, {
        "label": d["metadata"]["case_label"],
        "full_wrf_exe": d["metadata"]["full_wrf_exe"],
        "fields": results,
        "pass": bool(passed),
    }


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
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
        print(f"=== FP32 CASE {cid} {res['label']} -> {status} ===")
        for field, metrics in res["fields"].items():
            if "max_abs" in metrics:
                print(
                    f"  {field:8s} max_abs={metrics['max_abs']:.3e} "
                    f"max_rel={metrics['max_rel']:.3e} {'ok' if metrics['pass'] else 'FAIL'}"
                )
            else:
                print(
                    f"  {field:8s} jax={metrics['jax']:.6e} oracle={metrics['oracle']:.6e} "
                    f"abs_err={metrics['abs_err']:.3e} {'ok' if metrics['pass'] else 'FAIL'}"
                )
        print()

    for cid in CASE_IDS:
        ok, res = compare_case(SAVE_FP64, cid)
        fp64_cases[str(cid)] = res
        fp64_pass = fp64_pass and ok

    source_checksums = read_text_if_present(SAVE_FP32 / "wrf_source_checksums.txt")
    fp64_checksums = read_text_if_present(SAVE_FP64 / "wrf_source_checksums.txt")
    report = {
        "scheme": "Kessler warm rain (mp_physics=1)",
        "verdict": "PASS" if canonical_pass and fp64_pass else "FAIL",
        "overall_pass": bool(canonical_pass and fp64_pass),
        "canonical_fp32_pass": bool(canonical_pass),
        "fp64_precision_audit_pass": bool(fp64_pass),
        "oracle": {
            "source": "$WRF_PRISTINE_ROOT/phys/module_mp_kessler.F",
            "source_unmodified": True,
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled the unmodified WRF Kessler module; "
                "this is not a full wrf.exe integration savepoint."
            ),
            "fp32_savepoints": "proofs/v060/savepoints",
            "fp64_savepoints": "proofs/v060/savepoints_fp64",
            "fp32_source_checksums": source_checksums,
            "fp64_source_checksums": fp64_checksums,
            "fp32_build_manifest": read_text_if_present(SAVE_FP32 / "build_manifest.txt"),
            "fp64_build_manifest": read_text_if_present(SAVE_FP64 / "build_manifest.txt"),
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": git_head(),
        "predeclared_tolerances": PREDECLARED_TOL,
        "fp64_dust_handling": (
            "Canonical WRF state parity is against the fp32 default REAL build. "
            "The fp64 build uses the same unmodified source with -fdefault-real-8 "
            "as an additional precision audit; no Kessler diagnostics required "
            "switching the binding reference away from fp32."
        ),
        "cases": canonical_cases,
        "fp64_audit_cases": fp64_cases,
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("OVERALL:", report["verdict"])
    print("wrote", REPORT)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
