#!/usr/bin/env python3
"""Parity gate for the classic RRTM longwave port (ra_lw_physics=1).

The oracle savepoints come from the unmodified WRF ``phys/module_ra_rrtm.F``
classic RRTM path driven through ``RRTMLWRAD`` with the shipped AER
``RRTM_DATA`` lookup tables.  The canonical verdict compares the JAX column
endpoint against the fp64 oracle savepoints and reports residuals for
``RTHRATEN``, ``GLW``, and ``OLR`` over all seven predeclared columns.

This gate never self-compares and never fabricates a pass.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAVE_FP32 = HERE / "savepoints"
SAVE_FP64 = HERE / "savepoints_fp64"
REPORT = HERE / "rrtm_lw_savepoint_parity_report.json"
CASE_IDS = (1, 2, 3, 4, 5, 6, 7)

# Frozen before the port was implemented. Do not loosen without a new sprint
# contract and proof review.
PREDECLARED_TOL = {
    "rthraten_abs": 1.0e-8,
    "rthraten_rel": 1.0e-3,
    "glw_rel": 1.0e-3,
    "glw_abs": 1.0e-1,
    "olr_rel": 1.0e-3,
    "olr_abs": 1.0e-1,
}


def _jax_module_present() -> bool:
    return importlib.util.find_spec("gpuwrf.physics.ra_lw_rrtm") is not None


def _enable_jax_x64() -> None:
    import jax  # type: ignore

    jax.config.update("jax_enable_x64", True)


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def oracle_summary(save_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cid in CASE_IDS:
        p = save_dir / f"rrtm_lw_case_{cid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rth = np.asarray(d["columns"]["RTHRATEN"], dtype=np.float64)
        out[str(cid)] = {
            "label": d["scalars"]["REGIME"],
            "GLW": float(d["scalars"]["GLW"]),
            "OLR": float(d["scalars"]["OLR"]),
            "TSK": float(d["scalars"]["TSK"]),
            "max_abs_rthraten": float(np.max(np.abs(rth))),
            "col_sum_rthraten": float(np.sum(rth)),
            "full_wrf_exe": int(d["scalars"]["FULL_WRF_EXE"]),
        }
    return out


def _case_state(d: dict[str, Any]):
    import jax.numpy as jnp  # type: ignore

    def c(name: str):
        return jnp.asarray(np.asarray(d["columns"][name], dtype=np.float64)[None, :])

    from gpuwrf.physics.ra_lw_rrtm import RRTMLWColumnState  # type: ignore

    return RRTMLWColumnState(
        T=c("T"),
        t8w=c("T8W"),
        p=c("P"),
        p8w=c("P8W"),
        qv=c("QV"),
        qc=c("QC"),
        qr=c("QR"),
        qi=c("QI"),
        qs=c("QS"),
        qg=c("QG"),
        cloud_fraction=c("CLDFRA"),
        dz=c("DZ"),
        rho=c("RHO"),
        emiss=float(d["scalars"]["EMISS"]),
        tsk=float(d["scalars"]["TSK"]),
    )


def _rthraten_status(jax_rth: np.ndarray, oracle_rth: np.ndarray) -> dict[str, Any]:
    diff = np.abs(jax_rth - oracle_rth)
    max_abs = float(np.max(diff))
    max_idx = int(np.argmax(diff))
    scale = max(float(np.max(np.abs(oracle_rth))), PREDECLARED_TOL["rthraten_abs"])
    max_rel = max_abs / scale
    abs_ratio = max_abs / PREDECLARED_TOL["rthraten_abs"]
    rel_ratio = max_rel / PREDECLARED_TOL["rthraten_rel"]
    pass_abs = max_abs <= PREDECLARED_TOL["rthraten_abs"]
    pass_rel = max_rel <= PREDECLARED_TOL["rthraten_rel"]
    if rel_ratio <= abs_ratio:
        controlling_metric = "max_rel"
        controlling_value = max_rel
        controlling_limit = PREDECLARED_TOL["rthraten_rel"]
        normalized_ratio = rel_ratio
    else:
        controlling_metric = "max_abs"
        controlling_value = max_abs
        controlling_limit = PREDECLARED_TOL["rthraten_abs"]
        normalized_ratio = abs_ratio
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "argmax_k": max_idx,
        "scale": scale,
        "tol_abs": PREDECLARED_TOL["rthraten_abs"],
        "tol_rel": PREDECLARED_TOL["rthraten_rel"],
        "pass_abs": bool(pass_abs),
        "pass_rel": bool(pass_rel),
        "pass": bool(pass_abs or pass_rel),
        "controlling_metric": controlling_metric,
        "controlling_value": controlling_value,
        "controlling_limit": controlling_limit,
        "normalized_ratio": normalized_ratio,
    }


def _scalar_status(value: float, oracle: float, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    abs_err = abs(value - oracle)
    rel_err = abs_err / max(abs(oracle), 1.0e-300)
    limit_abs = max(abs_tol, rel_tol * abs(oracle))
    return {
        "value": value,
        "oracle": oracle,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "limit_abs": limit_abs,
        "tol_abs": abs_tol,
        "tol_rel": rel_tol,
        "pass": bool(abs_err <= limit_abs),
        "controlling_metric": "abs_err",
        "controlling_value": abs_err,
        "controlling_limit": limit_abs,
        "normalized_ratio": abs_err / limit_abs if limit_abs else 0.0,
    }


def _worst_from_cases(dataset: str, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    worst: dict[str, Any] | None = None
    for cid, case in cases.items():
        for field, status in case["fields"].items():
            candidate = {
                "dataset": dataset,
                "case_id": cid,
                "case_label": case["label"],
                "field": field,
                "metric": status["controlling_metric"],
                "value": status["controlling_value"],
                "limit": status["controlling_limit"],
                "normalized_ratio": status["normalized_ratio"],
            }
            if worst is None or candidate["normalized_ratio"] > worst["normalized_ratio"]:
                worst = candidate
    return worst or {
        "dataset": dataset,
        "case_id": None,
        "case_label": None,
        "field": None,
        "metric": None,
        "value": 0.0,
        "limit": 1.0,
        "normalized_ratio": 0.0,
    }


def compare_with_jax(save_dir: Path) -> tuple[bool, dict[str, dict[str, Any]], dict[str, Any]]:
    _enable_jax_x64()
    from gpuwrf.physics.ra_lw_rrtm import solve_rrtm_lw_column  # type: ignore

    cases: dict[str, dict[str, Any]] = {}
    all_pass = True
    for cid in CASE_IDS:
        d = json.loads((save_dir / f"rrtm_lw_case_{cid}.json").read_text(encoding="utf-8"))
        out = solve_rrtm_lw_column(_case_state(d))

        pi = np.asarray(d["columns"]["PI"], dtype=np.float64)
        jax_rth = np.asarray(out.heating_rate, dtype=np.float64)[0] / pi
        oracle_rth = np.asarray(d["columns"]["RTHRATEN"], dtype=np.float64)

        fields = {
            "RTHRATEN": _rthraten_status(jax_rth, oracle_rth),
            "GLW": _scalar_status(
                float(np.asarray(out.glw, dtype=np.float64)[0]),
                float(d["scalars"]["GLW"]),
                PREDECLARED_TOL["glw_abs"],
                PREDECLARED_TOL["glw_rel"],
            ),
            "OLR": _scalar_status(
                float(np.asarray(out.olr, dtype=np.float64)[0]),
                float(d["scalars"]["OLR"]),
                PREDECLARED_TOL["olr_abs"],
                PREDECLARED_TOL["olr_rel"],
            ),
        }
        case_pass = all(field["pass"] for field in fields.values())
        all_pass = all_pass and case_pass
        cases[str(cid)] = {
            "label": d["scalars"]["REGIME"],
            "pass": bool(case_pass),
            "fields": fields,
            # Compatibility keys for quick review of older reports.
            "rthraten_max_abs": fields["RTHRATEN"]["max_abs"],
            "rthraten_rel": fields["RTHRATEN"]["max_rel"],
            "glw_err": fields["GLW"]["abs_err"],
            "olr_err": fields["OLR"]["abs_err"],
        }
    return bool(all_pass), cases, _worst_from_cases(save_dir.name, cases)


def main() -> int:
    present = _jax_module_present()
    report: dict[str, Any] = {
        "scheme": "classic RRTM longwave (ra_lw_physics=1)",
        "jax_port_present": present,
        "jax_module": "src/gpuwrf/physics/ra_lw_rrtm.py",
        "canonical_dataset": "fp64_savepoints",
        "git_head": git_head(),
        "predeclared_tolerances": PREDECLARED_TOL,
        "oracle": {
            "source": "$WRF_PRISTINE_ROOT/phys/module_ra_rrtm.F",
            "entry": "RRTMLWRAD -> RRTM (AER 16-band k-distribution LW)",
            "lookup_asset": "RRTM_DATA / RRTM_DATA_DBL (big-endian, frecord-marker=4)",
            "source_unmodified": True,
            "full_wrf_exe": False,
            "full_wrf_exe_note": (
                "Standalone driver compiled the unmodified WRF module_ra_rrtm.F + "
                "module_ra_clWRF_support.F, loaded the RRTM_DATA tables via "
                "rrtminit(allowed_to_read=.TRUE.), and called RRTMLWRAD; not a full "
                "wrf.exe savepoint."
            ),
            "fp32_savepoints": "proofs/v060/savepoints (rrtm_lw_case_*.json)",
            "fp64_savepoints": "proofs/v060/savepoints_fp64 (rrtm_lw_case_*.json)",
            "fp32_source_checksums": read_text_if_present(SAVE_FP32 / "rrtm_lw_wrf_source_checksums.txt"),
            "fp64_source_checksums": read_text_if_present(SAVE_FP64 / "rrtm_lw_wrf_source_checksums.txt"),
            "fp32_build_manifest": read_text_if_present(SAVE_FP32 / "rrtm_lw_build_manifest.txt"),
            "fp64_build_manifest": read_text_if_present(SAVE_FP64 / "rrtm_lw_build_manifest.txt"),
            "fp64_oracle_summary": oracle_summary(SAVE_FP64),
        },
    }

    if not present:
        report["verdict"] = "FAIL"
        report["overall_pass"] = False
        report["note"] = "JAX RRTM-LW port is absent; no parity pass is fabricated."
        print("RRTM-LW JAX parity: FAIL (JAX port absent)")
    else:
        fp64_ok, fp64_cases, fp64_worst = compare_with_jax(SAVE_FP64)
        fp32_ok, fp32_cases, fp32_worst = compare_with_jax(SAVE_FP32)
        report["verdict"] = "PASS" if fp64_ok else "FAIL"
        report["overall_pass"] = bool(fp64_ok)
        report["canonical_fp64_cases"] = fp64_cases
        report["secondary_fp32_cases"] = fp32_cases
        report["secondary_fp32_pass"] = bool(fp32_ok)
        report["worst_residual"] = fp64_worst
        report["secondary_fp32_worst_residual"] = fp32_worst
        print("RRTM-LW JAX parity:", report["verdict"])
        for cid, case in fp64_cases.items():
            rth = case["fields"]["RTHRATEN"]
            glw = case["fields"]["GLW"]
            olr = case["fields"]["OLR"]
            print(
                f"  fp64 case {cid} {case['label']:24s} pass={case['pass']} "
                f"RTH_abs={rth['max_abs']:.3e} RTH_rel={rth['max_rel']:.3e} "
                f"GLW_err={glw['abs_err']:.3e} OLR_err={olr['abs_err']:.3e}"
            )
        print(
            "  worst fp64 residual "
            f"{fp64_worst['case_id']}:{fp64_worst['field']} "
            f"{fp64_worst['metric']}={fp64_worst['value']:.3e} "
            f"(limit {fp64_worst['limit']:.3e})"
        )
        print("  secondary fp32 parity:", "PASS" if fp32_ok else "FAIL")

    with REPORT.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print("wrote", REPORT)
    return 0 if report.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
