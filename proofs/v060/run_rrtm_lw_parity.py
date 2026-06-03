#!/usr/bin/env python3
"""Parity gate skeleton for the classic RRTM longwave port (ra_lw_physics=1).

The WRF oracle (gold savepoints from the UNMODIFIED phys/module_ra_rrtm.F via
proofs/v060/oracle/rrtm_lw_build_and_run.sh) is COMPLETE and physically
validated.  The JAX column port (src/gpuwrf/physics/ra_lw_rrtm.py) is a scoped
follow-on: classic RRTM is a 16-band k-distribution scheme (~7.6k Fortran LOC +
the RRTM_DATA lookup asset) on the same order as the existing 2k-LOC RRTMG-LW
JAX port.

This runner is the turnkey gate for that follow-on.  When the JAX module is
present it compares RTHRATEN/GLW/OLR against the gold savepoints; until then it
emits an honest report recording oracle provenance and a JAX_PORT_PRESENT=false
verdict.  It NEVER fabricates a pass.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAVE_FP32 = HERE / "savepoints"
SAVE_FP64 = HERE / "savepoints_fp64"
REPORT = HERE / "rrtm_lw_savepoint_parity_report.json"
CASE_IDS = (1, 2, 3, 4, 5, 6, 7)

# PREDECLARED TOLERANCES (frozen now, before any JAX port exists). RRTM column
# runs in r4 in WRF; the canonical fp32 gold carries single-precision dust
# through the 16-band radiative transfer.
PREDECLARED_TOL = {
    "rthraten_abs": 1.0e-8,   # K/s absolute floor (near-zero layers)
    "rthraten_rel": 1.0e-3,   # relative on column max heating rate
    "glw_rel": 1.0e-3,        # surface downwelling LW (W/m^2)
    "glw_abs": 1.0e-1,
    "olr_rel": 1.0e-3,        # TOA outgoing LW (W/m^2)
    "olr_abs": 1.0e-1,
}


def _jax_module_present() -> bool:
    return importlib.util.find_spec("gpuwrf.physics.ra_lw_rrtm") is not None


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def oracle_summary(save_dir: Path) -> dict:
    out = {}
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


def compare_with_jax() -> tuple[bool, dict]:
    from gpuwrf.physics.ra_lw_rrtm import (  # type: ignore  # noqa: E402
        RRTMLWColumnState,
        solve_rrtm_lw_column,
    )

    canonical_cases = {}
    canonical_pass = True
    for cid in CASE_IDS:
        d = json.loads((SAVE_FP32 / f"rrtm_lw_case_{cid}.json").read_text(encoding="utf-8"))

        def c(name):
            import jax.numpy as jnp
            return jnp.asarray(np.asarray(d["columns"][name], dtype=np.float64)[None, :])

        state = RRTMLWColumnState(
            T=c("T"), t8w=c("T8W"), p=c("P"), p8w=c("P8W"),
            qv=c("QV"), qc=c("QC"), qr=c("QR"), qi=c("QI"), qs=c("QS"), qg=c("QG"),
            cloud_fraction=c("CLDFRA"), dz=c("DZ"), rho=c("RHO"),
            emiss=float(d["scalars"]["EMISS"]), tsk=float(d["scalars"]["TSK"]),
        )
        out = solve_rrtm_lw_column(state)
        pi = np.asarray(d["columns"]["PI"], dtype=np.float64)
        jax_rth = np.asarray(out.heating_rate)[0] / pi
        oracle_rth = np.asarray(d["columns"]["RTHRATEN"], dtype=np.float64)
        scale = max(float(np.max(np.abs(oracle_rth))), PREDECLARED_TOL["rthraten_abs"])
        max_abs = float(np.max(np.abs(jax_rth - oracle_rth)))
        rth_ok = (max_abs / scale <= PREDECLARED_TOL["rthraten_rel"]) or (
            max_abs <= PREDECLARED_TOL["rthraten_abs"]
        )

        glw_err = abs(float(out.glw[0]) - float(d["scalars"]["GLW"]))
        olr_err = abs(float(out.olr[0]) - float(d["scalars"]["OLR"]))
        glw_ok = glw_err <= max(PREDECLARED_TOL["glw_rel"] * abs(d["scalars"]["GLW"]), PREDECLARED_TOL["glw_abs"])
        olr_ok = olr_err <= max(PREDECLARED_TOL["olr_rel"] * abs(d["scalars"]["OLR"]), PREDECLARED_TOL["olr_abs"])

        ok = bool(rth_ok and glw_ok and olr_ok)
        canonical_pass = canonical_pass and ok
        canonical_cases[str(cid)] = {
            "label": d["scalars"]["REGIME"],
            "rthraten_max_abs": max_abs,
            "rthraten_rel": max_abs / scale,
            "glw_err": glw_err,
            "olr_err": olr_err,
            "pass": ok,
        }
    return canonical_pass, canonical_cases


def main() -> int:
    present = _jax_module_present()
    report = {
        "scheme": "classic RRTM longwave (ra_lw_physics=1)",
        "jax_port_present": present,
        "git_head": git_head(),
        "predeclared_tolerances": PREDECLARED_TOL,
        "oracle": {
            "source": "/home/enric/src/wrf_pristine/WRF/phys/module_ra_rrtm.F",
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
        report["verdict"] = "ORACLE_READY_JAX_PORT_PENDING"
        report["overall_pass"] = False
        report["note"] = (
            "WRF gold oracle built + physically validated (GLW/OLR/RTHRATEN sane "
            "across 7 clear/cloudy/cold/tropical columns). JAX column port "
            "src/gpuwrf/physics/ra_lw_rrtm.py is the scoped follow-on; this gate "
            "will compare against the gold savepoints once it lands. No pass is "
            "fabricated in its absence."
        )
        print("RRTM-LW: oracle READY, JAX port PENDING (no fabricated pass).")
        for cid, s in report["oracle"]["fp64_oracle_summary"].items():
            print(
                f"  oracle case {cid} {s['label']:24s} GLW={s['GLW']:8.2f} "
                f"OLR={s['OLR']:7.2f} max|RTH|={s['max_abs_rthraten']:.3e}"
            )
    else:
        ok, cases = compare_with_jax()
        report["verdict"] = "PASS" if ok else "FAIL"
        report["overall_pass"] = bool(ok)
        report["canonical_fp32_cases"] = cases
        print("RRTM-LW JAX parity:", report["verdict"])

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", REPORT)
    # Exit 0 when the oracle is ready (pre-port) or the JAX parity passes; exit 1
    # only when a present JAX port FAILS parity.
    return 0 if (report.get("overall_pass") or not present) else 1


if __name__ == "__main__":
    raise SystemExit(main())
