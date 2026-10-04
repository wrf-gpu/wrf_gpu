#!/usr/bin/env python3
"""Compare current JAX CAM-UW scaffold against the WRF Fortran oracle outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics.bl_camuw import camuw_columns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASES = (1, 2, 3)
REPORT = HERE / "camuw_current_jax_vs_wrf.json"

FIELD_MAP = {
    "u": "RUBLTEN",
    "v": "RVBLTEN",
    "theta": "RTHBLTEN",
    "qv": "RQVBLTEN",
    "qc": "RQCBLTEN",
    "qi": "RQIBLTEN",
    "tke": "TKE_PBL",
    "kvm": "KVM3D",
    "kvh": "KVH3D",
    "smaw": "SMAW3D",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(case: int) -> dict:
    with (HERE / f"camuw_case_{case}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _col(d: dict, name: str) -> np.ndarray:
    return np.asarray(d["columns"][name], dtype=np.float64)


def _metric(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = np.asarray(actual, dtype=np.float64).ravel()
    b = np.asarray(expected, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    err = np.abs(a - b)
    max_abs = float(np.max(err))
    rms = float(np.sqrt(np.mean(err * err)))
    scale = max(float(np.max(np.abs(b))), 1.0e-30)
    idx = int(np.argmax(err))
    return {
        "max_abs": max_abs,
        "rms": rms,
        "max_rel_to_oracle_scale": float(max_abs / scale),
        "argmax_flat": idx,
        "jax_at_argmax": float(a[idx]),
        "wrf_at_argmax": float(b[idx]),
        "oracle_scale": scale,
    }


def _run_one(d: dict) -> dict:
    scalars = d["scalars"]
    z_at_w = _col(d, "Z_AT_W")
    dz = np.diff(z_at_w)
    # WRF module_model_constants.F: epsq2 = 0.2, used by camuwpblinit on restart=false.
    tke_initial = np.full_like(_col(d, "U"), 0.2)
    out = camuw_columns(
        jnp.asarray(_col(d, "U")[None, :], jnp.float64),
        jnp.asarray(_col(d, "V")[None, :], jnp.float64),
        jnp.asarray(_col(d, "T")[None, :], jnp.float64),
        jnp.asarray(_col(d, "TH")[None, :], jnp.float64),
        jnp.asarray(_col(d, "QV")[None, :], jnp.float64),
        jnp.asarray(_col(d, "QC")[None, :], jnp.float64),
        jnp.asarray(_col(d, "QI")[None, :], jnp.float64),
        jnp.asarray(_col(d, "P")[None, :], jnp.float64),
        jnp.asarray(_col(d, "EXNER")[None, :], jnp.float64),
        jnp.asarray(dz[None, :], jnp.float64),
        jnp.asarray(_col(d, "Z")[None, :], jnp.float64),
        jnp.asarray(tke_initial[None, :], jnp.float64),
        hfx=jnp.asarray([scalars["HFX"]], jnp.float64),
        qfx=jnp.asarray([scalars["QFX"]], jnp.float64),
        ust=jnp.asarray([scalars["USTAR"]], jnp.float64),
        wspd=jnp.asarray([float(np.hypot(_col(d, "U")[0], _col(d, "V")[0]))], jnp.float64),
        dt=float(scalars["DT"]),
    )
    out_np = {k: np.asarray(v)[0] for k, v in out.items()}
    metrics = {}
    for jax_name, wrf_name in FIELD_MAP.items():
        if wrf_name in ("TKE_PBL", "KVM3D", "KVH3D", "SMAW3D"):
            metrics[jax_name] = _metric(out_np[jax_name], _col(d, wrf_name))
        else:
            metrics[jax_name] = _metric(out_np[jax_name], _col(d, wrf_name))
    metrics["pblh"] = _metric(np.asarray([out_np["pblh"]]), np.asarray([scalars["PBLH"]]))
    return {
        "case": scalars["CASE"],
        "regime": scalars["REGIME"],
        "metrics": metrics,
    }


def main() -> int:
    cases = [_run_one(_load(c)) for c in CASES]
    worst = max(
        (
            (case["case"], field, metric["max_abs"], metric["max_rel_to_oracle_scale"])
            for case in cases
            for field, metric in case["metrics"].items()
        ),
        key=lambda item: item[2],
    )
    report = {
        "schema": "gpuwrf.v023.camuw_current_jax_vs_wrf.v1",
        "verdict": "FAIL",
        "reason": (
            "The current src/gpuwrf/physics/bl_camuw.py endpoint is a v0.22 "
            "CAM-style scaffold, not a WRF Fortran CAM-UW transcription."
        ),
        "jax_platform": jax.default_backend(),
        "jax_x64": bool(jax.config.jax_enable_x64),
        "cases": cases,
        "worst_by_abs": {
            "case": worst[0],
            "field": worst[1],
            "max_abs": worst[2],
            "max_rel_to_oracle_scale": worst[3],
        },
        "oracle": {
            "type": "standalone driver linked against unmodified pristine-WRF CAM-UW objects",
            "full_wrf_exe": False,
            "self_compare": False,
            "wrf_sources_sha256": {
                str(path): _sha256(path)
                for path in [
                    Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_camuwpbl_driver.F"),
                    Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_cam_bl_eddy_diff.F"),
                    Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_cam_bl_diffusion_solver.F"),
                ]
            },
        },
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"verdict={report['verdict']} worst={report['worst_by_abs']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
