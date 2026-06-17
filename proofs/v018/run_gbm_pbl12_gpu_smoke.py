#!/usr/bin/env python3
"""v0.18 PBL12 GBM GPU smoke proof.

The strict WRF-oracle parity proof is ``run_gbm_pbl12_parity.py``. This script
only asserts that the operational JAX/vmap endpoint compiles/runs on a GPU and
returns finite arrays.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.physics.bl_gbm import gbm_columns  # noqa: E402

SAVEPOINT = ROOT / "proofs" / "v017" / "savepoints_fp64" / "gbm" / "gbm_case_1.json"
REPORT_PATH = ROOT / "proofs" / "v018" / "gbm_pbl12_gpu_smoke.json"


def _load() -> dict[str, Any]:
    return json.loads(SAVEPOINT.read_text(encoding="utf-8"))


def _col(data: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _main() -> None:
    data = _load()
    s = data["scalars"]

    columns = [
        jnp.asarray(_col(data, name)[None, :], dtype=jnp.float64)
        for name in ("U", "V", "T", "QV", "P", "PI", "DZ", "TKE_PBL")
    ]
    qc = jnp.zeros_like(columns[3])
    scalars = {
        name: jnp.asarray([s[name]], dtype=jnp.float64)
        for name in ("PSFC", "ZNT", "UST", "HFX", "QFX", "TSK", "GZ1OZ0", "WSPD", "BR", "PSIM", "PSIH", "XLAND")
    }

    @jax.jit
    def run(u, v, t, qv, p, pii, dz, tke, qc, psfc, znt, ust, hfx, qfx, tsk, gz, wspd, br, psim, psih, xland):
        return gbm_columns(
            u, v, t, qv, qc, p, pii, dz, tke,
            psfc=psfc,
            znt=znt,
            ust=ust,
            hfx=hfx,
            qfx=qfx,
            tsk=tsk,
            gz1oz0=gz,
            wspd=wspd,
            br=br,
            psim=psim,
            psih=psih,
            dt=s["DT"],
            xland=xland,
        )

    args = tuple(columns) + (qc,) + tuple(
        scalars[name]
        for name in ("PSFC", "ZNT", "UST", "HFX", "QFX", "TSK", "GZ1OZ0", "WSPD", "BR", "PSIM", "PSIH", "XLAND")
    )
    out = run(*args)
    jax.block_until_ready(out["theta"])

    arrays = {name: np.asarray(value) for name, value in out.items()}
    finite = {name: bool(np.all(np.isfinite(value))) for name, value in arrays.items()}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    backend = jax.default_backend()
    devices = [str(device) for device in jax.devices()]
    verdict = "PASS" if backend == "gpu" and all(finite.values()) else "FAIL"
    report = {
        "schema": "gpuwrf.v018.gbm_pbl12_gpu_smoke",
        "scheme": "GBM moist TKE PBL (bl_pbl_physics=12)",
        "candidate": "gpuwrf.physics.bl_gbm.gbm_columns",
        "savepoint": str(SAVEPOINT.relative_to(ROOT)),
        "backend": backend,
        "devices": devices,
        "verdict": verdict,
        "all_finite": bool(all(finite.values())),
        "finite": finite,
        "shapes": shapes,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
