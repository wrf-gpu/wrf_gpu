#!/usr/bin/env python3
"""v0.18 PBL11 Shin-Hong GPU smoke proof.

This is a device residency/JIT smoke for the operational JAX/vmap endpoint. The
strict oracle parity proof is ``run_shinhong_pbl11_parity.py``; this script only
asserts that the endpoint compiles/runs on a GPU and returns finite arrays.
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

from gpuwrf.physics.bl_shinhong import shinhong_columns  # noqa: E402

SAVEPOINT = ROOT / "proofs" / "v090" / "savepoints" / "shinhong_case_1.json"
REPORT_PATH = ROOT / "proofs" / "v018" / "shinhong_pbl11_gpu_smoke.json"


def _load() -> dict[str, Any]:
    return json.loads(SAVEPOINT.read_text())


def _col(data: dict[str, Any], name: str) -> np.ndarray:
    return np.asarray(data["columns"][name], dtype=np.float64)


def _main() -> None:
    data = _load()
    s = data["scalars"]

    columns = [
        jnp.asarray(_col(data, name)[None, :], dtype=jnp.float64)
        for name in ("U", "V", "T", "QV", "P", "PDI", "PI", "DZ", "TKE_PBL")
    ]
    scalars = {
        name: jnp.asarray([s[name]], dtype=jnp.float64)
        for name in ("PSFC", "ZNT", "UST", "HFX", "QFX", "WSPD", "BR", "PSIM", "PSIH", "XLAND", "U10", "V10")
    }

    @jax.jit
    def run(*args):
        return shinhong_columns(
            *args[:9],
            psfc=args[9],
            znt=args[10],
            ust=args[11],
            hfx=args[12],
            qfx=args[13],
            wspd=args[14],
            br=args[15],
            psim=args[16],
            psih=args[17],
            dt=s["DT"],
            xland=args[18],
            u10=args[19],
            v10=args[20],
            dx=s["DX"],
            dy=s["DY"],
        )

    args = tuple(columns) + tuple(
        scalars[name] for name in ("PSFC", "ZNT", "UST", "HFX", "QFX", "WSPD", "BR", "PSIM", "PSIH", "XLAND", "U10", "V10")
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
        "schema": "gpuwrf.v018.shinhong_pbl11_gpu_smoke",
        "scheme": "Shin-Hong scale-aware PBL (bl_pbl_physics=11)",
        "candidate": "gpuwrf.physics.bl_shinhong.shinhong_columns",
        "savepoint": str(SAVEPOINT.relative_to(ROOT)),
        "backend": backend,
        "devices": devices,
        "verdict": verdict,
        "all_finite": bool(all(finite.values())),
        "finite": finite,
        "shapes": shapes,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
