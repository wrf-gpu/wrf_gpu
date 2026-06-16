#!/usr/bin/env python3
"""v0.17 MYNN BouLac ONZ oracle.

Proves the actual ``GPUWRF_MYNN_BOULAC_ONZ=1`` dispatcher path is bit-identical
to the dense BouLac length for fp64 inputs.  The production ONZ path is the
source-chunked dense schedule with chunk=1; it preserves dense arithmetic order
while avoiding the full ``(B, nz, nz)`` live matrix.

Run on CPU:

  taskset -c 0-3 env PYTHONPATH=src:. JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
    python proofs/perf/v017/boulac_onz_oracle.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ["GPUWRF_MYNN_BOULAC_ONZ"] = "1"
os.environ["GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN"] = "0"
os.environ["GPUWRF_MYNN_BOULAC_FP32"] = "0"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import mynn_pbl as MYNN


OUT = Path("proofs/perf/v017/boulac_onz_oracle.json")
V015_ORACLE = Path("proofs/perf/v015/boulac_nz_oracle.py")


def _load_v015_oracle():
    spec = importlib.util.spec_from_file_location("_v015_boulac_oracle", V015_ORACLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V015_ORACLE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _max_abs_pair(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
    return float(max(np.max(np.abs(left[0] - right[0])), np.max(np.abs(left[1] - right[1]))))


def _array_equal_pair(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1]))


def _as_np(pair) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(pair[0], np.float64), np.asarray(pair[1], np.float64)


def main() -> int:
    v015 = _load_v015_oracle()
    cases = v015.make_cases()
    nz = int(cases[0][1].shape[0])

    dz_b = np.stack([c[1] for c in cases])
    theta_b = np.stack([c[2] for c in cases])
    qtke_b = np.stack([c[3] for c in cases])
    zw_b = np.stack([c[4][:-1] for c in cases])

    zw = jnp.asarray(zw_b, dtype=jnp.float64)
    dz = jnp.asarray(dz_b, dtype=jnp.float64)
    qtke = jnp.asarray(qtke_b, dtype=jnp.float64)
    theta = jnp.asarray(theta_b, dtype=jnp.float64)

    dense = _as_np(MYNN._boulac_length_dense(zw, dz, qtke, theta))
    dispatched = _as_np(MYNN._boulac_length(zw, dz, qtke, theta))
    production = _as_np(MYNN._boulac_length_onz_production(zw, dz, qtke, theta))
    legacy_scan = _as_np(MYNN._boulac_length_onz(zw, dz, qtke, theta))

    dispatch_max_abs = _max_abs_pair(dispatched, dense)
    production_max_abs = _max_abs_pair(production, dense)
    legacy_scan_max_abs = _max_abs_pair(legacy_scan, dense)

    per_case = []
    worst_wrf = {"case": None, "field": None, "max_abs": -1.0, "max_rel": -1.0}
    for i, (name, _dz, theta_i, qtke_i, zw_full) in enumerate(cases):
        ref = v015.wrf_boulac_length_column(zw_full, _dz, qtke_i, theta_i)
        for field, got, want in (
            ("lb1", dispatched[0][i], ref[0]),
            ("lb2", dispatched[1][i], ref[1]),
        ):
            diff = np.abs(got - want)
            scale = np.maximum(np.abs(want), 1.0e-12)
            max_abs = float(np.max(diff))
            max_rel = float(np.max(diff / scale))
            rec = {"case": name, "field": field, "max_abs": max_abs, "max_rel": max_rel}
            per_case.append(rec)
            if max_abs > worst_wrf["max_abs"]:
                worst_wrf = dict(rec)

    bit_identical = (
        dispatch_max_abs == 0.0
        and production_max_abs == 0.0
        and _array_equal_pair(dispatched, dense)
        and _array_equal_pair(production, dense)
    )
    verdict = "PASS" if bit_identical else "FAIL"

    payload = {
        "schema": "gpuwrf.v017.boulac_onz_oracle.v1",
        "scope": "GPUWRF_MYNN_BOULAC_ONZ=1 dispatched path vs dense BouLac",
        "verdict": verdict,
        "mandatory_bit_identity": {
            "max_abs": dispatch_max_abs,
            "array_equal": _array_equal_pair(dispatched, dense),
            "requirement": "max_abs == 0.0",
        },
        "production_impl": {
            "description": "source-chunked dense BouLac, source chunk=1",
            "o_nz_working_set": True,
            "gpuwrf_mynn_boulac_onz": bool(MYNN._MYNN_BOULAC_ONZ),
            "gpuwrf_mynn_boulac_onz_legacy_scan": bool(MYNN._MYNN_BOULAC_ONZ_LEGACY_SCAN),
            "gpuwrf_mynn_boulac_fp32": bool(MYNN._MYNN_BOULAC_FP32),
            "production_vs_dense_max_abs": production_max_abs,
        },
        "legacy_scan_characterization": {
            "legacy_scan_vs_dense_max_abs": legacy_scan_max_abs,
            "note": "legacy scan is kept only for diagnostics; production ONZ dispatch does not use it",
        },
        "wrf_reference": {
            "source": str(V015_ORACLE),
            "routine": "module_bl_mynnedmf.F:2192-2338, loop-for-loop NumPy transcription",
            "worst_vs_wrf": worst_wrf,
            "per_case": per_case,
        },
        "nz": nz,
        "n_cases": len(cases),
        "regimes": [c[0] for c in cases],
        "no_clamps_or_tolerance_changes": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "verdict": verdict,
        "dispatch_vs_dense_max_abs": dispatch_max_abs,
        "production_vs_dense_max_abs": production_max_abs,
        "legacy_scan_vs_dense_max_abs": legacy_scan_max_abs,
        "worst_vs_wrf": worst_wrf,
        "out": str(OUT),
    }, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
