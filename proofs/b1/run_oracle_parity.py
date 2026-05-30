"""B1 REAL WRF-oracle parity runner (raw big-endian .f64 oracle).

Feeds the verified WRF v4.7.1 Thompson ``mp_gt_driver`` INPUT fields (raw
big-endian float64 per ``manifest.json``) into the JAX Thompson step and
compares the JAX outputs to the WRF ``mp_gt_driver`` OUTPUT fields, per field,
against the frozen Phase-B transcription tolerance ladder, using the
inactive-physical moist mask.  This is JAX-vs-WRF (NOT a self-compare).

Run:
  taskset -c 0-3 OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 \
    python3 proofs/b1/run_oracle_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import gpuwrf  # noqa: F401  (enables jax_enable_x64 at import — fp64 throughout)

from gpuwrf.validation.tier1_thompson import ORACLE_DIR, ORACLE_ARTIFACT, run_oracle_parity_f64


def main() -> dict:
    record = run_oracle_parity_f64(oracle_dir=ORACLE_DIR, out=ORACLE_ARTIFACT)
    summary = {
        "status": record.get("status"),
        "pass": record.get("pass"),
        "n_columns": record.get("n_columns"),
        "moist_cells": record.get("moist_cells"),
        "water_closure_max_rel_residual": record.get("water_closure_max_rel_residual"),
        "water_closure_pass": record.get("water_closure_pass"),
        "per_field_pass": {k: v["pass"] for k, v in record.get("per_field", {}).items()},
    }
    print(json.dumps(summary, indent=2))
    return record


if __name__ == "__main__":
    main()
