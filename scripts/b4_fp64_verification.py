#!/usr/bin/env python3
"""B4 fp64 / precision-defeat verification for the lateral-boundary path.

Confirms that ``apply_lateral_boundaries`` does NOT silently re-introduce fp32
on the operational path, in two regimes:

A. **force_fp64 regime** (idealized / fp64-locked): after the operational
   precision enforcement upcasts the State (incl. all ``*_bdy`` leaves) to
   float64, the boundary-applied State stays float64 for every prognostic and
   every boundary leaf is read at float64.

B. **operational perf regime** (ADR-007 PRECISION_MATRIX): each field keeps its
   storage dtype.  The crucial precision-defeat guard is that an fp64-locked
   prognostic (w, p_total, ph_total, mu_total, p/ph/mu perturbations) is NEVER
   downcast to fp32 by the boundary apply, and that its forcing leaf is itself
   fp64 (matching the field), so the relaxation arithmetic runs in fp64.  The
   fp32-gated fields (u, v, theta, qv) stay fp32 with fp32 leaves -- this is the
   sanctioned perf storage, not a defeat (the data source is fp32 wrfout).

Writes proofs/b4/fp64_verification.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import jax.numpy as jnp

from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG, apply_lateral_boundaries
from gpuwrf.contracts.precision import PRECISION_MATRIX
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import _enforce_operational_precision

DEFAULT_RUN = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")

PROGNOSTIC = ("u", "v", "w", "theta", "qv", "p_total", "p_perturbation",
              "ph_total", "ph_perturbation", "mu_total", "mu_perturbation")
BDY_LEAVES = ("u_bdy", "v_bdy", "w_bdy", "theta_bdy", "qv_bdy", "p_bdy",
              "pb_bdy", "ph_bdy", "phb_bdy", "mu_bdy", "mub_bdy")
FP64_LOCKED = ("w", "p_total", "p_perturbation", "ph_total", "ph_perturbation",
               "mu_total", "mu_perturbation")


def _dtype(state, field):
    return str(np.asarray(getattr(state, field)).dtype)


def run(run_dir: Path) -> dict:
    cfg = DEFAULT_BOUNDARY_CONFIG
    case = build_replay_case(str(run_dir), domain="d02")

    # Regime A: force_fp64
    st_a = _enforce_operational_precision(case.state, force_fp64=True)
    out_a = apply_lateral_boundaries(st_a, jnp.asarray(0.0), 6.0, cfg)
    a_prog = {f: _dtype(out_a, f) for f in PROGNOSTIC}
    a_bdy = {f: _dtype(st_a, f) for f in BDY_LEAVES}
    a_all_fp64 = all(v == "float64" for v in a_prog.values()) and all(v == "float64" for v in a_bdy.values())

    # Regime B: operational perf matrix (no force_fp64)
    st_b = case.state  # build_replay_case leaves perf-matrix dtypes
    out_b = apply_lateral_boundaries(st_b, jnp.asarray(0.0), 6.0, cfg)
    b_prog = {f: _dtype(out_b, f) for f in PROGNOSTIC}
    # fp64-locked prognostics must NOT have been downcast by the boundary apply
    b_locked_ok = all(b_prog[f] == "float64" for f in FP64_LOCKED)
    # each fp64-locked field's forcing leaf is fp64 too (no fp32 pollution)
    leaf_for = {"w": "w_bdy", "p_perturbation": "p_bdy", "ph_perturbation": "ph_bdy",
                "mu_perturbation": "mu_bdy"}
    b_leaf_ok = all(_dtype(st_b, leaf_for[f]) == "float64" for f in leaf_for)
    # storage dtypes match the frozen ADR-007 matrix (no drift)
    matrix_ok = all(
        _dtype(out_b, f) == ("float64" if PRECISION_MATRIX[f][0] == jnp.float64 else "float32")
        for f in PROGNOSTIC
    )

    status = "PASS" if (a_all_fp64 and b_locked_ok and b_leaf_ok and matrix_ok) else "FAIL"
    return {
        "artifact_type": "b4_fp64_verification",
        "status": status,
        "run_dir": str(run_dir),
        "regime_A_force_fp64": {
            "prognostic_dtypes_after_apply": a_prog,
            "boundary_leaf_dtypes": a_bdy,
            "all_fp64": a_all_fp64,
        },
        "regime_B_operational_perf_matrix": {
            "prognostic_dtypes_after_apply": b_prog,
            "fp64_locked_not_downcast": b_locked_ok,
            "fp64_locked_forcing_leaves_are_fp64": b_leaf_ok,
            "matches_adr007_matrix": matrix_ok,
        },
        "conclusion": (
            "Boundary apply preserves float64 end-to-end under force_fp64, and on the "
            "operational perf path never downcasts an fp64-locked prognostic nor reads "
            "its forcing leaf in fp32. fp32-gated u/v/theta/qv stay fp32 (sanctioned "
            "ADR-007 storage; the wrfout data source is itself fp32)."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path, default=ROOT / "proofs/b4/fp64_verification.json")
    args = ap.parse_args(argv)
    payload = run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
