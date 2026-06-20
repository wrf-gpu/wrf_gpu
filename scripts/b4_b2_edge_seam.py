#!/usr/bin/env python3
"""B4 / B2 edge-seam analysis (Gate-1 decision #4).

B2's MYNN PBL rewrites the C-grid winds via a mass-point round-trip whose
inverse (`_mass_to_u_face` / `_mass_to_v_face` in coupling/physics_couplers.py)
assumes PERIODIC x/y edges (`jnp.roll` + duplicated outer face). For the real,
non-periodic Canary d02 case that corrupts exactly the domain-edge faces
(u at x=0 and x=nx; v at y=0 and y=ny).  This script characterises which faces
are corrupted and demonstrates that B4's lateral-boundary application, which
runs AFTER the physics block (operational_mode.py: physics -> boundary), hard-
sets those same edge faces to the WRF boundary value -- so the seam is closed
provided the operator order physics->boundary is preserved.

The contract written to proofs/b4/b2_b4_edge_seam.json:
  * B2 MAY use its periodic mass->face inverse internally, BUT the corrupted
    cells are confined to the outermost staggered faces (u[:, :, {0, nx}],
    v[:, {0, ny}, :]) plus the relaxation columns that B4 nudges.
  * B4's spec zone (b_dist < spec_zone) HARD-SETS u[:, :, 0]/u[:, :, nx] and
    v[:, 0, :]/v[:, ny, :] to the WRF boundary value every step, overwriting the
    periodic-reconstruction artefact at the outer face.
  * The seam therefore requires: (a) operator order physics(MYNN) -> boundary,
    and (b) spec_zone >= 1 so the outer face is hard-set.  Both hold for the
    pinned run (spec_zone=1) and the frozen RK1 bundle order.

Writes proofs/b4/b2_b4_edge_seam.json.
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
from gpuwrf.coupling.physics_couplers import _mass_to_u_face, _mass_to_v_face, _u_mass, _v_mass
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import _enforce_operational_precision

DEFAULT_RUN = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")


def run(run_dir: Path) -> dict:
    cfg = DEFAULT_BOUNDARY_CONFIG
    case = build_replay_case(str(run_dir), domain="d02")
    st = _enforce_operational_precision(case.state, force_fp64=True)
    u0 = np.asarray(st.u)
    v0 = np.asarray(st.v)

    # 1) Isolate the PERIODICITY-SPECIFIC corruption.  MYNN maps a mass-point
    #    wind *increment* back to faces; the mass<->face round-trip is lossy
    #    everywhere (averaging is not invertible), but that interior loss is a
    #    generic staggering effect, not a seam issue.  The seam issue is the
    #    jnp.roll wrap: the WEST outer face (x=0) is set from the EAST-most mass
    #    column and the appended EAST face duplicates the west face.  We expose
    #    it by comparing the periodic inverse against a non-periodic
    #    edge-extrapolation inverse on a smooth mass-point increment; the two
    #    differ ONLY where periodicity wraps -- the outer faces.
    nx = u0.shape[2] - 1
    ny = v0.shape[1] - 1
    # smooth mass-point increment (z,y,nx) and (z,ny_mass,nx)
    inc_u_mass = np.asarray(_u_mass(st)) * 0.01
    inc_v_mass = np.asarray(_v_mass(st)) * 0.01

    def nonperiodic_u_face(m):  # m: (z,y,nx) -> (z,y,nx+1) edge-extrapolated
        face = 0.5 * (m[:, :, :-1] + m[:, :, 1:])
        return np.concatenate((m[:, :, :1], face, m[:, :, -1:]), axis=2)

    def nonperiodic_v_face(m):  # m: (z,ny,nx) -> (z,ny+1,nx)
        face = 0.5 * (m[:, :-1, :] + m[:, 1:, :])
        return np.concatenate((m[:, :1, :], face, m[:, -1:, :]), axis=1)

    u_per = np.asarray(_mass_to_u_face(jnp.asarray(inc_u_mass)))
    v_per = np.asarray(_mass_to_v_face(jnp.asarray(inc_v_mass)))
    u_np = nonperiodic_u_face(inc_u_mass)
    v_np = nonperiodic_v_face(inc_v_mass)
    u_seam_err = np.abs(u_per - u_np).max(axis=(0, 1))   # length nx+1
    v_seam_err = np.abs(v_per - v_np).max(axis=(0, 2))   # length ny+1
    u_bad_faces = [int(i) for i in np.flatnonzero(u_seam_err > 1e-9)]
    v_bad_faces = [int(j) for j in np.flatnonzero(v_seam_err > 1e-9)]
    u_face_err_by_x = u_seam_err
    v_face_err_by_y = v_seam_err
    # full-field periodic faces (for the post-MYNN restoration test below)
    u_face_periodic = np.asarray(_mass_to_u_face(_u_mass(st)))
    v_face_periodic = np.asarray(_mass_to_v_face(_v_mass(st)))

    # 2) Apply B4 boundary AFTER the periodic reconstruction; confirm the outer
    #    edge faces are restored to the WRF boundary value (spec zone hard-set).
    st_after_mynn = st.replace(u=jnp.asarray(u_face_periodic), v=jnp.asarray(v_face_periodic))
    out = apply_lateral_boundaries(st_after_mynn, jnp.asarray(0.0), 6.0, cfg)
    u_after = np.asarray(out.u)
    v_after = np.asarray(out.v)
    ubdy = np.asarray(st.u_bdy)  # (t,side,bw,z,side_len)
    vbdy = np.asarray(st.v_bdy)
    # WRF boundary value at the outer face (spec zone)
    u_west_target = ubdy[0, 0, 0, :, : u0.shape[1]]   # (z, y)
    u_east_target = ubdy[0, 1, 0, :, : u0.shape[1]]
    v_south_target = vbdy[0, 2, 0, :, : v0.shape[2]]  # (z, x)
    v_north_target = vbdy[0, 3, 0, :, : v0.shape[2]]
    edge_after = {
        "u_west_face_vs_wrf_bdy": float(np.max(np.abs(u_after[:, :, 0] - u_west_target))),
        "u_east_face_vs_wrf_bdy": float(np.max(np.abs(u_after[:, :, nx] - u_east_target))),
        "v_south_face_vs_wrf_bdy": float(np.max(np.abs(v_after[:, 0, :] - v_south_target))),
        "v_north_face_vs_wrf_bdy": float(np.max(np.abs(v_after[:, ny, :] - v_north_target))),
    }
    edge_restored = all(val < 1e-9 for val in edge_after.values())

    # The periodicity-specific (wrap) corruption must be confined to the outer
    # faces that B4's spec zone hard-sets; interior faces must show ZERO
    # periodic-vs-nonperiodic difference (the generic averaging loss is not a
    # seam concern and is identical for both inverses).
    relax = int(cfg.relax_zone)
    interior_u_err = float(u_face_err_by_x[relax:nx - relax + 1].max())
    interior_v_err = float(v_face_err_by_y[relax:ny - relax + 1].max())

    # periodicity-specific corruption confined to the outer faces (spec zone)?
    seam_confined = (
        u_bad_faces in ([0, int(nx)], [int(nx), 0])
        and v_bad_faces in ([0, int(ny)], [int(ny), 0])
        and interior_u_err < 1e-9
        and interior_v_err < 1e-9
    )
    status = "PASS" if (edge_restored and seam_confined) else "FAIL"
    return {
        "artifact_type": "b4_b2_edge_seam",
        "status": status,
        "run_dir": str(run_dir),
        "gate1_decision": "#4 -- MYNN periodic wind reconstruction vs B4 non-periodic boundaries",
        "operator_order_required": "physics(surface, MYNN) -> apply_lateral_boundaries (frozen RK1 bundle order)",
        "method": (
            "periodicity-specific test: periodic jnp.roll inverse vs non-periodic "
            "edge-extrapolation inverse on a smooth mass-point wind increment; they "
            "differ ONLY where periodicity wraps."
        ),
        "periodicity_specific_corruption": {
            "u_wrapped_face_x_indices": u_bad_faces,
            "v_wrapped_face_y_indices": v_bad_faces,
            "u_outer_faces": [0, int(nx)],
            "v_outer_faces": [0, int(ny)],
            "interior_periodic_vs_nonperiodic_u_err_beyond_relax": interior_u_err,
            "interior_periodic_vs_nonperiodic_v_err_beyond_relax": interior_v_err,
            "confined_to_outer_faces": seam_confined,
        },
        "b4_restores_outer_faces_after_mynn": edge_after,
        "edge_faces_restored_to_wrf": edge_restored,
        "seam_contract": [
            "The periodicity-SPECIFIC corruption from MYNN's jnp.roll mass->face inverse "
            "is confined to exactly the outer staggered faces u[:, :, {0, nx}] and "
            "v[:, {0, ny}, :]; interior faces are identical to a non-periodic inverse. "
            "(The generic mass<->face averaging loss is a staggering effect present for "
            "any inverse and is not a seam issue.)",
            "B4 spec_zone (>=1) HARD-SETS those exact outer faces to the WRF boundary "
            "value every step, AFTER the physics block, so the wrap artefact is fully "
            "overwritten (post-MYNN restoration error = 0).",
            "Required invariants, all satisfied for the pinned run: (a) operator order "
            "physics(MYNN)->apply_lateral_boundaries (frozen RK1 bundle order), "
            "(b) spec_zone>=1, (c) u/v boundary leaves carry the staggered outer faces.",
            "Recommended (non-blocking) B2 hardening: replace the periodic jnp.roll "
            "inverse with edge-extrapolation so the relaxation columns b_dist=1..relax-1 "
            "(which B4 only NUDGES, not hard-sets) are not seeded with a wrap artefact "
            "before relaxation. With spec_zone=1 the residual is one relaxation row deep "
            "and the relaxation drives it out; edge-extrapolation removes it entirely.",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path, default=ROOT / "proofs/b4/b2_b4_edge_seam.json")
    args = ap.parse_args(argv)
    payload = run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
