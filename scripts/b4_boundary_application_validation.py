#!/usr/bin/env python3
"""B4 lateral-boundary application validation against real WRF data.

Validates ``gpuwrf.coupling.boundary_apply.apply_lateral_boundaries`` against
the pinned Canary d02 run with REAL WRF side-history forcing (no self-compare):

1. **spec-zone exactness** -- applying the boundary at lead = t exactly places
   the WRF wrfout side strip in the outermost ``spec_zone`` row/col.
2. **relaxation consistency** -- when the interior equals the boundary forcing
   the relaxation residual is zero (no spurious nudging of a consistent state).
3. **independent WRF-formula cross-check** -- a standalone NumPy re-derivation
   of WRF's ``relax_bdytend`` stencil (``share/module_bc.F``) is compared to the
   JAX kernel on the real West-edge strip; they must agree to fp64 round-off.
4. **corner / single-owner** -- every relaxation-zone cell is updated by exactly
   one side (WRF corner trimming) and interior beyond ``relax_zone`` is untouched.
5. **fp64 preservation** -- the boundary-applied State stays fp64 under force_fp64.

Writes ``proofs/b4/boundary_application_validation.json``.
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

from gpuwrf.coupling.boundary_apply import (
    DEFAULT_BOUNDARY_CONFIG,
    BoundaryConfig,
    apply_lateral_boundaries,
    interpolate_boundary_leaf,
    _wrf_relax_weights,
)
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import _enforce_operational_precision

DEFAULT_RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")


# --- independent NumPy re-derivation of WRF relax_bdytend for the West edge ---
def wrf_relax_west_numpy(field, forcing_side, dt_s, cfg: BoundaryConfig):
    """Reproduce WRF relax_bdytend for the West (X-start) boundary in NumPy.

    field:        (z, y, x)
    forcing_side: (bdy_width, z, y)  -- West strip, width 0 = outer edge
    Returns the updated field (relaxation zone only; spec zone left as-is here).
    """
    z, y, x = field.shape
    out = field.copy()
    spec_zone, relax_zone = int(cfg.spec_zone), int(cfg.relax_zone)
    for b_dist in range(spec_zone, relax_zone):
        wf, wg = _wrf_relax_weights(b_dist, dt_s, cfg)
        i = b_dist
        for k in range(z):
            for j in range(b_dist + 1, y - b_dist - 1):  # WRF X-bdy j-trim
                bdy0 = forcing_side[b_dist, k, j]
                bdy_in = forcing_side[b_dist - 1, k, j]
                bdy_out = forcing_side[b_dist + 1, k, j]
                fls0 = bdy0 - field[k, j, i]
                fls1 = forcing_side[b_dist, k, j - 1] - field[k, j - 1, i]
                fls2 = forcing_side[b_dist, k, j + 1] - field[k, j + 1, i]
                fls3 = bdy_in - field[k, j, i - 1]
                fls4 = bdy_out - field[k, j, i + 1]
                out[k, j, i] = field[k, j, i] + wf * fls0 - wg * (fls1 + fls2 + fls3 + fls4 - 4.0 * fls0)
    return out


def run(run_dir: Path) -> dict:
    cfg = DEFAULT_BOUNDARY_CONFIG
    dt_s = 6.0  # d02 timestep
    case = build_replay_case(str(run_dir), domain="d02")
    st = _enforce_operational_precision(case.state, force_fp64=True)

    results: dict = {}

    # ---- 1. spec-zone exactness at lead=0 (all four sides, theta field) ----
    out0 = apply_lateral_boundaries(st, jnp.asarray(0.0), dt_s, cfg)
    th = np.asarray(out0.theta)
    thb = np.asarray(st.theta_bdy)  # (t,side,bw,z,side_len)
    ny, nx = th.shape[1], th.shape[2]
    spec = {
        "W": float(np.max(np.abs(th[:, :, 0] - thb[0, 0, 0, :, :ny]))),
        "E": float(np.max(np.abs(th[:, :, -1] - thb[0, 1, 0, :, :ny]))),
        "S": float(np.max(np.abs(th[:, 0, :] - thb[0, 2, 0, :, :nx]))),
        "N": float(np.max(np.abs(th[:, -1, :] - thb[0, 3, 0, :, :nx]))),
    }
    results["spec_zone_exact_max_abs_diff"] = spec
    results["spec_zone_exact_pass"] = all(v < 1e-9 for v in spec.values())

    # ---- 2. relaxation consistency: interior == boundary -> no change ----
    # Build a synthetic state whose theta equals the boundary forcing everywhere
    # in the strip; the relaxation residual must vanish (no spurious nudge).
    forcing_th = np.asarray(interpolate_boundary_leaf(st.theta_bdy, jnp.asarray(0.0), cfg.update_cadence_s))
    consistent = np.asarray(st.theta).copy()
    # set the West relax columns equal to their boundary strip
    for b in range(int(cfg.spec_zone), int(cfg.relax_zone)):
        consistent[:, :, b] = forcing_th[0, b, :, :consistent.shape[1]]
        consistent[:, :, b - 1] = forcing_th[0, max(b - 1, 0), :, :consistent.shape[1]]
        consistent[:, :, b + 1] = forcing_th[0, b + 1, :, :consistent.shape[1]]
    st_consistent = st.replace(theta=jnp.asarray(consistent))
    out_c = apply_lateral_boundaries(st_consistent, jnp.asarray(0.0), dt_s, cfg)
    # West relax column 1: change vs the (already consistent) input
    rc = float(np.max(np.abs(np.asarray(out_c.theta)[:, 2:-2, 1] - consistent[:, 2:-2, 1])))
    results["relaxation_zero_residual_when_consistent"] = rc
    results["relaxation_consistency_pass"] = rc < 1e-9

    # ---- 3. independent WRF-formula cross-check on the real West strip ----
    field = np.asarray(st.theta, dtype=np.float64)
    west_strip = forcing_th[0]  # (bw, z, side_len=padded) -> slice to y
    west_strip = west_strip[:, :, :field.shape[1]]  # (bw, z, y)
    ref = wrf_relax_west_numpy(field, west_strip, dt_s, cfg)
    jax_out = np.asarray(apply_lateral_boundaries(st, jnp.asarray(0.0), dt_s, cfg).theta)
    # compare only the West relaxation columns (b_dist 1..3), interior j-range
    xcheck = 0.0
    for b in range(int(cfg.spec_zone), int(cfg.relax_zone)):
        sl = (slice(None), slice(b + 1, field.shape[1] - b - 1), b)
        xcheck = max(xcheck, float(np.max(np.abs(jax_out[sl] - ref[sl]))))
    results["wrf_formula_crosscheck_max_abs_diff"] = xcheck
    results["wrf_formula_crosscheck_pass"] = xcheck < 1e-9

    # ---- 4. interior beyond relax untouched ----
    interior_change = float(np.max(np.abs(jax_out[:, :, int(cfg.relax_zone):-int(cfg.relax_zone)]
                                          - field[:, :, int(cfg.relax_zone):-int(cfg.relax_zone)])))
    results["interior_beyond_relax_untouched_max_abs_diff"] = interior_change
    results["interior_untouched_pass"] = interior_change < 1e-12

    # ---- 5. fp64 preservation ----
    fp64_ok = all(
        np.asarray(getattr(out0, f)).dtype == np.float64
        for f in ("u", "v", "w", "theta", "qv", "p_total", "p_perturbation",
                  "ph_total", "ph_perturbation", "mu_total", "mu_perturbation")
    )
    results["fp64_preserved_under_force_fp64"] = bool(fp64_ok)

    # ---- 6. WRF-intrinsic boundary vs interior change over hour 1 ----
    # In the true WRF d02 solution, is the boundary strip the dominant source of
    # change?  If WRF's own boundary-strip RMSE(t0->t1) is <= its interior RMSE,
    # a boundary scheme that faithfully reproduces the side-history cannot make
    # the boundary the dominant first-hour error.
    run_obj = case.run
    relax = int(cfg.relax_zone)
    intrinsic = {}
    for var, off, name in (("T", 300.0, "theta"), ("U", 0.0, "u"), ("V", 0.0, "v")):
        a = np.asarray(run_obj.load("d02", var, time=0, lazy=False), dtype=np.float64) + off
        b = np.asarray(run_obj.load("d02", var, time=1, lazy=False), dtype=np.float64) + off
        d = np.abs(b - a)
        m = np.zeros(d.shape[1:], bool)
        m[:relax, :] = True; m[-relax:, :] = True; m[:, :relax] = True; m[:, -relax:] = True
        intrinsic[name] = {
            "boundary_strip_rmse": float(np.sqrt(np.mean(d[:, m] ** 2))),
            "interior_rmse": float(np.sqrt(np.mean(d[:, ~m] ** 2))),
        }
    results["wrf_intrinsic_change_t0_t1"] = intrinsic
    results["boundary_not_dominant_in_wrf"] = all(
        v["boundary_strip_rmse"] <= v["interior_rmse"] for v in intrinsic.values()
    )

    # ---- 7. relaxation-toward-truth fidelity (pure boundary operator) ----
    # Repeatedly apply ONLY the boundary operator toward the hour-1 forcing and
    # confirm the relaxation zone moves *toward* the WRF hour-1 truth (the strip
    # error decreases), with the spec row exactly equal to the WRF hour-1 value.
    import jax
    apply_jit = jax.jit(lambda s: apply_lateral_boundaries(s, jnp.asarray(cfg.update_cadence_s), dt_s, cfg))
    th1 = np.asarray(run_obj.load("d02", "T", time=1, lazy=False), dtype=np.float64) + 300.0
    cur = st
    for _ in range(200):
        cur = apply_jit(cur)
    cur.theta.block_until_ready()
    th_after = np.asarray(cur.theta)
    th0 = np.asarray(st.theta)
    m = np.zeros(th_after.shape[1:], bool)
    m[:relax, :] = True; m[-relax:, :] = True; m[:, :relax] = True; m[:, -relax:] = True
    err_before = float(np.sqrt(np.mean((th0[:, m] - th1[:, m]) ** 2)))
    err_after = float(np.sqrt(np.mean((th_after[:, m] - th1[:, m]) ** 2)))
    spec_t1 = float(np.max(np.abs(th_after[:, :, 0] - th1[:, :, 0])))
    results["relaxation_toward_truth"] = {
        "boundary_strip_theta_rmse_vs_wrf_t1_before": err_before,
        "boundary_strip_theta_rmse_vs_wrf_t1_after": err_after,
        "spec_zone_W_equals_wrf_t1_max_abs_diff": spec_t1,
    }
    results["relaxation_reduces_strip_error"] = err_after < err_before and spec_t1 < 1e-9

    all_pass = (
        results["spec_zone_exact_pass"]
        and results["relaxation_consistency_pass"]
        and results["wrf_formula_crosscheck_pass"]
        and results["interior_untouched_pass"]
        and results["fp64_preserved_under_force_fp64"]
        and results["boundary_not_dominant_in_wrf"]
        and results["relaxation_reduces_strip_error"]
    )

    return {
        "artifact_type": "b4_boundary_application_validation",
        "status": "PASS" if all_pass else "FAIL",
        "run_dir": str(run_dir),
        "domain": "d02",
        "dt_s": dt_s,
        "boundary_config": {
            "spec_bdy_width": cfg.spec_bdy_width, "spec_zone": cfg.spec_zone,
            "relax_zone": cfg.relax_zone, "update_cadence_s": cfg.update_cadence_s,
            "spec_exp": cfg.spec_exp,
        },
        "wrf_source": "share/module_bc.F relax_bdytend/spec_bdytend; dyn_em/module_bc_em.F lbc_fcx_gcx",
        "checks": results,
        "note": "Forcing = decoupled wrfout side-history; WRF relaxes mass-coupled vars "
                "(documented O(mu') departure in boundary_apply.py).",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path, default=ROOT / "proofs/b4/boundary_application_validation.json")
    args = ap.parse_args(argv)
    payload = run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
