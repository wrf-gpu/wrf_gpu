"""Locate WHERE the residual surface w@k0 lives and whether it is a real
terrain-flow feature or a numerical artifact.

Runs the FIXED coupled model (no_rrtmg, memory-light) as one scan to a target
hour, then on the final state reports:
  - the (k0) global |w| max and its (j,i) location
  - the terrain gradient dz/dx, dz/dy at that cell
  - the low-3-level u/v there and the recomputed kinematic w_surface
  - whether the large w cells coincide with the steepest terrain
This tells us if the kinematic BC is faithfully tracking the (possibly large)
low-level flow over steep terrain, or if a spurious low-level wind perturbation
is being amplified.
"""
from __future__ import annotations
import argparse, dataclasses, json
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
import gpuwrf.runtime.operational_mode as om
from gpuwrf.runtime.operational_mode import _physics_boundary_step, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry


def _zero_rthraten(state, *a, **k):
    return jnp.zeros_like(state.theta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--out", type=str, default="proofs/stability/surface_w_hotspot.json")
    args = ap.parse_args()

    # radiation off (memory-light); MYNN+surface+thompson ON
    om.rrtmg_theta_tendency = _zero_rthraten

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    cadence = 180
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=cadence, time_utc=case.run_start,
    )
    dt_s = 10.0
    steps = int(round(args.hours * 3600.0 / dt_s))
    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(carry, step_index):
        return _physics_boundary_step(carry, nl, step_index, run_radiation=False, debug=False), None

    init = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def run(carry):
        c, _ = jax.lax.scan(body, carry, indices)
        return c

    final = run(init)
    jax.block_until_ready(final.state.w)
    s = final.state

    w0 = np.asarray(s.w[0])               # (ny, nx) surface face
    aw0 = np.abs(w0)
    jmax, imax = np.unravel_index(int(aw0.argmax()), aw0.shape)

    # terrain ht = surface geopotential / g (ph_total bottom face)
    g = 9.81
    ht = np.asarray(s.ph_total[0]) / g  # (ny, nx)

    u = np.asarray(s.u)  # (nz, ny, nx+1)
    v = np.asarray(s.v)  # (nz, ny+1, nx)

    def around(arr2d, j, i, r=1):
        j0, j1 = max(0, j - r), min(arr2d.shape[0], j + r + 1)
        i0, i1 = max(0, i - r), min(arr2d.shape[1], i + r + 1)
        return arr2d[j0:j1, i0:i1].tolist()

    # terrain gradients (cell-centred, edge-padded)
    dz_dx = np.zeros_like(ht); dz_dy = np.zeros_like(ht)
    dz_dx[:, 1:-1] = 0.5 * (ht[:, 2:] - ht[:, :-2])
    dz_dy[1:-1, :] = 0.5 * (ht[2:, :] - ht[:-2, :])

    # near-surface u/v at the hotspot (lowest 3 mass levels, mass-collocated)
    u_mass = 0.5 * (u[:, :, :-1] + u[:, :, 1:])  # (nz, ny, nx)
    v_mass = 0.5 * (v[:, :-1, :] + v[:, 1:, :])  # (nz, ny, nx)

    # correlation of |w0| with terrain steepness |grad ht|
    grad = np.hypot(dz_dx, dz_dy)
    finite = np.isfinite(aw0) & np.isfinite(grad)
    if finite.sum() > 10:
        corr = float(np.corrcoef(aw0[finite].ravel(), grad[finite].ravel())[0, 1])
    else:
        corr = None

    out = {
        "scope": "surface w@k0 hotspot localization (FIXED coupler, no_rrtmg)",
        "hours": args.hours, "steps": steps, "run_dir": str(run_dir),
        "w0_absmax": float(aw0.max()),
        "w0_argmax_ji": [int(jmax), int(imax)],
        "w0_around_hotspot": around(w0, jmax, imax, 2),
        "ht_at_hotspot": float(ht[jmax, imax]),
        "ht_around_hotspot": around(ht, jmax, imax, 2),
        "grad_ht_at_hotspot_m_per_cell": [float(dz_dx[jmax, imax]), float(dz_dy[jmax, imax])],
        "max_grad_ht_m_per_cell": float(grad.max()),
        "u_mass_low3_at_hotspot": [float(u_mass[k, jmax, imax]) for k in range(3)],
        "v_mass_low3_at_hotspot": [float(v_mass[k, jmax, imax]) for k in range(3)],
        "corr_absw0_vs_terrain_steepness": corr,
        "w0_p99": float(np.percentile(aw0, 99)),
        "w0_p50": float(np.percentile(aw0, 50)),
        "w0_mean": float(aw0.mean()),
        "n_cells_gt_30ms": int((aw0 > 30).sum()),
        "n_cells_total": int(aw0.size),
        "u_global_absmax": float(np.abs(u).max()),
        "v_global_absmax": float(np.abs(v).max()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
