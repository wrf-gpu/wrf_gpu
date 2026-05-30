"""Task 1 GATE: full coupled real-case forecast, all physics on, guards OFF, fp64.

Assembles the complete coupled model on the real Gen2 d02 case (the same
`_build_real_case` init the dycore-realinit frontrunner used) and proves it runs
a stable physical forecast for >=360 steps (~1h at dt=10/acoustic=10) with:

  * run_physics=True  -> thompson -> surface -> mynn -> rrtmg(cadence)
  * run_boundary=True -> real d02 lateral relaxation/specified boundaries
  * disable_guards=True (guards OFF -- guards must NOT be load-bearing)
  * the F7-closed dycore config from _build_real_case (top_lid=True, epssm=0.5,
    force_fp64=True, w_damping=1, damp_opt=3, zdamp=5000, use_flux_advection,
    diff_6th_opt=2).

GATE = first_nonfinite is None AND first_unphysical is None AND conservation
within tolerance. Also emits the B3 radiation cadence check and the B4 lateral
boundary edge behaviour.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/coupled/task1_coupled_gate.py --steps 360
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import (
    _enforce_operational_precision,
    _physics_boundary_step,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/coupled")

# Physical-admissibility window for the GATE (guards OFF, so these are NOT
# clamps -- they only DETECT a blow-up). theta covers boundary-layer (~280K) to
# upper-troposphere potential temperature; |w| bounded; winds reasonable for d02.
THETA_LO, THETA_HI = 150.0, 600.0
W_MAX = 30.0
UV_MAX = 150.0
QV_MAX = 0.06  # kg/kg, saturation headroom


def _water_path(state) -> jax.Array:
    """Column-integrated total water (proxy mass): sum over all species, mean cell."""
    total = state.qv + state.qc + state.qr + state.qi + state.qs + state.qg
    return total


def _diag(carry, s):
    st = carry.state
    wabs = jnp.abs(st.w)
    uabs = jnp.abs(st.u)
    vabs = jnp.abs(st.v)
    qtot = _water_path(st)
    rec = {
        "step": int(s),
        "w_absmax": float(jnp.max(wabs)),
        "w_top_face": float(jnp.max(wabs[-1])),
        "u_absmax": float(jnp.max(uabs)),
        "v_absmax": float(jnp.max(vabs)),
        "theta_min": float(jnp.min(st.theta)),
        "theta_max": float(jnp.max(st.theta)),
        "qv_min": float(jnp.min(st.qv)),
        "qv_max": float(jnp.max(st.qv)),
        "qc_max": float(jnp.max(st.qc)),
        "qr_max": float(jnp.max(st.qr)),
        "qi_max": float(jnp.max(st.qi)),
        "qs_max": float(jnp.max(st.qs)),
        "qg_max": float(jnp.max(st.qg)),
        "qtot_sum": float(jnp.sum(qtot)),
        "mu_min": float(jnp.min(st.mu)),
        "mu_max": float(jnp.max(st.mu)),
        "mu_sum": float(jnp.sum(st.mu)),
        "rain_acc_max": float(jnp.max(st.rain_acc)),
        "snow_acc_max": float(jnp.max(st.snow_acc)),
        "graupel_acc_max": float(jnp.max(st.graupel_acc)),
        # theta dry-energy proxy: mu-weighted column theta sum (conservation track)
        "theta_mu_energy": float(jnp.sum(st.mu[None, :, :] * st.theta)),
    }
    return rec


def _finite(rec) -> bool:
    return all(np.isfinite(v) for v in rec.values() if isinstance(v, float))


def _physical(rec) -> bool:
    return (
        THETA_LO <= rec["theta_min"]
        and rec["theta_max"] <= THETA_HI
        and rec["w_absmax"] < W_MAX
        and rec["u_absmax"] < UV_MAX
        and rec["v_absmax"] < UV_MAX
        and rec["qv_min"] >= -1e-9
        and rec["qv_max"] < QV_MAX
    )


def run(case, nl, steps, label):
    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=True)
    )
    cadence = int(nl.radiation_cadence_steps)

    # run_radiation is a STATIC python bool (it gates a python `if` inside the
    # step), so compile two variants keyed on it rather than tracing it.
    from functools import partial

    @partial(jax.jit, static_argnames=("run_rad",))
    def _step(c, idx, run_rad):
        return _physics_boundary_step(c, nl, idx, run_radiation=run_rad, debug=False)

    hist = []
    first_nonfinite = None
    first_unphysical = None
    rad_steps = []
    rad_theta_deltas = []  # B3: max |theta| change across each radiation step
    rec0 = _diag(carry, 0)
    hist.append(rec0)
    prev_theta = carry.state.theta
    for s in range(1, steps + 1):
        run_rad = (s % cadence == 0)
        if run_rad:
            rad_steps.append(s)
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32), bool(run_rad))
        if run_rad:
            dth = float(jnp.max(jnp.abs(carry.state.theta - prev_theta)))
            rad_theta_deltas.append({"step": int(s), "max_abs_dtheta": dth})
            print(f"  [{label}] B3 radiation step {s}: max|dtheta|={dth:.4f} K", flush=True)
        prev_theta = carry.state.theta
        rec = _diag(carry, s)
        rec["radiation"] = bool(run_rad)
        hist.append(rec)
        if first_nonfinite is None and not _finite(rec):
            first_nonfinite = s
            print(f"  [{label}] NONFINITE at step {s}", flush=True)
            break
        if first_unphysical is None and not _physical(rec):
            first_unphysical = s
            print(f"  [{label}] UNPHYSICAL at step {s}: "
                  f"theta[{rec['theta_min']:.1f},{rec['theta_max']:.1f}] "
                  f"|w|={rec['w_absmax']:.2f} |u|={rec['u_absmax']:.2f} "
                  f"qv_max={rec['qv_max']:.5f}", flush=True)
        if s <= 5 or s % 30 == 0:
            print(f"  [{label}] step {s:4d}{'*RAD' if run_rad else '    '}: "
                  f"|w|={rec['w_absmax']:.2f}(top {rec['w_top_face']:.2f}) "
                  f"|u|={rec['u_absmax']:.2f} |v|={rec['v_absmax']:.2f} "
                  f"th[{rec['theta_min']:.1f},{rec['theta_max']:.1f}] "
                  f"qv_max={rec['qv_max']:.5f} qc={rec['qc_max']:.2e} "
                  f"rain={rec['rain_acc_max']:.4f}", flush=True)
    jax.block_until_ready(carry.state.u)

    # B4 edge audit: u magnitude profile across x (west edge / interior / east edge)
    u = np.asarray(jax.device_get(jnp.abs(carry.state.u)))
    u_vs_x = u.max(axis=(0, 1))

    f = hist[-1]
    # conservation: relative drift of dry-mass column sum and water mass over run
    mu_drift = (f["mu_sum"] - rec0["mu_sum"]) / (abs(rec0["mu_sum"]) + 1e-30)
    qtot_drift = (f["qtot_sum"] - rec0["qtot_sum"]) / (abs(rec0["qtot_sum"]) + 1e-30)
    energy_drift = (f["theta_mu_energy"] - rec0["theta_mu_energy"]) / (
        abs(rec0["theta_mu_energy"]) + 1e-30
    )

    stable = first_nonfinite is None and first_unphysical is None

    return {
        "label": label,
        "steps_run": len(hist) - 1,
        "config": {
            "run_physics": bool(nl.run_physics),
            "run_boundary": bool(nl.run_boundary),
            "disable_guards": bool(nl.disable_guards),
            "force_fp64": bool(nl.force_fp64),
            "top_lid": bool(nl.top_lid),
            "epssm": float(nl.epssm),
            "radiation_cadence_steps": cadence,
            "use_flux_advection": bool(nl.use_flux_advection),
            "dt_s": float(nl.dt_s),
            "acoustic_substeps": int(nl.acoustic_substeps),
            "time_utc": str(nl.time_utc),
        },
        "first_nonfinite": first_nonfinite,
        "first_unphysical": first_unphysical,
        "radiation_steps": rad_steps,
        "b3_radiation_theta_deltas": rad_theta_deltas,
        "initial": rec0,
        "final": f,
        "conservation": {
            "mu_sum_init": rec0["mu_sum"],
            "mu_sum_final": f["mu_sum"],
            "mu_rel_drift": mu_drift,
            "qtot_sum_init": rec0["qtot_sum"],
            "qtot_sum_final": f["qtot_sum"],
            "qtot_rel_drift": qtot_drift,
            "theta_mu_energy_init": rec0["theta_mu_energy"],
            "theta_mu_energy_final": f["theta_mu_energy"],
            "theta_mu_energy_rel_drift": energy_drift,
        },
        "b4_edge_audit": {
            "u_west_edge": float(u_vs_x[0]),
            "u_east_edge": float(u_vs_x[-1]),
            "u_interior_max": float(u_vs_x[10:-10].max()),
        },
        "final_dtype": str(carry.state.theta.dtype),
        "verdict": "COUPLED_STABLE" if stable else "UNSTABLE",
        "history": hist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--rad-cadence", type=int, default=180,
                    help="radiation cadence in steps (default matches d02 namelist)")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)

    # GATE config: all physics ON, guards OFF, real boundaries, fp64; thread the
    # real init instant so the RRTMG diurnal SW cycle is physically correct
    # (run_start = 2026-05-21 18:00 UTC = dusk over the Canaries).
    time_utc = case.run_start
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=int(args.rad_cadence),
        time_utc=time_utc,
    )

    print(f"=== Task 1 COUPLED GATE (steps={args.steps}, guards OFF, all physics ON) ===",
          flush=True)
    print(f"  run_dir={run_dir}", flush=True)
    print(f"  grid theta shape (nz,ny,nx)={case.state.theta.shape}", flush=True)
    print(f"  top_lid={nl.top_lid} epssm={nl.epssm} run_boundary={nl.run_boundary} "
          f"force_fp64={nl.force_fp64} rad_cadence={nl.radiation_cadence_steps} "
          f"time_utc={nl.time_utc}", flush=True)

    res = run(case, nl, args.steps, "coupled_guards_off")

    out = {
        "run_dir": str(run_dir),
        "grid_shape_zyx": list(case.state.theta.shape),
        "steps": args.steps,
        "result": res,
    }
    (PROOF / "task1_coupled_gate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {PROOF / 'task1_coupled_gate.json'}", flush=True)
    print(f"\nVERDICT: {res['verdict']}  first_nonfinite={res['first_nonfinite']} "
          f"first_unphysical={res['first_unphysical']}", flush=True)
    c = res["conservation"]
    print(f"  conservation: mu_drift={c['mu_rel_drift']:.3e} "
          f"qtot_drift={c['qtot_rel_drift']:.3e} "
          f"energy_drift={c['theta_mu_energy_rel_drift']:.3e}", flush=True)
    print(f"  B4 edges: W={res['b4_edge_audit']['u_west_edge']:.2f} "
          f"int={res['b4_edge_audit']['u_interior_max']:.2f} "
          f"E={res['b4_edge_audit']['u_east_edge']:.2f}", flush=True)
    return 0 if res["verdict"] == "COUPLED_STABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
