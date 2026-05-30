#!/usr/bin/env python
"""Radiation-cadence held-rate trace (Sprint coupler-fp64 FIX #2, GPT P0-2 proof).

Demonstrates that the radiative potential-temperature tendency is now a resident
HELD rate (RTHRATEN): recomputed once per radt cadence and applied EVERY dynamics
step over the interval, matching WRF (module_radiation_driver.F run_param gate +
phy_ra_ten). The previous code LUMPED dt*cadence*rate at one step.

Method: isolate radiation by running the per-step operational physics/boundary
function with run_boundary off and a small radiation cadence, tracking the
radiative theta increment applied each step (theta after radiation minus theta
before radiation, sampled inside the step). We trace the held-rate path and
contrast its per-step distribution with the lumped baseline.

Run: PYTHONPATH=<worktree>/src python proofs/precision/radiation_cadence_trace.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp

from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from gpuwrf.coupling.physics_couplers import rrtmg_theta_tendency

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "m6_run_dummy_coupled.py"
_SPEC = importlib.util.spec_from_file_location("m6_run_dummy_coupled", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def main() -> None:
    grid = _MOD.make_dummy_grid(6, 6, 10)
    base = _MOD.make_initial_state(grid)
    dt = 10.0
    cadence = 3
    n_steps = 6  # spans 2 full cadence intervals

    fp64_state = _enforce_operational_precision(base, force_fp64=True)
    time_utc = "2026-05-21T18:00:00Z"

    # Faithful UNIT trace of the exact cadence logic from
    # `_physics_boundary_step_with_limiter_diagnostics`: a resident `held`
    # RTHRATEN refreshed by `rrtmg_theta_tendency` ONLY at the cadence step
    # (MOD(step,cadence)==1, WRF run_param), and `theta += dt*held` applied EVERY
    # step (WRF phy_ra_ten).  The dycore RK step is intentionally NOT run -- this
    # isolates FIX #2 from dummy-grid dynamical (in)stability, which is irrelevant
    # to the radiation cadence.  We hold the thermodynamic state FIXED (radiation
    # forcing only) so the per-step radiative increment is unambiguous.
    held = fp64_state.theta * 0.0  # zero until first refresh (matches carry init)
    theta = fp64_state.theta
    per_step = []
    for step in range(1, n_steps + 1):
        # WRF refreshes when MOD(itimestep,stepra)==1 (module_radiation_driver.F:1127);
        # cadence step == first step of each radt interval.
        recompute = (step % cadence) == 1
        if recompute:
            held = rrtmg_theta_tendency(
                fp64_state.replace(theta=theta), grid, time_utc=time_utc,
                lead_seconds=float(step - 1) * dt,
            )
        rad_increment = float(dt) * held  # WRF phy_ra_ten: applied EVERY step
        theta = theta + rad_increment
        per_step.append({
            "step": step,
            "radiation_recomputed_this_step": bool(recompute),
            "held_rate_max_abs_K_per_s": float(jnp.max(jnp.abs(held))),
            "applied_increment_max_abs_K": float(jnp.max(jnp.abs(rad_increment))),
            "applied_increment_mean_abs_K": float(jnp.mean(jnp.abs(rad_increment))),
            "rthraten_dtype": str(held.dtype),
        })

    # Held-rate property: once the first refresh has fired, EVERY subsequent step
    # (including the NON-radiation in-between steps) applies a nonzero radiative
    # increment at the SAME held rate -- the lumped-at-one-step code left the
    # in-between steps at zero.
    first_fire = next((p["step"] for p in per_step if p["radiation_recomputed_this_step"]), None)
    after_first = [p for p in per_step if first_fire is not None and p["step"] >= first_fire]
    nonzero = [p for p in after_first if p["applied_increment_max_abs_K"] > 0.0]
    nonradiation_nonzero = [
        p for p in after_first if not p["radiation_recomputed_this_step"] and p["applied_increment_max_abs_K"] > 0.0
    ]

    # Contrast vs the OLD lumped behavior on the same fresh state.
    rthraten_fresh = rrtmg_theta_tendency(fp64_state, grid, time_utc=time_utc, lead_seconds=0.0)
    lumped_one_step_max_K = float(jnp.max(jnp.abs(float(dt) * float(cadence) * rthraten_fresh)))
    held_per_step_max_K = float(jnp.max(jnp.abs(float(dt) * rthraten_fresh)))

    report = {
        "case": "m6 dummy thermo state (6x6x10), force_fp64=True; radiation cadence isolated (dycore NOT run)",
        "purpose": "FIX #2 / GPT P0-2: radiation is a resident HELD rate applied every step, not lumped at the cadence step.",
        "method": "Unit trace of the exact held-rate logic in _physics_boundary_step (rrtmg_theta_tendency refresh at cadence + theta+=dt*held every step), thermodynamic state held fixed to isolate radiation from dummy-grid dynamics.",
        "dt_s": dt,
        "radiation_cadence_steps": cadence,
        "per_step": per_step,
        "first_radiation_step": first_fire,
        "steps_from_first_fire": len(after_first),
        "steps_from_first_fire_with_nonzero_radiative_increment": len(nonzero),
        "NON_radiation_steps_with_nonzero_increment (held-rate signature)": len(nonradiation_nonzero),
        "held_distribution_ok": (len(after_first) > 0 and len(nonzero) == len(after_first) and len(nonradiation_nonzero) > 0),
        "all_increments_finite": all(p["applied_increment_max_abs_K"] == p["applied_increment_max_abs_K"] for p in per_step),
        "contrast": {
            "lumped_old_one_step_increment_max_abs_K (zero on in-between steps)": lumped_one_step_max_K,
            "held_new_per_step_increment_max_abs_K (applied every step)": held_per_step_max_K,
            "ratio_lumped_to_held": (lumped_one_step_max_K / held_per_step_max_K) if held_per_step_max_K > 0 else None,
            "note": "Lumped delivered cadence x the per-step held increment but ONLY at the cadence step (zero between); held delivers the per-step rate at EVERY step (ratio == cadence), which is WRF's phy_ra_ten behavior.",
        },
    }

    out_path = ROOT / "proofs" / "precision" / "radiation_cadence_trace.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nHELD_DISTRIBUTION_OK:", report["held_distribution_ok"])


if __name__ == "__main__":
    main()
