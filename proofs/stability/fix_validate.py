"""Validate the FIX direction (probe only, no src edit):

Hypothesis: MYNN's _mass_to_w_face round-trip re-introduces nonzero w at the rigid-lid
top face (k=nz), which seeds the 15.04h top-of-column w eruption.  If we wrap the real
mynn_adapter so that it RESTORES w to its pre-MYNN value (MYNN does not solve w, so it
should not change w at all), the rigid lid is preserved and the blowup should vanish.

Runs the SAME single-compile scan_trace machinery with a patched mynn_adapter:
  mynn_fixed(state, dt, grid) = real_mynn(state, dt, grid).replace(w=state.w)

This is the minimal correct behavior (MYNN must not alter w) and a direct test of
whether the top-w-face corruption is load-bearing for the instability.
"""
from __future__ import annotations
import argparse, dataclasses, json, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
import gpuwrf.runtime.operational_mode as om
from gpuwrf.runtime.operational_mode import (
    _physics_boundary_step, _enforce_operational_precision)
from gpuwrf.runtime.operational_state import initial_operational_carry

_real_mynn = om.mynn_adapter
_real_rrtmg = om.rrtmg_adapter


def _identity_dt(state, dt, *a, **k):
    return state


def _mynn_preserve_w(state, dt, grid=None):
    """Real MYNN, but restore w to its input value (MYNN does not solve w)."""
    out = _real_mynn(state, dt, grid)
    return out.replace(w=state.w)


def _absmax_and_k(field):
    af = jnp.abs(field)
    lev = jnp.max(af.reshape(af.shape[0], -1), axis=1)
    return jnp.max(af), jnp.argmax(lev).astype(jnp.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=17.0)
    args = ap.parse_args()
    # Memory-light: keep RRTMG off (irrelevant to the blowup); MYNN -> preserve-w fix.
    om.rrtmg_adapter = _identity_dt
    om.mynn_adapter = _mynn_preserve_w

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    cadence = 180
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=True, radiation_cadence_steps=cadence,
                             time_utc=case.run_start)
    dt_s = 10.0
    steps = int(round(args.hours * 3600.0 / dt_s))
    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(carry, step_index):
        nxt = _physics_boundary_step(carry, nl, step_index, run_radiation=False, debug=False)
        s = nxt.state
        w_m, w_k = _absmax_and_k(s.w)
        u_m, u_k = _absmax_and_k(s.u)
        th_m, th_k = _absmax_and_k(s.theta)
        finite = (jnp.isfinite(s.theta).all() & jnp.isfinite(s.w).all()
                  & jnp.isfinite(s.u).all() & jnp.isfinite(s.ph).all())
        return nxt, jnp.stack([w_m, u_m, th_m, w_k.astype(jnp.float64),
                               u_k.astype(jnp.float64), th_k.astype(jnp.float64),
                               finite.astype(jnp.float64)])

    init = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def run(carry):
        return jax.lax.scan(body, carry, indices)

    t0 = time.perf_counter()
    _, outs = run(init)
    jax.block_until_ready(outs)
    wall = time.perf_counter() - t0
    outs = np.asarray(jax.device_get(outs))
    fin = outs[:, 6]
    onset = int(np.argmin(fin)) + 1 if (fin < 0.5).any() else None
    out = {"scope": "FIX validation: MYNN preserve-w (do not let MYNN alter w)",
           "hours": args.hours, "steps": steps, "wall_s": round(wall, 1),
           "onset_step": onset, "onset_hours": (onset * dt_s / 3600.0) if onset else None,
           "all_finite_to_end": bool((fin >= 0.5).all()),
           "hourly": [{"h": round((i + 1) * dt_s / 3600.0, 2),
                       "fin": int(outs[i, 6]),
                       "w": round(float(outs[i, 0]), 3), "w_k": int(outs[i, 3]),
                       "u": round(float(outs[i, 1]), 3), "u_k": int(outs[i, 4])}
                      for i in range(359, steps, 360)]}
    Path("proofs/stability/fix_validate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[fix=preserve-w] onset_step={onset} onset_h={out['onset_hours']} "
          f"all_finite={out['all_finite_to_end']} wall={wall:.1f}s", flush=True)
    for r in out["hourly"]:
        print(f"  {r['h']:.0f}h fin={r['fin']} w={r['w']}@k{r['w_k']} u={r['u']}@k{r['u_k']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
