"""ONE-compile per-step trace of the coupled forecast (onset localizer).

Avoids the per-segment recompile entirely: builds a SINGLE jax.lax.scan over all
steps (like run_forecast_operational_single_scan) but emits a tiny per-step summary
as the scan's stacked output -- theta/w/u/v abs-max, their worst vertical level, and
a finiteness flag.  Memory is bounded (scan carry = one State; outputs are 7 scalars
x n_steps = trivial).  Compile is O(1) in forecast length.  Reuses the SAME
_physics_boundary_step the production scan uses, so the trajectory is the production
trajectory.

This gives the EXACT onset step + the growth curve of every key field + which level
blows first, in ONE compiled program, for any --variant (full / no_physics /
no_rrtmg / no_boundary / no_raydamp / no_wdamp / opentop).

Run:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
    PYTHONPATH=src taskset -c 2-3 python proofs/stability/scan_trace.py \
      --hours 24 --variant full
"""
from __future__ import annotations
import argparse, dataclasses, json, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
import gpuwrf.runtime.operational_mode as om
from gpuwrf.runtime.operational_mode import (
    _physics_boundary_step, _enforce_operational_precision, _steps_for_hours)
from gpuwrf.runtime.operational_state import initial_operational_carry


def _identity_dt(state, dt, *a, **k):
    return state


def _zero_rthraten(state, *a, **k):
    """No-op radiation: return a zero RTHRATEN (theta-shaped, theta dtype)."""
    return jnp.zeros_like(state.theta)


# The state-returning physics adapters (thompson/surface/mynn) are no-op'd by
# substituting an identity; the radiation path is NO LONGER an adapter -- after
# Agent A's rewrite (HEAD 6c45f9c) it is the HELD-RATE primitive
# `rrtmg_theta_tendency` called inside
# `_physics_boundary_step_with_limiter_diagnostics`, so "no_rrtmg" is now done by
# zeroing that tendency (theta += dt*0 = no radiative heating).
_NOOP_TARGET = {
    "thompson_adapter": ("thompson_adapter", _identity_dt),
    "mynn_adapter":     ("mynn_adapter", _identity_dt),
    "surface_adapter":  ("surface_adapter", _identity_dt),
    # legacy alias "rrtmg_adapter" preserved so existing variants below still mean
    # "radiation off"; it now retargets the held-rate tendency primitive.
    "rrtmg_adapter":    ("rrtmg_theta_tendency", _zero_rthraten),
}
_REAL = {n: getattr(om, n) for n in
         ("thompson_adapter", "mynn_adapter", "rrtmg_theta_tendency", "surface_adapter")}

# variant -> (namelist kwargs, set of adapter-noop keys)
VARIANTS = {
    "full":        (dict(run_physics=True), set()),
    "no_physics":  (dict(run_physics=False), set()),
    "no_rrtmg":    (dict(run_physics=True), {"rrtmg_adapter"}),
    "no_thompson": (dict(run_physics=True), {"thompson_adapter"}),
    "no_mynn":     (dict(run_physics=True), {"mynn_adapter"}),
    "no_surface":  (dict(run_physics=True), {"surface_adapter"}),
    "no_boundary": (dict(run_physics=True, run_boundary=False), set()),
    "no_raydamp":  (dict(run_physics=True, damp_opt=0), set()),
    "no_wdamp":    (dict(run_physics=True, w_damping=0), set()),
    "no_damp_all": (dict(run_physics=True, w_damping=0, damp_opt=0), set()),
    "opentop":     (dict(run_physics=True, top_lid=False), set()),
    # --- memory-light bisection on top of no_rrtmg (no 8GB RRTMG transient) ---
    "norad_no_mynn":     (dict(run_physics=True), {"rrtmg_adapter", "mynn_adapter"}),
    "norad_no_thompson": (dict(run_physics=True), {"rrtmg_adapter", "thompson_adapter"}),
    "norad_no_surface":  (dict(run_physics=True), {"rrtmg_adapter", "surface_adapter"}),
    "norad_only_mynn":   (dict(run_physics=True), {"rrtmg_adapter", "thompson_adapter", "surface_adapter"}),
    # top-BC / damping sensitivity on the (memory-light) no_rrtmg blowup:
    "norad_no_raydamp":  (dict(run_physics=True, damp_opt=0), {"rrtmg_adapter"}),
    "norad_no_damp_all": (dict(run_physics=True, w_damping=0, damp_opt=0), {"rrtmg_adapter"}),
    # --- damping-coverage / coupled-path probes (residual w-growth, 2026-05-30) ---
    # strengthen the Rayleigh/w damping on the memory-light coupled path: if the
    # residual w-growth is a damping-coverage gap, this should suppress it.
    "norad_strong_damp": (dict(run_physics=True, zdamp=10000.0, dampcoef=0.5),
                          {"rrtmg_adapter"}),
}


def _absmax_and_k(field):
    """abs-max over the whole field and the vertical level (axis 0) of that max."""
    af = jnp.abs(field)
    lev = jnp.max(af.reshape(af.shape[0], -1), axis=1)  # (nz,)
    return jnp.max(af), jnp.argmax(lev).astype(jnp.int32)


def build_scan(nl, steps, run_physics, cadence):
    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(carry, step_index):
        run_radiation = (jnp.equal(jnp.mod(step_index, cadence), 0)
                         if run_physics else False)
        nxt = _physics_boundary_step(carry, nl, step_index, run_radiation=run_radiation, debug=False)
        s = nxt.state
        th_m, th_k = _absmax_and_k(s.theta)
        w_m, w_k = _absmax_and_k(s.w)
        u_m, u_k = _absmax_and_k(s.u)
        v_m, v_k = _absmax_and_k(s.v)
        finite = (jnp.isfinite(s.theta).all() & jnp.isfinite(s.w).all()
                  & jnp.isfinite(s.u).all() & jnp.isfinite(s.v).all()
                  & jnp.isfinite(s.ph).all() & jnp.isfinite(s.qv).all())
        out = jnp.stack([th_m, w_m, u_m, v_m,
                         th_k.astype(jnp.float64), w_k.astype(jnp.float64),
                         u_k.astype(jnp.float64), v_k.astype(jnp.float64),
                         finite.astype(jnp.float64)])
        return nxt, out

    return body, indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--variant", type=str, default="full")
    ap.add_argument("--out", type=str, default="proofs/stability/scan_trace")
    args = ap.parse_args()
    if args.variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {list(VARIANTS)}")
    nlkw, noops = VARIANTS[args.variant]
    # restore all real adapters first, then apply each requested noop to its target
    for sym, real in _REAL.items():
        setattr(om, sym, real)
    for key in noops:
        target_sym, repl = _NOOP_TARGET[key]
        setattr(om, target_sym, repl)

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    cadence = 180
    base = dict(run_physics=True, run_boundary=True, disable_guards=True,
                radiation_cadence_steps=cadence, time_utc=case.run_start)
    base.update(nlkw)
    nl = dataclasses.replace(case.namelist, **base)
    dt_s = 10.0
    steps = int(round(args.hours * 3600.0 / dt_s))

    body, indices = build_scan(nl, steps, bool(nl.run_physics), cadence)
    init = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def run(carry):
        c, outs = jax.lax.scan(body, carry, indices)
        return c, outs

    t0 = time.perf_counter()
    final, outs = run(init)
    jax.block_until_ready(outs)
    wall = time.perf_counter() - t0
    outs = np.asarray(jax.device_get(outs))  # (steps, 9)
    names = ["theta_absmax", "w_absmax", "u_absmax", "v_absmax",
             "theta_k", "w_k", "u_k", "v_k", "finite"]
    finite_col = outs[:, 8]
    first_bad = int(np.argmin(finite_col)) if (finite_col < 0.5).any() else None
    # onset step = first step where finite flag drops to 0
    onset_step = (first_bad + 1) if (first_bad is not None and finite_col[first_bad] < 0.5) else None

    out = {"scope": "ONE-compile per-step coupled trace",
           "variant": args.variant, "run_dir": str(run_dir), "hours": args.hours,
           "steps": steps, "wall_s": round(wall, 1),
           "namelist": {"run_physics": bool(nl.run_physics), "run_boundary": bool(nl.run_boundary),
                        "top_lid": bool(nl.top_lid), "epssm": float(nl.epssm),
                        "w_damping": int(nl.w_damping), "damp_opt": int(nl.damp_opt),
                        "noops": sorted(noops)},
           "onset_step": onset_step,
           "onset_hours": (onset_step * dt_s / 3600.0) if onset_step else None,
           "all_finite_to_end": bool((finite_col >= 0.5).all())}
    # hourly subsample of the growth curve (every 360 steps) + the 20 steps around onset
    def row(i):
        return {"step": i + 1, "hours": round((i + 1) * dt_s / 3600.0, 3),
                **{names[j]: (round(float(outs[i, j]), 3) if j < 4 else int(outs[i, j]))
                   for j in range(9)}}
    hourly = [row(i) for i in range(359, steps, 360)]
    near = []
    if onset_step:
        lo = max(0, onset_step - 12); hi = min(steps, onset_step + 3)
        near = [row(i) for i in range(lo, hi)]
    out["hourly_curve"] = hourly
    out["near_onset"] = near
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fn = Path(f"{args.out}_{args.variant}.json")
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.variant}] onset_step={onset_step} onset_h={out['onset_hours']} "
          f"all_finite={out['all_finite_to_end']} wall={wall:.1f}s", flush=True)
    for r in hourly:
        print(f"  {r['hours']:.1f}h fin={r['finite']} th={r['theta_absmax']}@k{r['theta_k']} "
              f"w={r['w_absmax']}@k{r['w_k']} u={r['u_absmax']}@k{r['u_k']} "
              f"v={r['v_absmax']}@k{r['v_k']}", flush=True)
    print(f"wrote {fn}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
