"""Dycore-only (run_physics=False) long-run walk to localize the 9-24h blowup.

KEY: with run_physics=False the _advance_chunk graph does NOT recompile on varying
start_step (proved in recompile_probe.log: start=4/7/4 are 0.1s cache hits), and the
compile is cheap (no RRTMG transient), so a full 24h hour-by-hour walk is fast after
ONE compile.  This isolates whether the dycore + lateral boundaries ALONE (the
ROOT_CAUSE 'Mode B' periodic-advection-on-a-LAM hypothesis) goes non-finite in 9-24h,
which the prior dycore-realinit proofs only tested to 1h.

If dycore-only ALSO blows up in 9-24h -> the blowup is structural (advection/boundary/
acoustic/damping), independent of physics.  If dycore-only stays finite to 24h -> the
blowup REQUIRES physics (radiation lump / surface-PBL / microphysics).

Records per-LEVEL abs-max of u,v,w,theta,ph and the (k,j,i) of the global |max|.
Writes JSONL incrementally.
"""
from __future__ import annotations
import argparse, dataclasses, json, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

FIELDS = ("u", "v", "w", "theta", "ph")


def rep(state, name):
    a = np.asarray(jax.device_get(getattr(state, name)), dtype=np.float64)
    fm = np.isfinite(a)
    fv = a[fm]
    d = {"finite": bool(fm.all())}
    if fv.size:
        d["absmax"] = float(np.abs(fv).max())
        masked = np.where(fm, np.abs(a), -np.inf)
        d["absmax_idx"] = [int(x) for x in np.unravel_index(int(np.argmax(masked)), a.shape)]
        lev = np.where(fm, np.abs(a), 0.0).reshape(a.shape[0], -1).max(axis=1)
        d["worst_level"] = int(np.argmax(lev))
        d["per_level_absmax"] = [round(float(x), 3) for x in lev]
    if not d["finite"]:
        bad = (~fm).reshape(a.shape[0], -1).sum(axis=1)
        d["first_nonfinite_level"] = int(np.argwhere(bad > 0).ravel()[0])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seg-steps", type=int, default=360)
    ap.add_argument("--run-boundary", type=int, default=1)
    ap.add_argument("--out", type=str, default="proofs/stability/dycore_only_walk")
    args = ap.parse_args()
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    dt_s = 10.0
    steps = int(round(args.hours * 3600.0 / dt_s))
    seg = args.seg_steps
    nl = dataclasses.replace(case.namelist, run_physics=False,
                             run_boundary=bool(args.run_boundary), disable_guards=True,
                             radiation_cadence_steps=180, time_utc=case.run_start)
    tag = "bndy" if args.run_boundary else "nobndy"
    fh = Path(f"{args.out}_{tag}.jsonl").open("w")
    fh.write(json.dumps({"kind": "meta", "run_dir": str(run_dir), "run_boundary": bool(args.run_boundary),
                         "hours": args.hours, "seg": seg, "physics": False,
                         "top_lid": bool(nl.top_lid), "epssm": float(nl.epssm),
                         "w_damping": int(nl.w_damping), "damp_opt": int(nl.damp_opt)}) + "\n"); fh.flush()
    carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))
    start = 1; t0 = time.perf_counter()
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32), n_steps=int(n), cadence=180)
        jax.block_until_ready(carry.state.theta)
        end = start + n - 1; lead = end * dt_s / 3600.0
        reps = {f: rep(carry.state, f) for f in FIELDS}
        finite = all(r["finite"] for r in reps.values())
        rec = {"kind": "hour", "lead_hours": round(lead, 2), "finite": finite, "wall_s": round(time.perf_counter()-t0,1)}
        for f in FIELDS:
            rec[f] = {"absmax": reps[f].get("absmax"), "k": reps[f].get("worst_level"),
                      "idx": reps[f].get("absmax_idx"), "finite": reps[f]["finite"]}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"[{tag}] {lead:.1f}h fin={finite} "
              f"u={reps['u'].get('absmax'):.1f}@k{reps['u'].get('worst_level')} "
              f"v={reps['v'].get('absmax'):.1f}@k{reps['v'].get('worst_level')} "
              f"w={reps['w'].get('absmax'):.1f}@k{reps['w'].get('worst_level')} "
              f"th={reps['theta'].get('absmax'):.0f}@k{reps['theta'].get('worst_level')} "
              f"wall={rec['wall_s']}s", flush=True)
        if not finite:
            print(f"  >>> {tag} NON-FINITE at {lead:.1f}h: "
                  f"{[f for f in FIELDS if not reps[f]['finite']]}", flush=True)
            break
        start += n
    fh.close()
    print(f"DONE {tag}", flush=True)


if __name__ == "__main__":
    main()
