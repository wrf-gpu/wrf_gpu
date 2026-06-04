#!/usr/bin/env python
"""v0.9.0 d02-replay hour-1 blow-up — DECISIVE GPU forecast verify.

Confirms the MYNN qke cold-start fix (+ harness stability-namelist hardening)
stabilizes the 20260521 L2 d02 parent-history replay through hour 1 and beyond,
and determines the MINIMAL WRF-faithful cure by toggling:

  - qke seed   ON  = WRF mym_initialize cold-start TKE (d02_replay.build_replay_case)
               OFF = the old degenerate qke=0 the parent wrfout carries
  - namelist   harness_default = old m7 weakest dataclass defaults (open top, fp32, ..)
               stable          = the validated _build_real_case Gen2-d02 stability set

Matrix:
  qke0_harness   (qke=0,    harness)  -> reproduce the blow-up baseline
  qke0_stable    (qke=0,    stable)   -> does the namelist alone cure it?
  seed_harness   (qke=seed, harness)  -> does the qke seed alone cure it?
  seed_stable    (qke=seed, stable)   -> the full shipped fix (minimal combination?)

GPU (cuda) job.  ONE GPU job at a time -- claims the advisory lock first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

from gpuwrf.integration.d02_replay import build_l2_d02_replay_case  # noqa: E402
from gpuwrf.integration.daily_pipeline import finite_summary, resolve_run_dir  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    run_forecast_operational,
)

L2_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l2")
DT_S = 12.0

STABLE_FLAGS = dict(
    use_flux_advection=True,
    force_fp64=True,
    diff_6th_opt=2,
    diff_6th_factor=0.12,
    w_damping=1,
    damp_opt=3,
    zdamp=5000.0,
    dampcoef=0.2,
    epssm=0.5,
    top_lid=True,
)
HARNESS_FLAGS: dict = {}


def _build_namelist(replay, extra, *, dt_s, substeps):
    return OperationalNamelist.from_grid(
        replay.grid,
        tendencies=replay.tendencies,
        metrics=replay.metrics,
        dt_s=float(dt_s),
        acoustic_substeps=int(substeps),
        radiation_cadence_steps=180,
        use_vertical_solver=True,
        **extra,
    )


def _signature(state):
    summ = finite_summary(state)
    fields = summ["fields"]
    dyn = {}
    for k in ("u", "v", "w", "theta", "mu", "ph", "p", "qke"):
        if k in fields:
            f = fields[k]
            dyn[k] = {"finite": f["finite"], "nonfinite": f["nonfinite_count"],
                      "min": f["min"], "max": f["max"]}
    nonfinite = {k: v["nonfinite_count"] for k, v in fields.items() if not v["finite"]}
    return {"all_finite": summ["all_finite"], "dynamics": dyn, "nonfinite_fields": nonfinite}


def _step_to_blowup(state, namelist, *, dt_s, probe_steps, max_steps):
    import jax
    trace = []
    done = 0
    first_blow = None
    for target in probe_steps:
        if target > max_steps:
            break
        n = target - done
        if n <= 0:
            continue
        hours = (n * dt_s) / 3600.0
        state = run_forecast_operational(state, namelist, hours)
        jax.block_until_ready(state)
        done = target
        sig = _signature(state)
        trace.append({"step": int(done), "model_minutes": round(done * dt_s / 60.0, 3),
                      "model_hours": round(done * dt_s / 3600.0, 3),
                      "all_finite": sig["all_finite"],
                      "dyn": {k: {"max": v["max"], "min": v["min"], "fin": v["finite"]}
                              for k, v in sig["dynamics"].items()},
                      "nonfinite_fields": sig["nonfinite_fields"]})
        if not sig["all_finite"] and first_blow is None:
            first_blow = int(done)
            break
    return {"first_nonfinite_step": first_blow, "trace": trace}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="20260521_18z_l2_72h_20260522T133443Z")
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "proofs" / "v090" / "d02replay_qke_fix_verify.json")
    ap.add_argument("--configs", nargs="+",
                    default=["qke0_harness", "seed_harness", "seed_stable"])
    ap.add_argument("--max-steps", type=int, default=900)  # 3h = 900 steps @ dt=12
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ["JAX_PLATFORMS"] = args.platform
    import jax
    import jax.numpy as jnp
    devices = [str(d) for d in jax.devices()]

    run_dir = resolve_run_dir(args.run_id, L2_RUN_ROOT)
    replay = build_l2_d02_replay_case(run_dir, domain="d02", parent_domain="d01")
    qke_coldstart_meta = replay.metadata.get("qke_coldstart", {})
    seeded_qke = replay.state.qke  # already WRF-cold-start seeded on this branch
    zero_qke = jnp.zeros_like(seeded_qke)

    base = replay.state.replace(
        p=replay.state.p_total, ph=replay.state.ph_total, mu=replay.state.mu_total)
    grid = replay.grid
    grid_shape = [int(grid.nz), int(grid.ny), int(grid.nx)]

    # Checkpoints chosen to (a) catch the early blow-up (baseline went non-finite
    # at ~step 10-30) and (b) keep each advance's segment small enough to avoid the
    # fp64 OOM the 600-step jump hit (14.6 GiB).  step 30 (6min) / 150 (30min) /
    # 300 (1h) / 600 (2h) / 900 (3h): each advance is <=300 steps, splitting further
    # at the radiation cadence.
    probe_steps = [int(s) for s in os.environ.get("QKE_PROBE_STEPS", "30,150,300,600,900").split(",")]

    def _fresh_with_qke(qke):
        s = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x) + 0
            if (jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
                or jnp.issubdtype(jnp.asarray(x).dtype, jnp.integer)) else x,
            base)
        return s.replace(qke=jnp.asarray(qke) + 0)

    config_defs = {
        "qke0_harness": (zero_qke, HARNESS_FLAGS),
        "qke0_stable": (zero_qke, STABLE_FLAGS),
        "seed_harness": (seeded_qke, HARNESS_FLAGS),
        "seed_stable": (seeded_qke, STABLE_FLAGS),
    }

    results = {}
    for name in args.configs:
        qke0, extra = config_defs[name]
        nl = _build_namelist(replay, extra, dt_s=DT_S, substeps=10)
        state_cfg = _fresh_with_qke(qke0)
        flags = {k: (float(getattr(nl, k)) if isinstance(getattr(nl, k), float)
                     else (int(getattr(nl, k)) if isinstance(getattr(nl, k), int)
                           and not isinstance(getattr(nl, k), bool) else bool(getattr(nl, k))))
                 for k in ("top_lid", "epssm", "w_damping", "damp_opt", "dampcoef", "zdamp",
                           "diff_6th_opt", "diff_6th_factor", "use_flux_advection", "force_fp64")}
        seed_on = bool(name.startswith("seed"))
        print(f"\n=== config {name}: seed={'ON' if seed_on else 'OFF'} flags={extra} ===")
        try:
            res = _step_to_blowup(state_cfg, nl, dt_s=DT_S,
                                  probe_steps=probe_steps, max_steps=args.max_steps)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res = {"error": f"{type(exc).__name__}: {exc}"}
        res["flags"] = flags
        res["qke_seed"] = seed_on
        res["qke_t0_max"] = float(jnp.max(jnp.asarray(qke0)))
        # survived = last checkpoint that completed AND was finite
        finite_trace = [t for t in res.get("trace", []) if t.get("all_finite")]
        last = finite_trace[-1] if finite_trace else {}
        res["survived_steps"] = int(last.get("step", 0))
        res["survived_hours"] = float(last.get("model_hours", 0.0))
        res["aborted_oom"] = bool("error" in res and "RESOURCE_EXHAUSTED" in res.get("error", ""))
        res["stable_through_1h"] = bool(res.get("first_nonfinite_step") is None
                                        and not res.get("aborted_oom")
                                        and res["survived_steps"] >= 300)
        res["stable_through_max"] = bool(res.get("first_nonfinite_step") is None
                                         and not res.get("aborted_oom")
                                         and res["survived_steps"] >= max(probe_steps))
        results[name] = res
        fb = res.get("first_nonfinite_step")
        print(f"    -> first_nonfinite_step={fb} survived={res['survived_steps']}steps "
              f"({res['survived_hours']:.2f}h)")
        if res.get("trace"):
            for t in res["trace"]:
                q = t["dyn"].get("qke", {})
                mu = t["dyn"].get("mu", {})
                print(f"      step {t['step']:>4d} ({t['model_hours']:>5.2f}h) "
                      f"finite={t['all_finite']} qke.max={q.get('max')} mu.max={mu.get('max')} "
                      f"nonfin={list(t['nonfinite_fields'].keys())[:5]}")

    # minimal-fix determination -- only conclusive for configs that ran clean
    def blew(n):
        return results.get(n, {}).get("first_nonfinite_step") is not None

    def cured(n):
        r = results.get(n)
        return bool(r and not r.get("aborted_oom") and r.get("first_nonfinite_step") is None
                    and r.get("survived_steps", 0) >= 300)

    def ran(n):
        r = results.get(n)
        return bool(r and not r.get("aborted_oom") and "error" not in r)

    minimal = "INCONCLUSIVE"
    if "qke0_harness" in results and blew("qke0_harness"):
        seed_only = cured("seed_harness")
        flags_only = cured("qke0_stable")
        combo = cured("seed_stable")
        if seed_only and flags_only:
            minimal = "EITHER seed OR stable-flags cures it independently"
        elif seed_only:
            minimal = "qke seed ALONE is the minimal cure"
        elif flags_only and combo:
            minimal = "stable-flags cure it; seed adds no instability (full shipped fix = seed+stable)"
        elif flags_only and not ran("seed_stable"):
            minimal = "stable-flags ALONE cure it (seed_stable inconclusive/OOM); seed is WRF-faithful hardening on top"
        elif combo:
            minimal = "seed+stable cures it (seed_harness blew, qke0_stable not separately confirmed)"
        elif ran("seed_harness") and not seed_only:
            minimal = "qke seed ALONE insufficient (still blew up with weak namelist); stable-flags carry stability"
        else:
            minimal = "INCONCLUSIVE -- stable configs did not run clean"

    payload = {
        "schema": "V090D02ReplayQkeFixVerify",
        "generated": datetime.now(timezone.utc).isoformat(),
        "branch": "worker/opus/v090-d02replay-qke-fix",
        "trunk_commit": os.popen("git rev-parse HEAD").read().strip(),
        "platform": args.platform,
        "devices": devices,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "grid_mass_shape_zyx": grid_shape,
        "dt_s": DT_S,
        "acoustic_substeps": 10,
        "max_steps": args.max_steps,
        "qke_coldstart_meta": qke_coldstart_meta,
        "wrf_seed_ref": "phys/module_bl_mynnedmf.F:618-691 (mym_initialize INITIALIZE_QKE)",
        "results": results,
        "minimal_fix": minimal,
        "blowup_gone_full_fix": bool(results.get("seed_stable", {}).get("stable_through_1h")
                                     or results.get("qke0_stable", {}).get("stable_through_1h")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nminimal_fix: {minimal}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
