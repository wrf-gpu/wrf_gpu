"""XLA control-flow/command-buffer flag A/B on the warm FAST loop (dt54, 67 steps, seg34).

Child mode (--arm): one XLA_FLAGS binding per process (flags bind at backend init).
Builds the case, one warm-up, 3 timed reps of run_forecast_operational_segmented,
writes JSON {median_s, reps, theta_sha256}.  Bitwise check = identical sha256 across
arms (deterministic fp64 kernels on the same GPU/build).
"""
import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

WORKTREE_SRC = "<USER_HOME>/src/wrf_gpu2_wt/hostforensic/src"
sys.path.insert(0, WORKTREE_SRC)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.integration import daily_pipeline as daily  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
SEG = 34
STEPS = 67
DT = 54.0


def build_case():
    config = daily.DailyPipelineConfig(run_id=str(RUN_DIR), run_root=RUN_DIR.parent, hours=1, domain="d01")
    case, _ = daily._build_real_case(config)
    state = case.state
    capture = daily._capture_boundary_leaves(state, case.namelist)
    cadence = daily._boundary_window_cadence_s(case.namelist)
    record_cadence = float((case.metadata.get("boundary") or {}).get("interval_seconds") or cadence)
    if capture:
        state = daily._rewindow_boundary_leaves(state, capture, segment_start_s=0.0,
                                                record_cadence_s=record_cadence, window_s=cadence)
    nl = dataclasses.replace(case.namelist, dt_s=DT)
    return state, nl, STEPS * DT / 3600.0


def state_hash(st) -> str:
    h = hashlib.sha256()
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(st)
    import numpy as np
    for path, leaf in leaves_with_paths:
        a = jax.device_get(leaf)
        a = getattr(a, "value", a)
        a = np.asarray(a)
        h.update(str(jax.tree_util.keystr(path)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--flag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    state, nl, hours = build_case()
    # warm-up (compiles + autotune + cache load)
    out = om.run_forecast_operational_segmented(state, nl, hours, segment_steps=SEG)
    jax.block_until_ready(out.theta)
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = om.run_forecast_operational_segmented(state, nl, hours, segment_steps=SEG)
        jax.block_until_ready(out.theta)
        times.append(time.perf_counter() - t0)
    res = {
        "arm": args.arm,
        "flag": args.flag,
        "xla_flags_env": os.environ.get("XLA_FLAGS", ""),
        "reps_s": times,
        "median_s": statistics.median(times),
        "step_ms": 1000.0 * statistics.median(times) / STEPS,
    }
    try:
        res["theta_sha256"] = state_hash(out)
    except Exception as e:  # keep timing evidence even if hashing fails
        res["theta_sha256_error"] = repr(e)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "theta_sha256"}, indent=1), flush=True)
    print("sha256", res["theta_sha256"][:16], flush=True)


if __name__ == "__main__":
    main()
