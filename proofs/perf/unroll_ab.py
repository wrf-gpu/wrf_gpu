"""A/B runner for the acoustic-substep scan unroll (one config per invocation).

Reads GPUWRF_ACOUSTIC_UNROLL from the env (the source reads the same var at trace
time), runs the warmed coupled real-d02 forecast, and writes:
  * warmed per-step ms (marginal (120-step - 30-step)/90),
  * the final state field arrays (saved to .npz) for a bitwise cross-run compare.

Run twice (unroll=1 baseline, unroll=N) then compare with --compare.

  GPUWRF_ACOUSTIC_UNROLL=1 PYTHONPATH=src ... python proofs/perf/unroll_ab.py --tag u1
  GPUWRF_ACOUSTIC_UNROLL=4 PYTHONPATH=src ... python proofs/perf/unroll_ab.py --tag u4
  python proofs/perf/unroll_ab.py --compare u1 u4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

PROOF = Path("proofs/perf")
FIELDS = ("u", "v", "w", "theta", "qv", "p_total", "ph_total", "mu_total")


def _run_one(tag: str) -> int:
    import jax
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    # Production segmented entry: chunks memory via the radiation loop (the
    # single-scan entry OOMs at 60-120 steps when the GPU is shared). The unroll
    # edit lives in _acoustic_scan, used by ALL entries, so this exercises it.
    from gpuwrf.runtime.operational_mode import run_forecast_operational as run_fn

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=180, time_utc=case.run_start,
    )
    dt_s = float(nl.dt_s)

    def builder():
        return _build_real_case(cfg)[0].state

    def warm(h):
        o = run_fn(builder(), nl, h); jax.block_until_ready(o.theta)

    def t(h, k=3):
        best = 1e9
        for _ in range(k):
            st = builder()
            t0 = time.perf_counter()
            o = run_fn(st, nl, h); jax.block_until_ready(o.theta)
            best = min(best, time.perf_counter() - t0)
        return best

    h_s, h_b = 30 * dt_s / 3600.0, 120 * dt_s / 3600.0
    warm(h_s); warm(h_b)
    ws, wb = t(h_s), t(h_b)
    per_step_ms = (wb - ws) / 90.0 * 1000.0

    # final state at 30 steps for bitwise compare
    final = run_fn(builder(), nl, h_s)
    jax.block_until_ready(final.theta)
    arrs = {f: np.asarray(jax.device_get(getattr(final, f)), dtype=np.float64) for f in FIELDS}

    PROOF.mkdir(parents=True, exist_ok=True)
    np.savez(PROOF / f"unroll_ab_{tag}.npz", **arrs)
    meta = {
        "tag": tag,
        "unroll_env": os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"),
        "device": str(jax.devices()[0]),
        "per_step_ms": per_step_ms,
        "warm_30_s": ws, "warm_120_s": wb,
    }
    (PROOF / f"unroll_ab_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


def _compare(tag_a: str, tag_b: str) -> int:
    a = np.load(PROOF / f"unroll_ab_{tag_a}.npz")
    b = np.load(PROOF / f"unroll_ab_{tag_b}.npz")
    meta_a = json.loads((PROOF / f"unroll_ab_{tag_a}.json").read_text())
    meta_b = json.loads((PROOF / f"unroll_ab_{tag_b}.json").read_text())
    max_abs = {f: float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS}
    bitwise = max(max_abs.values()) == 0.0
    speedup = meta_a["per_step_ms"] / meta_b["per_step_ms"] if meta_b["per_step_ms"] else float("nan")
    out = {
        "scope": f"acoustic-substep unroll A/B: {tag_a} (unroll={meta_a['unroll_env']}) vs {tag_b} (unroll={meta_b['unroll_env']})",
        "baseline_per_step_ms": meta_a["per_step_ms"],
        "unrolled_per_step_ms": meta_b["per_step_ms"],
        "speedup": speedup,
        "max_abs_diff_per_field": max_abs,
        "bitwise_identical": bitwise,
        "verdict": ("SAFE+FASTER" if (bitwise and speedup > 1.02)
                    else "SAFE-NO-GAIN" if bitwise
                    else "NOT-BITWISE-IDENTICAL"),
    }
    (PROOF / "unroll_ab_verdict.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0 if bitwise else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        return _compare(args.compare[0], args.compare[1])
    return _run_one(args.tag or f"u{os.environ.get('GPUWRF_ACOUSTIC_UNROLL', '1')}")


if __name__ == "__main__":
    raise SystemExit(main())
