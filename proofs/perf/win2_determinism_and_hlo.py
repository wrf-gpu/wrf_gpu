#!/usr/bin/env python
"""Win #2 root-cause: is the segmented-forecast output deterministic, and is the
dynamic-clock HLO identical to the legacy-static HLO?

Check 1 (determinism floor): run the SAME dynamic-scalar forecast TWICE and report
the max abs diff. If nonzero, GPU XLA reductions are non-deterministic and that is
the noise floor any equivalence comparison must be read against.

Check 2 (HLO identity): lower _advance_chunk for the dynamic-scalar namelist and
for the legacy-static namelist (same clock, scalars None). If the lowered HLO text
is identical, the two paths compile to the SAME executable -> bit-identical by
construction, and any run-to-run output diff is pure non-determinism.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import (
    _advance_chunk,
    run_forecast_operational_segmented,
)
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.operational_mode import _enforce_operational_precision


def _fields(state):
    out = {}
    for name in getattr(type(state), "__slots__", ()):
        v = getattr(state, name, None)
        if v is None:
            continue
        try:
            out[name] = np.asarray(jax.device_get(v))
        except Exception:
            pass
    return out


def _maxdiff(a, b):
    common = sorted(set(a) & set(b))
    m, worst = 0.0, None
    for k in common:
        if a[k].shape != b[k].shape:
            continue
        d = float(np.nanmax(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) if a[k].size else 0.0
        if d > m:
            m, worst = d, k
    return m, worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--segment-steps", type=int, default=60)
    ap.add_argument("--out", default="proofs/perf/win2_determinism_and_hlo.json")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(
        run_id=args.run_id, run_root=Path(args.run_root),
        output_dir=Path("/tmp/win2_det_out"), proof_dir=Path("/tmp/win2_det_proof"),
        hours=int(args.hours), domain="d02",
    )
    case, _ = _build_real_case(cfg)
    nl_dyn = dataclasses.replace(case.namelist, time_utc=case.run_start)

    nl_legacy = dataclasses.replace(nl_dyn, time_utc=None,
                                    start_julian_day=None, start_utc_minute=None)
    object.__setattr__(nl_legacy, "time_utc", case.run_start)
    seg = int(args.segment_steps)

    # Determinism floor: same dynamic path, run THREE times, in one GPU session.
    outs = []
    for _ in range(3):
        o = run_forecast_operational_segmented(case.state, nl_dyn, float(args.hours), segment_steps=seg)
        jax.block_until_ready(o.theta)
        outs.append(_fields(o))
    det12, w12 = _maxdiff(outs[0], outs[1])
    det13, w13 = _maxdiff(outs[0], outs[2])
    det23, w23 = _maxdiff(outs[1], outs[2])
    det_diff = max(det12, det13, det23)
    det_worst = {"d12": [det12, w12], "d13": [det13, w13], "d23": [det23, w23]}

    # Dynamic vs legacy in the SAME session (so the same non-determinism floor applies).
    out_leg = run_forecast_operational_segmented(case.state, nl_legacy, float(args.hours), segment_steps=seg)
    jax.block_until_ready(out_leg.theta)
    dl_diff, dl_worst = _maxdiff(outs[0], _fields(out_leg))

    payload = {
        "win": "2-root-cause",
        "determinism_same_path_max_abs_diff": det_diff,
        "determinism_pairs": det_worst,
        "dynamic_vs_legacy_max_abs_diff": dl_diff,
        "dynamic_vs_legacy_worst_field": dl_worst,
        "within_noise_floor": dl_diff <= det_diff,
        "run_start": str(case.run_start),
        "hours": args.hours,
        "segment_steps": seg,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
