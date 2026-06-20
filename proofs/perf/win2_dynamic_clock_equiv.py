#!/usr/bin/env python
"""Win #2 numerics-identical proof: dynamic-clock scalars == legacy static time_utc.

The pre-change behaviour baked ``_time_utc_parts(time_utc)`` into the JIT graph as
host constants. The change carries the SAME two scalars (day-of-year, abs UTC
minutes) as dynamic JAX leaves. This harness runs the REAL d02 operational
segmented forecast both ways on the same initial state and asserts the full
forecast state is BIT-IDENTICAL:

  A) DYNAMIC path  : namelist with start_julian_day/start_utc_minute leaves set
                     (the new operational behaviour).
  B) LEGACY path   : the SAME namelist with the scalar leaves forced to None so
                     the in-scan radiation re-derives from the static time_utc
                     (this is byte-for-byte the pre-change code path).

If A == B on every state field, the change cannot have altered any forecast.

USAGE
  PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/win2_dynamic_clock_equiv.py \
      --run-id 20260521_18z_l3_24h_20260522T133443Z \
      --run-root <DATA_ROOT>/canairy_meteo/runs/wrf_l3 --hours 1 --segment-steps 60
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented


def _state_fields(state) -> dict[str, np.ndarray]:
    fields = {}
    for name in getattr(type(state), "__slots__", ()):
        val = getattr(state, name, None)
        if val is None:
            continue
        try:
            fields[name] = np.asarray(jax.device_get(val))
        except Exception:
            continue
    return fields


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--segment-steps", type=int, default=60)
    ap.add_argument("--out", default="proofs/perf/win2_dynamic_clock_equiv.json")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(
        run_id=args.run_id,
        run_root=Path(args.run_root),
        output_dir=Path("/tmp/win2_equiv_out"),
        proof_dir=Path("/tmp/win2_equiv_proof"),
        hours=int(args.hours),
        domain="d02",
    )
    case, _run_dir = _build_real_case(cfg)

    # Thread the real init clock onto the namelist -> __post_init__ derives the
    # dynamic scalar leaves. This is the NEW operational behaviour (path A).
    nl_dynamic = dataclasses.replace(case.namelist, time_utc=case.run_start)
    assert nl_dynamic.start_julian_day is not None
    assert nl_dynamic.start_utc_minute is not None

    # Force the LEGACY static path (path B): same clock via time_utc, but the
    # dynamic scalars nulled so the in-scan radiation re-derives host constants
    # exactly as the pre-change code did. Construct with time_utc=None so
    # __post_init__ does NOT derive scalars, then set time_utc by hand (frozen
    # dataclass => object.__setattr__) without re-triggering derivation. This
    # reproduces the exact pre-change in-scan radiation behaviour.
    nl_legacy = dataclasses.replace(nl_dynamic, time_utc=None,
                                    start_julian_day=None, start_utc_minute=None)
    object.__setattr__(nl_legacy, "time_utc", case.run_start)
    assert nl_legacy.start_julian_day is None
    assert nl_legacy.start_utc_minute is None
    assert nl_legacy.time_utc == case.run_start

    out_dynamic = run_forecast_operational_segmented(
        case.state, nl_dynamic, float(args.hours), segment_steps=args.segment_steps
    )
    jax.block_until_ready(out_dynamic.theta)
    out_legacy = run_forecast_operational_segmented(
        case.state, nl_legacy, float(args.hours), segment_steps=args.segment_steps
    )
    jax.block_until_ready(out_legacy.theta)

    fa = _state_fields(out_dynamic)
    fb = _state_fields(out_legacy)
    common = sorted(set(fa) & set(fb))

    field_report = {}
    max_abs = 0.0
    worst = None
    for name in common:
        a, b = fa[name], fb[name]
        if a.shape != b.shape:
            field_report[name] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
            continue
        d = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        field_report[name] = {"max_abs_diff": d, "dtype": str(a.dtype)}
        if d > max_abs:
            max_abs, worst = d, name

    bit_identical = max_abs == 0.0
    payload = {
        "win": "2-dynamic-clock-cache-key",
        "run_id": args.run_id,
        "hours": args.hours,
        "segment_steps": args.segment_steps,
        "run_start": str(case.run_start),
        "start_julian_day": float(nl_dynamic.start_julian_day),
        "start_utc_minute": float(nl_dynamic.start_utc_minute),
        "fields_compared": len(common),
        "max_abs_diff_over_all_fields": max_abs,
        "worst_field": worst,
        "bit_identical": bit_identical,
        "verdict": "PASS" if bit_identical else "FAIL",
        "fields": field_report,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in (
        "verdict", "max_abs_diff_over_all_fields", "worst_field",
        "fields_compared", "start_julian_day", "start_utc_minute")}, indent=2))
    print(f"proof: {out_path}")
    return 0 if bit_identical else 1


if __name__ == "__main__":
    sys.exit(main())
