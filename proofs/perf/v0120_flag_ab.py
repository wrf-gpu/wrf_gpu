#!/usr/bin/env python3
"""v0.12.0 XLA-flag A/B harness for the SAFE launch-tax speedup.

Runs the EXACT production standalone path (``_build_real_case`` +
``run_forecast_operational_segmented``, force_fp64 per the operational config)
for ``--hours`` hours, drops hour-1 (compile), reports the warm per-forecast-hour
wall, and emits per-field min/max/finite of the FINAL state so the same case run
under two different ``XLA_FLAGS`` can be diffed: the safe-speedup claim is that
``--xla_gpu_graph_min_graph_size=1`` (CUDA-graph capture of short fusion chains)
only perturbs results at fp64 machine-epsilon (it changes the kernel
launch/scheduling, NOT the arithmetic), as the prior dynamics-only probe found
(proofs/perf/fusion_results.md).

The XLA flag is taken from the ambient ``XLA_FLAGS`` env (set by the caller); this
script just records what it sees and measures the warm step under it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--domain", default="d01")
    ap.add_argument("--hours", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import jax
    import numpy as np

    from gpuwrf.integration.daily_pipeline import (
        DailyPipelineConfig,
        _build_real_case,
        _commit_to_operational_device,
        finite_summary,
    )
    from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented

    config = DailyPipelineConfig(
        run_id=str(args.input_dir.resolve()),
        run_root=args.input_dir.parent,
        hours=int(args.hours),
        output_dir=Path("/tmp/_flagab_out"),
        proof_dir=Path("/tmp/_flagab_proof"),
        domain=args.domain,
    )
    case, _ = _build_real_case(config)
    state = _commit_to_operational_device(case.state)
    namelist = case.namelist

    per_hour: list[float] = []
    for _ in range(int(args.hours)):
        t0 = time.perf_counter()
        state = run_forecast_operational_segmented(state, namelist, 1.0)
        jax.block_until_ready(state)
        per_hour.append(time.perf_counter() - t0)

    summary = finite_summary(state)

    # Final-state fingerprint for cross-flag numerics-neutrality check.
    fields = {}
    for name in ("u", "v", "theta", "w", "ph_total", "p_total", "mu_total", "qv", "qke"):
        arr = getattr(state, name, None)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        fields[name] = {
            "min": float(np.nanmin(a)),
            "max": float(np.nanmax(a)),
            "mean": float(np.nanmean(a)),
            "l2": float(np.sqrt(np.nansum(a * a))),
        }

    # warm = drop hour-1 (compile); use mean of remaining hours.
    warm = per_hour[1:] if len(per_hour) >= 2 else per_hour
    warm_s_per_fc_hr = float(sum(warm) / len(warm)) if warm else None

    result = {
        "schema": "GpuwrfV0120FlagAB",
        "label": args.label,
        "domain": args.domain,
        "hours": int(args.hours),
        "device": str(jax.devices()[0]),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "wall_clock_per_hour_s": per_hour,
        "warm_s_per_forecast_hour": warm_s_per_fc_hr,
        "all_finite": summary["all_finite"],
        "force_fp64": bool(namelist.force_fp64),
        "final_state_fingerprint": fields,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(json.dumps({
        "label": args.label,
        "warm_s_per_forecast_hour": warm_s_per_fc_hr,
        "all_finite": summary["all_finite"],
        "xla_flags": result["xla_flags"],
    }, indent=2))
    return 0 if summary["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
