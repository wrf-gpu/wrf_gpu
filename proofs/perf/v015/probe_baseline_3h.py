#!/usr/bin/env python
"""v0.15 kernel probe — baseline reproduction of the v0.14 perf-triage protocol.

Runs the Switzerland d01 reinit-h36 replay (128x128x44, dt=18s, force_fp64)
for 3 forecast hours through the production daily pipeline `_run_forecast_sequence`
(the EXACT path the v0.14 triage measured: per-hour run_forecast_operational call
+ finite_summary + wrfout write + land refresh + boundary rewindow).

Output: proofs/perf/v015/baseline_3h.json with per-hour walls and per-step ms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT = Path("/tmp/v015_perf/baseline_3h")
PROOF = Path(__file__).resolve().parent / "baseline_3h.json"

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=3,
    output_dir=OUT,
    proof_dir=OUT / "proofs",
    run_root=PROBE,
    domain="d01",
)

t0 = time.perf_counter()
result = dp._run_forecast_sequence(config, output_dir=config.output_dir)
total = time.perf_counter() - t0

payload = {
    "schema": "V015KernelProbeBaseline",
    "case": "Switzerland d01 reinit h36 replay, 128x128x44, dt=18s, force_fp64",
    "entry": "dp._run_forecast_sequence -> run_forecast_operational (1 jit call/hour)",
    "per_hour_wall_s": [round(x, 3) for x in result.per_hour_wall_s],
    "total_wall_s": round(total, 3),
    "steps_per_hour": 200,
    "steady_state_ms_per_step_hour3": round(result.per_hour_wall_s[-1] / 200.0 * 1000.0, 2),
    "n_files": len(result.output_files),
}
PROOF.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
