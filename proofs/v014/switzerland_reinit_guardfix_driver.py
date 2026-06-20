#!/usr/bin/env python
"""h36 re-init PRODUCTION run with the v0.14 theta-limiter ceiling fix (500->1000 K).

Full production config (physics ON, guards ON); the only change vs the
gpu_output_phys_tendf baseline is _THETA_LIMITER_MAX_K = 1000.0 in
operational_mode.py (the Switzerland venting root-cause fix).  3 forecast hours
-> wrfout h37/h38/h39 for the binding depth-8 budget_between excess + the
warm-bias / u-dipole collapse + 2h stability.
"""

from pathlib import Path

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=3,
    output_dir=PROBE / "gpu_output_guardfix",
    proof_dir=PROBE / "proofs_guardfix",
    run_root=PROBE,
    domain="d01",
)

result = dp._run_forecast_sequence(config, output_dir=config.output_dir)
print("guardfix run complete")
