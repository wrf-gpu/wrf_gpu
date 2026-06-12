#!/usr/bin/env python
"""h36 re-init probe with ALL physics OFF (run_physics=False) — pure dycore+LBC.

Bisects the v0.14 Switzerland +0.5 K/h interior warm bias / vertical u-dipole:
radiation-off left both unchanged (gpu_output_norad), so if the k15-k25 warm
band persists here the source is the dycore transport chain (vertical advection
/ stage omega / acoustic theta closure), not any physics scheme.
"""

import dataclasses
from pathlib import Path

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=2,
    output_dir=PROBE / "gpu_output_nophys",
    proof_dir=PROBE / "proofs_nophys",
    run_root=PROBE,
    domain="d01",
)


def nophys_case_builder(cfg):
    case, run_dir = dp._build_real_case(cfg)
    namelist = dataclasses.replace(case.namelist, run_physics=False)
    case = dataclasses.replace(case, namelist=namelist)
    print("nophys driver: patched run_physics ->", case.namelist.run_physics)
    return case, run_dir


result = dp._run_forecast_sequence(
    config,
    output_dir=config.output_dir,
    case_builder=nophys_case_builder,
)
print("files:", [str(p) for p in result.files])
