#!/usr/bin/env python
"""h36 re-init probe with microphysics genuinely OFF (mp_physics=0).

The CLI/daily path hardcodes the default physics suite (mp=8) and ignores the
case namelist.input physics options, so the namelist-edit variant ran Thompson
bit-identically.  This driver patches the resolved OperationalNamelist to
mp_physics=0 before the forecast loop.
"""

import dataclasses
from pathlib import Path

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=3,
    output_dir=PROBE / "gpu_output_nomp2",
    proof_dir=PROBE / "proofs_nomp2",
    run_root=PROBE,
    domain="d01",
)


def nomp_case_builder(cfg):
    case, run_dir = dp._build_real_case(cfg)
    namelist = dataclasses.replace(case.namelist, mp_physics=0)
    case = dataclasses.replace(case, namelist=namelist)
    print(f"nomp driver: patched mp_physics -> {case.namelist.mp_physics}")
    return case, run_dir


result = dp._run_forecast_sequence(
    config,
    output_dir=config.output_dir,
    case_builder=nomp_case_builder,
)
print("files:", [str(p) for p in result.files])
