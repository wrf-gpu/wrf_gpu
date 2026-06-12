#!/usr/bin/env python
"""h36 re-init probe with radiation genuinely OFF (ra_sw=ra_lw=0).

Bisects the v0.14 Switzerland +0.5 K/h interior tropospheric warm bias
(proofs/v014/switzerland_midlevel_momentum_budget.json): if the warm bias
collapses with radiation off, the heating lives in the RTHRATEN lane
(RRTMG magnitude/sign or the rad_rk_tendf t_tendf cadence); if it persists,
the heating is in the dycore theta budget (advection/ww/EOS).
"""

import dataclasses
from pathlib import Path

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=2,
    output_dir=PROBE / "gpu_output_norad",
    proof_dir=PROBE / "proofs_norad",
    run_root=PROBE,
    domain="d01",
)


def norad_case_builder(cfg):
    case, run_dir = dp._build_real_case(cfg)
    namelist = dataclasses.replace(case.namelist, ra_sw_physics=0, ra_lw_physics=0)
    case = dataclasses.replace(case, namelist=namelist)
    print(
        "norad driver: patched ra_sw_physics ->",
        case.namelist.ra_sw_physics,
        "ra_lw_physics ->",
        case.namelist.ra_lw_physics,
    )
    return case, run_dir


result = dp._run_forecast_sequence(
    config,
    output_dir=config.output_dir,
    case_builder=norad_case_builder,
)
print("files:", [str(p) for p in result.files])
