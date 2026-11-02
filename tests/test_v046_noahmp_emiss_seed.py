"""Noah-MP land EMISS seed == WRF landuse_init EMISS (v46-heavyrain HV04).

WRF's first radiation call (first_rk_step_part1) precedes the first Noah-MP call, so RRTMG sees the landuse_init
EMISS = SFEM(IS,ISN) (physics_init.F:1970-1971), which is also the CPU-WRF lead-zero history EMISS. The port seeded
a uniform 0.97; on the real Swiss case that made the step-1 RTHRATEN 3.3 % stronger at k0 (HV04: with the WRF seed
the step-1 RTHRATEN equals CPU-WRF at every level). Oracle = original CPU-WRF lead-zero history, every land cell.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (ROOT / "examples/switzerland_d01", "d01",
     Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_00:00:00")),
    *((Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run"), d,
       Path(f"<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run/wrfout_{d}_2026-02-28_00:00:00"))
      for d in ("d01", "d02", "d03")),
)


@pytest.mark.parametrize("run_dir,domain,cpu", CASES, ids=lambda c: str(c)[-30:])
def test_noahmp_emiss_seed_matches_cpu_wrf_lead_zero(run_dir, domain, cpu):
    if not cpu.exists() or not (run_dir / f"wrfinput_{domain}").exists():
        pytest.skip("original CPU-WRF lead-zero history not mounted")
    land, _static, _meta = build_noahmp_land_state(run_dir, domain)
    emiss = np.asarray(jax.device_get(land.emiss), dtype=np.float32)
    with Dataset(cpu) as ds:
        reference = np.asarray(ds["EMISS"][0], dtype=np.float32)
        is_land = np.asarray(ds["XLAND"][0]) < 1.5
    assert is_land.any()
    # the old uniform seed must not survive on land (deletion-sensitive: 0.97 fails here on every case)
    assert not np.all(emiss[is_land] == np.float32(0.97))
    np.testing.assert_array_equal(emiss[is_land], reference[is_land])
