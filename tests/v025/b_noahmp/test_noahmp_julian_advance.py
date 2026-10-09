"""Noah-MP phenology clock = WRF grid%julian at every LSM call (GPUWRF_NOAHMP_JULIAN_ADVANCE, mass-opus 18:34Z).

WRF passes the CURRENT clock to the LSM: surface_driver JULIAN_IN=grid%julian (first_rk_step_part1.F:624), where
grid%julian = REAL(dayOfYear_r8) - 1.0 (module_domain.F:2165) is advanced only after solve (domain_clockadvance,
module_integrate.F:393-396) -> the clock at the START of own step n, init + (n-1)*dt; YEARLEN follows the current
year (YR from currentTime, first_rk_step_part1.F:231 -> noahmpdrv.F:678-684). The port passed the run-start
julian to every step: LAI/XSAI frozen at tau0 (0502 d01 up to 0.128 LAI by tau72).
Gates: (1) the real step function hands Noah-MP the step-start clock (independent host WRF julian, incl. a leap
year and the New-Year wrap); (2) per-frame LAI/XSAI == CPU-WRF wrfout (WN3 0502/0227, d01-d03, every 6 h) on all
land cells; (3) flag unset = the old run-start clock objects; the frozen clock and a step-END clock fail (2);
non-finite candidate/reference values fail the frame gate (E200).
"""
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.physics_couplers import time_utc_clock_base
from gpuwrf.integration.nested_pipeline import _wrf_julian_yearlen
from gpuwrf.physics.wrf_clwrf_ghg import _wrf_julian, clwrf_gas_clock
from gpuwrf.runtime import operational_mode as op

ROOT = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen")
GATE = Path(__file__).resolve().parents[3] / "proofs/noahmp/julian_advance/frame_gate.py"


class CapturedClock(Exception):
    pass


def _clock_base(start):
    julian, yearlen = _wrf_julian_yearlen(start)
    rad_julian, rad_minute = time_utc_clock_base(start)
    return SimpleNamespace(noahmp_julian=jnp.asarray(julian, jnp.float64), noahmp_yearlen=jnp.asarray(yearlen, jnp.float64),
                           ghg_clock=jax.tree_util.tree_map(jnp.asarray, clwrf_gas_clock(start)),
                           rad_julian=rad_julian, rad_minute=rad_minute, lower_boundary=None)


def _step_clock(monkeypatch, start, dt, step):
    """Run the real step function to the Noah-MP call; return the clock it receives."""
    shape = (2, 3)
    z = jnp.zeros(shape)

    class Box(SimpleNamespace):
        def replace(self, **kw):
            return Box(**{**vars(self), **kw})

    state = Box(t_skin=z + 290.0, xland=z + 1.0)
    carry = Box(state=state, noahmp_land=Box(t_skin=state.t_skin, tslb=jnp.zeros((4, *shape))),
                radiation_diagnostics=True, rthraten=None, census=None)
    namelist = SimpleNamespace(lower_boundary=None, noahmp_static=None, dt_s=dt, boundary_config=None, run_physics=True,
                               rad_rk_tendf=0, mp_physics=0, sf_sfclay_physics=5, cu_physics=0, use_noahmp=True,
                               radiation_interval_s=1800., radiation_cadence_steps=int(1800 / dt), grid=None,
                               time_utc=start, radiation_static=None, topo_shading=0, slope_rad=0,
                               topo_shadow_length_m=25000., noahmp_julian=-1.0, noahmp_yearlen=-1.0)

    def noahmp(new, land, *a, **kw):
        raise CapturedClock(kw["clock"])

    monkeypatch.setattr(op, "rrtmg_theta_tendency",
                        lambda old, grid, **kw: (None, SimpleNamespace(swnorm=z, glw=z, coszen=z)))
    monkeypatch.setitem(op.SFCLAY_SCAN_ADAPTERS, 5, lambda new, *a, **kw: new)
    monkeypatch.setattr(op, "_noahmp_params", lambda nml: (None, None))
    monkeypatch.setattr(op, "noahmp_surface_step", noahmp)
    with pytest.raises(CapturedClock) as captured:
        op._physics_boundary_step_with_limiter_diagnostics(carry, namelist, jnp.asarray(step, jnp.int32),
                                                           run_radiation=True, clock_base=_clock_base(start))
    return captured.value.args[0]


@pytest.mark.parametrize("start,dt,step", [
    (datetime(2026, 5, 3, 0, tzinfo=timezone.utc), 6.0, 1),
    (datetime(2026, 5, 3, 0, tzinfo=timezone.utc), 6.0, 43201),        # +72 h
    (datetime(2026, 2, 28, 0, tzinfo=timezone.utc), 54.0, 4801),       # +72 h, non-leap Feb -> Mar
    (datetime(2028, 2, 28, 18, tzinfo=timezone.utc), 18.0, 9601),      # leap year, +48 h through Feb 29
    (datetime(2026, 12, 31, 18, tzinfo=timezone.utc), 18.0, 2001),     # New-Year wrap: 2027 Jan 1 04:00
])
def test_step_hands_noahmp_the_wrf_step_start_clock(monkeypatch, start, dt, step):
    monkeypatch.setenv("GPUWRF_NOAHMP_JULIAN_ADVANCE", "1")
    clock = _step_clock(monkeypatch, start, dt, step)
    now = start + timedelta(seconds=(step - 1) * dt)
    assert float(clock.julian) == float(_wrf_julian(now))
    assert float(clock.yearlen) == (366.0 if now.year % 4 == 0 else 365.0)


def test_flag_unset_keeps_the_run_start_clock_objects(monkeypatch):
    monkeypatch.delenv("GPUWRF_NOAHMP_JULIAN_ADVANCE", raising=False)
    start = datetime(2026, 5, 3, 0, tzinfo=timezone.utc)
    base = _clock_base(start)
    assert op._noahmp_clock(None, base, 3600.0) == (base.noahmp_julian, base.noahmp_yearlen)
    clock = _step_clock(monkeypatch, start, 6.0, 43201)
    assert float(clock.julian) == float(base.noahmp_julian) and float(clock.yearlen) == float(base.noahmp_yearlen)


def _frame_gate(case, clock, tmp_path, monkeypatch):
    import importlib.util
    run = ROOT / f"wg_{case}" / "run/run"
    if not run.exists():
        pytest.skip("read-only WN3 CPU reference unavailable")
    spec = importlib.util.spec_from_file_location("julian_frame_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    monkeypatch.setenv("GPUWRF_NOAHMP_JULIAN_ADVANCE", "1")   # registered: the gate's clock_for flips it, teardown restores
    from netCDF4 import Dataset
    worst = 0.0
    for dom in ("d01", "d02", "d03"):
        files = sorted(glob.glob(f"{run}/wrfout_{dom}_*"))
        with Dataset(files[0]) as d0:
            dt = float(d0.DT)
            start = datetime.strptime(d0.START_DATE, "%Y-%m-%d_%H:%M:%S")
            ivg, lat, land = (np.asarray(d0[v][0]) for v in ("IVGTYP", "XLAT", "LANDMASK"))
            inputs = gate.inputs(ivg, lat, np.asarray(d0["TV"][0]), np.asarray(d0["SNOWH"][0]))
        for path in files[6::6]:
            t = datetime.strptime(path.split(f"wrfout_{dom}_")[1], "%Y-%m-%d_%H:%M:%S")
            lead = (t - start).total_seconds() - (0.0 if clock == "step_end" else dt)
            lai, sai = gate.phenology(*inputs, gate.clock_for(start, lead, "frozen" if clock == "frozen" else "advance"))
            with Dataset(path) as d:
                for name, port in (("LAI", lai), ("XSAI", sai)):
                    worst = max(worst, gate.frame_error(port, d[name][0], land > 0.5))
    return worst


@pytest.mark.parametrize("case", ("20260502_18z_a1", "20260227_18z_a1"))
def test_lai_xsai_match_cpu_wrf_every_frame(case, tmp_path, monkeypatch):
    assert _frame_gate(case, "advance", tmp_path, monkeypatch) <= 1e-6


@pytest.mark.parametrize("clock", ("frozen", "step_end"))
def test_frame_gate_rejects_frozen_and_step_end_clocks(clock, tmp_path, monkeypatch):
    assert _frame_gate("20260502_18z_a1", clock, tmp_path, monkeypatch) > 1e-6


def _gate_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("julian_frame_gate", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("side", ("candidate", "reference"))
def test_frame_error_rejects_nonfinite_values(side, bad):
    """E200 (review-b RB138): a NaN/+-Inf candidate or reference on a land cell fails; water cells are not scored."""
    gate = _gate_module()
    ref = np.linspace(0.5, 3.0, 12).reshape(3, 4)
    land = np.ones((3, 4), bool)
    land[0, 0] = False
    assert gate.frame_error(ref.copy(), ref, land) == 0.0
    port, refb = ref.copy(), ref.copy()
    (port if side == "candidate" else refb)[1, 2] = bad
    assert gate.frame_error(port, refb, land) == np.inf
    port, refb = ref.copy(), ref.copy()
    (port if side == "candidate" else refb)[0, 0] = bad      # water cell: outside the gate mask
    assert gate.frame_error(port, refb, land) == 0.0
