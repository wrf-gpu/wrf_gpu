"""WRF-grounded operand order; solver stubs only expose caller operands."""
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import jax
import numpy as np
from netCDF4 import Dataset, chartostring
import pytest

from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.io.lower_boundary import load_lower_boundary
from gpuwrf.runtime import operational_mode as op

ROOT = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen")


@dataclass(frozen=True)
class Surface:
    t_skin: object
    xland: object

    def replace(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Land:
    t_skin: object
    tslb: object

    def replace(self, **kw):
        return replace(self, **kw)


@dataclass(frozen=True)
class Carry:
    state: object
    noahmp_land: object
    radiation_diagnostics: object = True
    rthraten: object = None
    census: object = None

    def replace(self, **kw):
        return replace(self, **kw)


class CapturedSurface(Exception):
    pass


@pytest.mark.parametrize("case", ("20260227_18z_a1", "20260502_18z_a1", "20260614_18z_a1"))
@pytest.mark.parametrize("domain,dt", (("d02", 18), ("d03", 6)))
@pytest.mark.parametrize("record", (1, 2, 3))
def test_resumed_alarm_radiation_old_surface_new(monkeypatch, case, domain, dt, record):
    source = ROOT / f"wg_{case}" / "run/run"
    if not source.exists():
        pytest.skip("read-only WN3 CPU reference unavailable")
    nml = parse_namelist(source / "namelist.input")
    with Dataset(source / f"wrflowinp_{domain}") as ds:
        times = [datetime.strptime(str(t), "%Y-%m-%d_%H:%M:%S") for t in chartostring(ds["Times"][:])]
    with Dataset(source / f"wrfout_{domain}_{times[record]:%Y-%m-%d_%H:%M:%S}") as ds:
        previous_tsk = np.asarray(ds["TSK"][0])
        previous_tslb = np.asarray(ds["TSLB"][0])
        water = np.asarray(ds["XLAND"][0]) > 1.5
    boundary = load_lower_boundary(source, nml, domain, run_start=times[0], dt_s=dt, shape=water.shape)
    activation = int(boundary.activation_steps[record])
    state = Surface(jnp.asarray(previous_tsk), jnp.asarray(np.where(water, 2., 1.)))
    carry = Carry(state, Land(state.t_skin, jnp.asarray(previous_tslb)))
    namelist = SimpleNamespace(lower_boundary=boundary, noahmp_static=None,
        dt_s=dt, boundary_config=None, run_physics=True, rad_rk_tendf=0,
        mp_physics=0, sf_sfclay_physics=5, cu_physics=0, use_noahmp=True,
        radiation_interval_s=1800., radiation_cadence_steps=int(1800 / dt),
        grid=None, time_utc=times[0], radiation_static=None,
        topo_shading=1, slope_rad=1, topo_shadow_length_m=25000.,
        noahmp_julian=0., noahmp_yearlen=365.)
    observed = []

    def radiation(old, grid, **kw):
        observed.append("radiation")
        np.testing.assert_array_equal(old.t_skin, previous_tsk)
        np.testing.assert_array_equal(kw["land_state"].tslb, previous_tslb)
        z = jnp.zeros_like(old.t_skin)
        return None, SimpleNamespace(swnorm=z, glw=z, coszen=z)

    def sfclay(new, *a, **kw):
        observed.append("sfclay")
        np.testing.assert_array_equal(np.asarray(new.t_skin)[water], np.asarray(boundary.sst[record])[water])
        return new

    def noahmp(new, land, *a, **kw):
        observed.append("noahmp")
        expected = np.asarray(boundary.sst[record])[water]
        np.testing.assert_array_equal(np.asarray(new.t_skin)[water], expected)
        np.testing.assert_array_equal(np.asarray(land.t_skin)[water], expected)
        np.testing.assert_array_equal(np.asarray(land.tslb[0])[water], expected)
        np.testing.assert_array_equal(np.asarray(land.tslb[1:]), previous_tslb[1:])
        raise CapturedSurface()

    monkeypatch.setattr(op, "rrtmg_theta_tendency", radiation)
    monkeypatch.setitem(op.SFCLAY_SCAN_ADAPTERS, 5, sfclay)
    monkeypatch.setattr(op, "_noahmp_params", lambda nml: (None, None))
    monkeypatch.setattr(op, "noahmp_surface_step", noahmp)
    # This is the first step of a resumed segment at6/12/18h; use its absolute
    # one-based step number, not the segment-local1.
    with pytest.raises(CapturedSurface):
        op._physics_boundary_step_with_limiter_diagnostics(carry, namelist,
            jnp.asarray(activation + 1, jnp.int32), run_radiation=True)
    assert observed == ["radiation", "sfclay", "noahmp"]


def test_compiled_resumed_segment_retains_radiation_then_surface_order(monkeypatch):
    """Control regression only: actual fori scheduler, sentinel solver arithmetic."""
    import importlib.util
    from gpuwrf.io.lower_boundary import LowerBoundary
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("order_fixture", Path(__file__).with_name("test_rrtm_lw_operational_wiring.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = fixture._state(grid).replace(t_skin=jnp.full((2, 2), 280.), xland=jnp.full((2, 2), 2.))
    state = op._enforce_operational_precision(state, force_fp64=True)
    sst = jnp.asarray([np.full((2, 2), 280.), np.full((2, 2), 285.)], jnp.float32)
    boundary = LowerBoundary(jnp.asarray([0, 2], jnp.int32), sst,
        jnp.zeros_like(sst), jnp.zeros_like(sst), jnp.full_like(sst, .08))
    nml = replace(fixture._namelist(grid), lower_boundary=boundary, mp_physics=0,
        bl_pbl_physics=0, sf_sfclay_physics=999, cu_physics=0, use_noahmp=False,
        run_boundary=False, disable_guards=True, force_fp64=True, rad_rk_tendf=0,
        radiation_cadence_steps=2, radiation_interval_s=20.)
    carry = initial_operational_carry(state).replace(radiation_diagnostics=(state.t_skin,))
    monkeypatch.setattr(op, "rrtmg_theta_tendency", lambda st, *a, **kw:
        (jnp.zeros_like(st.theta), (st.t_skin,)))
    monkeypatch.setattr(op, "surface_adapter", lambda st, *a, **kw: st.replace(ustar=st.t_skin))
    monkeypatch.setattr(op, "_rk_scan_step", lambda c, *a, **kw: c)
    op._advance_chunk_fori.clear_cache()
    try:
        whole = op._advance_chunk_fori(carry, nml, jnp.asarray(1, jnp.int32), n_steps=3, cadence=2)
        first = op._advance_chunk_fori(carry, nml, jnp.asarray(1, jnp.int32), n_steps=2, cadence=2)
        resumed = op._advance_chunk_fori(first, nml, jnp.asarray(3, jnp.int32), n_steps=1, cadence=2)
        np.testing.assert_array_equal(resumed.radiation_diagnostics[0], np.full((2, 2), 280.))
        np.testing.assert_array_equal(resumed.state.ustar, np.full((2, 2), 285.))
        for a, b in zip(jax.tree_util.tree_leaves(whole), jax.tree_util.tree_leaves(resumed), strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        op._advance_chunk_fori.clear_cache()
