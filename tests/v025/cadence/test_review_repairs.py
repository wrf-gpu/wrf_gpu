"""Production caller/control checks against critic's pristine WRF oracles."""
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.diagnostics import census
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_initial_held_output import fixture


@pytest.fixture(scope="module")
def wrf_oracles(tmp_path_factory, wrf_fortran_compiler, frozen_wrf_oracles):
    sources = Path(__file__).with_name("oracles")
    out = tmp_path_factory.mktemp("wrf-review-oracles")
    for name in ("control_oracle", "solar_oracle"):
        subprocess.run([wrf_fortran_compiler, "-ffree-line-length-none",
                        str(sources / f"{name}.f90"), "-o", str(out / name)],
                       check=True, capture_output=True)
    def run(name, rows):
        text = "".join(" ".join(map(str, row)) + "\n" for row in rows)
        result = subprocess.run([str(out / name)], input=text, text=True,
                                capture_output=True, check=True)
        return np.asarray([[float(x) for x in line.split()]
                           for line in result.stdout.splitlines()])
    return run


def seed(state, nca=-100., rates=None):
    if rates is None:
        rates = tuple(jnp.zeros_like(state.theta) for _ in range(6)) + (jnp.zeros_like(state.t_skin),)
    return initial_operational_carry(state).replace(
        cumulus_carry=(jnp.zeros_like(state.theta), jnp.full_like(state.t_skin, nca)),
        cumulus_tendencies=rates, census=census.initial_census())


@pytest.mark.parametrize("dt,cu,rad", [(54., 6, 33), (18., 17, 100), (54., 1, 1)])
def test_production_caller_phases_and_resumed_segments(monkeypatch, wrf_oracles, dt, cu, rad):
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = op._enforce_operational_precision(fixture._state(grid), force_fp64=True)
    nml = dataclasses.replace(fixture._namelist(grid), dt_s=dt,
        mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=999, use_noahmp=False,
        cu_physics=1, cumulus_cadence_steps=cu, cudt_minutes=5.,
        radiation_cadence_steps=rad, run_boundary=False, disable_guards=True, force_fp64=True)
    length = 2 * rad + 1
    expected = wrf_oracles("control_oracle", [
        (step, dt, cu, rad, 5., 30., -100., 0., 0.) for step in range(1, length + 1)])
    # Replace column arithmetic only; retain the actual fori loop, boundary
    # caller clock, physics forcing, KF gate and radiation driver predicate.
    monkeypatch.setattr(op, "kf_adapter", lambda st, dt, w, n, **kw:
                        (kw["held_tendencies"], w, n))
    monkeypatch.setattr(op, "surface_adapter", lambda st, *a, **kw: st)
    monkeypatch.setattr(op, "_rk_scan_step", lambda carry, *a, **kw: carry)
    def radiation(st, *args, **kw):
        return jnp.zeros_like(st.theta), (jnp.full_like(st.t_skin, kw["lead_seconds"]),)
    monkeypatch.setattr(op, "rrtmg_theta_tendency", radiation)
    initial = seed(state).replace(radiation_diagnostics=(jnp.zeros_like(state.t_skin),))
    op._advance_chunk_fori.clear_cache()
    try:
        # A split before a cadence boundary must preserve the absolute step
        # index, including starts that are neither step1 nor call boundaries.
        splits = sorted(set((1, min(4, length), min(rad, length), length + 1)))
        carry = initial
        previous = {"kf_calls": 0, "radiation_tendency_calls": 0}
        for start, stop in zip(splits, splits[1:]):
            carry = op._advance_chunk_fori(carry, nml, jnp.asarray(start),
                                           n_steps=stop - start, cadence=rad)
            work = dict(zip(census.WORK, map(int, jax.device_get(carry.census.work))))
            for name, column in (("kf_calls", 0), ("radiation_tendency_calls", 1)):
                assert work[name] - previous[name] == int(expected[start-1:stop-1, column].sum())
                previous[name] = work[name]
            # The retained flux records this caller's step-entry clock.
            last_call = np.flatnonzero(expected[:stop-1, 1])[-1]
            np.testing.assert_array_equal(carry.radiation_diagnostics[0], last_call * dt)
        whole = op._advance_chunk_fori(initial, nml, jnp.asarray(1), n_steps=length, cadence=rad)
        for actual, uninterrupted in zip(jax.tree.leaves(carry), jax.tree.leaves(whole), strict=True):
            np.testing.assert_array_equal(actual, uninterrupted)
        # Audit every actual call phase too: aggregate counts can hide a
        # shifted predicate within a segment.
        carry = initial
        seen = {"kf_calls": [], "radiation_tendency_calls": []}
        previous = {name: 0 for name in seen}
        for step in range(1, length + 1):
            carry = op._advance_chunk_fori(carry, nml, jnp.asarray(step), n_steps=1, cadence=rad)
            work = dict(zip(census.WORK, map(int, jax.device_get(carry.census.work))))
            for name in seen:
                if work[name] > previous[name]:
                    seen[name].append(step)
                previous[name] = work[name]
        assert seen["kf_calls"] == (np.flatnonzero(expected[:, 0]) + 1).tolist()
        assert seen["radiation_tendency_calls"] == (np.flatnonzero(expected[:, 1]) + 1).tolist()
    finally:
        op._advance_chunk_fori.clear_cache()


@pytest.mark.parametrize("nca", [27., 54., 81., 108.])
def test_kf_consumes_final_step_then_clears_returned_carry(monkeypatch, wrf_oracles, nca):
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = op._enforce_operational_precision(fixture._state(grid), force_fp64=True)
    dt, rqv, rain, rth = 54., 1.e-5, .01, 2.e-3
    rates = (jnp.full_like(state.theta, rth), jnp.full_like(state.qv, rqv),
             *(jnp.zeros_like(state.theta) for _ in range(4)), jnp.full_like(state.t_skin, rain))
    nml = SimpleNamespace(dt_s=dt, cumulus_cadence_steps=6, cudt_minutes=5., grid=grid)
    # Tracing the unexecuted branch calls Python, so use a concrete shape-safe
    # adapter; the runtime census must remain unchanged on step2.
    monkeypatch.setattr(op, "kf_adapter", lambda st, dt, w, n, **kw:
                        (kw["held_tendencies"], w, n))
    out, carry = jax.jit(lambda st, c: op._kf_cadence_step(st, c, nml, jnp.asarray(2)))(
        state, seed(state, nca, rates))
    expected = wrf_oracles("control_oracle", [(2, dt, 6, 33, 5., 30., nca, rqv, rain)])[0, 2:]
    got = [float((out.qv-state.qv)[0, 0, 0]), float(carry.cumulus_tendencies[1][0, 0, 0]),
           float(carry.cumulus_carry[1][0, 0]), float(out.rainc_acc[0, 0])]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1.e-7)
    # Exact WRF conv_t_tendf_to_moist at OLD state, not a product of updated fields.
    ratio = 461.6 / 287.
    theta_rate = (1 + ratio * state.qv) * rth + ratio * state.theta / (1 + ratio * state.qv) * rqv
    np.testing.assert_allclose(out.theta, state.theta + dt * theta_rate, rtol=1.e-13, atol=0)
    assert int(carry.census.work[census.WORK.index("kf_calls")]) == 0


def test_first_production_step_refreshes_complete_heating_before_consumption(monkeypatch):
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = op._enforce_operational_precision(fixture._state(grid), force_fp64=True)
    nml = dataclasses.replace(fixture._namelist(grid), dt_s=54.,
        mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=999, use_noahmp=False,
        cu_physics=0, radiation_cadence_steps=33, run_boundary=False, disable_guards=True, force_fp64=True)
    initial = initial_operational_carry(state).replace(
        radiation_diagnostics=(jnp.zeros_like(state.t_skin),), census=census.initial_census())
    monkeypatch.setattr(op, "surface_adapter", lambda st, *a, **kw: st)
    monkeypatch.setattr(op, "_rk_scan_step", lambda carry, *a, **kw: carry)
    monkeypatch.setattr(op, "rrtmg_theta_tendency", lambda st, *a, **kw:
        (jnp.full_like(st.theta, 1.e-4), (jnp.full_like(st.t_skin, kw["lead_seconds"]),)))
    op._advance_chunk_fori.clear_cache()
    try:
        carry = op._advance_chunk_fori(initial, nml, jnp.asarray(1), n_steps=1, cadence=33)
        np.testing.assert_allclose(carry.state.theta - state.theta, .0054, rtol=0, atol=1.e-12)
        np.testing.assert_array_equal(carry.rthraten, 1.e-4)
        np.testing.assert_array_equal(carry.radiation_diagnostics[0], 0.)
        assert int(carry.census.work[census.WORK.index("radiation_tendency_calls")]) == 1
    finally:
        op._advance_chunk_fori.clear_cache()


@pytest.mark.parametrize("minute,day", [(720., 0), (1430., 0), (5., 1)])
def test_midpoint_preserves_current_julian_declination_and_eot(wrf_oracles, minute, day):
    julian, radt, lat, lon = 207. + day, 30., 28.3, -16.4
    expected = wrf_oracles("solar_oracle", [(julian, minute, 0., radt, lat, lon)])[0, :3]
    base = (jnp.asarray(207.), jnp.asarray(minute))
    current = float(day * 86400)
    got = pc._compute_solar_geometry(jnp.asarray(lat), jnp.asarray(lon), None, current,
        clock_base=base, solar_lead_seconds=current + 900.)
    np.testing.assert_allclose([got.declination_rad, got.coszen, got.hour_angle_rad], expected,
                               rtol=0, atol=1.e-6)
