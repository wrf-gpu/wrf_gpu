from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.flux_advection import couple_velocities_periodic
from gpuwrf.runtime import operational_mode as runtime


jax.config.update("jax_enable_x64", True)


def _fixture():
    rng = np.random.default_rng(9314)
    nz, ny, nx = 5, 7, 8
    u = jnp.asarray(rng.normal(size=(nz, ny, nx + 1)))
    v = jnp.asarray(rng.normal(size=(nz, ny + 1, nx)))
    mu = jnp.asarray(8.0e4 + 2.0e3 * rng.normal(size=(ny, nx)))
    c1h = jnp.asarray(np.linspace(0.9, 0.4, nz))
    c2h = jnp.asarray(np.linspace(100.0, 500.0, nz))
    dnw = jnp.asarray(-np.linspace(0.12, 0.05, nz))
    msftx = jnp.asarray(0.995 + 0.01 * rng.random(size=(ny, nx)))
    msfuy = jnp.asarray(0.995 + 0.01 * rng.random(size=(ny, nx + 1)))
    msfvx = jnp.asarray(0.995 + 0.01 * rng.random(size=(ny + 1, nx)))
    metrics = SimpleNamespace(
        c1h=c1h,
        c2h=c2h,
        dnw=dnw,
        msftx=msftx,
        msfuy=msfuy,
        msfvx=msfvx,
        msfux=msfuy,
        msfvy=msfvx,
    )
    namelist = SimpleNamespace(
        metrics=metrics,
        grid=SimpleNamespace(projection=SimpleNamespace(dx_m=1000.0, dy_m=1200.0)),
    )
    state = SimpleNamespace(u=u, v=v, mu_total=mu)
    return state, namelist


def _numpy_calc_ww_cp(state, namelist):
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    mu = np.asarray(state.mu_total)
    m = namelist.metrics
    c1 = np.asarray(m.c1h)[:, None, None]
    c2 = np.asarray(m.c2h)[:, None, None]
    muu = np.concatenate(
        [
            0.5 * (np.pad(mu, ((0, 0), (1, 0)), mode="edge")[:, 1:] +
                   np.pad(mu, ((0, 0), (1, 0)), mode="edge")[:, :-1]),
            mu[:, -1:],
        ],
        axis=1,
    )
    muv = np.concatenate(
        [
            0.5 * (np.pad(mu, ((1, 0), (0, 0)), mode="edge")[1:] +
                   np.pad(mu, ((1, 0), (0, 0)), mode="edge")[:-1]),
            mu[-1:, :],
        ],
        axis=0,
    )
    ru = (c1 * muu[None] + c2) * u / np.asarray(m.msfuy)[None]
    rv = (c1 * muv[None] + c2) * v / np.asarray(m.msfvx)[None]
    dnw = np.asarray(m.dnw)[:, None, None]
    divv = np.asarray(m.msftx)[None] * dnw * (
        (ru[:, :, 1:] - ru[:, :, :-1]) / namelist.grid.projection.dx_m
        + (rv[:, 1:, :] - rv[:, :-1, :]) / namelist.grid.projection.dy_m
    )
    dmdt = np.sum(divv, axis=0, keepdims=True)
    increments = -(dnw * np.asarray(m.c1h)[:, None, None] * dmdt) - divv
    out = np.zeros((u.shape[0] + 1,) + mu.shape, dtype=np.float64)
    out[1:-1] = np.cumsum(increments, axis=0)[:-1]
    return out


def test_candidate_routes_independent_edge_faithful_calc_ww_cp_into_transport(monkeypatch):
    state, namelist = _fixture()
    monkeypatch.setattr(runtime, "_specified_adv_degrade_active", lambda _: True)
    monkeypatch.setattr(runtime, "_stage_transport_omega_ownership_enabled", lambda: True)

    actual = runtime._stage_transport_velocities(state, namelist)
    oracle = _numpy_calc_ww_cp(state, namelist)
    np.testing.assert_allclose(np.asarray(actual.rom), oracle, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_array_equal(np.asarray(actual.rom[0]), 0.0)
    np.testing.assert_array_equal(np.asarray(actual.rom[-1]), 0.0)

    periodic = couple_velocities_periodic(
        state.u,
        state.v,
        state.mu_total,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        dnw=namelist.metrics.dnw,
        rdx=1.0 / namelist.grid.projection.dx_m,
        rdy=1.0 / namelist.grid.projection.dy_m,
        msfuy=namelist.metrics.msfuy,
        msfvx=namelist.metrics.msfvx,
        msftx=namelist.metrics.msftx,
        msfux=namelist.metrics.msfux,
        msfvy=namelist.metrics.msfvy,
    )
    assert float(np.max(np.abs(np.asarray(actual.rom - periodic.rom)))) > 1.0e-3


def test_retained_selector_preserves_periodic_rom_exactly(monkeypatch):
    state, namelist = _fixture()
    monkeypatch.setattr(runtime, "_specified_adv_degrade_active", lambda _: True)
    monkeypatch.setattr(runtime, "_stage_transport_omega_ownership_enabled", lambda: False)
    actual = runtime._stage_transport_velocities(state, namelist)
    retained = couple_velocities_periodic(
        state.u,
        state.v,
        state.mu_total,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        dnw=namelist.metrics.dnw,
        rdx=1.0 / namelist.grid.projection.dx_m,
        rdy=1.0 / namelist.grid.projection.dy_m,
        msfuy=namelist.metrics.msfuy,
        msfvx=namelist.metrics.msfvx,
        msftx=namelist.metrics.msftx,
        msfux=namelist.metrics.msfux,
        msfvy=namelist.metrics.msfvy,
    )
    np.testing.assert_array_equal(np.asarray(actual.rom), np.asarray(retained.rom))


def test_source_keeps_single_static_owner_without_interface_or_observer():
    import inspect

    source = inspect.getsource(runtime._stage_transport_velocities)
    assert source.count("rom = stage_omega_specified(") == 1
    assert "rom=rom" in source
    for forbidden in (
        "debug.callback",
        "pure_callback",
        "io_callback",
        "device_get",
        "block_until_ready",
    ):
        assert forbidden not in source
