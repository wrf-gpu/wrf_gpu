from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.dynamics.core.coupled import CoupledCoreConfig
from gpuwrf.dynamics.core.dycore import DycoreCoreConfig
from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf


def _inputs(*, periodic_x: bool = True, specified: bool = False, nested: bool = False) -> AdvanceMuTInputs:
    rng = np.random.default_rng(40040)
    nz, ny, nx = 3, 6, 7
    mu = rng.normal(20.0, 2.0, size=(ny, nx))
    mut = rng.normal(85000.0, 120.0, size=(ny, nx))
    mu_work = rng.normal(0.5, 0.1, size=(ny, nx))
    theta = rng.normal(5.0, 0.5, size=(nz, ny, nx))
    return AdvanceMuTInputs(
        ww=jnp.asarray(rng.normal(0.0, 0.1, size=(nz + 1, ny, nx))),
        ww_1=jnp.asarray(rng.normal(0.0, 0.1, size=(nz + 1, ny, nx))),
        u=jnp.asarray(rng.normal(1.0, 0.5, size=(nz, ny, nx + 1))),
        u_1=jnp.asarray(rng.normal(1.0, 0.5, size=(nz, ny, nx + 1))),
        v=jnp.asarray(rng.normal(-0.5, 0.5, size=(nz, ny + 1, nx))),
        v_1=jnp.asarray(rng.normal(-0.5, 0.5, size=(nz, ny + 1, nx))),
        mu=jnp.asarray(mu),
        mut=jnp.asarray(mut),
        muave=jnp.asarray(rng.normal(0.0, 0.1, size=(ny, nx))),
        muts=jnp.asarray(mut + mu_work),
        muu=jnp.asarray(rng.normal(85020.0, 120.0, size=(ny, nx + 1))),
        muv=jnp.asarray(rng.normal(85020.0, 120.0, size=(ny + 1, nx))),
        mudf=jnp.asarray(rng.normal(0.0, 0.1, size=(ny, nx))),
        theta=jnp.asarray(theta),
        theta_1=jnp.asarray(theta + rng.normal(0.0, 0.01, size=(nz, ny, nx))),
        theta_ave=jnp.asarray(theta),
        theta_tend=jnp.asarray(rng.normal(0.0, 1.0e-5, size=(nz, ny, nx))),
        mu_tend=jnp.asarray(rng.normal(0.0, 1.0e-4, size=(ny, nx))),
        dnw=jnp.asarray([-0.25, -0.35, -0.40]),
        fnm=jnp.asarray([0.0, 0.55, 0.60]),
        fnp=jnp.asarray([0.0, 0.45, 0.40]),
        rdnw=jnp.asarray([-4.0, -2.857142857142857, -2.5]),
        c1h=jnp.asarray([0.8, 0.6, 0.4]),
        c2h=jnp.asarray([1000.0, 2000.0, 3000.0]),
        msfuy=jnp.asarray(rng.uniform(0.95, 1.05, size=(ny, nx + 1))),
        msfvx_inv=jnp.asarray(rng.uniform(0.95, 1.05, size=(ny + 1, nx))),
        msftx=jnp.asarray(rng.uniform(0.95, 1.05, size=(ny, nx))),
        msfty=jnp.asarray(rng.uniform(0.95, 1.05, size=(ny, nx))),
        rdx=1.0 / 3000.0,
        rdy=1.0 / 3000.0,
        dts=15.0,
        epssm=0.5,
        periodic_x=periodic_x,
        specified=specified,
        nested=nested,
    )


def _assert_2d_edges_unchanged(after, before):
    np.testing.assert_array_equal(after[0, :], before[0, :])
    np.testing.assert_array_equal(after[-1, :], before[-1, :])
    np.testing.assert_array_equal(after[:, 0], before[:, 0])
    np.testing.assert_array_equal(after[:, -1], before[:, -1])


def _assert_3d_edges_unchanged(after, before):
    np.testing.assert_array_equal(after[:, 0, :], before[:, 0, :])
    np.testing.assert_array_equal(after[:, -1, :], before[:, -1, :])
    np.testing.assert_array_equal(after[:, :, 0], before[:, :, 0])
    np.testing.assert_array_equal(after[:, :, -1], before[:, :, -1])


def test_specified_nonperiodic_keeps_lateral_mass_and_theta_edges_unchanged():
    inp = _inputs(periodic_x=False, specified=True)
    out = advance_mu_t_wrf(inp)

    for name in ("mu", "mudf", "muts", "muave"):
        before = np.asarray(getattr(inp, name))
        _assert_2d_edges_unchanged(np.asarray(out[name]), before)
    _assert_2d_edges_unchanged(np.asarray(out["dmdt"]), np.zeros_like(np.asarray(out["dmdt"])))
    for name in ("ww", "theta"):
        _assert_3d_edges_unchanged(np.asarray(out[name]), np.asarray(getattr(inp, name)))
    for name in ("dvdxi", "theta_tendency"):
        _assert_3d_edges_unchanged(np.asarray(out[name]), np.zeros_like(np.asarray(out[name])))
    _assert_3d_edges_unchanged(np.asarray(out["wdtn"]), np.zeros_like(np.asarray(out["wdtn"])))

    assert np.max(np.abs(np.asarray(out["mu"])[1:-1, 1:-1] - np.asarray(inp.mu)[1:-1, 1:-1])) > 0.0


def test_specified_periodic_x_keeps_y_edges_but_advances_x_edges():
    inp = _inputs(periodic_x=True, specified=True)
    out = advance_mu_t_wrf(inp)

    np.testing.assert_array_equal(np.asarray(out["mu"])[0, :], np.asarray(inp.mu)[0, :])
    np.testing.assert_array_equal(np.asarray(out["mu"])[-1, :], np.asarray(inp.mu)[-1, :])
    west_delta = np.max(np.abs(np.asarray(out["mu"])[1:-1, 0] - np.asarray(inp.mu)[1:-1, 0]))
    east_delta = np.max(np.abs(np.asarray(out["mu"])[1:-1, -1] - np.asarray(inp.mu)[1:-1, -1]))
    assert max(west_delta, east_delta) > 0.0


def test_legacy_periodic_path_still_advances_edge_mass_cells():
    inp = _inputs()
    out = advance_mu_t_wrf(inp)

    edge_delta = max(
        float(np.max(np.abs(np.asarray(out["mu"])[0, :] - np.asarray(inp.mu)[0, :]))),
        float(np.max(np.abs(np.asarray(out["mu"])[-1, :] - np.asarray(inp.mu)[-1, :]))),
        float(np.max(np.abs(np.asarray(out["mu"])[:, 0] - np.asarray(inp.mu)[:, 0]))),
        float(np.max(np.abs(np.asarray(out["mu"])[:, -1] - np.asarray(inp.mu)[:, -1]))),
    )
    assert edge_delta > 0.0


def test_shared_core_configs_thread_lbc_flags_to_acoustic_config():
    dycore_cfg = DycoreCoreConfig(
        dt=60.0,
        dx=9000.0,
        dy=9000.0,
        periodic_x=False,
        specified=True,
        nested=False,
    )
    acoustic_cfg = dycore_cfg.acoustic_config()
    assert acoustic_cfg.periodic_x is False
    assert acoustic_cfg.specified is True
    assert acoustic_cfg.nested is False

    coupled_cfg = CoupledCoreConfig(
        dt=60.0,
        dx=9000.0,
        dy=9000.0,
        periodic_x=False,
        specified=False,
        nested=True,
    )
    coupled_acoustic_cfg = coupled_cfg.dycore_config().acoustic_config()
    assert coupled_acoustic_cfg.periodic_x is False
    assert coupled_acoustic_cfg.specified is False
    assert coupled_acoustic_cfg.nested is True
