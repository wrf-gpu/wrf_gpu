"""CPU/algebra gates for pristine-WRF acoustic-averaged scalar mass fluxes."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.flux_advection import CoupledVelocities
from gpuwrf.runtime.operational_mode import (
    _finalize_time_averaged_scalar_mass_fluxes,
    _scalar_transport_velocities_from_sumflux,
)


def _fixture(seed: int = 234):
    rng = np.random.default_rng(seed)
    nz, ny, nx = 4, 7, 8
    prep = SimpleNamespace(
        c1h=jnp.asarray(rng.uniform(0.2, 1.1, size=nz)),
        c2h=jnp.asarray(rng.uniform(10.0, 80.0, size=nz)),
        muu=jnp.asarray(rng.uniform(600.0, 1100.0, size=(ny, nx + 1))),
        muv=jnp.asarray(rng.uniform(600.0, 1100.0, size=(ny + 1, nx))),
        msfuy=jnp.asarray(rng.uniform(0.8, 1.3, size=(ny, nx + 1))),
        msfvx=jnp.asarray(rng.uniform(0.8, 1.3, size=(ny + 1, nx))),
        u_save=jnp.asarray(rng.normal(size=(nz, ny, nx + 1))),
        v_save=jnp.asarray(rng.normal(size=(nz, ny + 1, nx))),
        ww_save=jnp.asarray(rng.normal(size=(nz + 1, ny, nx))),
    )
    return rng, prep


@pytest.mark.parametrize("n_sound", [1, 2, 4])
def test_sumflux_matches_literal_numpy_oracle(n_sound: int) -> None:
    """Independent literal translation of WRF's final ``sumflux`` algebra."""

    rng, prep = _fixture(seed=234 + n_sound)
    u_steps = rng.normal(size=(n_sound,) + tuple(prep.u_save.shape))
    v_steps = rng.normal(size=(n_sound,) + tuple(prep.v_save.shape))
    w_steps = rng.normal(size=(n_sound,) + tuple(prep.ww_save.shape))
    acoustic = SimpleNamespace(
        ru_m=jnp.asarray(np.sum(u_steps, axis=0)),
        rv_m=jnp.asarray(np.sum(v_steps, axis=0)),
        ww_m=jnp.asarray(np.sum(w_steps, axis=0)),
    )

    got = jax.jit(
        lambda ru_m, rv_m, ww_m: _finalize_time_averaged_scalar_mass_fluxes(
            SimpleNamespace(ru_m=ru_m, rv_m=rv_m, ww_m=ww_m),
            prep,
            number_of_small_timesteps=n_sound,
        )
    )(acoustic.ru_m, acoustic.rv_m, acoustic.ww_m)

    c1 = np.asarray(prep.c1h)[:, None, None]
    c2 = np.asarray(prep.c2h)[:, None, None]
    expected_u = np.sum(u_steps, axis=0) / n_sound + (
        (c1 * np.asarray(prep.muu)[None, :, :] + c2)
        * np.asarray(prep.u_save)
        / np.asarray(prep.msfuy)[None, :, :]
    )
    expected_v = np.sum(v_steps, axis=0) / n_sound + (
        (c1 * np.asarray(prep.muv)[None, :, :] + c2)
        * np.asarray(prep.v_save)
        / np.asarray(prep.msfvx)[None, :, :]
    )
    expected_w = np.sum(w_steps, axis=0) / n_sound + np.asarray(prep.ww_save)
    np.testing.assert_allclose(got.ru_full, expected_u, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(got.rv_full, expected_v, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(got.ww, expected_w, rtol=2e-15, atol=2e-15)


def test_sumflux_preserves_scalar_stagger_and_static_metadata() -> None:
    rng, prep = _fixture(seed=9000)
    fluxes = _finalize_time_averaged_scalar_mass_fluxes(
        SimpleNamespace(
            ru_m=jnp.asarray(rng.normal(size=prep.u_save.shape)),
            rv_m=jnp.asarray(rng.normal(size=prep.v_save.shape)),
            ww_m=jnp.asarray(rng.normal(size=prep.ww_save.shape)),
        ),
        prep,
        number_of_small_timesteps=4,
    )
    nz = int(prep.u_save.shape[0])
    ny = int(prep.u_save.shape[1])
    nx = int(prep.v_save.shape[-1])
    marker = jnp.asarray(rng.normal(size=(ny, nx)))
    stage = CoupledVelocities(
        ru=jnp.zeros((nz, ny, nx)),
        rv=jnp.zeros((nz, ny, nx)),
        rom=jnp.zeros((nz + 1, ny, nx)),
        msftx=marker,
        msfux=marker,
        msfvy=marker,
        msfvx=marker,
        specified=True,
        ru_full=jnp.zeros_like(prep.u_save),
        rv_full=jnp.zeros_like(prep.v_save),
    )

    got = _scalar_transport_velocities_from_sumflux(stage, fluxes)
    np.testing.assert_array_equal(got.ru, fluxes.ru_full[..., :nx])
    np.testing.assert_array_equal(got.rv, fluxes.rv_full[:, :ny, :])
    np.testing.assert_array_equal(got.rom, fluxes.ww)
    np.testing.assert_array_equal(got.ru_full, fluxes.ru_full)
    np.testing.assert_array_equal(got.rv_full, fluxes.rv_full)
    assert got.specified is True
    assert got.msftx is marker
    assert got.msfux is marker
    assert got.msfvy is marker
    assert got.msfvx is marker


def test_sumflux_fails_closed_without_substeps_or_accumulators() -> None:
    _, prep = _fixture()
    with pytest.raises(ValueError, match="at least one"):
        _finalize_time_averaged_scalar_mass_fluxes(
            SimpleNamespace(ru_m=None, rv_m=None, ww_m=None),
            prep,
            number_of_small_timesteps=0,
        )
    with pytest.raises(ValueError, match="accumulators"):
        _finalize_time_averaged_scalar_mass_fluxes(
            SimpleNamespace(ru_m=None, rv_m=None, ww_m=None),
            prep,
            number_of_small_timesteps=1,
        )
