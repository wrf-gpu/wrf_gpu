"""B51 / review-realall C2: the product glue feeds pg_buoy_w WRF's grid%mu_2 (prep.mu_save).

WRF rk_tendency passes grid%mu_2 to pg_buoy_w (module_em.F:730). The port used
prep.mut - prep.mub, a rounded REAL total minus the base (E133). Spies the real
call inside _acoustic_core_state_from_prep and checks the mass operand bitwise on
operands where f32(mub + mu') - mub != mu' (deletion-sensitive), moist and dry.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as op


class _Stop(Exception):
    pass


def _operands(nz=4, ny=3, nx=5):
    rng = np.random.default_rng(7)
    mub = jnp.asarray(9.0e4 + 1.0e3 * rng.random((ny, nx)), jnp.float32)
    mu2 = jnp.asarray(-500.0 + 1000.0 * rng.random((ny, nx)), jnp.float32)
    zeros = jnp.zeros((nz, ny, nx), jnp.float32)
    state = SimpleNamespace(theta=zeros + 300.0, p_perturbation=zeros,
                            qv=zeros + 1e-2, qc=zeros, qr=zeros, qi=zeros, qs=zeros, qg=zeros)
    prep = SimpleNamespace(entry_state=state, theta_offset=300.0, phb=jnp.zeros((nz + 1, ny, nx), jnp.float32),
                           theta_work=zeros, mub=mub, mut=mub + mu2, mu_save=mu2)
    vec = lambda n: jnp.ones((n,), jnp.float32)
    namelist = SimpleNamespace(boundary_config=SimpleNamespace(nested_frozen_wrf_boundary_bundle=False),
                               metrics=SimpleNamespace(c1f=vec(nz + 1), c2f=vec(nz + 1), rdnw=vec(nz), rdn=vec(nz),
                                                       msfty=jnp.ones((ny, nx), jnp.float32)))
    return prep, namelist, mu2


@pytest.mark.parametrize("moist", (True, False))
def test_pg_buoy_w_receives_grid_mu_2(monkeypatch, moist):
    prep, namelist, mu2 = _operands()
    # Precondition: the rounded-total form really differs on these operands.
    assert not np.array_equal(np.asarray(prep.mut - prep.mub), np.asarray(mu2))
    seen = {}

    def spy(p, mu, *args, **kwargs):
        seen["mu"] = mu
        raise _Stop

    monkeypatch.setattr(op, "_moist_cqw_enabled", lambda: moist)
    monkeypatch.setattr(op, "pg_buoy_w_moist" if moist else "pg_buoy_w_dry", spy)
    with pytest.raises(_Stop):
        op._acoustic_core_state_from_prep(None, prep, None, namelist, None)
    assert np.array_equal(np.asarray(seen["mu"]), np.asarray(mu2))
