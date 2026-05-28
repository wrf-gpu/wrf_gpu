from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.dynamics.core.acoustic import AcousticCoreState, _decouple_theta_after_advance


def _acoustic_state(theta: jnp.ndarray, theta_1: jnp.ndarray) -> AcousticCoreState:
    nz, ny, nx = theta.shape
    surf = (ny, nx)
    u = jnp.zeros((nz, ny, nx + 1), dtype=jnp.float64)
    v = jnp.zeros((nz, ny + 1, nx), dtype=jnp.float64)
    w = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    mass = jnp.zeros(surf, dtype=jnp.float64)
    return AcousticCoreState(
        ww=w,
        ww_1=w,
        u=u,
        u_1=u,
        v=v,
        v_1=v,
        w=w,
        mu=mass,
        mut=90000.0 + mass,
        muave=mass,
        muts=90000.0 + mass,
        muu=jnp.zeros((ny, nx + 1), dtype=jnp.float64),
        muv=jnp.zeros((ny + 1, nx), dtype=jnp.float64),
        mudf=mass,
        theta=theta,
        theta_1=theta_1,
        theta_ave=theta,
        theta_tend=jnp.zeros_like(theta),
        mu_tend=mass,
        ph_tend=w,
        ph=w,
        p=jnp.zeros_like(theta),
        t_2ave=theta,
        dnw=jnp.ones((nz,), dtype=jnp.float64),
        fnm=jnp.ones((nz,), dtype=jnp.float64) * 0.5,
        fnp=jnp.ones((nz,), dtype=jnp.float64) * 0.5,
        rdnw=jnp.ones((nz,), dtype=jnp.float64),
        c1h=jnp.linspace(0.2, 0.8, nz, dtype=jnp.float64),
        c2h=jnp.linspace(1000.0, 4000.0, nz, dtype=jnp.float64),
        msfuy=jnp.ones((ny, nx + 1), dtype=jnp.float64),
        msfvx_inv=jnp.ones((ny + 1, nx), dtype=jnp.float64),
        msftx=jnp.ones((ny, nx), dtype=jnp.float64),
        msfty=jnp.ones((ny, nx), dtype=jnp.float64),
    )


def test_decouple_theta_after_advance_uses_theta_1_reference_state():
    theta = jnp.ones((3, 2, 2), dtype=jnp.float64) * 10.0
    theta_1 = jnp.ones((3, 2, 2), dtype=jnp.float64) * 12.0
    state = _acoustic_state(theta, theta_1)
    theta_mass = jnp.ones_like(theta) * 5.0
    muts_new = state.muts + 100.0

    got = _decouple_theta_after_advance(state, theta_mass, muts_new)

    numerator_theta_1 = theta_mass + state.theta_1 * (state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None])
    numerator_running_theta = theta_mass + state.theta * (state.c1h[:, None, None] * state.mut[None, :, :] + state.c2h[:, None, None])
    denominator = state.c1h[:, None, None] * muts_new[None, :, :] + state.c2h[:, None, None]
    expected = numerator_theta_1 / denominator
    wrong = numerator_running_theta / denominator

    assert float(jnp.max(jnp.abs(got - expected))) <= 1.0e-12
    assert float(jnp.max(jnp.abs(got - wrong))) > 1.0
