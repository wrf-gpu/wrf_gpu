import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.physics_couplers import _wrf_phy_prep_rho_from_state
from gpuwrf.physics.surface_constants import EP1, G, P0_PA, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics


def _column_state(**overrides):
    qv = overrides.pop("qv", 0.1)
    p = overrides.pop("p", 90000.0)
    t_air = overrides.pop("t_air", 300.0)
    theta = t_air * (P0_PA / p) ** R_D_OVER_CP
    values = {
        "u": 2.0,
        "v": 0.0,
        "theta": theta,
        "qv": qv,
        "p": p,
        "dz": 50.0,
        "t_air": t_air,
        "t_skin": 295.0,
        "psfc": 91000.0,
        "xland": 1.0,
        "lakemask": 0.0,
        "mavail": 0.3,
        "roughness_m": 0.1,
        "ustar": 0.2,
        "qsfc": qv / (1.0 + qv),
        "hfx": 0.0,
        "qfx": 0.0,
        "pblh": 1000.0,
        "dx_m": 3000.0,
    }
    values.update(overrides)
    state = SimpleNamespace()
    for name, value in values.items():
        arr = jnp.asarray([[value]], dtype=jnp.float64)
        if name in {"u", "v", "theta", "qv", "p", "dz", "t_air"}:
            arr = arr[..., None]
        setattr(state, name, arr)
    return state


def test_first_timestep_br_clamps_to_wrf_narrow_limit():
    cold = _column_state(u=0.1, v=0.0, t_air=280.0, t_skin=330.0, qv=0.01)

    first = surface_layer_with_diagnostics(cold, first_timestep=True)
    warm = surface_layer_with_diagnostics(cold, first_timestep=False)

    assert np.asarray(first.br).item() == -2.0
    assert np.asarray(warm.br).item() == -4.0


def test_virtual_theta_uses_specific_humidity_qvsh_for_br():
    state = _column_state()
    diag = surface_layer_with_diagnostics(state, first_timestep=True)

    qv = 0.1
    qvsh = qv / (1.0 + qv)
    p = 90000.0
    psfc = 91000.0
    t_air = 300.0
    t_skin = 295.0
    dz = 50.0
    thx = t_air * (P0_PA / p) ** R_D_OVER_CP
    thgb = t_skin * (P0_PA / psfc) ** R_D_OVER_CP
    qsfc = qvsh
    expected = (G / thx) * (0.5 * dz) * (thx * (1.0 + EP1 * qvsh) - thgb * (1.0 + EP1 * qsfc)) / (2.0**2)
    mixing_ratio_wrong = (G / thx) * (0.5 * dz) * (thx * (1.0 + EP1 * qv) - thgb * (1.0 + EP1 * qsfc)) / (2.0**2)

    actual = np.asarray(diag.br).item()
    assert actual == np.asarray(jnp.clip(expected, -2.0, 2.0)).item()
    assert abs(actual - mixing_ratio_wrong) > 0.05


def test_wrf_phy_prep_rho_reconstructs_alt_density():
    alt = np.asarray([[[0.84]]], dtype=np.float32)
    qv = np.asarray([[[0.02]]], dtype=np.float32)
    p_top = np.float32(10000.0)
    p_up = np.float32(80000.0)
    p_down = np.float32(90000.0)
    p_mid = np.float32(0.5 * (float(p_up) + float(p_down)))
    dph = (alt * p_mid * np.log(p_down / p_up)).astype(np.float32)

    state = SimpleNamespace(
        mu_total=jnp.asarray([[1.0]], dtype=jnp.float64),
        ph_total=jnp.asarray(np.concatenate([np.zeros_like(dph), dph], axis=0), dtype=jnp.float64),
        qv=jnp.asarray(qv, dtype=jnp.float64),
    )
    metrics = SimpleNamespace(
        c3f=jnp.asarray([p_down - p_top, p_up - p_top], dtype=jnp.float64),
        c4f=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        c3h=jnp.asarray([p_mid - p_top], dtype=jnp.float64),
        c4h=jnp.asarray([0.0], dtype=jnp.float64),
        p_top=jnp.asarray(p_top, dtype=jnp.float64),
    )

    rho = _wrf_phy_prep_rho_from_state(state, metrics)

    np.testing.assert_allclose(np.asarray(rho), (1.0 + qv) / alt, rtol=0.0, atol=2.0e-7)
