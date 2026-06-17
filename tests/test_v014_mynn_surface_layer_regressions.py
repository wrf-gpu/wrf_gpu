import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.physics_couplers import _wrf_phy_prep_rho_from_state
from gpuwrf.physics.surface_constants import EP1, G, P0_PA, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
from gpuwrf.runtime.operational_mode import _assert_nonzero_initial_mu_total


def _tiny_grid(nz: int) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    return GridSpec(
        Projection(kind="lambert", lat_0=28.0, lon_0=-16.0, dx_m=1000.0, dy_m=1000.0, nx=1, ny=1),
        TerrainProvenance(
            source_path="synthetic",
            sha256="0" * 64,
            shape=(1, 1),
            units="m",
            projection_transform="identity",
            max_elevation_m=0.0,
            coastline_sanity_check_passed=True,
        ),
        VerticalCoord(kind="hybrid_eta", nz=nz, top_pressure_pa=5000.0, eta_levels=eta),
        BCMetadata(source="ideal", fields=(), update_cadence_h=1, interpolation="linear", restart_compatible=True),
        eta,
        jnp.zeros((1, 1), dtype=jnp.float64),
        halo_width=1,
    )


def _tiny_state(nz: int) -> State:
    fields = {
        name: jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(name))
        for name, shape in _state_field_shapes(_tiny_grid(nz), mp_physics=8).items()
    }
    return State(**fields)


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
    alt = np.full((3, 1, 1), 0.84, dtype=np.float32)
    qv = np.full((3, 1, 1), 0.02, dtype=np.float32)
    p_top = np.float32(10000.0)
    p_faces = np.asarray([90000.0, 80000.0, 70000.0, 60000.0], dtype=np.float32)
    p_mid = np.float32(0.5) * (p_faces[:-1] + p_faces[1:])
    dph = (alt[:, 0, 0] * p_mid * np.log(p_faces[:-1] / p_faces[1:])).astype(np.float32)
    ph = np.concatenate([[0.0], np.cumsum(dph, dtype=np.float32)]).astype(np.float32)[:, None, None]

    state = _tiny_state(3).replace(
        mu_total=jnp.asarray([[1.0]], dtype=jnp.float64),
        ph_total=jnp.asarray(ph, dtype=jnp.float64),
        qv=jnp.asarray(qv, dtype=jnp.float64),
    )
    metrics = SimpleNamespace(
        c3f=jnp.asarray(p_faces - p_top, dtype=jnp.float64),
        c4f=jnp.zeros((4,), dtype=jnp.float64),
        c3h=jnp.asarray(p_mid - p_top, dtype=jnp.float64),
        c4h=jnp.zeros((3,), dtype=jnp.float64),
        p_top=jnp.asarray(p_top, dtype=jnp.float64),
    )

    rho = _wrf_phy_prep_rho_from_state(state, metrics)

    np.testing.assert_allclose(np.asarray(rho), (1.0 + qv) / alt, rtol=0.0, atol=3.0e-7)


def test_wrf_phy_prep_rho_uses_state_replace_synced_totals():
    qv = np.full((3, 1, 1), 0.005, dtype=np.float32)
    ph = np.asarray([0.0, 5000.0 * G, 10000.0 * G, 15000.0 * G], dtype=np.float32)[:, None, None]
    state = _tiny_state(3).replace(
        mu_total=jnp.full((1, 1), 90000.0, dtype=jnp.float64),
        ph_total=jnp.asarray(ph, dtype=jnp.float64),
        qv=jnp.asarray(qv, dtype=jnp.float64),
    )
    np.testing.assert_allclose(np.asarray(state.mu), np.asarray(state.mu_total))
    np.testing.assert_allclose(np.asarray(state.ph), np.asarray(state.ph_total))
    metrics = SimpleNamespace(
        c3f=jnp.asarray([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=jnp.float64),
        c4f=jnp.zeros((4,), dtype=jnp.float64),
        c3h=jnp.asarray([5.0 / 6.0, 0.5, 1.0 / 6.0], dtype=jnp.float64),
        c4h=jnp.zeros((3,), dtype=jnp.float64),
        p_top=jnp.asarray(5000.0, dtype=jnp.float64),
    )

    rho = np.asarray(_wrf_phy_prep_rho_from_state(state, metrics))

    assert np.all(np.isfinite(rho))
    assert np.min(rho) > 0.0


def test_operational_init_fails_loud_on_zero_mu_total():
    state = _tiny_state(3)

    with pytest.raises(ValueError, match="zero mu_total"):
        _assert_nonzero_initial_mu_total(state)
