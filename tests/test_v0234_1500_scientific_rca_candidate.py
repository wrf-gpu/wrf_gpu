"""CPU-only gates for the exact nested spec-ring cadence correction."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    spec_bdyupdate_ph_tendency_inloop,
    specified_boundary_tendency,
)
from gpuwrf.dynamics.core import acoustic
from scripts import v0234_1500_scientific_rca_source_oracle as oracle


CFG = BoundaryConfig(
    spec_bdy_width=5,
    spec_zone=1,
    relax_zone=4,
    update_cadence_s=18.0,
    force_geopotential=False,
    nested_frozen_wrf_boundary_bundle=True,
)


def _record(*, z_len: int, side_len: int) -> tuple[jnp.ndarray, np.ndarray]:
    values = np.zeros((4, 5, z_len, side_len), dtype=np.float64)
    for side in range(4):
        for level in range(z_len):
            values[side, 0, level, :] = (
                100.0 * side + 10.0 * level + np.arange(side_len)
            )
    records = np.stack((np.zeros_like(values), 18.0 * values), axis=0)
    return jnp.asarray(records), values


def _manual_spec(values: np.ndarray, *, z_len: int, ny: int, nx: int) -> np.ndarray:
    out = np.zeros((z_len, ny, nx), dtype=np.float64)
    # Boundary storage order is W, E, S, N.  Y sides own corners.
    out[:, 1:-1, 0] = values[0, 0, :z_len, 1 : ny - 1]
    out[:, 1:-1, -1] = values[1, 0, :z_len, 1 : ny - 1]
    out[:, 0, :] = values[2, 0, :z_len, :nx]
    out[:, -1, :] = values[3, 0, :z_len, :nx]
    return out


def test_spec_boundary_tendency_closes_mass_u_and_v_staggers() -> None:
    for z_len, ny, nx in ((3, 8, 10), (3, 9, 9), (4, 8, 9)):
        records, values = _record(z_len=z_len, side_len=max(ny, nx) + 3)
        actual = specified_boundary_tendency(
            records,
            6.0,
            18.0,
            z_len=z_len,
            y_len=ny,
            x_len=nx,
            dtype=jnp.float64,
            config=CFG,
        )
        np.testing.assert_array_equal(
            np.asarray(actual), _manual_spec(values, z_len=z_len, ny=ny, nx=nx)
        )


def test_exact_ph_update_matches_independent_numpy_and_keeps_advanced_interior() -> None:
    rng = np.random.default_rng(23415)
    nzp1, ny, nx = 4, 7, 8
    ph_before = rng.normal(size=(nzp1, ny, nx))
    # The advance_w output MUST differ from the pre-advance array everywhere so
    # that any regression back to the rejected single-operand formula (which
    # returned the pre-advance interior and froze the interior geopotential;
    # the v0.23.4 step-200 d03 Ni blocker) turns this gate red.
    ph_advanced = ph_before + 1.0 + rng.normal(size=(nzp1, ny, nx))
    ph_save = 100.0 + rng.normal(size=(nzp1, ny, nx))
    ph_tend = rng.normal(size=(nzp1, ny, nx))
    mu_tend = rng.normal(size=(ny, nx))
    muts = 9000.0 + rng.normal(size=(ny, nx))
    c1f = np.linspace(0.2, 1.0, nzp1)
    c2f = np.linspace(2.0, 5.0, nzp1)
    dts = 0.6
    actual = np.asarray(
        spec_bdyupdate_ph_tendency_inloop(
            jnp.asarray(ph_advanced),
            jnp.asarray(ph_before),
            jnp.asarray(ph_tend),
            jnp.asarray(ph_save),
            jnp.asarray(mu_tend),
            jnp.asarray(muts),
            jnp.asarray(c1f),
            jnp.asarray(c2f),
            dts,
            CFG,
        )
    )
    expected_ring = oracle.wrf_spec_ph_update(
        ph_before,
        ph_save,
        ph_tend,
        mu_tend[None, :, :],
        muts[None, :, :],
        c1f[:, None, None],
        c2f[:, None, None],
        dts,
    )
    expected = ph_advanced.copy()
    expected[:, 0, :] = expected_ring[:, 0, :]
    expected[:, -1, :] = expected_ring[:, -1, :]
    expected[:, 1:-1, 0] = expected_ring[:, 1:-1, 0]
    expected[:, 1:-1, -1] = expected_ring[:, 1:-1, -1]
    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=3.0e-13)
    np.testing.assert_array_equal(actual[:, 1:-1, 1:-1], ph_advanced[:, 1:-1, 1:-1])
    assert np.all(
        np.abs(actual[:, 1:-1, 1:-1] - ph_before[:, 1:-1, 1:-1]) > 1.0e-8
    ), "interior must come from the advanced field, never the pre-advance array"


def test_nested_production_branch_uses_pre_substep_additive_ring_and_not_muave() -> None:
    source = inspect.getsource(acoustic.acoustic_substep_core)
    uv_source = inspect.getsource(acoustic.advance_uv_wrf)
    assert "state.u + dts * state.u_work_bdy" in uv_source
    assert "state.v + dts * state.v_work_bdy" in uv_source
    assert "state.mu.astype(mu_new.dtype)" in source
    assert "state.muts.astype(muts_new.dtype)" in source
    assert "state.theta_coupled_work.astype(theta_coupled.dtype)" in source
    nested_block = source.split(
        "if bool(cfg.nested_frozen_wrf_boundary_bundle):", 1
    )[1].split("else:", 1)[0]
    assert "muave_new = _pin_ring" not in nested_block
    assert "spec_bdyupdate_ph_tendency_inloop" in source
    assert "state_for_w.w" in source


def test_source_oracle_predicts_ninety_percent_first_substep_lead() -> None:
    wrf = oracle.wrf_spec_walk(0.0, 1.0, 0.6, 10)
    old = oracle.endpoint_pin_walk(0.0, 1.0, 0.6, 10)
    assert np.isclose(float((old[1] - wrf[1]) / wrf[-1]), 0.9, rtol=0.0, atol=3e-16)
