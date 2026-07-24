"""WRF-source regression and compatibility gates for static RRTMG gases."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.rrtmg_constants import CH4_VMR, CO2_VMR, N2O_VMR
from gpuwrf.physics.rrtmg_lw import (
    RRTMGLWColumnState,
    _CFC_VMR,
    solve_rrtmg_lw_column,
)
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time


jax.config.update("jax_enable_x64", True)


def _table(tmp_path: Path) -> Path:
    path = tmp_path / "CAMtr_volume_mixing_ratio.SSP245"
    path.write_text(
        "## year | co2 | n2o | ch4 | cfc11 | cfc12\n"
        "## source-faithful focused bracket\n"
        "2024  426.069    335.186   1951.461    207.208    476.333\n"
        "2025  429.030    335.980   1960.651    204.429    471.673\n"
        "2026  432.011    336.779   1969.541    201.646    467.030\n"
        "2027  435.008    337.581   1978.155    198.879    462.431\n",
        encoding="ascii",
    )
    return path


def test_clwrf_ssp245_interpolation_matches_wrf_default_real_replay(
    tmp_path: Path,
) -> None:
    table = _table(tmp_path)
    d03 = clwrf_ssp245_gases_for_time(
        "2025-03-01T00:00:00Z", table_path=table
    )
    np.testing.assert_array_equal(
        np.asarray(d03, dtype=np.float64),
        np.asarray(
            (
                0.00042817414821614197,
                3.3575049186939293e-07,
                1.9579946568973677e-06,
                2.0523224440220293e-10,
                4.730199296449112e-10,
            ),
            dtype=np.float64,
        ),
    )

    real_wrf = clwrf_ssp245_gases_for_time(
        "2026-04-28T18:00:00+00:00", table_path=table
    )
    np.testing.assert_array_equal(
        np.asarray(real_wrf, dtype=np.float64),
        np.asarray(
            (
                0.00043162917303889273,
                3.366766427747367e-07,
                1.9684022349994908e-06,
                2.0200244590691145e-10,
                4.676246698024074e-10,
            ),
            dtype=np.float64,
        ),
    )


def test_clwrf_run_date_fraction_and_timezone_are_material(tmp_path: Path) -> None:
    table = _table(tmp_path)
    midnight = clwrf_ssp245_gases_for_time(
        "2026-04-28T00:00:00Z", table_path=table
    )
    eighteen_utc = clwrf_ssp245_gases_for_time(
        "2026-04-28T18:00:00Z", table_path=table
    )
    equivalent = clwrf_ssp245_gases_for_time(
        "2026-04-28T19:00:00+01:00", table_path=table
    )
    assert eighteen_utc == equivalent
    assert midnight != eighteen_utc
    assert eighteen_utc.co2_vmr > midnight.co2_vmr


def test_clwrf_reuses_the_co2_selected_bracket_for_every_gas(
    tmp_path: Path,
) -> None:
    """Pin the non-obvious single ``valid_years(co2r, ...)`` source rule."""

    table = _table(tmp_path)
    text = table.read_text(encoding="ascii").replace(
        "2025  429.030    335.980",
        "2025  429.030   -999.000",
    )
    table.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match="negative gas value"):
        clwrf_ssp245_gases_for_time(
            "2025-03-01T00:00:00Z",
            table_path=table,
        )


def _lw_state() -> RRTMGLWColumnState:
    temperature = jnp.asarray([[291.0, 264.0, 235.0]], dtype=jnp.float64)
    pressure = jnp.asarray([[90000.0, 48000.0, 13000.0]], dtype=jnp.float64)
    zero = jnp.zeros_like(temperature)
    return RRTMGLWColumnState(
        temperature,
        pressure,
        jnp.full_like(temperature, 2.0e-3),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([292.0], dtype=jnp.float64),
        jnp.asarray([0.98], dtype=jnp.float64),
        jnp.full_like(temperature, 700.0),
        jnp.full_like(temperature, 0.8),
    )


def _sw_state() -> RRTMGSWColumnState:
    temperature = jnp.asarray([[291.0, 264.0, 235.0]], dtype=jnp.float64)
    pressure = jnp.asarray([[90000.0, 48000.0, 13000.0]], dtype=jnp.float64)
    zero = jnp.zeros_like(temperature)
    return RRTMGSWColumnState(
        temperature,
        pressure,
        jnp.full_like(temperature, 2.0e-3),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([0.15], dtype=jnp.float64),
        jnp.asarray([0.55], dtype=jnp.float64),
        jnp.full_like(temperature, 700.0),
        jnp.full_like(temperature, 0.8),
    )


def test_omitted_gas_metadata_is_solver_byte_identical_to_historical_constants() -> None:
    lw_legacy = _lw_state()
    lw_explicit = lw_legacy.replace(
        co2_vmr=CO2_VMR,
        n2o_vmr=N2O_VMR,
        ch4_vmr=CH4_VMR,
        cfc11_vmr=float(_CFC_VMR[1]),
        cfc12_vmr=float(_CFC_VMR[2]),
    )
    lw_old = solve_rrtmg_lw_column(lw_legacy, debug=False)
    lw_new = solve_rrtmg_lw_column(lw_explicit, debug=False)

    sw_legacy = _sw_state()
    sw_explicit = sw_legacy.replace(
        co2_vmr=CO2_VMR,
        n2o_vmr=N2O_VMR,
        ch4_vmr=CH4_VMR,
    )
    sw_old = solve_rrtmg_sw_column(sw_legacy, debug=False)
    sw_new = solve_rrtmg_sw_column(sw_explicit, debug=False)
    jax.block_until_ready(
        (lw_old.surface_down, lw_new.surface_down, sw_old.surface_down, sw_new.surface_down)
    )
    for old, new in ((lw_old, lw_new), (sw_old, sw_new)):
        for old_field, new_field in zip(old, new, strict=True):
            if old_field is None or new_field is None:
                assert old_field is new_field
            else:
                np.testing.assert_array_equal(np.asarray(old_field), np.asarray(new_field))


def test_gas_metadata_is_static_pytree_state_and_must_be_complete() -> None:
    lw = _lw_state()
    sw = _sw_state()
    lw_leaves = jax.tree_util.tree_leaves(lw)
    sw_leaves = jax.tree_util.tree_leaves(sw)
    lw_gases = lw.replace(
        co2_vmr=4.2e-4,
        n2o_vmr=3.3e-7,
        ch4_vmr=1.9e-6,
        cfc11_vmr=2.0e-10,
        cfc12_vmr=4.7e-10,
    )
    sw_gases = sw.replace(co2_vmr=4.2e-4, n2o_vmr=3.3e-7, ch4_vmr=1.9e-6)
    assert len(jax.tree_util.tree_leaves(lw_gases)) == len(lw_leaves)
    assert len(jax.tree_util.tree_leaves(sw_gases)) == len(sw_leaves)
    assert jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(lw_gases), lw_leaves
    ).co2_vmr == 4.2e-4
    assert jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(sw_gases), sw_leaves
    ).co2_vmr == 4.2e-4

    with pytest.raises(ValueError, match="all supplied or all None"):
        lw.replace(co2_vmr=4.2e-4)
    with pytest.raises(ValueError, match="all supplied or all None"):
        sw.replace(co2_vmr=4.2e-4)
    with pytest.raises(ValueError, match="finite and positive"):
        lw.replace(
            co2_vmr=np.nan,
            n2o_vmr=3.3e-7,
            ch4_vmr=1.9e-6,
            cfc11_vmr=2.0e-10,
            cfc12_vmr=4.7e-10,
        )
