"""WRF-source regression and compatibility gates for RRTMG gases."""

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
    solve_rrtmg_lw_m9_flux_slices,
)
from gpuwrf.physics.rrtmg_sw import (
    RRTMGSWColumnState,
    solve_rrtmg_sw_column,
    solve_rrtmg_sw_m9_flux_slices,
)
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


# v0.25 S3 calendar amendment (2026-09-23): the expected values below are the
# compiled WRF ``read_CAMgases`` on this table (WRF's preprocessed
# module_ra_clWRF_support.f90 sha256 b8e02d7e..., WRF gfortran flags, julian =
# REAL(dayOfYear_r8) - 1.0; oracle in .agent/sprints/2026-09-23-v0250-s3-date-
# correct-ghg-cache/wrf_clwrf_oracle).  The ORIGINAL host values pinned before
# the amendment came from a one-based-calendar replay that ran 12 h ahead of WRF.
_WRF_ORACLE_2025_03_01T00Z = (
    0.0004281701048356831,
    3.3574941421059136e-07,
    1.957982126281826e-06,
    2.0523605734600383e-10,
    4.730263272669047e-10,
)
_WRF_ORACLE_2026_04_28T18Z = (
    0.00043162510234999967,
    3.3667555829042865e-07,
    1.968390115608005e-06,
    2.0200626424722753e-10,
    4.676310439983451e-10,
)
_ORIGINAL_HOST_ONE_BASED_2025_03_01T00Z = (
    0.00042817414821614197,
    3.3575049186939293e-07,
    1.9579946568973677e-06,
    2.0523224440220293e-10,
    4.730199296449112e-10,
)
_ORIGINAL_HOST_ONE_BASED_2026_04_28T18Z = (
    0.00043162917303889273,
    3.366766427747367e-07,
    1.9684022349994908e-06,
    2.0200244590691145e-10,
    4.676246698024074e-10,
)


def test_clwrf_ssp245_interpolation_matches_wrf_default_real_replay(
    tmp_path: Path,
) -> None:
    table = _table(tmp_path)
    for when, wrf_oracle, original_host in (
        ("2025-03-01T00:00:00Z", _WRF_ORACLE_2025_03_01T00Z,
         _ORIGINAL_HOST_ONE_BASED_2025_03_01T00Z),
        ("2026-04-28T18:00:00+00:00", _WRF_ORACLE_2026_04_28T18Z,
         _ORIGINAL_HOST_ONE_BASED_2026_04_28T18Z),
    ):
        got = np.asarray(
            clwrf_ssp245_gases_for_time(when, table_path=table), dtype=np.float64
        )
        np.testing.assert_array_equal(got, np.asarray(wrf_oracle, dtype=np.float64))
        # The superseded one-based calendar must not come back.
        assert not np.any(got == np.asarray(original_host, dtype=np.float64)), when


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


def test_gas_metadata_is_dynamic_pytree_state_and_must_be_complete() -> None:
    """v0.25 S1: gas scalars are dynamic children.

    As aux they were `jax.jit` compile constants, so the coupler's clock-advanced
    SSP245 values minted a new SW/LW executable at every wrfout output time. As
    children the program depends on geometry only, while the values themselves
    still reach the computation."""

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
    # Supplying the gases adds exactly one leaf per gas (they are children now).
    assert len(jax.tree_util.tree_leaves(lw_gases)) == len(lw_leaves) + 5
    assert len(jax.tree_util.tree_leaves(sw_gases)) == len(sw_leaves) + 3
    # ...and their VALUES no longer live in the treedef, so two different gas
    # sets share one program structure.
    lw_other = lw_gases.replace(
        co2_vmr=4.3e-4,
        n2o_vmr=3.4e-7,
        ch4_vmr=2.0e-6,
        cfc11_vmr=2.1e-10,
        cfc12_vmr=4.8e-10,
    )
    sw_other = sw_gases.replace(co2_vmr=4.3e-4, n2o_vmr=3.4e-7, ch4_vmr=2.0e-6)
    assert jax.tree_util.tree_structure(lw_gases) == jax.tree_util.tree_structure(lw_other)
    assert jax.tree_util.tree_structure(sw_gases) == jax.tree_util.tree_structure(sw_other)
    # A flatten/unflatten round trip returns the same values through the leaves.
    for state in (lw_gases, sw_gases):
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert jax.tree_util.tree_unflatten(treedef, leaves).co2_vmr == 4.2e-4
    # The LW model top stays STATIC: it is shape metadata (WRF rrtmg_lwinit
    # NLAYERS), so it must keep specializing the program.
    top_a = lw_gases.replace(top_pressure_pa=5000.0)
    top_b = lw_gases.replace(top_pressure_pa=4000.0)
    assert jax.tree_util.tree_structure(top_a) != jax.tree_util.tree_structure(top_b)
    assert top_a.tree_flatten()[1] == (5000.0,)
    assert sw_gases.tree_flatten()[1] == ()

    for first, second in ((lw_gases, lw_other), (sw_gases, sw_other)):
        consume = jax.jit(lambda state: state.co2_vmr + state.ch4_vmr)
        assert float(consume(first)) != float(consume(second))
        assert consume._cache_size() == 1

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
    with pytest.raises((TypeError, ValueError)):
        sw_gases.replace(co2_vmr=object())


def _m9_columns(ncol: int = 3, nz: int = 12):
    """Small WRF-shaped columns for the real M9 JIT boundaries."""

    p_ifc = np.linspace(101000.0, 5000.0, nz + 1)[None, :].repeat(ncol, axis=0)
    p = 0.5 * (p_ifc[:, 1:] + p_ifc[:, :-1])
    t = np.linspace(295.0, 215.0, nz)[None, :].repeat(ncol, axis=0)
    t_ifc = np.linspace(297.0, 213.0, nz + 1)[None, :].repeat(ncol, axis=0)
    z = lambda v: jnp.asarray(np.full((ncol, nz), v))  # noqa: E731
    common = dict(
        T=jnp.asarray(t), p=jnp.asarray(p), qv=z(5.0e-3), qc=z(0.0), qi=z(0.0), qs=z(0.0),
        qg=z(0.0), cloud_fraction=z(0.0), dz=z(800.0), rho=z(0.8),
        pressure_interfaces=jnp.asarray(p_ifc), temperature_interfaces=jnp.asarray(t_ifc),
    )
    return common, ncol


def _m9_gases(scale: float):
    return dict(co2_vmr=4.2e-4 * scale, n2o_vmr=3.3e-7 * scale, ch4_vmr=1.9e-6 * scale)


def test_m9_flux_slices_do_not_respecialize_across_gas_values() -> None:
    """R1: lock the production SW/LW M9 JIT boundaries, not a toy lambda."""

    common, ncol = _m9_columns()
    lw_kw = dict(common, surface_temperature=jnp.full((ncol,), 295.0),
                 surface_emissivity=jnp.full((ncol,), 0.98), top_pressure_pa=5000.0)
    sw_kw = dict(common, surface_albedo=jnp.full((ncol,), 0.2), coszen=jnp.full((ncol,), 0.6))
    lw_a = RRTMGLWColumnState(**lw_kw, **_m9_gases(1.0), cfc11_vmr=2.0e-10, cfc12_vmr=4.7e-10)
    lw_b = RRTMGLWColumnState(**lw_kw, **_m9_gases(1.01), cfc11_vmr=2.1e-10, cfc12_vmr=4.8e-10)
    sw_a = RRTMGSWColumnState(**sw_kw, **_m9_gases(1.0))
    sw_b = RRTMGSWColumnState(**sw_kw, **_m9_gases(1.01))

    lw0 = solve_rrtmg_lw_m9_flux_slices._cache_size()
    out_a = solve_rrtmg_lw_m9_flux_slices(lw_a, debug=False, column_tile_cols=2)
    out_b = solve_rrtmg_lw_m9_flux_slices(lw_b, debug=False, column_tile_cols=2)
    assert solve_rrtmg_lw_m9_flux_slices._cache_size() == lw0 + 1
    assert float(jnp.max(jnp.abs(out_a.surface_down - out_b.surface_down))) > 0.0

    sw0 = solve_rrtmg_sw_m9_flux_slices._cache_size()
    solve_rrtmg_sw_m9_flux_slices(sw_a, debug=False, topography=None, column_tile_cols=2)
    solve_rrtmg_sw_m9_flux_slices(sw_b, debug=False, topography=None, column_tile_cols=2)
    assert solve_rrtmg_sw_m9_flux_slices._cache_size() == sw0 + 1

    # The shape-setting LW model top remains a positive specialization control.
    solve_rrtmg_lw_m9_flux_slices(lw_a.replace(top_pressure_pa=4000.0), debug=False,
                                  column_tile_cols=2)
    assert solve_rrtmg_lw_m9_flux_slices._cache_size() == lw0 + 2
