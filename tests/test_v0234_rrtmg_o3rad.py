"""WRF-source regression and compatibility gates for ``o3input=2`` O3RAD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.rrtmg_lw import (
    RRTMGLWColumnState,
    _flatten_lw_state,
    _lw_extended_pressure_profiles,
    _lw_o3_profile_vmr,
    _lw_o3_vmr_for_state,
    _pad_lw_state,
    _slice_lw_state,
)
from gpuwrf.physics.rrtmg_sw import (
    RRTMGSWColumnState,
    _flatten_sw_state,
    _pad_sw_state,
    _pressure_interfaces as _sw_pressure_interfaces,
    _slice_sw_state,
    _sw_o3_vmr_for_state,
    _wrf_o3_vmr,
)
from gpuwrf.physics.wrf_cam_ozone import (
    OZONE_ASSET,
    OZONE_ASSET_SHA256,
    _load_wrf_cam_ozone_asset,
    wrf_cam_ozone_profile,
    wrf_ozn_time_indices_factors,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = ROOT / "data" / "fixtures" / "wrf-cam-ozone-v1.json"
EXPECTED_SOURCE_HASHES = {
    "latitude": "761597f3454b99e3dbf15d2621b1ede88269948e96e47f4695b79fabd3cc6348",
    "pressure": "df26a938273b22d2d04ec52b7297827a316cac0061fab020519241cdc0e61d48",
    "mixing_ratio": "2a13ee25809672e0419e40062c1fc753580adbf604bfc67a9166ef77c815dbc0",
}
EXPECTED_PAYLOAD_HASHES = {
    "latitude_deg": "aa3a9daacbf5d545e033aa23942cb8689d931858e69abac4a6a8d3733c6369cf",
    "pressure_pa": "f33be37c277974f486dbb4359363025cf37b8e549aa6775fbd80dc0dc297ac38",
    "ozone_vmr": "82e16042e17db543b97f77c11b08a5fa2e6ef6bc068018f58a6fb82df1b9c01d",
}
DATE_OZ = np.asarray(
    [16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _wrf_time_reference(julian_day_1based: int, utc_minute: float):
    """Independent scalar replay of ``radiation_driver::ozn_time_int``."""

    day = np.float32(julian_day_1based) + np.float32(
        np.float32(utc_minute) / np.float32(1440.0)
    )
    ijul = int(day)
    fraction = np.float32(day - np.float32(ijul))
    ijul %= 365
    if ijul == 0:
        ijul = 365
    day = np.float32(fraction + np.float32(ijul))
    month_plus = 0
    for month, date_oz in enumerate(DATE_OZ):
        if date_oz > day:
            month_plus = month
            break
    month_minus = month_plus - 1 if month_plus > 0 else 11
    cday_plus = DATE_OZ[month_plus]
    cday_minus = DATE_OZ[month_minus]
    if month_plus == 0:
        delta = np.float32(cday_plus + np.float32(365.0) - cday_minus)
        if day > cday_plus:
            factor1 = np.float32(
                np.float32(cday_plus + np.float32(365.0) - day) / delta
            )
            factor2 = np.float32(np.float32(day - cday_minus) / delta)
        else:
            factor1 = np.float32(np.float32(cday_plus - day) / delta)
            factor2 = np.float32(
                np.float32(day + np.float32(365.0) - cday_minus) / delta
            )
    else:
        delta = np.float32(cday_plus - cday_minus)
        factor1 = np.float32(np.float32(cday_plus - day) / delta)
        factor2 = np.float32(np.float32(day - cday_minus) / delta)
    return month_minus, month_plus, factor1, factor2, day


def _wrf_profile_reference(
    latitude_deg: np.ndarray,
    pressure_pa: np.ndarray,
    *,
    julian_day_1based: int,
    utc_minute: float,
) -> np.ndarray:
    """Independent loop replay of ``oznini`` + time and pressure interpolation."""

    latitude_table, pressure_table, ozone_table = _load_wrf_cam_ozone_asset()
    month_minus, month_plus, factor1, factor2, _ = _wrf_time_reference(
        julian_day_1based, utc_minute
    )
    output = np.empty_like(pressure_pa, dtype=np.float32)
    for column in np.ndindex(latitude_deg.shape):
        latitude = np.float32(latitude_deg[column])
        latitude_index = 0
        if latitude >= latitude_table[-1]:
            latitude_index = latitude_table.size - 2
        elif latitude > latitude_table[0]:
            while (
                latitude > latitude_table[latitude_index + 1]
                and latitude_index < latitude_table.size - 1
            ):
                latitude_index += 1
        monthly = np.empty((12, pressure_table.size), dtype=np.float32)
        for month in range(12):
            for level in range(pressure_table.size):
                slope = np.float32(
                    np.float32(
                        ozone_table[month, latitude_index + 1, level]
                        - ozone_table[month, latitude_index, level]
                    )
                    / np.float32(
                        latitude_table[latitude_index + 1]
                        - latitude_table[latitude_index]
                    )
                )
                monthly[month, level] = np.float32(
                    ozone_table[month, latitude_index, level]
                    + np.float32(
                        slope * np.float32(latitude - latitude_table[latitude_index])
                    )
                )
        timed = np.empty(pressure_table.size, dtype=np.float32)
        for level in range(pressure_table.size):
            timed[level] = np.float32(
                np.float32(monthly[month_minus, level] * factor1)
                + np.float32(monthly[month_plus, level] * factor2)
            )
        for level in range(pressure_pa.shape[-1]):
            pressure = np.float32(pressure_pa[column + (level,)])
            if pressure < pressure_table[0]:
                output[column + (level,)] = np.float32(
                    np.float32(timed[0] * pressure) / pressure_table[0]
                )
            elif pressure > pressure_table[-1]:
                output[column + (level,)] = timed[-1]
            elif pressure == pressure_table[0]:
                output[column + (level,)] = timed[0]
            else:
                upper = 0
                while not (
                    pressure_table[upper] < pressure
                    and pressure <= pressure_table[upper + 1]
                ):
                    upper += 1
                    if upper == pressure_table.size - 1:
                        # Exact top-table endpoint follows the source's carried
                        # kupper fallback and the final adjacent interval.
                        upper -= 1
                        break
                dpu = np.float32(pressure - pressure_table[upper])
                dpl = np.float32(pressure_table[upper + 1] - pressure)
                numerator = np.float32(
                    np.float32(timed[upper] * dpl)
                    + np.float32(timed[upper + 1] * dpu)
                )
                output[column + (level,)] = np.float32(
                    numerator / np.float32(dpl + dpu)
                )
    return output


def test_cam_ozone_asset_is_hash_pinned_to_the_wrf_run_tables() -> None:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    assert _sha256(OZONE_ASSET) == OZONE_ASSET_SHA256
    assert manifest["asset"]["sha256"] == OZONE_ASSET_SHA256
    assert {
        name: entry["sha256"]
        for name, entry in manifest["wrf_sources"].items()
    } == EXPECTED_SOURCE_HASHES
    latitude, pressure, ozone = _load_wrf_cam_ozone_asset()
    arrays = {"latitude_deg": latitude, "pressure_pa": pressure, "ozone_vmr": ozone}
    assert {name: _payload_sha(value) for name, value in arrays.items()} == EXPECTED_PAYLOAD_HASHES
    assert latitude.dtype == pressure.dtype == ozone.dtype == np.dtype("float32")
    assert latitude.shape == (64,)
    assert pressure.shape == (59,)
    assert ozone.shape == (12, 64, 59)


@pytest.mark.parametrize(
    ("julian", "minute", "expected"),
    (
        (1, 0.0, (11, 0, np.float32(15.0 / 31.0), np.float32(16.0 / 31.0), np.float32(1.0))),
        (60, 0.0, (1, 2, np.float32(0.5), np.float32(0.5), np.float32(60.0))),
        (118, 1080.0, (3, 4, np.float32(17.25 / 31.0), np.float32(13.75 / 31.0), np.float32(118.75))),
        (365, 0.0, (11, 0, np.float32(16.0 / 31.0), np.float32(15.0 / 31.0), np.float32(365.0))),
    ),
)
def test_ozn_time_int_matches_wrf_365_day_run_date_handling(
    julian: int,
    minute: float,
    expected,
) -> None:
    actual = tuple(
        np.asarray(value).item()
        for value in wrf_ozn_time_indices_factors(julian, minute)
    )
    reference = _wrf_time_reference(julian, minute)
    assert actual == tuple(np.asarray(value).item() for value in reference)
    assert actual == tuple(np.asarray(value).item() for value in expected)


def test_cam_profile_matches_independent_wrf_float32_loop_replay() -> None:
    latitude = np.asarray([[-90.0, 28.3], [31.1, 90.0]], dtype=np.float32)
    pressure = np.broadcast_to(
        np.asarray([100.0, 7000.0, 50000.0, 110000.0], dtype=np.float32),
        latitude.shape + (4,),
    ).copy()
    reference = _wrf_profile_reference(
        latitude,
        pressure,
        julian_day_1based=118,
        utc_minute=1080.0,
    )
    actual = np.asarray(
        wrf_cam_ozone_profile(
            latitude,
            pressure,
            julian_day_1based=118,
            utc_minute=1080.0,
        )
    )
    # Each JAX vector expression is the same WRF REAL expression as the scalar
    # replay. A single float32 ULP permits backend multiply-add contraction;
    # it is a representation bound, not an empirical physics tolerance.
    np.testing.assert_array_max_ulp(actual, reference, maxulp=1)


def _column_states(ozone_vmr=None):
    temperature = jnp.asarray(
        [[[291.0, 267.0, 238.0], [292.0, 268.0, 239.0]]], dtype=jnp.float64
    )
    pressure = jnp.asarray(
        [[[90000.0, 51000.0, 17000.0], [91000.0, 52000.0, 18000.0]]],
        dtype=jnp.float64,
    )
    zero = jnp.zeros_like(temperature)
    lw = RRTMGLWColumnState(
        temperature,
        pressure,
        jnp.full_like(temperature, 1.0e-3),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([[292.0, 293.0]], dtype=jnp.float64),
        jnp.asarray([[0.97, 0.98]], dtype=jnp.float64),
        jnp.full_like(temperature, 500.0),
        jnp.full_like(temperature, 0.8),
        5000.0,
        ozone_vmr=ozone_vmr,
    )
    sw = RRTMGSWColumnState(
        temperature,
        pressure,
        jnp.full_like(temperature, 1.0e-3),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([[0.12, 0.18]], dtype=jnp.float64),
        jnp.asarray([[0.35, 0.42]], dtype=jnp.float64),
        jnp.full_like(temperature, 500.0),
        jnp.full_like(temperature, 0.8),
        ozone_vmr=ozone_vmr,
    )
    return lw, sw


def test_optional_o3rad_leaves_reproduce_wrf_shifted_top_layers_and_tiling() -> None:
    ozone = jnp.asarray(
        [[[2.0e-8, 3.0e-7, 1.0e-6], [2.1e-8, 3.1e-7, 1.1e-6]]],
        dtype=jnp.float32,
    )
    lw, sw = _column_states(ozone)
    lw_model_interfaces, lw_interfaces, _ = _lw_extended_pressure_profiles(
        lw.p, lw.top_pressure_pa
    )
    lw_climatology = _lw_o3_profile_vmr(lw_interfaces)
    lw_actual = _lw_o3_vmr_for_state(lw, lw_interfaces, lw.p.shape[-1])
    lw_shift = ozone.astype(lw_interfaces.dtype)[..., -1] - lw_climatology[..., 2]
    lw_expected_top = lw_climatology[..., 3:] + lw_shift[..., None]
    lw_expected_top = jnp.where(
        lw_expected_top <= 0.0, lw_climatology[..., 3:], lw_expected_top
    )
    np.testing.assert_array_equal(np.asarray(lw_actual[..., :3]), np.asarray(ozone))
    np.testing.assert_array_equal(np.asarray(lw_actual[..., 3:]), np.asarray(lw_expected_top))
    assert lw_model_interfaces.shape[-1] == 4

    sw_interfaces = _sw_pressure_interfaces(sw.p)
    sw_climatology = _wrf_o3_vmr(sw_interfaces)
    sw_actual = _sw_o3_vmr_for_state(sw, sw_interfaces)
    sw_shift = ozone.astype(sw_interfaces.dtype)[..., -1] - sw_climatology[..., 2]
    sw_expected_top = sw_climatology[..., 3:] + sw_shift[..., None]
    sw_expected_top = jnp.where(
        sw_expected_top <= 0.0, sw_climatology[..., 3:], sw_expected_top
    )
    np.testing.assert_array_equal(np.asarray(sw_actual[..., :3]), np.asarray(ozone))
    np.testing.assert_array_equal(np.asarray(sw_actual[..., 3:]), np.asarray(sw_expected_top))

    assert len(jax.tree_util.tree_leaves(lw)) == 13
    assert len(jax.tree_util.tree_leaves(sw)) == 14
    lw_sliced = _slice_lw_state(
        _pad_lw_state(_flatten_lw_state(lw, (1, 2), 2), 2, 4),
        jnp.asarray(0, dtype=jnp.int32),
        2,
        4,
    )
    sw_sliced = _slice_sw_state(
        _pad_sw_state(_flatten_sw_state(sw, (1, 2), 2), 2, 4),
        jnp.asarray(0, dtype=jnp.int32),
        2,
        4,
    )
    np.testing.assert_array_equal(np.asarray(lw_sliced.ozone_vmr), np.reshape(np.asarray(ozone), (2, 3)))
    np.testing.assert_array_equal(np.asarray(sw_sliced.ozone_vmr), np.reshape(np.asarray(ozone), (2, 3)))


def test_omitted_o3rad_keeps_historical_path_and_shape_is_fail_closed() -> None:
    lw, sw = _column_states()
    lw_interfaces = _lw_extended_pressure_profiles(lw.p, lw.top_pressure_pa)[1]
    sw_interfaces = _sw_pressure_interfaces(sw.p)
    assert _lw_o3_vmr_for_state(lw, lw_interfaces, lw.p.shape[-1]) is None
    assert _sw_o3_vmr_for_state(sw, sw_interfaces) is None
    assert len(jax.tree_util.tree_leaves(lw)) == 12
    assert len(jax.tree_util.tree_leaves(sw)) == 13
    # The ozone argument is appended after every pre-existing optional
    # parameter, so historical positional gas calls retain their meaning.
    assert RRTMGLWColumnState(
        *(
            getattr(lw, name)
            for name in RRTMGLWColumnState.__slots__[:12]
        ),
        lw.top_pressure_pa,
        None,
        None,
        4.2e-4,
        3.3e-7,
        1.9e-6,
        2.0e-10,
        4.7e-10,
    ).ozone_vmr is None
    assert RRTMGSWColumnState(
        *(
            getattr(sw, name)
            for name in RRTMGSWColumnState.__slots__[:12]
        ),
        sw.solar_source_scale,
        None,
        None,
        4.2e-4,
        3.3e-7,
        1.9e-6,
    ).ozone_vmr is None
    with pytest.raises(ValueError, match="column-layer shape"):
        lw.replace(ozone_vmr=jnp.ones(lw.p.shape[:-1] + (2,)))
    with pytest.raises(ValueError, match="column-layer shape"):
        sw.replace(ozone_vmr=jnp.ones(sw.p.shape[:-1] + (2,)))
