"""Unit tests for fail-fast namelist option support checks."""

from __future__ import annotations

import pytest

from gpuwrf.io.namelist_check import (
    SUPPORTED_OPTIONS,
    UnsupportedNamelistOption,
    validate_supported_namelist,
)


def test_supported_physics_and_dynamics_config_passes() -> None:
    config = {
        "physics": {
            "mp_physics": [8, 8],
            "cu_physics": [0, 0],
            "bl_pbl_physics": [5, 5],
            "sf_sfclay_physics": [5, 5],
            "sf_surface_physics": [4, 4],
            "ra_sw_physics": [4, 4],
            "ra_lw_physics": [4, 4],
            "sf_urban_physics": [0, 0],
        },
        "dynamics": {
            "rk_order": 3,
            "diff_6th_opt": 2,
            "diff_opt": 2,
            "km_opt": 1,
            "w_damping": 1,
            "damp_opt": 3,
        },
    }

    validate_supported_namelist(config)


def test_unsupported_selected_option_raises_actionable_error() -> None:
    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist(
            {
                "physics": {
                    "mp_physics": [8, 2],
                    "cu_physics": [2, 0],
                },
            }
        )

    message = str(excinfo.value)
    assert "physics.mp_physics domain 2 selected 2" in message
    assert "supported values: 0, 1, 6, 8, 10, 16" in message
    assert "physics.cu_physics domain 1 selected 2" in message
    assert "1=Kain-Fritsch" in message
    assert "Action:" in message


def test_registry_records_supported_active_suite() -> None:
    assert SUPPORTED_OPTIONS["mp_physics"].supported_values == frozenset({0, 1, 6, 8, 10, 16})
    assert SUPPORTED_OPTIONS["bl_pbl_physics"].supported_values == frozenset({0, 1, 5, 7})
    assert SUPPORTED_OPTIONS["sf_sfclay_physics"].supported_values == frozenset({0, 1, 5, 7})
    assert SUPPORTED_OPTIONS["sf_surface_physics"].supported_values == frozenset({0, 2, 4})
    assert SUPPORTED_OPTIONS["cu_physics"].supported_values == frozenset({0, 1, 3, 6, 16})
    assert SUPPORTED_OPTIONS["ra_sw_physics"].supported_values == frozenset({0, 4})
    assert SUPPORTED_OPTIONS["ra_lw_physics"].supported_values == frozenset({0, 4})
