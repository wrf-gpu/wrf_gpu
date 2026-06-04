"""Unit tests for fail-fast namelist option support checks."""

from __future__ import annotations

import pytest

from gpuwrf.contracts.physics_registry import ACCEPTED_NAMELIST_OPTIONS
from gpuwrf.io.namelist_check import (
    SUPPORTED_OPTIONS,
    UnsupportedNamelistOption,
    validate_supported_namelist,
)
from gpuwrf.io.wrf_scheme_catalog import (
    WRF_SCHEME_CATALOG,
    is_recognized_wrf_option,
    wrf_scheme_name,
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
                    "mp_physics": [8, 5],
                    "cu_physics": [5, 0],
                },
            }
        )

    message = str(excinfo.value)
    # mp=5 (Ferrier) and cu=5 (Grell-3D) are both *recognized* WRF v4 schemes
    # that are not yet implemented -> the specific "NOT YET IMPLEMENTED" message.
    assert "physics.mp_physics domain 2=5" in message
    assert "Ferrier" in message
    assert "NOT YET IMPLEMENTED" in message
    assert "Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 16" in message
    assert "physics.cu_physics domain 1=5" in message
    assert "Grell 3D ensemble" in message
    assert "1=Kain-Fritsch" in message
    assert "Action:" in message


def test_registry_records_supported_active_suite() -> None:
    assert SUPPORTED_OPTIONS["mp_physics"].supported_values == frozenset({0, 1, 2, 3, 4, 6, 8, 10, 16})
    assert SUPPORTED_OPTIONS["bl_pbl_physics"].supported_values == frozenset({0, 1, 2, 5, 7, 8})
    assert SUPPORTED_OPTIONS["sf_sfclay_physics"].supported_values == frozenset({0, 1, 2, 5, 7})
    assert SUPPORTED_OPTIONS["sf_surface_physics"].supported_values == frozenset({0, 2, 4})
    assert SUPPORTED_OPTIONS["cu_physics"].supported_values == frozenset({0, 1, 2, 3, 6, 16})
    assert SUPPORTED_OPTIONS["ra_sw_physics"].supported_values == frozenset({0, 1, 4})
    assert SUPPORTED_OPTIONS["ra_lw_physics"].supported_values == frozenset({0, 1, 4})


def test_myj_janjic_pairing_is_enforced() -> None:
    validate_supported_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}})
    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [5]}})
    message = str(excinfo.value)
    assert "MYJ PBL and Janjic Eta surface layer are a mandatory WRF pair" in message
    assert "Select both option values as 2" in message


# --------------------------------------------------------------------------- #
# Three-outcome validation: (a) implemented, (b) recognized-not-implemented,  #
# (c) not-a-recognized-WRF-option. WRF-USER-FRIENDLY namelist compatibility.  #
# --------------------------------------------------------------------------- #


def _selection_for(excinfo: "pytest.ExceptionInfo[UnsupportedNamelistOption]", key: str):
    selections = [s for s in excinfo.value.selections if s.key == key]
    assert selections, f"expected a rejected selection for {key}"
    return selections[0]


def test_every_accepted_option_is_a_recognized_wrf_option() -> None:
    """The port's accepted matrix must be a subset of the WRF v4 catalog.

    Guards against an accepted scheme being mis-reported as 'not a recognized
    WRF option' (which would be a catalog transcription bug).
    """

    for key, accepted in ACCEPTED_NAMELIST_OPTIONS.items():
        catalog = WRF_SCHEME_CATALOG.get(key)
        assert catalog is not None, f"no WRF catalog for accepted key {key}"
        for code in accepted:
            assert is_recognized_wrf_option(key, code), (
                f"accepted {key}={code} is not in the WRF v4 catalog"
            )


def test_catalog_spot_checks_against_wrf_readme() -> None:
    """Spot-check faithful WRF v4 code->name mappings from README.namelist."""

    assert wrf_scheme_name("mp_physics", 8).name.startswith("Thompson")
    assert "Morrison" in wrf_scheme_name("mp_physics", 10).name
    assert "P3" in wrf_scheme_name("mp_physics", 50).name
    assert "Kain-Fritsch" in wrf_scheme_name("cu_physics", 1).name
    assert "Tiedtke" in wrf_scheme_name("cu_physics", 6).name
    assert "MYNN" in wrf_scheme_name("bl_pbl_physics", 5).name
    assert "QNSE" in wrf_scheme_name("bl_pbl_physics", 4).name
    assert "Noah-MP" in wrf_scheme_name("sf_surface_physics", 4).name
    assert "RUC" in wrf_scheme_name("sf_surface_physics", 3).name
    assert "RRTMG" in wrf_scheme_name("ra_lw_physics", 4).name
    assert "Dudhia" in wrf_scheme_name("ra_sw_physics", 1).name
    # Not WRF options:
    assert wrf_scheme_name("mp_physics", 99) is None
    assert wrf_scheme_name("cu_physics", 42) is None


def test_implemented_scheme_passes() -> None:
    """An implemented Thompson/KF/MYNN/Noah-MP/RRTMG suite is accepted."""

    validate_supported_namelist(
        {
            "physics": {
                "mp_physics": [8],
                "cu_physics": [1],
                "bl_pbl_physics": [5],
                "sf_sfclay_physics": [5],
                "sf_surface_physics": [4],
                "ra_lw_physics": [4],
                "ra_sw_physics": [4],
            }
        }
    )


@pytest.mark.parametrize(
    "key, value, scheme_substring",
    [
        ("mp_physics", 28, "Thompson"),  # aerosol-aware Thompson
        ("mp_physics", 50, "P3"),
        ("bl_pbl_physics", 4, "QNSE"),
        ("cu_physics", 5, "Grell 3D"),
        ("sf_surface_physics", 3, "RUC"),
        ("sf_surface_physics", 5, "CLM4"),
        ("sf_sfclay_physics", 3, "GFS"),
        ("ra_lw_physics", 5, "Goddard"),
        ("ra_sw_physics", 2, "Goddard"),
    ],
)
def test_recognized_but_unimplemented_scheme_names_the_status(
    key: str, value: int, scheme_substring: str
) -> None:
    """A valid WRF v4 scheme the port lacks must fail closed with a specific message."""

    cfg = {"physics": {key: [value]}}
    # bl=4 QNSE/sf=3 GFS would also trip MYJ pairing only if 2; safe here.
    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist(cfg)

    sel = _selection_for(excinfo, key)
    assert sel.outcome == "not_yet_implemented"
    assert sel.value == value
    assert scheme_substring in (sel.wrf_scheme or "")

    message = str(excinfo.value)
    assert f"{key}={value}" in message
    assert "NOT YET IMPLEMENTED" in message
    assert scheme_substring in message


@pytest.mark.parametrize(
    "key, value",
    [
        ("mp_physics", 99),
        ("mp_physics", 12),
        ("cu_physics", 42),
        ("bl_pbl_physics", 6),  # removed in WRF 4.5
        ("sf_surface_physics", 9),
        ("ra_lw_physics", 2),  # 2 is sw-only, not a valid lw option
    ],
)
def test_invalid_wrf_value_reports_not_recognized(key: str, value: int) -> None:
    """A value that is not a WRF v4 option fails closed with the 'not recognized' message."""

    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist({"physics": {key: [value]}})

    sel = _selection_for(excinfo, key)
    assert sel.outcome == "invalid_wrf_option"
    assert sel.wrf_scheme is None

    message = str(excinfo.value)
    assert f"{key}={value} is not a recognized WRF v4" in message


def test_reference_failclosed_schemes_keep_specific_messages() -> None:
    """GF/New-Tiedtke/RRTM-LW/Dudhia/MYJ/Janjic remain accepted with their notes.

    These options ARE in the port's accepted/reference matrix, so they must not
    be flagged at the namelist layer (their fail-closed-in-scan behavior is a
    downstream runtime concern, kept accurate in the SUPPORTED_OPTIONS text).
    """

    # cu=3 Grell-Freitas, cu=16 New Tiedtke: accepted (reference).
    validate_supported_namelist({"physics": {"cu_physics": [3]}})
    validate_supported_namelist({"physics": {"cu_physics": [16]}})
    # ra_lw=1 classic RRTM, ra_sw=1 Dudhia: accepted (isolated savepoint).
    validate_supported_namelist({"physics": {"ra_lw_physics": [1], "ra_sw_physics": [1]}})
    # MYJ(2)+Janjic(2) reference pair: accepted.
    validate_supported_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}})

    # And the supported-option notes still carry the reference/fail-closed text.
    assert "Grell-Freitas" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "New Tiedtke" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "MYJ" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "Janjic Eta" in SUPPORTED_OPTIONS["sf_sfclay_physics"].implemented
    assert "RRTM longwave" in SUPPORTED_OPTIONS["ra_lw_physics"].implemented
    assert "Dudhia shortwave" in SUPPORTED_OPTIONS["ra_sw_physics"].implemented


def test_recognized_but_unimplemented_dynamics_option() -> None:
    """diff_opt=1 / km_opt=4 are real-data WRF defaults the port does not yet wire."""

    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist({"dynamics": {"diff_opt": 1, "km_opt": 4}})
    message = str(excinfo.value)
    assert "diff_opt=1" in message
    assert "km_opt=4" in message
    assert "NOT YET IMPLEMENTED" in message
    # And km_opt=99 is not a recognized WRF option.
    with pytest.raises(UnsupportedNamelistOption) as excinfo2:
        validate_supported_namelist({"dynamics": {"km_opt": 99}})
    assert "km_opt=99 is not a recognized WRF v4" in str(excinfo2.value)


def test_fortran_repeat_count_syntax_is_expanded() -> None:
    """WRF ``N*value`` repeat syntax expands to N per-domain values."""

    # 3*8 -> [8, 8, 8] all implemented -> passes.
    validate_supported_namelist("&physics\n mp_physics = 3*8,\n/")
    # 2*28 -> two domains of a recognized-but-unimplemented scheme.
    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist("&physics\n mp_physics = 2*28,\n/")
    sels = [s for s in excinfo.value.selections if s.key == "mp_physics"]
    assert len(sels) == 2
    assert all(s.outcome == "not_yet_implemented" for s in sels)
    assert all(s.value == 28 for s in sels)


def test_real_wrf_namelist_input_is_consumable() -> None:
    """A standard Fortran namelist.input (all groups) parses and validates.

    Uses the pristine WRF em_real oracle namelist when present (an implemented
    Thompson/KF/MYNN/Noah-MP/RRTMG suite); otherwise an inline equivalent.
    """

    from pathlib import Path

    oracle = Path(
        "/home/enric/src/wrf_pristine/WRF/test/em_real/oracle_run/namelist.input"
    )
    if oracle.is_file():
        from gpuwrf.io.namelist_check import _parse_wrf_namelist

        parsed = _parse_wrf_namelist(oracle.read_text())
        # All standard WRF groups present.
        for group in ("time_control", "domains", "physics", "dynamics", "bdy_control"):
            assert group in parsed, f"missing &{group} in parsed namelist"
        # The oracle's diff_opt=1/km_opt=4 real-data defaults fail closed with the
        # specific not-yet-implemented message (honest: those are not yet wired).
        with pytest.raises(UnsupportedNamelistOption) as excinfo:
            validate_supported_namelist(oracle)
        assert "NOT YET IMPLEMENTED" in str(excinfo.value)
    else:  # pragma: no cover - pristine tree not on this host
        text = (
            "&time_control\n run_hours = 24,\n/\n"
            "&domains\n max_dom = 1,\n/\n"
            "&physics\n mp_physics = 8,\n cu_physics = 1,\n"
            " bl_pbl_physics = 5,\n sf_sfclay_physics = 5,\n"
            " sf_surface_physics = 4,\n ra_lw_physics = 4,\n ra_sw_physics = 4,\n/\n"
            "&dynamics\n diff_opt = 2,\n km_opt = 1,\n/\n"
            "&bdy_control\n spec_bdy_width = 5,\n/\n"
        )
        validate_supported_namelist(text)
