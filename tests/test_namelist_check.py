"""Unit tests for fail-fast namelist option support checks."""

from __future__ import annotations

import pytest

from gpuwrf.contracts.physics_registry import ACCEPTED_NAMELIST_OPTIONS
from gpuwrf.io.namelist_check import (
    SUPPORTED_OPTIONS,
    NotOperationallyWiredError,
    UnsupportedNamelistOption,
    UnsupportedSchemeError,
    validate_namelist,
    validate_operational_namelist,
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
            "sf_lake_physics": [0, 0],
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
                    # cu=7 (Zhang-McFarlane/CAM5) is a recognized WRF v4 cumulus
                    # scheme the port does not accept (cu=5 Grell-3D / 14 KSAS are
                    # now reference-only, accepted-at-namelist; cu=7 stays
                    # RECOGNIZED_FAIL_CLOSED).
                    "cu_physics": [7, 0],
                },
            }
        )

    message = str(excinfo.value)
    # mp=5 (Ferrier) and cu=7 (Zhang-McFarlane) are both *recognized* WRF v4
    # schemes that are not yet implemented -> the "NOT YET IMPLEMENTED" message.
    assert "physics.mp_physics domain 2=5" in message
    assert "Ferrier" in message
    assert "NOT YET IMPLEMENTED" in message
    assert "Supported mp_physics values: 0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16" in message
    assert "physics.cu_physics domain 1=7" in message
    assert "Zhang-McFarlane" in message
    assert "1=Kain-Fritsch" in message
    assert "Action:" in message


def test_registry_records_supported_active_suite() -> None:
    # v0.16 adds mp=28 (aerosol-aware Thompson); v0.18 harvests mp=13 (SBU-YLin),
    # mp=24 (WSM7) + mp=26 (WDM7) hail, and mp=97 (Goddard GCE).
    assert SUPPORTED_OPTIONS["mp_physics"].supported_values == frozenset({0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 24, 26, 28, 97})
    # bl=3 GFS, bl=11 Shin-Hong, and bl=12 GBM are operational
    # (savepoint/reference-parity proven, scan-wired). bl=4/10/16/17 are v0.18
    # reference-only with real pristine-WRF module oracles.
    assert SUPPORTED_OPTIONS["bl_pbl_physics"].supported_values == frozenset({0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 99})
    assert SUPPORTED_OPTIONS["sf_sfclay_physics"].supported_values == frozenset({0, 1, 2, 3, 5, 7, 91})
    assert SUPPORTED_OPTIONS["sf_surface_physics"].supported_values == frozenset({0, 1, 2, 3, 4, 7, 8})
    assert SUPPORTED_OPTIONS["cu_physics"].supported_values == frozenset({0, 1, 2, 3, 4, 5, 6, 14, 16, 93, 94, 95, 96, 99})
    # v0.18 RA tail adds ra_lw/sw=3 (CAM) and 7 (FLG/UCLA) as real-WRF exact-driver
    # reference-only oracles alongside ra_sw=5 (new Goddard) and ra_lw/sw=99
    # (GFDL-Eta). ra_lw/sw=14 (RRTMG-K) and 24 (fast RRTMG) are NOT accepted --
    # they are compiled-out of standard WRF (see test_v018_ra_tail_oracle.py).
    assert SUPPORTED_OPTIONS["ra_sw_physics"].supported_values == frozenset({0, 1, 2, 3, 4, 5, 7, 99})
    # ra_lw=5 (GSFC/Goddard NUWRF LW) is v0.13 Tier-3 reference-only: namelist-
    # accepted (selectable for a single-column reference comparison) but
    # fail-closed in the operational scan (proofs/v013/t3_gsfc_lw_oracle.py).
    # v0.17/v0.18 also accept ra_lw/sw=3/7/99 and ra_sw=5 as reference-only selections.
    # ra_lw=31 (Held-Suarez idealized radiation) is a v0.18-harvested operationally
    # scan-wired no-kernel-change endpoint (savepoint-parity-proven against the
    # unmodified phys/module_ra_hs.F at fp64).
    assert SUPPORTED_OPTIONS["ra_lw_physics"].supported_values == frozenset({0, 1, 3, 4, 5, 7, 31, 99})
    assert SUPPORTED_OPTIONS["sf_urban_physics"].supported_values == frozenset({0})
    assert SUPPORTED_OPTIONS["sf_lake_physics"].supported_values == frozenset({0})


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
    assert "BEP" in wrf_scheme_name("sf_urban_physics", 2).name
    assert "lake model" in wrf_scheme_name("sf_lake_physics", 1).name
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
                "sf_urban_physics": [0],
                "sf_lake_physics": [0],
            }
        }
    )


@pytest.mark.parametrize(
    "key, value, scheme_substring",
    [
        # mp=28 (aerosol-aware Thompson) became IMPLEMENTED in v0.16; mp=29
        # (RCON, the liquid-phase-modified Thompson-aero variant) remains the
        # recognized-but-unimplemented Thompson-family example.
        ("mp_physics", 29, "Thompson"),
        ("mp_physics", 50, "P3"),
        # bl=4/10/16/17 are now reference-only with real pristine-WRF module
        # oracles; bl=9 CAM-UW is operationally scan-wired in v0.22.
        ("cu_physics", 7, "Zhang-McFarlane"),
        # sf_surface=3 (RUC) + 8 (SSiB) are now v0.17 Tier-3 REFERENCE-ONLY
        # (namelist-accepted for a single-column reference comparison, fail-closed
        # in the operational scan), so they are no longer "recognized-but-
        # unimplemented" fail-at-namelist examples -- see the reference-only test in
        # tests/test_scheme_catalog_fail_closed.py. sf_surface=5 (CLM4) remains
        # recognized-but-unimplemented.
        ("sf_surface_physics", 5, "CLM4"),
        # sf_sfclay=3 (GFS) + 91 (old-MM5) are now v0.13 Tier-3 implemented; the
        # remaining unimplemented surface-layer option is QNSE (4).
        ("sf_sfclay_physics", 4, "QNSE"),
        # ra_lw=5 (GSFC/Goddard NUWRF LW) is v0.13 Tier-3 REFERENCE-ONLY and ra_lw/sw=3
        # (CAM), 7 (FLG/UCLA), 99 (GFDL-Eta) are v0.18 RA-tail REFERENCE-ONLY (real-WRF
        # exact-driver oracles staged in proofs/v018/savepoints/ra_tail_wrf;
        # namelist-accepted for reference comparison, fail-closed in the operational
        # scan) -- so none of them is a "recognized-but-unimplemented" fail-at-namelist
        # example any more. See the reference-only tests in
        # tests/test_scheme_catalog_fail_closed.py and tests/test_v018_ra_tail_oracle.py.
        # ra_lw/sw=14 (RRTMG-K) and 24 (fast RRTMG) are class-(c) compiled-out of
        # standard WRF -- covered as fail-closed in tests/test_v018_ra_tail_oracle.py.
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

    # cu=3 Grell-Freitas is operational; cu=5/16/93/99 are accepted (reference).
    validate_supported_namelist({"physics": {"cu_physics": [3]}})
    validate_supported_namelist({"physics": {"cu_physics": [4, 5, 16, 93, 94, 95, 96, 99]}})
    # ra_lw=1 classic RRTM, ra_sw=1 Dudhia: accepted (isolated savepoint).
    validate_supported_namelist({"physics": {"ra_lw_physics": [1], "ra_sw_physics": [1]}})
    # MYJ(2)+Janjic(2) reference pair: accepted.
    validate_supported_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}})
    # v0.18 PBL reference endpoints: accepted for single-column module-oracle work.
    validate_supported_namelist({"physics": {"bl_pbl_physics": [4, 10, 16, 17]}})

    # And the supported-option notes still carry the reference/fail-closed text.
    assert "Grell-Freitas" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "SAS family" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "New Tiedtke" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "Grell-Devenyi" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "previous Kain-Fritsch" in SUPPORTED_OPTIONS["cu_physics"].implemented
    assert "MYJ" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "QNSE" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "TEMF" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "EEPS" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "KEPS" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "CAM-UW" in SUPPORTED_OPTIONS["bl_pbl_physics"].implemented
    assert "Janjic Eta" in SUPPORTED_OPTIONS["sf_sfclay_physics"].implemented
    assert "RRTM longwave" in SUPPORTED_OPTIONS["ra_lw_physics"].implemented
    assert "Dudhia shortwave" in SUPPORTED_OPTIONS["ra_sw_physics"].implemented


def test_recognized_but_unimplemented_dynamics_option() -> None:
    """Wired diffusion/km_opt paths validate; unrecognized km_opt still fails."""

    # diff_opt=1/km_opt=4 is the v0.9.0 2-D Smagorinsky path (dynamics/explicit_diffusion.py,
    # parity proofs/v090/diffopt1_smagorinsky_parity.json) -- accepted, no raise.
    validate_supported_namelist({"dynamics": {"diff_opt": 1, "km_opt": 4}})
    validate_supported_namelist({"dynamics": {"diff_opt": 2, "km_opt": 2}})
    validate_supported_namelist({"dynamics": {"diff_opt": 2, "km_opt": 3}})
    validate_supported_namelist({"dynamics": {"diff_opt": 2, "km_opt": 5}})
    # km_opt=99 is not a recognized WRF option.
    with pytest.raises(UnsupportedNamelistOption) as excinfo2:
        validate_supported_namelist({"dynamics": {"km_opt": 99}})
    assert "km_opt=99 is not a recognized WRF v4" in str(excinfo2.value)


def test_fortran_repeat_count_syntax_is_expanded() -> None:
    """WRF ``N*value`` repeat syntax expands to N per-domain values."""

    # 3*8 -> [8, 8, 8] all implemented -> passes.
    validate_supported_namelist("&physics\n mp_physics = 3*8,\n/")
    # 2*29 -> two domains of a recognized-but-unimplemented scheme (RCON;
    # mp=28 aerosol-aware Thompson became implemented in v0.16).
    with pytest.raises(UnsupportedNamelistOption) as excinfo:
        validate_supported_namelist("&physics\n mp_physics = 2*29,\n/")
    sels = [s for s in excinfo.value.selections if s.key == "mp_physics"]
    assert len(sels) == 2
    assert all(s.outcome == "not_yet_implemented" for s in sels)
    assert all(s.value == 29 for s in sels)


# --------------------------------------------------------------------------- #
# Operational-strict validation (validate_operational_namelist).              #
# The OPERATIONAL run path must additionally REJECT reference-only schemes     #
# (oracle-backed but not wired into the operational GPU scan) so that          #
# ``gpuwrf run`` can never silently substitute a different scheme.             #
# --------------------------------------------------------------------------- #


def test_operational_validator_passes_implemented_suite() -> None:
    """An operationally-wired Thompson/KF/MYNN/Noah-MP/RRTMG suite passes."""

    validate_operational_namelist(
        {
            "physics": {
                "mp_physics": [8],
                "cu_physics": [1],
                "bl_pbl_physics": [5],
                "sf_sfclay_physics": [5],
                "sf_surface_physics": [4],
                "ra_lw_physics": [4],
                "ra_sw_physics": [4],
            },
            "dynamics": {"rk_order": 3, "diff_opt": 1, "km_opt": 4},
        }
    )


@pytest.mark.parametrize(
    "key, value, scheme_substring, alt_substring",
    [
        # ra_sw_physics=1 (Dudhia) and ra_lw_physics=1 (classic RRTM) are NOW
        # operationally scan-wired (see test_operational_validator_accepts_wired_*
        # below); they are no longer reference-only rejections.
        ("cu_physics", 16, "New Tiedtke", "cu_physics=6"),
        ("cu_physics", 4, "Scale-aware GFS SAS", "cu_physics=1/2/3/6"),
        ("cu_physics", 93, "Grell-Devenyi", "cu_physics=3"),
        ("cu_physics", 99, "previous Kain-Fritsch", "cu_physics=1"),
        ("bl_pbl_physics", 4, "QNSE", "bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99"),
        ("bl_pbl_physics", 10, "TEMF", "bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99"),
        ("bl_pbl_physics", 16, "epsilon", "bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99"),
        ("bl_pbl_physics", 17, "TPE", "bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99"),
    ],
)
def test_operational_validator_rejects_reference_only_scheme(
    key: str, value: int, scheme_substring: str, alt_substring: str
) -> None:
    """Reference-only schemes (cumulus + v0.17 radiation longtail) fail closed on
    the operational run path with the supported alternative."""

    with pytest.raises(NotOperationallyWiredError) as excinfo:
        validate_operational_namelist({"physics": {key: [value]}})

    sel = _selection_for(excinfo, key)
    assert sel.outcome == "reference_only_not_operational"
    assert sel.value == value
    assert scheme_substring in (sel.wrf_scheme or "")

    message = str(excinfo.value)
    assert f"{key}={value}" in message
    assert "SILENTLY" in message
    assert "NOT operationally wired" in message
    assert alt_substring in message
    # The rejected reference-only code must NOT be advertised as operational.
    assert f"Operationally-wired {key} values:" in message


def test_operational_validator_accepts_wired_dudhia_sw() -> None:
    """ra_sw_physics=1 (Dudhia) is now operationally scan-wired, so the operational
    validator ACCEPTS it (paired with the operational RRTMG longwave). This is the
    contract flip from REFERENCE_ONLY -> IMPLEMENTED."""

    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

    assert classify_scheme("ra_sw_physics", 1).status is SupportStatus.IMPLEMENTED
    # Accepted on its own and alongside the operational RRTMG longwave.
    validate_operational_namelist({"physics": {"ra_sw_physics": [1]}})
    validate_operational_namelist(
        {"physics": {"ra_sw_physics": [1], "ra_lw_physics": [4]}}
    )


@pytest.mark.parametrize(
    "key, value, scheme_substring",
    [
        ("ra_sw_physics", 3, "CAM"),
        ("ra_sw_physics", 5, "Goddard"),
        ("ra_sw_physics", 7, "FLG"),
        ("ra_sw_physics", 99, "GFDL"),
        ("ra_lw_physics", 3, "CAM"),
        ("ra_lw_physics", 5, "Goddard"),
        ("ra_lw_physics", 7, "FLG"),
        ("ra_lw_physics", 99, "GFDL"),
    ],
)
def test_operational_validator_rejects_reference_only_radiation(
    key: str, value: int, scheme_substring: str
) -> None:
    """v0.17/v0.18 radiation-longtail selections (CAM/Goddard/FLG/GFDL-Eta) are
    accepted for reference work but not operationally wired until real Fortran
    parity and faithful JAX kernels exist."""

    validate_namelist({"physics": {key: [value]}})
    with pytest.raises(NotOperationallyWiredError) as excinfo:
        validate_operational_namelist({"physics": {key: [value]}})
    sel = _selection_for(excinfo, key)
    assert sel.outcome == "reference_only_not_operational"
    assert sel.value == value
    assert scheme_substring in (sel.wrf_scheme or "")


def test_operational_validator_accepts_wired_rrtm_lw() -> None:
    """ra_lw_physics=1 (classic AER RRTM LW) is now operationally scan-wired, so the
    operational validator ACCEPTS it -- on its own and in every (ra_sw, ra_lw)
    combination of the two wired SW/LW schemes. This is the contract flip from
    REFERENCE_ONLY -> IMPLEMENTED."""

    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

    assert classify_scheme("ra_lw_physics", 1).status is SupportStatus.IMPLEMENTED
    validate_operational_namelist({"physics": {"ra_lw_physics": [1]}})
    for ra_sw in (1, 4):
        for ra_lw in (1, 4):
            validate_operational_namelist(
                {"physics": {"ra_sw_physics": [ra_sw], "ra_lw_physics": [ra_lw]}}
            )


def test_operational_validator_accepts_myj_janjic_pair() -> None:
    """v0.13: MYJ PBL (bl=2) + Janjic Eta surface layer (sf=2) is now an
    IMPLEMENTED operational pair -- both the validation and operational layers
    accept it (the pair is jit/vmap-traceable + scan-wired)."""

    # Reference (validation) layer accepts the pair.
    validate_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}})
    # The operational layer now ALSO accepts it (no longer reference-only).
    validate_operational_namelist(
        {"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}}
    )


def test_operational_validator_rejects_unpaired_myj() -> None:
    """The mandatory MYJ<->Janjic pairing still fails closed when only one is set."""

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_operational_namelist(
            {"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [5]}}
        )
    assert any(s.key == "myj_pairing" for s in excinfo.value.selections)


def test_operational_validator_rejects_reference_only_per_domain() -> None:
    """A multi-domain namelist with a reference-only scheme on one domain is
    rejected for that domain. Both radiation options ra_sw=1/ra_lw=1 are now wired,
    so cu=16 (New-Tiedtke, still reference-only) drives the per-domain reject."""

    with pytest.raises(NotOperationallyWiredError) as excinfo:
        validate_operational_namelist({"physics": {"cu_physics": [1, 16]}})
    sel = _selection_for(excinfo, "cu_physics")
    assert sel.domain_index == 2
    assert sel.value == 16


def test_operational_validator_still_rejects_unimplemented_and_out_of_scope() -> None:
    """The operational validator subsumes the full validate_namelist checks:
    recognized-but-unimplemented schemes and out-of-scope features still fail."""

    # mp=29 (RCON) stays recognized-but-unimplemented (mp=28 aerosol-aware
    # Thompson became implemented in v0.16).
    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_operational_namelist({"physics": {"mp_physics": [29]}})
    assert "NOT YET IMPLEMENTED" in str(excinfo.value)

    with pytest.raises(UnsupportedSchemeError) as excinfo2:
        validate_operational_namelist({"chem": {"chem_opt": 401}})
    assert "out-of-scope" in str(excinfo2.value)


def test_validate_namelist_still_accepts_reference_only_schemes() -> None:
    """REGRESSION GUARD: the validation layer must KEEP accepting reference-only
    schemes (other callers run reference comparisons against them). Only the
    OPERATIONAL validator rejects them."""

    validate_namelist(
        {"physics": {"ra_lw_physics": [1], "ra_sw_physics": [1], "cu_physics": [16]}}
    )
    validate_namelist({"physics": {"bl_pbl_physics": [2], "sf_sfclay_physics": [2]}})
    validate_namelist({"physics": {"bl_pbl_physics": [4, 10, 16, 17]}})


def test_not_operationally_wired_is_an_unsupported_scheme_error() -> None:
    """The CLI catches UnsupportedSchemeError; the operational rejection must be it."""

    assert issubclass(NotOperationallyWiredError, UnsupportedSchemeError)


def test_real_wrf_namelist_input_is_consumable() -> None:
    """A standard Fortran namelist.input (all groups) parses and validates.

    Uses the pristine WRF em_real oracle namelist when present (an implemented
    Thompson/KF/MYNN/Noah-MP/RRTMG suite); otherwise an inline equivalent.
    """

    from pathlib import Path

    oracle = Path(
        "/home/user/src/wrf_pristine/WRF/test/em_real/oracle_run/namelist.input"
    )
    if oracle.is_file():
        from gpuwrf.io.namelist_check import _parse_wrf_namelist

        parsed = _parse_wrf_namelist(oracle.read_text())
        # All standard WRF groups present.
        for group in ("time_control", "domains", "physics", "dynamics", "bdy_control"):
            assert group in parsed, f"missing &{group} in parsed namelist"
        # The oracle's diff_opt=1/km_opt=4 real-data defaults are now the wired
        # 2-D Smagorinsky path (v0.9.0), so a standard em_real namelist with an
        # implemented Thompson/KF/MYNN/Noah-MP/RRTMG suite validates cleanly.
        validate_supported_namelist(oracle)
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
