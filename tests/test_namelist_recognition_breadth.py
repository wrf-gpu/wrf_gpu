"""Proof object for sprint B5 -- namelist recognition breadth.

v0.12.0 honesty contract: a real WRF ``namelist.input`` that selects dynamics/
advection switches, MYNN-EDMF sub-options, gravity-wave drag, physics-cadence
intervals, slope/topo radiation and out-of-scope feature switches must get an
HONEST per-key verdict from :func:`validate_namelist`:

* IMPLEMENTED selections (and operationally-wired control values) pass silently;
* a RECOGNIZED-but-not-operationally-wired control value fails closed with a
  NAMED reason + alternative (never a silent ignore, never a generic error);
* an OUT_OF_SCOPE feature switch fails closed as a named scope decision.

The single source of truth for every per-key verdict is the catalog
(``gpuwrf.io.scheme_catalog``); this test asserts the validator surfaces those
verdicts and -- critically -- that an unsupported-but-recognized key is REJECTED
rather than silently accepted.
"""

from __future__ import annotations

import pytest

from gpuwrf.io.namelist_check import (
    UnsupportedSchemeError,
    collect_namelist_warnings,
    validate_namelist,
    validate_operational_namelist,
)
from gpuwrf.io.scheme_catalog import (
    APPROXIMATED_CONTROL_KEYS,
    IMPLEMENTED_CONTROL_KEYS,
    RECOGNIZED_CONTROL_KEYS,
    SupportStatus,
    assert_catalog_consistent,
    classify_control,
)


# A representative REAL WRF namelist (one key per line, as WRF emits them) that
# mixes: an implemented physics suite, operationally-wired control values, a
# batch of recognized-but-unwired controls, an implemented slope/topo radiation
# pair, and out-of-scope feature switches. Modeled on the pristine em_real
# oracle namelist (cudt=5, gwd_opt=1, radt=30, bldt=0, topo_shading/slope_rad=1,
# moist_adv_opt/scalar_adv_opt=1).
REAL_WRF_NAMELIST = """\
&time_control
 run_hours = 24,
/
&domains
 max_dom = 2,
/
&physics
 mp_physics = 8, 8,
 cu_physics = 1, 1,
 bl_pbl_physics = 5, 5,
 sf_sfclay_physics = 5, 5,
 sf_surface_physics = 4, 4,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 30, 30,
 bldt = 0, 0,
 cudt = 5, 5,
 icloud_bl = 1,
 bl_mynn_tkeadvect = .true.,
 bl_mynn_edmf = 1,
 bl_mynn_mixqt = 0,
 topo_shading = 1, 1,
 slope_rad = 1, 1,
 sst_update = 0,
 windfarm_opt = 1,
/
&dynamics
 rk_order = 3,
 diff_opt = 1,
 km_opt = 4,
 gwd_opt = 1, 0,
 moist_adv_opt = 2, 2,
 scalar_adv_opt = 1, 1,
 h_sca_adv_order = 5,
 v_sca_adv_order = 3,
 h_mom_adv_order = 5,
 v_mom_adv_order = 3,
/
&fdda
 grid_fdda = 1,
/
&bdy_control
 spec_bdy_width = 5,
/
"""


def _selections_by_key(exc: UnsupportedSchemeError) -> dict[str, list]:
    out: dict[str, list] = {}
    for sel in exc.selections:
        out.setdefault(sel.key, []).append(sel)
    return out


def test_catalog_stays_internally_consistent() -> None:
    """The new recognized-control tables must not break the catalog invariants."""

    assert_catalog_consistent()


def test_real_wrf_namelist_yields_honest_per_key_verdicts() -> None:
    """Feed a representative real WRF namelist; assert each key's honest verdict.

    This is the B5 proof object. Every key below is asserted to land in exactly
    one honest bucket: IMPLEMENTED (passes), RECOGNIZED-fail-closed-with-reason,
    or OUT_OF_SCOPE -- and the unsupported-but-recognized keys are REJECTED with
    a named reason rather than silently accepted.
    """

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist(REAL_WRF_NAMELIST)
    exc = excinfo.value
    by_key = _selections_by_key(exc)
    message = str(exc)

    # --- (a) Recognized-but-NOT-operationally-wired controls -> named fail. -- #
    # gwd_opt=1 IS now implemented (orographic GWD + flow blocking, the faithful
    # bl_gwdo_run port). Both wired values (1 = on, 0 = off) must therefore NOT
    # be flagged as failures -- it is no longer an unsupported control.
    assert "gwd_opt" not in by_key

    # moist_adv_opt=2 (positive-definite transport variant) on BOTH domains.
    assert len(by_key["moist_adv_opt"]) == 2
    assert all(s.value == 2 for s in by_key["moist_adv_opt"])
    assert "positive-definite" in message

    # cudt=5 (cumulus sub-stepping cadence): NO LONGER fail-closed. It is a
    # conservative approximation (the port runs cumulus every step), so it must
    # NOT appear as a failure -- a naive user with a real WRF namelist must RUN.
    assert "cudt" not in by_key

    # icloud_bl=1 and the MYNN TKE-advection logical -- recognized, not wired.
    assert "icloud_bl" in by_key
    assert "bl_mynn_tkeadvect" in by_key
    # Each named reason + the "NOT silently ignored" honesty phrase is present.
    for key in ("moist_adv_opt", "icloud_bl", "bl_mynn_tkeadvect"):
        for sel in by_key[key]:
            assert sel.outcome == "recognized_control_not_wired"
            assert sel.action.strip(), f"{key} fail-closed without an alternative"
    assert "NOT silently ignored" in message

    # --- (b) OUT_OF_SCOPE feature switches -> named scope decision. ---------- #
    assert "windfarm_opt" in by_key
    assert by_key["windfarm_opt"][0].outcome == "out_of_scope"
    assert "Wind-farm" in message
    assert "grid_fdda" in by_key
    assert by_key["grid_fdda"][0].outcome == "out_of_scope"

    # --- (c) IMPLEMENTED / wired controls are NOT flagged. ------------------- #
    # slope_rad=1 and topo_shading=1 ARE implemented (RRTMG SW slope/shadow) and
    # must NOT appear as failures -- do not falsely fail-close an implemented
    # feature.
    for wired in (
        "gwd_opt",
        "slope_rad",
        "topo_shading",
        "radt",
        "bldt",
        "cudt",  # cudt>0 is an approximation WARNING now, never a failure
        "scalar_adv_opt",
        "h_sca_adv_order",
        "v_sca_adv_order",
        "h_mom_adv_order",
        "v_mom_adv_order",
        "bl_mynn_edmf",
        "bl_mynn_mixqt",
        "sst_update",
        "mp_physics",
        "bl_pbl_physics",
    ):
        assert wired not in by_key, f"{wired} was wrongly fail-closed (it is implemented/wired/off)"

    # And the cudt=5 approximation surfaces as a non-fatal WARNING (run proceeds).
    cudt_warnings = [w for w in collect_namelist_warnings(REAL_WRF_NAMELIST) if "cudt" in w]
    assert cudt_warnings, "cudt=5 must surface a non-fatal approximation warning"
    assert any("every dynamics step" in w.lower() for w in cudt_warnings)


def test_implemented_slope_topo_radiation_is_not_failed() -> None:
    """slope_rad=1 / topo_shading=1 are implemented; classify_control confirms."""

    assert classify_control("slope_rad", 1).status is SupportStatus.IMPLEMENTED
    assert classify_control("topo_shading", 1).status is SupportStatus.IMPLEMENTED
    # A clean meteorology namelist that enables only implemented features passes.
    validate_namelist(
        {
            "physics": {
                "mp_physics": [8],
                "bl_pbl_physics": [5],
                "slope_rad": [1],
                "topo_shading": [1],
                "radt": [30],
                "bldt": [0],
                "moist_adv_opt": [1],
                "scalar_adv_opt": [1],
            }
        }
    )


def test_unsupported_recognized_key_is_rejected_not_silently_accepted() -> None:
    """The core honesty requirement: a recognized-but-unwired key must RAISE.

    Previously these keys were absent from SUPPORTED_OPTIONS and therefore
    SILENTLY IGNORED. Each must now fail closed with its own named reason.
    """

    # NOTE: cudt/bldt are deliberately EXCLUDED here -- a positive cadence is a
    # conservative approximation (run-every-step), surfaced as a non-fatal
    # warning rather than a fail-closed rejection. See
    # test_cadence_keys_are_approximation_warnings_not_rejections.
    unwired_examples = {
        "gwd_opt": 3,  # GSL drag suite -- not wired (gwd_opt=1 IS implemented)
        "moist_adv_opt": 3,
        "scalar_adv_opt": 4,
        "h_sca_adv_order": 6,
        "v_sca_adv_order": 5,
        "icloud_bl": 1,
        "bl_mynn_edmf": 0,
        "bl_mynn_mixqt": 1,
        "radt": 0,
        "slope_rad": 2,
    }
    for key, value in unwired_examples.items():
        with pytest.raises(UnsupportedSchemeError) as excinfo:
            validate_namelist({"dynamics": {key: value}})
        sel = [s for s in excinfo.value.selections if s.key == key]
        assert sel, f"{key}={value} was not rejected (silently accepted?)"
        assert sel[0].outcome == "recognized_control_not_wired"
        # A specific named reason + alternative, not a generic error.
        assert sel[0].action.strip(), f"{key} rejected without an alternative recipe"
        assert sel[0].wrf_scheme, f"{key} rejected without a control label"


def test_recognized_control_keyspace_is_covered() -> None:
    """Every key this sprint promised to recognize is in the catalog namespace."""

    promised_recognized = {
        "gwd_opt",
        "moist_adv_opt",
        "scalar_adv_opt",
        "h_sca_adv_order",
        "v_sca_adv_order",
        "h_mom_adv_order",
        "v_mom_adv_order",
        "icloud_bl",
        "bl_mynn_tkeadvect",
        "bl_mynn_edmf",
        "bl_mynn_edmf_mom",
        "bl_mynn_edmf_tke",
        "bl_mynn_edmf_dd",
        "bl_mynn_mixscalars",
        "bl_mynn_mixqt",
        "bl_mynn_mixlength",
        "radt",
        "bldt",
        "cudt",
    }
    assert promised_recognized <= RECOGNIZED_CONTROL_KEYS
    # slope_rad/topo_shading are recognized AND implemented (not fail-closed).
    assert {"slope_rad", "topo_shading"} <= IMPLEMENTED_CONTROL_KEYS
    # cudt/bldt are the APPROXIMATED cadence controls (warn, do not reject).
    assert {"cudt", "bldt"} == APPROXIMATED_CONTROL_KEYS
    assert APPROXIMATED_CONTROL_KEYS <= RECOGNIZED_CONTROL_KEYS


# --------------------------------------------------------------------------- #
# PROOF OBJECT -- naive-user cadence-key usability.                            #
# A real Canary/WRF production namelist (cudt=5, gwd_opt=1, radt=30,           #
# cu_physics=1, bldt=0) must RUN out-of-box via the standalone CLI validator   #
# -- the cumulus/PBL cadence keys WARN (conservative every-step approximation) #
# rather than REJECT. Genuine wrong-substitutions still fail closed.           #
# --------------------------------------------------------------------------- #

# Modeled on the real production namelist
# /mnt/data/canairy_meteo/runs/wrf_l3/20260503_18z_l3_24h_*/namelist.input
# (cudt=5,5,..., gwd_opt=1, radt=30, cu_physics=1,0,..., bldt=0, slope_rad=1,
# topo_shading=1, mp=8 Thompson, bl=5 MYNN, sfclay=5, sf_surface=4 Noah-MP,
# ra_lw=4/ra_sw=4 RRTMG, diff_opt=1/km_opt=4, damp_opt=3, moist/scalar_adv=1).
REAL_CANARY_NAMELIST = """\
&time_control
 run_hours = 24,
 input_from_file = .true., .true., .true.,
/
&domains
 time_step = 18,
 max_dom = 3,
 e_we = 94, 160, 94,
 e_sn = 60, 67, 76,
/
&physics
 mp_physics = 8, 8, 8,
 bl_pbl_physics = 5, 5, 5,
 sf_sfclay_physics = 5, 5, 5,
 sf_surface_physics = 4, 4, 4,
 ra_lw_physics = 4, 4, 4,
 ra_sw_physics = 4, 4, 4,
 cu_physics = 1, 0, 0,
 cudt = 5, 5, 5,
 radt = 30, 30, 30,
 bldt = 0, 0, 0,
 topo_shading = 1, 1, 1,
 slope_rad = 1, 1, 1,
 sf_urban_physics = 0, 0, 0,
 sst_update = 0,
/
&dynamics
 w_damping = 1,
 diff_opt = 1,
 km_opt = 4,
 diff_6th_opt = 2, 2, 2,
 damp_opt = 3,
 non_hydrostatic = .true., .true., .true.,
 moist_adv_opt = 1, 1, 1,
 scalar_adv_opt = 1, 1, 1,
 gwd_opt = 1,
/
&bdy_control
 spec_bdy_width = 5,
/
"""


def test_real_canary_namelist_proceeds_with_cudt_warning() -> None:
    """(a) The real Canary production namelist must PROCEED (no raise) and emit a
    cudt cadence WARNING -- the naive-user out-of-box fix.

    cudt=5 (and a positive bldt, were it set) is a conservative approximation
    (the GPU port runs cumulus/PBL every step), so the operational validator does
    NOT reject it. gwd_opt=1 is implemented; radt=30 is the radiation cadence;
    everything else in this namelist is implemented/wired."""

    # Neither the validation layer nor the strict OPERATIONAL run path may raise.
    validate_namelist(REAL_CANARY_NAMELIST)
    validate_operational_namelist(REAL_CANARY_NAMELIST)

    # The cudt approximation is surfaced as a non-fatal warning naming it.
    warnings = collect_namelist_warnings(REAL_CANARY_NAMELIST)
    cudt_warnings = [w for w in warnings if "cudt" in w]
    assert cudt_warnings, "real Canary cudt=5 must emit a cadence warning"
    text = " ".join(cudt_warnings).lower()
    assert "every dynamics step" in text or "every step" in text
    assert "approximation" in text


def test_cadence_keys_are_approximation_warnings_not_rejections() -> None:
    """cudt>0 and bldt>0 are RECOGNIZED_APPROXIMATED (warn, proceed) -- never a
    fail-closed rejection."""

    for key, value in (("cudt", 5), ("bldt", 5), ("cudt", 10.5)):
        support = classify_control(key, value)
        assert support is not None
        assert support.status is SupportStatus.RECOGNIZED_APPROXIMATED, (
            f"{key}={value} should be RECOGNIZED_APPROXIMATED, got {support.status}"
        )
        assert support.reason.strip()
        # A positive cadence does NOT raise from either validator.
        validate_namelist({"physics": {key: [value]}})
        validate_operational_namelist({"physics": {key: [value]}})
        # And it surfaces a warning.
        warnings = collect_namelist_warnings({"physics": {key: [value]}})
        assert any(key in w for w in warnings)

    # cudt=0 / bldt=0 (the exactly-wired every-step request) pass with NO warning.
    assert classify_control("cudt", 0).status is SupportStatus.IMPLEMENTED
    assert classify_control("bldt", 0).status is SupportStatus.IMPLEMENTED
    assert collect_namelist_warnings({"physics": {"cudt": [0], "bldt": [0]}}) == []


def test_genuine_wrong_substitutions_still_fail_closed() -> None:
    """(b/c/d) The naive-user fix must NOT weaken fail-closed for genuine
    wrong-substitutions:

    * (b) moist_adv_opt=2 (a different, unimplemented advection scheme) RAISES;
    * (c) cu_physics=16 (New-Tiedtke, reference-only -> would silently become a
      different cumulus scheme on the operational scan) RAISES on the operational
      path (ra_lw=1/ra_sw=1 are now wired, so cumulus drives this check);
    * (d) grid_fdda=1 (out-of-scope feature) RAISES.
    """

    # (b) different advection scheme -> still fail closed.
    with pytest.raises(UnsupportedSchemeError) as exc_b:
        validate_namelist({"dynamics": {"moist_adv_opt": [2]}})
    assert any(s.key == "moist_adv_opt" for s in exc_b.value.selections)
    assert "positive-definite" in str(exc_b.value)

    # (c) reference-only scheme -> operational run still fail closed.
    with pytest.raises(UnsupportedSchemeError) as exc_c:
        validate_operational_namelist({"physics": {"cu_physics": [16]}})
    assert any(s.key == "cu_physics" for s in exc_c.value.selections)

    # (d) out-of-scope feature -> still fail closed.
    with pytest.raises(UnsupportedSchemeError) as exc_d:
        validate_namelist({"fdda": {"grid_fdda": 1}})
    assert any(s.key == "grid_fdda" for s in exc_d.value.selections)


def test_operational_validator_also_rejects_unwired_controls() -> None:
    """The strict operational entrypoint inherits the recognized-control checks."""

    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_operational_namelist({"dynamics": {"gwd_opt": 3}})
    assert any(s.key == "gwd_opt" for s in excinfo.value.selections)


def test_wired_control_values_pass_silently() -> None:
    """The operationally-wired control values must NOT raise (no false positives)."""

    validate_namelist(
        {
            "physics": {
                "mp_physics": [8],
                "bl_pbl_physics": [5],
                "icloud_bl": [0],
                "bl_mynn_tkeadvect": [".false."],
                "bl_mynn_edmf": [1],
                "bl_mynn_edmf_mom": [1],
                "bl_mynn_edmf_tke": [0],
                "bl_mynn_mixscalars": [1],
                "bl_mynn_mixqt": [0],
                "bl_mynn_mixlength": [2],
                "radt": [15],
                "bldt": [0],
                "cudt": [0],
            },
            "dynamics": {
                "gwd_opt": [1],  # orographic GWD ON -- now implemented
                "moist_adv_opt": [1],
                "scalar_adv_opt": [1],
                "h_sca_adv_order": [5],
                "v_sca_adv_order": [3],
                "h_mom_adv_order": [5],
                "v_mom_adv_order": [3],
            },
        }
    )
