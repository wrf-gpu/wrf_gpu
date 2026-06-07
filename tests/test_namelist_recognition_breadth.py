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
    validate_namelist,
    validate_operational_namelist,
)
from gpuwrf.io.scheme_catalog import (
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

    # cudt=5 (cumulus sub-stepping cadence) on BOTH domains; the port runs
    # cumulus every step (cudt=0).
    assert len(by_key["cudt"]) == 2
    assert "every dynamics step" in message.lower() or "every step" in message.lower()

    # icloud_bl=1 and the MYNN TKE-advection logical -- recognized, not wired.
    assert "icloud_bl" in by_key
    assert "bl_mynn_tkeadvect" in by_key
    # Each named reason + the "NOT silently ignored" honesty phrase is present.
    for key in ("moist_adv_opt", "cudt", "icloud_bl", "bl_mynn_tkeadvect"):
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

    unwired_examples = {
        "gwd_opt": 3,  # GSL drag suite -- not wired (gwd_opt=1 IS implemented)
        "moist_adv_opt": 3,
        "scalar_adv_opt": 4,
        "h_sca_adv_order": 6,
        "v_sca_adv_order": 5,
        "icloud_bl": 1,
        "bl_mynn_edmf": 0,
        "bl_mynn_mixqt": 1,
        "cudt": 10,
        "bldt": 5,
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
