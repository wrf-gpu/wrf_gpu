"""v0.20.0 #102 — G0-verdict patch unit test.

Exercises the hardened Gate-0 reducer (`scripts/v020_g0_verdict.py`) against the
v2 production contract's B2a/B2b/B2c rules
(`proofs/v020/fp32_proto/S4_S7_PRODUCTION_PLAN.md` Gate 0, D-critique B2).

Required scenarios (per the integration brief STEP 3):
  - null manifest                          -> NOT a KILL (B2b: never kill on no
                                              classification)  [INCONCLUSIVE]
  - clean production speed                 -> SPEED-GO
  - fp32 shrank storage but island-capped  -> GO-to-S7  (NOT a KILL, B2b)
  - identical walls + no shrink            -> real-KILL (the 1.106x signature)
  - fits but not fast                      -> CAPABILITY-GO
  - bottleneck moved out of scope          -> REDIRECT (B2c)

Plus B2a guards: the GO decision uses the PRODUCTION arm, never the aggressive
ceiling; an in-scope moved bottleneck is a valid Speed-GO.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v020_g0_verdict.py"
SPEC = importlib.util.spec_from_file_location("v020_g0_verdict", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

decide = MODULE.decide


# --------------------------------------------------------------------------- #
# manifest builders
# --------------------------------------------------------------------------- #
def _grid(oom_target: bool = True) -> dict:
    return {"name": "test_grid", "cols": 211600, "fp64_oom_target": oom_target}


def _manifest(fp64=None, fp32_production=None, fp32_aggressive=None, grid=None):
    m = {"grid": grid if grid is not None else _grid()}
    if fp64 is not None:
        m["fp64"] = fp64
    if fp32_production is not None:
        m["fp32_production"] = fp32_production
    if fp32_aggressive is not None:
        m["fp32_aggressive"] = fp32_aggressive
    return m


# --------------------------------------------------------------------------- #
# 1) null manifest -> KILL? NO. B2b forbids killing without classification.
# --------------------------------------------------------------------------- #
def test_null_manifest_is_not_a_kill_inconclusive():
    res = decide({})
    assert res["verdict"] == "G0-INCONCLUSIVE", res["verdict"]
    assert res["gates"]["G0_KILL"] is False
    assert res["b2b_classification"]["real_kill"] is False
    # the action must explicitly refuse to kill on this evidence
    assert "do not kill" in res["action"].lower() or "not kill" in res["action"].lower()


def test_null_production_arm_all_none_inconclusive_not_kill():
    m = _manifest(
        fp64={"ms_per_step": None, "peak_vram_mib": None, "fit": True, "dominant_kernel_ms": None},
        fp32_production={"ms_per_step": None, "peak_vram_mib": None, "fit": None,
                         "dominant_kernel_ms": None, "transient_fp32_fraction": None},
    )
    res = decide(m)
    assert res["verdict"] == "G0-INCONCLUSIVE", res
    assert res["gates"]["G0_KILL"] is False


# --------------------------------------------------------------------------- #
# 2) clean production speed -> SPEED-GO
# --------------------------------------------------------------------------- #
def test_clean_production_speed_is_speed_go():
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 9.0,            # 2.22x on the PRODUCTION arm
            "peak_vram_mib": 5000.0,        # VRAM lower
            "fit": True,
            "dominant_kernel_ms": 2.2,      # dominant kernel actually faster
            "transient_fp32_fraction": 0.7, # transient majority fp32-able
            "storage_shrank": True,
        },
    )
    res = decide(m)
    assert res["verdict"] == "G0-SPEED-GO", res
    assert res["gates"]["G0_SPEED_GO"] is True
    assert res["metrics"]["warm_speedup_x_production"] == pytest.approx(20.0 / 9.0)


# --------------------------------------------------------------------------- #
# 3) fp32 shrank storage/transient but wall island/launch-capped -> GO-to-S7
#    (B2b: this is the S7 target condition, NEVER a KILL)
# --------------------------------------------------------------------------- #
def test_storage_shrank_but_island_capped_is_go_to_s7_not_kill():
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 18.5,            # only ~1.08x -> NOT on a 2x trajectory
            "peak_vram_mib": 5500.0,        # VRAM lower: storage shrank
            "fit": True,
            "dominant_kernel_ms": 5.0,      # dominant kernel wall unchanged (island-capped)
            "transient_fp32_fraction": 0.65,
            "storage_shrank": True,
            "walls_identical": True,        # walls identical BUT storage DID shrink
            "wall_capped_by_launch_or_island": True,
        },
    )
    res = decide(m)
    assert res["verdict"] == "G0-GO-TO-S7", res
    assert res["gates"]["G0_GO_TO_S7"] is True
    assert res["gates"]["G0_KILL"] is False
    assert res["b2b_classification"]["real_kill"] is False
    assert res["b2b_classification"]["go_to_s7_target_condition"] is True


def test_storage_shrank_island_capped_via_derived_flags_is_go_to_s7():
    # No explicit wall_capped/storage flags: derive from VRAM-lower + dominant
    # kernel NOT faster. Must still resolve to GO-to-S7, never KILL.
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 18.0,
            "peak_vram_mib": 5500.0,        # VRAM lower -> storage_shrank proxy True
            "fit": True,
            "dominant_kernel_ms": 5.0,      # not faster -> wall capped
            "transient_fp32_fraction": 0.6,
        },
    )
    res = decide(m)
    assert res["verdict"] == "G0-GO-TO-S7", res
    assert res["gates"]["G0_KILL"] is False


# --------------------------------------------------------------------------- #
# 4) identical walls AND no storage shrink -> real KILL (1.106x signature)
# --------------------------------------------------------------------------- #
def test_identical_walls_no_shrink_is_real_kill():
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 18.1,            # ~1.105x, the 1.106 signature
            "peak_vram_mib": 8000.0,        # VRAM identical: no storage shrink
            "fit": True,
            "dominant_kernel_ms": 5.0,      # identical wall
            "transient_fp32_fraction": 0.6,
            "storage_shrank": False,        # explicit: storage did NOT shrink
            "walls_identical": True,
        },
    )
    res = decide(m)
    assert res["verdict"] == "G0-KILL", res
    assert res["gates"]["G0_KILL"] is True
    assert res["b2b_classification"]["real_kill"] is True
    assert res["b2b_classification"]["go_to_s7_target_condition"] is False


def test_real_kill_requires_both_no_shrink_and_identical_walls():
    # no shrink but walls NOT identical (dominant faster) -> must NOT be a real kill
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 14.0,
            "peak_vram_mib": 8000.0,        # no shrink
            "fit": True,
            "dominant_kernel_ms": 3.0,      # but dominant kernel IS faster
            "transient_fp32_fraction": 0.6,
            "storage_shrank": False,
            "walls_identical": False,
        },
    )
    res = decide(m)
    assert res["verdict"] != "G0-KILL", res
    assert res["b2b_classification"]["real_kill"] is False


# --------------------------------------------------------------------------- #
# 5) fits-but-not-fast -> CAPABILITY-GO
# --------------------------------------------------------------------------- #
def test_fits_but_not_fast_is_capability_go():
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 24000.0, "fit": False, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 21.0,            # NOT faster (slightly slower even)
            "peak_vram_mib": 12000.0,       # VRAM lower -> fits the grid fp64 OOMs on
            "fit": True,
            "dominant_kernel_ms": 5.2,      # not faster
            "transient_fp32_fraction": 0.3, # transient NOT majority fp32 -> speed gate fails
        },
        grid=_grid(oom_target=True),
    )
    res = decide(m)
    assert res["verdict"] == "G0-CAPABILITY-GO", res
    assert res["gates"]["G0_CAPABILITY_GO"] is True
    assert res["gates"]["G0_SPEED_GO"] is False


# --------------------------------------------------------------------------- #
# 6) bottleneck moved OUT of scope -> REDIRECT (B2c), not Speed-GO, not KILL
# --------------------------------------------------------------------------- #
def test_bottleneck_moved_out_of_scope_is_redirect():
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 18.0,            # ~1.1x, not fast
            "peak_vram_mib": 8000.0,        # no VRAM win -> capability fails too
            "fit": True,
            "dominant_kernel_ms": 5.0,
            "transient_fp32_fraction": 0.6,
            "storage_shrank": False,
            "walls_identical": False,       # NOT the 1.106 signature -> not a real kill
            "bottleneck_moved": True,
            "bottleneck_in_scope": False,   # OUT of S4/S7 scope, no scheduled follow-on
            "bottleneck_component": "MYNN_qke",
        },
    )
    res = decide(m)
    assert res["verdict"] == "G0-REDIRECT", res
    assert res["gates"]["G0_REDIRECT"] is True
    assert res["gates"]["G0_SPEED_GO"] is False
    assert res["gates"]["G0_KILL"] is False
    assert "MYNN_qke" in res["action"]


def test_bottleneck_moved_out_of_scope_does_not_become_speed_go():
    # Even with VRAM lower + transient majority, an OUT-OF-SCOPE moved bottleneck
    # alone must NOT satisfy the 2x trajectory (B2c).
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 18.0,            # ~1.1x: below 1.5x and not dom-faster
            "peak_vram_mib": 5000.0,        # VRAM lower
            "fit": True,
            "dominant_kernel_ms": 5.0,      # not faster
            "transient_fp32_fraction": 0.7, # majority fp32
            "storage_shrank": True,
            "bottleneck_moved": True,
            "bottleneck_in_scope": False,
            "bottleneck_component": "radiation_rrtmg",
        },
    )
    res = decide(m)
    assert res["gates"]["G0_SPEED_GO"] is False, res
    # storage shrank -> the honest outcome is GO-to-S7 (not REDIRECT, not KILL);
    # the out-of-scope bottleneck does not by itself force a speed-go.
    assert res["verdict"] in ("G0-GO-TO-S7", "G0-REDIRECT"), res
    assert res["verdict"] != "G0-SPEED-GO"


# --------------------------------------------------------------------------- #
# B2a — production arm decides; aggressive ceiling is context only
# --------------------------------------------------------------------------- #
def test_b2a_aggressive_does_not_drive_go():
    # aggressive arm is a blazing 3x, production arm is a flat ~1.1x with the
    # 1.106 signature -> verdict must follow PRODUCTION (real kill), not aggressive.
    m = _manifest(
        fp64={"ms_per_step": 30.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 6.0},
        fp32_production={
            "ms_per_step": 27.2,            # ~1.1x
            "peak_vram_mib": 8000.0,        # no shrink
            "fit": True,
            "dominant_kernel_ms": 6.0,      # identical wall
            "transient_fp32_fraction": 0.6,
            "storage_shrank": False,
            "walls_identical": True,
        },
        fp32_aggressive={"ms_per_step": 10.0, "peak_vram_mib": 5000.0, "dominant_kernel_ms": 2.0},
    )
    res = decide(m)
    assert res["decided_on"].startswith("fp32_production"), res["decided_on"]
    assert res["verdict"] == "G0-KILL", res  # production says real-kill despite aggressive 3x
    # aggressive speedup IS reported, just as context
    assert res["metrics"]["warm_speedup_x_aggressive_CONTEXT_ONLY"] == pytest.approx(3.0)
    assert "context only" in res["b2a_note"].lower()


def test_b2a_legacy_single_fp32_arm_is_flagged_as_fallback():
    m = {
        "grid": _grid(),
        "fp64": {"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        "fp32": {"ms_per_step": 9.0, "peak_vram_mib": 5000.0, "fit": True,
                 "dominant_kernel_ms": 2.0, "transient_fp32_fraction": 0.7, "storage_shrank": True},
    }
    res = decide(m)
    assert "fallback" in res["decided_on"].lower()
    # still decides correctly on that single arm
    assert res["verdict"] == "G0-SPEED-GO", res


# --------------------------------------------------------------------------- #
# B2c — an IN-SCOPE moved bottleneck IS a valid Speed-GO
# --------------------------------------------------------------------------- #
def test_b2c_bottleneck_moved_in_scope_is_speed_go():
    m = _manifest(
        fp64={"ms_per_step": 20.0, "peak_vram_mib": 8000.0, "fit": True, "dominant_kernel_ms": 5.0},
        fp32_production={
            "ms_per_step": 18.0,            # ~1.1x raw, but...
            "peak_vram_mib": 5000.0,        # VRAM lower
            "fit": True,
            "dominant_kernel_ms": 5.0,
            "transient_fp32_fraction": 0.7, # majority fp32
            "storage_shrank": True,
            "bottleneck_moved": True,
            "bottleneck_in_scope": True,    # addressed by S4/S7 / named follow-on
            "bottleneck_component": "acoustic_substep_launch",
        },
    )
    res = decide(m)
    assert res["verdict"] == "G0-SPEED-GO", res
    assert res["b2c_bottleneck"]["moved_in_scope_counts_as_speed_go"] is True


# --------------------------------------------------------------------------- #
# CLI smoke: --emit-template then --manifest round-trips (driver contract)
# --------------------------------------------------------------------------- #
def test_cli_template_roundtrip(tmp_path):
    tmpl = tmp_path / "g0.json"
    out = tmp_path / "verdict.json"
    rc = MODULE.main(["--emit-template", str(tmpl)])
    assert rc == 0 and tmpl.is_file()
    rc = MODULE.main(["--manifest", str(tmpl), "--out", str(out)])
    assert rc == 0 and out.is_file()
    import json
    res = json.loads(out.read_text())
    # all-null template must NOT be a KILL (B2b safety)
    assert res["verdict"] == "G0-INCONCLUSIVE"
    assert res["gates"]["G0_KILL"] is False
