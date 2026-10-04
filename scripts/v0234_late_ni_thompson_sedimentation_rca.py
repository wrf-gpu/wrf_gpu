#!/usr/bin/env python3
"""CPU source-literal proof of the V0234 late-Ni Thompson root cause.

The exact GPU arm retained the complete ordinary input/output carries around
d01 native step 1148.  This script consumes those immutable carries on CPU,
replays the production Thompson column at the first catastrophic column, and
then executes the source stages one at a time.  It also evaluates the exact
WRF-v4.7.1 ice mass/number balance from module_mp_thompson.F:3033-3055 before
sedimentation.  It never initializes or queries a GPU and never edits a model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-late-ni-rootcause"
DEFAULT_OUTPUT = SPRINT / "THOMPSON_SEDIMENTATION_ROOT_CAUSE.json"
EXACT_RESULT = SPRINT / "GPU_EXACT_FIRST_RED_RESULT.json"
EXACT_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_gpt_late_ni_df242a850d6ebab1_exact"
)
INPUT_CARRY = EXACT_DIR / "last-green-advance-input-d01-step-1147.pkl"
OUTPUT_CARRY = EXACT_DIR / "first-red-advance-output-d01-step-1148.pkl"
PORT_SOURCE = ROOT / "src/gpuwrf/physics/thompson_column.py"
WRF_SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_thompson.F")

EXPECTED_INPUT_SHA256 = "0bfc31891eb1da86326a2f5965cfdb8eb981a131d8c7c6a13ad69d2b3acb8414"
EXPECTED_OUTPUT_SHA256 = "a1bfe8d2919e3c5efae51e2af200dfaa36622407694a22a9bec3969ea8a5a51e"
EXPECTED_EXACT_RESULT_SHA256 = "621a0dd925dd8ee596ec413502b766d34e5467534928ee26aa7786fe1ea227d6"
EXPECTED_MODEL_TREE = "e627605f6a8bc0dc23f5c474be4bb532b99297c1"
EXPECTED_MODEL_COMMIT = "d762692a89d5b4031181ecf634cbd53f2c1511bb"
TARGET_YX = (48, 40)
TARGET_K = 3
ICE_SOURCE_K = 19
DT_SECONDS = 54.0

# Exact Thompson constants used by pristine WRF v4.7.1 for mp_physics=8.
R1 = 1.0e-12
R2 = 1.0e-6
PI = 3.1415926536
AM_I = PI * 890.0 / 6.0
BM_I = 3.0
OBMI = 1.0 / BM_I
CIG1 = 1.0
CIG2 = 6.0
OIG1 = 1.0 / CIG1
OIG2 = 1.0 / CIG2
CIE2 = 4.0
ICE_NUMBER_CEILING_M3 = 999.0e3


class RootCauseFailure(RuntimeError):
    """Raised when an immutable input or causal invariant does not hold."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise RootCauseFailure(f"missing proof input: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RootCauseFailure(f"sha256 mismatch for {path}: {digest}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = deepcopy(payload)
    canonical.pop("canonical_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_stats(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
    }
    if np.any(finite):
        selected = array[finite]
        result.update(
            finite_min=float(np.min(selected)),
            finite_max=float(np.max(selected)),
            finite_max_abs=float(np.max(np.abs(selected))),
            finite_argmax_flat=int(np.flatnonzero(finite)[np.argmax(selected)]),
        )
    return result


def wrf_pre_sed_ice_number_balance(qi: Any, ni: Any, rho: Any) -> np.ndarray:
    """Literal vector form of module_mp_thompson.F:3033-3055.

    Inputs/return are mixing-ratio number (kg^-1).  The WRF scratch ``xri`` is
    kg m^-3 and ``xni`` is m^-3.  Active ice is constrained to a 5--300 um
    mass-weighted mean diameter and 999e3 m^-3 maximum before fall speeds are
    formed.  Inactive ice is zero in the prognostic return.
    """

    qi_a = np.asarray(qi, dtype=np.float64)
    ni_a = np.asarray(ni, dtype=np.float64)
    rho_a = np.asarray(rho, dtype=np.float64)
    xri = np.maximum(R1, qi_a * rho_a)
    xni = np.maximum(R2, ni_a * rho_a)
    lami = (AM_I * CIG2 * OIG1 * xni / xri) ** OBMI
    xdi = CIE2 / lami

    small = (xri > R1) & (xdi < 5.0e-6)
    large = (xri > R1) & (xdi > 300.0e-6)
    lami_small = CIE2 / 5.0e-6
    lami_large = CIE2 / 300.0e-6
    xni_small = np.minimum(
        ICE_NUMBER_CEILING_M3,
        CIG1 * OIG2 * xri / AM_I * lami_small**BM_I,
    )
    xni_large = CIG1 * OIG2 * xri / AM_I * lami_large**BM_I
    xni = np.where(small, xni_small, np.where(large, xni_large, xni))
    xni = np.minimum(xni, ICE_NUMBER_CEILING_M3)
    return np.where(qi_a > R1, xni / rho_a, 0.0)


def ice_diameter_um(qi: float, ni: float, rho: float) -> float:
    xri = max(R1, float(qi) * float(rho))
    xni = max(R2, float(ni) * float(rho))
    lami = (AM_I * CIG2 * OIG1 * xni / xri) ** OBMI
    return float(CIE2 / lami * 1.0e6)


def source_authority() -> dict[str, Any]:
    wrf = WRF_SOURCE.read_text(encoding="utf-8")
    port = PORT_SOURCE.read_text(encoding="utf-8")
    assertions = {
        "wrf_pre_sed_balance_5um": "if (xDi.lt. 5.E-6) then" in wrf,
        "wrf_pre_sed_balance_300um": "elseif (xDi.gt. 300.E-6) then" in wrf,
        "wrf_pre_sed_number_ceiling": "if (xni.gt.999.E3)" in wrf,
        "wrf_balance_precedes_updated_ri_ni": wrf.index("!..Cloud ice mass/number balance")
        < wrf.index("!..Update variables for TAU+1 before condensation & sedimention."),
        "wrf_no_nstep_cap": "nstep = MAX(nstep, INT(DT/delta_tp + 1.))" in wrf,
        "port_sedimentation_precedes_finish": port.index("state, precip = _sedimentation")
        < port.index("state = _finish(state)", port.index("state, precip = _sedimentation")),
        "port_nstep_is_clipped": "nstep = jnp.clip(nstep, 1.0, float(NSED_MAX))" in port,
    }
    if not all(assertions.values()):
        raise RootCauseFailure(f"source authority assertion failed: {assertions}")
    tree = subprocess.check_output(
        (
            "git", "-C", str(ROOT), "rev-parse",
            f"{EXPECTED_MODEL_COMMIT}:src/gpuwrf",
        ),
        text=True,
    ).strip()
    if tree != EXPECTED_MODEL_TREE:
        raise RootCauseFailure(f"accepted model tree changed: {tree}")
    ancestor = subprocess.run(
        (
            "git", "-C", str(ROOT), "merge-base", "--is-ancestor",
            EXPECTED_MODEL_COMMIT, "HEAD",
        ),
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RootCauseFailure(
            f"accepted model commit is not ancestral: {EXPECTED_MODEL_COMMIT}"
        )
    return {
        "assertions": assertions,
        "accepted_src_gpuwrf_tree": tree,
        "accepted_src_gpuwrf_commit": EXPECTED_MODEL_COMMIT,
        "accepted_src_gpuwrf_commit_is_ancestor": ancestor,
        "pristine_wrf": authenticated_file(WRF_SOURCE),
        "port_source": authenticated_file(PORT_SOURCE),
        "wrf_line_anchors": {
            "ice_mass_number_balance": "3033-3055",
            "tendency_updated_ri_ni": "3226-3229",
            "ice_fall_speed_and_adaptive_nstep": "3673-3700",
        },
        "port_line_anchors": {
            "nstep_static_clip": "1622-1641",
            "ice_fall_speed_from_raw_state_Ni": "1811-1820",
            "sedimentation_before_finish": "2113-2119",
        },
    }


def _stage_snapshot(state: Any) -> dict[str, Any]:
    return {
        "T": array_stats(state.T),
        "qi": array_stats(state.qi),
        "Ni": array_stats(state.Ni),
    }


def build_proof() -> dict[str, Any]:
    # Imports are intentionally delayed so merely importing this proof helper
    # never initializes JAX.  The executable itself is required to run on CPU.
    import jax
    import jax.numpy as jnp

    from gpuwrf.coupling.physics_couplers import _thompson_column_from_state
    from gpuwrf.physics import thompson_column as tc

    if jax.default_backend() != "cpu":
        raise RootCauseFailure(f"CPU-only proof resolved backend {jax.default_backend()!r}")

    inputs = {
        "exact_result": authenticated_file(EXACT_RESULT, EXPECTED_EXACT_RESULT_SHA256),
        "last_green_input": authenticated_file(INPUT_CARRY, EXPECTED_INPUT_SHA256),
        "first_red_output": authenticated_file(OUTPUT_CARRY, EXPECTED_OUTPUT_SHA256),
    }
    with INPUT_CARRY.open("rb") as stream:
        before_carry = pickle.load(stream)
    with OUTPUT_CARRY.open("rb") as stream:
        after_carry = pickle.load(stream)

    full_column = _thompson_column_from_state(before_carry.state)
    y, x = TARGET_YX
    column = jax.tree_util.tree_map(lambda a: jnp.asarray(a[y, x, :]), full_column)

    public_out, public_precip = tc.step_thompson_column_with_precip(
        column, DT_SECONDS, debug=False
    )
    public_out = jax.tree_util.tree_map(lambda a: np.asarray(a), public_out)
    public_precip = {name: float(np.asarray(value)) for name, value in public_precip.items()}

    state = tc._cast_state(column, tc._work_dtype())
    state = tc._clip_species(state)
    state = tc._reset_mp8_graupel_number(state)
    staged: dict[str, Any] = {"input": _stage_snapshot(state)}
    state = tc._warm_rain_collection(state, DT_SECONDS)
    cold_rates = (
        tc._cold_collection_rates(state, DT_SECONDS, tc.COLD_COLLECTION_TABLES)
        if tc._cold_collection_enabled()
        else tc._zero_cold_collection_rates(state)
    )
    state, graupel_melt, vts_boost, cold_rates = tc._ice_sources_with_process_flags(
        state, DT_SECONDS, cold_collection_rates=cold_rates
    )
    if tc._cold_collection_enabled():
        state = tc._apply_cold_collection_rates(state, DT_SECONDS, cold_rates)
    state, cloud_condensed = tc._saturation_adjustment_with_condensation(
        state, DT_SECONDS
    )
    state = tc._rain_evaporation(
        state,
        DT_SECONDS,
        skip_evaporation=cloud_condensed,
        graupel_melt=graupel_melt,
    )
    staged["immediately_before_sedimentation"] = _stage_snapshot(state)
    pre_sed = state

    speeds = tc._fall_speeds(pre_sed, vts_boost)
    ice_speed = np.maximum(np.asarray(speeds[2]), np.asarray(speeds[3]))
    dz = np.maximum(np.asarray(pre_sed.dz), 1.0)
    raw_nstep = np.floor(DT_SECONDS * ice_speed / dz + 1.0)
    max_nstep_k = int(np.argmax(raw_nstep))

    capped_ice_nstep = tc._nstep_per_column(
        speeds[2], speeds[2], jnp.asarray(dz), DT_SECONDS
    )
    direct_ice_qi, direct_ice_ni, direct_ice_ppt = tc._sed_one_species(
        pre_sed.qi,
        pre_sed.Ni,
        speeds[2],
        speeds[3],
        jnp.asarray(dz),
        pre_sed.rho,
        DT_SECONDS,
        capped_ice_nstep,
    )

    current_sed, current_precip = tc._sedimentation(
        pre_sed, DT_SECONDS, vts_boost=vts_boost
    )
    staged["immediately_after_current_sedimentation"] = _stage_snapshot(current_sed)
    current_melt = tc._instant_melt_freeze(current_sed, DT_SECONDS)
    staged["after_current_instant_melt_freeze"] = _stage_snapshot(current_melt)
    current_finish = tc._finish(current_melt)
    staged["after_current_finish"] = _stage_snapshot(current_finish)

    balanced_ni = wrf_pre_sed_ice_number_balance(
        np.asarray(pre_sed.qi), np.asarray(pre_sed.Ni), np.asarray(pre_sed.rho)
    )
    # Existing _finish contains the same WRF diameter algebra but is currently
    # called too late.  This equality cross-checks the independent literal form.
    existing_late_balance = np.asarray(tc._finish(pre_sed).Ni)
    if not np.array_equal(balanced_ni, existing_late_balance):
        raise RootCauseFailure("literal WRF balance differs from existing late balance")
    balanced_pre_sed = pre_sed.replace(Ni=jnp.asarray(balanced_ni))
    balanced_speeds = tc._fall_speeds(balanced_pre_sed, vts_boost)
    balanced_ice_speed = np.maximum(
        np.asarray(balanced_speeds[2]), np.asarray(balanced_speeds[3])
    )
    balanced_raw_nstep = np.floor(DT_SECONDS * balanced_ice_speed / dz + 1.0)
    balanced_max_k = int(np.argmax(balanced_raw_nstep))
    balanced_sed, balanced_precip = tc._sedimentation(
        balanced_pre_sed, DT_SECONDS, vts_boost=vts_boost
    )
    balanced_finish = tc._finish(
        tc._instant_melt_freeze(balanced_sed, DT_SECONDS)
    )

    rho = np.asarray(pre_sed.rho)
    qi_before = np.asarray(pre_sed.qi)
    current_qi = np.asarray(current_sed.qi)
    balanced_qi = np.asarray(balanced_sed.qi)
    mass_before = float(np.sum(qi_before * rho * dz))
    mass_current = float(np.sum(current_qi * np.asarray(current_sed.rho) * dz))
    mass_balanced = float(np.sum(balanced_qi * np.asarray(balanced_sed.rho) * dz))

    ordinary_ni = float(np.asarray(after_carry.state.Ni)[TARGET_K, y, x])
    replay_ni = float(np.asarray(public_out.Ni)[TARGET_K])
    replay_abs_error = abs(replay_ni - ordinary_ni)
    replay_rel_error = replay_abs_error / abs(ordinary_ni)
    if replay_rel_error > 1.0e-12:
        raise RootCauseFailure(f"Thompson replay did not reproduce ordinary Ni: {replay_rel_error}")
    if int(np.argmax(np.asarray(current_sed.qi))) != TARGET_K:
        raise RootCauseFailure("staged sedimentation explosion moved from target k")
    if not np.array_equal(np.asarray(direct_ice_qi), np.asarray(current_sed.qi)):
        raise RootCauseFailure("direct ice upwind scan differs from full sedimentation qi")
    if not np.array_equal(np.asarray(direct_ice_ni), np.asarray(current_sed.Ni)):
        raise RootCauseFailure("direct ice upwind scan differs from full sedimentation Ni")
    if float(raw_nstep[max_nstep_k]) <= float(tc.NSED_MAX):
        raise RootCauseFailure("raw ice CFL did not exceed the static port cap")
    if float(np.max(balanced_raw_nstep)) > float(tc.NSED_MAX):
        raise RootCauseFailure("source-balanced ice CFL still exceeds the port cap")
    if mass_current / mass_before < 1.0e12:
        raise RootCauseFailure("current sedimentation did not reproduce catastrophic mass creation")
    if abs(mass_balanced - mass_before) > 1.0e-15:
        raise RootCauseFailure("source-balanced sedimentation did not conserve retained column ice")

    ksrc = ICE_SOURCE_K
    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.late-ni-thompson-sedimentation-root-cause.v1",
        "verdict": "LATE_NI_ROOT_CAUSE_THOMPSON_PRESED_ICE_BALANCE_OMITTED",
        "scientific_falsification": True,
        "model_edit_performed": False,
        "gpu_actions": 0,
        "exact_event": {
            "domain": "d01",
            "input_native_step": 1147,
            "output_native_step": 1148,
            "valid_time_utc": "2025-03-01T11:13:12Z",
            "target_yxk": [y, x, TARGET_K],
            "dt_seconds": DT_SECONDS,
        },
        "inputs": inputs,
        "source_authority": source_authority(),
        "public_thompson_reproduction": {
            "ordinary_output_Ni_target": ordinary_ni,
            "cpu_public_thompson_Ni_target": replay_ni,
            "absolute_error": replay_abs_error,
            "relative_error": replay_rel_error,
            "same_target_and_magnitude": replay_rel_error <= 1.0e-12,
            "cpu_public_output": _stage_snapshot(public_out),
            "cpu_public_precip_mm": public_precip,
        },
        "source_stage_ladder": staged,
        "first_catastrophic_operation": {
            "name": "gpuwrf.physics.thompson_column._sed_one_species (ice channel)",
            "outer_stage": "gpuwrf.physics.thompson_column._sedimentation",
            "direct_ice_scan_equals_full_sedimentation_qi_bytes": True,
            "direct_ice_scan_equals_full_sedimentation_Ni_bytes": True,
            "direct_ice_scan_capped_nstep": int(np.asarray(capped_ice_nstep)),
            "direct_ice_scan_pptice_mm": float(np.asarray(direct_ice_ppt)),
            "qi_before_max": float(np.max(qi_before)),
            "qi_before_max_k": int(np.argmax(qi_before)),
            "qi_after_max": float(np.max(current_qi)),
            "qi_after_max_k": int(np.argmax(current_qi)),
            "ice_column_mass_before_kg_m2": mass_before,
            "ice_column_mass_after_kg_m2": mass_current,
            "mass_creation_factor": mass_current / mass_before,
            "current_pptice_mm": float(np.asarray(current_precip["ice"])),
        },
        "causal_chain": {
            "active_source_level_k": ksrc,
            "pre_balance_qi": float(np.asarray(pre_sed.qi)[ksrc]),
            "pre_balance_Ni": float(np.asarray(pre_sed.Ni)[ksrc]),
            "pre_balance_diameter_um": ice_diameter_um(
                np.asarray(pre_sed.qi)[ksrc],
                np.asarray(pre_sed.Ni)[ksrc],
                np.asarray(pre_sed.rho)[ksrc],
            ),
            "balanced_Ni": float(balanced_ni[ksrc]),
            "balanced_diameter_um": ice_diameter_um(
                np.asarray(pre_sed.qi)[ksrc], balanced_ni[ksrc], np.asarray(pre_sed.rho)[ksrc]
            ),
            "filled_down_max_speed_m_s": float(ice_speed[max_nstep_k]),
            "filled_down_max_speed_k": max_nstep_k,
            "filled_down_max_speed_dz_m": float(dz[max_nstep_k]),
            "raw_wrf_nstep": int(raw_nstep[max_nstep_k]),
            "port_static_nstep_cap": int(tc.NSED_MAX),
            "balanced_filled_down_max_speed_m_s": float(
                balanced_ice_speed[balanced_max_k]
            ),
            "balanced_raw_wrf_nstep": int(balanced_raw_nstep[balanced_max_k]),
        },
        "source_balanced_counterfactual": {
            "literal_balance_equals_existing_late_finish_balance_bytes": True,
            "sedimentation_output": _stage_snapshot(balanced_sed),
            "final_output": _stage_snapshot(balanced_finish),
            "ice_column_mass_before_kg_m2": mass_before,
            "ice_column_mass_after_kg_m2": mass_balanced,
            "ice_column_mass_error_kg_m2": mass_balanced - mass_before,
            "pptice_mm": float(np.asarray(balanced_precip["ice"])),
        },
        "root_cause": (
            "The port applies WRF's 5--300 um ice mass/number balance only in the "
            "post-sedimentation finish. Pristine WRF applies it before forming ri/ni "
            "and ice fall speeds. The retained finite qi with near-zero Ni therefore "
            "produces an 856.96 m/s filled-down speed and requires 911 explicit "
            "substeps; the port silently clips that to 16, violating the upwind CFL "
            "and creating 7.93e13 times the column ice mass. Melt/freeze and the late "
            "finish then convert that finite mass catastrophe into the observed Ni."
        ),
        "handoff": {
            "narrow_source_faithful_fix": (
                "Apply module_mp_thompson.F:3033-3055 ice mass/number balance after "
                "ice tendencies and before fall-speed/sedimentation construction; "
                "retain the final balance as source-faithful writeback/defense."
            ),
            "independent_critic": "Opus 4.8 max in a fresh worktree, manager-directed",
            "gpu_needed_for_critic_phase": False,
            "fix_status": "UNIMPLEMENTED_IN_THIS_SPRINT",
        },
    }
    proof["canonical_sha256"] = canonical_sha256(proof)
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    proof = build_proof()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(proof["verdict"])
    print(proof["canonical_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
