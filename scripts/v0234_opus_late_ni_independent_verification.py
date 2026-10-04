#!/usr/bin/env python3
"""Independent Opus verification of the v0234 late-Ni Thompson ice-balance root cause.

This is deliberately NOT a re-run of the predecessor's RCA script.  The WRF
formulas below are transcribed directly from pristine
``module_mp_thompson.F`` by this sprint's owner, and the failure quantities
(fall speed, substep count, mass amplification) are recomputed from first
principles rather than read back from the predecessor's proof JSON.  The
predecessor's numbers are used only as *expected* values to agree or disagree
with, and every disagreement is reported.

Run (CPU only):
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    taskset -c 13,14,15,29,30,31 nice -n 15 ionice -c 3 \
    python scripts/v0234_opus_late_ni_independent_verification.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --- Retained exact carries from the authorized GPU discriminator arm ---------
EXACT_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_gpt_late_ni_df242a850d6ebab1_exact"
)
INPUT_CARRY = EXACT_DIR / "last-green-advance-input-d01-step-1147.pkl"
EXPECTED_INPUT_SHA256 = "0bfc31891eb1da86326a2f5965cfdb8eb981a131d8c7c6a13ad69d2b3acb8414"

TARGET_YX = (48, 40)
DT_SECONDS = 54.0

# --- WRF constants, read by me from pristine module_mp_thompson.F ------------
# mu_i = 0.0                (line 105)
# bm_i = 3.0                (line 138)
# am_i = PI*rho_i/6.0       (line 137), rho_i = 890 kg/m3
# R1 = 1.E-12               (line 183)
# R2 = 1.E-6                (line 184)
# cie(2) = bm_i+mu_i+1 = 4  (line 688)
# cig(1) = WGAMMA(mu_i+1) = Gamma(1) = 1        (line 694)
# cig(2) = WGAMMA(bm_i+mu_i+1) = Gamma(4) = 6   (line 695)
# oig1 = 1/cig(1) = 1, oig2 = 1/cig(2) = 1/6    (lines 701-702)
# obmi = 1/bm_i = 1/3                           (line 703)
WRF_R1 = 1.0e-12
WRF_R2 = 1.0e-6
WRF_MU_I = 0.0
WRF_BM_I = 3.0
WRF_RHO_I = 890.0
WRF_AM_I = np.pi * WRF_RHO_I / 6.0
WRF_CIE2 = WRF_BM_I + WRF_MU_I + 1.0  # 4.0
WRF_CIG1 = 1.0  # Gamma(1)
WRF_CIG2 = 6.0  # Gamma(4)
WRF_OIG1 = 1.0 / WRF_CIG1
WRF_OIG2 = 1.0 / WRF_CIG2
WRF_OBMI = 1.0 / WRF_BM_I
WRF_NI_CEILING = 999.0e3  # 999 per cm^3 == 500 xtals/litre comment, line 3034
WRF_DI_MIN = 5.0e-6
WRF_DI_MAX = 300.0e-6

# Predecessor's claims, held here only as expectations to test.
CLAIMED = {
    "filled_down_max_speed_m_s": 856.9557946103141,
    "raw_wrf_nstep": 911,
    "port_static_nstep_cap": 16,
    "balanced_filled_down_max_speed_m_s": 0.6391697902895398,
    "balanced_raw_wrf_nstep": 1,
    "pre_balance_diameter_um": 402219.7893092471,
    "balanced_diameter_um": 300.0000000000001,
    "balanced_Ni": 4138.382776150909,
    "mass_creation_factor": 79329183651091.42,
    "active_source_level_k": 19,
}


class VerificationFailure(RuntimeError):
    """Raised when an independent check contradicts the claimed root cause."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def wrf_ice_balance_number_density(ri: np.ndarray, ni: np.ndarray) -> np.ndarray:
    """Literal transcription of module_mp_thompson.F:3033-3055.

    Takes the *post-tendency* ice mass density ``ri`` (kg m^-3) and number
    density ``ni`` (m^-3) and returns the balanced number density that WRF's
    modified ``niten`` produces.  WRF writes the balance as a tendency
    correction ``niten = (xni - ni1d*rho)*odts*orho``; since ``odts = 1/dtsave``
    the post-tendency number density is exactly ``xni``, so returning ``xni`` is
    an exact restatement, not an approximation.
    """

    xri = np.maximum(WRF_R1, ri)
    xni = np.maximum(WRF_R2, ni)

    lami = (WRF_AM_I * WRF_CIG2 * WRF_OIG1 * xni / xri) ** WRF_OBMI
    xdi = WRF_CIE2 / lami

    lami_small = WRF_CIE2 / WRF_DI_MIN
    xni_small = np.minimum(
        WRF_NI_CEILING, WRF_CIG1 * WRF_OIG2 * xri / WRF_AM_I * lami_small**WRF_BM_I
    )
    lami_large = WRF_CIE2 / WRF_DI_MAX
    xni_large = WRF_CIG1 * WRF_OIG2 * xri / WRF_AM_I * lami_large**WRF_BM_I

    out = np.where(xdi < WRF_DI_MIN, xni_small, np.where(xdi > WRF_DI_MAX, xni_large, xni))
    # `else` branch of the `if (xri .gt. R1)` test: niten = -ni1d*odts -> post = 0.
    out = np.where(xri > WRF_R1, out, 0.0)
    # Trailing ceiling, lines 3053-3055.
    return np.minimum(np.maximum(out, 0.0), WRF_NI_CEILING)


def wrf_ice_fall_speeds(rhof: np.ndarray, ri: np.ndarray, ni: np.ndarray, active: np.ndarray):
    """Literal transcription of module_mp_thompson.F:3678-3691 with fill-down.

    Returns (vtik, vtnik) after WRF's `vtik(k) = vtik(k+1)` fill-down, which
    runs top->bottom.  Index 0 here is the surface (port convention), so the
    fill-down walks from the last index toward index 0.
    """

    from gpuwrf.physics.thompson_column import AV_I, BV_I, CIG3, CIG6, CIG7, OIG2

    lami = (WRF_AM_I * WRF_CIG2 * WRF_OIG1 * ni / ri) ** WRF_OBMI
    ilami = 1.0 / lami
    vtik = rhof * AV_I * CIG3 * OIG2 * ilami**BV_I
    vtnik = rhof * AV_I * CIG6 / CIG7 * ilami**BV_I

    vtik = np.where(active, vtik, 0.0)
    vtnik = np.where(active, vtnik, 0.0)
    for k in range(len(vtik) - 2, -1, -1):
        if not active[k]:
            vtik[k] = vtik[k + 1]
            vtnik[k] = vtnik[k + 1]
    return vtik, vtnik


def wrf_adaptive_nstep(vt: np.ndarray, dz: np.ndarray, dt: float) -> int:
    """module_mp_thompson.F:3693-3697 — nstep = MAX_k INT(DT/(dzq/vt) + 1).

    Only levels with `vtik(k) > 1.E-3` participate, and WRF applies NO cap.
    """

    nstep = 0
    for k in range(len(vt)):
        if vt[k] > 1.0e-3:
            delta_tp = dz[k] / vt[k]
            nstep = max(nstep, int(dt / delta_tp + 1.0))
    return nstep


def stats(value: Any) -> dict[str, Any]:
    arr = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(arr)
    return {
        "shape": list(arr.shape),
        "nonfinite_count": int((~finite).sum()),
        "finite_max": float(arr[finite].max()) if finite.any() else None,
        "finite_min": float(arr[finite].min()) if finite.any() else None,
        "finite_argmax_flat": int(np.argmax(np.where(finite, arr, -np.inf))) if finite.any() else None,
    }


def agree(name: str, got: float, expected: float, rtol: float, disagreements: list) -> dict:
    if expected == 0.0:
        ok = abs(got) <= rtol
        rel = abs(got)
    else:
        rel = abs(got - expected) / abs(expected)
        ok = rel <= rtol
    if not ok:
        disagreements.append({"quantity": name, "independent": got, "claimed": expected, "rel": rel})
    return {"independent": got, "claimed": expected, "relative_difference": rel, "agrees": bool(ok)}


def build_verification() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from gpuwrf.coupling.physics_couplers import _thompson_column_from_state
    from gpuwrf.physics import thompson_column as tc

    if jax.default_backend() != "cpu":
        raise VerificationFailure(f"CPU-only verification resolved backend {jax.default_backend()!r}")

    input_sha = sha256_file(INPUT_CARRY)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise VerificationFailure(f"retained step-1147 carry SHA mismatch: {input_sha}")

    with INPUT_CARRY.open("rb") as stream:
        carry = pickle.load(stream)

    full_column = _thompson_column_from_state(carry.state)
    y, x = TARGET_YX
    column = jax.tree_util.tree_map(lambda a: jnp.asarray(a[y, x, :]), full_column)

    disagreements: list = []
    report: dict[str, Any] = {
        "schema": "gpuwrf.v0234.opus-late-ni-independent-verification.v1",
        "inputs": {
            "last_green_input": {"path": str(INPUT_CARRY), "sha256": input_sha},
            "target_yx": list(TARGET_YX),
            "dt_seconds": DT_SECONDS,
        },
        "gpu_actions": 0,
    }

    # --- G2/G4: what does the production kernel do on this column TODAY? -----
    # This script is re-runnable before and after the fix.  The physics
    # derivations below never call the (possibly fixed) _fall_speeds, so the
    # root-cause evidence is reproducible either way; only this section reports
    # the live kernel state.
    fix_present = hasattr(tc, "_balance_ice_number")
    public_out, _ = tc.step_thompson_column_with_precip(column, DT_SECONDS, debug=False)
    public_out = jax.tree_util.tree_map(np.asarray, public_out)
    catastrophic = bool(np.max(public_out.Ni) > 1.0e9)
    report["g2_g4_public_kernel_state"] = {
        "fix_present_in_tree": fix_present,
        "Ni": stats(public_out.Ni),
        "qi": stats(public_out.qi),
        "T": stats(public_out.T),
        "catastrophic": catastrophic,
        "interpretation": (
            "G4 GREEN: fix present and the exact failing column is finite and physical"
            if fix_present and not catastrophic
            else "G2 GREEN: unfixed tree reproduces the late-Ni catastrophe"
            if not fix_present and catastrophic
            else "INCONSISTENT"
        ),
    }
    if fix_present and catastrophic:
        raise VerificationFailure("fix is present but the catastrophe still reproduces")
    if not fix_present and not catastrophic:
        raise VerificationFailure("unfixed kernel did NOT reproduce the late-Ni catastrophe")

    # --- Walk the operator split to the pre-sedimentation state ---------------
    state = tc._cast_state(column, tc._work_dtype())
    state = tc._clip_species(state)
    state = tc._reset_mp8_graupel_number(state)
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
    state, cloud_condensed = tc._saturation_adjustment_with_condensation(state, DT_SECONDS)
    pre_sed = tc._rain_evaporation(
        state, DT_SECONDS, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt
    )

    qi = np.asarray(pre_sed.qi, dtype=np.float64)
    Ni = np.asarray(pre_sed.Ni, dtype=np.float64)
    rho = np.asarray(pre_sed.rho, dtype=np.float64)
    dz = np.maximum(np.asarray(pre_sed.dz, dtype=np.float64), 1.0)

    report["pre_sedimentation_state"] = {
        "qi": stats(qi),
        "Ni": stats(Ni),
        "all_finite": bool(np.isfinite(qi).all() and np.isfinite(Ni).all()),
    }
    if not report["pre_sedimentation_state"]["all_finite"]:
        raise VerificationFailure("pre-sedimentation state already nonfinite; mechanism differs")

    # --- G1/G3: my own WRF-literal fall-speed and nstep derivation ------------
    ri = np.maximum(qi * rho, WRF_R1)
    ni_raw = np.maximum(Ni * rho, WRF_R2)
    active_ice = qi > WRF_R1
    rhof = np.sqrt(np.asarray(tc.RHO_NOT, dtype=np.float64) / np.maximum(rho, WRF_R1))

    vtik_raw, _ = wrf_ice_fall_speeds(rhof, ri, ni_raw, active_ice)
    nstep_raw = wrf_adaptive_nstep(vtik_raw, dz, DT_SECONDS)

    ksrc = int(np.argmax(qi))
    lami_src = (WRF_AM_I * WRF_CIG2 * WRF_OIG1 * ni_raw[ksrc] / ri[ksrc]) ** WRF_OBMI
    xdi_src_um = (WRF_CIE2 / lami_src) * 1.0e6

    ni_bal = wrf_ice_balance_number_density(ri, ni_raw)
    vtik_bal, _ = wrf_ice_fall_speeds(rhof, ri, np.maximum(ni_bal, WRF_R2), active_ice)
    nstep_bal = wrf_adaptive_nstep(vtik_bal, dz, DT_SECONDS)
    lami_bal = (WRF_AM_I * WRF_CIG2 * WRF_OIG1 * max(ni_bal[ksrc], WRF_R2) / ri[ksrc]) ** WRF_OBMI
    xdi_bal_um = (WRF_CIE2 / lami_bal) * 1.0e6

    report["g1_g3_independent_derivation"] = {
        "active_source_level_k": agree(
            "active_source_level_k", float(ksrc), float(CLAIMED["active_source_level_k"]), 0.0, disagreements
        ),
        "unbalanced_max_ice_fall_speed_m_s": agree(
            "filled_down_max_speed_m_s",
            float(vtik_raw.max()),
            CLAIMED["filled_down_max_speed_m_s"],
            1e-9,
            disagreements,
        ),
        "unbalanced_raw_wrf_nstep": agree(
            "raw_wrf_nstep", float(nstep_raw), float(CLAIMED["raw_wrf_nstep"]), 0.0, disagreements
        ),
        "unbalanced_source_diameter_um": agree(
            "pre_balance_diameter_um", xdi_src_um, CLAIMED["pre_balance_diameter_um"], 1e-9, disagreements
        ),
        # The predecessor's `balanced_filled_down_max_speed_m_s` is in fact the
        # speed at the ice SOURCE level k19, not the column maximum.  Two layers
        # (k19, k20) both balance to exactly 300 um; at a fixed diameter the
        # speed scales with rhof = sqrt(rho_not/rho), so the higher, thinner-air
        # layer k20 is marginally faster and owns the true column max.  Both are
        # reported here.  This is a LABELLING correction: nstep is 1 either way
        # and no causal conclusion changes.
        "balanced_source_level_ice_fall_speed_m_s": agree(
            "balanced_speed_at_source_level_k19",
            float(vtik_bal[ksrc]),
            CLAIMED["balanced_filled_down_max_speed_m_s"],
            1e-9,
            disagreements,
        ),
        "balanced_true_column_max_ice_fall_speed_m_s": float(vtik_bal.max()),
        "balanced_true_column_max_level_k": int(np.argmax(vtik_bal)),
        "balanced_raw_wrf_nstep": agree(
            "balanced_raw_wrf_nstep", float(nstep_bal), float(CLAIMED["balanced_raw_wrf_nstep"]), 0.0, disagreements
        ),
        "balanced_source_diameter_um": agree(
            "balanced_diameter_um", xdi_bal_um, CLAIMED["balanced_diameter_um"], 1e-9, disagreements
        ),
        "balanced_source_Ni_per_kg": agree(
            "balanced_Ni", float(ni_bal[ksrc] / rho[ksrc]), CLAIMED["balanced_Ni"], 1e-9, disagreements
        ),
        "port_static_nstep_cap": int(tc.NSED_MAX),
        "cfl_violation_factor": float(nstep_raw) / float(tc.NSED_MAX),
        "surface_dz_m": float(dz[0]),
        "unbalanced_source_Ni_per_kg": float(Ni[ksrc]),
        "source_qi_kg_kg": float(qi[ksrc]),
    }

    # --- Mass amplification through the port's own sedimentation -------------
    _qi_c, _ni_c, _ppt_c = tc._sed_one_species(
        pre_sed.qi,
        pre_sed.Ni,
        jnp.asarray(vtik_raw),
        jnp.asarray(vtik_raw),
        jnp.asarray(dz),
        pre_sed.rho,
        DT_SECONDS,
        tc._nstep_per_column(jnp.asarray(vtik_raw), jnp.asarray(vtik_raw), jnp.asarray(dz), DT_SECONDS),
    )
    mass_before = float(np.sum(qi * rho * dz))
    mass_after_unbalanced = float(np.sum(np.asarray(_qi_c, dtype=np.float64) * rho * dz))
    report["mass_amplification_unbalanced"] = {
        "ice_column_mass_before_kg_m2": mass_before,
        "ice_column_mass_after_kg_m2": mass_after_unbalanced,
        "factor": agree(
            "mass_creation_factor",
            mass_after_unbalanced / mass_before,
            CLAIMED["mass_creation_factor"],
            1e-6,
            disagreements,
        ),
    }

    # --- Domain-wide incidence: how isolated is this column? ------------------
    full_qi = np.asarray(full_column.qi, dtype=np.float64)
    full_Ni = np.asarray(full_column.Ni, dtype=np.float64)
    full_rho = np.asarray(full_column.rho, dtype=np.float64)
    ri_dom = np.maximum(full_qi * full_rho, WRF_R1)
    ni_dom = np.maximum(full_Ni * full_rho, WRF_R2)
    lami_dom = (WRF_AM_I * WRF_CIG2 * WRF_OIG1 * ni_dom / ri_dom) ** WRF_OBMI
    xdi_dom = WRF_CIE2 / lami_dom
    unbalanced = (full_qi > WRF_R1) & ((xdi_dom > WRF_DI_MAX) | (xdi_dom < WRF_DI_MIN))
    report["domain_incidence_at_step_1147_input"] = {
        "cells_total": int(full_qi.size),
        "cells_with_ice": int((full_qi > WRF_R1).sum()),
        "cells_outside_5_300um_band": int(unbalanced.sum()),
        "columns_outside_band": int(unbalanced.any(axis=-1).sum()),
        "max_out_of_band_diameter_um": float(np.max(np.where(unbalanced, xdi_dom, 0.0)) * 1.0e6),
        "note": "measured on the retained step-1147 INPUT state (before this step's tendencies)",
    }

    report["disagreements_with_predecessor"] = disagreements
    report["predecessor_corrections"] = [
        {
            "field": "causal_chain.balanced_filled_down_max_speed_m_s",
            "claimed": CLAIMED["balanced_filled_down_max_speed_m_s"],
            "corrected_true_column_max": float(vtik_bal.max()),
            "class": "LABELLING_CORRECTION_NOT_MECHANISM_CHANGE",
            "detail": (
                "0.63917 m/s is the balanced speed at the ice source level k19, not "
                "the filled-down column maximum (0.66098 m/s at k20). Both give WRF "
                "nstep=1, so the root cause, the fix and every causal conclusion are "
                "unaffected."
            ),
        }
    ]
    report["verdict"] = (
        "OPUS_INDEPENDENT_CONFIRMATION_OF_THOMPSON_PRESED_ICE_BALANCE_OMISSION"
        if not disagreements
        else "OPUS_INDEPENDENT_DISAGREEMENT"
    )
    report["canonical_sha256"] = canonical_sha256(report)
    return report


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / (
        ".agent/sprints/2026-07-21-v0234-opus-late-ni-fix/OPUS_INDEPENDENT_VERIFICATION.json"
    )
    try:
        report = build_verification()
    except VerificationFailure as exc:
        print(f"RED: {exc}", file=sys.stderr)
        return 1
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    print(f"verdict: {report['verdict']}")
    return 0 if not report["disagreements_with_predecessor"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
