"""Release defaults: the validated fp32 fast paths are ON unless the operator opts out.

Applied once at ``import gpuwrf`` BEFORE any submodule reads its import-time
flag, so a run with NO flag file resolves exactly the switch set of
``scripts/bench/flags_lw9.env`` (+ GPUWRF_SPEC_RING_SELECT and the
REAL_ALL pair): same environment,
hence the same AOT cheap keys and the same traced programs as a flags-file run.

Opt-outs: any explicitly set value wins (``GPUWRF_DYN_FP32=0`` keeps the legacy
path for that family); ``GPUWRF_FAST_DEFAULTS=0`` leaves the environment
untouched (all legacy defaults, the pre-v0.3 behaviour).

No imports beyond ``os``: this must run before JAX is imported.
"""

from __future__ import annotations

import os

# Exact resolved switch set of the LW9 release candidate (integrate LW9,
# flags_lw9.env sha 84f310de) plus the in-place ring select (b-carry 48824c974)
# and the WRF REAL dycore + carry pair (S2-DYN / b-carry G10: -9.8 % ordinary root),
# plus the merged LW10 physics levers (MYNN plume sums / tridiag REAL, LW band sums, KISS kernel).
# plus the b-core acoustic (A1c split w, A2 mass blocks, masked uv) and PD species-loop kernels (LW11).
# plus the fused large-step u/v PGF + Coriolis + curvature kernel on nested domains and the rhs_ph REAL
# stencil kernel (b-diff #15, LW12).
# plus the v0.3.1 b-core levers: blocked acoustic w recurrence (A1-SL, bitwise) and the Noah-MP layer lists +
# column kernels (#14, Noah-gate equivalent).
FAST_PATH_DEFAULTS: dict[str, str] = {
    # dynamics
    "GPUWRF_DYN_FP32": "1",
    "GPUWRF_DYN_ADVECTION_FP32": "1",
    "GPUWRF_DYN_DIFFUSION_FP32": "1",
    "GPUWRF_DYN_RK_FP32": "1",
    "GPUWRF_DYN_CARRY_FP32": "1",
    "GPUWRF_DYN_REAL_ALL": "1",  # S2-DYN; paired with CARRY_REAL_ALL (b-carry G10)
    "GPUWRF_CARRY_REAL_ALL": "1",
    "GPUWRF_DYN_PD_FP32": "1",
    "GPUWRF_ACOUSTIC_ALIAS": "1",
    "GPUWRF_ACOUSTIC_W2PASS": "3",  # b-core A1c split w phase (1dc7f2dd6; supersedes A1 =1)
    "GPUWRF_ACOUSTIC_MASS_BLOCK": "1",  # b-core A2 (b774ee8eb)
    "GPUWRF_ACOUSTIC_UV_MASKED": "1",  # b-core (1dc7f2dd6)
    "GPUWRF_DYN_PD_SPECIES_LOOP": "1",  # b-core #16 (6aac4bd5c)
    "GPUWRF_ACOUSTIC_W_RECUR_SL": "1",  # b-core A1-SL (8ee208a92): blocked w recurrence, bitwise (W1b/W1c)
    "GPUWRF_BOUNDARY_FP32": "1",
    "GPUWRF_SPEC_RING_SELECT": "1",
    "GPUWRF_DYN_GLUE_FUSED": "momuvn_rhsph_uvn_pin",  # b-diff #15 rhs_ph stencil + nested-only fused u/v, pinned; BD82 + nested-only u/v momentum advection
    # b-core v0.3.2 P2 row-major layout pins (d02 transposes 441 -> 223), now ALSO on the root step (no "nested"
    # scope; e41-class): unpinned root programs carry the land/surface family x-major and XLA merges its
    # transposes into > 500-instr surface fusions (E41 HLO500, Monica STACK 392); pinned programs are clean.
    "GPUWRF_LAYOUT_PIN": "ac_carry_cols_cum",
    # land surface / surface layer / GWDO
    "GPUWRF_NOAHMP_NATIVE_REAL": "1",
    "GPUWRF_NOAHMP_ITERATION_BARRIER": "1",
    "GPUWRF_NOAHMP_LAYER_SELECT": "1",
    "GPUWRF_NOAHMP_LAYER_LISTS": "1",  # b-core #14 (2cab1a234): snow-water column + SOILWATER on layer lists
    "GPUWRF_NOAHMP_COLUMN_KERNELS": "1",  # b-core #14: canopy/bare Newton, SOILWATER, snow-water column kernels
    "GPUWRF_SFCLAY_NATIVE_REAL": "1",
    "GPUWRF_GWDO_NATIVE_REAL": "1",
    # PBL
    "GPUWRF_EDMF_FUSED_PLUME": "1",
    "GPUWRF_MYNN_FP32_PLUME": "1",
    "GPUWRF_MYNN_FP32_COLUMNS": "1",
    "GPUWRF_MYNN_FP32_BOULAC": "0",
    "GPUWRF_MYNN_CLOUDMIX": "1",
    "GPUWRF_MYNN_PLUME_SUMS": "1",  # b-phys a5f4c0126 (compile LW10b 13.2 gate PASS)
    "GPUWRF_MYNN_TRIDIAG_REAL": "1",
    # cumulus
    "GPUWRF_KF_RESIDENT_TABLES": "1",
    "GPUWRF_KF_COLUMN_FP32": "1",
    # radiation
    "GPUWRF_MCICA_JUMPAHEAD": "1",
    "GPUWRF_MCICA_LEGACY_FP64": "0",
    "GPUWRF_RRTMG_LW_FUSED_TRANSFER": "1",
    "GPUWRF_RRTMG_SW_FUSED_QUADRATURE": "1",
    "GPUWRF_RRTMG_COLUMN_TILE_COLS": "4096",
    "GPUWRF_RRTMG_LW_BAND_SUMS": "1",  # b-phys a5f4c0126
    "GPUWRF_MCICA_KISS_KERNEL": "1",  # BP61 (stream bitwise vs jump-ahead)
    "GPUWRF_RRTMG_SW_BAND_SUMS": "1",  # lever-phys LP02: fused SW band kernel (GPU tier1 + LP02 receipts)
    # microphysics
    "GPUWRF_THOMPSON_FP32": "0",
    "GPUWRF_THOMPSON_NATIVE_REAL": "1",
    "GPUWRF_THOMPSON_FULL_COLUMN": "1",
    "GPUWRF_THOMPSON_COLUMN_SED": "1",
    "GPUWRF_THOMPSON_SED_FP32": "1",
    "GPUWRF_THOMPSON_SED_PREP_FUSED": "0",
    "GPUWRF_THOMPSON_COLUMN_LAYOUT": "1",
    "GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT": "1",
    "GPUWRF_THOMPSON_IMPLICIT_SED": "0",
    # output
    "GPUWRF_HISTORY_INSTEP_DIAG": "1",
    # dispatch: donate the carry into the advance executable (D13: wrfout 6/6 bytes OFF==ON, restart exact)
    "GPUWRF_CARRY_DONATE": "1",
    # v0.3.3 FINAL RC (manager 2026-10-06T07:12:53Z): the merged WRF-fidelity fixes (each merged flag-OFF after its critic).
    # GPUWRF_MYNN_PLUME_CLOUD_BASE (BP91) was OFF until the WRF-faithful MYNN closure set existed; it is ON in the v0.3.3 DELTA
    # below together with SHSM_FLOORS/TKEPROD_UP/ELH_BUDGET/RHOSFC_WRF/CONDENSATION_WRF (MYNNSET72 72 h guard PASS, manager 01:03:20Z).
    "GPUWRF_RRTMG_MAXRAND": "1",  # RE01 McICA cldovrlp=2 maximum-random (integrate)
    "GPUWRF_RRTMG_MP_RE": "1",  # RE01 Thompson radii into RRTMG (b-column)
    "GPUWRF_MYNN_SGS_MIXING_RATIO": "1",  # BP86
    "GPUWRF_MYNN_SCALE_AWARE": "1",  # BP87
    "GPUWRF_W_SURFACE_RESET": "1",  # b-core LL01 set_w_surface
    "GPUWRF_MYNN_SFC_WSPD": "1",  # BP90
    "GPUWRF_SPEC_W_WORK_COPY": "1",  # B44 residual
    "GPUWRF_MYNN_FLTV_WRF": "1",  # BD91
    "GPUWRF_MYNN_DHEAT": "1",  # BP92 early-seed root cause (MYNN heating restore)
    "GPUWRF_MYNN_PSIQ_FLUX_WRF": "1",  # BP95 MYNN-SL flux-loop PSIQ form
    "GPUWRF_W_DAMP_STAGE": "1",  # BD85 WRF w_damp once per RK stage (b-core)
    "GPUWRF_ACOUSTIC_NO_MU_FLOOR": "1",  # ledger row 7: drop the non-WRF dry-mass floor (b-core)
    "GPUWRF_NOAHMP_JULIAN_ADVANCE": "1",  # Noah-MP phenology clock = WRF grid%julian at each LSM call (b-thompson)
    "GPUWRF_NEST_O3_FROM_PARENT": "1",  # nest ozone from the post-step parent (WRF interp_fcn SINT, fid-q2)
    "GPUWRF_ROOT_SCALAR_BDY_RK1": "1",  # BD92 root lateral moist/scalar BC in the RK1 sc_tend (WRF solve_em, b-diff)
    "GPUWRF_THOMPSON_MIXED_PHASE_WRF": "1",  # mixed-phase (1)-(6) + B55 + early exits (b-thompson; cost disclosed)
    "GPUWRF_MYNN_DMP_KTOP_BOUND": "1",  # BP98 WRF F6602 KTOP bound in the DMP shallow-cu SGS overwrite (b-phys)
    "GPUWRF_MYNN_ELB_MF": "1",  # BP99 WRF mym_length CASE1 elb_mf (b-phys)
    "GPUWRF_MYNN_PHY_EXNER": "1",  # BP96 WRF pi_phy (P+PB) into every MYNN column consumer (b-phys)
    "GPUWRF_MYNN_PSIG_CLAMP": "1",  # BP100 WRF SCALE_AWARE Psig_bl/Psig_shcu clamp to [0,1] (b-phys)
    "GPUWRF_MYNN_QNI_MIXING": "1",  # BP101 WRF MYNN mixes Thompson qni (bl_mynn_cloudmix/mixscalars default 1, b-phys)
    # v0.3.3 release DELTA (manager 2026-10-06T08:53:26Z, validated at wave end):
    "GPUWRF_MYNN_SFC_PBLH_WRF": "1",  # BP102 WRF SF_mynn gust uses the supplied PBLH unfloored (b-phys)
    "GPUWRF_KF_REAL_CONSTANTS": "1",  # BD94-B4 WRF REAL ALIQ/GDRY in the native KF column (b-column)
    "GPUWRF_WRITER_RVOVRD_REAL": "1",  # BD94-B5 WRF REAL RVOVRD in the wrfout writer diagnostics (cadence-out)
    "GPUWRF_PHY_PREP_REAL_CONST": "1",  # BP103 WRF REAL(r_v)/REAL(r_d) in the native PHY_PREP consumers (b-phys)
    "GPUWRF_THOMPSON_WRF_CONSTANTS": "1",  # BD94 WRF REAL Thompson constants (AM_R/AM_I/D0I + WGAMMA products) on the native REAL path (b-thompson)
    "GPUWRF_MYNN_REAL_CONSTANTS": "1",  # BD94-A/B1 13 WRF REAL MYNN/SL/Noah PARAMETER values (P608, EP1, CP, R_D, ...; b-phys)
    "GPUWRF_SOLAR_JULIAN_WRF": "1",  # X5/BP106 WRF fractional 0-based grid%julian for declination/EoT at the radiation call (b-phys)
    "GPUWRF_NEST_SCALAR_SPEC_FINAL": "1",  # BD95 WRF spec_bdy_final re-pins every nest moist/scalar spec zone each step (b-diff)
    "GPUWRF_RRTMG_REAL_CONSTANTS": "1",  # BP105 WRF REAL RRTMG constants (BPADE, grav 9.8066, heatfac) + model g 9.81 cloud mass (b-phys)
    "GPUWRF_KF_PHYD_WRF": "1",  # 0227 NAMED BUG: KF gets WRF P_HYD (+ independent REAL pi, no-trigger NCA kept) like KF_ETA_CPS (b-column)
    "GPUWRF_PHYS_TEND_RK_WRF": "1",  # D2/D3 WRF physics->RK coupling of PBL moist tendencies + held radiation rate (b-diff, main 150628ddf)
    "GPUWRF_KF_TEND_RK_WRF": "1",  # D1 KF coupled tendencies through RK, held-rate lifetime (b-column, main 0a7a4a72c)
    "GPUWRF_MYNN_PLUME_CLOUD_BASE": "1",  # BP91 WRF DMP plume cloud-base geometry (b-phys; with the closure set below)
    "GPUWRF_MYNN_SHSM_FLOORS": "1",  # BP113 WRF mym_turbulence Sm/Sh + MF floors (b-phys)
    "GPUWRF_MYNN_TKEPROD_UP": "1",  # BP113 WRF TKEprod_up from DMP_mf with el_prev (b-phys)
    "GPUWRF_MYNN_ELH_BUDGET": "1",  # BP113 WRF elh in pdq/pdt (b-phys)
    "GPUWRF_MYNN_RHOSFC_WRF": "1",  # BP113 WRF surface density from actual PSFC (b-phys)
    "GPUWRF_MYNN_CONDENSATION_WRF": "1",  # BP113 WRF condensation_edmf QC carry + early exit (b-phys 758466507)
    # v0.3.3 W2 fix wave (station DAMAGE R32-60 -> W2; manager 2026-10-07T06:31:32Z: ONE W2 defaults commit LAST, 8 keys ON):
    "GPUWRF_NOAH_URBAN_SOIL_PARAMS": "1",  # W2 WRF hard-coded URBAN soil SMCMAX/SMCREF/SMCWLT/SMCDRY/CSOIL before FRZX (b-thompson 477fbe93c)
    "GPUWRF_LAND_Q2_CAP_POSTLSM": "1",  # W2 WRF land Q2 cap after the Noah-MP overwrite, entry QV (b-phys b4b36bcc6)
    "GPUWRF_HISTORY_SWDOWN_HORIZONTAL": "1",  # W2 WRF horizontal SWDOWN history on slopes (mass-sol 4f00d82d3)
    "GPUWRF_MYNN_PREDICT_SAW_FLOOR": "1",  # W2 WRF mym_predict plume diffusion floors (b-phys 33d059a3e)
    "GPUWRF_NOAH_CANWATER_ELAI": "1",  # W2 WRF CANWATER capacities from PHENOLOGY ELAI/ESAI/FVEG (b-thompson cce6dab19)
    "GPUWRF_MYNN_SOURCE_PARITY2": "1",  # W2 WRF REAL DMP GTR + MEAN floor face extent (b-phys 778aeee64)
    "GPUWRF_MYNN_SFC_PSI_REAL": "1",  # W2 WRF REAL psi tables + stable-heat reciprocal in MYNN-SL (b-phys 36fe022a0)
    "GPUWRF_NOAH_QSFC_WRF": "1",  # W2 WRF grid%QSFC = BARE_FLUX QSFC on land (b-thompson 3814b53fc)
    # v0.3.3 W3: original WRF REAL reftra_sw PIFM + conservative SSA (approved 5ab322a76).
    "GPUWRF_RRTMG_SW_REFTRA_WRF": "1",
    # run-mode values the flag file pins (kept identical for cheap-key parity)
    "GPUWRF_CENSUS": "0",
    "JAX_ENABLE_X64": "true",
    "JAX_PALLAS_USE_MOSAIC_GPU": "false",
}

FAST_DEFAULTS_STATUS: dict[str, object] = {}


def apply_fast_path_defaults() -> dict[str, object]:
    """Set every unset release switch; never overwrite an operator value."""

    FAST_DEFAULTS_STATUS.clear()
    if os.environ.get("GPUWRF_FAST_DEFAULTS", "1").strip().lower() in {"0", "false", "no", "off"}:
        FAST_DEFAULTS_STATUS.update(enabled=False, applied={}, kept={})
        return FAST_DEFAULTS_STATUS
    applied: dict[str, str] = {}
    kept: dict[str, str] = {}
    for name, value in FAST_PATH_DEFAULTS.items():
        if name in os.environ:
            kept[name] = os.environ[name]
        else:
            os.environ[name] = value
            applied[name] = value
    FAST_DEFAULTS_STATUS.update(enabled=True, applied=applied, kept=kept)
    return FAST_DEFAULTS_STATUS
