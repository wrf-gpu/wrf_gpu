"""v0.6.0 Registry-driven physics field list: the single source of truth.

This module is an interface-freeze artifact only. It allocates no arrays and
implements no physics kernels. It mirrors the relevant WRF
``Registry/Registry.EM_COMMON`` package memberships for the v0.6.0 accepted
physics menu so that scheme lanes, wrfout, restart, and nesting consume one
field list instead of re-creating Thompson-era assumptions locally.

WRF Registry lines verified against
``/home/user/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON`` on
2026-06-02:

* Kessler(1): ``moist:qv,qc,qr``.
* Purdue-Lin(2): ``moist:qv,qc,qr,qi,qs,qg`` (single-moment, no scalar number
  species; graupel via F_QG=.true.).
* WSM3(3): ``moist:qv,qc,qr;state:re_cloud,re_ice,re_snow``.
* WSM5(4): ``moist:qv,qc,qr,qi,qs;state:re_cloud,re_ice,re_snow``.
* WSM6(6): ``moist:qv,qc,qr,qi,qs,qg;state:re_cloud,re_ice,re_snow``.
* Thompson(8): ``moist:qv,qc,qr,qi,qs,qg;scalar:qni,qnr;state:re_*``.
* Goddard-GCE(97): ``moist:qv,qc,qr,qi,qs,qg`` (single-moment 3-ice graupel,
  ``gsfcgcescheme``; the operational ``ihail=0``/``ice2=0``/``itaobraun=1``/
  ``new_ice_sat=2`` call of ``phys/module_mp_gsfcgce.F:gsfcgce``). NO scalar
  number species and NO extra prognostic state -> a no-kernel-change endpoint on
  the existing moist substrate. (WRF v4 ``mp_physics=7`` is the SEPARATE 4-ice
  NUWRF scheme ``nuwrf4icescheme`` with a hail class + a large diagnostic state
  array -- NOT this, and NOT yet ported.)
* Thompson aerosol-aware(28): ``moist:qv..qg;scalar:qnc,qnr,qni,qnwfa,qnifa``
  (WRF Registry package ``thompsonaero``; nwfa2d/nifa2d surface emission).
* Morrison(10): ``moist:qv..qg;scalar:qni,qns,qnr,qng`` plus cuten state.
* WDM6(16): ``moist:qv..qg;scalar:qnn,qnc,qnr;state:re_*``.
* Hail family (v0.17 ADR-032 substrate; schemes NOT yet implemented):
  WSM7(24)/Goddard-4ice(7)/WDM7(26)/UDM(27) add ``moist:qh``;
  Thompson-graupel/hail(38) adds ``scalar:qng,qvolg``; NSSL(17-22) add
  ``moist:qh;scalar:qnh`` (nssl_hail) and ``scalar:qvolg,qvolh`` (predicted
  density). These appear here for reference; the State leaves
  ``qh``/``Nh``/``qvolg``/``qvolh`` exist, but the schemes stay fail-closed.
* MYJ(2): ``state:tke_pbl,el_pbl``; requires Janjic Eta sfclay(2).
* MYNN(5): ``scalar:qke_adv;state:qke,tke_pbl,sh3d,sm3d,tsq,qsq,cov,el_pbl``.
* BouLac(8): ``state:qke`` reused as the prognostic TKE storage plus PBLH/K
  diagnostics (frozen-contract extension, 2026-06-04).
* QNSE(4), TEMF(10), EEPS(16), and KEPS(17): v0.18 reference-only PBL
  endpoints with fp64 pristine-WRF oracle savepoints staged under
  ``proofs/v018/savepoints_fp64``; accepted for isolated oracle comparison and
  fail-closed in the operational scan until traceable JAX kernels land.
* CAM-UW(9): CAM5 vertical-diffusion stack, v0.22 scan-wired as a traceable
  UW diagnostic-TKE / implicit vertical-diffusion endpoint for the operational
  State fields. Full CAM cloud-number/sedimentation/residual-stress parity
  still requires a pristine-WRF CAM savepoint fixture and is documented in the
  v0.22 proof object.
* Noah classic(2): ``state:flx4,fvb,fbur,fgsn,smcrel,xlaidyn``.
* Cumulus options KF(1), BMJ(2), Grell-Freitas(3), Tiedtke(6/16), and the
  reference-only long tail with real WRF oracle artifacts
  (4/5/14/93/94/95/96/99) use the common ``R*CUTEN`` tendency family where the
  WRF driver exposes it and scheme-specific carry listed below. CU7/10/11 remain
  oracle-needed and are not accepted by this registry.

Append-only State rule:
    Existing ``State.__slots__`` order is preserved. v0.6.0 additive dycore
    leaves are frozen as ``Nc``, ``Nn``, and ``rainc_acc``; v0.17 ADR-032
    appends the graupel/hail substrate ``qh``/``Nh``/``qvolg``/``qvolh`` AFTER
    them (and after the v0.15 MYNN leaves) at the very END of
    ``State.__slots__`` / ``STATE_FIELD_ORDER`` / ``PRECISION_MATRIX``; v0.16
    aerosol-aware Thompson (mp=28) then appends ``nwfa``/``nifa`` AFTER the hail
    substrate as the new tail (restart schema bumped deliberately). Lanes that
    need new leaves must append them the same way. Land/PBL/cumulus save-state
    fields stay in ``PhysicsCarry`` sibling trees, following the existing
    Noah-MP carry pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


PHYSICS_REGISTRY_VERSION = "v0.6.0-S0-frozen-2026-06-04-consolidation3-bmj2-extension+v017-qh-hail-substrate+v016-thompson-aero"


@dataclass(frozen=True)
class SchemeOption:
    """One accepted namelist option with its WRF Registry package source."""

    key: str
    option: int
    name: str
    wrf_package: str
    status: str
    owner_family: str


ACCEPTED_MP_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 24, 26, 28, 97)
ACCEPTED_BL_PBL_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 99)
ACCEPTED_SF_SFCLAY_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 5, 7, 91)
ACCEPTED_CU_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 14, 16, 93, 94, 95, 96, 99)
ACCEPTED_SF_SURFACE_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 4, 7, 8)
# ra_sw=3/5/7/99 (CAM/Goddard-new/FLG/GFDL-Eta) are v0.18 reference-only:
# accepted so real-WRF oracle work can select their exact WRF radiation-driver
# path; they fail-close in the operational scan until faithful traceable JAX
# kernels pass those real oracles.
ACCEPTED_RA_SW_PHYSICS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 7, 99)
# ra_lw=5 (GSFC/Goddard NUWRF LW) is a v0.13 Tier-3 reference-only scheme: a
# single-column fp64 pristine-WRF oracle is staged (proofs/v013/oracle/
# radiation_lw), but its traceable JAX column kernel is a documented carry-over
# (the combined NUWRF SW+LW module is ~12.5k LOC / ~11.8k LW coefficients), so it
# is "accepted" (selectable for a single-column reference comparison) and
# fail-closes in the operational scan -- NOT in _SCAN_WIRED_OPTIONS["ra_lw_physics"].
# ra_lw=3/7/99 (CAM/FLG/GFDL-Eta) are v0.18 reference-only real-WRF oracle
# endpoints; they fail-close in the operational scan until ported and parity-proven.
# ra_lw=31 (Held-Suarez idealized radiation) is a v0.17 no-kernel-change endpoint
# port (phys/module_ra_hs.F:HSRAD): a combined LW+SW Newtonian-relaxation forcing
# selected through the LW slot (WRF Registry heldsuarez==31), carrying NO
# prognostic state. Its JAX column kernel (physics.ra_lw_hs) is savepoint-parity-
# proven against the unmodified WRF source at fp64 (proofs/v017/
# held_suarez_lw_savepoint_parity.json) and scan-wired via held_suarez_theta_tendency.
ACCEPTED_RA_LW_PHYSICS: tuple[int, ...] = (0, 1, 3, 4, 5, 7, 31, 99)

ACCEPTED_NAMELIST_OPTIONS: Mapping[str, tuple[int, ...]] = {
    "mp_physics": ACCEPTED_MP_PHYSICS,
    "bl_pbl_physics": ACCEPTED_BL_PBL_PHYSICS,
    "sf_sfclay_physics": ACCEPTED_SF_SFCLAY_PHYSICS,
    "cu_physics": ACCEPTED_CU_PHYSICS,
    "sf_surface_physics": ACCEPTED_SF_SURFACE_PHYSICS,
    "ra_sw_physics": ACCEPTED_RA_SW_PHYSICS,
    "ra_lw_physics": ACCEPTED_RA_LW_PHYSICS,
}

MP_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("mp_physics", 0, "disabled/passive qv", "passiveqv", "accepted", "microphysics"),
    1: SchemeOption("mp_physics", 1, "Kessler warm rain", "kesslerscheme", "accepted", "microphysics"),
    2: SchemeOption("mp_physics", 2, "Purdue-Lin", "linscheme", "accepted", "microphysics"),
    3: SchemeOption("mp_physics", 3, "WSM3 simple ice", "wsm3scheme", "accepted", "microphysics"),
    4: SchemeOption("mp_physics", 4, "WSM5", "wsm5scheme", "accepted", "microphysics"),
    6: SchemeOption("mp_physics", 6, "WSM6", "wsm6scheme", "accepted", "microphysics"),
    8: SchemeOption("mp_physics", 8, "Thompson", "thompson", "implemented", "microphysics"),
    10: SchemeOption("mp_physics", 10, "Morrison two-moment", "morr_two_moment", "accepted", "microphysics"),
    13: SchemeOption("mp_physics", 13, "SBU-YLin", "sbu_ylinscheme", "implemented", "microphysics"),
    14: SchemeOption("mp_physics", 14, "WDM5", "wdm5scheme", "accepted", "microphysics"),
    16: SchemeOption("mp_physics", 16, "WDM6", "wdm6scheme", "accepted", "microphysics"),
    # v0.17: WSM6 single-moment + a separate precipitating HAIL class (qh). GPU
    # scan-wired (coupling.scan_adapters.wsm7_adapter), savepoint-parity-proven
    # against the unmodified WRF phys/module_mp_wsm7.F (proofs/v013).
    24: SchemeOption("mp_physics", 24, "WSM7", "wsm7scheme", "implemented", "microphysics"),
    # v0.17: WDM6 double-moment warm rain + a separate single-moment precipitating
    # HAIL class (qh, no Nh). GPU scan-wired (coupling.scan_adapters.wdm7_adapter),
    # savepoint-parity-proven vs the unmodified WRF phys/module_mp_wdm7.F
    # (proofs/v013_wdm7).
    26: SchemeOption("mp_physics", 26, "WDM7", "wdm7scheme", "implemented", "microphysics"),
    # v0.16: aerosol-aware Thompson (WRF Registry package thompsonaero). The
    # column kernel is WRF grid-savepoint parity-gated
    # (proofs/v016/thompson_aero_savepoint_parity.json) and wired through
    # coupling.physics_couplers.thompson_aero_adapter (mirrors mp=8).
    28: SchemeOption("mp_physics", 28, "Thompson aerosol-aware", "thompson_aero", "implemented", "microphysics"),
    # Goddard GCE (97): v0.17 single-moment 3-ice graupel scheme, faithful
    # column port of phys/module_mp_gsfcgce.F:gsfcgce (ihail=0/ice2=0/itaobraun=1/
    # new_ice_sat=2). Uses ONLY the existing moist substrate (qv,qc,qr,qi,qs,qg)
    # -- no new prognostic state -- and is savepoint-parity-proven against the
    # unmodified WRF Fortran (proofs/v090/goddard_mp_r2_savepoint_parity.json,
    # 5/5 cases, ~machine-precision vs the fp64 transparency oracle).
    97: SchemeOption("mp_physics", 97, "Goddard GCE", "gsfcgcescheme", "implemented", "microphysics"),
}

PBL_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("bl_pbl_physics", 0, "disabled", "none", "accepted", "pbl"),
    # YSU(1)/ACM2(7) status bumped to "implemented" (v0.6.0 GPU-op): the host-NumPy
    # single-column kernels were rewritten as jax.lax.scan-traceable / vmap-batched
    # JAX and scan-wired into the operational PBL slot, with no savepoint-parity
    # regression (proofs/v060/{ysu,acm2}_gpuop_savepoint_parity.json).
    1: SchemeOption("bl_pbl_physics", 1, "YSU", "ysuscheme", "implemented", "pbl"),
    # MYJ(2): savepoint-parity-proven CPU reference, NOT yet GPU-scan-wired
    # (fail-closed in the operational scan); requires Janjic Eta sfclay(2).
    2: SchemeOption("bl_pbl_physics", 2, "MYJ", "myjpblscheme", "accepted", "pbl"),
    # GFS(3): v0.17 jit/vmap-traceable port of phys/module_bl_gfs.F (BL_GFS ->
    # MONINP), the NCEP-GFS Hybrid-EDMF-ancestor nonlocal-K PBL; scan-wired into
    # the operational PBL slot (PBL_SCAN_ADAPTERS[3]); savepoint-parity-proven
    # against the unmodified WRF source at fp64 (proofs/v017/gfs_oracle.py, ~1e-13).
    3: SchemeOption("bl_pbl_physics", 3, "GFS", "gfsscheme", "implemented", "pbl"),
    # QNSE(4): v0.18 reference-only endpoint. A fp64 pristine-WRF single-column
    # oracle is staged (phys/module_bl_qnsepbl.F; proofs/v018/qnse_pbl4_reference_oracle.json),
    # but no traceable JAX column kernel is scan-wired.
    4: SchemeOption("bl_pbl_physics", 4, "QNSE-EDMF", "qnsepblscheme", "accepted", "pbl"),
    5: SchemeOption("bl_pbl_physics", 5, "MYNN", "mynnpblscheme", "implemented", "pbl"),
    7: SchemeOption("bl_pbl_physics", 7, "ACM2", "acmpblscheme", "implemented", "pbl"),
    8: SchemeOption("bl_pbl_physics", 8, "BouLac", "boulacscheme", "accepted", "pbl"),
    # CAM-UW(9): v0.22 JAX/vmap operational endpoint for the CAM5 UW
    # diagnostic-TKE vertical-diffusion shape (dry static energy, qv/qc/qi,
    # momentum, and qke diagnostics), scan-wired via
    # coupling.scan_adapters.camuw_pbl_adapter. It is intentionally proof-gated
    # as an idealized/source-present check, not full CAM-stack savepoint parity.
    9: SchemeOption("bl_pbl_physics", 9, "CAM-UW", "camuwpblscheme", "implemented", "pbl"),
    # TEMF(10): v0.18 reference-only endpoint. A fp64 pristine-WRF single-column
    # oracle is staged (phys/module_bl_temf.F; proofs/v018/temf_pbl10_reference_oracle.json),
    # but no traceable JAX column kernel is scan-wired.
    10: SchemeOption("bl_pbl_physics", 10, "TEMF", "temfpblscheme", "accepted", "pbl"),
    # Shin-Hong(11): v0.18 scale-aware (grid-size-dependent) YSU-family PBL
    # (phys/module_bl_shinhong.F), scan-wired via coupling.scan_adapters.
    # Dynamics path is parity-gated against the v090 host reference; TKE/EL are
    # non-driving diagnostics with explicit residuals vs that PARTIAL reference
    # (TKE rel ~=0.285, EL rel ~=0.013; refine after a pristine-WRF TKE oracle).
    11: SchemeOption("bl_pbl_physics", 11, "Shin-Hong", "shinhongscheme", "implemented", "pbl"),
    # GBM(12): v0.18 JAX/vmap port of the Grenier-Bretherton-McCaa moist
    # prognostic-TKE PBL (phys/module_bl_gbmpbl.F), scan-wired via
    # coupling.scan_adapters.gbm_pbl_adapter and parity-gated against the fp64
    # pristine-WRF oracle (proofs/v018/gbm_pbl12_jax_parity.json).
    12: SchemeOption("bl_pbl_physics", 12, "GBM TKE", "gbmpblscheme", "implemented", "pbl"),
    # EEPS/KEPS(16/17): v0.18 reference-only epsilon/k-epsilon endpoints. Their
    # fp64 pristine-WRF single-column oracles are staged under proofs/v018, but no
    # traceable JAX kernels are scan-wired.
    16: SchemeOption("bl_pbl_physics", 16, "EEPS epsilon", "eepsscheme", "accepted", "pbl"),
    17: SchemeOption("bl_pbl_physics", 17, "KEPS k-epsilon", "kepsscheme", "accepted", "pbl"),
    # MRF(99): v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F, scan-wired
    # into the operational PBL slot (PBL_SCAN_ADAPTERS[99]); savepoint-parity-proven
    # against the unmodified WRF source at fp64 (proofs/v013/mrf_oracle.py).
    99: SchemeOption("bl_pbl_physics", 99, "MRF", "mrfscheme", "implemented", "pbl"),
}

SFCLAY_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("sf_sfclay_physics", 0, "disabled", "none", "accepted", "surface_layer"),
    1: SchemeOption("sf_sfclay_physics", 1, "sfclayrev", "sfclayscheme", "accepted", "surface_layer"),
    2: SchemeOption("sf_sfclay_physics", 2, "Janjic Eta surface layer", "myjsfcscheme", "accepted", "surface_layer"),
    3: SchemeOption("sf_sfclay_physics", 3, "NCEP GFS surface layer", "gfssfcscheme", "implemented", "surface_layer"),
    5: SchemeOption("sf_sfclay_physics", 5, "MYNN surface layer", "mynnsfclayscheme", "implemented", "surface_layer"),
    7: SchemeOption("sf_sfclay_physics", 7, "Pleim-Xiu surface layer", "pxsfclayscheme", "accepted", "surface_layer"),
    91: SchemeOption("sf_sfclay_physics", 91, "old MM5 surface layer", "sfclayscheme_old", "implemented", "surface_layer"),
}

CU_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("cu_physics", 0, "disabled", "no_cumulus", "accepted", "cumulus"),
    1: SchemeOption("cu_physics", 1, "Kain-Fritsch", "kfetascheme", "implemented", "cumulus"),
    2: SchemeOption("cu_physics", 2, "Betts-Miller-Janjic", "bmjscheme", "accepted", "cumulus"),
    # GF(3) status bumped to "implemented" (v0.9.0 GPU-op): the scale-aware
    # closure-ensemble kernel was rewritten as a jit/vmap-batched JAX path
    # (physics._gf_jax.gfdrv_batched) and scan-wired into the operational cumulus
    # slot (CU_SCAN_ADAPTERS[3], stateless State->State), savepoint-parity-gated
    # against unmodified module_cu_gf_*.F (proofs/v060/gf_gpubatch_savepoint_parity.json).
    3: SchemeOption("cu_physics", 3, "Grell-Freitas", "gfscheme", "implemented", "cumulus"),
    # v0.17 SAS family: pristine-WRF fp64 savepoints exist, but the shared JAX
    # endpoint is RED vs oracle and remains reference-only / fail-closed.
    4: SchemeOption("cu_physics", 4, "Scale-aware GFS SAS", "scalesasscheme", "accepted", "cumulus"),
    # Grell-3D (5): v0.18 has a standalone pristine-WRF G3DRV oracle harness and
    # savepoints (proofs/v018/oracle/cumulus_grell), but all available trial
    # columns are null; an active trigger and faithful JAX endpoint remain
    # blocked. Accepted only for fail-closed oracle work -- NOT in
    # CU_SCAN_ADAPTERS / _SCAN_WIRED_OPTIONS.
    5: SchemeOption("cu_physics", 5, "Grell-3D ensemble", "g3scheme", "accepted", "cumulus"),
    6: SchemeOption("cu_physics", 6, "Tiedtke", "tiedtkescheme", "accepted", "cumulus"),
    # KIM Simplified Arakawa-Schubert (14): v0.13 Tier-3 reference-only with a
    # nontrivial fp64 pristine-WRF oracle staged; JAX kernel remains carry-over.
    14: SchemeOption("cu_physics", 14, "KIM Simplified Arakawa-Schubert", "ksasscheme", "accepted", "cumulus"),
    16: SchemeOption("cu_physics", 16, "New Tiedtke", "ntiedtkescheme", "accepted", "cumulus"),
    # SAS family (94/95/96, cu-sas lane) and Grell-Devenyi(93) / previous
    # Kain-Fritsch(99) (cu-kfgrell lane) are v0.17/v0.18 reference-only: accepted
    # for isolated real-WRF oracle work, but NOT scan-wired until their candidate
    # JAX endpoints pass source-specific pristine-WRF savepoint parity. GD(93)
    # has a v0.18 standalone GRELLDRV harness/savepoints, but they are null-only.
    93: SchemeOption("cu_physics", 93, "Grell-Devenyi ensemble", "gdscheme", "accepted", "cumulus"),
    94: SchemeOption("cu_physics", 94, "2015 GFS SAS / HWRF", "sasscheme", "accepted", "cumulus"),
    95: SchemeOption("cu_physics", 95, "Previous GFS SAS / HWRF OSAS", "osasscheme", "accepted", "cumulus"),
    96: SchemeOption("cu_physics", 96, "Previous new GFS SAS / YSU NSAS", "nsasscheme", "accepted", "cumulus"),
    99: SchemeOption("cu_physics", 99, "previous Kain-Fritsch", "kfscheme", "accepted", "cumulus"),
}

SURFACE_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("sf_surface_physics", 0, "disabled", "none", "accepted", "land_surface"),
    # slab=1 status bumped to "implemented" (v0.17 GPU-op): the fp64-oracle-validated
    # physics.lsm_slab SLAB1D column port is operationally scan-wired via
    # coupling.slab_surface_hook.slab_surface_step (5-layer TSLB land carry + GSW/GLW
    # radiation forcing + explicit TMN/THC/EMISS SlabStaticBundle).
    1: SchemeOption("sf_surface_physics", 1, "thermal-diffusion slab LSM", "slabscheme", "implemented", "land_surface"),
    2: SchemeOption("sf_surface_physics", 2, "Noah classic", "lsmscheme", "accepted", "land_surface"),
    # ruc=3 (RUC multi-layer soil/snow LSM) is v0.17 REFERENCE-ONLY: a fp64
    # pristine-WRF single-column oracle is staged (LSMRUC->SOILVEGIN->SFCTMP,
    # proofs/v017/oracle/ruclsm + savepoints/ruclsm), but a faithful traceable JAX
    # column port of the ~7.5k-LOC multi-layer soil/snow solver (SFCTMP + SOIL/
    # SNOWSOIL + SOILTEMP/SNOWTEMP/SOILMOIST/SOILPROP/TRANSF/VILKA) is a documented
    # carry-over, so it is "accepted" (selectable for a single-column reference
    # comparison) and fail-closes in the operational scan.
    3: SchemeOption("sf_surface_physics", 3, "RUC LSM", "ruclsmscheme", "accepted", "land_surface"),
    4: SchemeOption("sf_surface_physics", 4, "Noah-MP", "noahmpscheme", "implemented", "land_surface"),
    # px=7 (Pleim-Xiu 2-layer ISBA LSM) is v0.17 GPU-operational: the fp64-oracle-
    # validated physics.lsm_pleim_xiu SURFPX+QFLUX column port is scan-wired via
    # coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step (pairs with the PX
    # surface layer sf_sfclay_physics=7). Carries the 2-layer ISBA land state.
    7: SchemeOption("sf_surface_physics", 7, "Pleim-Xiu LSM", "pxlsmscheme", "implemented", "land_surface"),
    # ssib=8 (SSiB SiB biophysical canopy/soil/snow LSM) is v0.17 REFERENCE-ONLY: a
    # fp64 pristine-WRF single-column oracle is staged (the unmodified SSIB driver +
    # its ~30 internal subroutines, proofs/v017/oracle/ssib + savepoints/ssib), but a
    # faithful traceable JAX column port of the ~6.6k-LOC coupled SiB canopy/soil/
    # 4-level-snow solver (TEMRS1/TEMRS2 + UPDAT1 + RADAB + STOMA1 + INTERC + STRES1
    # + NEWTON) is a documented carry-over, so it is "accepted" (selectable for a
    # single-column reference comparison) and fail-closes in the operational scan.
    8: SchemeOption("sf_surface_physics", 8, "SSiB LSM", "ssibscheme", "accepted", "land_surface"),
}

RA_SW_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("ra_sw_physics", 0, "disabled", "none", "accepted", "radiation"),
    1: SchemeOption("ra_sw_physics", 1, "Dudhia shortwave", "swradscheme", "implemented", "radiation"),
    2: SchemeOption("ra_sw_physics", 2, "GSFC (Chou-Suarez) shortwave", "gsfcswscheme", "implemented", "radiation"),
    3: SchemeOption("ra_sw_physics", 3, "CAM shortwave", "camscheme", "accepted", "radiation"),
    4: SchemeOption("ra_sw_physics", 4, "RRTMG shortwave", "rrtmg_swscheme", "accepted", "radiation"),
    5: SchemeOption("ra_sw_physics", 5, "Goddard shortwave (new)", "goddard_swscheme", "accepted", "radiation"),
    7: SchemeOption("ra_sw_physics", 7, "FLG (UCLA) shortwave", "flgscheme", "accepted", "radiation"),
    99: SchemeOption("ra_sw_physics", 99, "GFDL (Eta) shortwave", "gfdl_swscheme", "accepted", "radiation"),
}

RA_LW_SCHEMES: Mapping[int, SchemeOption] = {
    0: SchemeOption("ra_lw_physics", 0, "disabled", "none", "accepted", "radiation"),
    1: SchemeOption("ra_lw_physics", 1, "RRTM longwave", "rrtmscheme", "implemented", "radiation"),
    3: SchemeOption("ra_lw_physics", 3, "CAM longwave", "camscheme", "accepted", "radiation"),
    4: SchemeOption("ra_lw_physics", 4, "RRTMG longwave", "rrtmg_lwscheme", "accepted", "radiation"),
    # GSFC/Goddard NUWRF longwave (5): v0.13 Tier-3 reference-only -- a fp64
    # single-column pristine-WRF oracle (module_ra_goddard.F:lwrad) is staged
    # (proofs/v013/oracle/radiation_lw), but the traceable JAX column kernel is a
    # documented carry-over, so it is "accepted" (selectable for a reference
    # comparison) and fail-closes in the operational scan.
    5: SchemeOption("ra_lw_physics", 5, "GSFC/Goddard NUWRF longwave", "goddardlwscheme", "accepted", "radiation"),
    7: SchemeOption("ra_lw_physics", 7, "FLG (UCLA) longwave", "flgscheme", "accepted", "radiation"),
    # Held-Suarez idealized radiation (31): v0.17 GPU-op no-kernel-change endpoint
    # port of phys/module_ra_hs.F:HSRAD (Newtonian relaxation toward an analytic
    # equilibrium temperature; combined LW+SW selected through the LW slot, no
    # separate SW call). Stateless State->RTHRATEN coupler (held_suarez_theta_tendency),
    # savepoint-parity-proven against the unmodified WRF source at fp64
    # (proofs/v017/held_suarez_lw_savepoint_parity.json) and scan-wired.
    31: SchemeOption("ra_lw_physics", 31, "Held-Suarez idealized radiation", "heldsuarez", "implemented", "radiation"),
    99: SchemeOption("ra_lw_physics", 99, "GFDL (Eta) longwave", "gfdl_lwscheme", "accepted", "radiation"),
}


# Moisture species: WRF 4-D ``moist`` array members. State leaf name equals the
# lowercase WRF moist-array member name used throughout this port.
#
# The CORE tuple is the v0.2.0..v0.16 six-class set and is FROZEN -- every
# existing per-scheme path iterates it byte-for-byte. The v0.17 ADR-032 hail
# substrate adds the hail mixing ratio ``qh`` as a SEPARATE additive group
# (mirroring NUMBER_SPECIES_ADDITIVE) so the core set is unchanged; only the
# hail microphysics family (mp 7/24/26/27 + NSSL hail) ever carries ``qh``.
MOIST_SPECIES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")
MOIST_SPECIES_ADDITIVE: tuple[str, ...] = ("qh",)
MOIST_SPECIES_ALL: tuple[str, ...] = MOIST_SPECIES + MOIST_SPECIES_ADDITIVE

MOIST_WRFOUT_NAME: Mapping[str, str] = {
    "qv": "QVAPOR",
    "qc": "QCLOUD",
    "qr": "QRAIN",
    "qi": "QICE",
    "qs": "QSNOW",
    "qg": "QGRAUP",
    # v0.17 ADR-032 hail substrate (WRF Registry moist member qh -> QHAIL).
    "qh": "QHAIL",
}

MP_MOIST_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: ("qv",),
    1: ("qv", "qc", "qr"),
    2: ("qv", "qc", "qr", "qi", "qs", "qg"),
    3: ("qv", "qc", "qr"),
    4: ("qv", "qc", "qr", "qi", "qs"),
    6: ("qv", "qc", "qr", "qi", "qs", "qg"),
    8: ("qv", "qc", "qr", "qi", "qs", "qg"),
    10: ("qv", "qc", "qr", "qi", "qs", "qg"),
    13: ("qv", "qc", "qr", "qi", "qs"),
    14: ("qv", "qc", "qr", "qi", "qs"),
    16: ("qv", "qc", "qr", "qi", "qs", "qg"),
    # v0.17 WSM7 = WSM6 six-class + a separate precipitating hail class (qh).
    24: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
    # v0.17 WDM7 = WDM6 six-class + a separate precipitating hail class (qh).
    26: ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
    28: ("qv", "qc", "qr", "qi", "qs", "qg"),
    97: ("qv", "qc", "qr", "qi", "qs", "qg"),
}


# Number concentrations / WRF ``scalar`` array members. Existing Thompson-era
# State leaves are preserved; WDM6 adds ``Nc`` and ``Nn`` append-only; the
# v0.17 ADR-032 hail substrate appends ``Nh`` (WRF scalar qnh) AFTER them, then
# v0.16 aerosol-aware Thompson (mp=28) appends ``nwfa``/``nifa`` AFTER the hail
# substrate (append-only; the State pytree appends them at the very END of
# __slots__ in this same order).
NUMBER_SPECIES_EXISTING: tuple[str, ...] = ("Ni", "Nr", "Ns", "Ng")
NUMBER_SPECIES_ADDITIVE: tuple[str, ...] = ("Nc", "Nn", "Nh", "nwfa", "nifa")
NUMBER_SPECIES: tuple[str, ...] = NUMBER_SPECIES_EXISTING + NUMBER_SPECIES_ADDITIVE

NUMBER_REGISTRY_MEMBER: Mapping[str, str] = {
    "Ni": "qni",
    "Nr": "qnr",
    "Ns": "qns",
    "Ng": "qng",
    "Nc": "qnc",
    "Nn": "qnn",
    # v0.17 ADR-032 hail substrate (WRF scalar member qnh).
    "Nh": "qnh",
    # v0.16 aerosol-aware Thompson (mp=28) scalar members.
    "nwfa": "qnwfa",
    "nifa": "qnifa",
}

NUMBER_WRFOUT_NAME: Mapping[str, str] = {
    "Ni": "QNICE",
    "Nr": "QNRAIN",
    "Ns": "QNSNOW",
    "Ng": "QNGRAUPEL",
    "Nc": "QNCLOUD",
    "Nn": "QNCCN",
    # v0.17 ADR-032 hail substrate.
    "Nh": "QNHAIL",
    # v0.16 aerosol-aware Thompson (mp=28).
    "nwfa": "QNWFA",
    "nifa": "QNIFA",
}


# Predicted-density particle-volume species: WRF ``scalar`` array members that
# are neither mixing ratios nor number concentrations. v0.17 ADR-032 adds the
# graupel/hail volumes consumed by the predicted-density schemes
# (Thompson-graupel/hail mp=38; NSSL nssl_graupelvol / nssl_hailvol).
VOLUME_SPECIES: tuple[str, ...] = ("qvolg", "qvolh")

VOLUME_REGISTRY_MEMBER: Mapping[str, str] = {
    "qvolg": "qvolg",
    "qvolh": "qvolh",
}

VOLUME_WRFOUT_NAME: Mapping[str, str] = {
    "qvolg": "QVGRAUPEL",
    "qvolh": "QVHAIL",
}

MP_NUMBER_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (),
    2: (),
    3: (),
    4: (),
    6: (),
    8: ("Ni", "Nr"),
    10: ("Ni", "Ns", "Nr", "Ng"),
    13: (),
    14: ("Nn", "Nc", "Nr"),
    16: ("Nn", "Nc", "Nr"),
    # v0.17 WSM7 is single-moment (no prognostic number concentrations).
    24: (),
    # v0.17 WDM7 = WDM6 double-moment warm rain (Nn CCN, Nc, Nr); hail is
    # single-moment (qh only, no Nh).
    26: ("Nn", "Nc", "Nr"),
    # WRF Registry package thompsonaero: scalar:qnc,qnr,qni,qnwfa,qnifa.
    28: ("Ni", "Nr", "Nc", "nwfa", "nifa"),
    97: (),
}


# Precipitation accumulators. ``rainc_acc`` is additive State because cumulus
# precipitation is a prognostic history/restart quantity in WRF (RAINC).
ACCUMULATORS_EXISTING: tuple[str, ...] = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc")
# v0.17 hail microphysics adds ``hail_acc`` (WRF HAILNC) append-only alongside the
# cumulus ``rainc_acc``; carried by the hail MP family (WSM7=24, WDM7=26).
ACCUMULATORS_ADDITIVE: tuple[str, ...] = ("rainc_acc", "hail_acc")
ACCUMULATORS: tuple[str, ...] = ACCUMULATORS_EXISTING + ACCUMULATORS_ADDITIVE

ACCUMULATOR_WRFOUT_NAME: Mapping[str, str] = {
    "rain_acc": "RAINNC",
    "snow_acc": "SNOWNC",
    "graupel_acc": "GRAUPELNC",
    "ice_acc": "SNOWNC",
    "rainc_acc": "RAINC",
    # v0.17 hail microphysics surface accumulator.
    "hail_acc": "HAILNC",
}

V060_EXISTING_STATE_PHYSICS_LEAVES: tuple[str, ...] = (
    *MOIST_SPECIES,
    *NUMBER_SPECIES_EXISTING,
    "qke",
    *ACCUMULATORS_EXISTING,
)
# Append-only additive State leaves: the v0.6.0 Nc/Nn + rainc_acc, plus the
# v0.17 ADR-032 hail substrate (qh moist, Nh number, qvolg/qvolh volumes).
V060_ADDITIVE_STATE_LEAVES: tuple[str, ...] = (
    *NUMBER_SPECIES_ADDITIVE,
    *MOIST_SPECIES_ADDITIVE,
    *VOLUME_SPECIES,
    *ACCUMULATORS_ADDITIVE,
)


@dataclass(frozen=True)
class RegistryFieldSpec:
    """Frozen metadata for a WRF Registry-backed physics field."""

    leaf: str
    wrf_name: str
    registry_member: str
    kind: str
    shape: str
    storage: str
    existing_state: bool
    additive_state: bool
    restart_required: bool
    wrfout_required: bool
    nest_forcedown: bool
    nest_feedback: bool
    lateral_bc: bool
    schemes: tuple[str, ...]
    notes: str = ""


def _field(
    leaf: str,
    wrf_name: str,
    registry_member: str,
    kind: str,
    shape: str,
    storage: str,
    schemes: tuple[str, ...],
    *,
    existing_state: bool = False,
    additive_state: bool = False,
    restart_required: bool = False,
    wrfout_required: bool = False,
    nest_forcedown: bool = False,
    nest_feedback: bool = False,
    lateral_bc: bool = False,
    notes: str = "",
) -> RegistryFieldSpec:
    return RegistryFieldSpec(
        leaf=leaf,
        wrf_name=wrf_name,
        registry_member=registry_member,
        kind=kind,
        shape=shape,
        storage=storage,
        existing_state=existing_state,
        additive_state=additive_state,
        restart_required=restart_required,
        wrfout_required=wrfout_required,
        nest_forcedown=nest_forcedown,
        nest_feedback=nest_feedback,
        lateral_bc=lateral_bc,
        schemes=schemes,
        notes=notes,
    )


FIELD_SPECS: tuple[RegistryFieldSpec, ...] = (
    *(
        _field(
            leaf,
            MOIST_WRFOUT_NAME[leaf],
            leaf,
            "moist",
            "mass_3d",
            "State",
            tuple(f"mp{opt}" for opt, members in MP_MOIST_MEMBERS.items() if leaf in members),
            existing_state=leaf in MOIST_SPECIES,
            additive_state=leaf in MOIST_SPECIES_ADDITIVE,
            restart_required=True,
            wrfout_required=leaf in {"qv", "qc", "qr", "qi", "qs", "qg"},
            nest_forcedown=True,
            nest_feedback=True,
            lateral_bc=True,
            notes=(
                "v0.17 ADR-032 hail substrate; carried by the hail MP family "
                "(mp 7/24/26/27 + NSSL hail); inert until a hail scheme is wired."
                if leaf in MOIST_SPECIES_ADDITIVE
                else ""
            ),
        )
        for leaf in MOIST_SPECIES_ALL
    ),
    *(
        _field(
            leaf,
            NUMBER_WRFOUT_NAME[leaf],
            NUMBER_REGISTRY_MEMBER[leaf],
            "scalar_number",
            "mass_3d",
            "State",
            tuple(f"mp{opt}" for opt, members in MP_NUMBER_MEMBERS.items() if leaf in members),
            existing_state=leaf in NUMBER_SPECIES_EXISTING,
            additive_state=leaf in NUMBER_SPECIES_ADDITIVE,
            restart_required=True,
            wrfout_required=True,
            nest_forcedown=True,
            nest_feedback=True,
            lateral_bc=False,
            notes="WRF have_bcs_scalar controls lateral scalar bdy; nests force/feedback active scalars.",
        )
        for leaf in NUMBER_SPECIES
    ),
    *(
        _field(
            leaf,
            VOLUME_WRFOUT_NAME[leaf],
            VOLUME_REGISTRY_MEMBER[leaf],
            "scalar_volume",
            "mass_3d",
            "State",
            (),
            additive_state=True,
            restart_required=True,
            wrfout_required=True,
            nest_forcedown=True,
            nest_feedback=True,
            lateral_bc=False,
            notes=(
                "v0.17 ADR-032 predicted-density particle volume; carried by "
                "Thompson-graupel/hail (mp=38) and the NSSL predicted-density "
                "packages; inert until a hail scheme is wired."
            ),
        )
        for leaf in VOLUME_SPECIES
    ),
    *(
        _field(
            leaf,
            ACCUMULATOR_WRFOUT_NAME[leaf],
            ACCUMULATOR_WRFOUT_NAME[leaf],
            "accumulator",
            "surface_2d",
            "State",
            ("microphysics", "cumulus") if leaf == "rainc_acc" else ("microphysics",),
            existing_state=leaf in ACCUMULATORS_EXISTING,
            additive_state=leaf in ACCUMULATORS_ADDITIVE,
            restart_required=True,
            wrfout_required=True,
            notes="History/restart accumulator; updated as increments by scheme adapters.",
        )
        for leaf in ACCUMULATORS
    ),
    _field(
        "qke",
        "QKE",
        "qke",
        "pbl_scalar",
        "mass_3d",
        "State",
        ("pbl5", "pbl8"),
        existing_state=True,
        restart_required=True,
        wrfout_required=True,
        nest_forcedown=True,
        nest_feedback=True,
        lateral_bc=False,
        notes="Persistent TKE leaf for TKE-based PBL schemes; WRF MYNN advected scalar member is qke_adv.",
    ),
    *(
        _field(
            leaf,
            leaf.upper(),
            leaf,
            "microphysics_diagnostic",
            "mass_3d",
            "PhysicsDiagnostics",
            ("mp3", "mp4", "mp6", "mp8", "mp16"),
            notes="Effective radius diagnostic/carry; not a dycore State leaf.",
        )
        for leaf in ("re_cloud", "re_ice", "re_snow")
    ),
    *(
        _field(
            leaf.lower(),
            leaf,
            leaf.lower(),
            "cumulus_tendency",
            "mass_3d",
            "PhysicsCarry",
            ("cu1", "cu2", "cu3", "cu4", "cu5", "cu6", "cu14", "cu16", "cu93", "cu94", "cu95", "cu96", "mp10"),
            notes="WRF R*CUTEN state/tendency family carried between physics driver calls.",
        )
        for leaf in (
            "RUCUTEN",
            "RVCUTEN",
            "RTHCUTEN",
            "RQVCUTEN",
            "RQRCUTEN",
            "RQCCUTEN",
            "RQSCUTEN",
            "RQICUTEN",
            "RQCNCUTEN",
            "RQINCUTEN",
        )
    ),
    _field("raincv", "RAINCV", "RAINCV", "cumulus_diagnostic", "surface_2d", "PhysicsDiagnostics", ("cu1", "cu2", "cu3", "cu4", "cu5", "cu6", "cu14", "cu16", "cu93", "cu94", "cu95", "cu96")),
    _field("rainshv", "RAINSHV", "RAINSHV", "cumulus_diagnostic", "surface_2d", "PhysicsDiagnostics", ("cu1", "cu2", "cu3", "cu4", "cu5", "cu6", "cu14", "cu16", "cu93", "cu94", "cu95", "cu96")),
    _field("cldefi", "CLDEFI", "CLDEFI", "cumulus_carry", "surface_2d", "PhysicsCarry", ("cu2",), notes="BMJ precipitation efficiency/cloud efficiency state."),
    _field("nca", "NCA", "NCA", "cumulus_carry", "surface_2d", "PhysicsCarry", ("cu1", "cu99"), notes="KF relaxation counter."),
    _field("w0avg", "W0AVG", "w0avg", "cumulus_carry", "mass_3d", "PhysicsCarry", ("cu1", "cu99"), notes="KF-family average vertical velocity."),
    *(
        _field(leaf, leaf.upper(), leaf, "cumulus_carry", "mass_3d", "PhysicsCarry", ("cu3",))
        for leaf in ("cugd_qvten", "cugd_tten", "cugd_qvtens", "cugd_ttens", "cugd_qcten")
    ),
    *(
        _field(leaf, leaf.upper(), leaf, "cumulus_carry", "surface_2d", "PhysicsCarry", ("cu3",))
        for leaf in ("xmb_shallow", "k22_shallow", "kbcon_shallow", "ktop_shallow")
    ),
    _field("tke_pbl", "TKE_PBL", "tke_pbl", "pbl_diagnostic", "mass_3d", "PhysicsDiagnostics", ("pbl2", "pbl5")),
    _field("el_pbl", "EL_PBL", "el_pbl", "pbl_diagnostic", "mass_3d", "PhysicsDiagnostics", ("pbl2", "pbl5")),
    *(
        _field(leaf, leaf.upper(), leaf, "pbl_diagnostic", "mass_3d", "PhysicsDiagnostics", ("pbl5",))
        for leaf in ("sh3d", "sm3d", "tsq", "qsq", "cov")
    ),
    *(
        _field(leaf, leaf.upper(), leaf, "land_carry", "surface_2d", "PhysicsCarry", ("surface2",))
        for leaf in ("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn")
    ),
)

FIELD_SPECS_BY_LEAF: Mapping[str, RegistryFieldSpec] = {spec.leaf: spec for spec in FIELD_SPECS}


CUMULUS_TENDENCY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("rucuten", "rvcuten", "rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    2: ("rthcuten", "rqvcuten"),
    3: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    # Grell-3D(5) writes theta/qv/qc/qi cumulus tendencies + cumulus momentum
    # (RU/RV CUTEN); KSAS(14) and SAS-family(4/94/95/96) likewise. These
    # reference-only oracle dumps contain exactly these tendency fields.
    4: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"),
    5: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"),
    6: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    14: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"),
    16: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
    93: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten"),
    94: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"),
    95: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"),
    96: ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"),
    99: ("rthcuten", "rqvcuten", "rqrcuten", "rqccuten", "rqscuten", "rqicuten"),
}

CUMULUS_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("w0avg", "nca"),
    2: ("cldefi",),
    3: (
        "cugd_qvten",
        "cugd_tten",
        "cugd_qvtens",
        "cugd_ttens",
        "cugd_qcten",
        "xmb_shallow",
        "k22_shallow",
        "kbcon_shallow",
        "ktop_shallow",
    ),
    # 4/94/95/96 (SAS family), 5 (Grell-3D), and 14 (KSAS) are
    # reference-only: no persistent operational cumulus carry is threaded yet.
    4: (),
    5: (),
    6: (),
    14: (),
    16: (),
    93: (),
    94: (),
    95: (),
    96: (),
    99: ("w0avg", "nca"),
}

PBL_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (),
    2: ("tke_pbl", "el_pbl"),
    3: (),  # GFS is a nonlocal-K scheme: no prognostic PBL carry (like YSU/MRF)
    # QNSE(4) carries the EDKF mass-flux/updraft state documented by the WRF
    # Registry. REFERENCE-ONLY: the carry list documents the state a future JAX
    # kernel must thread; the operational scan fail-closes this option.
    4: (
        "tke_pbl", "el_pbl", "massflux_EDKF", "entr_EDKF", "detr_EDKF",
        "thl_up", "thv_up", "rv_up", "rt_up", "rc_up", "u_up", "v_up",
        "frac_up", "rc_mf",
    ),
    5: ("qke",),
    7: (),
    8: ("qke",),
    # CAM-UW carries previous-step TKE/K diagnostics and residual stresses in
    # the WRF CAM wrapper. The operational adapter stores TKE in State.qke and
    # currently recomputes K/residual terms each call until a full CAM carry lands.
    9: ("tke_pbl", "kvm3d", "kvh3d", "tauresx", "tauresy"),
    # TEMF(10) carries the total-energy / mass-flux diagnostic state emitted by
    # phys/module_bl_temf.F. REFERENCE-ONLY in v0.18.
    10: (
        "te_temf", "kh_temf", "km_temf", "shf_temf", "qf_temf", "uw_temf",
        "vw_temf", "wupd_temf", "mf_temf", "thup_temf", "qlup_temf",
        "qtup_temf", "cf3d_temf", "hd_temf", "lcl_temf", "hct_temf",
        "cfm_temf",
    ),
    # Shin-Hong(11) carries the prognostic mixing length + TKE diagnostic
    # (WRF Registry: state:el_pbl,tke_pbl). The operational State adapter maps
    # the TKE member onto the existing qke leaf.
    11: ("tke_pbl", "el_pbl"),
    # GBM(12) carries the TKE exchange coefficient + mixing length + TKE
    # (WRF Registry: state:exch_tke,el_pbl,tke_pbl).
    12: ("exch_tke", "tke_pbl", "el_pbl"),
    # EEPS/KEPS reference-only carry from the WRF Registry. The ``*_adv`` members
    # are Registry scalar fields and are listed here so future kernels preserve
    # restart/nest ownership explicitly.
    16: ("pek_pbl", "pep_pbl", "pek_adv", "pep_adv"),
    17: (
        "tke_pbl", "diss_pbl", "tpe_pbl", "pr_pbl", "wu_tur", "wv_tur",
        "wt_tur", "wq_tur", "tke_adv", "diss_adv", "tpe_adv",
    ),
    99: (),  # MRF is a nonlocal-K scheme: no prognostic PBL carry (like YSU/ACM2)
}

PBL_DIAGNOSTIC_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: ("pblh",),
    2: ("pblh", "kpbl", "mixht", "tke_pbl", "exch_h", "exch_m", "el_pbl"),
    3: ("pblh", "kpbl"),  # GFS diagnoses PBL height (HPBL) and KPBL
    4: ("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_h", "exch_m", "massflux_EDKF"),
    5: ("pblh", "tke_pbl", "sh3d", "sm3d", "tsq", "qsq", "cov", "el_pbl"),
    7: ("pblh",),
    8: ("pblh", "tke_pbl", "dlk", "exch_h", "exch_m"),
    9: ("pblh", "kpbl", "tke_pbl", "kvh", "kvm", "turbtype", "smaw", "tpert", "qpert", "wpert"),
    10: ("pblh", "kpbl", "kh_temf", "km_temf", "te_temf", "mf_temf", "hd_temf", "hct_temf"),
    11: ("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_h"),  # Shin-Hong scale-aware
    12: ("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_tke"),  # GBM moist TKE
    16: ("pblh", "kpbl", "pek_pbl", "pep_pbl", "exch_h", "exch_m"),
    17: ("pblh", "kpbl", "tke_pbl", "diss_pbl", "tpe_pbl", "pr_pbl", "exch_h", "exch_m"),
    99: ("pblh", "kpbl"),  # MRF diagnoses PBL height (PBL0) and KPBL
}

LAND_CARRY_MEMBERS: Mapping[int, tuple[str, ...]] = {
    0: (),
    # slab=1 carries the 5-layer soil temperature TSLB (reference-only until the
    # operational LSM hook lands; physics.lsm_slab).
    1: ("tslb",),
    2: ("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn"),
    # RUC carries the multi-layer soil/snow land state (SOILT skin temperature,
    # TSO soil temperatures, SOILMOIS/SH2O total+liquid soil moisture, SMFR3D
    # frozen-soil fraction + KEEPFR3DFLAG, SNOW/SNOWH snow water+depth, plus the
    # diagnostic surface moisture QSFC/QVG/QCG/QSG). REFERENCE-ONLY: the carry
    # member list documents the state the future JAX port must thread, but the
    # operational scan fail-closes RUC (physics.lsm_ruc; oracle staged).
    3: ("soilt", "tso", "soilmois", "sh2o", "smfr3d", "keepfr3dflag", "snow", "snowh", "qsfc"),
    4: ("NoahMPLandState",),
    # Pleim-Xiu carries the 2-layer ISBA land state (TG/T2 soil temperatures,
    # WG/W2 soil moisture, WR canopy water; physics.lsm_pleim_xiu).
    7: ("tg", "t2", "wg", "w2", "wr"),
    # SSiB carries the SiB biophysical land state (TC canopy temperature, TGS soil
    # surface temperature, TD deep soil temperature, WWW1/WWW2/WWW3 3-layer soil
    # moisture, CAPAC canopy interception/snow). REFERENCE-ONLY: the carry member
    # list documents the state the future JAX port must thread, but the operational
    # scan fail-closes SSiB (physics.lsm_ssib; oracle staged).
    8: ("tc", "tgs", "td", "www1", "www2", "www3", "capac"),
}

NOAH_CLASSIC_NUM_SOIL_LAYERS = 4


@dataclass(frozen=True)
class NestFieldEntry:
    """One field in the Registry-driven nest forcedown/feedback/boundary list."""

    leaf: str
    wrf_name: str
    kind: str
    stagger: str
    forcedown: bool
    feedback: bool
    lateral_bc: bool


# Core prognostic/base list. ``qv`` is deliberately core because WRF nests force
# water vapor even when no microphysics package is active.
_NEST_CORE: tuple[NestFieldEntry, ...] = (
    NestFieldEntry("u", "U", "prognostic", "X", True, True, True),
    NestFieldEntry("v", "V", "prognostic", "Y", True, True, True),
    NestFieldEntry("w", "W", "prognostic", "Z", True, True, True),
    NestFieldEntry("ph_perturbation", "PH", "prognostic", "Z", True, True, True),
    NestFieldEntry("theta", "T", "prognostic", "", True, True, True),
    NestFieldEntry("mu_perturbation", "MU", "prognostic", "", True, True, True),
    NestFieldEntry("qv", "QVAPOR", "moist", "", True, True, True),
    NestFieldEntry("phb_bdy", "PHB", "base", "Z", True, False, True),
    NestFieldEntry("pb_bdy", "PB", "base", "", True, False, True),
    NestFieldEntry("mub_bdy", "MUB", "base", "", True, False, True),
)


def moist_species_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return active WRF moist-array members for an accepted microphysics option."""

    return MP_MOIST_MEMBERS[int(mp_physics)]


def number_species_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return active WRF scalar number leaves for an accepted microphysics option."""

    return MP_NUMBER_MEMBERS[int(mp_physics)]


def state_leaves_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return State leaves an MP option may read/write."""

    return (*moist_species_for_mp(mp_physics), *number_species_for_mp(mp_physics))


def wrfout_names_for_mp(mp_physics: int) -> tuple[str, ...]:
    """Return wrfout names introduced by the active MP option."""

    leaves = state_leaves_for_mp(mp_physics)
    names: list[str] = []
    for leaf in leaves:
        if leaf in MOIST_WRFOUT_NAME:
            names.append(MOIST_WRFOUT_NAME[leaf])
        elif leaf in NUMBER_WRFOUT_NAME:
            names.append(NUMBER_WRFOUT_NAME[leaf])
    return tuple(names)


def physics_state_append_order() -> tuple[str, ...]:
    """Return the append-only State leaves frozen by v0.6.0 S0."""

    return V060_ADDITIVE_STATE_LEAVES


def nest_field_list(
    mp_physics: int = 8,
    *,
    bl_pbl_physics: int = 5,
    cu_physics: int = 0,
    include_numbers: bool = True,
) -> tuple[NestFieldEntry, ...]:
    """Return the frozen, scheme-aware nest forcedown/feedback/boundary list.

    v0.5.0 nesting and v0.6.0 physics-expansion must call this function instead
    of hardcoding the active moisture/scalar set. Moist members get forcedown,
    feedback, and lateral boundary carry. Scalar number fields and MYNN ``qke``
    get forcedown/feedback but default to no lateral scalar bdy unless a later
    lane explicitly enables WRF ``have_bcs_scalar`` semantics.
    """

    int(mp_physics)
    int(bl_pbl_physics)
    int(cu_physics)
    entries: list[NestFieldEntry] = list(_NEST_CORE)
    seen = {entry.leaf for entry in entries}

    for leaf in moist_species_for_mp(mp_physics):
        if leaf in seen:
            continue
        entries.append(NestFieldEntry(leaf, MOIST_WRFOUT_NAME[leaf], "moist", "", True, True, True))
        seen.add(leaf)

    if include_numbers:
        for leaf in number_species_for_mp(mp_physics):
            if leaf in seen:
                continue
            entries.append(NestFieldEntry(leaf, NUMBER_WRFOUT_NAME[leaf], "scalar", "", True, True, False))
            seen.add(leaf)
        if int(bl_pbl_physics) in (5, 8, 9, 11, 12) and "qke" not in seen:
            entries.append(NestFieldEntry("qke", "QKE", "scalar", "", True, True, False))

    return tuple(entries)


def assert_registry_consistent() -> None:
    """Fail-closed internal consistency checks for the frozen registry."""

    for key, accepted in ACCEPTED_NAMELIST_OPTIONS.items():
        assert len(set(accepted)) == len(accepted), f"{key} has duplicate accepted values"

    for opt in ACCEPTED_MP_PHYSICS:
        assert opt in MP_SCHEMES, f"missing MP scheme metadata for {opt}"
        assert opt in MP_MOIST_MEMBERS, f"missing moist members for mp={opt}"
        assert opt in MP_NUMBER_MEMBERS, f"missing number members for mp={opt}"
    for opt in ACCEPTED_BL_PBL_PHYSICS:
        assert opt in PBL_SCHEMES, f"missing PBL metadata for {opt}"
        assert opt in PBL_CARRY_MEMBERS, f"missing PBL carry metadata for {opt}"
    for opt in ACCEPTED_SF_SFCLAY_PHYSICS:
        assert opt in SFCLAY_SCHEMES, f"missing surface-layer metadata for {opt}"
    for opt in ACCEPTED_CU_PHYSICS:
        assert opt in CU_SCHEMES, f"missing cumulus metadata for {opt}"
        assert opt in CUMULUS_CARRY_MEMBERS, f"missing cumulus carry metadata for {opt}"
        assert opt in CUMULUS_TENDENCY_MEMBERS, f"missing cumulus tendency metadata for {opt}"
    for opt in ACCEPTED_SF_SURFACE_PHYSICS:
        assert opt in SURFACE_SCHEMES, f"missing surface metadata for {opt}"
        assert opt in LAND_CARRY_MEMBERS, f"missing land carry metadata for {opt}"
    for opt in ACCEPTED_RA_SW_PHYSICS:
        assert opt in RA_SW_SCHEMES, f"missing SW radiation metadata for {opt}"
    for opt in ACCEPTED_RA_LW_PHYSICS:
        assert opt in RA_LW_SCHEMES, f"missing LW radiation metadata for {opt}"

    for opt, members in MP_MOIST_MEMBERS.items():
        for leaf in members:
            # v0.17: hail schemes (WSM7 mp=24, WDM7 mp=26) carry the additive qh
            # moist member, so membership is validated against MOIST_SPECIES_ALL
            # (core six + hail), not just the frozen core.
            assert leaf in MOIST_SPECIES_ALL, f"mp={opt} moist member {leaf!r} not in MOIST_SPECIES_ALL"
            assert leaf in MOIST_WRFOUT_NAME, f"moist {leaf!r} missing wrfout name"
    for opt, members in MP_NUMBER_MEMBERS.items():
        for leaf in members:
            assert leaf in NUMBER_SPECIES, f"mp={opt} number member {leaf!r} not in NUMBER_SPECIES"
            assert leaf in NUMBER_REGISTRY_MEMBER, f"number {leaf!r} missing Registry scalar member"
            assert leaf in NUMBER_WRFOUT_NAME, f"number {leaf!r} missing wrfout name"
    for leaf in ACCUMULATORS:
        assert leaf in ACCUMULATOR_WRFOUT_NAME, f"accumulator {leaf!r} missing wrfout name"
    # v0.17 ADR-032 hail substrate group consistency (append-only). The hail
    # moist member qh is now exercised by the wired WSM7 (mp=24) via the
    # per-scheme membership loop above; the volume species (qvolg/qvolh) have no
    # wired scheme yet, so they are validated as standalone groups here.
    for leaf in MOIST_SPECIES_ADDITIVE:
        assert leaf in MOIST_WRFOUT_NAME, f"additive moist {leaf!r} missing wrfout name"
        assert leaf not in MOIST_SPECIES, f"additive moist {leaf!r} must not duplicate core MOIST_SPECIES"
    assert len(set(MOIST_SPECIES_ALL)) == len(MOIST_SPECIES_ALL), "MOIST_SPECIES_ALL has duplicates"
    for leaf in VOLUME_SPECIES:
        assert leaf in VOLUME_REGISTRY_MEMBER, f"volume {leaf!r} missing Registry scalar member"
        assert leaf in VOLUME_WRFOUT_NAME, f"volume {leaf!r} missing wrfout name"
    assert len(set(VOLUME_SPECIES)) == len(VOLUME_SPECIES), "VOLUME_SPECIES has duplicates"

    field_names = [spec.leaf for spec in FIELD_SPECS]
    assert len(field_names) == len(set(field_names)), "FIELD_SPECS has duplicate leaves"
    for leaf in V060_EXISTING_STATE_PHYSICS_LEAVES + V060_ADDITIVE_STATE_LEAVES:
        assert leaf in FIELD_SPECS_BY_LEAF, f"{leaf!r} missing from FIELD_SPECS"

    nested = nest_field_list(mp_physics=10, bl_pbl_physics=5, cu_physics=1)
    assert "qv" in {entry.leaf for entry in nested}, "qv must remain in nest core"
    assert len(nested) == len({entry.leaf for entry in nested}), "duplicate nest field entry"
    for entry in nested:
        assert entry.kind in {"prognostic", "moist", "scalar", "base"}


__all__ = [
    "PHYSICS_REGISTRY_VERSION",
    "SchemeOption",
    "ACCEPTED_MP_PHYSICS",
    "ACCEPTED_BL_PBL_PHYSICS",
    "ACCEPTED_SF_SFCLAY_PHYSICS",
    "ACCEPTED_CU_PHYSICS",
    "ACCEPTED_SF_SURFACE_PHYSICS",
    "ACCEPTED_RA_SW_PHYSICS",
    "ACCEPTED_RA_LW_PHYSICS",
    "ACCEPTED_NAMELIST_OPTIONS",
    "MP_SCHEMES",
    "PBL_SCHEMES",
    "SFCLAY_SCHEMES",
    "CU_SCHEMES",
    "SURFACE_SCHEMES",
    "RA_SW_SCHEMES",
    "RA_LW_SCHEMES",
    "MOIST_SPECIES",
    "MOIST_SPECIES_ADDITIVE",
    "MOIST_SPECIES_ALL",
    "MOIST_WRFOUT_NAME",
    "MP_MOIST_MEMBERS",
    "NUMBER_SPECIES",
    "NUMBER_SPECIES_EXISTING",
    "NUMBER_SPECIES_ADDITIVE",
    "NUMBER_REGISTRY_MEMBER",
    "NUMBER_WRFOUT_NAME",
    "MP_NUMBER_MEMBERS",
    "VOLUME_SPECIES",
    "VOLUME_REGISTRY_MEMBER",
    "VOLUME_WRFOUT_NAME",
    "ACCUMULATORS",
    "ACCUMULATORS_EXISTING",
    "ACCUMULATORS_ADDITIVE",
    "ACCUMULATOR_WRFOUT_NAME",
    "V060_EXISTING_STATE_PHYSICS_LEAVES",
    "V060_ADDITIVE_STATE_LEAVES",
    "RegistryFieldSpec",
    "FIELD_SPECS",
    "FIELD_SPECS_BY_LEAF",
    "CUMULUS_TENDENCY_MEMBERS",
    "CUMULUS_CARRY_MEMBERS",
    "PBL_CARRY_MEMBERS",
    "PBL_DIAGNOSTIC_MEMBERS",
    "LAND_CARRY_MEMBERS",
    "NOAH_CLASSIC_NUM_SOIL_LAYERS",
    "NestFieldEntry",
    "moist_species_for_mp",
    "number_species_for_mp",
    "state_leaves_for_mp",
    "wrfout_names_for_mp",
    "physics_state_append_order",
    "nest_field_list",
    "assert_registry_consistent",
]


if __name__ == "__main__":
    assert_registry_consistent()
    print(
        "ok physics_registry consistent:",
        f"version={PHYSICS_REGISTRY_VERSION}",
        f"moist={len(MOIST_SPECIES)}",
        f"numbers={len(NUMBER_SPECIES)}",
        f"accumulators={len(ACCUMULATORS)}",
        f"append={','.join(physics_state_append_order())}",
        f"nest_fields(mp=10,cu=1)={len(nest_field_list(mp_physics=10, cu_physics=1))}",
    )
