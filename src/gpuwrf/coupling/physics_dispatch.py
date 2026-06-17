"""v0.6.0 operational physics-suite dispatcher (manager integration patch).

This module is the single fail-closed router from a namelist's selected physics
options to the per-scheme JAX kernels that passed WRF-savepoint parity in the
v0.6.0 lanes. It owns the ``(family, option) -> scheme`` selection table, the
per-scheme calling convention + carry/tendency metadata sourced from the frozen
S0 registry, and the GPU-runnability flag the integrated forecast gate consumes.

Design contract (S0 ``V0.6.0-S0-PLAN.md``):

* Accepted options are exactly the frozen S0 accept-matrix
  (``physics_registry.ACCEPTED_*``). Anything outside it FAILS CLOSED here, in
  addition to ``io.namelist_check.validate_supported_namelist`` (defence in
  depth: the dispatcher must never silently fall through to a default scheme).
* Each scheme records ``gpu_runnable``. KF (cu=1), BMJ (cu=2), Grell-Freitas
  (cu=3), Tiedtke (cu=6), and the jit/vmap'd microphysics / PBL / surface-layer /
  Noah-MP/Noah-classic kernels are GPU-runnable. Tiedtke (cu=6) is GPU-batched via
  ``cumulus_tiedtke_jax`` and scan-wired (``CU_SCAN_ADAPTERS[6]``) when the
  runtime can provide WRF ``RQVFTEN`` from active flux-form moisture advection,
  savepoint-gated against unmodified ``module_cu_tiedtke.F``
  (proofs/v060/tiedtke_gpubatch_savepoint_parity.json). Grell-Freitas (cu=3) is
  the v0.9.0 GPU-batched jit/vmap port (``physics._gf_jax.gfdrv_batched``,
  stateless State->State, scan-wired ``CU_SCAN_ADAPTERS[3]``), savepoint-gated
  against unmodified ``module_cu_gf_*.F``
  (proofs/v060/gf_gpubatch_savepoint_parity.json). New Tiedtke (cu=16)
  shares the Tiedtke kernel but is NOT separately savepoint-gated by a distinct
  WRF source path, so it is flagged ``gpu_runnable=False`` and fail-closed in the
  operational scan. The integrated GPU forecast gate excludes any combo
  containing a non-GPU-runnable scheme.
* ``cugd_*`` correction (S0 manager carry-note): the inert ``cugd_*`` carry for
  Grell-Freitas is NOT threaded as State; GF and Tiedtke are routed through the
  combined ``R*CUTEN`` tendency + ``RAINCV``/``PRATEC`` family with shallow
  diagnostics, exactly as the frozen registry ``CUMULUS_TENDENCY_MEMBERS`` lists.

This module does NOT run a forecast and allocates nothing at import. It is a
pure selection/metadata layer; the operational step calls
:func:`resolve_physics_suite` to obtain the validated suite and routes through
the recorded scheme entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gpuwrf.contracts.physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
    CU_SCHEMES,
    CUMULUS_CARRY_MEMBERS,
    CUMULUS_TENDENCY_MEMBERS,
    LAND_CARRY_MEMBERS,
    MP_SCHEMES,
    PBL_SCHEMES,
    SFCLAY_SCHEMES,
    SURFACE_SCHEMES,
    state_leaves_for_mp,
)


# Default operational suite = the v0.2.0 validated baseline. When a namelist does
# not pin a physics option, the dispatcher resolves to these so the integrated
# operational step is byte-for-byte the validated v0.2.0 path until a caller
# explicitly selects a different scheme.
DEFAULT_MP_PHYSICS = 8        # Thompson
DEFAULT_BL_PBL_PHYSICS = 5    # MYNN
DEFAULT_SF_SFCLAY_PHYSICS = 5  # MYNN surface layer
DEFAULT_CU_PHYSICS = 0        # no cumulus (resolved grid-scale only)
DEFAULT_SF_SURFACE_PHYSICS = 4  # Noah-MP


# --- Surface-layer <-> PBL pairing rule (fail-closed) -------------------------
# WRF's own ``module_physics_init.F`` (CASE blocks under ``pbl_select``) FATAL-ERRORs
# unless each PBL is paired with a surface-layer scheme whose Monin-Obukhov forcing
# the PBL was designed to consume (the ``isfc`` requirement): the surface_driver
# writes HFX/QFX/BR/PSIM/PSIH/U10/V10/ZNT and pbl_driver reads those SAME fields
# (dyn_em/module_first_rk_step_part1.F:594 -> :1113).
#
# In THIS reimplementation the YSU(1)/ACM2(7)/BouLac(8)/Shin-Hong(11)/GBM(12)/MRF(99) PBL scan adapters
# (coupling.scan_adapters) re-derive the per-cell surface forcing they consume via
# the REVISED-MM5 surface layer (``_pbl_surface_forcing`` ->
# ``surface_layer.surface_layer_with_diagnostics``) because the frozen State carries
# only the B2 kinematic flux handles (ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/
# fltv) and NOT the stability functions PSIM/PSIH, the bulk Richardson BR, or U10/V10
# those PBL kernels also require. The revised-MM5 re-derivation is faithful ONLY when
# the SELECTED surface-layer scheme IS revised-MM5 (sf_sfclay=1). Pairing one of these
# PBLs with any OTHER surface layer would SILENTLY substitute revised-MM5 forcing for
# the requested scheme's forcing -- an honest-failure (presenting a different scheme's
# result as the requested one), so we FAIL CLOSED here.
#
# MYNN(5) and MYJ(2) are exempt: MYNN consumes the SELECTED scheme's kinematic flux
# handles directly from State (physics_couplers._surface_fluxes_from_state), and MYJ
# re-runs the Janjic surface layer it is mandatorily paired with (sf_sfclay=2). bl=0
# (no PBL) needs no surface forcing.
#
# The set below names the PBL options whose forcing is re-derived from revised-MM5,
# and hence are faithful ONLY with sf_sfclay_physics=1 (revised-MM5). This matches
# WRF's isfc==1 requirement for YSU/MRF (sf in {1,91}); we further restrict to {1}
# because only the revised-MM5 forcing path is wired into these adapters (the old-MM5
# sf=91 forcing is NOT separately threaded into the PBL re-derivation).
_PBL_REQUIRES_REVISED_MM5_SFCLAY: frozenset[int] = frozenset({1, 7, 8, 11, 12, 99})
_REVISED_MM5_SFCLAY_OPTION = 1


class UnsupportedSchemeSelection(ValueError):
    """Raised when a namelist selects a physics option the dispatcher cannot route."""


@dataclass(frozen=True)
class SchemeEntry:
    """One routable physics scheme: option metadata + calling convention."""

    family: str
    option: int
    name: str
    owner_module: str
    entrypoint: str
    convention: str  # "mp_flat" | "column_state" | "land_step" | "state_adapter" | "disabled"
    gpu_runnable: bool
    reads_state: tuple[str, ...] = ()
    writes_state: tuple[str, ...] = ()
    carry_members: tuple[str, ...] = ()
    tendency_members: tuple[str, ...] = ()
    accumulators: tuple[str, ...] = ()
    notes: str = ""


def _mp_entry(option: int, module: str, entrypoint: str, *, gpu: bool, adapter: bool = False) -> SchemeEntry:
    leaves = state_leaves_for_mp(option)
    return SchemeEntry(
        family="microphysics",
        option=option,
        name=MP_SCHEMES[option].name,
        owner_module=module,
        entrypoint=entrypoint,
        convention="state_adapter" if adapter else "mp_flat",
        gpu_runnable=gpu,
        reads_state=("theta", "qv", "p", *leaves),
        writes_state=("theta", *leaves),
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc"),
    )


# --- Microphysics (mp_physics) -------------------------------------------------
# mp=8 Thompson is wired through the EXISTING validated coupling.physics_couplers
# .thompson_adapter (State->State); the rest expose the S0 *_physics_tendency /
# *_tendency flat-array entrypoints returning a frozen PhysicsTendency.
_MP_ENTRIES: dict[int, SchemeEntry] = {
    0: SchemeEntry("microphysics", 0, "disabled/passive qv", "", "", "disabled", True,
                   reads_state=("qv",), writes_state=()),
    1: _mp_entry(1, "gpuwrf.physics.microphysics_kessler", "kessler_physics_tendency", gpu=True),
    2: _mp_entry(2, "gpuwrf.physics.microphysics_lin", "lin_physics_tendency", gpu=True),
    3: _mp_entry(3, "gpuwrf.physics.microphysics_wsm3", "wsm3_physics_tendency", gpu=True),
    4: _mp_entry(4, "gpuwrf.physics.microphysics_wsm5", "wsm5_physics_tendency", gpu=True),
    6: _mp_entry(6, "gpuwrf.physics.microphysics_wsm6", "wsm6_physics_tendency", gpu=True),
    8: _mp_entry(8, "gpuwrf.coupling.physics_couplers", "thompson_adapter", gpu=True, adapter=True),
    10: _mp_entry(10, "gpuwrf.physics.microphysics_morrison", "morrison_tendency", gpu=True),
    13: _mp_entry(13, "gpuwrf.physics.microphysics_sbu_ylin", "sbu_ylin_physics_tendency", gpu=True),
    14: _mp_entry(14, "gpuwrf.physics.microphysics_wdm5", "wdm5_physics_tendency", gpu=True),
    16: _mp_entry(16, "gpuwrf.physics.microphysics_wdm6", "wdm6_physics_tendency", gpu=True),
    # v0.17 WSM7 = WSM6 + separate precipitating hail (qh + hail_acc).
    24: _mp_entry(24, "gpuwrf.physics.microphysics_wsm7", "wsm7_physics_tendency", gpu=True),
    # v0.17 WDM7 = WDM6 double-moment + separate single-moment hail (qh + hail_acc).
    26: _mp_entry(26, "gpuwrf.physics.microphysics_wdm7", "wdm7_physics_tendency", gpu=True),
    # mp=28 aerosol-aware Thompson (v0.16): wired through the State adapter in
    # coupling.physics_couplers (mirrors mp=8), advancing the moist species +
    # Ni/Nr/Ns/Ng + the aerosol-aware prognostics Nc/nwfa/nifa and applying the
    # WRF fake surface aerosol emission each step.
    28: _mp_entry(28, "gpuwrf.coupling.physics_couplers", "thompson_aero_adapter", gpu=True, adapter=True),
    # mp=97 Goddard GCE single-moment 3-ice (gsfcgce): jit/vmap column port,
    # savepoint-parity-proven against unmodified phys/module_mp_gsfcgce.F
    # (proofs/v090/goddard_mp_r2_savepoint_parity.json). No new prognostic state.
    97: _mp_entry(97, "gpuwrf.physics.microphysics_goddard", "goddard_physics_tendency", gpu=True),
}

# --- PBL (bl_pbl_physics) ------------------------------------------------------
# bl=5 MYNN is wired through the EXISTING coupling.physics_couplers.mynn_adapter.
# bl=1 YSU / bl=7 ACM2 / bl=8 BouLac are the v0.6.0 jax.lax.scan-traceable /
# vmap-batched rewrites (physics.pbl_{ysu,acm2,boulac}.*_columns) -- GPU-operational and
# scan-wired via coupling.scan_adapters.PBL_SCAN_ADAPTERS. ``gpu_runnable=True`` is
# now genuine (the host-NumPy single-column path was replaced; per-case savepoint
# parity re-passes on the traceable path). The dispatcher entry point names the
# per-column ``step_*_column`` (which now also routes through the traceable kernel).
_PBL_ENTRIES: dict[int, SchemeEntry] = {
    0: SchemeEntry("pbl", 0, "disabled", "", "", "disabled", True),
    1: SchemeEntry("pbl", 1, PBL_SCHEMES[1].name, "gpuwrf.physics.pbl_ysu", "step_ysu_column",
                   "column_state", True,
                   reads_state=("u", "v", "theta", "qv"), writes_state=("u", "v", "theta", "qv")),
    2: SchemeEntry("pbl", 2, PBL_SCHEMES[2].name, "gpuwrf.physics.pbl_myj", "step_myj_pbl_column",
                   "column_state", True,
                   reads_state=("u", "v", "theta", "qv", "tke_pbl"),
                   writes_state=("u", "v", "theta", "qv"),
                   carry_members=("tke_pbl", "el_pbl"),
                   notes="MUST pair with sf_sfclay_physics=2 (Janjic Eta surface layer)."),
    # GFS(3): v0.17 jit/vmap-traceable port of phys/module_bl_gfs.F, scan-wired as a
    # State->State adapter (coupling.scan_adapters.gfs_pbl_adapter). Nonlocal-K, no
    # prognostic PBL carry; consumes the revised-MM5 surface forcing (sf_sfclay=1).
    3: SchemeEntry("pbl", 3, PBL_SCHEMES[3].name, "gpuwrf.coupling.scan_adapters", "gfs_pbl_adapter",
                   "state_adapter", True,
                   reads_state=("u", "v", "theta", "qv"), writes_state=("u", "v", "theta", "qv")),
    5: SchemeEntry("pbl", 5, PBL_SCHEMES[5].name, "gpuwrf.coupling.physics_couplers", "mynn_adapter",
                   "state_adapter", True,
                   reads_state=("u", "v", "theta", "qv", "qke"),
                   writes_state=("u", "v", "theta", "qv", "qke"), carry_members=("qke",)),
    7: SchemeEntry("pbl", 7, PBL_SCHEMES[7].name, "gpuwrf.physics.pbl_acm2", "step_acm2_column",
                   "column_state", True,
                   reads_state=("u", "v", "theta", "qv"), writes_state=("u", "v", "theta", "qv")),
    8: SchemeEntry("pbl", 8, PBL_SCHEMES[8].name, "gpuwrf.physics.pbl_boulac", "step_boulac_column",
                   "column_state", True,
                   reads_state=("u", "v", "theta", "qv", "qc", "qke"),
                   writes_state=("u", "v", "theta", "qv", "qc", "qke"), carry_members=("qke",)),
    # Shin-Hong(11): v0.18 JAX/vmap port of the scale-aware YSU-family PBL,
    # scan-wired as a State->State adapter. It consumes revised-MM5 surface
    # forcing and grid dx/dy for the scale-aware partition functions.
    11: SchemeEntry("pbl", 11, PBL_SCHEMES[11].name, "gpuwrf.coupling.scan_adapters", "shinhong_pbl_adapter",
                    "state_adapter", True,
                    reads_state=("u", "v", "theta", "qv", "qke"),
                    writes_state=("u", "v", "theta", "qv", "qke"), carry_members=("qke",)),
    # GBM(12): v0.18 JAX/vmap port of phys/module_bl_gbmpbl.F, scan-wired as a
    # State->State adapter. It consumes revised-MM5 surface forcing and advances
    # cloud water plus prognostic TKE (qke) in addition to the driving PBL fields.
    12: SchemeEntry("pbl", 12, PBL_SCHEMES[12].name, "gpuwrf.coupling.scan_adapters", "gbm_pbl_adapter",
                    "state_adapter", True,
                    reads_state=("u", "v", "theta", "qv", "qc", "qke"),
                    writes_state=("u", "v", "theta", "qv", "qc", "qke"), carry_members=("qke",)),
    # MRF(99): v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F, scan-wired as a
    # State->State adapter (coupling.scan_adapters.mrf_pbl_adapter). Nonlocal-K, no
    # prognostic PBL carry; consumes the revised-MM5 surface forcing (sf_sfclay=1).
    99: SchemeEntry("pbl", 99, PBL_SCHEMES[99].name, "gpuwrf.coupling.scan_adapters", "mrf_pbl_adapter",
                    "state_adapter", True,
                    reads_state=("u", "v", "theta", "qv"), writes_state=("u", "v", "theta", "qv")),
}

# --- Surface layer (sf_sfclay_physics) -----------------------------------------
# sf_sfclay=5 MYNN surface layer is wired through the existing
# coupling.physics_couplers.surface_adapter; sfclay=1 revised-MM5 and sfclay=7
# Pleim-Xiu expose step_*_column -> PhysicsStepResult writing the flux handles.
_SFCLAY_WRITES = ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv")
_SFCLAY_READS = ("u", "v", "theta", "qv", "t_skin", "soil_moisture")
_SFCLAY_ENTRIES: dict[int, SchemeEntry] = {
    0: SchemeEntry("surface_layer", 0, "disabled", "", "", "disabled", True),
    1: SchemeEntry("surface_layer", 1, SFCLAY_SCHEMES[1].name, "gpuwrf.physics.sfclay_revised_mm5",
                   "step_sfclay_revised_mm5_column", "column_state", True,
                   reads_state=_SFCLAY_READS, writes_state=_SFCLAY_WRITES),
    2: SchemeEntry("surface_layer", 2, SFCLAY_SCHEMES[2].name, "gpuwrf.physics.sfclay_janjic",
                   "step_janjic_sfclay_column", "column_state", True,
                   reads_state=(*_SFCLAY_READS, "tke_pbl"), writes_state=_SFCLAY_WRITES,
                   carry_members=("tke_pbl",),
                   notes="MUST pair with bl_pbl_physics=2 (MYJ PBL)."),
    5: SchemeEntry("surface_layer", 5, SFCLAY_SCHEMES[5].name, "gpuwrf.coupling.physics_couplers",
                   "surface_adapter", "state_adapter", True,
                   reads_state=_SFCLAY_READS, writes_state=_SFCLAY_WRITES),
    7: SchemeEntry("surface_layer", 7, SFCLAY_SCHEMES[7].name, "gpuwrf.physics.sfclay_pleim_xiu",
                   "step_pxsfclay_column", "column_state", True,
                   reads_state=_SFCLAY_READS, writes_state=_SFCLAY_WRITES),
    # v0.13 Tier-3: NCEP-GFS (3) + old-MM5 (91) surface layers. Both are jit/vmap
    # batched columns that produce HFX/QFX/USTAR; coupling.scan_adapters wraps them
    # to write the B2 kinematic flux handles. fp64 pristine-WRF oracle-validated.
    3: SchemeEntry("surface_layer", 3, SFCLAY_SCHEMES[3].name, "gpuwrf.physics.sfclay_gfs",
                   "sf_gfs_columns", "column_state", True,
                   reads_state=_SFCLAY_READS, writes_state=_SFCLAY_WRITES,
                   notes="NCEP-GFS Monin-Obukhov surface layer; scan-wired via "
                   "coupling.scan_adapters.gfs_sfclay_adapter."),
    91: SchemeEntry("surface_layer", 91, SFCLAY_SCHEMES[91].name, "gpuwrf.physics.sfclay_old_mm5",
                    "sfclay_old_mm5_columns", "column_state", True,
                    reads_state=_SFCLAY_READS, writes_state=_SFCLAY_WRITES,
                    notes="Classic MM5 4-regime Monin-Obukhov surface layer; scan-wired via "
                    "coupling.scan_adapters.sfclay_old_mm5_adapter."),
}

# --- Cumulus (cu_physics) ------------------------------------------------------
# KF (cu=1), BMJ (cu=2), Grell-Freitas (cu=3), and Tiedtke (cu=6) are jit/vmap'd
# operational GPU cumulus paths (scan-wired via CU_SCAN_ADAPTERS). Tiedtke requires
# runtime RQVFTEN/RQVBLTEN from active moisture advection and the PBL qv increment.
# Grell-Freitas (cu=3) is the v0.9.0 GPU-batched jit/vmap port of the scale-aware
# closure-ensemble kernel (physics._gf_jax.gfdrv_batched), savepoint-parity gated.
# New Tiedtke (cu=16) and the SAS family (cu=4/94/95/96) stay
# gpu_runnable=False / fail-closed until their distinct WRF source paths pass
# traceable JAX parity. All route through the combined R*CUTEN tendency +
# RAINCV/PRATEC family (S0 cugd_* correction: no inert cugd_* State carry for GF).
_CU_ENTRIES: dict[int, SchemeEntry] = {
    0: SchemeEntry("cumulus", 0, "disabled", "", "", "disabled", True),
    1: SchemeEntry("cumulus", 1, CU_SCHEMES[1].name, "gpuwrf.physics.cumulus_kf", "step_kf_column",
                   "column_state", True,
                   reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                   writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
                   carry_members=CUMULUS_CARRY_MEMBERS[1], tendency_members=CUMULUS_TENDENCY_MEMBERS[1],
                   accumulators=("rainc_acc",)),
    2: SchemeEntry("cumulus", 2, CU_SCHEMES[2].name, "gpuwrf.physics.cumulus_bmj", "step_bmj_column",
                   "column_state", True,
                   reads_state=("theta", "qv", "p", "ph", "xland"),
                   writes_state=("theta", "qv"),
                   carry_members=CUMULUS_CARRY_MEMBERS[2],
                   tendency_members=CUMULUS_TENDENCY_MEMBERS[2],
                   accumulators=("rainc_acc",),
                   notes="Frozen-contract extension for BMJ adjustment cumulus; carries CLDEFI."),
    3: SchemeEntry("cumulus", 3, CU_SCHEMES[3].name, "gpuwrf.physics._gf_jax",
                   "gfdrv_batched", "column_state", True,
                   reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                   writes_state=("theta", "qv", "qc", "qi"),
                   tendency_members=CUMULUS_TENDENCY_MEMBERS[3], accumulators=("rainc_acc",),
                   notes="GPU-batched (jit/vmap) scale-aware Grell-Freitas; scan-wired via "
                   "CU_SCAN_ADAPTERS[3] (stateless State->State); savepoint-gated vs unmodified "
                   "module_cu_gf_deep.F/module_cu_gf_sh.F/module_cu_gf_wrfdrv.F "
                   "(proofs/v060/gf_gpubatch_savepoint_parity.json). cugd_* carry DROPPED per S0 "
                   "correction -- routed via R*CUTEN + RAINCV/PRATEC + shallow diags."),
    4: SchemeEntry("cumulus", 4, CU_SCHEMES[4].name, "gpuwrf.physics.cumulus_sas",
                   "step_sas_family_column", "column_state", False,
                   reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "ph"),
                   writes_state=("theta", "qv", "qc", "qi", "u", "v"),
                   tendency_members=CUMULUS_TENDENCY_MEMBERS[4], accumulators=("rainc_acc",),
                   notes="v0.17 reference-only: fp64 pristine-WRF oracle exists for "
                   "phys/module_cu_scalesas.F:CU_SCALESAS, but the shared JAX "
                   "endpoint is RED vs oracle (proofs/v017/sas_family_parity.json); "
                   "fail-closed in the operational scan."),
    6: SchemeEntry("cumulus", 6, CU_SCHEMES[6].name, "gpuwrf.physics.cumulus_tiedtke_jax", "tiedtke_column_jax",
                   "column_state", True,
                   reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                   writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
                   tendency_members=CUMULUS_TENDENCY_MEMBERS[6], accumulators=("rainc_acc",),
                   notes="GPU-batched (jit/vmap) Tiedtke; scan-wired via CU_SCAN_ADAPTERS[6] "
                   "when runtime supplies WRF RQVFTEN/RQVBLTEN from active moisture "
                   "advection/PBL forcing; "
                   "savepoint-gated vs unmodified module_cu_tiedtke.F "
                   "(proofs/v060/tiedtke_gpubatch_savepoint_parity.json); tendency-only carry."),
    16: SchemeEntry("cumulus", 16, CU_SCHEMES[16].name, "gpuwrf.physics.cumulus_tiedtke", "step_tiedtke_column",
                    "column_state", False,
                    reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                    writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
                    tendency_members=CUMULUS_TENDENCY_MEMBERS[16], accumulators=("rainc_acc",),
                    notes="New Tiedtke option spec; shares the Tiedtke kernel but is NOT separately "
                    "savepoint-gated by a distinct WRF source path -- accepted/fail-closed in the "
                    "operational GPU scan, NOT parity-proven for cu=16 specifically."),
    93: SchemeEntry("cumulus", 93, CU_SCHEMES[93].name, "gpuwrf.physics.cumulus_grell_devenyi",
                    "step_grell_devenyi_column", "column_state", False,
                    reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                    writes_state=("theta", "qv", "qc", "qi"),
                    tendency_members=CUMULUS_TENDENCY_MEMBERS[93], accumulators=("rainc_acc",),
                    notes="v0.18 RED/reference-only. A pristine-WRF GRELLDRV harness/savepoints exist, "
                    "but current trial columns are null-only; no source-specific traceable JAX endpoint "
                    "has passed parity; not scan-wired."),
    94: SchemeEntry("cumulus", 94, CU_SCHEMES[94].name, "gpuwrf.physics.cumulus_sas",
                    "step_sas_family_column", "column_state", False,
                    reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "ph"),
                    writes_state=("theta", "qv", "qc", "qi", "u", "v"),
                    tendency_members=CUMULUS_TENDENCY_MEMBERS[94], accumulators=("rainc_acc",),
                    notes="v0.17 reference-only: fp64 pristine-WRF oracle exists for "
                    "phys/module_cu_sas.F:CU_SAS, but the shared JAX endpoint is "
                    "RED vs oracle (proofs/v017/sas_family_parity.json); fail-closed."),
    95: SchemeEntry("cumulus", 95, CU_SCHEMES[95].name, "gpuwrf.physics.cumulus_sas",
                    "step_sas_family_column", "column_state", False,
                    reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "ph"),
                    writes_state=("theta", "qv", "qc", "qi", "u", "v"),
                    tendency_members=CUMULUS_TENDENCY_MEMBERS[95], accumulators=("rainc_acc",),
                    notes="v0.17 reference-only: fp64 pristine-WRF oracle exists for "
                    "phys/module_cu_osas.F:CU_OSAS, but the shared JAX endpoint is "
                    "RED vs oracle (proofs/v017/sas_family_parity.json); fail-closed."),
    96: SchemeEntry("cumulus", 96, CU_SCHEMES[96].name, "gpuwrf.physics.cumulus_sas",
                    "step_sas_family_column", "column_state", False,
                    reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "ph"),
                    writes_state=("theta", "qv", "qc", "qi", "u", "v"),
                    tendency_members=CUMULUS_TENDENCY_MEMBERS[96], accumulators=("rainc_acc",),
                    notes="v0.17 reference-only: fp64 pristine-WRF oracle exists for "
                    "phys/module_cu_nsas.F:CU_NSAS, but the shared JAX endpoint is "
                    "RED vs oracle (proofs/v017/sas_family_parity.json); fail-closed."),
    99: SchemeEntry("cumulus", 99, CU_SCHEMES[99].name, "gpuwrf.physics.cumulus_kf_previous",
                    "step_previous_kf_column", "column_state", False,
                    reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                    writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
                    carry_members=CUMULUS_CARRY_MEMBERS[99],
                    tendency_members=CUMULUS_TENDENCY_MEMBERS[99], accumulators=("rainc_acc",),
                    notes="v0.17 RED/reference-only candidate: reuses the KF-eta family endpoint for "
                    "comparison only; not parity-proven against phys/module_cu_kf.F:KFCPS and not "
                    "scan-wired."),
}

# --- Land surface (sf_surface_physics) -----------------------------------------
# sf_surface=4 Noah-MP is the EXISTING coupling.noahmp_surface_hook path (wired in
# operational_mode via namelist.use_noahmp). sf_surface=2 Noah classic exposes the
# lsm_noah_classic.sflx_step land step + a 4-layer land carry (S0 land carry).
_SURFACE_ENTRIES: dict[int, SchemeEntry] = {
    0: SchemeEntry("land_surface", 0, "disabled", "", "", "disabled", True),
    # v0.17 slab LSM (1): jit/vmap column port + fp64 oracle, now OPERATIONAL via
    # coupling.slab_surface_hook.slab_surface_step (5-layer TSLB land carry +
    # GSW/GLW radiation forcing + TMN/THC/EMISS static bundle; FLHC/FLQC recovered
    # from the resident surface-layer kinematic flux handles).
    1: SchemeEntry("land_surface", 1, SURFACE_SCHEMES[1].name, "gpuwrf.coupling.slab_surface_hook",
                   "slab_surface_step", "state_adapter", True,
                   reads_state=("t_skin", "mavail"),
                   writes_state=("t_skin",),
                   carry_members=("tslb",),
                   notes="5-layer thermal-diffusion slab LSM; fp64 pristine-WRF oracle "
                   "(proofs/v013/t3_surface_lsm_oracle.json, SLAB1D) -- scan-wired in "
                   "runtime.operational_mode via an explicit slab_static SlabStaticBundle."),
    2: SchemeEntry("land_surface", 2, SURFACE_SCHEMES[2].name, "gpuwrf.physics.lsm_noah_classic",
                   "sflx_step", "land_step", True,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin", "soil_moisture", "mavail"),
                   carry_members=("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn"),
                   notes="Owns a 4-layer (num_soil_layers=4) land carry; does NOT reinterpret the "
                   "2-D State.soil_moisture as a 4-layer field (S0 land carry rule)."),
    # v0.17 RUC LSM (3): REFERENCE-ONLY. A fp64 pristine-WRF single-column oracle is
    # staged (proofs/v017/oracle/ruclsm; LSMRUC->SOILVEGIN->SFCTMP, unmodified source),
    # but the ~7.5k-LOC multi-layer soil/snow JAX column kernel is a documented
    # carry-over, so it is gpu_runnable=False / fail-closed in the operational scan.
    3: SchemeEntry("land_surface", 3, SURFACE_SCHEMES[3].name, "gpuwrf.physics.lsm_ruc",
                   "ruc_column", "land_step", False,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin", "soil_moisture", "mavail"),
                   carry_members=LAND_CARRY_MEMBERS[3],
                   notes="RUC multi-layer soil/snow LSM; fp64 pristine-WRF oracle staged "
                   "(proofs/v017/oracle/ruclsm) but the faithful JAX column port "
                   "(SFCTMP+SOIL+SOILTEMP+SOILMOIST+SOILPROP+TRANSF) is a carry-over -- "
                   "accepted/fail-closed in the operational GPU scan, NOT parity-proven as a "
                   "JAX kernel yet."),
    4: SchemeEntry("land_surface", 4, SURFACE_SCHEMES[4].name, "gpuwrf.coupling.noahmp_surface_hook",
                   "noahmp_surface_step", "state_adapter", True,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin", "soil_moisture", "mavail"),
                   carry_members=("NoahMPLandState",)),
    # v0.17 Pleim-Xiu LSM (7): fp64-oracle-validated SURFPX+QFLUX port, OPERATIONAL
    # via coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step (2-layer ISBA land
    # carry + GSW/GLW radiation + PleimXiuStaticBundle; pairs with sf_sfclay=7).
    7: SchemeEntry("land_surface", 7, SURFACE_SCHEMES[7].name, "gpuwrf.coupling.pleim_xiu_surface_hook",
                   "pleim_xiu_surface_step", "state_adapter", True,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin",),
                   carry_members=("tg", "t2", "wg", "w2", "wr")),
    # v0.17 SSiB LSM (8): REFERENCE-ONLY. A fp64 pristine-WRF single-column oracle is
    # staged (proofs/v017/oracle/ssib; the unmodified SSIB driver), but the ~6.6k-LOC
    # coupled SiB canopy/soil/4-level-snow JAX column kernel is a documented carry-over,
    # so it is gpu_runnable=False / fail-closed in the operational scan.
    8: SchemeEntry("land_surface", 8, SURFACE_SCHEMES[8].name, "gpuwrf.physics.lsm_ssib",
                   "ssib_column", "land_step", False,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin", "soil_moisture", "mavail"),
                   carry_members=LAND_CARRY_MEMBERS[8],
                   notes="SSiB SiB biophysical canopy/soil/snow LSM; fp64 pristine-WRF oracle "
                   "staged (proofs/v017/oracle/ssib) but the faithful JAX column port "
                   "(TEMRS1/TEMRS2+UPDAT1+RADAB+STOMA1+INTERC+STRES1+NEWTON) is a carry-over -- "
                   "accepted/fail-closed in the operational GPU scan, NOT parity-proven as a "
                   "JAX kernel yet."),
}


_FAMILY_TABLE: Mapping[str, tuple[dict[int, SchemeEntry], tuple[int, ...]]] = {
    "microphysics": (_MP_ENTRIES, tuple(ACCEPTED_MP_PHYSICS)),
    "pbl": (_PBL_ENTRIES, tuple(ACCEPTED_BL_PBL_PHYSICS)),
    "surface_layer": (_SFCLAY_ENTRIES, tuple(ACCEPTED_SF_SFCLAY_PHYSICS)),
    "cumulus": (_CU_ENTRIES, tuple(ACCEPTED_CU_PHYSICS)),
    "land_surface": (_SURFACE_ENTRIES, tuple(ACCEPTED_SF_SURFACE_PHYSICS)),
}

_NAMELIST_KEY = {
    "microphysics": "mp_physics",
    "pbl": "bl_pbl_physics",
    "surface_layer": "sf_sfclay_physics",
    "cumulus": "cu_physics",
    "land_surface": "sf_surface_physics",
}

_DEFAULT_OPTION = {
    "microphysics": DEFAULT_MP_PHYSICS,
    "pbl": DEFAULT_BL_PBL_PHYSICS,
    "surface_layer": DEFAULT_SF_SFCLAY_PHYSICS,
    "cumulus": DEFAULT_CU_PHYSICS,
    "land_surface": DEFAULT_SF_SURFACE_PHYSICS,
}


def scheme_entry(family: str, option: int) -> SchemeEntry:
    """Return the routable scheme entry for ``(family, option)``, fail-closed."""

    if family not in _FAMILY_TABLE:
        raise UnsupportedSchemeSelection(f"unknown physics family {family!r}")
    table, accepted = _FAMILY_TABLE[family]
    opt = int(option)
    if opt not in accepted or opt not in table:
        raise UnsupportedSchemeSelection(
            f"{_NAMELIST_KEY[family]}={opt} is not a routable v0.6.0 scheme; "
            f"accepted: {sorted(accepted)}"
        )
    return table[opt]


@dataclass(frozen=True)
class PhysicsSuite:
    """Validated set of selected schemes for one operational run.

    ``gpu_gate_ready`` is True only when every NON-disabled selected scheme is
    GPU-runnable -- the integrated GPU forecast gate excludes any combo with a
    fail-closed scheme (Grell-Freitas cu=3 / New Tiedtke cu=16).
    """

    microphysics: SchemeEntry
    pbl: SchemeEntry
    surface_layer: SchemeEntry
    cumulus: SchemeEntry
    land_surface: SchemeEntry

    @property
    def entries(self) -> tuple[SchemeEntry, ...]:
        return (self.microphysics, self.pbl, self.surface_layer, self.cumulus, self.land_surface)

    @property
    def gpu_gate_ready(self) -> bool:
        return all(e.gpu_runnable for e in self.entries if e.convention != "disabled")

    @property
    def non_gpu_schemes(self) -> tuple[str, ...]:
        return tuple(
            f"{_NAMELIST_KEY[e.family]}={e.option} ({e.name})"
            for e in self.entries
            if e.convention != "disabled" and not e.gpu_runnable
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schemes": {
                _NAMELIST_KEY[e.family]: {
                    "option": e.option,
                    "name": e.name,
                    "module": e.owner_module,
                    "entrypoint": e.entrypoint,
                    "convention": e.convention,
                    "gpu_runnable": e.gpu_runnable,
                }
                for e in self.entries
            },
            "gpu_gate_ready": self.gpu_gate_ready,
            "non_gpu_schemes": list(self.non_gpu_schemes),
        }


def _option_from(config: Any, family: str) -> int:
    """Read a family's option from a namelist-like config, defaulting per family."""

    key = _NAMELIST_KEY[family]
    value = None
    if isinstance(config, Mapping):
        if key in config:
            value = config[key]
        else:
            for section in config.values():
                if isinstance(section, Mapping) and key in section:
                    value = section[key]
                    break
    else:
        value = getattr(config, key, None)
    if value is None:
        # Noah-MP back-compat: the operational namelist toggles land via use_noahmp
        # rather than an explicit sf_surface_physics field. Honor it so an existing
        # v0.2.0 namelist resolves Noah-MP vs the bulk surface path correctly.
        if family == "land_surface":
            use_noahmp = (
                config.get("use_noahmp") if isinstance(config, Mapping)
                else getattr(config, "use_noahmp", None)
            )
            if use_noahmp is True:
                return 4
            if use_noahmp is False:
                return 2
        return _DEFAULT_OPTION[family]
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return int(value)


def resolve_physics_suite(config: Any) -> PhysicsSuite:
    """Resolve a namelist/config to a validated :class:`PhysicsSuite` (fail-closed).

    ``config`` may be an ``OperationalNamelist``, a flat mapping, or a nested
    WRF-style ``{"physics": {...}}`` mapping. Each family's option is read (with
    the v0.2.0 baseline default when unset) and routed via :func:`scheme_entry`,
    which raises :class:`UnsupportedSchemeSelection` on anything outside the S0
    accept-matrix. This is the loud, single dispatch decision point.
    """

    pbl_opt = _option_from(config, "pbl")
    sfclay_opt = _option_from(config, "surface_layer")
    if (pbl_opt == 2) != (sfclay_opt == 2):
        raise UnsupportedSchemeSelection(
            "MYJ pairing violation: bl_pbl_physics=2 and sf_sfclay_physics=2 "
            "must be selected together; no fallback surface-layer/PBL pairing is WRF-faithful."
        )
    if (
        pbl_opt in _PBL_REQUIRES_REVISED_MM5_SFCLAY
        and sfclay_opt != _REVISED_MM5_SFCLAY_OPTION
    ):
        raise UnsupportedSchemeSelection(
            f"surface-layer/PBL pairing violation: bl_pbl_physics={pbl_opt} "
            f"(YSU/ACM2/BouLac/Shin-Hong/GBM/MRF) re-derives its surface-layer forcing via the "
            f"revised-MM5 surface layer, so it is faithful ONLY with "
            f"sf_sfclay_physics=1 (revised-MM5); selected sf_sfclay_physics="
            f"{sfclay_opt}. Running this pairing would SILENTLY substitute revised-MM5 "
            f"surface forcing for the requested scheme. Select sf_sfclay_physics=1, "
            f"or use bl_pbl_physics=5 (MYNN, consumes the selected scheme's State flux "
            f"handles) / bl_pbl_physics=2 (MYJ, pairs with sf_sfclay_physics=2)."
        )
    return PhysicsSuite(
        microphysics=scheme_entry("microphysics", _option_from(config, "microphysics")),
        pbl=scheme_entry("pbl", pbl_opt),
        surface_layer=scheme_entry("surface_layer", sfclay_opt),
        cumulus=scheme_entry("cumulus", _option_from(config, "cumulus")),
        land_surface=scheme_entry("land_surface", _option_from(config, "land_surface")),
    )


def dispatch_matrix() -> dict[str, Any]:
    """Return the full routable dispatch matrix (one row per accepted option).

    Used by the integration proof object: lists every accepted physics option,
    the scheme it routes to, its calling convention, and GPU-runnability.
    """

    rows: list[dict[str, Any]] = []
    for family, (table, accepted) in _FAMILY_TABLE.items():
        for opt in accepted:
            if opt not in table:
                # Reference-only options (e.g. v0.13 Tier-3 cu=5/14) are
                # namelist-accepted but NOT operationally routable -- they
                # fail-close in the scan. They have no routable entry, so they
                # are excluded from the routable dispatch matrix.
                continue
            e = table[opt]
            rows.append(
                {
                    "namelist_key": _NAMELIST_KEY[family],
                    "family": family,
                    "option": opt,
                    "name": e.name,
                    "module": e.owner_module,
                    "entrypoint": e.entrypoint,
                    "convention": e.convention,
                    "gpu_runnable": e.gpu_runnable,
                    "carry_members": list(e.carry_members),
                    "tendency_members": list(e.tendency_members),
                    "accumulators": list(e.accumulators),
                    "notes": e.notes,
                }
            )
    return {
        "default_suite": {
            "mp_physics": DEFAULT_MP_PHYSICS,
            "bl_pbl_physics": DEFAULT_BL_PBL_PHYSICS,
            "sf_sfclay_physics": DEFAULT_SF_SFCLAY_PHYSICS,
            "cu_physics": DEFAULT_CU_PHYSICS,
            "sf_surface_physics": DEFAULT_SF_SURFACE_PHYSICS,
            "note": "v0.2.0 validated baseline = Thompson/MYNN/MYNN-sfclay/Noah-MP, no cumulus.",
        },
        "rows": rows,
    }


__all__ = [
    "DEFAULT_MP_PHYSICS",
    "DEFAULT_BL_PBL_PHYSICS",
    "DEFAULT_SF_SFCLAY_PHYSICS",
    "DEFAULT_CU_PHYSICS",
    "DEFAULT_SF_SURFACE_PHYSICS",
    "UnsupportedSchemeSelection",
    "SchemeEntry",
    "PhysicsSuite",
    "scheme_entry",
    "resolve_physics_suite",
    "dispatch_matrix",
]
