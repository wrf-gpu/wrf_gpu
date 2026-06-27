"""Machine-readable WRF v4 scheme support catalog -- the public honesty contract.

This module answers, for *every* WRF v4 option of the common physics/dynamics
namelist groups (plus the major feature switches such as WRF-Chem, WRF-Fire,
FDDA, moving nests and stochastic physics), the single question a
WRF developer evaluating this port cares about:

    "If I set this in my namelist, does the GPU port run it, refuse it with a
     named reason, or is it a documented out-of-scope decision?"

Every option resolves to exactly one :class:`SupportStatus`:

* ``IMPLEMENTED``            -- operationally GPU-scan-wired and run normally.
* ``REFERENCE_ONLY``        -- a recognized WRF v4 scheme with an oracle-backed
                               reference path (GREEN parity where claimed, or an
                               explicitly measured RED gap) that is NOT yet
                               wired into the operational GPU scan. It is
                               *accepted* by the namelist validator (so a
                               reference / single-column comparison can be run)
                               but the operational forecast scan fail-closes it
                               loudly with a named reason -- never a silent wrong
                               result.
* ``RECOGNIZED_FAIL_CLOSED``-- a valid WRF v4 option that the port does not
                               implement at all. Selecting it fails closed with a
                               message naming the WRF scheme, the reason, and the
                               supported alternative.
* ``RECOGNIZED_APPROXIMATED``-- a recognized WRF v4 *cadence* control whose
                               requested value the port does not honor exactly,
                               but whose effect is a documented CONSERVATIVE
                               approximation rather than a wrong-scheme
                               substitution: the cumulus/PBL cadence keys
                               (``cudt``/``bldt``) ask the port to sub-step those
                               physics every N minutes, but the GPU port calls
                               them EVERY dynamics step (more frequent than
                               requested). Selecting a positive cadence does NOT
                               fail closed -- the run PROCEEDS and a WARNING names
                               the approximation. This mirrors the operational
                               pipeline, which already runs cumulus/PBL every
                               step regardless of ``cudt``/``bldt``. It is NEVER
                               used for a genuine wrong-substitution (a different
                               scheme / unimplemented advection variant): those
                               stay ``RECOGNIZED_FAIL_CLOSED``.
* ``OUT_OF_SCOPE``          -- a documented design decision NOT to port this
                               capability (coupled chemistry, wildfire, hydrology,
                               moving/vortex-following nests, FDDA/4DVAR
                               nudging, stochastic physics).
                               Selecting it fails closed with the scope decision
                               and the reason.

Honesty rules followed when authoring this catalog (do not relax them):

* ``IMPLEMENTED`` is asserted ONLY for options that are actually threaded into
  the operational GPU scan. The ground truth was read from
  ``runtime.operational_mode._SCAN_WIRED_OPTIONS`` and the adapter registries in
  ``coupling.scan_adapters`` (MP/CU/PBL/SFCLAY), and cross-checked against
  ``contracts.physics_registry`` (the frozen accept-matrix). See
  ``assert_catalog_consistent`` for the machine-checked invariants that keep this
  module from drifting away from those authorities.
* When a scheme has oracle-backed reference evidence but is not operationally
  wired, it is ``REFERENCE_ONLY`` (a caveat), never ``IMPLEMENTED`` (an
  over-claim). The reason must say whether the comparison is GREEN or RED.
* When in doubt about a scheme's support, the safe classification is
  ``RECOGNIZED_FAIL_CLOSED`` (refuse loudly), never ``IMPLEMENTED``.

The full WRF v4 code->name enumeration is owned by
``gpuwrf.io.wrf_scheme_catalog`` (transcribed from ``WRF/run/README.namelist``).
This module classifies each of those codes. Feature-switch keys that have no
integer-enumerated WRF catalog (e.g. ``chem_opt``, ``grid_fdda``, ``ifire``,
``sf_ocean_physics``) are classified directly as ``OUT_OF_SCOPE`` truthy switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from gpuwrf.contracts.physics_registry import ACCEPTED_NAMELIST_OPTIONS
from gpuwrf.io.wrf_scheme_catalog import (
    WRF_PARAM_LABEL,
    WRF_SCHEME_CATALOG,
    wrf_scheme_name,
)


class SupportStatus(str, Enum):
    """Support classification for a WRF namelist scheme/feature selection."""

    IMPLEMENTED = "implemented"
    REFERENCE_ONLY = "reference_only"
    RECOGNIZED_APPROXIMATED = "recognized_approximated"
    RECOGNIZED_FAIL_CLOSED = "recognized_fail_closed"
    OUT_OF_SCOPE = "out_of_scope"

    @property
    def passes_namelist_check(self) -> bool:
        """True iff this status is *accepted* by ``validate_namelist``.

        ``IMPLEMENTED`` and ``REFERENCE_ONLY`` pass the namelist validator
        (REFERENCE_ONLY then fail-closes in the operational scan with a named
        reason). ``RECOGNIZED_APPROXIMATED`` also passes -- the run PROCEEDS,
        with a warning naming the conservative approximation (cumulus/PBL
        cadence run every step). ``RECOGNIZED_FAIL_CLOSED`` and ``OUT_OF_SCOPE``
        are rejected at the namelist layer so the user fails fast with a helpful
        message.
        """

        return self in (
            SupportStatus.IMPLEMENTED,
            SupportStatus.REFERENCE_ONLY,
            SupportStatus.RECOGNIZED_APPROXIMATED,
        )


@dataclass(frozen=True)
class SchemeSupport:
    """Support classification for one ``key=code`` (or one feature switch).

    ``alternative`` is the supported scheme / transition recipe surfaced to the
    user; ``reason`` explains *why* the option is not run on the operational GPU
    scan. ``wrf_name`` is the WRF scheme name when ``code`` is an enumerated WRF
    v4 option (``None`` for boolean feature switches).
    """

    key: str
    code: int | bool
    status: SupportStatus
    reason: str
    alternative: str
    wrf_name: str | None = None


# --------------------------------------------------------------------------- #
# Ground truth: operationally GPU-scan-wired options.                         #
# Mirrors runtime.operational_mode._SCAN_WIRED_OPTIONS +                      #
# coupling.scan_adapters.{MP,CU,SFCLAY,PBL}_SCAN_ADAPTERS, verified           #
# 2026-06-07. assert_catalog_consistent() enforces these match the           #
# physics_registry accept-matrix and the WRF v4 catalog.                      #
# --------------------------------------------------------------------------- #
_IMPLEMENTED: Mapping[str, frozenset[int]] = {
    # mp=13 (SBU-YLin) is the v0.18-harvested sbu_ylin single-moment scheme
    # (savepoint-parity-proven against unmodified phys/module_mp_sbu_ylin.F,
    # proofs/v017/sbu_ylin oracle; scan-wired).
    # mp=24 WSM7 (WSM6 + separate precipitating hail) + mp=26 WDM7 (WDM6
    # double-moment + separate single-moment hail) are GPU scan-wired
    # (coupling.scan_adapters.{wsm7,wdm7}_adapter; the qh hail State substrate
    # ADR-032 carries them end-to-end), savepoint-parity-proven against the
    # unmodified phys/module_mp_wsm7.F (proofs/v013) and module_mp_wdm7.F
    # (proofs/v013_wdm7), 6/6 PASS each.
    # mp=28 (aerosol-aware Thompson) is the v0.16 thompson_aero_adapter
    # (coupling.physics_couplers; WRF grid-savepoint parity-gated,
    # proofs/v016/thompson_aero_savepoint_parity.json; scan-wired in
    # runtime.operational_mode._SCAN_WIRED_OPTIONS / _physics_step_forcing).
    # mp=97 Goddard GCE (gsfcgce, single-moment 3-ice) is operationally scan-wired
    # (coupling.scan_adapters.MP_SCAN_ADAPTERS[97] = goddard_adapter, plain
    # State->State on the existing moist substrate), savepoint-parity-proven against
    # unmodified phys/module_mp_gsfcgce.F (proofs/v090/goddard_mp_r2_savepoint_parity.json).
    "mp_physics": frozenset({0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 24, 26, 28, 97}),
    "cu_physics": frozenset({0, 1, 2, 3, 6}),
    # bl=2 MYJ + sf=2 Janjic Eta are the v0.13 traceable MYJ pair (operationally
    # scan-wired via physics.myj_adapters + runtime.operational_mode; mandatory pair).
    # bl=99 MRF is the v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F
    # (savepoint-parity gated, proofs/v013/mrf_oracle.py); consumes the revised-MM5
    # surface layer (sf_sfclay=1), no new surface partner needed.
    # bl=3 GFS is the v0.17 jit/vmap-traceable port of phys/module_bl_gfs.F
    # (BL_GFS -> MONINP, savepoint-parity gated, proofs/v017/gfs_oracle.py ~1e-13);
    # nonlocal-K, consumes the revised-MM5 surface layer (sf_sfclay=1).
    # bl=9 CAM-UW is the v0.22 CAM5 UW diagnostic-TKE / implicit vertical-
    # diffusion endpoint. It is operationally scan-wired with an idealized
    # source-present proof, not full pristine-WRF CAM-stack savepoint parity.
    # bl=11 Shin-Hong is the v0.18 JAX/vmap scale-aware PBL port: dynamics-green
    # operational, with explicit non-driving TKE/EL diagnostic caveat vs the v090
    # PARTIAL reference (TKE rel ~=0.285, EL rel ~=0.013). bl=12 GBM is the
    # v0.18 JAX/vmap moist prognostic-TKE PBL port.
    "bl_pbl_physics": frozenset({0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 99}),
    # sf_sfclay 3 (NCEP-GFS) + 91 (old-MM5) are v0.13 Tier-3 scan-wired surface
    # layers (coupling.scan_adapters.{gfs_sfclay_adapter,sfclay_old_mm5_adapter};
    # fp64 pristine-WRF oracle-validated; B2 kinematic flux handles).
    "sf_sfclay_physics": frozenset({0, 1, 2, 3, 5, 7, 91}),
    # sf_surface=1 (5-layer thermal-diffusion slab LSM) is v0.17 operationally
    # scan-wired (coupling.slab_surface_hook.slab_surface_step over the fp64
    # pristine-WRF-oracle-validated physics.lsm_slab.slab_columns SLAB1D port;
    # advances the 5-layer TSLB land carry from GSW/GLW + a TMN/THC/EMISS bundle).
    # sf_surface=7 (Pleim-Xiu 2-layer ISBA LSM) is v0.17 operationally scan-wired
    # (coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step over the fp64
    # pristine-WRF-oracle-validated physics.lsm_pleim_xiu SURFPX+QFLUX port).
    "sf_surface_physics": frozenset({0, 1, 2, 4, 7}),
    # ra_lw=1 (classic AER RRTM 16-band LW) is now operationally scan-wired
    # (coupling.physics_couplers.rrtm_lw_theta_tendency over the JAX-traceable
    # physics.ra_lw_rrtm_jax kernel, dispatched in runtime.operational_mode by
    # OperationalNamelist.ra_lw_physics; SW selected independently).
    # ra_lw=31 (Held-Suarez idealized radiation, phys/module_ra_hs.F:HSRAD) is a
    # v0.17 no-kernel-change endpoint port: a stateless combined LW+SW Newtonian
    # relaxation selected through the LW slot, operationally scan-wired
    # (coupling.physics_couplers.held_suarez_theta_tendency; dispatch fail-closes
    # any SW selection since HSRAD is the sole radiative call), savepoint-parity-
    # proven against the unmodified WRF source at fp64
    # (proofs/v017/held_suarez_lw_savepoint_parity.json).
    "ra_lw_physics": frozenset({0, 1, 4, 31}),
    # ra_sw=1 (Dudhia, Stephens-1984 broadband SW) and ra_sw=2 (GSFC/Chou-Suarez
    # multi-band delta-Eddington SW) are now operationally scan-wired
    # (coupling.physics_couplers.dudhia_sw_theta_tendency / gsfc_sw_theta_tendency,
    # dispatched in runtime.operational_mode by OperationalNamelist.ra_sw_physics).
    # ra_sw=3/5/7/99 (CAM/Goddard-new/FLG/GFDL-Eta) are accepted reference-only
    # in v0.18 for real-WRF oracle/parity work, not operational scan-wiring.
    "ra_sw_physics": frozenset({0, 1, 2, 4}),
}

# Recognized WRF schemes with an oracle-backed reference path that the
# operational scan fail-closes (selectable for a reference comparison, NOT an
# operational run). Some entries are GREEN; some are RED and say so in the reason.
# reason = the named scan-unwired reason; alternative = the operational swap.
_REFERENCE_ONLY: Mapping[str, dict[int, tuple[str, str]]] = {
    "cu_physics": {
        # v0.17 SAS family: all four requested codes now have fp64 pristine-WRF
        # single-column savepoints, but the shared JAX endpoint is still RED
        # against those oracles. They are reference-only and fail-closed.
        4: (
            "Scale-aware GFS SAS has v0.17 single-column fp64 pristine-WRF "
            "savepoints (module_cu_scalesas.F), but the shared JAX endpoint is "
            "RED vs oracle, so it is fail-closed in the operational GPU scan.",
            "Use cu_physics=1/2/3/6 for operational runs; use "
            "proofs/v017/run_sas_family_parity.py for SAS oracle comparisons.",
        ),
        # v0.13 Tier-3 cumulus batch: New-Tiedtke(16) / KSAS(14) / Grell-3D(5) each
        # have a single-column fp64 pristine-WRF oracle staged
        # (proofs/v013/oracle/cumulus, savepoints/cumulus), but their traceable JAX
        # column kernels are a documented carry-over. They are REFERENCE_ONLY:
        # registry-accepted so a single-column reference comparison can be run, but
        # fail-closed in the operational GPU scan (never silently wrong).
        5: (
            "Grell-3D ensemble has a v0.13 single-column fp64 pristine-WRF oracle "
            "staged, but its traceable JAX column kernel is not yet ported, so it "
            "is fail-closed in the operational GPU scan.",
            "Use cu_physics=3 (Grell-Freitas, GPU-operational) or 1/2/6.",
        ),
        14: (
            "KIM Simplified Arakawa-Schubert has a v0.13 single-column fp64 "
            "pristine-WRF oracle staged, but its traceable JAX column kernel is not "
            "yet ported, so it is fail-closed in the operational GPU scan.",
            "Use cu_physics=1/2/3/6 (Kain-Fritsch / BMJ / Grell-Freitas / Tiedtke; "
            "Tiedtke also requires active flux-form moisture advection for RQVFTEN).",
        ),
        16: (
            "New Tiedtke has module-specific v0.13 single-column fp64 pristine-WRF "
            "savepoints from phys/module_cu_ntiedtke.F, but no faithful traceable "
            "JAX kernel or CU scan adapter is wired yet, so it is fail-closed in "
            "the operational GPU scan.",
            "Use cu_physics=6 (modified Tiedtke with active flux-form moisture "
            "advection for RQVFTEN) or 1/3.",
        ),
        93: (
            "Grell-Devenyi ensemble is recognized for v0.17 oracle work, but no "
            "source-specific traceable JAX column endpoint has passed parity "
            "against unmodified phys/module_cu_gd.F:GRELLDRV, so it is fail-"
            "closed in the operational GPU scan.",
            "Use cu_physics=3 (Grell-Freitas, GPU-operational) or 1/2/6.",
        ),
        94: (
            "2015 GFS SAS / HWRF has v0.17 single-column fp64 pristine-WRF "
            "savepoints (module_cu_sas.F), but the shared JAX endpoint is RED "
            "vs oracle, so it is fail-closed in the operational GPU scan.",
            "Use cu_physics=1/2/3/6 for operational runs; use "
            "proofs/v017/run_sas_family_parity.py for SAS oracle comparisons.",
        ),
        95: (
            "Previous GFS SAS / HWRF OSAS has v0.17 single-column fp64 pristine-WRF "
            "savepoints (module_cu_osas.F), but the shared JAX endpoint is RED "
            "vs oracle, so it is fail-closed in the operational GPU scan.",
            "Use cu_physics=1/2/3/6 for operational runs; use "
            "proofs/v017/run_sas_family_parity.py for SAS oracle comparisons.",
        ),
        96: (
            "Previous new GFS SAS / YSU NSAS has v0.17 single-column fp64 "
            "pristine-WRF savepoints (module_cu_nsas.F), but the shared JAX "
            "endpoint is RED vs oracle, so it is fail-closed in the operational "
            "GPU scan.",
            "Use cu_physics=1/2/3/6 for operational runs; use "
            "proofs/v017/run_sas_family_parity.py for SAS oracle comparisons.",
        ),
        99: (
            "Previous Kain-Fritsch is recognized for v0.17 oracle work, but the "
            "available GPU endpoint is the KF-eta family (cu_physics=1), not a "
            "parity-proven port of unmodified phys/module_cu_kf.F:KFCPS, so it "
            "is fail-closed in the operational GPU scan.",
            "Use cu_physics=1 (Kain-Fritsch eta, GPU-operational) or 0/2/3/6.",
        ),
    },
    # sf_surface_physics=1 (5-layer thermal-diffusion slab LSM) was REFERENCE_ONLY
    # (JAX-ported + fp64 oracle, no operational hook); it is now operationally
    # scan-wired via coupling.slab_surface_hook.slab_surface_step (IMPLEMENTED
    # above), so it is no longer listed here.
    # sf_surface_physics=3 (RUC) + 8 (SSiB) are v0.17 Tier-3 REFERENCE-ONLY: each has
    # a fp64 pristine-WRF single-column oracle staged (proofs/v017/oracle/{ruclsm,ssib},
    # built from the unmodified WRF LSMRUC / SSIB drivers, NOT a self-compare), but the
    # faithful traceable JAX column kernels for these large multi-layer/biophysical
    # solvers are documented carry-overs, so both are namelist-accepted (selectable for a
    # single-column reference comparison) and fail-closed in the operational GPU scan with
    # a named reason (never silently wrong).
    "sf_surface_physics": {
        3: (
            "RUC multi-layer soil/snow LSM has a v0.17 single-column fp64 pristine-WRF "
            "oracle staged (LSMRUC->SOILVEGIN->SFCTMP, proofs/v017/oracle/ruclsm), but its "
            "faithful traceable JAX column kernel (the ~7.5k-LOC soil/snow solver) is not "
            "yet ported, so it is fail-closed in the operational GPU scan.",
            "Use sf_surface_physics=4 (Noah-MP, GPU-operational), 2 (Noah classic), 1 (slab) "
            "or 7 (Pleim-Xiu).",
        ),
        8: (
            "SSiB SiB biophysical canopy/soil/snow LSM has a v0.17 single-column fp64 "
            "pristine-WRF oracle staged (the unmodified SSIB driver, proofs/v017/oracle/ssib), "
            "but its faithful traceable JAX column kernel (the ~6.6k-LOC coupled SiB solver) "
            "is not yet ported, so it is fail-closed in the operational GPU scan.",
            "Use sf_surface_physics=4 (Noah-MP, GPU-operational), 2 (Noah classic), 1 (slab) "
            "or 7 (Pleim-Xiu).",
        ),
    },
    # v0.18 RA tail: CAM/Goddard/FLG/GFDL-Eta have exact-driver real-WRF
    # savepoints in proofs/v018/savepoints/ra_tail_wrf. They remain REFERENCE_ONLY
    # because no faithful traceable JAX kernels are wired into the operational scan.
    "ra_lw_physics": {
        3: (
            "CAM longwave has a v0.18 real-WRF driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra3_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_cam.F:CAMRAD), but no "
            "faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_lw_physics=4 (RRTMG, GPU-operational default) or 1 (classic RRTM).",
        ),
        5: (
            "GSFC/Goddard NUWRF longwave has a v0.13 single-column fp64 pristine-WRF "
            "oracle staged (module_ra_goddard.F:lwrad) and a v0.18 real-WRF paired "
            "driver oracle (proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json), "
            "but its traceable JAX column kernel is not operationally scan-wired, so "
            "it is fail-closed in the operational GPU scan.",
            "Use ra_lw_physics=4 (RRTMG, GPU-operational default) or 1 (classic RRTM).",
        ),
        7: (
            "FLG/UCLA longwave has a v0.18 real-WRF driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra7_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_flg.F:RAD_FLG), but no "
            "faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_lw_physics=4 (RRTMG, GPU-operational default) or 1 (classic RRTM).",
        ),
        99: (
            "GFDL-Eta longwave has a v0.18 real-WRF paired driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra99_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_gfdleta.F:ETARA), but no "
            "faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_lw_physics=4 (RRTMG, GPU-operational default) or 1 (classic RRTM).",
        ),
    },
    "ra_sw_physics": {
        3: (
            "CAM shortwave has a v0.18 real-WRF driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra3_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_cam.F:CAMRAD), but no "
            "faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_sw_physics=4 (RRTMG), 1 (Dudhia), or 2 (GSFC/Chou-Suarez).",
        ),
        5: (
            "New Goddard shortwave has a v0.18 real-WRF paired driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_goddard.F:goddardrad), "
            "but no faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_sw_physics=4 (RRTMG), 1 (Dudhia), or 2 (GSFC/Chou-Suarez).",
        ),
        7: (
            "FLG/UCLA shortwave has a v0.18 real-WRF driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra7_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_flg.F:RAD_FLG), but no "
            "faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_sw_physics=4 (RRTMG), 1 (Dudhia), or 2 (GSFC/Chou-Suarez).",
        ),
        99: (
            "GFDL-Eta shortwave has a v0.18 real-WRF paired driver oracle "
            "(proofs/v018/savepoints/ra_tail_wrf/ra99_wrf_real.json, "
            "module_radiation_driver.F dispatch to module_ra_gfdleta.F:ETARA), but no "
            "faithful JAX column kernel is operationally scan-wired, so it is "
            "fail-closed in the operational GPU scan.",
            "Use ra_sw_physics=4 (RRTMG), 1 (Dudhia), or 2 (GSFC/Chou-Suarez).",
        ),
    },
    "bl_pbl_physics": {
        4: (
            "QNSE-EDMF PBL has a v0.18 fp64 pristine-WRF single-column oracle "
            "staged (unmodified phys/module_bl_qnsepbl.F; "
            "proofs/v018/qnse_pbl4_reference_oracle.json), but no traceable JAX "
            "column kernel is scan-wired, so it is fail-closed in the operational "
            "GPU scan.",
            "Use bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99 for operational runs; use "
            "proofs/v018/run_qnse_pbl4_oracle_check.py for QNSE oracle comparisons.",
        ),
        10: (
            "TEMF PBL has a v0.18 fp64 pristine-WRF single-column oracle staged "
            "(unmodified phys/module_bl_temf.F; "
            "proofs/v018/temf_pbl10_reference_oracle.json), but no traceable JAX "
            "column kernel is scan-wired, so it is fail-closed in the operational "
            "GPU scan.",
            "Use bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99 for operational runs; use "
            "proofs/v018/run_temf_pbl10_oracle_check.py for TEMF oracle comparisons.",
        ),
        16: (
            "EEPS epsilon PBL has a v0.18 fp64 pristine-WRF single-column oracle "
            "staged (unmodified phys/module_bl_eepsilon.F; "
            "proofs/v018/eeps_pbl16_reference_oracle.json), but no traceable JAX "
            "column kernel is scan-wired, so it is fail-closed in the operational "
            "GPU scan.",
            "Use bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99 for operational runs; use "
            "proofs/v018/run_eeps_pbl16_oracle_check.py for EEPS oracle comparisons.",
        ),
        17: (
            "KEPS k-epsilon PBL has a v0.18 fp64 pristine-WRF single-column oracle "
            "staged (unmodified phys/module_bl_keps.F; "
            "proofs/v018/keps_pbl17_reference_oracle.json), but no traceable JAX "
            "column kernel is scan-wired, so it is fail-closed in the operational "
            "GPU scan.",
            "Use bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99 for operational runs; use "
            "proofs/v018/run_keps_pbl17_oracle_check.py for KEPS oracle comparisons.",
        ),
    },
    # bl_pbl_physics=2 (MYJ) + sf_sfclay_physics=2 (Janjic Eta) were REFERENCE_ONLY
    # (host-NumPy savepoint kernels); they are now operationally scan-wired as a
    # mandatory pair via the JAX-traceable physics.bl_myj / physics.sf_myj rewrites
    # (IMPLEMENTED above), so they are no longer listed here. bl=3 GFS is IMPLEMENTED.
    # bl_pbl_physics=9 (CAM-UW) is now IMPLEMENTED as a v0.22 operational
    # scaffold with WRF-source/idealized proof; full CAM-stack parity remains a
    # named caveat in its proof object rather than a REFERENCE_ONLY status here.
    # ra_lw_physics=1 (classic RRTM LW) was REFERENCE_ONLY (host-NumPy kernel); it
    # is now operationally scan-wired via the JAX-traceable physics.ra_lw_rrtm_jax
    # rewrite (IMPLEMENTED above), so it is no longer listed here.
    # ra_sw_physics=1 (Dudhia) was REFERENCE_ONLY; it is now operationally
    # scan-wired (IMPLEMENTED above), so it is no longer listed here.
}


def _label(key: str) -> str:
    return WRF_PARAM_LABEL.get(key, key)


# Per-key fallback alternative text used for RECOGNIZED_FAIL_CLOSED schemes.
_DEFAULT_ALTERNATIVE: Mapping[str, str] = {
    "mp_physics": "Use one of mp_physics=0/1/2/3/4/6/8/10/13/14/16/24/26/28/97 (8=Thompson is the "
    "operational default; 13=SBU-YLin; 24=WSM7 / 26=WDM7 add a precipitating hail class; "
    "28=aerosol-aware Thompson; 97=Goddard GCE single-moment 3-ice).",
    "cu_physics": "Use one of cu_physics=0/1/2/3/6 (1=Kain-Fritsch eta, "
    "3=Grell-Freitas, 6=Tiedtke requires active flux-form moisture advection "
    "for RQVFTEN). Reference-only cumulus options 4/5/14/16/93/94/95/96/99 "
    "fail-close in the operational scan.",
    "bl_pbl_physics": "Use one of bl_pbl_physics=0/1/2/3/5/7/8/9/11/12/99 (5=MYNN, 1=YSU, 2=MYJ "
    "[pair with sf_sfclay_physics=2], 3=GFS, 7=ACM2, 8=BouLac, 9=CAM-UW, "
    "11=Shin-Hong, 12=GBM, 99=MRF). PBL4/10/16/17 are accepted reference-only and "
    "fail-close in the operational scan.",
    "sf_sfclay_physics": "Use one of sf_sfclay_physics=0/1/2/3/5/7/91 (5=MYNN-SL, "
    "1=revised-MM5, 2=Janjic Eta [pair with bl_pbl_physics=2], 3=NCEP-GFS, "
    "7=Pleim-Xiu, 91=old-MM5).",
    "sf_surface_physics": "Use sf_surface_physics=4 (Noah-MP), 2 (Noah classic), 1 (slab) "
    "or 7 (Pleim-Xiu); 3=RUC and 8=SSiB are reference-only (fp64 oracle staged, JAX "
    "kernel carry-over).",
    "ra_lw_physics": "Use ra_lw_physics=4 (RRTMG) or 1 (classic RRTM); 3/5/7/99 are reference-only, 14/24 are compiled-out in this WRF build.",
    "ra_sw_physics": "Use ra_sw_physics=4 (RRTMG), 1 (Dudhia) or 2 (GSFC/Chou-Suarez); 3/5/7/99 are reference-only, 14/24 are compiled-out in this WRF build.",
    "diff_opt": "Use diff_opt=0/1/2 (1+km_opt=4 = 2-D Smagorinsky real-data "
    "default; 2+km_opt=1/2/3/5 = constant-K / 3-D turbulence closures).",
    "km_opt": "Use km_opt=0/1/2/3/4/5 (4 with diff_opt=1 = 2-D Smagorinsky; "
    "1/2/3/5 with diff_opt=2 = constant-K / 3-D TKE / 3-D Smagorinsky / SMS-3DTKE).",
    "damp_opt": "Use damp_opt=0 (off) or 3 (upper-level w-Rayleigh).",
    "diff_6th_opt": "Use diff_6th_opt=0 (off) or 2 (monotonic 6th-order filter).",
    "rk_order": "Use rk_order=3 (WRF RK3).",
    "w_damping": "Use w_damping=0 or 1.",
    "sf_urban_physics": "Set sf_urban_physics=0; BEP/BEM need the G3 urban state/oracle port before they can run.",
    "sf_lake_physics": "Set sf_lake_physics=0; the WRF lake column/state/oracle port is not operationally wired.",
}


# --------------------------------------------------------------------------- #
# Dynamics / numerics: implemented integer codes + the Smagorinsky note.      #
# These are gated by namelist_check.SUPPORTED_OPTIONS today; the catalog      #
# mirrors that so the public table is complete and consistent.                #
# --------------------------------------------------------------------------- #
_DYNAMICS_IMPLEMENTED: Mapping[str, frozenset[int]] = {
    "diff_opt": frozenset({0, 1, 2}),
    "km_opt": frozenset({0, 1, 2, 3, 4, 5}),
    "damp_opt": frozenset({0, 3}),
    "diff_6th_opt": frozenset({0, 2}),
    "rk_order": frozenset({3}),
    "w_damping": frozenset({0, 1}),
    # Urban/lake feature switches: only "off" (0) is a real path; active codes
    # are recognized G3 targets that fail closed with source/oracle reasons.
    "sf_urban_physics": frozenset({0}),
    "sf_lake_physics": frozenset({0}),
}

# Dynamics options that ARE valid WRF codes but are fail-closed by the port,
# with an explicit reason + transition recipe.
# Per-(key, code) fail-closed reason override for RECOGNIZED_FAIL_CLOSED schemes.
# Despite the historical name, this is consulted for ANY namelist key in
# classify_scheme (not only dynamics): it supplies a specific reason in place of
# the generic "NOT YET IMPLEMENTED" string when the truth is more precise.
_PER_CODE_FAIL_CLOSED_REASON: Mapping[str, dict[int, str]] = {
    "sf_urban_physics": {
        1: "G3 URBAN fail-closed scaffold: WRF single-layer UCM is recognized, "
        "but this sprint targets BEP/BEM and no UCM oracle/kernel/state carry is "
        "operationally wired. The GPU port keeps urban effects limited to land-use "
        "categories until a source-specific urban canopy port lands.",
        2: "G3 URBAN fail-closed scaffold: WRF BEP (phys/module_sf_bep.F:BEP, "
        "~3.5k LOC) needs the Registry bepscheme state package, urban mapping "
        "tables, vertical urban-grid carry, and a pristine-WRF single-column "
        "oracle before any GPU scan wiring. No faithful JAX BEP kernel is shipped "
        "in this one-pass attempt.",
        3: "G3 URBAN fail-closed scaffold: WRF BEP+BEM combines "
        "phys/module_sf_bep.F:BEP with phys/module_sf_bem.F:BEM (~6.2k LOC total) "
        "and the Registry bep_bemscheme state package (building energy, HVAC, PV, "
        "green-roof, drainage and multi-layer urban-grid fields). No source-"
        "specific pristine-WRF oracle or faithful JAX kernel is shipped in this "
        "one-pass attempt.",
    },
    "sf_lake_physics": {
        1: "G3 LAKE fail-closed scaffold: WRF lake model "
        "(phys/module_sf_lake.F:Lake/LakeMain/lakeini, ~5.4k LOC) needs lake "
        "depth/category initialization, snow/ice/water column carry, tridiagonal "
        "thermal solves, hydrology, Monin-Obukhov fluxes and a pristine-WRF "
        "single-column oracle before GPU scan wiring. No faithful JAX lake kernel "
        "is shipped in this one-pass attempt.",
    },
    # mp=24 WSM7 was fail-closed before v0.17 (it carries a separate precipitating
    # hail class qh the operational moist-state pytree did not hold). v0.17 added
    # the qh hail State substrate (ADR-032) + the hail surface accumulator and
    # scan-wired WSM7 (coupling.scan_adapters.wsm7_adapter); it is now IMPLEMENTED
    # (see _IMPLEMENTED["mp_physics"]), so it no longer has a fail-closed reason.
    "mp_physics": {
        5: "REFERENCE-WITH-REAL-ORACLE / fail-closed: Ferrier-HRW (new Eta, "
        "operational High-Resolution Window) has exact pristine-WRF FER_HIRES "
        "savepoints under proofs/v018/mp_oracles/ferrier_hires for "
        "phys/module_mp_fer_hires.F:FER_HIRES. The staged MP95 ETAMP_NEW oracle "
        "drives phys/module_mp_etanew.F and is not reused for MP5. The JAX "
        "endpoint is NOT YET "
        "IMPLEMENTED: it carries lumped total condensate qt/CWM plus "
        "f_ice_phy/f_rain_phy/f_rimef_phy state and ETAMPNEW_DATA lookup-table "
        "control flow that is not present in the operational State or scan "
        "interface.",
        95: "REFERENCE-WITH-REAL-ORACLE / fail-closed: Ferrier old Eta "
        "(etampnew) has exact pristine-WRF ETAMP_NEW oracle artifacts under "
        "proofs/v018/mp_oracles/ferrier_etanew for "
        "phys/module_mp_etanew.F:ETAMP_NEW. The operational JAX endpoint is "
        "still NOT YET IMPLEMENTED: the scheme carries lumped total condensate "
        "qt/CWM plus f_ice_phy/f_rain_phy/f_rimef_phy state and ETAMPNEW_DATA "
        "lookup-table control flow that is not present in the operational State "
        "or scan interface.",
        96: "PROVEN-NO-OP at the microphysics step: MadWRF mp_physics=96 is "
        "source-verified as a no-op inside "
        "phys/module_microphysics_driver.F CASE (MADWRF_MP), which only emits "
        "wrf_debug; the real MadWRF cloud initialization/nudging lives outside "
        "the microphysics step, so there is no faithful MP kernel to wire and it "
        "is NOT YET IMPLEMENTED as an operational GPU microphysics option.",
        7: "REFERENCE-WITH-REAL-ORACLE / fail-closed: Goddard 4-ice / NUWRF has "
        "exact pristine-WRF oracle artifacts under "
        "proofs/v018/mp_oracles/goddard4ice for "
        "phys/module_mp_gsfcgce_4ice_nuwrf.F. The JAX kernel is NOT YET "
        "IMPLEMENTED: this is the separate hail-class scheme with qh, hail "
        "accumulators, large phys*/re_* carry, and the long saticel_s 4-ice path, "
        "not the operational mp=97 Goddard GCE 3-ice port.",
        38: "REFERENCE-WITH-REAL-ORACLE / fail-closed: Thompson graupel-hail has "
        "exact pristine-WRF oracle artifacts under proofs/v018/mp_oracles/thompgh "
        "for phys/module_mp_thompson.F. The JAX kernel is NOT YET IMPLEMENTED: "
        "its hail path uses variable-density graupel via qvolg/Ng and hail-aware "
        "collision/terminal-velocity tables that are not the operational "
        "fixed-density mp=8/28 Thompson path.",
        9: "REFERENCE-WITH-REAL-ORACLE / fail-closed: Milbrandt-Yau 2-moment "
        "has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp9 for "
        "phys/module_mp_milbrandt2mom.F selected by mp_physics=9. The JAX "
        "endpoint is NOT YET IMPLEMENTED: it is a real multimoment bulk scheme "
        "and requires qh plus qnc/qnr/qni/qns/qng/qnh number-state transport "
        "before operational scan wiring.",
        11: "PROVEN-IRRELEVANT for the lean operational GPU target: CAM 5.1 "
        "microphysics is recognized in phys/module_mp_cammgmp_driver.F, but its "
        "WRF source warns that QME3D is wrong without CAM macrophysics, "
        "convective cloud fraction is unavailable to microphysics, and the "
        "outputs are not currently consumed by RRTMG. It is CAM-specific and "
        "NOT YET IMPLEMENTED in the operational microphysics State/PhysicsCarry "
        "contract.",
        17: "PROVEN-IRRELEVANT / SUPERSEDED legacy NSSL option: "
        "phys/module_mp_nssl_2mom.F recognizes it, but doc/README.NSSLmp says to "
        "use mp_physics=18 with modifier flags going forward. It is NOT YET "
        "IMPLEMENTED as a separate GPU scheme; MP18 is the reference-oracle-"
        "backed exact NSSL target and carries the qh/qnh/qvolg/qvolh-style "
        "state/oracle work.",
        18: "ORACLE-ABSENT / fail-closed: NSSL 2-moment 4-ice with predicted "
        "CCN is recognized from phys/module_mp_nssl_2mom.F and the WRF Registry "
        "nssl_2mom package (mp_physics=18), but no local single-column WRF oracle "
        "artifact is present in this worktree. The JAX endpoint is NOT YET "
        "IMPLEMENTED: it needs NSSL number/hail-volume state (qh/qnh/qvolg/qvolh), "
        "qnn CCN controls, and a source-specific oracle before scan wiring.",
        19: "PROVEN-IRRELEVANT / SUPERSEDED legacy NSSL option: "
        "phys/module_mp_nssl_2mom.F recognizes it, but doc/README.NSSLmp maps it "
        "to mp_physics=18 with nssl_2moment_on=0 and nssl_ccn_on=1. It is NOT "
        "YET IMPLEMENTED as a separate GPU scheme; MP18 is the reference-oracle-"
        "backed exact NSSL target and carries the "
        "qh/qnh/qvolg/qvolh-style state/oracle work.",
        21: "PROVEN-IRRELEVANT / SUPERSEDED legacy NSSL option: "
        "phys/module_mp_nssl_2mom.F recognizes it, but doc/README.NSSLmp maps it "
        "to mp_physics=18 with nssl_2moment_on=0, nssl_hail_on=0, "
        "nssl_ccn_on=0, and nssl_density_on=0. It is NOT YET IMPLEMENTED as a "
        "separate GPU scheme; MP18 is the reference-oracle-backed exact NSSL "
        "target and carries the qh/qnh/qvolg/qvolh-style state/oracle work.",
        22: "PROVEN-IRRELEVANT / SUPERSEDED legacy NSSL option: "
        "phys/module_mp_nssl_2mom.F recognizes it, but doc/README.NSSLmp maps it "
        "to mp_physics=18 with nssl_hail_on=0 and nssl_ccn_on=1. It is NOT YET "
        "IMPLEMENTED as a separate GPU scheme; MP18 is the reference-oracle-"
        "backed exact NSSL target and carries the qh/qnh/qvolg/qvolh-style "
        "state/oracle work.",
        27: "REFERENCE-WITH-REAL-ORACLE / fail-closed: UDM 7-class "
        "microphysics has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp27 for phys/module_mp_udm.F "
        "selected by mp_physics=27. The JAX endpoint is NOT YET IMPLEMENTED: it "
        "is a UFS double-moment scheme and still needs its qh plus qnn/qnc/qnr "
        "number-state contract and kernel.",
        29: "REFERENCE-WITH-REAL-ORACLE / fail-closed: RCON has active "
        "pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp29 for phys/module_mp_rcon.F "
        "selected by mp_physics=29. The JAX endpoint is NOT YET IMPLEMENTED: "
        "this January-2025 Thompson aerosol-aware variant adds cloudnc/black-"
        "carbon aerosol state beyond the operational mp=28 Thompson-aero path "
        "and needs its own aerosol-state ADR.",
        30: "PROVEN-IRRELEVANT / RESEARCH-ONLY for the lean operational GPU "
        "target: HUJI fast spectral-bin microphysics is recognized in "
        "phys/module_mp_fast_sbm.F, is a spectral-bin research scheme guarded by "
        "the BUILD_SBM_FAST compile path, and requires bin-state transport plus "
        "external SBM lookup tables. It is NOT YET IMPLEMENTED as an operational "
        "bulk-scheme adapter.",
        32: "PROVEN-IRRELEVANT / RESEARCH-ONLY for the lean operational GPU "
        "target: HUJI full spectral-bin microphysics is recognized in "
        "phys/module_mp_full_sbm.F and requires full spectral-bin-state and "
        "external SBM lookup-table architecture. It is NOT YET IMPLEMENTED as an "
        "operational bulk-scheme adapter.",
        40: "ORACLE-ABSENT / fail-closed: Morrison aerosol microphysics is "
        "recognized from phys/module_mp_morr_two_moment_aero.F selected by "
        "mp_physics=40 (the aerosol-coupled extension of Morrison "
        "module_mp_morr_two_moment.F), but no local single-column WRF oracle "
        "artifact is present in this worktree. The JAX endpoint is NOT YET "
        "IMPLEMENTED: it extends Morrison with aerosol activation/effectiveness, "
        "CCN diagnostics, and the CESM_RCP4.5 aerosol-data dependency beyond the "
        "current operational mp=10 interface.",
        50: "REFERENCE-WITH-REAL-ORACLE / fail-closed: P3 1-category "
        "microphysics has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp50 for phys/module_mp_p3.F "
        "selected by mp_physics=50. The JAX endpoint is NOT YET IMPLEMENTED: "
        "the P3 family needs qir/qib/rime-density particle property state, "
        "lookup-table initialization, and a dedicated State ADR before "
        "operational scan wiring.",
        51: "REFERENCE-WITH-REAL-ORACLE / fail-closed: P3 1-category + cloud-"
        "number microphysics has active pristine-WRF full-model oracle artifacts "
        "under proofs/v018/mp_oracles/wrf_full_model/mp51 for "
        "phys/module_mp_p3.F selected by mp_physics=51. The JAX endpoint is NOT "
        "YET IMPLEMENTED: the P3 family needs qnc/qir/qib/rime-density particle "
        "property state, lookup-table initialization, and a dedicated State ADR.",
        52: "REFERENCE-WITH-REAL-ORACLE / fail-closed: P3 2-category "
        "microphysics has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp52 for phys/module_mp_p3.F "
        "selected by mp_physics=52. The JAX endpoint is NOT YET IMPLEMENTED: it "
        "adds qi2/qni2/qir2/qib2 second-ice-category state plus P3 particle-"
        "property/rime-density carry that the operational State does not yet "
        "expose.",
        53: "REFERENCE-WITH-REAL-ORACLE / fail-closed: P3 1-category 3-moment "
        "microphysics has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp53 for phys/module_mp_p3.F "
        "selected by mp_physics=53. The JAX endpoint is NOT YET IMPLEMENTED: it "
        "adds qzi and P3 particle-property/rime-density carry that need a "
        "dedicated State ADR and kernel.",
        55: "PROVEN-IRRELEVANT / RESEARCH-ONLY for the lean operational GPU "
        "target: Jensen-ISHMAEL is recognized in phys/module_mp_jensen_ishmael.F "
        "as an initial-release habit research model with external ishmael-*.bin "
        "tables. It is NOT YET IMPLEMENTED: it uses multiple ice habits "
        "(qi2/qi3), habit volume/axis scalars, phii/itype diagnostics, and "
        "habit-state carry outside the current microphysics interface.",
        56: "REFERENCE-WITH-REAL-ORACLE / fail-closed: NTU multi-moment "
        "microphysics has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp56 for phys/module_mp_ntu.F "
        "selected by mp_physics=56. The JAX endpoint is NOT YET IMPLEMENTED: it "
        "requires qh plus many liquid/ice/aerosol moments "
        "(qdc/qtc/qcc/qrc/qnin/fi/fs/vi/vs/vg/ai/as/ag/ah/i3m) and an NTU "
        "State ADR before operational scan wiring.",
    },
    #
    # ra_lw/sw_physics=14 (RRTMG-K / KIAPS) and =24 (fast RRTMG, GPU/MIC) are NOT
    # a port gap: they are compiled OUT of standard WRF itself. Their source
    # modules (phys/module_ra_rrtmg_{lwk,swk,lwf,swf}.F) are bare dummy stubs
    # guarded by `#if( BUILD_RRTMK != 1)` / `#if( BUILD_RRTMG_FAST != 1)`, and the
    # pristine configure.wrf sets `-DBUILD_RRTMK=0` / `-DBUILD_RRTMG_FAST=0`. The
    # radiation_driver CASEs are likewise `#if( BUILD_* == 1)`-gated, so selecting
    # 14/24 in unmodified WRF reaches the driver's default branch and aborts with
    # "The longwave/shortwave option does not exist". They are therefore
    # documented as class-(c) computationally-unavailable and fail closed here --
    # there is no real oracle to build because the scheme cannot run in this build.
    "ra_lw_physics": {
        14: "RRTMG-K (KIAPS) longwave is compiled OUT of standard WRF "
        "(phys/module_ra_rrtmg_lwk.F is a `#if( BUILD_RRTMK != 1)` dummy stub; "
        "pristine configure.wrf sets -DBUILD_RRTMK=0), so it cannot run even in "
        "unmodified WRF -- selecting it hits the radiation_driver default abort.",
        24: "fast RRTMG (GPU/MIC) longwave is compiled OUT of standard WRF "
        "(phys/module_ra_rrtmg_lwf.F is a `#if( BUILD_RRTMG_FAST != 1)` dummy "
        "stub; pristine configure.wrf sets -DBUILD_RRTMG_FAST=0), so it cannot "
        "run even in unmodified WRF -- selecting it hits the driver default abort.",
    },
    "ra_sw_physics": {
        14: "RRTMG-K (KIAPS) shortwave is compiled OUT of standard WRF "
        "(phys/module_ra_rrtmg_swk.F is a `#if( BUILD_RRTMK != 1)` dummy stub; "
        "pristine configure.wrf sets -DBUILD_RRTMK=0), so it cannot run even in "
        "unmodified WRF -- selecting it hits the radiation_driver default abort.",
        24: "fast RRTMG (GPU/MIC) shortwave is compiled OUT of standard WRF "
        "(phys/module_ra_rrtmg_swf.F is a `#if( BUILD_RRTMG_FAST != 1)` dummy "
        "stub; pristine configure.wrf sets -DBUILD_RRTMG_FAST=0), so it cannot "
        "run even in unmodified WRF -- selecting it hits the driver default abort.",
    },
    "damp_opt": {
        1: "Diffusive upper-level damping is not implemented.",
        2: "Rayleigh damping (idealized-only) is not implemented; the real-data "
        "path uses damp_opt=3 (w-Rayleigh).",
    },
    "diff_6th_opt": {
        1: "6th-order diffusion *with* up-gradient flux is not implemented; the "
        "port uses the monotonic (no up-gradient) variant diff_6th_opt=2.",
    },
    "rk_order": {
        2: "RK2 time integration is not implemented; the port is RK3-only.",
    },
}

_SCHEME_FAIL_CLOSED_REASON: Mapping[str, dict[int, str]] = {}


# Physics options that ARE valid WRF v4 codes but are documented v0.18->v1.0
# ARCHITECTURE BOUNDARIES the port fails closed with a SPECIFIC reason +
# alternative (reason, alternative), rather than the generic "NOT YET
# IMPLEMENTED" text. Reserved for schemes whose faithful single-column oracle is
# itself architecture-scale (so a v0.18-session port would necessarily become a
# happy-path/self-compare) -- selecting one raises a clear named error, NEVER a
# silent substitution or a stub kernel. CLM4 (sf=5) + CTSM (sf=6) are the
# CAM/CLM/CTSM-family land-surface boundary (carried to the v1.0 ADR).
_PHYSICS_FAIL_CLOSED_REASON: Mapping[str, dict[int, tuple[str, str]]] = {
    "sf_surface_physics": {
        5: (
            "CLM4 (Community Land Model v4) is a recognized WRF v4 land-surface "
            "option that is a documented v0.18->v1.0 ARCHITECTURE BOUNDARY, NOT a "
            "happy-path stub: phys/module_sf_clm.F is ~61.5k LOC built around a "
            "single global clmtype PFT/column/landunit/gridcell subgrid hierarchy "
            "initialized from an EXTERNAL CLM surface dataset (PFT fractions + soil "
            "colour/sand/clay) plus netCDF MEGAN/SNICAR inputs, so a faithful "
            "single-column pristine-WRF oracle is itself architecture-scale "
            "(multi-session) -- it is deferred to the v1.0 CAM/CLM/CTSM-family ADR "
            "and fails closed (never silently substituted by another LSM).",
            "Use sf_surface_physics=4 (Noah-MP, GPU-operational), 2 (Noah classic), "
            "1 (slab) or 7 (Pleim-Xiu); 3=RUC / 8=SSiB are reference-only. CLM4 is "
            "carried to the v1.0 CAM/CLM/CTSM ADR.",
        ),
        6: (
            "CTSM (Community Terrestrial Systems Model) is a recognized WRF v4 "
            "land-surface option that is a documented v0.18->v1.0 ARCHITECTURE "
            "BOUNDARY: phys/module_sf_ctsm.F is compiled only under -DWRF_USE_CTSM "
            "and runs the FULL external CESM/CTSM land model through the LILAC "
            "coupler (its WRF-side Registry state package is empty -- it carries NO "
            "in-core prognostic land state), so there is no in-core WRF physics to "
            "build a single-column oracle from without the external coupled-model "
            "build. It is out of scope for the in-core GPU dycore port and fails "
            "closed.",
            "Use sf_surface_physics=4 (Noah-MP), 2 (Noah classic), 1 (slab) or 7 "
            "(Pleim-Xiu). CTSM is carried to the v1.0 CAM/CLM/CTSM ADR.",
        ),
    },
}


# --------------------------------------------------------------------------- #
# OUT-OF-SCOPE feature switches (documented design decisions). These are       #
# boolean/positive-int feature gates with NO enumerated WRF v4 scheme catalog. #
# A *truthy* (non-zero / .true.) selection fails closed as a scope decision.   #
# Keys are matched case-insensitively in any namelist section.                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OutOfScopeFeature:
    """A WRF capability the port deliberately does not implement."""

    key: str
    feature: str
    reason: str
    alternative: str


OUT_OF_SCOPE_FEATURES: tuple[OutOfScopeFeature, ...] = (
    OutOfScopeFeature(
        "windfarm_opt", "Wind-farm / wind-turbine drag parameterization",
        "The wind-farm turbine-drag parameterization (windfarm_opt) is out of "
        "scope for this port.",
        "Set windfarm_opt=0.",
    ),
    OutOfScopeFeature(
        "grid_sfdda", "FDDA surface-analysis nudging",
        "Surface-analysis FDDA nudging (grid_sfdda) is out of scope; this is a "
        "pure forecast-integration port.",
        "Set grid_sfdda=0; assimilate offline and start from the analysis.",
    ),
    OutOfScopeFeature(
        "chem_opt", "WRF-Chem coupled chemistry/aerosols",
        "Coupled gas-phase chemistry + aerosols (WRF-Chem) is out of scope for "
        "this meteorology-focused GPU port.",
        "Run a meteorology-only configuration (chem_opt=0); use offline/CTM "
        "chemistry if you need composition.",
    ),
    OutOfScopeFeature(
        "ifire", "WRF-Fire (SFIRE) wildfire spread",
        "The WRF-Fire / SFIRE level-set wildfire-spread coupling is out of scope.",
        "Set ifire=0.",
    ),
    OutOfScopeFeature(
        "wrf_hydro", "WRF-Hydro hydrological coupling",
        "WRF-Hydro surface/subsurface hydrological routing coupling is out of scope.",
        "Set wrf_hydro=0 / disable the WRF-Hydro coupler.",
    ),
    OutOfScopeFeature(
        "grid_fdda", "FDDA analysis/observation nudging",
        "Four-dimensional data assimilation (analysis/spectral/observation "
        "nudging) is out of scope; this is a pure forecast-integration port.",
        "Set grid_fdda=0 (and obs_nudge_opt=0); assimilate offline and start "
        "from the analysis.",
    ),
    OutOfScopeFeature(
        "obs_nudge_opt", "FDDA observation nudging",
        "Observation-nudging FDDA is out of scope.",
        "Set obs_nudge_opt=0.",
    ),
    OutOfScopeFeature(
        "sf_ocean_physics", "Coupled ocean mixed-layer / 3-D ocean",
        "The coupled ocean mixed-layer / 3-D ocean (sf_ocean_physics) is out of "
        "scope.",
        "Set sf_ocean_physics=0; SST is read from the input as a lower boundary.",
    ),
    OutOfScopeFeature(
        "sst_update", "Time-varying lower-boundary SST update",
        "Time-varying SST/lower-boundary auxinput updates (sst_update) are not "
        "wired in this single-input forecast path.",
        "Set sst_update=0; SST is fixed from the initial condition.",
    ),
    OutOfScopeFeature(
        "stoch_force_opt", "Stochastic physics forcing (generic)",
        "Stochastic physics forcing is out of scope (deterministic port).",
        "Set stoch_force_opt=0.",
    ),
    OutOfScopeFeature(
        "sppt", "Stochastically Perturbed Physics Tendencies (SPPT)",
        "SPPT stochastic perturbation is out of scope (deterministic port).",
        "Set sppt=0 / num_stoch_levels=0.",
    ),
    OutOfScopeFeature(
        "skebs", "Stochastic Kinetic-Energy Backscatter (SKEBS)",
        "SKEBS stochastic backscatter is out of scope (deterministic port).",
        "Set skebs=0.",
    ),
    OutOfScopeFeature(
        "spp", "Stochastically Perturbed Parameterizations (SPP)",
        "SPP stochastic parameterization perturbation is out of scope.",
        "Set spp=0 (and spp_conv/spp_pbl/spp_mp/spp_lsm=0).",
    ),
    OutOfScopeFeature(
        "rand_perturb", "Random-field stochastic perturbation",
        "Random-field stochastic perturbation is out of scope (deterministic port).",
        "Set rand_perturb=0.",
    ),
    OutOfScopeFeature(
        "vortex_interval", "Moving / vortex-following nest",
        "Moving (vortex-following or prescribed-path) nests are out of scope; the "
        "port supports static (fixed-position) one-way/two-way nests only.",
        "Use a static nest (do not set a moving-nest interval); set "
        "vortex_interval / num_moves accordingly to disable.",
    ),
    OutOfScopeFeature(
        "num_moves", "Moving nest (prescribed-move)",
        "Prescribed moving nests (num_moves>0) are out of scope; static nests only.",
        "Set num_moves=0.",
    ),
)

# Integer-enumerated out-of-scope codes live here when a recognized WRF code is a
# permanent product boundary rather than a future-port target. G3 moves urban
# BEP/BEM from out-of-scope to recognized fail-closed scaffold, so this is empty.
_OUT_OF_SCOPE_CODES: Mapping[str, dict[int, OutOfScopeFeature]] = {}

OUT_OF_SCOPE_FEATURE_KEYS: frozenset[str] = frozenset(
    f.key.lower() for f in OUT_OF_SCOPE_FEATURES
)
_OUT_OF_SCOPE_FEATURE_BY_KEY: Mapping[str, OutOfScopeFeature] = {
    f.key.lower(): f for f in OUT_OF_SCOPE_FEATURES
}


# --------------------------------------------------------------------------- #
# Recognized non-enumerated CONTROLS (no full WRF code->name catalog).         #
#                                                                              #
# These are real WRF namelist keys -- dynamics/advection switches, the         #
# MYNN-EDMF sub-option family, and physics-cadence intervals -- that the port  #
# RECOGNIZES but only wires for a SPECIFIC operational value (or value set).   #
# A recognized key set to a value the operational scan does NOT wire fails     #
# CLOSED with a named reason (RECOGNIZED_FAIL_CLOSED) -- never silently        #
# ignored.  A value the scan DOES wire is IMPLEMENTED.                         #
#                                                                              #
# This is recognition, NOT new implementation: the wired-value sets below are  #
# the ALREADY-existing operational behaviour, read from the code authorities   #
# on 2026-06-07:                                                               #
#   * advection orders frozen to h=5 / v=3 (dynamics/flux_advection.py:9-15);  #
#   * no positive-definite/monotonic scalar transport variants                 #
#     (moist_adv_opt/scalar_adv_opt 2/3/4 unimplemented -- differential        #
#     analysis P1-6);                                                          #
#   * gwd_opt=1 orographic gravity-wave drag + flow blocking IS implemented    #
#     (physics/gwd_gwdo.py; coupling.physics_couplers.gwdo_adapter; faithful   #
#     bl_gwdo_run port, oracle-validated vs pristine WRF). gwd_opt=3 (GSL) is  #
#     NOT wired;                                                               #
#   * MYNN-EDMF wired sub-config bl_mynn_edmf=1 / edmf_mom=1 / edmf_tke=0 /     #
#     mixscalars=1 / mixqt=0 / edmf_dd=0 (physics/mynn_edmf.py:7-13),          #
#     mixlength 1|2 (physics/mynn_constants.py);                               #
#   * radt honoured as the radiation cadence (radiation_cadence_steps;         #
#     nested_pipeline.py:61); bldt/cudt unread -> PBL/cumulus run every step,  #
#     so only the every-step value 0 is faithful.                             #
# Slope/topo radiation (slope_rad=1 / topo_shading=1) ARE implemented (RRTMG   #
# SW slope-radiation + topographic-shadow path, coupling.physics_couplers.     #
# _rrtmg_topography_state) and are classified IMPLEMENTED here, NOT failed.    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecognizedControl:
    """A recognized non-enumerated WRF control key with a wired-value set.

    ``wired`` is the set of values the operational scan actually runs (each is
    classified ``IMPLEMENTED``).  Any other value is ``RECOGNIZED_FAIL_CLOSED``
    with ``unwired_reason`` (a ``str`` or a ``value -> str`` callable) naming why
    the port does not run it, plus ``alternative`` (the wired recipe).
    ``integer`` is False for cadence intervals that may be fractional minutes.

    ``approximated`` marks a control whose unwired values are a documented
    CONSERVATIVE approximation rather than a wrong-substitution: an unwired value
    is classified ``RECOGNIZED_APPROXIMATED`` (a non-raising warning) instead of
    ``RECOGNIZED_FAIL_CLOSED``. This is reserved for the cumulus/PBL cadence keys
    (``cudt``/``bldt``): a positive interval asks the port to sub-step the scheme
    every N minutes, but the GPU port runs it EVERY dynamics step -- more
    frequent than requested, which cannot silently produce a wrong scheme, only a
    slightly more-expensive, more-up-to-date tendency. The run proceeds.
    """

    key: str
    label: str
    wired: frozenset[int]
    unwired_reason: object  # str | Callable[[int|float], str]
    alternative: str
    integer: bool = True
    approximated: bool = False
    # Warning text surfaced when an ``approximated`` control is set to an unwired
    # value (the run proceeds). ``None`` for non-approximated controls.
    approximation_note: object = None  # str | Callable[[int|float], str] | None

    def approximation_for(self, value: object) -> str:
        note = self.approximation_note
        if note is None:
            return ""
        return note(value) if callable(note) else str(note)

    def reason_for(self, value: object) -> str:
        reason = self.unwired_reason
        return reason(value) if callable(reason) else str(reason)


_ADV_ORDER_REASON = (
    "the port freezes the WRF advection orders to h=5 / v=3 (5th-order "
    "horizontal, 3rd-order vertical -- the WRF real-data default); other "
    "advection orders are not wired (dynamics/flux_advection.py)."
)
_ADV_OPT_REASON = (
    "recognized; the port wires the standard h5/v3 transport (0) and the "
    "WRF-canonical positive-definite (1) and monotonic (2) flux limiters "
    "(dynamics/flux_advection.advect_scalar_flux_limited, selected on the final "
    "RK3 stage; module_advect_em.F advect_scalar_pd/advect_scalar_mono). The WENO "
    "(3) and WENO-positive-definite (4) reconstruction variants are NOT yet "
    "scan-wired."
)

_RECOGNIZED_CONTROLS: tuple[RecognizedControl, ...] = (
    # --- Dynamics / advection --------------------------------------------- #
    RecognizedControl(
        "gwd_opt", "gravity-wave-drag",
        frozenset({0, 1}),
        "recognized; the port wires gwd_opt=1 (orographic gravity-wave drag + "
        "flow blocking, the Kim-GWDO of Choi & Hong 2015 -- a faithful port of "
        "module_bl_gwdo/bl_gwdo_run, physics/gwd_gwdo.py + coupling.gwdo_adapter, "
        "oracle-validated vs pristine WRF). gwd_opt=3 (GSL drag suite) is not "
        "wired. Requires the sub-grid orography statics (VAR/CON/OA1-4/OL1-4) "
        "carried in wrfinput.",
        "Use gwd_opt=0 (off) or 1 (orographic GWD on); gwd_opt=3 is not wired.",
    ),
    RecognizedControl(
        "moist_adv_opt", "moisture-advection",
        frozenset({0, 1, 2}),
        _ADV_OPT_REASON,
        "Use moist_adv_opt=0 (standard h5/v3), 1 (positive-definite) or 2 "
        "(monotonic); the WENO variants (3/4) are not wired.",
    ),
    RecognizedControl(
        "scalar_adv_opt", "scalar-advection",
        frozenset({0, 1, 2}),
        _ADV_OPT_REASON,
        "Use scalar_adv_opt=0 (standard h5/v3), 1 (positive-definite) or 2 "
        "(monotonic); the WENO variants (3/4) are not wired.",
    ),
    RecognizedControl(
        "h_sca_adv_order", "horizontal-scalar-advection-order",
        frozenset({5}),
        _ADV_ORDER_REASON,
        "Use h_sca_adv_order=5.",
    ),
    RecognizedControl(
        "v_sca_adv_order", "vertical-scalar-advection-order",
        frozenset({3}),
        _ADV_ORDER_REASON,
        "Use v_sca_adv_order=3.",
    ),
    RecognizedControl(
        "h_mom_adv_order", "horizontal-momentum-advection-order",
        frozenset({5}),
        _ADV_ORDER_REASON,
        "Use h_mom_adv_order=5.",
    ),
    RecognizedControl(
        "v_mom_adv_order", "vertical-momentum-advection-order",
        frozenset({3}),
        _ADV_ORDER_REASON,
        "Use v_mom_adv_order=3.",
    ),
    # --- PBL / cloud sub-options ------------------------------------------ #
    RecognizedControl(
        "icloud_bl", "PBL-cloud-coupling",
        frozenset({0}),
        "recognized; the bl_pbl=MYNN <-> radiation sub-grid cloud-fraction "
        "coupling (icloud_bl=1) is NOT scan-wired in v0.12.0 (the MYNN cloud "
        "fraction is computed but not fed to the radiation cloud overlap).",
        "Set icloud_bl=0.",
    ),
    RecognizedControl(
        "bl_mynn_tkeadvect", "MYNN-TKE-advection",
        frozenset({0}),
        "recognized; MYNN prognostic-TKE horizontal advection "
        "(bl_mynn_tkeadvect=.true., the qke_adv scalar) is NOT scan-wired in "
        "v0.12.0 (qke is carried but not advected as a transported scalar).",
        "Set bl_mynn_tkeadvect=.false. (0).",
    ),
    RecognizedControl(
        "bl_mynn_edmf", "MYNN-EDMF-massflux",
        frozenset({1}),
        "recognized; the port wires the WRF-default MYNN-EDMF mass-flux ON "
        "(bl_mynn_edmf=1, physics/mynn_edmf.py). The EDMF-off path is not "
        "separately wired.",
        "Use bl_mynn_edmf=1 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_edmf_mom", "MYNN-EDMF-momentum-massflux",
        frozenset({1}),
        "recognized; the port wires the WRF-default EDMF momentum mass-flux ON "
        "(bl_mynn_edmf_mom=1).",
        "Use bl_mynn_edmf_mom=1 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_edmf_tke", "MYNN-EDMF-TKE-massflux",
        frozenset({0}),
        "recognized; the port wires the WRF-default EDMF TKE mass-flux OFF "
        "(bl_mynn_edmf_tke=0). The TKE mass-flux path is not wired.",
        "Use bl_mynn_edmf_tke=0 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_edmf_dd", "MYNN-EDMF-downdraft",
        frozenset({0}),
        "recognized; the MYNN-EDMF stochastic downdraft (bl_mynn_edmf_dd=1) is "
        "not wired (the port runs the no-downdraft default).",
        "Use bl_mynn_edmf_dd=0 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_mixscalars", "MYNN-EDMF-scalar-mixing",
        frozenset({1}),
        "recognized; the port wires the WRF-default EDMF scalar mixing ON "
        "(bl_mynn_mixscalars=1).",
        "Use bl_mynn_mixscalars=1 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_mixqt", "MYNN-EDMF-total-water-mixing",
        frozenset({0}),
        "recognized; the port mixes qv/qc separately (bl_mynn_mixqt=0, the WRF "
        "'mix water vapor only' path); the total-water (qt) mixing variant "
        "(bl_mynn_mixqt=1) is not wired.",
        "Use bl_mynn_mixqt=0 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_mixlength", "MYNN-mixing-length",
        frozenset({1, 2}),
        "recognized; the port wires the WRF MYNN mixing-length options 1 "
        "(nonlocal/BouLac-blend) and 2 (local); other mixing-length options "
        "are not wired.",
        "Use bl_mynn_mixlength=1 or 2.",
    ),
    # --- Aerosol-aware Thompson (mp=28) input sub-options ------------------ #
    # v0.16: the port runs ONLY the WRF thompson_init climatological self-init
    # path (use_aero_icbc=.false.; nwfa/nifa cold-started from the BL-following
    # exponential profiles + the fake surface nwfa2d emission). Reading aerosol
    # ICs/BCs from WPS/met_em (use_aero_icbc=.true.) is NOT wired, so a .true.
    # selection fails CLOSED (never a silent wrong-input-path run).
    RecognizedControl(
        "use_aero_icbc", "Thompson-aerosol-ICBC-input",
        frozenset({0}),
        "recognized; aerosol-aware Thompson (mp=28) is wired ONLY for the "
        "climatological self-init path (use_aero_icbc=.false.: thompson_init "
        "BL-following nwfa/nifa profiles + fake surface nwfa2d emission). "
        "Reading QNWFA/QNIFA ICs/BCs from WPS/met_em (use_aero_icbc=.true.) "
        "is not wired.",
        "Set use_aero_icbc=.false. (climatological aerosol self-init).",
    ),
    RecognizedControl(
        "wif_input_opt", "water-ice-friendly-aerosol-input",
        frozenset({1}),
        "recognized; only wif_input_opt=1 is accepted with the mp=28 "
        "climatological self-init path. Other water-/ice-friendly aerosol "
        "input modes are not wired.",
        "Use wif_input_opt=1.",
    ),
    RecognizedControl(
        "aer_init_opt", "aerosol-init",
        frozenset({0, 1}),
        "recognized; the port wires aer_init_opt=0/1 (the WRF thompson_init "
        "climatological aerosol initialization family). Other aerosol-init "
        "modes are not wired.",
        "Use aer_init_opt=0 or 1.",
    ),
    # --- Physics cadence intervals (minutes) ------------------------------ #
    RecognizedControl(
        "radt", "radiation-cadence",
        frozenset(),  # any positive value honoured; see classify_control()
        "radt is honoured as the radiation call cadence "
        "(radiation_cadence_steps = round(radt*60/dt_s)); a positive interval "
        "is recognized and implemented.",
        "Set radt to the desired radiation interval in minutes (e.g. 30).",
        integer=False,
    ),
    RecognizedControl(
        "bldt", "PBL-cadence",
        frozenset({0}),
        "recognized; the port calls the PBL scheme EVERY dynamics step "
        "(bldt=0 semantics). A nonzero PBL sub-stepping interval (bldt>0) is "
        "not implemented.",
        "Set bldt=0 (call PBL every step), or accept the every-step approximation.",
        integer=False,
        approximated=True,
        approximation_note=(
            lambda v: (
                f"bldt={v} cadence not honored; the GPU port runs the PBL scheme "
                f"EVERY dynamics step -- more frequently than the requested "
                f"{v}-minute sub-stepping interval, a conservative approximation "
                f"(more up-to-date boundary-layer tendencies, never a different "
                f"scheme). The run proceeds. Set bldt=0 to request this exactly."
            )
        ),
    ),
    RecognizedControl(
        "cudt", "cumulus-cadence",
        frozenset({0}),
        "recognized; the port calls the cumulus scheme EVERY dynamics step "
        "(cudt=0 semantics). A nonzero cumulus sub-stepping interval (cudt>0) "
        "is not implemented.",
        "Set cudt=0 (call cumulus every step), or accept the every-step approximation.",
        integer=False,
        approximated=True,
        approximation_note=(
            lambda v: (
                f"cudt={v} cadence not honored; the GPU port runs the cumulus "
                f"scheme EVERY dynamics step -- more frequently than the requested "
                f"{v}-minute sub-stepping interval, a conservative approximation "
                f"(more up-to-date convective tendencies, never a different "
                f"scheme). The run proceeds. Set cudt=0 to request this exactly."
            )
        ),
    ),
)

RECOGNIZED_CONTROL_KEYS: frozenset[str] = frozenset(
    c.key.lower() for c in _RECOGNIZED_CONTROLS
)
# Cadence controls whose unwired (positive) values are a non-raising,
# conservative approximation (run-every-step) rather than a fail-closed
# rejection: cudt / bldt. A naive user pointing the standalone CLI at a real
# WRF namelist (cudt=5, bldt=0) must RUN, not be rejected.
APPROXIMATED_CONTROL_KEYS: frozenset[str] = frozenset(
    c.key.lower() for c in _RECOGNIZED_CONTROLS if c.approximated
)
_RECOGNIZED_CONTROL_BY_KEY: Mapping[str, RecognizedControl] = {
    c.key.lower(): c for c in _RECOGNIZED_CONTROLS
}

# Implemented non-enumerated controls (radiation slope/topo path). slope_rad=1
# and topo_shading=1 ARE wired (RRTMG SW slope-radiation + topographic-shadow);
# slope_rad=2 (the WRF "slope + shadow" combined flag) is NOT separately wired.
_IMPLEMENTED_CONTROLS: Mapping[str, frozenset[int]] = {
    "slope_rad": frozenset({0, 1}),
    "topo_shading": frozenset({0, 1}),
}
_IMPLEMENTED_CONTROL_REASON: Mapping[str, tuple[str, str, str]] = {
    # key: (label, unwired-reason, alternative)
    "slope_rad": (
        "slope-radiation",
        "recognized; the port wires slope_rad=1 (RRTMG SW slope-radiation). "
        "slope_rad=2 (WRF combined slope+shadow flag) is not separately wired.",
        "Use slope_rad=0 (off) or 1 (slope radiation on).",
    ),
    "topo_shading": (
        "topographic-shading",
        "recognized; the port wires topo_shading=1 (RRTMG SW topographic "
        "shadowing). Other topo_shading values are not wired.",
        "Use topo_shading=0 (off) or 1 (topographic shadowing on).",
    ),
}
IMPLEMENTED_CONTROL_KEYS: frozenset[str] = frozenset(_IMPLEMENTED_CONTROLS)


def classify_control(key: str, value: object) -> SchemeSupport | None:
    """Classify a recognized non-enumerated WRF control key, or return ``None``.

    Returns ``None`` when ``key`` is not a recognized control (so the caller can
    fall through to silent-pass for keys the port deliberately does not gate).
    Otherwise returns a :class:`SchemeSupport`:

    * ``IMPLEMENTED`` when ``value`` is in the operationally-wired set;
    * ``RECOGNIZED_FAIL_CLOSED`` (with a named reason + alternative) otherwise.

    Booleans/strings (``.true.``/``.false.``) are coerced to ``1``/``0`` so a
    Fortran-style ``bl_mynn_tkeadvect = .false.`` reads as the wired value ``0``.
    """

    lkey = key.lower()

    impl = _IMPLEMENTED_CONTROLS.get(lkey)
    if impl is not None:
        code = _coerce_int_or_bool(value)
        label, reason, alternative = _IMPLEMENTED_CONTROL_REASON[lkey]
        if isinstance(code, bool):
            code = int(code)
        if code in impl:
            return SchemeSupport(
                key=lkey,
                code=code,
                status=SupportStatus.IMPLEMENTED,
                reason="Operationally wired into the GPU scan.",
                alternative="",
                wrf_name=label,
            )
        return SchemeSupport(
            key=lkey,
            code=code,
            status=SupportStatus.RECOGNIZED_FAIL_CLOSED,
            reason=reason,
            alternative=alternative,
            wrf_name=label,
        )

    control = _RECOGNIZED_CONTROL_BY_KEY.get(lkey)
    if control is None:
        return None

    # radt: any positive interval is honoured as the radiation cadence.
    if lkey == "radt":
        numeric = _coerce_number(value)
        wired = numeric is not None and numeric > 0
        status = SupportStatus.IMPLEMENTED if wired else SupportStatus.RECOGNIZED_FAIL_CLOSED
        return SchemeSupport(
            key=lkey,
            code=_coerce_int_or_bool(value),
            status=status,
            reason=(
                control.reason_for(value)
                if wired
                else "recognized; radt must be a positive radiation interval "
                "(minutes) to set the radiation call cadence."
            ),
            alternative=control.alternative,
            wrf_name=control.label,
        )

    # ``_coerce_number`` understands Fortran logicals (``.false.`` -> 0.0) and
    # numeric strings; it is the reliable read for set membership. Fall back to
    # ``_coerce_int_or_bool`` only when the value is genuinely non-numeric.
    numeric = _coerce_number(value)
    if control.integer:
        if numeric is not None and float(numeric).is_integer():
            compare: object = int(numeric)
            code: object = compare
        else:
            code = _coerce_int_or_bool(value)
            compare = int(code) if isinstance(code, bool) else code
    else:
        # Treat an exact integer (e.g. 0.0 minutes) as its int for set membership.
        if numeric is not None and float(numeric).is_integer():
            compare = int(numeric)
        elif numeric is not None:
            compare = numeric
        else:
            compare = _coerce_int_or_bool(value)
        code = compare

    if isinstance(compare, int) and compare in control.wired:
        return SchemeSupport(
            key=lkey,
            code=code,
            status=SupportStatus.IMPLEMENTED,
            reason="Operationally wired into the GPU scan.",
            alternative="",
            wrf_name=control.label,
        )
    # An unwired value of an APPROXIMATED cadence control is a non-raising
    # WARNING (the run proceeds), not a fail-closed rejection: the GPU port runs
    # the scheme every step, a conservative approximation of the requested
    # sub-stepping cadence -- it can never become a wrong scheme.
    if control.approximated:
        return SchemeSupport(
            key=lkey,
            code=code,
            status=SupportStatus.RECOGNIZED_APPROXIMATED,
            reason=control.approximation_for(value),
            alternative=control.alternative,
            wrf_name=control.label,
        )
    return SchemeSupport(
        key=lkey,
        code=code,
        status=SupportStatus.RECOGNIZED_FAIL_CLOSED,
        reason=control.reason_for(value),
        alternative=control.alternative,
        wrf_name=control.label,
    )


def _coerce_number(value: object) -> float | None:
    """Best-effort numeric read of a namelist value (``None`` if non-numeric)."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().strip("'\"").lower()
        if text in {".true.", "true", "t", "yes"}:
            return 1.0
        if text in {".false.", "false", "f", "no", ""}:
            return 0.0
        try:
            return float(text.replace("d", "e").replace("D", "e"))
        except ValueError:
            return None
    return None


def classify_scheme(key: str, code: int) -> SchemeSupport:
    """Classify one ``key=code`` selection into a :class:`SchemeSupport`.

    Handles the seven enumerated physics groups plus the gated dynamics keys.
    For out-of-scope *feature switches* (``chem_opt`` etc.) use
    :func:`classify_feature_switch` instead -- those have no enumerated catalog.
    """

    code = int(code)

    # 1) Out-of-scope enumerated codes, if any.
    oos_codes = _OUT_OF_SCOPE_CODES.get(key)
    if oos_codes is not None and code in oos_codes:
        feat = oos_codes[code]
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.OUT_OF_SCOPE,
            reason=feat.reason,
            alternative=feat.alternative,
            wrf_name=_scheme_name_or_none(key, code),
        )

    # 2) Operationally implemented (physics + dynamics).
    impl = _IMPLEMENTED.get(key) or _DYNAMICS_IMPLEMENTED.get(key)
    if impl is not None and code in impl:
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.IMPLEMENTED,
            reason="Operationally wired into the GPU scan.",
            alternative="",
            wrf_name=_scheme_name_or_none(key, code),
        )

    # 3) Reference-only (oracle-backed, not scan-wired).
    ref = _REFERENCE_ONLY.get(key)
    if ref is not None and code in ref:
        reason, alternative = ref[code]
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.REFERENCE_ONLY,
            reason=reason,
            alternative=alternative,
            wrf_name=_scheme_name_or_none(key, code),
        )

    # 4) Recognized WRF v4 option, not implemented -> fail closed.
    scheme = wrf_scheme_name(key, code)
    if scheme is not None:
        phys = _PHYSICS_FAIL_CLOSED_REASON.get(key, {}).get(code)
        if phys is not None:
            # A documented v0.18->v1.0 architecture-boundary scheme (e.g. CLM4 /
            # CTSM): a SPECIFIC named reason + alternative, never the generic
            # "NOT YET IMPLEMENTED" text -- selecting it errors cleanly.
            reason, alternative = phys
        else:
            per_code = _PER_CODE_FAIL_CLOSED_REASON.get(key, {}) | _SCHEME_FAIL_CLOSED_REASON.get(key, {})
            reason = per_code.get(
                code,
                f"{scheme.name} is a recognized WRF v4 {_label(key)} option that is "
                f"NOT YET IMPLEMENTED in the GPU port.",
            )
            alternative = _DEFAULT_ALTERNATIVE.get(key, "")
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.RECOGNIZED_FAIL_CLOSED,
            reason=reason,
            alternative=alternative,
            wrf_name=scheme.name,
        )

    # 5) Not a recognized WRF v4 option at all. Modeled as fail-closed with a
    #    distinct reason (the namelist validator reports it as "not recognized").
    if key in WRF_SCHEME_CATALOG:
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.RECOGNIZED_FAIL_CLOSED,
            reason=f"{code} is not a recognized WRF v4 {_label(key)} option.",
            alternative=_DEFAULT_ALTERNATIVE.get(key, ""),
            wrf_name=None,
        )

    # Unknown key (no catalog): treat as fail-closed-unknown.
    return SchemeSupport(
        key=key,
        code=code,
        status=SupportStatus.RECOGNIZED_FAIL_CLOSED,
        reason=f"{key} is not a gated namelist option in this port.",
        alternative="",
        wrf_name=None,
    )


def classify_feature_switch(key: str, value: object) -> SchemeSupport | None:
    """Classify an out-of-scope *feature switch* (e.g. ``chem_opt``, ``sppt``).

    Returns an ``OUT_OF_SCOPE`` :class:`SchemeSupport` when ``key`` is a known
    out-of-scope feature switch AND ``value`` is truthy (non-zero / ``.true.``).
    Returns ``None`` when the key is not an out-of-scope feature switch or the
    switch is off (so a meteorology-only namelist that leaves ``chem_opt=0``
    passes cleanly).
    """

    feat = _OUT_OF_SCOPE_FEATURE_BY_KEY.get(key.lower())
    if feat is None:
        return None
    if not _is_truthy(value):
        return None
    return SchemeSupport(
        key=feat.key,
        code=_coerce_int_or_bool(value),
        status=SupportStatus.OUT_OF_SCOPE,
        reason=feat.reason,
        alternative=feat.alternative,
        wrf_name=feat.feature,
    )


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().strip("'\"").lower()
        if text in {".true.", "true", "t", "yes"}:
            return True
        if text in {".false.", "false", "f", "no", ""}:
            return False
        try:
            return float(text.replace("d", "e").replace("D", "e")) != 0
        except ValueError:
            # A non-empty, non-boolean string (e.g. a filename) counts as "set".
            return True
    return bool(value)


def _coerce_int_or_bool(value: object) -> int | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip().strip("'\"").lower()
        if text in {".true.", "true", "t", "yes"}:
            return True
        try:
            return int(float(text.replace("d", "e").replace("D", "e")))
        except ValueError:
            return True
    return bool(value)


def _scheme_name_or_none(key: str, code: int) -> str | None:
    scheme = wrf_scheme_name(key, code)
    return scheme.name if scheme is not None else None


# Keys whose full WRF v4 code enumeration the catalog classifies per-code.
CATALOGED_SCHEME_KEYS: tuple[str, ...] = tuple(WRF_SCHEME_CATALOG.keys())


def iter_full_catalog() -> Iterable[SchemeSupport]:
    """Yield a :class:`SchemeSupport` for every WRF v4 code of every gated key.

    Used to render the docs support table and to assert catalog totals. Out-of-
    scope feature switches (which have no enumerated catalog) are reported by
    :data:`OUT_OF_SCOPE_FEATURES` separately.
    """

    for key, codes in WRF_SCHEME_CATALOG.items():
        for code in sorted(codes):
            yield classify_scheme(key, code)


def status_counts() -> dict[SupportStatus, int]:
    """Count enumerated ``key=code`` classifications per status (for reports)."""

    counts: dict[SupportStatus, int] = {s: 0 for s in SupportStatus}
    for support in iter_full_catalog():
        counts[support.status] += 1
    return counts


def assert_catalog_consistent() -> None:
    """Fail-closed invariants keeping the catalog honest vs the code authorities.

    * Every ``IMPLEMENTED`` enumerated code must be a recognized WRF v4 option.
    * The implemented physics set must equal the frozen ``ACCEPTED_*`` matrix
      MINUS the reference-only options (so ``IMPLEMENTED`` never over-claims a
      reference-only scheme, and never silently drops an accepted one).
    * Implemented / reference-only / out-of-scope-code sets must be disjoint.
    """

    for support in iter_full_catalog():
        if support.status is SupportStatus.IMPLEMENTED and support.key in WRF_SCHEME_CATALOG:
            assert support.wrf_name is not None, (
                f"implemented {support.key}={support.code} is not a recognized WRF option"
            )

    for key, impl in _IMPLEMENTED.items():
        ref = frozenset(_REFERENCE_ONLY.get(key, {}).keys())
        accepted = frozenset(ACCEPTED_NAMELIST_OPTIONS[key])
        # accepted == implemented ∪ reference_only (no over-claim, no silent drop).
        assert impl | ref == accepted, (
            f"{key}: implemented({sorted(impl)}) ∪ reference({sorted(ref)}) "
            f"!= accepted({sorted(accepted)})"
        )
        assert not (impl & ref), f"{key}: an option is both implemented and reference_only"

    for key, codes in _OUT_OF_SCOPE_CODES.items():
        impl = _IMPLEMENTED.get(key, frozenset()) | _DYNAMICS_IMPLEMENTED.get(key, frozenset())
        assert not (impl & set(codes)), f"{key}: an option is both implemented and out_of_scope"

    # The recognized-control key namespaces must be disjoint from each other,
    # from the out-of-scope feature switches, and from the enumerated catalog
    # keys -- so a key has exactly one classification authority and a value can
    # never be silently double-classified.
    assert not (RECOGNIZED_CONTROL_KEYS & IMPLEMENTED_CONTROL_KEYS), (
        "a control key is both a recognized-control and an implemented-control"
    )
    control_keys = RECOGNIZED_CONTROL_KEYS | IMPLEMENTED_CONTROL_KEYS
    assert not (control_keys & OUT_OF_SCOPE_FEATURE_KEYS), (
        "a control key is also an out-of-scope feature switch"
    )
    assert not (control_keys & {k.lower() for k in WRF_SCHEME_CATALOG}), (
        "a control key collides with an enumerated WRF scheme key"
    )
    for control in _RECOGNIZED_CONTROLS:
        assert control.alternative.strip(), f"{control.key} missing alternative"
        assert control.reason_for(0).strip(), f"{control.key} missing reason"
        if control.approximated:
            # An approximated control must supply a non-empty warning note for a
            # representative unwired value, so the surfaced warning is never blank.
            assert control.approximation_for(5).strip(), (
                f"{control.key} marked approximated but has no approximation note"
            )


__all__ = [
    "SupportStatus",
    "SchemeSupport",
    "OutOfScopeFeature",
    "OUT_OF_SCOPE_FEATURES",
    "OUT_OF_SCOPE_FEATURE_KEYS",
    "RecognizedControl",
    "RECOGNIZED_CONTROL_KEYS",
    "APPROXIMATED_CONTROL_KEYS",
    "IMPLEMENTED_CONTROL_KEYS",
    "classify_control",
    "CATALOGED_SCHEME_KEYS",
    "classify_scheme",
    "classify_feature_switch",
    "iter_full_catalog",
    "status_counts",
    "assert_catalog_consistent",
]


if __name__ == "__main__":  # pragma: no cover - manual audit entrypoint
    assert_catalog_consistent()
    counts = status_counts()
    print("scheme_catalog consistent. Enumerated key=code classifications:")
    for status, n in counts.items():
        print(f"  {status.value:24s} {n}")
    print(f"  out_of_scope feature switches: {len(OUT_OF_SCOPE_FEATURES)}")
