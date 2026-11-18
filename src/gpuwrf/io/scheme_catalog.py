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
                               (``bldt``) ask the port to sub-step those
                               physics every N minutes, but the GPU port calls
                               them EVERY dynamics step (more frequent than
                               requested). Selecting a positive cadence does NOT
                               fail closed -- the run PROCEEDS and a WARNING names
                               the approximation. This mirrors the operational
                               pipeline, which already runs cumulus/PBL every
                               step regardless of ``bldt``. KF uses WRF STEPCU
                               from ``cudt``. It is NEVER
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
  the operational scan. Explicit experimental CPU-only entries identify that
  scope and refuse GPU selection before JAX initialization. Ground truth comes from
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
    # mp=40 Morrison-aerosol (v0.3.4 O1) is scan-wired at WRF's stand-alone point
    # aercu_opt=0 (coupling.scan_adapters.morrison_aero_adapter; Nc = WRF qnc):
    # CPU-oracle-qualified (proofs/v034/f2_oracles/morrison_aero_opt0, fp64 port
    # <=1.6e-11 rel, fp32 REAL port in the base fp32 band, 7 regimes incl. ice
    # nucleation); GPU/coupled-forecast qualification pending. aercu_opt>0 needs
    # cu_physics=11 (MSKF); aerosol coupling is not ported and is refused.
    # Pristine physics_init leaves mp40 has_reqc/i/s=0 at aercu_opt=0.
    # Morrison radii activate in radiation_driver's aercu_opt>0 MSKF blend
    # (not ported, refused); Registry allocation alone does not enable them.
    # mp=18 NSSL 2-moment is the v0.3.4 JAX port (physics.nssl2mom; proofs/v034/f2_oracles/nssl_2mom):
    # CPU-oracle-qualified including cumulus number sources; single domains, GPU qualification pending.
    "mp_physics": frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 26, 28, 38, 40, 50, 51, 52, 53, 55, 56, 97, 11}),

    # cu=5 Grell-3D + cu=93 Grell-Devenyi (v0.3.4 o1-grell): line-faithful JAX ports
    # of G3DRV/conv_grell_spread3d and GRELLDRV, machine-precision vs pristine-WRF
    # multi-column tile oracles (proofs/v034); CPU-oracle-qualified; GPU/coupled-
    # forecast qualification pending. NOT oracle-verified (no staged column exercises
    # them; mutants M1/M2 survive): the CUP_enss keep_going downdraft-origin search and
    # GD neg_check. Real-case CPU smoke (Swiss d01) too slow on CPU: 3 root steps ran
    # before a 45 min timeout without a finiteness receipt; the 12x12 operational-step
    # smoke (tests/test_v034_grell_cumulus_wiring.py) is the standing coupled evidence.
    "cu_physics": frozenset({0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 94, 95, 96, 99}),
    # bl=2 MYJ + sf=2 Janjic Eta are the v0.13 traceable MYJ pair (operationally
    # scan-wired via physics.myj_adapters + runtime.operational_mode; mandatory pair).
    # bl=99 MRF is the v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F
    # (savepoint-parity gated, proofs/v013/mrf_oracle.py); consumes the revised-MM5
    # surface layer (sf_sfclay=1), no new surface partner needed.
    # bl=3 GFS is the v0.17 jit/vmap-traceable port of phys/module_bl_gfs.F
    # (BL_GFS -> MONINP, savepoint-parity gated, proofs/v017/gfs_oracle.py ~1e-13);
    # nonlocal-K; WRF admits it only with isfc=2 (module_physics_init.F:3743), so it
    # pairs with sf_sfclay=3 and consumes that single SF_GFS solve (t-cfix-sfc F13).
    # bl=11 Shin-Hong is the scale-aware PBL port, BYTE vs a pristine-WRF REAL4
    # public-call oracle incl. the TKE diagnostic (proofs/t_shinhong);
    # CPU-oracle-qualified, GPU/coupled-forecast qualification pending. bl=12 GBM is the
    # v0.18 JAX/vmap moist prognostic-TKE PBL port.
    # bl=9 CAM-UW is the v0.3.4 faithful r8 port (proofs/v034/camuw_oracle):
    # CPU-oracle-qualified; GPU/coupled-forecast qualification pending.
    "bl_pbl_physics": frozenset({0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17, 99}),
    # sf_sfclay 3 (NCEP-GFS) + 91 (old-MM5) are v0.13 Tier-3 scan-wired surface
    # layers (coupling.scan_adapters.{gfs_sfclay_adapter,sfclay_old_mm5_adapter};
    # fp64 pristine-WRF oracle-validated; B2 kinematic flux handles).
    "sf_sfclay_physics": frozenset({0, 1, 2, 3, 4, 5, 7, 10, 91}),
    # sf_surface=1 (5-layer thermal-diffusion slab LSM) is v0.17 operationally
    # scan-wired (coupling.slab_surface_hook.slab_surface_step over the fp64
    # pristine-WRF-oracle-validated physics.lsm_slab.slab_columns SLAB1D port;
    # advances the 5-layer TSLB land carry from GSW/GLW + a TMN/THC/EMISS bundle).
    # sf_surface=7 (Pleim-Xiu 2-layer ISBA LSM) is v0.17 operationally scan-wired
    # (coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step over the fp64
    # pristine-WRF-oracle-validated physics.lsm_pleim_xiu SURFPX+QFLUX port).
    # sf_surface=3 (RUC LSM) is v0.3.4 scan-wired (lane o1-ruc): faithful JAX LSMRUC
    # port physics.ruclsm via coupling.ruc_surface_hook.ruc_surface_step, CPU-oracle-
    # qualified (proofs/v034/oracle/ruclsm); see _IMPLEMENTED_NOTES.
    "sf_surface_physics": frozenset({0, 1, 2, 3, 4, 7, 8}),
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
    # ra_lw=3 (CAM LW) is v0.3.4 scan-wired (lane o1-camrad): physics.ra_cam port of
    # phys/module_ra_cam.F via coupling.cam_radiation; see _IMPLEMENTED_NOTES.
    "ra_lw_physics": frozenset({0, 1, 3, 4, 5, 7, 31, 99}),
    # ra_sw=1 (Dudhia, Stephens-1984 broadband SW) and ra_sw=2 (GSFC/Chou-Suarez
    # multi-band delta-Eddington SW) are now operationally scan-wired
    # (coupling.physics_couplers.dudhia_sw_theta_tendency / gsfc_sw_theta_tendency,
    # dispatched in runtime.operational_mode by OperationalNamelist.ra_sw_physics).
    # ra_sw=7/99 (FLG/GFDL-Eta) are accepted reference-only
    # in v0.18 for real-WRF oracle/parity work, not operational scan-wiring.
    # ra_sw=3 (CAM SW) is v0.3.4 scan-wired (lane o1-camrad), see _IMPLEMENTED_NOTES.
    "ra_sw_physics": frozenset({0, 1, 2, 3, 4, 5, 7, 99}),
}

# Recognized WRF schemes with an oracle-backed reference path that the
# operational scan fail-closes (selectable for a reference comparison, NOT an
# operational run). Some entries are GREEN; some are RED and say so in the reason.
# reason = the named scan-unwired reason; alternative = the operational swap.
_REFERENCE_ONLY: Mapping[str, dict[int, tuple[str, str]]] = {
    # v0.23 F2 staged single-column oracles for NSSL (18) and Morrison-aerosol (40); both graduated
    # to IMPLEMENTED in v0.3.4 (18: physics.nssl2mom, 40: aercu_opt=0) -- see _IMPLEMENTED.
    "mp_physics": {
    },
    "cu_physics": {
        # cu=4 scale-aware GFS SAS (v0.3.4 o1-sas, manager decision 2026-10-10): pristine WRF
        # ARW refuses it (share/module_check_a_mundo.F:671 FATAL; the call path would also pass
        # an absent DYNMM as DY). The faithful kernel (physics.cumulus_scalesas, bitwise vs the
        # pristine WRF REAL build of module_cu_scalesas.F, proofs/v034/scalesas) and its scan
        # wiring (coupling.scalesas_adapter) are Python-API only: module-oracle-qualified,
        # GPU/coupled qualification pending. The CLI refuses cu=4 pre-JAX like WRF.
        4: (
            "WRF ARW rejects cu_physics=4 (check_a_mundo); scale-aware SAS kernel is available via "
            "the Python API only, no CPU-WRF reference exists",
            "Use cu_physics=1/2/3/5/6/16/93 for CLI runs (WRF suggests 95, reference-only here); "
            "API: gpuwrf.physics.cumulus_scalesas / gpuwrf.coupling.scalesas_adapter.",
        ),
        # v0.13 Tier-3 cumulus batch: New-Tiedtke(16) / KSAS(14) / Grell-3D(5) each
        # have a single-column fp64 pristine-WRF oracle staged
        # (proofs/v013/oracle/cumulus, savepoints/cumulus), but their traceable JAX
        # column kernels are a documented carry-over. They are REFERENCE_ONLY:
        # registry-accepted so a single-column reference comparison can be run, but
        # fail-closed in the operational GPU scan (never silently wrong).
        # cu=16 New-Tiedtke graduated to IMPLEMENTED in v0.23 F2: faithful fp64
        # kernel machine-precision-proven vs the v0.13 oracle savepoints
        # (cumulus_ntiedtke + cumulus_ntiedtke_jax), scan-wired via
        # coupling.scan_adapters.ntiedtke_adapter.
        # cu=5 Grell-3D and cu=93 Grell-Devenyi graduated to IMPLEMENTED in v0.3.4
        # (o1-grell; CPU-oracle-qualified, GPU qualification pending).

    },
    # sf_surface_physics=1 (5-layer thermal-diffusion slab LSM) was REFERENCE_ONLY
    # (JAX-ported + fp64 oracle, no operational hook); it is now operationally
    # scan-wired via coupling.slab_surface_hook.slab_surface_step (IMPLEMENTED
    # above), so it is no longer listed here.
    # v0.18 RA tail: CAM/Goddard/FLG/GFDL-Eta have exact-driver real-WRF
    # savepoints in proofs/v018/savepoints/ra_tail_wrf. They remain REFERENCE_ONLY
    # because no faithful traceable JAX kernels are wired into the operational scan.
    "ra_lw_physics": {},
    "ra_sw_physics": {},
    "bl_pbl_physics": {},
    # bl_pbl_physics=2 (MYJ) + sf_sfclay_physics=2 (Janjic Eta) were REFERENCE_ONLY
    # (host-NumPy savepoint kernels); they are now operationally scan-wired as a
    # mandatory pair via the JAX-traceable physics.bl_myj / physics.sf_myj rewrites
    # (IMPLEMENTED above), so they are no longer listed here. bl=3 GFS is IMPLEMENTED.
    # bl_pbl_physics=9 (CAM-UW) is IMPLEMENTED (v0.3.4): the v023 oracle was itself
    # defective (no esinti saturation table); proofs/v034/camuw_oracle replaces it.
    # ra_lw_physics=1 (classic RRTM LW) was REFERENCE_ONLY (host-NumPy kernel); it
    # is now operationally scan-wired via the JAX-traceable physics.ra_lw_rrtm_jax
    # rewrite (IMPLEMENTED above), so it is no longer listed here.
    # ra_sw_physics=1 (Dudhia) was REFERENCE_ONLY; it is now operationally
    # scan-wired (IMPLEMENTED above), so it is no longer listed here.
}


# v0.3.4 (lane o1-nlbind, measured): scan-wired codes that ``gpuwrf run`` CANNOT
# run under the release defaults (REAL32 carry, GPUWRF_FAST_DEFAULTS on): through
# the real CLI loader on the Swiss RD11 case the production step cannot be traced
# (proofs/o1_nlbind/cli_support_matrix.json).  Until v0.3.3 the CLI never bound
# these keys, so the crash was masked by silently running the default scheme.
# They move IMPLEMENTED -> REFERENCE_ONLY (validated column kernels, not runnable
# operationally) so validate_operational_namelist refuses them before compute.
_RELEASE_CARRY_UNRUNNABLE: Mapping[str, dict[int, str]] = {
    "mp_physics": {
        # v0.4.0 O2 (o2-rootbdy): 24/28/40 now take WRF's flow_dep_bdy for their extra root
        # species (root_scalar_rk1_split flow_only) and are IMPLEMENTED again (_IMPLEMENTED_NOTES).
    },
}
_RELEASE_CARRY_ALTERNATIVE: Mapping[str, str] = {
    "mp_physics": "Use mp_physics=8 (Thompson, release default) or another code that "
    "`gpuwrf run` lists as operational.",
}
_IMPLEMENTED = {
    key: codes - frozenset(_RELEASE_CARRY_UNRUNNABLE.get(key, {}))
    for key, codes in _IMPLEMENTED.items()
}
_REFERENCE_ONLY = {
    **_REFERENCE_ONLY,
    **{
        key: {
            **_REFERENCE_ONLY.get(key, {}),
            **{
                code: (
                    f"savepoint-parity column kernel and scan adapter exist, but under the "
                    f"release defaults the operational step cannot be built: {why}. "
                    f"Not runnable by `gpuwrf run` (v0.3.4 CLI probe; <=v0.3.3 silently ran "
                    f"the default scheme instead)",
                    _RELEASE_CARRY_ALTERNATIVE[key],
                )
                for code, why in codes.items()
            },
        }
        for key, codes in _RELEASE_CARRY_UNRUNNABLE.items()
    },
}

def _label(key: str) -> str:
    return WRF_PARAM_LABEL.get(key, key)


# Per-key fallback alternative text used for RECOGNIZED_FAIL_CLOSED schemes.
_DEFAULT_ALTERNATIVE: Mapping[str, str] = {
    "mp_physics": "Use one of mp_physics=0/1/2/3/4/6/8/9/10/13/14/16/17/18/19/21/22/24/26/28/40/97 (8=Thompson is the "
    "operational default; 13=SBU-YLin; 18=NSSL 2-moment (single domains, supported cumulus; CPU-oracle-qualified, GPU/coupled-forecast qualification pending); 97=Goddard "
    "GCE single-moment 3-ice; 24=WSM7, 28=aerosol-aware Thompson, 40=Morrison-aerosol: single domains). "
    "10=Morrison, 14/16/26=WDM5/6/7: single domains; number transport CPU-qualified, GPU qualification pending.",
    "cu_physics": "Use one of cu_physics=0/1/2/3/5/6/14/16/93/94/95/96/99 (1=Kain-Fritsch eta, "
    "3=Grell-Freitas, 5=Grell-3D, 93=Grell-Devenyi, 6/16=Tiedtke family requires "
    "active flux-form moisture advection for RQVFTEN). Reference-only cumulus "
    "option 4 fails closed in the operational scan.",
    "bl_pbl_physics": "Use one of bl_pbl_physics=0/1/2/3/5/7/8/9/10/11/12/16/17/99 (5=MYNN, 1=YSU, 2=MYJ "
    "[pair with sf_sfclay_physics=2], 3=GFS [surface pair 3], 7=ACM2, 8=BouLac, 9=CAM-UW, 11=Shin-Hong, "
    "10=TEMF [surface pair 10], 12=GBM, 16/17=experimental EEPS/KEPS [surface pair 1], 99=MRF). "
    "PBL4 is reference-only and fails closed in the operational scan.",
    "sf_sfclay_physics": "Use one of sf_sfclay_physics=0/1/2/3/5/7/91 (5=MYNN-SL, "
    "1=revised-MM5, 2=Janjic Eta [pair with bl_pbl_physics=2], 3=NCEP-GFS, "
    "7=Pleim-Xiu, 91=old-MM5).",
    "sf_surface_physics": "Use sf_surface_physics=4 (Noah-MP), 2 (Noah classic), 1 (slab), "
    "3 (RUC, explicit ruc_static/ruc_land), 7 (Pleim-Xiu), or 8 (experimental SSiB; "
    "prepared three-layer input, revised-MM5/YSU, SW=1/LW=4).",
    "ra_lw_physics": "Use ra_lw_physics=4 (RRTMG), 1 (classic RRTM) or 3 (CAM); paired 5 is experimental, paired 7/99 are experimental CPU-only, 14/24 are compiled-out in this WRF build.",
    "ra_sw_physics": "Use ra_sw_physics=4 (RRTMG), 1 (Dudhia), 2 (GSFC/Chou-Suarez) or 3 (CAM); paired 5 is experimental, paired 7/99 are experimental CPU-only, 14/24 are compiled-out in this WRF build.",
    "diff_opt": "Use diff_opt=0/1/2 (1+km_opt=4 = 2-D Smagorinsky real-data "
    "default; 2+km_opt=3 = 3-D Smagorinsky).",
    "km_opt": "Use km_opt=4 with diff_opt=1 (2-D Smagorinsky) or km_opt=3 with diff_opt=2 "
    "(3-D Smagorinsky), or km_opt=2/5 with diff_opt=2 (experimental 3-D TKE/SMS-3DTKE).",
    "damp_opt": "Use damp_opt=0 (off) or 3 (upper-level w-Rayleigh).",
    "diff_6th_opt": "Use diff_6th_opt=0 (off) or 2 (monotonic 6th-order filter).",
    "rk_order": "Use rk_order=3 (WRF RK3).",
    "w_damping": "Use w_damping=0 or 1.",
    "sf_urban_physics": "Use sf_urban_physics=0/1/2/3 with Noah-MP for urban1/2/3; BEP/BEM2/3 require YSU1 or BouLac8, revised-MM5 SF1 and RRTMG4/4; MYJURB is absent.",
    "sf_lake_physics": "Set sf_lake_physics=0 for operational runs; the WRF lake model is G3 reference-only until the lake column/state/oracle port lands.",
}


# --------------------------------------------------------------------------- #
# Dynamics / numerics: implemented integer codes + the Smagorinsky note.      #
# These are gated by namelist_check.SUPPORTED_OPTIONS today; the catalog      #
# mirrors that so the public table is complete and consistent.                #
# --------------------------------------------------------------------------- #
_DYNAMICS_IMPLEMENTED: Mapping[str, frozenset[int]] = {
    "diff_opt": frozenset({0, 1, 2}),
    # Literal km2 and SMS km5 are CPU-oracle-qualified; forecast qualification pending.
    "km_opt": frozenset({0, 1, 2, 3, 4, 5}),
    "damp_opt": frozenset({0, 3}),
    "diff_6th_opt": frozenset({0, 2}),
    "rk_order": frozenset({3}),
    "w_damping": frozenset({0, 1}),
    # SLUCM, BEP/BEM and Lake are opt-in surface paths.
    # Per-scheme qualification is recorded once in _IMPLEMENTED_NOTES.
    "sf_urban_physics": frozenset({0, 1, 2, 3}),
    "sf_lake_physics": frozenset({0,1}),
}

# Dynamics options that ARE valid WRF codes but are fail-closed by the port,
# with an explicit reason + transition recipe.
# Per-(key, code) fail-closed reason override for RECOGNIZED_FAIL_CLOSED schemes.
# Despite the historical name, this is consulted for ANY namelist key in
# classify_scheme (not only dynamics): it supplies a specific reason in place of
# the generic "NOT YET IMPLEMENTED" string when the truth is more precise.
_PER_CODE_FAIL_CLOSED_REASON: Mapping[str, dict[int, str]] = {
    # mp=24 WSM7 was fail-closed before v0.17 (it carries a separate precipitating
    # hail class qh the operational moist-state pytree did not hold). v0.17 added
    # the qh hail State substrate (ADR-032) + the hail surface accumulator and
    # scan-wired WSM7 (coupling.scan_adapters.wsm7_adapter); it is now IMPLEMENTED
    # (see _IMPLEMENTED["mp_physics"]), so it no longer has a fail-closed reason.
    "mp_physics": {
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
        9: "REFERENCE-WITH-REAL-ORACLE / fail-closed: Milbrandt-Yau 2-moment "
        "has active pristine-WRF full-model oracle artifacts under "
        "proofs/v018/mp_oracles/wrf_full_model/mp9 for "
        "phys/module_mp_milbrandt2mom.F selected by mp_physics=9. The JAX "
        "endpoint is NOT YET IMPLEMENTED: it is a real multimoment bulk scheme "
        "and requires qh plus qnc/qnr/qni/qns/qng/qnh number-state transport "
        "before operational scan wiring.",
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
            "1 (slab), 3 (RUC), 7 (Pleim-Xiu) or 8 (experimental SSiB). CLM4 is "
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
        "Only the Fitch wind-farm parameterization (windfarm_opt=1) is ported; MAV=2 is not qualified.",
        "Set windfarm_opt=0 or 1 with MYNN=5/5 and turbine input files.",
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
        "grid_fdda", "FDDA analysis/spectral nudging",
        "Unknown grid_fdda selection; WRF supports 0 (off), 1 (PSU analysis) and 2 (spectral).",
        "Use grid_fdda=0, 1 or 2.",
    ),
    OutOfScopeFeature(
        "obs_nudge_opt", "FDDA observation nudging",
        "Observation nudging requires obs_nudge_opt=1; other codes are not WRF observation nudging.",
        "Use obs_nudge_opt=1 on a single root domain with MYNN PBL/SFCLAY and OBS_DOMAIN101, or set 0.",
    ),
    OutOfScopeFeature(
        "sf_ocean_physics", "Coupled ocean mixed-layer / 3-D ocean",
        "The PWP 3-D ocean (sf_ocean_physics=2) is not implemented.",
        "Use sf_ocean_physics=1 (1-D ocean mixed layer) or 0 (prescribed SST).",
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
        "spp_conv", "Stochastically Perturbed Convection Parameters (SPP)",
        "SPP convection perturbations are not operationally wired; spp_conv activates independently of spp.",
        "Set spp_conv=0 and spp=0.",
    ),
    OutOfScopeFeature(
        "spp_pbl", "Stochastically Perturbed PBL Parameters (SPP)",
        "SPP PBL perturbations are not operationally wired; spp_pbl activates independently of spp.",
        "Set spp_pbl=0 and spp=0.",
    ),
    OutOfScopeFeature(
        "spp_lsm", "Stochastically Perturbed Land-Surface Parameters (SPP)",
        "SPP land-surface perturbations are not operationally wired; spp_lsm activates independently of spp.",
        "Set spp_lsm=0 and spp=0.",
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
# permanent product boundary rather than a future-port target. Urban BEP/BEM
# have since graduated to the qualified opt-in path above.
_OUT_OF_SCOPE_CODES: Mapping[str, dict[int, OutOfScopeFeature]] = {}

#: Feature-switch values that ARE implemented although other values of the switch are
#: out of scope: grid_fdda=1 PSU grid analysis nudging (v0.4.0 lane o2-fdda,
#: assimilation.fdda_grid; &fdda details checked by io.fdda_namelist).
_IMPLEMENTED_FEATURE_VALUES: Mapping[str, frozenset[int]] = {"grid_fdda": frozenset({1, 2}),
    "obs_nudge_opt": frozenset({1}),
    **{key: frozenset({1}) for key in ("skebs", "sppt", "spp", "spp_conv",
        "spp_pbl", "spp_lsm", "rand_perturb", "stoch_force_opt")},
    "windfarm_opt": frozenset({1}),
    "sf_ocean_physics": frozenset({1}),
    "grid_sfdda": frozenset({1, 2}),
}

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
#     nested_pipeline.py:61); bldt unread -> PBL runs every step; KF cudt uses STEPCU,  #
#     so only the every-step value 0 is faithful.                             #
#   * Noah-MP soiltstep (&noah_mp, s): soil solved every land-surface call;    #
#     only soiltstep=0 is wired, a positive soil timestep fails closed.        #
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

_GECROS_REFUSAL_REASON = (
    "GECROS (opt_crop=2) is REFUSED: pristine WRF 4.7.1 never calls EMERG, "
    "so cold-start crops cannot emerge. Its unassigned module parameters "
    "(including CFO/YGO) remain zero; 8/8 active-column probes produced "
    "NaN/Inf yield in STATE_GECROS(45). No functional pristine WRF oracle "
    "exists. See proofs/s2_gecros/UPSTREAM_WRF_BUGS.md."
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
        "rad_diag", "gpuwrf-radiation-history", frozenset({0, 1}),
        "gpuwrf option (&gpuwrf): opt-in RRTMG4/4 radiation-call heating and input condensate history.",
        "Use rad_diag=0 (default) or rad_diag=1 with RRTMG4/4.",
    ),
    RecognizedControl(
        "icloud", "radiation-cloud-fraction",
        frozenset({0, 1, 2, 3}),
        "recognized; the port binds WRF icloud=1 (default, cal_cldfra1 + icloud_bl merge), "
        "icloud=2 (cal_cldfra2), icloud=3 (cal_cldfra3) and icloud=0 (cloud-free RRTMG radiation).",
        "Use icloud=1 (WRF default), 0, 2 or 3.",
    ),
    RecognizedControl(
        "icloud_bl", "PBL-cloud-coupling",
        frozenset({0, 1}),
        "WRF radiation_driver.F:1404-1431: icloud_bl=1 merges MYNN cloud fraction "
        "and condensate for icloud=1; icloud_bl=0 skips that radiation-input merge. "
        "The MYNN cloud producer is unchanged; icloud=0/2/3 bypasses this block.",
        "Use icloud_bl=0 or 1 (WRF default).",
    ),
    RecognizedControl(
        "bl_mynn_cloudpdf", "MYNN-cloud-PDF", frozenset({0, 1, 2}),
        "WRF CASE 0/1 with held TSQ/COV/SH3D, or default CASE 2. "
        "MYNN5 only; cycling, stochastic and budget-history combinations unqualified. GPU pending.",
        "Use bl_mynn_cloudpdf=0/1/2; one scalar shared by all domains.",
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
        "cycling", "WRF-cycling-lifecycle", frozenset({0, 1}),
        "WRF time_control cycling: fresh issue clock/buckets, start_em rebalance_cycl, "
        "and MYNN per-column initialization including its zero-QKE cycling quirk. "
        "Currently Thompson mp=8, MYNN bl=5, dfi_opt=0 only; coupled qualification pending.",
        "Omit cycling for the released cold-start behavior; cycled inputs require cycling=.true..",
    ),
    RecognizedControl(
        "use_input_w", "cycling-input-W", frozenset({0, 1}),
        "Cycling startup follows start_em:1475-1530: false rebuilds W, true retains "
        "nonzero input W; a zero surface field is initialized in either case. "
        "Explicit use_input_w is bound on the cycling path only.",
        "Use cycling=.true. with use_input_w=.false. (WRF default) or .true..",
    ),
    RecognizedControl(
        "bl_mynn_edmf", "MYNN-EDMF-massflux",
        frozenset({0,1}),
        "Native MYNN master mass-flux switch. Zero disables every plume source; "
        "nondefault mode requires bl_pbl_physics=5 and native fp32 source-leaf coupling. "
        "CPU-oracle-qualified; composed GPU qualification pending.",
        "Use bl_mynn_edmf=0/1 with MYNN5.",
    ),
    RecognizedControl(
        "bl_mynn_edmf_mom", "MYNN-EDMF-momentum-massflux",
        frozenset({0,1}),
        "recognized; the port wires the WRF-default EDMF momentum mass-flux ON "
        "(bl_mynn_edmf_mom=1). Zero is allowed only when the master bl_mynn_edmf=0 "
        "already disables all plume tendencies.",
        "Use bl_mynn_edmf_mom=1, or disable the master EDMF switch too.",
    ),
    RecognizedControl(
        "tke_budget", "WRF MYNN TKE budget history", frozenset({0,1}),
        "Opt-in separate MYNN history twin: qSHEAR/qBUOY/qDISS/qWT/Dtke. "
        "Requires MYNN5/native fp32. CPU-oracle-qualified; composed GPU qualification pending.",
        "Use tke_budget=0/1 with MYNN5.",
    ),
    RecognizedControl(
        "bl_mynn_output", "WRF MYNN EDMF history", frozenset({0,1}),
        "Opt-in separate MYNN history twin exports ten EDMF and six raw PBL rate fields. "
        "Requires MYNN5/native fp32. CPU-oracle-qualified; composed GPU qualification pending.",
        "Use bl_mynn_output=0/1 with MYNN5.",
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
        frozenset({1}),
        # v0.3.4 (o1-nlbind): no mixlength option reaches the MYNN port; it runs 1.
        "recognized; the port runs the WRF-default MYNN mixing length "
        "bl_mynn_mixlength=1 (nonlocal/BouLac blend, physics/mynn_constants.py); "
        "options 0 and 2 are not bound.",
        "Use bl_mynn_mixlength=1 (WRF default) or omit the key.",
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
        "Kain-Fritsch uses WRF STEPCU=max(1,NINT(cudt*60/dt)); "
        "calls on step 1 and STEPCU multiples, retaining tendencies between calls.",
        "Use a finite nonnegative cudt interval in minutes.",
        integer=False,
    ),
    RecognizedControl(
        "opt_crop", "Noah-MP crop model",
        frozenset({0,1}),
        lambda value: _GECROS_REFUSAL_REASON if _coerce_number(value) == 2 else
        "The requested Noah-MP crop model is not ported.",
        "Set opt_crop=0 (disabled) or 1 (Liu crop).",
    ),
    # --- Noah-MP soil cadence (&noah_mp, seconds) -------------------------- #
    RecognizedControl(
        "soiltstep", "noahmp-soil-cadence",
        frozenset({0}),
        "recognized; the port solves Noah-MP soil temperature and soil water at EVERY "
        "land-surface call (WRF soiltstep=0: soil_update_steps=1, calculate_soil every "
        "step, module_sf_noahmpdrv.F:648-675). A positive soil timestep in seconds "
        "(soil_update_steps=max(1,NINT(soiltstep/dt)); soil solved every "
        "soil_update_steps-th step from the ACC_* accumulated forcing) is not implemented.",
        "Set soiltstep=0 (the WRF default; soil processes every Noah-MP step).",
        integer=False,
    ),
)

from gpuwrf.io.noahmp_options import SUPPORTED as _NOAHMP_CONTROL_VALUES
_RECOGNIZED_CONTROLS += tuple(
    RecognizedControl(key,f'Noah-MP {key}',frozenset(values),
        (lambda value,key=key: f'Noah-MP {key}: only the WRF-defined, ported choices {_NOAHMP_CONTROL_VALUES[key]} are admitted; see docs/noahmp-options.md.'),
        f'Use a supported {key} value with native sf_surface_physics=4; see docs/noahmp-options.md.')
    for key,values in _NOAHMP_CONTROL_VALUES.items() if key!='opt_crop'
) + (RecognizedControl('sst_skin','WRF SST skin',frozenset({0,1}),
       'Only WRF sst_skin=0/1 is defined and ported; the selected native path requires Noah-MP/RRTMG and ice-free prescribed SST.',
       'Use sst_skin=0 or the admitted sst_skin=1 configuration.'),)

RECOGNIZED_CONTROL_KEYS: frozenset[str] = frozenset(
    c.key.lower() for c in _RECOGNIZED_CONTROLS
)
# Cadence controls whose unwired (positive) values are a non-raising,
# conservative approximation (run-every-step) rather than a fail-closed
# rejection: bldt. A naive user pointing the standalone CLI at a real
# WRF namelist (cudt=5, bldt=0) must RUN, not be rejected.
APPROXIMATED_CONTROL_KEYS: frozenset[str] = frozenset(
    c.key.lower() for c in _RECOGNIZED_CONTROLS if c.approximated
)
_RECOGNIZED_CONTROL_BY_KEY: Mapping[str, RecognizedControl] = {
    c.key.lower(): c for c in _RECOGNIZED_CONTROLS
}

# Implemented non-enumerated controls (radiation slope/topo path). slope_rad=1
# and topo_shading=1 ARE wired (RRTMG SW slope-radiation + topographic-shadow);
# WRF uses independent 0/1 switches; topo_shading affects surface flux only
# when slope_rad=1. There is no WRF slope_rad=2 combined mode.
_IMPLEMENTED_CONTROLS: Mapping[str, frozenset[int]] = {
    "swint_opt": frozenset({0, 1}),
    "dfi_opt": frozenset({0, 1, 2, 3}),
    "sst_update": frozenset({0, 1}),
    "slope_rad": frozenset({0, 1}),
    "topo_shading": frozenset({0, 1}),
    "tracer_opt": frozenset({0, 2}),
    "mfshconv": frozenset({0,1}),
    "topo_wind": frozenset({0, 1, 2}),
    "aer_opt": frozenset({0, 1, 3}),
}
_IMPLEMENTED_CONTROL_REASON: Mapping[str, tuple[str, str, str]] = {
    "swint_opt": (
        "WRF-Solar surface interpolation",
        "Only native swint_opt=1 with RRTMG4/4, Noah-MP4 and fixed dt is wired; FARMS swint_opt=2 is not ported.",
        "Use swint_opt=0 or the qualified swint_opt=1 configuration.",
    ),
    "dfi_opt": (
        "digital filter initialization (DFI)",
        "Single-domain native DFL/DDFI/TDFI; configuration-specific refusals are enforced by dfi_refusals.",
        "Use dfi_opt=0 or an admitted single-domain DFI configuration.",
    ),
    "sst_update": (
        "Time-varying lower-boundary SST update",
        "The standalone nested path wires sst_update=1 from NetCDF auxinput4 "
        "for ice-free prescribed SST; values other than 0/1 are not wired.",
        "Use sst_update=0 or 1 with io_form_auxinput4=2 and a positive interval.",
    ),
    # key: (label, unwired-reason, alternative)
    "slope_rad": (
        "slope-radiation",
        "recognized; WRF and the port use slope_rad=0 or 1. "
        "Enable topo_shading=1 separately for terrain shadows with slope_rad=1.",
        "Use slope_rad=0 (off) or 1 (slope radiation on).",
    ),
    "topo_shading": (
        "topographic-shading",
        "recognized; the port wires topo_shading=1 (RRTMG SW topographic "
        "shadowing). Other topo_shading values are not wired.",
        "Use topo_shading=0 (off) or 1 (topographic shadowing on).",
    ),
    "tracer_opt": (
        "non-Chem passive tracers",
        "CPU-oracle-qualified column mixing; resolved RK scalar transport, root/nest boundaries, "
        "history and full restart wired opt-in. GPU/coupled-forecast qualification pending. "
        "WRF-Chem tracer modes 1/3/4 are not implemented.",
        "Use tracer_opt=0/2; custom tracers use gpuwrf_tracer_file.",
    ),
    "topo_wind": (
        "topographic surface-wind correction (CTOPO/CTOPO2)",
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. start_em CTOPO/CTOPO2 byte-exact "
        "vs pristine WRF 4.7.1; they reach only YSU (bl=1) and Shin-Hong (bl=11) like WRF, so other PBLs "
        "run unchanged (warning). topo_wind=1 refuses wrfinput without VAR_SSO (GOT_VAR_SSO=0), "
        "topo_wind=2 needs the geogrid VAR field.",
        "Use topo_wind=0, or 1 with WPS varsso data, or 2, with bl_pbl_physics=1 or 11.",
    ),
    "mfshconv": ('QNSE shallow mass flux','ARW MFSHCONVPBL precedes QNSEPBL for mfshconv=1; WRF resets it to 0 outside bl=4.',
                 'Use mfshconv=0 or 1 with the mandatory bl=4/sf=4 pair.'),
    "aer_opt": (
        "radiation aerosol option",
        "Only aer_opt=1 (Tegen climatology) and aer_opt=3 (Thompson mp28 water/ice-friendly aerosols) in RRTMG SW "
        "(ra_sw_physics=4) are wired; aer_opt=2 (aer_type/AOD550 inputs) is not ported.",
        "Use aer_opt=0, aer_opt=1 with ra_sw_physics=4, or aer_opt=3 with mp_physics=28 and ra_sw_physics=4.",
    ),
}
IMPLEMENTED_CONTROL_KEYS: frozenset[str] = frozenset(_IMPLEMENTED_CONTROLS)

# CLI research operands, separate from WRF's namelist scheme-code namespace.
# These entries describe admission/qualification; they do not enable a scheme.
RUNTIME_PARAMETER_FAMILIES = {
    "thompson_parameters": {
        "option": "--runtime-parameters",
        "parameters": ("nt_c", "tno", "ato", "av_c", "av_i", "av_r", "autoconversion_threshold", "autoconversion_rate"),
        "runtime_parameters": ("nt_c", "tno", "ato", "av_c"),
        "experimental_refused_parameters": ("av_i", "av_r", "autoconversion_threshold", "autoconversion_rate"),
        "qualification": "Research candidate, native mp_physics=8. WRF PARAMETER/source-literal extensions require CPU-WRF recompilation. Public binder accepts Nt_c/TNO/ATO/AV_C; av_i/av_r/autoconversion are refused until their GPU/E41/cost gates pass. Isolated CPU default classes: TNO/ATO/autoconversion BYTE; AV_C qc/cloudw <=2 REAL ulp; Nt_c Nr<=3; AV_I qi<=3/Ni<=4/ice-precip<=1 REAL ulp; AV_R qr<=3/Nr<=6. Widened REAL precipitation carriers are not float64-ulp claims. All eight at defaults together: union of these classes (max Nr 6 REAL ulp), within the 25-member 1-REAL-ulp pristine-WRF floor. Product-root GPU gates remain open.",
    },
    "surface_controls": {
        "option": "--surface-controls",
        "parameters": ("z0bare", "z0mvt", "eg", "vcmx25", "mp", "albsat", "albdry", "rhol", "rhos",
                       "laim", "saim", "shdmax", "ctopo",
                       "smcmax", "dksat", "bexp", "psisat", "rsmin", "hvt", "hvb", "z0sno", "snow_emis"),
        "qualification": "Existing Track-P runtime path; z0bare is EXPERIMENTAL, a spatial extension of the WRF ENERGY ground-tile constant, not WRF-faithful (real GPU qualification pending). Authenticated CTOPO checkpoint integration is a research candidate. W2 soil/canopy/snow controls are research candidates; native E41/GPU cost remains pending.",
    },
    "diffusion_parameters": {
        "option": "--runtime-parameters",
        "parameters": ("c_s", "diff_6th_factor", "dampcoef", "zdamp", "w_alpha", "w_beta", "w_crit_cfl"),
        "qualification": "Research candidate: CPU WRF operator checks; GPU/E41/cost gate pending. c_s only diff_opt=1/km_opt=4.",
    },
    "mynn_parameters": {
        "option": "--runtime-parameters",
        "parameters": ("sqfac", "alp1", "cphm_unst", "am_st", "entrainment"),
        "qualification": "Research candidate: MYNN bl_pbl_physics=5; added length/stability/entrainment operands require native FP32 columns. Expanded CPU/GPU/E41/cost gates pending.",
    },
}


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
                reason=_IMPLEMENTED_NOTES.get((lkey, code), "Operationally wired into the GPU scan."),
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

    if lkey == "cudt":
        import math
        numeric = _coerce_number(value)
        wired = numeric is not None and math.isfinite(numeric) and numeric >= 0
        return SchemeSupport(
            key=lkey, code=_coerce_int_or_bool(value),
            status=SupportStatus.IMPLEMENTED if wired else SupportStatus.RECOGNIZED_FAIL_CLOSED,
            reason=control.reason_for(value), alternative=control.alternative,
            wrf_name=control.label,
        )

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
            reason=_IMPLEMENTED_NOTES.get((lkey, code), "Operationally wired into the GPU scan."),
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


# Honest per-scheme qualification notes for IMPLEMENTED options (default reason otherwise).
_IMPLEMENTED_NOTES: Mapping[tuple[str, int], str] = {
    ("icloud", 3): "GPU-qualified (v0.4.1: WN3 0227 3-nest 24 h D6 PASS 3/3 vs original CPU-WRF icloud=3, E41 PASS). WRF 4.7.1 "
    "radiation_driver icloud=3 (G. Thompson cal_cldfra3 + find_cloudLayers/adjust_*): scale-aware RH "
    "cloud fraction with radiation-only fake condensate; replaces the icloud=1 block incl. the icloud_bl "
    "MYNN merge exactly as WRF (MYNN keeps its own SGS chain). History CLDFRA = radiation input fraction. "
    "Literal REAL4 BYTE vs verbatim pristine on 640 real columns; compiled path within WRF FMA/input-ULP "
    "envelope, XLA log10 small-condensate branch <=15 ulp disclosed. RRTMG 4/4 only; mp 0/1/3/5/11/P3 "
    "refused. proofs/w41_icloud.",
    ("icloud", 0): "GPU-qualified (v0.4.2: WN3 0227 3-nest 24 h D6 PASS 3/3 vs original CPU-WRF icloud=0, E41 PASS). WRF 4.7.1 radiation_driver "
    "icloud=0: CLDFRA = 0 and no cloud-fraction branch (no icloud_bl merge); RRTMG LW/SW skip every condensate and "
    "effective-radius input (IF ICLOUD .ne. 0), so radiation is cloud-free while microphysics, MYNN SGS clouds and "
    "dynamics run unchanged (cloud-radiation decoupling switch). History CLDFRA = 0 like WRF. Port within the frozen "
    "RRTMG bounds of the pristine radiation_driver on 16 real WN3 columns; with the v0.4.1 SWDDIR fix (inert under "
    "icloud=0). RRTMG 4/4 only. proofs/w42_icloud0.",
    ("icloud", 2): "GPU-qualified (v0.4.1: WN3 0227 3-nest 24 h D6 PASS 3/3 vs original CPU-WRF icloud=2, E41 PASS). WRF 4.7.1 "
    "radiation_driver icloud=2 (cal_cldfra2: binary qc+qi > 1e-6), replaces the icloud=1 block incl. the icloud_bl "
    "MYNN merge exactly as WRF. History CLDFRA = radiation input fraction (binary). Literal BYTE vs pristine "
    "cal_cldfra2. RRTMG 4/4 only. proofs/w41_icloud, proofs/w41_icloud2.",
    ("mp_physics",11): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Experimental original nonchem CAM-MG/macrophysics/modal activation, REAL4 seam and held carry with WRF DOUBLE internal island. All48 original public calls literal BYTE; native CPU artifact within independently frozen WRF FMA/input-ULP bounds. No mandatory mp11/bl9/ra3 pairing in WRF; cu!=7 and shcu!=2 warn and zero their convection operands. Native CUDA column compiled by stock Triton 3.6 (pip triton==3.6.*, required for GPU mp11) and launched via jaxlib triton_kernel_call (XLA's bundled Triton miscompiles it to zero precipitation); GPU admitted again (see KNOWN_ISSUES.md#cam-mg-gpu-precipitation); GPU frozen pristine controls and an active 44-level column match the CPU literal; native E41 STACK 1240 B/255 regs disclosed; coupled D6 pending. Standalone shallow convection remains unported.",
    **{('cu_physics',option):'Experimental CPU-oracle-qualified MSKF/KF-CuP/CAM-ZM family; GPU/coupled-forecast/E41 qualification pending. '
       'Dedicated pristine WRF REAL/DOUBLE bodies, actual surface/PBL producers, resident cadence and full restart carry, '
       'WRF RK momentum/moisture/number sources and original convective radiation views. '
       'Thompson8/RRTMG4/4 only; CuP/MSKF use MYNN5/5, CAM-ZM uses CAM-UW9/5 or MYJ2/2 (MYJ/Noah-MP refused). '
       'Exactly calm MSKF singularity and small compiled deviations retained/disclosed; D6 decides. '
       'See docs/CUMULUS_CUPLUS.md and proofs/v040/cuplus/UPSTREAM_WRF_BUGS.md.' for option in (7,10,11)},
    ('moist_adv_opt', 0): "Standard WRF h5/v3 moisture transport (moist_adv_opt=0), single-domain release configuration. "
    "GPU-qualified vs CPU-WRF Swiss 24 h; E41 robustness flag: none (full scan PASS). D6 every hourly "
    "frame (25), all CPU fields inventoried; proofs/v040/gpuqual/moist0.",
    **{(key, 1): "WRF stochastic RK forcing and resident GNU14 REAL4 RNG/patterns: "
       "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
       "SPPT requires raw RK sources (PBL0/5, CU0/1; GWDO with MYNN); SPP-GSL GWDO3 refused. "
       "MYNN SPP includes surface, diffusion, cloud-PDF and capped entrainment terms. "
       "Original WRF convection/LSM SPP consumers are physical no-ops; rand_perturb is diagnostic. "
       "Parent SINT fields, exact full-carry restart, pre-cap counters; docs/stochastic.md. "
       "Native Triton IR preflight passed; actual GPU timing/memory/E41 not yet measured."
       for key in ("skebs", "sppt", "spp", "spp_conv", "spp_pbl", "spp_lsm",
                   "rand_perturb", "stoch_force_opt")},
    ('ra_lw_physics',5): 'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Experimental paired Goddard5/5 with MP8/use_mp_re=1; held original REAL4 tendencies/fluxes, explicit WRF DOUBLE solver. Upstream cloudy-SW uninitialized taux is mirrored as zero per documented compiler-defined reference; sensitivity is material (docs/UPSTREAM_WRF_BUGS.md). Exact-equator climatology, mixed radiation, other MP mediation and aerosol/topography feedback are refused. Surface support: Noah-MP/SLUCM, RUC and no-LSM control; slab, Noah-classic, Pleim-Xiu, lake and BEP/BEM pairings are refused.',
    ('ra_sw_physics',5): 'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Experimental paired Goddard5/5 with MP8/use_mp_re=1; held original REAL4 tendencies/fluxes, explicit WRF DOUBLE solver. Upstream cloudy-SW uninitialized taux is mirrored as zero per documented compiler-defined reference; sensitivity is material (docs/UPSTREAM_WRF_BUGS.md). Exact-equator climatology, mixed radiation, other MP mediation and aerosol/topography feedback are refused. Surface support: Noah-MP/SLUCM, RUC and no-LSM control; slab, Noah-classic, Pleim-Xiu, lake and BEP/BEM pairings are refused.',
    **{("mp_physics", code): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
       "Experimental P3 REAL column port: literal no-FMA BYTE against pristine WRF; compiled "
       "within independently measured WRF FMA spread and source-derived near-zero radius bound "
       "(proofs/s2_p3/COMPILED_BOUND.md). Conditional rime/volume, second ice or sqrt(N*Z) carry, "
       "root scalar/flow_dep_bdy, held cloud/ice radii with P3 RRTMG snow-optics mediation, "
       "Held WRF diameter/density/fall-speed history per category and lossless restart. Single domains only. Native resource/E41 flags "
       "and D6 forecast qualification pending."
       for code in (50,51,52,53)},
    ("mp_physics", 56): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
       "Experimental NTU multi-moment: literal REAL4/REAL8 pristine-WRF column, scalar, "
       "activation and cold aerosol initialization gates BYTE. All 28 prognostics, CCNTY1..4, "
       "root moist/scalar boundary transport, original REAL accumulated totals, history and restart. "
       "Single domains only; original DOUBLE gamma/aerosol lookup/Newton islands retained. "
       "Upstream compiler-zero initialization/scratch sensitivity disclosed in proofs/s2_ntu/UPSTREAM_WRF_BUGS.md. "
       "313 small compiled excesses beyond fixed original input-ULP/FMA controls are disclosed in proofs/s2_ntu/COMPILED_DEVIATIONS.json; D6 decides. "
       "Native column kernel compiles on GPU with typed global memory (whole Swiss step program 5 min 11 s; "
       "proofs/t_ntucompile). Actual GPU E41: FAIL, ntu_mp56_column STACK 3496 B per thread vs the 256 B default cap "
       "(opt-in resource disclosure; default hard gate unchanged; measured at 4 warps, the kernel now runs 2 warps with "
       "bitwise-identical outputs). Measured kernel-only runtime on two real cloudy 1764-column Swiss states: "
       "0.286-0.389 s per call (2 warps + aerosol-inversion NaN shortcut, bitwise-identical outputs; pre-shortcut 2-warp "
       "baseline 1.15-1.70 s; original DOUBLE islands on 1/64-rate fp64). D6 forecast qualification pending.",
    ("mp_physics", 55): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
       "Experimental Jensen-ISHMAEL: literal REAL4/REAL8 original public columns and "
       "evolved two-step habits BYTE. Three ice masses/numbers and six volume/aspect "
       "moments, conditional State, root transport, Registry history, lossless restart, "
       "held cloud/ice radii and WRF snow-optics mediation. Single domains only. "
       "Original gammln DOUBLE island retained. Small compiled rounding excesses are "
       "disclosed experimental deviations; D6 decides (docs/JENSEN_ISHMAEL.md). "
       "Native column kernel compiles and runs on GPU with typed global memory and interpretation-identical gather "
       "bounds (whole Swiss step program ~5-6 min; proofs/t_ntucompile). Actual GPU E41: FAIL, jensen_mp55_column "
       "STACK 5472 B per thread vs the 256 B default cap (opt-in resource disclosure; default hard gate unchanged). "
       "WRF fall speeds, diameters, densities and aspect ratios are held per habit with E78 restart "
       "validation (docs/physics/ice_diagnostics.md). Ice type and radar diagnostics remain omitted.",


    ('bl_pbl_physics', 1): ("YSU. Single domain: v0.4.0 composite c02 24 h D6 PASS vs original CPU-WRF (E41 FAIL disclosed). "
        "Nested-qualified (v0.4.2): Tenerife WN3 9/3/1 km 3-nest, issue 20260707_18z (ALISIOS G2 arm C2, gpuwrf v0.4.1), with sf_sfclay_physics=1: 24 h D6 PASS on d01-d03 vs an original CPU-WRF 4.7.1 twin with the same namelist and inputs (25 hourly frames per domain, 0 of 62 bounded fields over limit, NF0, every CPU variable present; worst-hour RMSE/limit 0.024, T2 d03). One case and date; E41 of the nested programs not measured. Receipt proofs/w41_nestqual/NESTQUAL_SUMMARY.json."),
    ('bl_pbl_physics', 7): ("ACM2 (Pleim). Single domain: v0.4.0 composite c05 24 h D6 PASS vs original CPU-WRF (E41 FAIL disclosed). "
        "Nested-qualified (v0.4.2): Tenerife WN3 9/3/1 km 3-nest, issue 20260707_18z (ALISIOS G2 arm C4, gpuwrf v0.4.1), with sf_sfclay_physics=1: 24 h D6 PASS on d01-d03 vs an original CPU-WRF 4.7.1 twin with the same namelist and inputs (25 hourly frames per domain, 0 of 62 bounded fields over limit, NF0, every CPU variable present; worst-hour RMSE/limit 0.069, V10 d03). One case and date; E41 of the nested programs not measured. Receipt proofs/w41_nestqual/NESTQUAL_SUMMARY.json."),
    ('bl_pbl_physics', 11): ("Shin-Hong scale-aware. Single domain: v0.4.0 composite c07 24 h D6 PASS vs original CPU-WRF (E41 FAIL disclosed). "
        "Nested-qualified (v0.4.2): Tenerife WN3 9/3/1 km 3-nest, issue 20260707_18z (ALISIOS G2 arm C3, gpuwrf v0.4.1), with sf_sfclay_physics=1: 24 h D6 PASS on d01-d03 vs an original CPU-WRF 4.7.1 twin with the same namelist and inputs (25 hourly frames per domain, 0 of 62 bounded fields over limit, NF0, every CPU variable present; worst-hour RMSE/limit 0.059, V10 d03). One case and date; E41 of the nested programs not measured. Receipt proofs/w41_nestqual/NESTQUAL_SUMMARY.json."),
    ('bl_pbl_physics', 16): 'EXPERIMENTAL: CPU-oracle-qualified against unchanged WRF 4.7.1 REAL32 with compiler-defined zero initialization and FP traps off (30 carried cases per scheme). Single domain: v0.4.0 composite c10 24 h D6 PASS vs original CPU-WRF (E41 FAIL disclosed). Nested-qualified (v0.4.2): Tenerife WN3 9/3/1 km 3-nest, issue 20260707_18z (ALISIOS G2 arm C7, gpuwrf v0.4.1), with sf_sfclay_physics=1: 24 h D6 PASS on d01-d03 vs an original CPU-WRF 4.7.1 twin with the same namelist and inputs (25 hourly frames per domain, 0 of 62 bounded fields over limit, NF0, every CPU variable present; worst-hour RMSE/limit 0.110, V10 d03). One case and date; E41 of the nested programs not measured. Receipt proofs/w41_nestqual/NESTQUAL_SUMMARY.json. Upstream undefined reads (KEPS ttke GET_PBLH:822, radsum:1068, top dtmdz; EEPS radsum:748) materially affect outputs under alternate initialization. This port explicitly mirrors zero semantics. Requires sf_sfclay_physics=1; transports Registry epsilon/TKE scalars and requires versioned restart carry. See docs/EEPS_KEPS.md and proofs/s2_keps/initialization_sensitivity.json.',
    ('bl_pbl_physics', 17): 'EXPERIMENTAL: CPU-oracle-qualified against unchanged WRF 4.7.1 REAL32 with compiler-defined zero initialization and FP traps off (30 carried cases per scheme); GPU/coupled-forecast qualification pending. Upstream undefined reads (KEPS ttke GET_PBLH:822, radsum:1068, top dtmdz; EEPS radsum:748) materially affect outputs under alternate initialization. This port explicitly mirrors zero semantics. Requires sf_sfclay_physics=1; transports Registry epsilon/TKE scalars and requires versioned restart carry. See docs/EEPS_KEPS.md and proofs/s2_keps/initialization_sensitivity.json.',
    ("aer_opt", 1): "WRF Tegen/ECMWF six-type monthly aerosol climatology in RRTMG SW (aerosol_in on domain 1, "
        "aer_time_int/aer_p_int at each radiation call, iaer=6 band optics; nests get the parent's aerod by SINT "
        "force-down like WRF Registry df=(p2c)). CPU-oracle-qualified; GPU 24 h WN3 0227 3-nest forecast matches "
        "original CPU-WRF (frozen D6 PASS on d01-d03, every field and frame). "
        "Requires ra_sw_physics=4; LW has no aerosol (as WRF without Chem). The aer_opt=1 SW path also emits WRF's "
        "unscaled (ztdbt_nodel) direct beam. See docs/AEROSOL.md and proofs/w41_aerosol/.",
    ("aer_opt", 3): "WRF Thompson water/ice-friendly aerosol optics in RRTMG SW (gt_aod on QNWFA/QNIFA + "
        "calc_aerosol_rrtmg_sw rural-type RH spectral tables at each radiation call, iaer=10; held TAOD5502D). "
        "CPU-oracle-qualified; GPU 24 h Swiss mp28 forecast matches original CPU-WRF (frozen D6 PASS, every field "
        "and frame). E41 on that run (A3c): FAIL on the conservative kInput rule, three optimized-HLO kInput "
        "transpose fusions of 658/879/1370 instructions, all from pre-existing Thompson mp28 source families (not "
        "byte-identical kernels; the v0.4.0 mp28 aer_opt=0 run had two, 741/1439); STACK max 16 B, no non-leaf "
        "callee >1k SASS; the aerosol code adds no flagged fusion. Requires mp_physics=28 (single domains, "
        "wif_input_opt=1) and ra_sw_physics=4. See docs/AEROSOL.md and proofs/w41_aerosol/.",
    ("swint_opt", 1): "WRF-Solar REAL surface interpolation, refreshed parameters only at radiation calls, "
        "surface fluxes each own step, held atmospheric heating/COSZEN, twelve REAL restart leaves. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending.",
    ('bl_pbl_physics',10): 'TEMF pair (bl=10/sfclay=10): CPU-oracle-qualified against unmodified WRF REAL module_bl_temf.F; GPU/coupled-forecast qualification pending. Stable and active shallow-cloud columns covered; no LSM or Noah-MP only. CPU parity requires AVX/no-FMA and disabled algsimp. GPU performance and column-kernel qualification pending.',
    ('sf_sfclay_physics',10): 'TEMF pair (bl=10/sfclay=10): CPU-oracle-qualified against unmodified WRF REAL module_sf_temfsfclay.F; GPU/coupled-forecast qualification pending. Requires TEMF PBL and no LSM or Noah-MP; WRF zero-mavail exchange divisions remain undefined.',
    ("grid_fdda", 2): "Spectral nudging through resident wrffdda analyses and WRF RK tendency lanes: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "REAL32 pristine spectral_nudging + FFTPACK5 references in proofs/s1_specnudge; "
    "wind, theta, perturbation geopotential and water vapor, linear vertical/PBL ramps and KTROP lid.",
    ("dfi_opt", 1): "Native single-domain DFL prepass (WRF forward filtering and midpoint launch). "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Configuration-specific admission, Noah-MP land handling and limitations: gpuwrf.dfi.control/driver12.",
    ("dfi_opt", 2): "Native single-domain DDFI prepass (adiabatic backward endpoint, diabatic forward filtering). "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Configuration-specific admission, Noah-MP land handling and limitations: gpuwrf.dfi.control/driver12.",
    ("dfi_opt", 3): "Native single-domain TDFI prepass (backward and forward filtering). CPU-oracle-qualified; Existing "
    "o2-dfi qualification and configuration-specific admission apply. GPU-qualified vs CPU-WRF Swiss 24 h; "
    "E41 robustness flag: none (full scan PASS). D6 every hourly frame (25), all CPU fields inventoried; "
    "proofs/v040/gpuqual/dfi3.",
    ("mp_physics", 17): "WRF 4.7.1 legacy alias of mp=18 with predicted CCN and hail: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Pristine check_a_mundo "
    "normalization and exact default-18 program/State/restart reuse: proofs/s2_mpmore. "
    "Same single-domain, scalar-transport, radiation and modifier limitations as mp=18; "
    "Custom &nssl_mp_params are refused; this is the WRF 4.7.1 alias, not the older CCN-off variant.",
    ("mp_physics", 22): "WRF legacy NSSL two-moment without hail, predicted CCN and graupel density: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Pristine REAL4/REAL8 active columns and actual one-ULP CCN threshold controls: proofs/s2_mpmore/nssl22. "
    "Six moist species, Nc/Nr/Ni/Ns/Ng/Nn and graupel volume; no hail mass/number/volume carry. "
    "Single domains; same scalar-advection and diffusion limits as18. Custom &nssl_mp_params refused.",
    ("mp_physics", 18): "Operationally wired into the GPU scan (NSSL two-moment, single domains): "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Pristine-WRF fp32/fp64 column savepoint-parity "
    "and cumulus number-source fixtures: proofs/v034/f2_oracles/nssl_2mom and proofs/v040/nssl_cumulus. "
    "Supported WRF cumulus schemes admitted, including KF held rain/snow sources after phy_prep_part2; "
    "WRF calcnfromcuten cloud/ice-only caller is a no-op. Root extra-species boundaries qualified; "
    "nested NSSL remains refused: WN3 N2/N6 numerical regions pass, but pre-mask qi/Ni events "
    "await WRF operator justification (proofs/s1_mpfix/NESTED_ADMISSION.json). WRF use_mp_re radii are CPU-oracle-qualified "
    "through held re_cloud/re_ice/re_snow and RRTMG4/4 (proofs/s1_mpfix).",
    **{("mp_physics", code): "Operationally wired into the GPU scan (single domains): WRF scalar "
       "transport of " + ("Ni/Nr/Ns/Ng" if code == 10 else "Nc/Nr/Nn") +
       "; root flow_dep_bdy, with QNN ccn_conc inflow for WDM; number fields read from wrfinput and "
       "retained by restart. Column kernel remains a documented fp64 island on the REAL32 carry "
       "(WDM7 hail qh also transported). CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
       "Nested runs refused until parent->child forcing is qualified."
       for code in (10, 14, 16, 26)},
    ('sf_lake_physics',1): 'WRF Lake/lakeini/LakeMain (phys/module_sf_lake.F): full snow/water/soil timestep, '
    'compact lake tiles bypass Noah-MP, held surface/radiation diagnostics and guarded restart/history. '
    'Requires Noah-MP, Thompson and MYNN/MYNN-SL; both CLI pipelines bind lake depth/elevation controls. '
    'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. '
    'Pristine 55-column initialization and 33 active trajectories (including 120-step freezing); docs/lake-model.md.',
    ("obs_nudge_opt", 1): "Root-domain station observation nudging: resident OBS_DOMAIN101 read/QC/map projection, "
    "multi-level soundings, time/radius/vertical weights, all WRF wind/temp/moisture flags and coefficients, "
    "ionf holds, dynamic-initialization ramp and negative-QV innovation scaling. WRF pressure nudging was removed; "
    "obs_nudge_pstr/coef_pstr and obs_rinsig remain inert as in pristine WRF. MYNN PBL=5/SFCLAY=5 currently required; "
    "new VAROBS/RKO/SAVWT carry is restarted. CPU-oracle-qualified; GPU/coupled-forecast qualification pending.",
    ("km_opt", 5): "Operational SMS-3DTKE (diff_opt=2, bl_pbl_physics=0): nonlocal flux, free-atmosphere "
    "parcel lengths, mesoscale coefficients, implicit momentum/heat/TKE/moisture solves and horizontal "
    "diffusion feed RK1-frozen sources and all three prognostic TKE RK stages. Separate REAL TKE, "
    "BR/RMOL/WSPD/GAMU/GAMV/USTM carry; restart retains every SMS field. "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Surface layer 0 or 5 and land model 0 or 4 only (surface layer 0 requires land 0); "
    "MM5 1/91 and other land SMS callers remain refused. "
    "WRF forbids PBL!=0, other surface-layer codes and cu_physics=11. "
    "Extra microphysics scalar families 10/14/16/18/24/26/28/40 remain refused, as for km_opt=2. "
    "Single domains only: live-child TKE/nonlocal carry needs the two-domain CPU-WRF gate. "
    "Non-default surface-flux/TKE transport options remain subject to the honour check.",
    ("km_opt", 2): "Operationally wired into the GPU scan: literal WRF 3-D TKE (diff_opt=2), tke_km/tke_rhs plus "
    "RK1-frozen diffusion, prognostic REAL State.tke distinct from PBL qke, three-stage TKE_advance with "
    "acoustic-averaged transport, PD update, bound_tke and flow_dep_bdy. Cold start seeds zero; restarts "
    "retain TKE; optional TKE history leaves PBL TKE_PBL/QKE separate. CPU-oracle-qualified against "
    "pristine WRF REAL4 diffusion and all three scalar RK stages. Swiss d01 release REAL32 CPU smoke "
    "(Pallas interpret + PTX shim): finite, zero guards. "
    "isfflx/tke_drag_coefficient/tke_heat_flux/tke_adv_opt/tke_mix6_off use WRF defaults 1/0/0/1/false; "
    "non-default CLI values remain refused by the honour check. CLI microphysics with unwired extra scalar "
    "diffusion (10/14/16/18/24/26/28/40) is refused. GPU-qualified vs CPU-WRF Swiss 24 h; E41 robustness "
    "flag: input_multiply_transpose_fusion.5: 1287 HLO > 500; input_transpose_fusion.197: 891 HLO > 500. "
    "D6 every hourly frame (25), all CPU fields inventoried; proofs/v040/gpuqual/km2.",
    ("mp_physics", 24): "Operationally wired into the GPU scan (WSM7, single domains): savepoint-parity "
    "column kernel run as a documented fp64 island under the release REAL32 carry "
    "(coupling.scan_adapters.FP64_ISLAND_MP_CODES); root lateral boundary for the hail species qh = "
    "WRF flow_dep_bdy (have_bcs_moist=.false., solve_em.F:2426), proven against the unchanged pristine "
    "routines on real PROD d01 operands (tests/v040/rootbdy). Parent->child forcing is implemented; "
    "nested runs remain refused despite WN3 N2/N6 numerical PASS: pre-mask qi/Ni events await "
    "WRF operator justification (proofs/s1_mpfix/NESTED_ADMISSION.json). WRF use_mp_re radii are CPU-oracle-qualified through held diagnostics "
    "and RRTMG4/4 (proofs/s1_mpfix). Current-candidate Swiss GPU E41: STACK 3864 B exceeds 256 B "
    "(experimental opt-in disclosure). CPU-qualified (Swiss d01 coupled smoke); "
    "GPU/coupled-forecast qualification pending.",
    ("mp_physics", 28): "Operationally wired into the GPU scan (aerosol-aware Thompson, single domains, "
    "use_aero_icbc=.false.): root lateral boundary for QNWFA/QNIFA = WRF flow_dep_bdy (aer_init_opt=0, "
    "solve_em.F:2993-3021) and for QNCLOUD = flow_dep_bdy (have_bcs_scalar=.false.); the droplet number Nc is "
    "transported like every WRF scalar (v0.4.0). Boundary route proven against the unchanged pristine "
    "routines on real PROD d01 operands (tests/v040/rootbdy). Parent->child forcing is implemented; "
    "nested runs remain refused: initial QNWFA/QNIFA fail the real WN3 original-WRF comparison "
    "in both domains (proofs/s1_mpfix/NESTED_ADMISSION.json). Variable-Nc effective radii are CPU-oracle-qualified through "
    "held diagnostics and RRTMG4/4 (proofs/s1_mpfix). Current-candidate Swiss GPU E41: STACK 928 B "
    "and kInput transpose 1510 optimized HLO instructions exceed 256 B/500 "
    "(experimental opt-in disclosure). CPU-qualified (Swiss d01 coupled smoke); GPU/coupled-forecast qualification "
    "pending.",
    ("mp_physics", 40): "Operationally wired into the GPU scan (Morrison-aerosol at aercu_opt=0, single "
    "domains): CPU-oracle-qualified column kernel (proofs/v034/f2_oracles/morrison_aero_opt0); root lateral "
    "boundary for QNSNOW/QNGRAUPEL/QNCLOUD = WRF flow_dep_bdy (have_bcs_scalar=.false., solve_em.F:2993-3021), "
    "proven against the unchanged pristine routines on real PROD d01 operands (tests/v040/rootbdy). "
    "aercu_opt>0 needs the unported MSKF aerosol coupling, which is refused. RRTMG uses default radii at "
    "aercu_opt=0, matching pristine WRF has_reqc/i/s=0 (proofs/v040/nssl_cumulus/radius_audit.json). "
    "Parent->child forcing of Ns/Ng/Nc is implemented; nested runs remain refused: parent boundary QNCLOUD "
    "and unresolved pre-mask Nc negatives fail the real WN3 N2/N6 gate "
    "(proofs/s1_mpfix/NESTED_ADMISSION.json). CPU-qualified "
    "(Swiss d01 coupled smoke); current-candidate Swiss GPU E41 STACK 16 B passes, "
    "kInput transpose 2226 optimized HLO instructions exceeds 500 (experimental opt-in disclosure). "
    "GPU/coupled-forecast qualification pending.",
    ("sf_urban_physics", 2): "Operationally wired into the scan: WRF BEP multi-layer canopy, canyon radiation, "
    "material heat diffusion, Noah-MP rural blend and YSU1/BouLac8 urban PBL tendencies. Requires Noah-MP4, "
    "revised-MM5 SF1 and RRTMG4/4; MYJ2 needs absent module_bl_myjurb and is refused. Default URBPARM.TBL "
    "UTYPE=2 only; LCZ/NUDAPT refused: grouped per-type geometry/carries/PBL sources are required. Compact REAL32 material carry, PBL sources and held SF1/radiation "
    "are restart-complete. CPU-oracle-qualified against pristine WRF REAL4/REAL8 full caller and PBL "
    "savepoints (proofs/v040/urban_bep_bem); conditioned REAL4 quantities use approved independent "
    "ROUND01 operands, state/REAL8/D6 gates unchanged. Authentic Madrid CPU coupled smoke; "
    "GPU/coupled-forecast qualification pending. GPU E41/launch cost unmeasured.",
    ("sf_urban_physics", 3): "Operationally wired into the scan: WRF BEP+BEM including indoor radiation, "
    "wall/window/floor/roof diffusion, occupants/equipment, ventilation, HVAC and moisture regulation, "
    "Noah-MP blend and YSU1/BouLac8 urban PBL tendencies. Requires Noah-MP4, revised-MM5 SF1 and RRTMG4/4; "
    "MYJ2 needs absent module_bl_myjurb and is refused. Default URBPARM.TBL UTYPE=2 only; LCZ/NUDAPT "
    "refused; LCZ needs grouped per-type geometry/carries/PBL sources. Rooftop PV (PV_FRAC_ROOF>0: radfluxspv/hsfluxpv, "
    "EP_PV_URB3D) and green roofs (GR_FLAG=1, GR_TYPE 1/2, class GR_FRAC_ROOF>0: roof_temp_veg/soil_moist/"
    "flux_flat_roof/wall_gr, irrigation IRHO, QGR/TGR/DRAINGR) are CPU-oracle-qualified (proofs/w41_writergaps: REAL8 "
    "0 violations on every field/step; REAL4 history producers inside WRF's REAL4-vs-REAL8 envelope, green-roof fluxes "
    "frozen caps); GR_FLAG=0 with GR_FRAC_ROOF>0 and GR_FLAG=1 without a class fraction stay refused. "
    "Compact REAL32 state and held SF1/radiation are restart-complete. CPU-oracle-qualified "
    "against pristine WRF REAL4/REAL8 full caller, AC on/off and PBL savepoints (proofs/v040/urban_bep_bem); "
    "conditioned REAL4 quantities use approved independent ROUND01 operands, state/REAL8/D6 gates unchanged. "
    "Authentic Madrid CPU coupled smoke. GPU: base BEP+BEM composite c57 E41 FAIL (disclosed); PV 0.3 + green roof 1.0 "
    "Madrid 9 h smoke finite with all fields written and E41 FAIL of the same class (STACK 976 B); incremental PV/green-roof "
    "launch/runtime/memory cost unknown (no matched PV/GR-off GPU control); coupled-forecast fidelity vs CPU-WRF pending.",
    ("sf_urban_physics", 1): "Operationally wired into the GPU scan (single-layer urban canopy model SLUCM, v0.4.0 lane o2-urban): "
    "faithful port of phys/module_sf_urban.F (urban_param_init/urban_var_init/urban) coupled exactly like "
    "WRF's noahmp_urban (Noah-MP runs urban cells as NATURAL with FVGMAX 0.96; FRC_URB2D blend of "
    "TSK/HFX/QFX/LH/GRDFLX/ALBEDO/QSFC/UST; surface-driver urban T2/Q2/U10/V10). Requires "
    "sf_surface_physics=4. CPU-oracle-qualified step-by-step against pristine WRF "
    "(proofs/v040/urban_slucm: 18 real Tenerife urban columns day/night + stress regimes, 3 URBPARM "
    "tables, fp64 <= 6e-12 rel, fp32 within the REAL rounding spread). URBPARM.TBL CH_SCHEME=1, "
    "GROPTION=1, IRI_SCHEME=1 and "
    "slucm_distributed_drag are refused, and so is a first model level within 15% of the canopy height "
    "ZDC+Z0C+2 (WRF's FATAL in urban). GPU-qualified vs CPU-WRF Madrid GFS 20261013_00z 24 h; E41 "
    "robustness flag: none (full scan PASS). D6 every hourly frame (25), all CPU fields inventoried; "
    "proofs/v040/gpuqual/urban1. Active urban land use at 31 grid points; cold shared validation wall "
    "441.600 s, sampled peak 1714 MiB. Additional opt-in LCZ51-61, NUDAPT morphology, diurnal AH/ALH and "
    "distributed/hourly AHE have CPU column/integration evidence (proofs/v040/urban_lcz), GPU qualification pending. "
    "NUDAPT LF1 is experimental one-cell zero-init parity only: WRF reuses undefined U10/V10 across tile cells; "
    "compiler initialization does not reset them per cell. Sealed two-cell pristine differences are "
    "HFX 6.273636 W/m2, U10 0.992088 m/s and TSK 0.301880 K; coupled CPU-WRF/D6 qualification pending "
    "(docs/UPSTREAM_WRF_BUGS.md). LCZ without morphology is unaffected; urban2/3 LCZ remains refused.",
    ("km_opt", 3): "CPU-ONLY: GPU selection refused CLI pre-JAX / API pre-device; the only GPU 24 h attempt became non-finite at 1 h, with no completed CPU-WRF reference (proofs/v040/gpuqual/km3/D6.json). Literal WRF diff_opt=2 3-D Smagorinsky "
    "(gpuwrf.dynamics.les3d_smagorinsky; RK1-frozen, vertical_diffusion_2 only with bl_pbl_physics=0; "
    "isfflx/tke_drag_coefficient/tke_heat_flux are not read from the namelist, WRF defaults 1/0/0 apply; "
    "moist diffusion reaches the root scalar update through the flux-form moisture transport). "
    "CPU-oracle-qualified against pristine WRF REAL4 module_diffusion_em on real terrain crops. Earlier "
    "coupled evidence was CPU-only (Swiss d01): 3 steps in the legacy fp64 carry and 1 step of the release "
    "REAL32 program with Pallas interpret mode + CPU-emulated PTX rounding, finite with 0 guard events; "
    "the 24 h GPU attempt failed with nonfinite u at the hour-1 observation; original CPU-WRF crashed at "
    "00:04:30 on Swiss and also crashed on a fresh authentic Madrid case, so no complete matched reference "
    "is available (proofs/v040/gpuqual/km3). GPU/coupled-forecast qualification pending.",
    ("cu_physics", 5): "Operationally wired into the GPU scan (Grell-3D ensemble, v0.3.4): line-faithful port of "
    "G3DRV/conv_grell_spread3d, CPU-oracle-qualified against the pristine-WRF multi-column tile oracle "
    "(proofs/v034); Positive cudt is refused. Not oracle-exercised: the CUP_enss downdraft-origin "
    "(keep_going) search. GPU-qualified vs CPU-WRF Swiss 24 h; E41 robustness flag: loop_add_fusion.79: "
    "1713 HLO > 500. D6 every hourly frame (25), all CPU fields inventoried; proofs/v040/gpuqual/cu5.",
    ("cu_physics", 93): "Operationally wired into the GPU scan (Grell-Devenyi ensemble, v0.3.4): line-faithful port of "
    "GRELLDRV, CPU-oracle-qualified against the pristine-WRF multi-column tile oracle (proofs/v034); "
    "Positive cudt is refused. Not oracle-exercised: the GD neg_check branch. GPU-qualified vs CPU-WRF "
    "Swiss 24 h; E41 robustness flag: none (full scan PASS). D6 every hourly frame (25), all CPU fields "
    "inventoried; proofs/v040/gpuqual/cu93.",
    ("bl_pbl_physics", 9): "Operationally wired into the GPU scan (CAM-UW, v0.3.4): CPU-oracle-qualified "
    "against proofs/v034/camuw_oracle; GPU/coupled-forecast qualification pending. Coupling caveat: WRF "
    "passes LW-only RTHRATENLW, the port passes the held TOTAL radiative heating RTHRATEN (LW+SW), so "
    "daytime cloud-top radiative forcing of the PBL is biased (exact at night).",
    ("sf_surface_physics", 3): "Scan-wired RUC LSM (coupling.ruc_surface_hook); CLI and API, MYNN-SL/Thompson, real.exe-prepared RUC "
    "wrfinput with 6 or 9 soil levels. CPU-oracle-qualified: physics.ruclsm matches the unmodified WRF "
    "LSMRUC oracle (proofs/v034/oracle/ruclsm, 48 regimes x 12 steps incl. snow/melt/frozen "
    "soil/convective frpcpn, fp64 <=1.3e-10 rel); RUCLSMINIT and SFCDIAGS_RUCLSM CPU-oracle-qualified "
    "(proofs/o2_ruccli); held 2-m history and full restart. Sea-ice/lake points fail closed. GPU-qualified "
    "vs CPU-WRF Madrid GFS 20261013_00z 24 h; E41 robustness flag: none (full scan PASS). D6 every hourly "
    "frame (25), all CPU fields inventoried; proofs/v040/gpuqual/sf3. Measured RUC input has 6 soil "
    "levels; 9-level input remains unqualified by this case.",
    ("ra_lw_physics", 3): "Operationally wired into the GPU scan (coupling.cam_radiation, held-rate RTHRATEN + CAM GLW to the "
    "land surface). CPU-oracle-qualified: physics.ra_cam matches the unmodified WRF camrad true caller "
    "(proofs/cam_rad CAM01, 210 real WN3 + augmented multi-region cloud columns, day/night; fluxes "
    "bit-identical in REAL, heating <= 2 ulp, held abs/ems path bit-identical); Absorptivities/emissivity "
    "are held and recomputed at WRF STEPABS cadence (cam_abs_freq_s, default 21600 s; carried in restart); "
    "XICE = 0. GPU-qualified vs CPU-WRF Swiss 24 h; E41 robustness flag: "
    "input_concatenate_exponential_transpose_fusion: 2141 HLO > 500; "
    "loop_add_concatenate_divide_log_multiply_power_sqrt_transpose_fusion: 1144 HLO > 500; "
    "loop_add_broadcast_clamp_concatenate_divide_exponential_maximum_multiply_sqrt_transpose_fusion: 1149 "
    "HLO > 500; loop_add_concatenate_transpose_fusion: 1274 HLO > 500; "
    "loop_add_concatenate_convert_exponential_multiply_transpose_fusion: 1426 HLO > 500; "
    "input_divide_select_transpose_fusion: 641 HLO > 500; STACK 2144 B > 256. D6 every hourly frame (25), "
    "all CPU fields inventoried; proofs/v040/gpuqual/ra3.",
    ("ra_sw_physics", 3): "Operationally wired into the GPU scan (coupling.cam_radiation, held-rate RTHRATEN + CAM SWDOWN = "
    "GSW/(1-ALBEDO) to the land surface). CPU-oracle-qualified against the unmodified WRF camrad true "
    "caller (proofs/cam_rad CAM01, day/night, cloudy/clear, uniform CAM aerosol climatology); XICE = 0. "
    "GPU-qualified vs CPU-WRF Swiss 24 h; E41 robustness flag: "
    "input_concatenate_exponential_transpose_fusion: 2141 HLO > 500; "
    "loop_add_concatenate_divide_log_multiply_power_sqrt_transpose_fusion: 1144 HLO > 500; "
    "loop_add_broadcast_clamp_concatenate_divide_exponential_maximum_multiply_sqrt_transpose_fusion: 1149 "
    "HLO > 500; loop_add_concatenate_transpose_fusion: 1274 HLO > 500; "
    "loop_add_concatenate_convert_exponential_multiply_transpose_fusion: 1426 HLO > 500; "
    "input_divide_select_transpose_fusion: 641 HLO > 500; STACK 2144 B > 256. D6 every hourly frame (25), "
    "all CPU fields inventoried; proofs/v040/gpuqual/ra3.",
    ("mp_physics", 9): "Milbrandt-Yau two-moment, six transported number species and separate hail: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. REAL32 pristine WRF active cold/hail "
    "module_mp_milbrandt2mom.F column and caller outputs in proofs/s2_milbrandt. Root flow_dep_bdy, mandatory per-step REFL_10CM, "
    "exact restart of all prognostics; WRF has_reqc/i/s=0 (unused local Dm diagnostics disclosed). "
    "Single domains only; nested extra-species forcing not qualified.",
    ("grid_fdda", 1): "PSU grid analysis nudging through resident wrffdda and WRF RK tendencies, "
        "continuous daily clock and held HFX_FDDA history/restart. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Original WRF analysis ingestion requires FieldType=104 metadata; "
        "successful completion alone does not establish ingestion.",
    ("windfarm_opt",1): "Fitch turbine drag and QKE source with WRF power/Cp interpolation and energy normalization; "
        "MYNN=5/5, windturbines[-ij].txt and wind-turbine-N.tbl required. POWER/QKE restart retained. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Literal REAL4 path BYTE; "
        "small compiled per-value contraction deviations disclosed in proofs/u2_windfarm/UPSTREAM_WRF_BUGS.md.",
    ("topo_wind", 1): "GPU-qualified vs CPU-WRF 24 h (release defaults, D6 every hourly frame, all CPU fields "
    "inventoried) on Swiss 3 km YSU+sfclay1 and Swiss 3 km Shin-Hong, and WN3 0227 d02/d03 YSU+sfclay1 (3/1 km "
    "one-way nests, feedback=0); proofs/w41_topowind/GPU_ARMS.json. E41 robustness flag (opt-in disclosure): "
    "overall E41 FAIL on every arm; STACK 400 B (Swiss YSU loop_maximum_minimum_select_subtract_fusion) and "
    "496 B (WN3 fused d02/d03, two loop_add_maximum_minimum_select_*subtract_fusion) in the revised-MM5 surface "
    "layer that topo_wind=1 re-fuses (same-source topo_wind=0 control 8 B; moving the YSU CTOPO2 blend out of "
    "the column did not change it and was reverted); Shin-Hong STACK 24 B; kInput transpose >500 HLO: Swiss YSU "
    "input_add_clamp_convert_divide_gather_select_transpose_fusion 570, loop_select_fusion.16/.17/.18 "
    "549/661/659, loop_broadcast_transpose_fusion 586; Shin-Hong input_subtract_transpose_fusion.1/.2 625/538, "
    "input_select_transpose_fusion.5 534, input_add_divide_multiply_select_subtract_transpose_fusion 725, "
    "input_concatenate_multiply_transpose_fusion 615; WN3 fused d02/d03 loop_add_divide_select_slice_fusion 926. "
    "Runtime cost unmeasured. Two-way CTOPO/CTOPO2 feedback (feedback=1) is CPU-tested only. "
    "WRF start_em CTOPO/CTOPO2 (Jimenez-Dudhia: VAR_SSO + terrain Laplacian) byte-exact vs pristine WRF 4.7.1; "
    "like WRF only YSU (bl=1) and Shin-Hong (bl=11) read them (momentum surface drag + CTOPO2 10 m wind blend "
    "into U10/V10); any other PBL runs unchanged with a warning. Refuses wrfinput without VAR_SSO "
    "(GOT_VAR_SSO=0, WRF start_em.F:222). Static fields recomputed at every init/resume.",
    ("topo_wind", 2): "GPU-qualified vs CPU-WRF 24 h only on WN3 0227 d01 YSU+sfclay1 9 km (release defaults, D6 "
    "every hourly frame, all CPU fields inventoried; proofs/w41_topowind/GPU_ARMS.json) plus pristine YSU oracle "
    "columns with real topo_wind=2 CTOPO; no Shin-Hong or other-domain topo_wind=2 forecast arm. E41 robustness "
    "flag (opt-in disclosure): overall E41 FAIL; STACK 88 B (PASS); kInput transpose >500 HLO: "
    "input_add_log_multiply_select_subtract_transpose_fusion 532, input_select_subtract_transpose_fusion 723, "
    "input_add_and_clamp_maximum_select_transpose_fusion 651. Runtime cost unmeasured. WRF start_em CTOPO "
    "(Mass-Ovens from the geogrid VAR; CTOPO2 stays 1) byte-exact vs pristine WRF 4.7.1; like WRF only YSU "
    "(bl=1) and Shin-Hong (bl=11) read it; any other PBL runs unchanged with a warning. Static fields "
    "recomputed at every init/resume.",
    ("tracer_opt", 2): "Non-Chem passive tracer operators are CPU-oracle-qualified; "
    "GPU/coupled-forecast qualification pending. Opt-in root and one-way nested transport "
    "(feedback=0), PBL mixing, custom sources/removal, history and full restart. "
    "The unchanged coupled max-error gate remains FAIL-explained: 1.3697e-4 > 1e-4. "
    "Operator exactness and mass conservation are accepted for lane scope; final composite D6 "
    "decides coupled qualification. Actual GPU launches, memory and timing remain unmeasured.",
    ("sf_ocean_physics",1): "WRF OML1D: seven resident REAL Registry fields, current surface fluxes, "
    "step-entry winds, depth/temperature anchors and optional relaxation. Both CLI pipelines bind/init; "
    "Noah-MP currently required; sst_update=1 and PWP 3-D refused. "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. docs/ocean-mixed-layer.md",
    ("sf_surface_physics", 8): "Experimental SSiB, CLI/API opt-in with real.exe-prepared three-layer soil, "
    "sfclay=1/YSU=1, mp=8, SW=1/LW=4, no sea ice/lake/urban. CPU-oracle-qualified literal REAL4: "
    "400/400 calls BYTE vs pristine WRF (proofs/s2_ssib), all 48 snow leaves and Registry history/restart. "
    "Compiled REAL4 has 56 disclosed field residuals beyond native FMA/input bounds in 300 wet/melt calls: "
    "maximum 0.0562 W/m2 flux and 0.000672 K temperature; no tolerance widened, QFX tightened by WRF HLAT. "
    "Pristine UPDAT1 SNOWW(2) undefined read uses compiler-defined zero; tested sensitivity 7.85e-12 m runoff. "
    "GPU/coupled-forecast qualification pending; D6 decides. Performance and other limitations: "
    "proofs/s2_ssib/README.md and UPSTREAM_WRF_BUGS.md.",
    ('grid_sfdda',1): 'PSU surface analysis nudging through resident auxinput9 analyses and WRF RK tendency lanes. '
    'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Requires grid_fdda=1, MYNN PBL/SFCLAY=5, '
    'Noah-MP=4; independent surface cadence, density smoothing, wind scaling and saturation restriction.',
    ('grid_sfdda',2): 'FASDAS atmospheric surface nudging with Noah-MP=4. '
    'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Requires grid_fdda=1 and MYNN PBL/SFCLAY=5. '
    'Classic Noah indirect soil nudging is refused; WRF Noah-MP has no indirect FASDAS soil update. '
    'SDA/QNORM/HFX_BOTH/QFX_BOTH history diagnostics are not written.',
    ("bl_pbl_physics",4): "Experimental QNSE with optional ARW EDMF (mfshconv=1), mandatory sf=4. "
    "No LSM or Noah-MP only; slab (1), Noah-classic (2) and Pleim-Xiu (7) caller wiring is refused. "
    "CPU-oracle-qualified: pristine literal REAL4 byte parity and finite compiled PBL input-ULP bounds. "
    "EDMF compiled rounding exceeds the pristine WRF FMA spread in 130 oracle values "
    "(max heating excess 1.01747e-6 K/s, RC_MF 6.49016e-9 kg/kg); experimental deviation documented in proofs/s2_qnse. "
    "GPU/coupled-forecast qualification pending; persistent surface exchange, TKE and MF condensate restart. "
    "Actual c26 GPU E41: FAIL; PBL STACK 7232 B / optional EDMF STACK 2952 B per thread versus the 256 B default cap. Opt-in resource disclosures; default hard gate unchanged. See proofs/t_qnsecompile/QUALIFICATION.json.",
    ("sf_sfclay_physics",4): "Experimental QNSE surface layer, mandatory bl=4. "
    "No LSM or Noah-MP only; slab (1), Noah-classic (2) and Pleim-Xiu (7) caller wiring is refused. "
    "CPU-oracle-qualified: pristine literal REAL4 byte parity with source-hashed WRF tables; finite compiled input-ULP bounds. "
    "GPU/coupled-forecast qualification pending. Surface solve retained through land and PBL.",
    ("cu_physics", 14): "KIM SAS, native WRF REAL32 CU_KSAS; CPU-oracle-qualified; "
    "GPU/coupled-forecast qualification pending. Initial five-column literal AVX path rain/T/QV/U/V BYTE; REAL8 "
    "scaled error <=6.94e-14. Disclosed default-ISA compiler contraction: rain abs 3.75509e-6 "
    "versus pristine WRF FMA spread 3.33786e-6 (excess 4.17233e-7); tiny QC/QI/QV cells also "
    "exceed that spread. Accepted by Owner; no tolerance/ISA change; final composite D6 is binding. "
    "Expanded455 native32 rain abs 1.31726265e-5 vs original FMA spread 6.19366765e-5; "
    "max same-cell excess 1.03116035e-5 in two cells. Held scalar/momentum cadence, same-step HPBL_HOLD; "
    "docs/SAS_FAMILY_SUPPORT.md.",
    ("cu_physics", 96): "GFS NSAS, native WRF REAL32 deep and own shallow; CPU-oracle-qualified; "
    "GPU/coupled-forecast qualification pending. REAL8 scaled error <=5.70e-13. Disclosed "
    "conditioning chain: FPVS max 2 ULP -> XAA0 1 ULP -> closure mass flux 44-68 ULP -> "
    "rain 5.798721e-6 relative (1.0430813e-6 absolute). Accepted by Owner; no tolerance/ISA change; "
    "final composite D6 is binding. Original i-row pressure limits, current HFX/QFX/HPBL, "
    "held scalar/momentum cadence. Expanded455 native32 rain abs 4.45032492e-5 vs original FMA "
    "spread 4.64038923e-4; max same-cell excess 2.55461782e-6 in13cells. Original FMA switches the "
    "same near-zero rain/cloud-level branch. docs/SAS_FAMILY_SUPPORT.md.",
    ("cu_physics", 99): "Experimental distinct old Kain-Fritsch, pristine module_cu_kf.F:KFCPS/KFPARA "
    "analytic thermodynamics and gradual freezing; CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Native REAL32 kernel/carry, full promoted-REAL8 ten-field gate passes with exact lifetimes. "
    "Disclosed compiled native conditioning: rain max abs 1.93715096e-6 vs original Fortran FMA "
    "spread 1.89244747e-6 (excess 4.47034836e-8); QI 6.66489086e-12 vs 4.57589522e-12, "
    "QV 8.46966941e-11 vs 7.84439180e-11. No tolerance/ISA change; final composite D6 binding. "
    "W0AVG/NCA and seven held rates restart; old NCA half-step trigger and <=0 clear, not KF-eta <=1. "
    "See docs/SAS_FAMILY_SUPPORT.md.",
    ("cu_physics", 95): "Experimental old GFS OSAS, distinct pristine OSASCNV and source-identical ARW SHALCV; "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Native REAL4 all eight outputs "
    "BYTE on 455 recorded-draw columns at two cadences (198/196 active); GFS DOUBLE internals. "
    "WRF draws from the system clock; gpuwrf is deterministic by design. Fixed seed 950040, "
    "Threefry2x32-20 key=(seed,domain), counter=(step,column); uint32 keys and clock restart. "
    "Promoted-REAL8 fully evaporated ~1e-20 rain has disclosed exact-zero branch sensitivity; "
    "original WRF FMA field spread covers maximum response; final composite D6 is binding. "
    "See docs/SAS_FAMILY_SUPPORT.md.",
    ("cu_physics", 94): "2015 GFS SAS, pristine WRF ARW CU_SAS/SASCNVN and OLD_ARW_SHALCV: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. WRF REAL4 driver outputs, "
    "DOUBLE kind_phys internals; WRF first-step/STEPCU held-rate cadence and sas_pgcon binding. "
    "No momentum tendencies coupled by WRF ARW. See docs/SAS_FAMILY_SUPPORT.md.",
    ("mp_physics",19): "WRF legacy NSSL one-moment with hail mass and predicted graupel volume: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Original REAL4/REAL8 active columns, native moment-specific branches, same-input original FMA controls: "
    "proofs/s2_mpmore/nssl1mom. Registry CCN scalar is advected and held by one-moment microphysics; "
    "no prognostic hydrometeor numbers or hail volume. No effective-radius handoff (original has_req flags0). "
    "Single domains; matching nonzero moist/scalar advection required; CCN0, km2 and custom nssl_mp_params refused. "
    "REFL_10CM held with the NSSL makediag lifetime from radardd02's ipconc=0 branch (fp32 BYTE vs pristine S6, "
    "proofs/t_classd). GPU E41/resources and forecast fidelity remain pending.",
    ("mp_physics",21): "WRF legacy NSSL one-moment without hail or prognostic numbers/volume: "
    "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Original REAL4/REAL8 active columns, native saturation/freezing branches, same-input original FMA controls: "
    "proofs/s2_mpmore/nssl1mom. Six moist species only; no effective-radius handoff (original has_req flags0). "
    "Single domains; custom nssl_mp_params refused. REFL_10CM held with the NSSL makediag lifetime from "
    "radardd02's ipconc=0 branch (fp32 BYTE vs pristine S6, proofs/t_classd). "
    "GPU E41/resources and forecast fidelity remain pending.",
    ("sf_surface_physics",4): "Requires each wrfinput MMINLU='MODIFIED_IGBP_MODIS_NOAH'; USGS and unknown/missing MMINLU are refused before land field reads or allocations. NUM_LAND_CAT alone does not identify the dataset. MODIS LCZ classes 51-61 remain accepted under existing urban-option requirements. Noah-MP includes the independent WRF glacier timestep (opt_gla=1/2). CPU-oracle-qualified; GPU/coupled-forecast qualification pending for glacier domains. Glacier QRAINXY=0 matches WRF's defined per-point cold-zero hold; arbitrary nonzero externally seeded QRAINXY is outside this qualification. opt_snf=5 is refused on glacier domains. Released Noah-MP default path is unchanged. Experimental opt_run=5 MMF: CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Original REAL4 cold initialization, soil columns and eight-neighbor groundwater transport; WTDDT cadence, complete restart and held history. Requires NCAR groundwater geography via GEOGRID.TBL.ARW.noahmp, one ice-free domain, no urban/lake, four STAS soil layers, opt_inf=1/2 and noahmp_acc_dt=0. Original-WRF input-ULP/FMA bounds; GPU performance, memory and E41 robustness unmeasured. proofs/s2_mmf/INPUT_REQUIREMENTS.md. Additional natural-soil controls: docs/noahmp-options.md; CPU-oracle-qualified, GPU/coupled-forecast qualification pending.",
    **{("mp_physics", code): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
       "Experimental Ferrier-Aligo REAL column kernel (module_mp_fer_hires.F): literal no-FMA execution is BYTE against "
       "36 pristine-WRF columns (tests/v040/mpmisc). Compiled CPU rounding deviations beyond "
       "the measured WRF FMA/input-ULP spread are explicitly disclosed in "
       "proofs/v040/mpmisc/COMPILER_DEVIATIONS.md; D6 decides final qualification. "
       "Native CUDA lowering passes; actual GPU E41/timing/VRAM remain unmeasured. "
       "Single domains, cu_physics=0 only: mp5 transports CWM and holds ice/rain/rime fractions; mp15 transports QRIMEF."
       for code in (5, 15)},
    ("mp_physics", 7): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Experimental REAL Goddard NUWRF four-ice kernel (module_mp_gsfcgce_4ice_nuwrf.F): all 18 original-WRF columns/36 public fields "
    "literal BYTE; all 5760 compiled prognostic values within independent 14-input-ULP/WRF-FMA bounds. "
    "Small diagnostic/ACPHYSF compiled excesses disclosed in proofs/v040/mpmisc/goddard4ice/COMPILED_DEVIATIONS.md. "
    "Separate hail mass, six accumulated latent budgets, six micron radii and held SR retained in restart. "
    "Original PHYS_TOT history sum is retained (720 independently qualified REAL values BYTE). "
    "Single domains, non-Chem defaults, WRF-order post-RK caller only. Native CUDA IR lowering passes; "
    "actual GPU E41/timing/VRAM remain unmeasured; D6 pending.",
    ("mp_physics", 38): "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "Experimental REAL Thompson variable-density hail+aerosol composition of module_mp_thompson.F: "
    "11 pristine-WRF columns (6160 physical values/77 public surfaces) and three evolving own-steps literal BYTE. "
    "Independent input-ULP/WRF-FMA bounds cover threshold branch sensitivity; 115/6237 small ordinary/fresh "
    "compiled excesses are disclosed in proofs/v040/mpmisc/thompson38/COMPILED_DEVIATIONS.md; D6 decides final qualification. "
    "Single domains, wif_input_opt=1/aer_init_opt=0 and zero black carbon; original volume/number/aerosol transport, "
    "held emissions/SR/REAL accumulated totals and strict restart retained; no effective-radius flags. "
    "JAX composition reuses shared Thompson equations, not a fused native column kernel. "
    "Original DOUBLE collision/freezing/rain-snow tables (973 MB) ride as device-resident traced operands "
    "(OperationalNamelist.thompson_hail_tables), not baked HLO constants; the AOT key still hashes their bytes. "
    "Actual GPU E41/timing/VRAM remain unmeasured.",
    ('ra_lw_physics', 7): "EXPERIMENTAL CPU-ONLY FLG, paired 7/7. CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "GPU refused before JAX: native GPU column kernel pending (v0.4.x). Qualified caller: MP8, cu0, Noah-MP4, supplied WRF statics; "
    "one cadence solve, held native rates/fluxes/clouds, exact restart. Literal REAL4 BYTE against pristine WRF; "
    "17 strict compiled band failures (3254 values): 7 within the independent native FMA envelope; 10 outside (285 values). Maximum band error 0.00139618 W/m2, source-bound ratio 12.8554; full-field LW TOA error 0.00961304 W/m2 versus native FMA spread 0.00119019 W/m2, heating error 4.44816e-7 K/s. Ten residual bands and full-field excesses are small experimental deviations only, D6 pending; not native-envelope passes. No GPU speed/memory claim. "
    "GFDL requires optional GNU Fortran once at initialization for bundled native CO2 tables. See docs/flg-gfdl-radiation.md and proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json (per-band independent bounds/dispositions/hashes).",
    ('ra_sw_physics', 7): "EXPERIMENTAL CPU-ONLY FLG, paired 7/7. CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "GPU refused before JAX: native GPU column kernel pending (v0.4.x). Qualified caller: MP8, cu0, Noah-MP4, supplied WRF statics; "
    "one cadence solve, held native rates/fluxes/clouds, exact restart. Literal REAL4 BYTE against pristine WRF; "
    "17 strict compiled band failures (3254 values): 7 within the independent native FMA envelope; 10 outside (285 values). Maximum band error 0.00139618 W/m2, source-bound ratio 12.8554; full-field LW TOA error 0.00961304 W/m2 versus native FMA spread 0.00119019 W/m2, heating error 4.44816e-7 K/s. Ten residual bands and full-field excesses are small experimental deviations only, D6 pending; not native-envelope passes. No GPU speed/memory claim. "
    "GFDL requires optional GNU Fortran once at initialization for bundled native CO2 tables. See docs/flg-gfdl-radiation.md and proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json (per-band independent bounds/dispositions/hashes).",
    ('ra_lw_physics', 99): "EXPERIMENTAL CPU-ONLY GFDL-Eta, paired 99/99. CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "GPU refused before JAX: native GPU column kernel pending (v0.4.x). Qualified caller: MP8, cu0, Noah-MP4, supplied WRF statics; "
    "one cadence solve, held native rates/fluxes/clouds, exact restart. Literal REAL4 BYTE against pristine WRF; "
    "Compiled GFDL residuals: SW flux error 0.0181 W/m2 and SW heating error 7.52e-8 K/s; disclosed experimental deviations, D6 pending, not strict fidelity passes. No GPU speed/memory claim. "
    "GFDL requires optional GNU Fortran once at initialization for bundled native CO2 tables. See docs/flg-gfdl-radiation.md and proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json (per-band independent bounds/dispositions/hashes).",
    ('ra_sw_physics', 99): "EXPERIMENTAL CPU-ONLY GFDL-Eta, paired 99/99. CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
    "GPU refused before JAX: native GPU column kernel pending (v0.4.x). Qualified caller: MP8, cu0, Noah-MP4, supplied WRF statics; "
    "one cadence solve, held native rates/fluxes/clouds, exact restart. Literal REAL4 BYTE against pristine WRF; "
    "Compiled GFDL residuals: SW flux error 0.0181 W/m2 and SW heating error 7.52e-8 K/s; disclosed experimental deviations, D6 pending, not strict fidelity passes. No GPU speed/memory claim. "
    "GFDL requires optional GNU Fortran once at initialization for bundled native CO2 tables. See docs/flg-gfdl-radiation.md and proofs/s2_flggfdl/receipts/FG93_flg_compiler_disposition.json (per-band independent bounds/dispositions/hashes).",
    **{(key,code): 'Native Noah-MP control, CPU-oracle-qualified; GPU/coupled-forecast qualification pending. '
       'Literal REAL4 original-source gates, disclosed small compiled deviations, exact active carries/restart. '
       'Glacier opt_gla=1/2 uses the approved independent WRF timestep; glacier opt_snf=5 is refused. QRAINXY=0 mirrors WRF cold-zero hold; arbitrary externally seeded nonzero QRAINXY is outside qualification. See docs/noahmp-options.md and proofs/u2_land. '
       + ('EXPERIMENTAL dynamic VIC: original unassigned YD follows compiler-zero policy; initialization sensitivity is material. '
          if (key=='opt_run' and code==8) or (key=='opt_infdv' and code!=0) else '')
       for key,codes in _NOAHMP_CONTROL_VALUES.items() for code in codes},
    ('sst_skin',1): 'WRF ocean_skin after radiation/aux4 and before surface physics; resident bulk SST/SSTSK/DTW with exact restart. '
       'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Noah-MP/RRTMG, ice-free prescribed SST. docs/noahmp-options.md.',
}


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
    if (impl is not None and code in impl) or code in _IMPLEMENTED_FEATURE_VALUES.get(key, ()):
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.IMPLEMENTED,
            reason=_IMPLEMENTED_NOTES.get((key, code), "Operationally wired into the GPU scan."),
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
    if _coerce_int_or_bool(value) in _IMPLEMENTED_FEATURE_VALUES.get(key.lower(), ()):
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
