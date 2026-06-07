"""Machine-readable WRF v4 scheme support catalog -- the public honesty contract.

This module answers, for *every* WRF v4 option of the common physics/dynamics
namelist groups (plus the major feature switches such as WRF-Chem, WRF-Fire,
FDDA, urban BEP/BEM, moving nests and stochastic physics), the single question a
WRF developer evaluating this port cares about:

    "If I set this in my namelist, does the GPU port run it, refuse it with a
     named reason, or is it a documented out-of-scope decision?"

Every option resolves to exactly one :class:`SupportStatus`:

* ``IMPLEMENTED``            -- operationally GPU-scan-wired and run normally.
* ``REFERENCE_ONLY``        -- a recognized WRF v4 scheme with a parity-proven
                               (savepoint / isolated / analytic-oracle) adapter
                               that is NOT yet wired into the operational GPU
                               scan. It is *accepted* by the namelist validator
                               (so a reference / single-column comparison can be
                               run) but the operational forecast scan fail-closes
                               it loudly with a named reason -- never a silent
                               wrong result.
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
                               multi-layer urban canopy, moving/vortex-following
                               nests, FDDA/4DVAR nudging, stochastic physics).
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
* When a scheme is parity-proven but not operationally wired, it is
  ``REFERENCE_ONLY`` (a caveat), never ``IMPLEMENTED`` (an over-claim).
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
    "mp_physics": frozenset({0, 1, 2, 3, 4, 6, 8, 10, 16}),
    "cu_physics": frozenset({0, 1, 2, 3, 6}),
    "bl_pbl_physics": frozenset({0, 1, 5, 7, 8}),
    "sf_sfclay_physics": frozenset({0, 1, 5, 7}),
    "sf_surface_physics": frozenset({0, 2, 4}),
    # ra_lw=1 (classic AER RRTM 16-band LW) is now operationally scan-wired
    # (coupling.physics_couplers.rrtm_lw_theta_tendency over the JAX-traceable
    # physics.ra_lw_rrtm_jax kernel, dispatched in runtime.operational_mode by
    # OperationalNamelist.ra_lw_physics; SW selected independently).
    "ra_lw_physics": frozenset({0, 1, 4}),
    # ra_sw=1 (Dudhia, Stephens-1984 broadband SW) is now operationally scan-wired
    # (coupling.physics_couplers.dudhia_sw_theta_tendency, dispatched in
    # runtime.operational_mode by OperationalNamelist.ra_sw_physics).
    "ra_sw_physics": frozenset({0, 1, 4}),
}

# Recognized WRF schemes with a parity-proven adapter that the operational scan
# fail-closes (selectable for a reference comparison, NOT an operational run).
# reason = the named scan-unwired reason; alternative = the operational swap.
_REFERENCE_ONLY: Mapping[str, dict[int, tuple[str, str]]] = {
    "cu_physics": {
        16: (
            "New Tiedtke is interface-compatible but not separately savepoint-"
            "gated by a distinct WRF source path; GPU-batching/gating is TODO, so "
            "it is fail-closed in the operational GPU scan.",
            "Use cu_physics=6 (modified Tiedtke, GPU-operational) or 1/3.",
        ),
    },
    "bl_pbl_physics": {
        2: (
            "MYJ TKE PBL has a WRF-savepoint-parity column adapter (CPU reference) "
            "but no operational GPU scan adapter/carry path yet.",
            "Use bl_pbl_physics=5 (MYNN), 1 (YSU) or 7 (ACM2) for the operational "
            "scan. If you must reference MYJ, pair it with sf_sfclay_physics=2.",
        ),
    },
    "sf_sfclay_physics": {
        2: (
            "Janjic Eta surface layer has a WRF-savepoint-parity column adapter "
            "(CPU reference) but no operational GPU scan adapter/carry path yet.",
            "Use sf_sfclay_physics=5 (MYNN-SL), 1 (revised-MM5) or 7 (Pleim-Xiu). "
            "Janjic Eta must pair with bl_pbl_physics=2.",
        ),
    },
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
    "mp_physics": "Use one of mp_physics=0/1/2/3/4/6/8/10/16 (8=Thompson is the "
    "operational default).",
    "cu_physics": "Use one of cu_physics=0/1/2/3/6 (1=Kain-Fritsch, 3=Grell-"
    "Freitas, 6=Tiedtke are GPU-operational).",
    "bl_pbl_physics": "Use one of bl_pbl_physics=0/1/5/7/8 (5=MYNN, 1=YSU, 7=ACM2, "
    "8=BouLac).",
    "sf_sfclay_physics": "Use one of sf_sfclay_physics=0/1/5/7 (5=MYNN-SL, "
    "1=revised-MM5, 7=Pleim-Xiu).",
    "sf_surface_physics": "Use sf_surface_physics=4 (Noah-MP) or 2 (Noah classic).",
    "ra_lw_physics": "Use ra_lw_physics=4 (RRTMG).",
    "ra_sw_physics": "Use ra_sw_physics=4 (RRTMG) or 1 (Dudhia); both GPU-operational.",
    "diff_opt": "Use diff_opt=0/1/2 (1+km_opt=4 = 2-D Smagorinsky real-data "
    "default; 2+km_opt=1 = constant-K).",
    "km_opt": "Use km_opt=0/1/4 (4 with diff_opt=1 = 2-D Smagorinsky; 1 with "
    "diff_opt=2 = constant-K).",
    "damp_opt": "Use damp_opt=0 (off) or 3 (upper-level w-Rayleigh).",
    "diff_6th_opt": "Use diff_6th_opt=0 (off) or 2 (monotonic 6th-order filter).",
    "rk_order": "Use rk_order=3 (WRF RK3).",
    "w_damping": "Use w_damping=0 or 1.",
    "sf_urban_physics": "Set sf_urban_physics=0 (urban canopy is not ported).",
}


# --------------------------------------------------------------------------- #
# Dynamics / numerics: implemented integer codes + the Smagorinsky note.      #
# These are gated by namelist_check.SUPPORTED_OPTIONS today; the catalog      #
# mirrors that so the public table is complete and consistent.                #
# --------------------------------------------------------------------------- #
_DYNAMICS_IMPLEMENTED: Mapping[str, frozenset[int]] = {
    "diff_opt": frozenset({0, 1, 2}),
    "km_opt": frozenset({0, 1, 4}),
    "damp_opt": frozenset({0, 3}),
    "diff_6th_opt": frozenset({0, 2}),
    "rk_order": frozenset({3}),
    "w_damping": frozenset({0, 1}),
    # Urban canopy: only "off" (0) is a real path; 1/2/3 are out_of_scope below.
    "sf_urban_physics": frozenset({0}),
}

# Dynamics options that ARE valid WRF codes but are fail-closed by the port,
# with an explicit reason + transition recipe. km_opt 2/3/5 (3-D TKE / 3-D
# Smagorinsky / SMS-3DTKE) are the notable real-data-LES selections the port
# does not implement; the operational horizontal-mixing path is the 2-D
# Smagorinsky (diff_opt=1/km_opt=4) or constant-K (diff_opt=2/km_opt=1).
_DYNAMICS_FAIL_CLOSED_REASON: Mapping[str, dict[int, str]] = {
    "km_opt": {
        2: "1.5-order 3-D TKE closure is not implemented (the port mixes "
        "vertically via the PBL scheme, not a prognostic 3-D TKE field).",
        3: "3-D Smagorinsky first-order closure is not implemented.",
        5: "SMS-3DTKE scale-adaptive LES/PBL closure is not implemented.",
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

# A small set of integer-enumerated keys whose NON-zero option codes are
# out_of_scope even though they appear in the WRF catalog: urban canopy 1/2/3.
# (Single-layer UCM and multi-layer BEP/BEM are all out of scope; only "no urban
# canopy" = 0 is supported.)
_OUT_OF_SCOPE_CODES: Mapping[str, dict[int, OutOfScopeFeature]] = {
    "sf_urban_physics": {
        1: OutOfScopeFeature(
            "sf_urban_physics", "single-layer urban canopy (UCM)",
            "The single-layer urban canopy model (UCM) is out of scope.",
            "Set sf_urban_physics=0 (urban is treated through the land-surface "
            "scheme's urban land-use categories).",
        ),
        2: OutOfScopeFeature(
            "sf_urban_physics", "multi-layer urban canopy (BEP)",
            "The multi-layer Building Effect Parameterization (BEP) urban canopy "
            "is out of scope.",
            "Set sf_urban_physics=0.",
        ),
        3: OutOfScopeFeature(
            "sf_urban_physics", "multi-layer urban canopy + building energy (BEM)",
            "The multi-layer BEP+BEM (Building Energy Model) urban canopy is out "
            "of scope.",
            "Set sf_urban_physics=0.",
        ),
    },
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
    "recognized; the port runs only the standard (non-positive-definite, "
    "non-monotonic) scalar transport. The WRF positive-definite (2), monotonic "
    "(3) and WENO (4) scalar/moisture transport variants are NOT yet scan-wired "
    "in v0.12.0."
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
        frozenset({0, 1}),
        _ADV_OPT_REASON,
        "Use moist_adv_opt=0/1 (standard transport); the PD/monotonic/WENO "
        "variants (2/3/4) are not wired.",
    ),
    RecognizedControl(
        "scalar_adv_opt", "scalar-advection",
        frozenset({0, 1}),
        _ADV_OPT_REASON,
        "Use scalar_adv_opt=0/1 (standard transport); the PD/monotonic/WENO "
        "variants (2/3/4) are not wired.",
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

    # 1) Out-of-scope enumerated codes (urban BEP/BEM, single-layer UCM).
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

    # 3) Reference-only (parity-proven, not scan-wired).
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
        per_code = _DYNAMICS_FAIL_CLOSED_REASON.get(key, {})
        reason = per_code.get(
            code,
            f"{scheme.name} is a recognized WRF v4 {_label(key)} option that is "
            f"NOT YET IMPLEMENTED in the GPU port.",
        )
        return SchemeSupport(
            key=key,
            code=code,
            status=SupportStatus.RECOGNIZED_FAIL_CLOSED,
            reason=reason,
            alternative=_DEFAULT_ALTERNATIVE.get(key, ""),
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
