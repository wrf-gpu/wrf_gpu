"""Fail-fast support check for WRF namelist/config options.

For v0.6.0, the physics values accepted here are the frozen interface matrix,
not a claim that every scheme is already wired into the operational dispatcher.
Unsupported option numbers still fail closed loudly. Per-scheme lanes must pass
their WRF savepoint parity gates before a non-Thompson suite can be used for an
integrated forecast.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gpuwrf.contracts.physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_RA_LW_PHYSICS,
    ACCEPTED_RA_SW_PHYSICS,
    ACCEPTED_SF_LAKE_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
    ACCEPTED_SF_URBAN_PHYSICS,
)
from gpuwrf.io.scheme_catalog import (
    APPROXIMATED_CONTROL_KEYS,
    IMPLEMENTED_CONTROL_KEYS,
    OUT_OF_SCOPE_FEATURE_KEYS,
    RECOGNIZED_CONTROL_KEYS,
    SchemeSupport,
    SupportStatus,
    classify_control,
    classify_feature_switch,
    classify_scheme,
    iter_full_catalog,
)
from gpuwrf.io.wrf_scheme_catalog import WRF_PARAM_LABEL, wrf_scheme_name


@dataclass(frozen=True)
class SupportedOption:
    """One supported-option registry entry."""

    key: str
    supported_values: frozenset[Any]
    implemented: str
    action: str


@dataclass(frozen=True)
class UnsupportedSelection:
    """A selected namelist/config value outside the faithful registry.

    ``outcome`` records *why* the value was rejected, so the caller (and the
    formatted message) can distinguish a recognized WRF v4 scheme that is not
    yet implemented in the GPU port from a value that is not a valid WRF option:

    * ``"not_yet_implemented"`` -- a recognized WRF v4 scheme that the port does
      not yet wire (``wrf_scheme`` names it);
    * ``"reference_only_not_operational"`` -- an oracle-backed WRF v4 scheme
      that is accepted by ``validate_namelist`` for a reference comparison but is
      NOT wired into the operational scan; the OPERATIONAL run path
      (``validate_operational_namelist``) rejects it to avoid a silent
      wrong-scheme run (``wrf_scheme`` names it);
    * ``"invalid_wrf_option"`` -- not a recognized WRF v4 option at all;
    * ``"recognized_control_not_wired"`` -- a recognized non-enumerated WRF
      control key (advection order, gwd_opt, a MYNN-EDMF sub-option, a
      physics-cadence interval) set to a value the operational scan does not
      wire (``wrf_scheme`` carries the control label);
    * ``"unsupported"`` -- generic rejection for keys without a WRF v4 catalog
      (e.g. a structural pairing constraint).
    """

    key: str
    location: str
    value: Any
    supported_values: tuple[Any, ...]
    implemented: str
    action: str
    domain_index: int | None = None
    outcome: str = "unsupported"
    wrf_scheme: str | None = None


class UnsupportedSchemeError(ValueError):
    """Raised when a namelist selects a scheme/feature the port will not run.

    This is the public ``validate_namelist`` failure type. It covers every
    fail-closed outcome -- a recognized-but-unimplemented WRF scheme, an invalid
    WRF option, a mandatory-pairing violation, and an out-of-scope feature
    (WRF-Chem, WRF-Fire, FDDA, stochastic physics, moving nests).
    """

    def __init__(self, selections: list[UnsupportedSelection]) -> None:
        self.selections = tuple(selections)
        super().__init__(_format_error(selections))


class UnsupportedNamelistOption(UnsupportedSchemeError):
    """Backward-compatible alias for the physics/dynamics scheme rejection.

    ``cli.py`` (and the existing test suite) catch ``UnsupportedNamelistOption``;
    keeping it a subclass of :class:`UnsupportedSchemeError` means the broader
    ``validate_namelist`` (which also rejects out-of-scope features) raises a
    type those callers still catch, while new callers can catch the umbrella
    ``UnsupportedSchemeError``.
    """


class NotOperationallyWiredError(UnsupportedSchemeError):
    """Raised by the OPERATIONAL run path for an oracle-backed-but-not-wired scheme.

    The validation layer (:func:`validate_namelist`) intentionally *accepts*
    ``REFERENCE_ONLY`` schemes so parity comparisons and RED oracle-inventory
    work can use the same WRF-option contract. The OPERATIONAL forecast scan,
    however, cannot actually select those schemes. Running them through
    ``gpuwrf run`` would therefore *silently substitute a different scheme* or
    route through a missing kernel, which violates the "no silent wrong path"
    contract. The operational entrypoint
    (:func:`validate_operational_namelist`) fail-closes them loudly, naming the
    requested scheme and the operational alternative.

    It subclasses :class:`UnsupportedSchemeError` -- the exact type the CLI
    ``run`` path catches -- so the operational rejection reaches the user as a
    clean fail-closed error with no traceback.
    """


SUPPORTED_OPTIONS: dict[str, SupportedOption] = {
    # Physics suite wired in runtime.operational_mode and coupling.physics_couplers.
    "mp_physics": SupportedOption(
        key="mp_physics",
        supported_values=frozenset(ACCEPTED_MP_PHYSICS),
        implemented="0=disabled/passive qv, 1=Kessler, 2=Purdue-Lin, 3=WSM3, 4=WSM5, 5=Ferrier, 6=WSM6, 7=Goddard4ice, "
        "8=Thompson, 10=Morrison, 13=SBU-YLin, 14=WDM5, 15=Ferrier advected-rime, 16=WDM6, 17=WRF 4.7.1 legacy NSSL alias of 18, "
        "18=NSSL 2-moment (single domains, supported cumulus; CPU-oracle-qualified, "
        "GPU/coupled-forecast qualification pending), 19=NSSL one-moment with hail/graupel density, "
        "21=NSSL one-moment without hail, 22=NSSL two-moment without hail, 24=WSM7, "
        "26=WDM7, 28=aerosol-aware Thompson, 40=Morrison-aerosol (aercu_opt=0), 97=Goddard GCE",
        action="Use one of the frozen accepted microphysics options; all other MP options remain unsupported.",
    ),
    "cu_physics": SupportedOption(
        key="cu_physics",
        supported_values=frozenset(ACCEPTED_CU_PHYSICS),
        implemented=(
            "0=disabled, 1=Kain-Fritsch, 2=Betts-Miller-Janjic (fp64 savepoint-parity), "
            "7=CAM-ZM, 10=KF-CuP, 11=MSKF (experimental CPU-oracle-qualified; GPU pending, profile limits apply), "
            "3=Grell-Freitas (v0.9.0 GPU-batched jit/vmap scale-aware adapter, savepoint-parity), "
            "4/94/95/96=SAS family (v0.17 fp64 pristine-WRF savepoints staged, "
            "shared JAX endpoint RED vs oracle; fail-closed in the GPU scan), "
            "6=Tiedtke (GPU-operational only with use_flux_advection=True "
            "so RQVFTEN is available); "
            "5=Grell-3D and 93=Grell-Devenyi (v0.3.4 line-faithful JAX ports, scan-wired; "
            "CPU-oracle-qualified vs pristine WRF, GPU qualification pending; cudt=0 only), "
            "14=KIM-SAS, 16=New Tiedtke, "
            "99=previous Kain-Fritsch (with 4/94/95/96=SAS family) are "
            "accepted/reference-only and fail-closed in the operational GPU scan"
        ),
        action=(
            "Use cu_physics=0/1/2/3/5/6/93 for the operational GPU scan; cu=6 requires "
            "active flux-form moisture advection (use_flux_advection=True); "
            "4/14/16/94/95/96/99 remain reference-only until source-specific "
            "WRF parity and scan wiring land."
        ),
    ),
    "bl_pbl_physics": SupportedOption(
        key="bl_pbl_physics",
        supported_values=frozenset(ACCEPTED_BL_PBL_PHYSICS),
        implemented=(
            "0=disabled, 1=YSU, 2=MYJ, 3=GFS, 5=MYNN, 7=ACM2, 8=BouLac, "
            "11=Shin-Hong, 12=GBM, 99=MRF "
            "(GPU-operational, scan-wired); 2=MYJ is the v0.13 jit/vmap-traceable MYJ pair (mandatorily paired with "
            "sf_sfclay_physics=2 Janjic Eta), savepoint-parity-proven; 3=GFS is the v0.17 "
            "jit/vmap-traceable port of phys/module_bl_gfs.F, savepoint-parity-proven; 99=MRF is the "
            "v0.13 jit/vmap-traceable port of phys/module_bl_mrf.F, savepoint-parity-proven. "
            "11=Shin-Hong is the v0.18 JAX/vmap scale-aware PBL port; "
            "12=GBM is the v0.18 JAX/vmap moist prognostic-TKE PBL port, fp64 "
            "parity-green vs the pristine-WRF savepoint oracle. "
            "9=CAM-UW is the v0.3.4 faithful r8 port of module_bl_camuwpbl_driver, "
            "parity-gated against the pristine-WRF oracle proofs/v034/camuw_oracle "
            "(CPU-oracle-qualified; GPU/coupled-forecast qualification pending; coupling caveat: "
            "it receives the held TOTAL RTHRATEN instead of WRF's LW-only RTHRATENLW, so daytime "
            "cloud-top radiative forcing is biased, exact at night). "
            "10=TEMF is CPU-oracle-qualified with mandatory TEMF surface layer 10; "
            "4=QNSE with sf=4 is experimental and scan-wired, including mfshconv=0/1. "
            "16=EEPS and 17=KEPS are accepted/reference-only "
            "v0.18 fp64 pristine-WRF oracle endpoints and fail-close in the "
            "operational GPU scan."
        ),
        action=(
            "Use bl_pbl_physics=0/1/2/3/4/5/7/8/9/10/11/12/99 for the operational GPU scan; 2=MYJ MUST "
            "pair with sf_sfclay_physics=2. "
            "Pair with the matching surface layer (MYNN<->5, ACM2<->7/1, YSU<->1, GFS<->1, "
            "Shin-Hong<->1, GBM<->1, MYJ<->2, MRF<->1, CAM-UW<->Noah-MP or 5). "
            "Use bl_pbl_physics=16/17 only for single-column oracle/reference "
            "comparisons."
        ),
    ),
    "sf_sfclay_physics": SupportedOption(
        key="sf_sfclay_physics",
        supported_values=frozenset(ACCEPTED_SF_SFCLAY_PHYSICS),
        implemented=(
            "0=disabled, 1=revised-MM5, 2=Janjic Eta, 3=NCEP-GFS, 4=experimental QNSE (paired with bl=4), 5=MYNN surface layer, "
            "7=Pleim-Xiu, 91=old-MM5 surface layer (all GPU-operational, scan-wired); "
            "10=TEMF is CPU-oracle-qualified, paired with TEMF PBL 10; GPU/forecast pending. "
            "2=Janjic Eta is the v0.13 jit/vmap-traceable MYJ pair (mandatorily paired "
            "with bl_pbl_physics=2 MYJ), savepoint-parity-proven; 3=NCEP-GFS and 91=old-MM5 "
            "are v0.13 Tier-3 fp64 pristine-WRF oracle-validated surface layers"
        ),
        action=(
            "Use sf_sfclay_physics=0/1/2/3/4/5/7/10/91 for the operational GPU scan; 2=Janjic Eta "
            "MUST pair with bl_pbl_physics=2. "
            "All other sfclay options remain unsupported. "
            "Use the PBL-compatible partner (MYNN-SL 5<->MYNN PBL 5, Pleim-Xiu 7<->ACM2 7, Janjic 2<->MYJ 2)."
        ),
    ),
    "sf_surface_physics": SupportedOption(
        key="sf_surface_physics",
        supported_values=frozenset(ACCEPTED_SF_SURFACE_PHYSICS),
        implemented="0=disabled; 1=thermal-diffusion slab LSM (explicit slab_static bundle); "
        "2=Noah classic (explicit static/land bundle); 4=Noah-MP (set use_noahmp=True); "
        "7=Pleim-Xiu 2-layer ISBA LSM (explicit px_static bundle; pairs with sf_sfclay=7); "
        "3=RUC LSM (explicit ruc_static/ruc_land bundles, MYNN-SL sf_sfclay=5; CPU-oracle-"
        "qualified vs proofs/v034/oracle/ruclsm, GPU/coupled-forecast qualification pending) "
        "-- all scan-wired; slab (1) + Pleim-Xiu (7) are fp64 pristine-WRF "
        "oracle-validated (physics.lsm_slab / physics.lsm_pleim_xiu). 8=SSiB is an "
        "experimental opt-in with prepared three-layer soil, revised-MM5/YSU, mp8, "
        "SW1/LW4; literal CPU-oracle-qualified, compiled residuals disclosed, "
        "GPU/coupled-forecast pending (proofs/s2_ssib). 5=CLM4 and 6=CTSM are "
        "CAM/CLM/CTSM ARCHITECTURE-BOUNDARY options that fail closed with a named reason "
        "(CLM4 is ~61.5k-LOC with a global clmtype + external surface-dataset; CTSM is the "
        "external CESM land model via LILAC) -- a faithful oracle is multi-session, carried "
        "to the v1.0 ADR; never a silent substitution.",
        action="Use sf_surface_physics=4 (Noah-MP), 2 (Noah classic), 1 (slab, with an "
        "explicit slab_static SlabStaticBundle), or 7 (Pleim-Xiu, with an explicit "
        "px_static PleimXiuStaticBundle), or 3 (RUC, with explicit ruc_static/ruc_land) for "
        "the operational scan; 8 (experimental SSiB) requires prepared soil and revised-MM5/YSU, SW1/LW4; 5 (CLM4) and 6 (CTSM) are "
        "documented v1.0 architecture-boundary (fail-closed, carried to the CAM/CLM/CTSM "
        "ADR); all other land-surface options remain unsupported.",
    ),
    "ra_sw_physics": SupportedOption(
        key="ra_sw_physics",
        supported_values=frozenset(ACCEPTED_RA_SW_PHYSICS),
        implemented="0=disabled, 1=Dudhia shortwave (Stephens-1984 broadband; GPU-operational, "
        "scan-wired held-rate RTHRATEN with RRTMG/classic-RRTM longwave), 2=GSFC (Chou-Suarez) "
        "shortwave (multi-band delta-Eddington; GPU-operational, jit/vmap-traceable port of "
        "phys/module_ra_gsfcsw.F scan-wired via gsfc_sw_theta_tendency), 4=RRTMG shortwave "
        "(GPU-operational; the operational radiation slot runs RRTMG SW+LW), "
        "3=CAM SW (v0.3.4: physics.ra_cam port of phys/module_ra_cam.F scan-wired via "
        "coupling.cam_radiation; CPU-oracle-qualified, GPU qualification pending), "
        "5=New Goddard SW (experimental paired5/5 MP8/use_mp_re=1, CPU-oracle-qualified), "
        "7=FLG/UCLA SW, and 99=GFDL-Eta SW "
        "(v0.18 REFERENCE-ONLY: exact-driver real-WRF oracles staged, but no "
        "faithful JAX kernel or operational scan wiring yet)",
        action="Use ra_sw_physics=4 (RRTMG SW+LW), 1 (Dudhia SW + RRTMG/RRTM LW), 2 (GSFC SW + "
        "RRTMG/RRTM LW) or 3 (CAM) for the operational SW path, or 0 when radiation is disabled. "
        "Use paired5/5 with MP8/use_mp_re=1 for experimental Goddard. ra_sw=7/99 are accepted for reference/parity development only and fail-close in "
        "the operational scan; the surface SWDOWN/flux history diagnostics remain RRTMG-derived. "
        "ra_sw=14 (RRTMG-K) and 24 (fast RRTMG) are compiled-out of standard WRF "
        "(configure.wrf BUILD_RRTMK=0 / BUILD_RRTMG_FAST=0) and cannot run even in unmodified WRF.",
    ),
    "ra_lw_physics": SupportedOption(
        key="ra_lw_physics",
        supported_values=frozenset(ACCEPTED_RA_LW_PHYSICS),
        implemented="0=disabled, 1=classic AER RRTM longwave (16-band k-distribution; "
        "GPU-operational, scan-wired held-rate RTHRATEN via the JAX-traceable "
        "physics.ra_lw_rrtm_jax port of phys/module_ra_rrtm.F), 4=RRTMG longwave "
        "(GPU-operational, default), 3=CAM LW (v0.3.4: physics.ra_cam port scan-wired via "
        "coupling.cam_radiation; CPU-oracle-qualified, GPU qualification pending), "
        "5=GSFC/Goddard NUWRF longwave (experimental paired5/5 MP8/use_mp_re=1, CPU-oracle-qualified), "
        "7=FLG/UCLA LW, and 99=GFDL-Eta longwave (v0.18 REFERENCE-ONLY: "
        "exact-driver real-WRF oracles staged; no faithful JAX kernel or operational scan wiring yet)",
        action="Use ra_lw_physics=4 (RRTMG), 1 (classic RRTM) or 3 (CAM) for the operational LW path, "
        "or paired5/5 with MP8/use_mp_re=1 for experimental Goddard, or 0 when radiation is disabled. ra_lw=7/99 "
        "are reference-only and fail-close in the operational scan. SW and LW are "
        "selected independently; the surface GLW history diagnostic remains RRTMG-derived. "
        "ra_lw=14 (RRTMG-K) and 24 (fast RRTMG) are compiled-out of standard WRF "
        "(configure.wrf BUILD_RRTMK=0 / BUILD_RRTMG_FAST=0) and cannot run even in unmodified WRF.",
    ),
    "use_mp_re": SupportedOption(
        key="use_mp_re", supported_values=frozenset({0, 1}),
        implemented="1=held MP radii (physics_init has_reqc/i/s: mp 3/4/6/8/14/16/18/22/24/26/28) with GPUWRF_RRTMG_MP_RE=1; 0=disable the candidate radius path",
        action="Use use_mp_re=1 (WRF default) with mp_physics=8 for the MP_RE candidate.",
    ),
    # Runtime/dynamics controls exposed by OperationalNamelist.
    "rk_order": SupportedOption(
        key="rk_order",
        supported_values=frozenset({3}),
        implemented="3=WRF RK3 outer loop",
        action="Use rk_order=3; other RK orders are not wired faithfully.",
    ),
    "diff_6th_opt": SupportedOption(
        key="diff_6th_opt",
        supported_values=frozenset({0, 2}),
        implemented="0=off, 2=WRF monotonic sixth-order horizontal filter",
        action="Use diff_6th_opt=2 for the operational filter or 0 when disabled.",
    ),
    "diff_opt": SupportedOption(
        key="diff_opt",
        supported_values=frozenset({0, 1, 2}),
        implemented="0=off, 1=coordinate-surface (eta) horizontal diffusion, "
        "2=physical-level constant-K / 3-D LES turbulence diffusion path",
        action="Use diff_opt=1/km_opt=4 for the real-data default 2-D Smagorinsky, "
        "diff_opt=2/km_opt=1 for constant-K, diff_opt=2/km_opt=2/3/5 for "
        "3-D TKE/Smagorinsky/SMS turbulence, or 0.",
    ),
    "km_opt": SupportedOption(
        key="km_opt",
        supported_values=frozenset({0, 1, 2, 3, 4, 5}),
        implemented="0=off, 1=constant-K coefficient (programmatic only; gpuwrf run refuses it), "
        "2=prognostic 3-D TKE (CPU-oracle-qualified, experimental), "
        "3=3-D Smagorinsky, 4=2-D Smagorinsky horizontal eddy viscosity; "
        "5=SMS-3DTKE (CPU-oracle-qualified, experimental; diff_opt=2/PBL=0/sfclay=0 or 5)",
        action="Use km_opt=4 with diff_opt=1 for the real-data default 2-D "
        "Smagorinsky, km_opt=2/3/5 with diff_opt=2 for 3-D TKE/Smagorinsky/SMS, or 0.",
    ),
    "w_damping": SupportedOption(
        key="w_damping",
        supported_values=frozenset({0, 1}),
        implemented="0=off, 1=WRF vertical-CFL w damping",
        action="Use w_damping=1 or 0; no other WRF w_damping option is implemented.",
    ),
    "damp_opt": SupportedOption(
        key="damp_opt",
        supported_values=frozenset({0, 3}),
        implemented="0=off, 3=upper-level Rayleigh w damping",
        action="Use damp_opt=3 for the implemented Rayleigh path or 0.",
    ),
    "sf_urban_physics": SupportedOption(
        key="sf_urban_physics",
        supported_values=frozenset(ACCEPTED_SF_URBAN_PHYSICS),
        implemented=(
            "0=disabled; 1=single-layer UCM (SLUCM) with sf_surface_physics=4 Noah-MP "
            "(CPU-oracle-qualified; GPU/coupled-forecast qualification pending); "
            "2=BEP and 3=BEP+BEM are G3 REFERENCE-ONLY "
            "(WRF source/object/Registry oracle inventory present, no faithful "
            "JAX kernel or numerical WRF single-column parity claim)"
        ),
        action=(
            "Use sf_urban_physics=0 or 1 (with sf_surface_physics=4) for operational runs; "
            "use 2/3 only for reference/oracle development until the urban state carry and "
            "faithful BEP/BEM kernels land."
        ),
    ),
    "sf_lake_physics": SupportedOption(
        key="sf_lake_physics",
        supported_values=frozenset(ACCEPTED_SF_LAKE_PHYSICS),
        implemented=(
            "0=disabled; 1=WRF lake model (lakeini/LakeMain), CPU-oracle-qualified; "
            "GPU/coupled-forecast qualification pending"
        ),
        action=(
            "sf_lake_physics=1 requires Noah-MP/Thompson/MYNN, WRF lake inputs and "
            "lake-depth data or use_lakedepth=0; see docs/lake-model.md."
        ),
    ),
}


def validate_supported_namelist(config: Any) -> None:
    """Raise if ``config`` selects a physics/dynamics option outside the registry.

    ``config`` may be a flat mapping, a nested WRF-style mapping such as
    ``{"physics": {"mp_physics": [8, 8]}}``, an object/dataclass with matching
    attributes, or a path to a simple WRF namelist file. Missing keys are ignored:
    this checker validates selected options, not namelist completeness.
    """

    config_obj = _coerce_config(config)
    failures: list[UnsupportedSelection] = []
    for key, spec in SUPPORTED_OPTIONS.items():
        found = _lookup(config_obj, key)
        if found is None:
            continue
        location, raw = found
        values = _domain_values(raw)
        for idx, value in enumerate(values):
            normalized = _normalize_value(value)
            if normalized in spec.supported_values:
                continue
            outcome, wrf_scheme = _classify_rejection(key, normalized)
            failures.append(
                UnsupportedSelection(
                    key=key,
                    location=location,
                    value=normalized,
                    supported_values=_sorted_supported_values(spec.supported_values),
                    implemented=spec.implemented,
                    action=spec.action,
                    domain_index=idx + 1 if len(values) > 1 else None,
                    outcome=outcome,
                    wrf_scheme=wrf_scheme,
                )
            )
    failures.extend(_myj_pairing_failures(config_obj))
    failures.extend(_temf_pairing_failures(config_obj))
    failures.extend(_qnse_pairing_failures(config_obj))
    if failures:
        raise UnsupportedNamelistOption(failures)


def validate_namelist(config: Any) -> None:
    """Validate a WRF namelist against the full support catalog; raise or pass.

    This is the public entrypoint. It runs the scheme/dynamics support check
    (recognized-but-unimplemented, invalid WRF options, mandatory pairings) AND
    the out-of-scope feature-switch check (WRF-Chem, WRF-Fire, WRF-Hydro, FDDA,
    stochastic physics, moving nests, coupled ocean, SST update).

    ``config`` may be a flat mapping, a nested WRF-style mapping
    (``{"physics": {"mp_physics": [8, 8]}}``), an object/dataclass with matching
    attributes, or a path / text of a Fortran ``namelist.input``.

    Raises :class:`UnsupportedSchemeError` (whose subclass
    :class:`UnsupportedNamelistOption` is what the CLI catches) with a message
    that names every offending option, the reason it will not run, and the
    supported alternative / transition recipe. Implemented (and accepted
    reference-only) selections pass silently.

    The cumulus/PBL CADENCE keys (``cudt``/``bldt``) are deliberately NOT
    fail-closed when positive: the GPU port runs those physics every dynamics
    step (a conservative approximation of the requested sub-stepping cadence, not
    a wrong-scheme substitution), so the namelist PASSES. The approximation is
    surfaced as a non-fatal warning via :func:`collect_namelist_warnings`, which
    the CLI prints before launching the forecast.
    """

    config_obj = _coerce_config(config)
    oos_failures = _out_of_scope_failures(config_obj)
    control_failures = _recognized_control_failures(config_obj)
    extra_failures = (oos_failures + control_failures + _cldovrlp_failures(config_obj)
                      + _aercu_opt_failures(config_obj))
    try:
        validate_supported_namelist(config_obj)
    except UnsupportedNamelistOption as exc:
        # Merge scheme failures with out-of-scope feature + recognized-control
        # failures into one actionable report rather than failing on only the
        # first category.
        raise UnsupportedSchemeError(list(exc.selections) + extra_failures) from None
    if extra_failures:
        raise UnsupportedSchemeError(extra_failures)


def validate_operational_namelist(config: Any) -> None:
    """Validate a namelist for an OPERATIONAL ``gpuwrf run``; raise or pass.

    This is the strict entrypoint the forecast CLI uses. It runs the full
    :func:`validate_namelist` check first (so recognized-but-unimplemented
    schemes, invalid WRF options, mandatory pairings and out-of-scope features
    fail closed exactly as before) AND THEN additionally rejects any selected
    scheme classified ``REFERENCE_ONLY`` by :func:`scheme_catalog.classify_scheme`.

    Why the extra rejection: ``REFERENCE_ONLY`` schemes have oracle-backed
    reference evidence (GREEN where claimed, or an explicitly measured RED gap)
    and so are accepted by :func:`validate_namelist` for a reference /
    single-column comparison or RED oracle-inventory work -- but they are NOT
    wired into the operational GPU scan. Some adapters have no operational scan
    carry-path (MYJ PBL + Janjic Eta surface layer were REFERENCE_ONLY through
    v0.12.0 but are now IMPLEMENTED via the v0.13 traceable pair). Running
    a reference-only scheme through ``gpuwrf run`` would therefore *silently
    substitute a different scheme* or route through a missing kernel, which
    violates the v0.12.0 "no silent wrong path" Scope-A contract. So the
    operational path refuses them loudly, naming the operational scheme the scan
    would actually run and the supported alternative.

    The authoritative IMPLEMENTED-vs-not decision comes from
    :class:`scheme_catalog.SupportStatus`: only ``IMPLEMENTED`` selections run on
    the operational scan; ``REFERENCE_ONLY`` (handled here) and
    ``RECOGNIZED_FAIL_CLOSED`` / ``OUT_OF_SCOPE`` (already rejected by
    :func:`validate_namelist`) all fail closed for an operational run.

    Raises :class:`NotOperationallyWiredError` (a subclass of
    :class:`UnsupportedSchemeError`) when a reference-only scheme is selected;
    re-raises :class:`UnsupportedSchemeError` from the base validation otherwise.
    Does NOT change :func:`validate_namelist`'s acceptance of reference-only
    schemes -- other (validation-layer) callers rely on that.
    """

    config_obj = _coerce_config(config)
    # Digital filter initialization: the native single-domain driver runs WRF's
    # DFI (dfi_opt=1/2/3, gpuwrf.dfi); every DFI configuration the port
    # does not reproduce are refused here, before any model import.
    failures = _dfi_admission_failures(config_obj) + _solar_admission_failures(config_obj) + _goddard_admission_failures(config_obj) + _mpmisc_admission_failures(config_obj) + _flg_gfdl_admission_failures(config_obj) + _cammg_admission_failures(config_obj) + _km3_admission_failures(config_obj)
    from gpuwrf.coupling.cuplus_config import selection_errors as cuplus_errors
    failures.extend(UnsupportedSelection(key='cu_physics',location='physics',value='cuplus',supported_values=(7,10,11),
        implemented=reason,action='Use a qualified cuplus producer/radiation profile; see docs/CUMULUS_CUPLUS.md.',
        outcome='recognized_control_not_wired',wrf_scheme='MSKF/KF-CuP/CAM-ZM') for reason in cuplus_errors(config_obj))
    if isinstance(config_obj,Mapping) and config_obj.get("stoch"):
        from gpuwrf.io.stochastic_config import stochastic_refusals
        failures.extend(UnsupportedSelection(key="stoch",location="&stoch",value="selected",
            supported_values=(0,1),implemented=reason,
            action="Use the qualified stochastic controls and WRF RK source-coupling profile.",
            outcome="recognized_control_not_wired",wrf_scheme="WRF stochastic physics")
            for reason in stochastic_refusals(config_obj))
        from gpuwrf.io.nesting_options import resolve_nesting_options
        from gpuwrf.io.stochastic_config import stochastic_nesting_refusals
        try:
            _feedback = resolve_nesting_options(config_obj)["feedback"]
        except ValueError:
            _feedback = False  # the nesting-option check reports the invalid value itself
        _max = (config_obj.get("domains") or {}).get("max_dom", 1)
        _max = int(_max[0] if isinstance(_max, (list, tuple)) else _max)
        failures.extend(UnsupportedSelection(key="stoch",location="&stoch",value="nested",
            supported_values=(0,1),implemented=reason,
            action="Use one-way nesting (feedback=0) and B=1 members for stochastic physics.",
            outcome="recognized_control_not_wired",wrf_scheme="WRF stochastic physics")
            for reason in stochastic_nesting_refusals(config_obj, tuple(f"d{i:02d}" for i in range(1, _max+1)),
                                                      feedback=_feedback, batch_size=1))
    from gpuwrf.io.windfarm import windfarm_errors
    selected_wind = _lookup(config_obj,'windfarm_opt')
    if selected_wind is not None:
        for index,option in enumerate(_domain_values(selected_wind[1])):
            messages = windfarm_errors(config_obj,f'd{index+1:02d}')
            if messages:
                failures.append(UnsupportedSelection(key='windfarm_opt',location=selected_wind[0],value=option,
                    supported_values=(0,1),implemented='; '.join(messages),action='Use Fitch=1 with MYNN=5/5 or disable windfarm_opt=0.',
                    outcome='recognized_control_not_wired',wrf_scheme='Fitch wind farm'))
    if failures:
        raise UnsupportedSchemeError(failures)
    # First: the existing full support + out-of-scope check (unchanged behavior).
    validate_namelist(config_obj)
    # Then: the operational-only strictness -- reject reference-only selections.
    ref_only = _reference_only_failures(config_obj)
    if ref_only:
        raise NotOperationallyWiredError(ref_only)


#: mp codes whose extra species (NSSL numbers/volumes, hail qh, Morrison-aero Ns/Ng/Nc, Thompson-aero
#: Nc/nwfa/nifa) have root forcing. Extra nest records/consumers are staged, but
#: nest-start interpolation and the real two-domain WRF gate remain incomplete.
def _goddard_admission_failures(config):
    from gpuwrf.io.goddard_config import selection_errors
    errors=selection_errors(config)
    return [UnsupportedSelection(key='ra_lw_physics',location='physics',value=5,supported_values=(5,),
        implemented=error,action='Use paired Goddard5/5 with Thompson8 and use_mp_re=1, no aerosol/topography feedback.',
        outcome='recognized_control_not_wired',wrf_scheme='Goddard radiation') for error in errors]

def _mpmisc_admission_failures(config):
    mp = _lookup(config, "mp_physics")
    cu = _lookup(config, "cu_physics")
    if mp is None:
        return []
    values = [_normalize_value(v) for v in _domain_values(cu[1])] if cu else [0]
    values = values or [0]
    failures = []
    for index, value in enumerate(_domain_values(mp[1])):
        option = _normalize_value(value)
        chosen = values[min(index, len(values) - 1)]
        if option in (5, 15) and chosen != 0:
            failures.append(UnsupportedSelection(
                key="cu_physics", location=cu[0], value=chosen, supported_values=(0,),
                implemented="Ferrier mp_physics=5/15 has no QSNOW category (WRF F_QS=false); the cumulus caller for this package is not qualified",
                action="Use cu_physics=0 with Ferrier.", outcome="recognized_control_not_wired",
                wrf_scheme="Ferrier cumulus package caller"))
        if option == 38:
            aerosol = _lookup(config, "aer_init_opt")
            if aerosol is not None:
                modes = _domain_values(aerosol[1])
                chosen_aerosol = _normalize_value(modes[min(index, len(modes)-1)]) if modes else 0
                if chosen_aerosol != 0:
                    failures.append(UnsupportedSelection(
                        key="aer_init_opt", location=aerosol[0], value=chosen_aerosol, supported_values=(0,),
                        implemented="Thompson-hail mp_physics=38 qualifies the default flow-dependent aerosol boundary only",
                        action="Use aer_init_opt=0 with mp_physics=38.",
                        outcome="recognized_control_not_wired", wrf_scheme="Thompson-hail aerosol root boundary"))
        if option not in (5, 7, 11, 15, 38):
            continue
        key = "have_bcs_moist" if option == 7 else "have_bcs_scalar"
        boundary = _lookup(config, key)
        if boundary is not None:
            flags = _domain_values(boundary[1])
            if flags and _normalize_value(flags[min(index, len(flags) - 1)]) == 1:
                failures.append(UnsupportedSelection(
                    key=key, location=boundary[0], value=True, supported_values=(False,),
                    implemented=f"mp_physics={option} extra-species specified records are not qualified; its WRF flow-dependent boundary path is implemented",
                    action=f"Use {key}=.false. (WRF default) with mp_physics={option}.",
                    outcome="recognized_control_not_wired", wrf_scheme="microphysics extra-species root boundary"))
    return failures


def _km3_admission_failures(config):
    from gpuwrf.io.km3_config import selection_errors
    return [UnsupportedSelection(
        key='km_opt', location='dynamics', value=3, supported_values=(),
        implemented=error, action='Set JAX_PLATFORMS=cpu or use default diff_opt=1/km_opt=4.',
        outcome='recognized_control_not_wired', wrf_scheme='CPU-only 3-D Smagorinsky')
        for error in selection_errors(config)]


def _cammg_admission_failures(config):
    from gpuwrf.io.cammg_config import selection_errors
    return [UnsupportedSelection(
        key='mp_physics', location='physics', value=11, supported_values=(),
        implemented=error, action='Set JAX_PLATFORMS=cpu or select another microphysics scheme.',
        outcome='recognized_control_not_wired', wrf_scheme='CPU-only CAM-MG')
        for error in selection_errors(config)]


def _flg_gfdl_admission_failures(config):
    from gpuwrf.io.flg_gfdl_config import selection_errors
    return [UnsupportedSelection(key='ra_lw_physics',location='physics',value=7,supported_values=(7,99),
        implemented=error,action='Use JAX_PLATFORMS=cpu with the qualified paired caller, or RRTMG4/4 on GPU.',
        outcome='recognized_control_not_wired',wrf_scheme='Experimental CPU-only FLG/GFDL')
        for error in selection_errors(config)]


def _solar_admission_failures(config):
    """Solar caller scope, shared by strict admission and the honour report."""
    selected = _lookup(config, "swint_opt")
    if selected is None or _normalize_value(_domain_values(selected[1])[0]) != 1:
        return []
    bad = []
    for key, expected in (("ra_sw_physics", 4), ("ra_lw_physics", 4), ("sf_surface_physics", 4),
                          ("solar_opt", 0), ("topo_shading", 0), ("slope_rad", 0),
                          ("sf_urban_physics", 0), ("sf_lake_physics", 0), ("use_adaptive_time_step", 0)):
        found = _lookup(config, key)
        if found is not None and any(_normalize_value(value) != expected for value in _domain_values(found[1])):
            bad.append(f"{key}={expected}")
    if not bad:
        return []
    return [UnsupportedSelection(
        key="swint_opt", location=selected[0], value=1, supported_values=(0, 1),
        implemented="Solar swint_opt=1 requires RRTMG4/4, Noah-MP4 and fixed dt; required " + ", ".join(bad),
        action="Use the qualified native RRTMG/Noah-MP configuration or set swint_opt=0.",
        outcome="recognized_control_not_wired", wrf_scheme="WRF-Solar interpolation")]


SINGLE_DOMAIN_ONLY_MP: frozenset[int] = frozenset({5, 7, 9, 10, 14, 15, 16, 17, 18, 19, 21, 22, 24, 26, 28, 38, 40, 50, 51, 52, 53, 55, 56, 11})

_NESTED_MP_FAILURES = {
    19: "NSSL one-moment nested forecast qualification has not been completed",
    21: "NSSL one-moment nested forecast qualification has not been completed",
    18: "the WN3 N2/N6 numerical comparison passes, but pre-mask qi/Ni negative events lack WRF operator justification",
    24: "the WN3 N2/N6 numerical comparison passes, but pre-mask qi/Ni negative events lack WRF operator justification",
    28: "initial QNWFA/QNIFA fail the WN3 original-WRF comparison in both domains",
    40: "parent boundary QNCLOUD fails the WN3 N2/N6 comparison and pre-mask Nc negative events remain unresolved",
}



def refuse_single_domain_mp_nested(config: Any, max_dom: int) -> None:
    """Refuse a live-nested run (``max_dom > 1``) with a single-domain-only microphysics, pre-JAX."""

    if int(max_dom) <= 1:
        return
    config_obj = _coerce_config(config)
    mp = _lookup(config_obj, "mp_physics")
    if mp is None:
        return
    bad = [_normalize_value(v) for v in _domain_values(mp[1])[: int(max_dom)]
           if _normalize_value(v) in SINGLE_DOMAIN_ONLY_MP]
    if bad:
        raise NotOperationallyWiredError([
            UnsupportedSelection(
                key="mp_physics",
                location=mp[0],
                value=bad[0],
                supported_values=tuple(sorted(SINGLE_DOMAIN_ONLY_MP)),
                implemented=f"mp_physics={bad[0]} runs on single domains only: parent->child coupled qualification remains refused because "
                + _NESTED_MP_FAILURES.get(bad[0], "the two-domain CPU-WRF gate has not passed")
                + ". Extra-species forcing and nest-start component tests do not replace this gate. "
                "Measured evidence: proofs/s1_mpfix/NESTED_ADMISSION.json.",
                action="Run a single domain (no --max-dom > 1), or use mp_physics=8 (Thompson) for nested runs.",
                outcome="recognized_control_not_wired",
                wrf_scheme="nest lateral forcing of extra microphysics species",
            )
        ])


def refuse_unqualified_nested_options(config: Any, max_dom: int) -> None:
    """Keep O2 single-domain qualification from implicitly admitting live nests.

    These are port evidence gaps, not WRF restrictions. Grid FDDA is per-domain
    and has independent coefficient/file/cursor binding; it is audited separately.
    """
    if int(max_dom) <= 1:
        return
    config_obj = _coerce_config(config)
    failures = []
    for key, code, reason in (
        ("sf_surface_physics", 3, "RUC land initialization, held diagnostics and child-subcycle evolution"),
        ("sf_urban_physics", 1, "SLUCM canopy initialization/carry and child surface fluxes"),
        ("km_opt", 2, "LES TKE advection/diffusion at live-child boundary and halo cells"),
        ("km_opt", 5, "SMS TKE and nonlocal carry at live-child boundary and halo cells"),
    ):
        found = _lookup(config_obj, key)
        if found is None:
            continue
        for domain, raw in enumerate(_domain_values(found[1])[:int(max_dom)], 1):
            if _normalize_value(raw) == code:
                failures.append(UnsupportedSelection(
                    key=key, location=found[0], value=code, supported_values=(),
                    implemented=f"d{domain:02d}: {reason} are CPU-qualified only for a single domain; "
                    "the two-domain CPU-WRF comparison is missing in this port",
                    action="Run a single domain or disable this option for the nested run.",
                    outcome="recognized_control_not_wired", wrf_scheme="unqualified live-nest option",
                ))
    if failures:
        raise NotOperationallyWiredError(failures)


def collect_namelist_warnings(config: Any) -> list[str]:
    """Return non-fatal approximation warnings for a namelist (never raises).

    A naive user pointing the standalone ``gpuwrf run`` at a real WRF
    ``namelist.input`` must not be REJECTED for the cumulus/PBL cadence keys.
    ``bldt`` asks the port to sub-step PBL physics every N minutes, but
    the GPU port runs PBL EVERY dynamics step. A positive ``bldt`` is NOT a fail-closed rejection
    (handled by :func:`validate_namelist`); instead it surfaces here as a
    WARNING string naming the approximation, while the run proceeds. The CLI
    prints these to stderr before launching the forecast. KF ``cudt`` follows
    WRF STEPCU and does not generate an approximation warning.

    Each warning is a single human-readable line. ``config`` accepts the same
    forms as :func:`validate_namelist` (flat/nested mapping, dataclass, or a
    namelist path/text). Returns ``[]`` when nothing is approximated. This
    function NEVER raises -- it is purely advisory; the fail-closed rejections
    remain the validators' job.
    """

    try:
        config_obj = _coerce_config(config)
    except Exception:  # noqa: BLE001 - advisory only; let validators report IO/parse errors
        return []

    warnings: list[str] = []
    for key in sorted(APPROXIMATED_CONTROL_KEYS):
        found = _lookup(config_obj, key)
        if found is None:
            continue
        location, raw = found
        values = _domain_values(raw)
        multi = len(values) > 1
        for idx, value in enumerate(values):
            support = classify_control(key, value)
            if support is None or support.status is not SupportStatus.RECOGNIZED_APPROXIMATED:
                continue
            domain = f" (domain {idx + 1})" if multi else ""
            # ``support.reason`` carries the catalog's approximation note.
            warnings.append(f"{location}{domain}: {support.reason}")
    if _rrtmg_selected(config_obj) and not _rrtmg_maxrand_resolved():
        location, values = _cldovrlp_values(config_obj)
        if 2 in values:
            warnings.append(
                f"{location}: cldovrlp=2 (WRF maximum-random McICA overlap) runs RANDOM overlap "
                "(cldovrlp=1) unless GPUWRF_RRTMG_MAXRAND=1"
            )
    return warnings


def _reference_only_failures(config: Any) -> list[UnsupportedSelection]:
    """Find selected physics schemes that are REFERENCE_ONLY (operational reject).

    Scans every gated physics key for a per-domain value that
    :func:`scheme_catalog.classify_scheme` classifies as ``REFERENCE_ONLY`` and
    builds a fail-closed selection whose message names the scheme, why it is not
    operationally wired, and the operational alternative (e.g. RRTMG=4). Only
    physics keys can be reference-only; dynamics keys never are, so scanning the
    physics keys is sufficient (and harmless if a dynamics key were scanned).
    """

    failures: list[UnsupportedSelection] = []
    for key in _REFERENCE_ONLY_CANDIDATE_KEYS:
        found = _lookup(config, key)
        if found is None:
            continue
        location, raw = found
        values = _domain_values(raw)
        for idx, value in enumerate(values):
            normalized = _normalize_value(value)
            if not isinstance(normalized, int):
                continue
            support = classify_scheme(key, normalized)
            if support.status is not SupportStatus.REFERENCE_ONLY:
                continue
            failures.append(
                UnsupportedSelection(
                    key=key,
                    location=location,
                    value=normalized,
                    # List only the truly operationally-wired (IMPLEMENTED) codes
                    # -- NOT the full accepted matrix (which includes this very
                    # reference-only code) -- so the message is honest about what
                    # the operational scan can actually run.
                    supported_values=_operationally_wired_values(key),
                    implemented=(
                        f"REFERENCE-ONLY (oracle-backed, NOT operationally wired): "
                        f"{support.reason}"
                    ),
                    action=support.alternative,
                    domain_index=idx + 1 if len(values) > 1 else None,
                    outcome="reference_only_not_operational",
                    wrf_scheme=support.wrf_name,
                )
            )
    return failures


def _operationally_wired_values(key: str) -> tuple[Any, ...]:
    """Sorted IMPLEMENTED (operationally scan-wired) codes for ``key``.

    Derived from the authoritative catalog so a reference-only code is never
    listed as operationally available.
    """

    impl = {
        support.code
        for support in iter_full_catalog()
        if support.key == key and support.status is SupportStatus.IMPLEMENTED
    }
    return _sorted_supported_values(frozenset(impl))


# Physics keys whose catalog classifies at least one code REFERENCE_ONLY.
# Derived from scheme_catalog.iter_full_catalog() so this stays in lockstep with
# the authoritative catalog classifications -- never hard-coded here. If a future
# scheme becomes reference-only (or graduates to implemented), this set tracks it
# automatically.
_REFERENCE_ONLY_CANDIDATE_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        support.key
        for support in iter_full_catalog()
        if support.status is SupportStatus.REFERENCE_ONLY
    )
)


def _out_of_scope_failures(config: Any) -> list[UnsupportedSelection]:
    """Scan every namelist section for truthy out-of-scope feature switches."""

    failures: list[UnsupportedSelection] = []
    for key in OUT_OF_SCOPE_FEATURE_KEYS:
        found = _lookup(config, key)
        if found is None:
            continue
        location, raw = found
        # A feature is "on" if ANY per-domain value is truthy.
        values = _domain_values(raw)
        support: SchemeSupport | None = None
        for value in values:
            support = classify_feature_switch(key, value)
            if support is not None:
                break
        if support is None:
            continue
        failures.append(
            UnsupportedSelection(
                key=support.key,
                location=location,
                value=support.code,
                supported_values=(),
                implemented=f"OUT OF SCOPE: {support.wrf_name}",
                action=support.alternative,
                outcome="out_of_scope",
                wrf_scheme=support.wrf_name,
            )
        )
    return failures


def _recognized_control_failures(config: Any) -> list[UnsupportedSelection]:
    """Scan recognized non-enumerated control keys for unwired values.

    Covers the dynamics/advection switches (``gwd_opt``, ``moist_adv_opt``,
    ``scalar_adv_opt``, the ``*_adv_order`` family), the implemented radiation
    slope/topo controls (``slope_rad``, ``topo_shading``), the MYNN-EDMF
    sub-option family (``bl_mynn_*``) and the physics-cadence intervals
    (``radt``/``bldt``/``cudt``). A recognized key set to a value the operational
    scan does NOT wire fails closed with the catalog's named reason; an
    operationally-wired value passes silently. Keys not in the recognized-control
    namespace are left to the rest of the validator (and silent-pass otherwise).
    """

    failures: list[UnsupportedSelection] = []
    for key in sorted(RECOGNIZED_CONTROL_KEYS | IMPLEMENTED_CONTROL_KEYS):
        found = _lookup(config, key)
        if found is None:
            continue
        location, raw = found
        values = _domain_values(raw)
        for idx, value in enumerate(values):
            support = classify_control(key, value)
            if support is None or support.status is not SupportStatus.RECOGNIZED_FAIL_CLOSED:
                continue
            failures.append(
                UnsupportedSelection(
                    key=key,
                    location=location,
                    value=support.code,
                    supported_values=(),
                    # Carry the catalog's named reason so the user sees *why* the
                    # value is not wired (not just the control label).
                    implemented=support.reason,
                    action=support.alternative,
                    domain_index=idx + 1 if len(values) > 1 else None,
                    outcome="recognized_control_not_wired",
                    wrf_scheme=support.wrf_name,
                )
            )
    return failures


_CLDOVRLP_DEFAULT = 2  # Registry.EM_COMMON cldovrlp (namelist physics, scalar)


def _rrtmg_maxrand_resolved() -> bool:
    """GPUWRF_RRTMG_MAXRAND as the traced McICA resolved it (env if not yet imported)."""
    module = sys.modules.get("gpuwrf.kernels.rad_mcica")
    if module is not None and hasattr(module, "_MAXRAND"):
        return bool(module._MAXRAND)
    return os.environ.get("GPUWRF_RRTMG_MAXRAND", "0") == "1"


def _rrtmg_selected(config: Any) -> bool:
    """RRTMG LW or SW on any domain; absent radiation keys mean the port's RRTMG default."""
    found = [f for f in (_lookup(config, "ra_lw_physics"), _lookup(config, "ra_sw_physics")) if f is not None]
    if not found:
        return True
    return any(_normalize_value(v) == 4 for _, raw in found for v in _domain_values(raw))


def _cldovrlp_values(config: Any) -> tuple[str, list[Any]]:
    found = _lookup(config, "cldovrlp")
    if found is None:
        return "physics.cldovrlp (WRF default)", [_CLDOVRLP_DEFAULT]
    location, raw = found
    return location, [_normalize_value(v) for v in _domain_values(raw)]


def _cldovrlp_failures(config: Any) -> list[UnsupportedSelection]:
    """RRTMG McICA cloud overlap: WRF cldovrlp 1 (random) or 2 (maximum-random) only.

    WRF icld = cldovrlp (module_ra_rrtmg_lw.F:12164, _sw.F:10762; a single
    &physics value). The port runs icld=1, or icld=2 under GPUWRF_RRTMG_MAXRAND=1
    (see :func:`apply_cldovrlp`); 0/3/4/5 fail closed instead of a silent overlap.
    """

    if not _rrtmg_selected(config):
        return []
    location, values = _cldovrlp_values(config)
    if all(value in (1, 2) for value in values) and len(set(values)) == 1:
        return []
    return [
        UnsupportedSelection(
            key="cldovrlp",
            location=location,
            value=values[0] if len(values) == 1 else tuple(values),
            supported_values=(1, 2),
            implemented="one cldovrlp: 1=random or 2=maximum-random (GPUWRF_RRTMG_MAXRAND=1) McICA overlap",
            action="Use cldovrlp=2 (WRF default; with GPUWRF_RRTMG_MAXRAND=1) or cldovrlp=1.",
            outcome="recognized_control_not_wired",
            wrf_scheme="RRTMG McICA cloud overlap",
        )
    ]


def _aercu_opt_failures(config: Any) -> list[UnsupportedSelection]:
    """CESM-NCSU prescribed aerosol (aercu_opt>0): fail closed, never ignored.

    WRF share/module_check_a_mundo.F:1407 makes aercu_opt>0 fatal unless
    cu_physics=11 (multi-scale KF, not ported) AND mp_physics=40; it also needs
    the CESM_RCP4.5_Aerosol_Data.dat climatology (module_physics_init.F:5545).
    The port runs mp_physics=40 at aercu_opt=0 (constant droplets) only.
    """

    found = _lookup(config, "aercu_opt")
    if found is None:
        return []
    location, raw = found
    return [
        UnsupportedSelection(
            key="aercu_opt",
            location=location,
            value=_normalize_value(value),
            supported_values=(0,),
            implemented="0=no CESM aerosol (mp_physics=40 runs constant-droplet Morrison-aerosol)",
            action="Set aercu_opt=0; aercu_opt>0 needs cu_physics=11 (MSKF), not ported.",
            outcome="recognized_control_not_wired",
            wrf_scheme="CESM-NCSU prescribed aerosol (aercu_opt)",
        )
        for value in _domain_values(raw)
        if _normalize_value(value) != 0
    ]


def apply_cldovrlp(config: Any) -> str | None:
    """Make the traced McICA overlap follow the namelist (WRF icld = cldovrlp), pre-JAX.

    GPUWRF_RRTMG_MAXRAND=1 implements cldovrlp=2; a cldovrlp=1 namelist pins it to
    0 for this process, i.e. today's random-overlap path bitwise. Call after the
    validators and before ``gpuwrf.kernels.rad_mcica`` is imported; raises if that
    module already resolved maximum-random. Returns a note when it pinned the flag.
    """

    config_obj = _coerce_config(config)
    if not _rrtmg_selected(config_obj) or _cldovrlp_values(config_obj)[1] != [1]:
        return None
    if os.environ.get("GPUWRF_RRTMG_MAXRAND", "0") != "1":
        return None
    module = sys.modules.get("gpuwrf.kernels.rad_mcica")
    if module is not None and getattr(module, "_MAXRAND", False):
        raise RuntimeError(
            "cldovrlp=1 but gpuwrf.kernels.rad_mcica already resolved GPUWRF_RRTMG_MAXRAND=1 "
            "(maximum-random); set GPUWRF_RRTMG_MAXRAND=0 before importing the radiation code"
        )
    os.environ["GPUWRF_RRTMG_MAXRAND"] = "0"
    return "cldovrlp=1: GPUWRF_RRTMG_MAXRAND pinned to 0 (WRF random McICA overlap)"


def _classify_rejection(key: str, value: Any) -> tuple[str, str | None]:
    """Classify a rejected ``key=value`` against the full WRF v4 catalog.

    Returns ``(outcome, wrf_scheme_name)``:

    * ``("not_yet_implemented", "<scheme name>")`` -- ``value`` is a recognized
      WRF v4 option for ``key`` that the GPU port does not yet implement /
      operationally wire;
    * ``("invalid_wrf_option", None)`` -- ``value`` is not a recognized WRF v4
      option for ``key`` (and ``key`` has a WRF v4 catalog);
    * ``("unsupported", None)`` -- ``key`` has no WRF v4 catalog (no enumeration
      to check against, e.g. a structural-only control).
    """

    if not isinstance(value, int):
        # Non-integer selections (e.g. a stray string) cannot be a WRF code.
        scheme = wrf_scheme_name(key, value) if isinstance(value, (int, float)) else None
        if scheme is not None:
            return "not_yet_implemented", scheme.name
        return "invalid_wrf_option", None

    scheme = wrf_scheme_name(key, value)
    if scheme is not None:
        return "not_yet_implemented", scheme.name
    if key in WRF_PARAM_LABEL:
        return "invalid_wrf_option", None
    return "unsupported", None


def _temf_pairing_failures(config: Any) -> list[UnsupportedSelection]:
    found=[_lookup(config,k) for k in ('bl_pbl_physics','sf_sfclay_physics')]
    values=[[_normalize_value(v) for v in _domain_values(f[1])] if f is not None else [None] for f in found]
    failures=[]
    for idx in range(max(map(len,values))):
        bl,sf=(v[min(idx,len(v)-1)] for v in values)
        if (bl==10)==(sf==10):
            continue
        failures.append(UnsupportedSelection(
            key='temf_pairing',location='bl_pbl_physics/sf_sfclay_physics',
            value={'bl_pbl_physics':bl,'sf_sfclay_physics':sf},
            supported_values=('bl_pbl_physics=10 with sf_sfclay_physics=10',),
            implemented='TEMF PBL and TEMF surface layer are a mandatory WRF pair',
            action='Select both option values as 10, or select neither as 10.',domain_index=idx+1))
    return failures


def _myj_pairing_failures(config: Any) -> list[UnsupportedSelection]:
    bl_found = _lookup(config, "bl_pbl_physics")
    sf_found = _lookup(config, "sf_sfclay_physics")
    if bl_found is None and sf_found is None:
        return []

    bl_location = bl_found[0] if bl_found is not None else "bl_pbl_physics"
    sf_location = sf_found[0] if sf_found is not None else "sf_sfclay_physics"
    bl_values = [_normalize_value(v) for v in _domain_values(bl_found[1])] if bl_found is not None else [None]
    sf_values = [_normalize_value(v) for v in _domain_values(sf_found[1])] if sf_found is not None else [None]
    ndom = max(len(bl_values), len(sf_values))

    failures: list[UnsupportedSelection] = []
    for idx in range(ndom):
        bl = bl_values[idx] if idx < len(bl_values) else bl_values[-1]
        sf = sf_values[idx] if idx < len(sf_values) else sf_values[-1]
        if (bl == 2) == (sf == 2):
            continue
        if bl != 2 and sf != 2:
            continue
        failures.append(
            UnsupportedSelection(
                key="myj_pairing",
                location=f"{bl_location}/{sf_location}",
                value={"bl_pbl_physics": bl, "sf_sfclay_physics": sf},
                supported_values=("bl_pbl_physics=2 with sf_sfclay_physics=2",),
                implemented="MYJ PBL and Janjic Eta surface layer are a mandatory WRF pair",
                action="Select both option values as 2, or select neither as 2.",
                domain_index=idx + 1 if ndom > 1 else None,
            )
        )
    return failures


def _qnse_pairing_failures(config):
    bl_found,sf_found=_lookup(config,'bl_pbl_physics'),_lookup(config,'sf_sfclay_physics')
    land_found=_lookup(config,'sf_surface_physics')
    bls=[_normalize_value(v) for v in _domain_values(bl_found[1])] if bl_found else [None]
    sfs=[_normalize_value(v) for v in _domain_values(sf_found[1])] if sf_found else [None]
    lands=[_normalize_value(v) for v in _domain_values(land_found[1])] if land_found else [None]
    failures=[]
    for index in range(max(len(bls),len(sfs),len(lands))):
        bl,sf=bls[min(index,len(bls)-1)],sfs[min(index,len(sfs)-1)]
        land=lands[min(index,len(lands)-1)]
        if (bl==4)!=(sf==4):
            failures.append(UnsupportedSelection(key='qnse_pairing',location='physics',
                value={'bl_pbl_physics':bl,'sf_sfclay_physics':sf},
                supported_values=('bl_pbl_physics=4 with sf_sfclay_physics=4',),
                implemented='QNSE PBL and surface layer are a mandatory WRF pair',
                action='Select both option values as 4, or select neither as 4.',domain_index=index+1))
        if bl==4 and land in (1,2,7):
            failures.append(UnsupportedSelection(key='sf_surface_physics',location=land_found[0],
                value=land,supported_values=(0,4),
                implemented='QNSE post-land physical heat/moisture fluxes and caller operands are unqualified for slab, Noah-classic and Pleim-Xiu',
                action='Select no LSM (sf_surface_physics=0) or Noah-MP (4) with QNSE.',domain_index=index+1))
    return failures


def _dfi_admission_failures(config_obj: Any) -> list[UnsupportedSelection]:
    """``&dfi_control`` admission: dfi_opt 0 or a reproducible DFL/DDFI/TDFI request."""

    dfi = _lookup(config_obj, "dfi_opt")
    if dfi is None:
        return []
    location, raw = dfi
    values = [_normalize_value(value) for value in _domain_values(raw)]
    action = "Set dfi_opt=0 for an unfiltered forecast, or use CPU-WRF for this DFI configuration."

    def failure(value: Any, implemented: str) -> UnsupportedSelection:
        return UnsupportedSelection(
            key="dfi_opt",
            location=location,
            value=value,
            supported_values=(0, 1, 2, 3),
            implemented=implemented,
            action=action,
            outcome="recognized_control_not_wired",
            wrf_scheme="digital filter initialization (DFI)",
        )

    if len(values) > 1:
        return [failure(values, "dfi_opt is a single WRF value (0=off, 1=DFL, 2=DDFI, 3=TDFI)")]
    if not values or values[0] == 0:
        return []
    if values[0] not in (1, 2, 3):
        return [failure(values[0], "0=disabled, 1=DFL, 2=DDFI, 3=TDFI (twice DFI)")]
    from gpuwrf.dfi.control import dfi_refusals

    if isinstance(config_obj, Mapping) and any(isinstance(v, Mapping) for v in config_obj.values()):
        nested = config_obj
    elif isinstance(config_obj, Mapping):
        nested = {"dfi_control": dict(config_obj)}
    else:
        nested = {"dfi_control": {"dfi_opt": values[0]}}
    return [failure(values[0], f"DFI refused: {reason}") for reason in dfi_refusals(nested)]


def _coerce_config(config: Any) -> Any:
    if isinstance(config, Path):
        return _parse_wrf_namelist(config.read_text())
    if isinstance(config, str):
        if "\n" in config or config.lstrip().startswith("&"):
            return _parse_wrf_namelist(config)
        path = Path(config)
        if path.exists():
            return _parse_wrf_namelist(path.read_text())
    return config


def _lookup(config: Any, key: str) -> tuple[str, Any] | None:
    if isinstance(config, Mapping):
        if key in config:
            return key, config[key]
        for section, values in config.items():
            if isinstance(values, Mapping) and key in values:
                return f"{section}.{key}", values[key]
        return None
    if hasattr(config, key):
        return key, getattr(config, key)
    fields = getattr(config, "__dataclass_fields__", {})
    if key in fields:
        return key, getattr(config, key)
    return None


def _domain_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        arr = np.asarray(value)
        return arr.reshape(-1).tolist()
    return [value]


def _sorted_supported_values(values: frozenset[Any]) -> tuple[Any, ...]:
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=repr))


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip().strip("'\"")
        lower = text.lower()
        if lower in {".true.", "true", "t"}:
            return True
        if lower in {".false.", "false", "f"}:
            return False
        try:
            number = float(text.replace("d", "e").replace("D", "e"))
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _parse_wrf_namelist(text: str) -> dict[str, dict[str, list[Any]]]:
    sections: dict[str, dict[str, list[Any]]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            current = line[1:].strip().lower()
            sections.setdefault(current, {})
            continue
        if line == "/":
            current = None
            continue
        if current is None or "=" not in line:
            continue
        key, rhs = line.split("=", 1)
        sections[current][key.strip().lower()] = _split_values(rhs)
    return sections


def _split_values(rhs: str) -> list[Any]:
    values: list[Any] = []
    for token in rhs.rstrip(",").split(","):
        token = token.strip()
        if not token:
            continue
        values.extend(_expand_repeat(token))
    return values


def _expand_repeat(token: str) -> list[Any]:
    """Expand a Fortran namelist repeat-count token ``N*value`` -> N copies.

    WRF namelists very commonly use ``3*8`` to mean ``8, 8, 8`` across max_dom
    domains. The ``*`` is the Fortran repeat operator, not multiplication.
    Tokens without a valid ``N*`` prefix are returned as a single normalized
    value unchanged (a bare ``*`` Fortran "no value" marker is dropped).
    """

    if "*" not in token:
        return [_normalize_value(token)]
    count_str, _, value_str = token.partition("*")
    count_str = count_str.strip()
    value_str = value_str.strip()
    if not count_str.isdigit():
        # Not a repeat count (e.g. an unexpected expression); keep verbatim.
        return [_normalize_value(token)]
    count = int(count_str)
    if not value_str:
        # Fortran ``N*`` with no value = "keep N defaults": nothing to validate.
        return []
    return [_normalize_value(value_str)] * count


def _format_error(selections: list[UnsupportedSelection]) -> str:
    lines = ["Unsupported namelist/config option(s) for the GPU-WRF faithful path:"]
    for item in selections:
        lines.append(_format_selection(item))
    return "\n".join(lines)


def _format_selection(item: UnsupportedSelection) -> str:
    domain = f" domain {item.domain_index}" if item.domain_index is not None else ""
    supported = ", ".join(repr(v) for v in item.supported_values)

    if item.outcome == "out_of_scope":
        # ``implemented`` carries "OUT OF SCOPE: <feature>"; ``action`` the alt.
        return (
            f"- {item.location}{domain}={item.value} ({item.wrf_scheme}): "
            f"{item.implemented} -- a documented out-of-scope decision for this "
            f"GPU port, NOT silently ignored. Action: {item.action}"
        )
    if item.outcome == "not_yet_implemented":
        label = WRF_PARAM_LABEL.get(item.key, item.key)
        return (
            f"- {item.location}{domain}={item.value} ({item.wrf_scheme}): recognized WRF v4 "
            f"{label} scheme, NOT YET IMPLEMENTED in the GPU port. "
            f"Supported {item.key} values: {supported}. "
            f"Implemented: {item.implemented}. Action: {item.action}"
        )
    if item.outcome == "reference_only_not_operational":
        label = WRF_PARAM_LABEL.get(item.key, item.key)
        scheme = item.wrf_scheme or f"{item.key}={item.value}"
        return (
            f"- {item.location}{domain}={item.value} ({scheme}): oracle-backed WRF v4 "
            f"{label} scheme, but NOT operationally wired into the GPU forecast scan. "
            f"Reason: {item.implemented}. "
            f"Running it would SILENTLY use a DIFFERENT scheme than requested "
            f"(the operational {label} path runs the implemented scheme instead) "
            f"or route through a missing kernel -- "
            f"refusing rather than producing a silent wrong-scheme run. "
            f"Operationally-wired {item.key} values: {supported}. Action: {item.action}"
        )
    if item.outcome == "recognized_control_not_wired":
        return (
            f"- {item.location}{domain}={item.value} ({item.wrf_scheme}): "
            f"recognized WRF namelist control, but the selected value is NOT "
            f"operationally wired in this GPU port -- {item.implemented}. "
            f"Fail-closed (NOT silently ignored). Action: {item.action}"
        )
    if item.outcome == "invalid_wrf_option":
        label = WRF_PARAM_LABEL.get(item.key, item.key)
        return (
            f"- {item.location}{domain}={item.value} is not a recognized WRF v4 {label} option. "
            f"Supported {item.key} values: {supported}. "
            f"Implemented: {item.implemented}. Action: {item.action}"
        )
    # Generic / structural rejection (no WRF v4 catalog for this key).
    return (
        f"- {item.location}{domain} selected {item.value!r}; supported values: "
        f"{supported}. Implemented: {item.implemented}. Action: {item.action}"
    )


__all__ = [
    "SUPPORTED_OPTIONS",
    "SupportedOption",
    "NotOperationallyWiredError",
    "UnsupportedNamelistOption",
    "UnsupportedSchemeError",
    "UnsupportedSelection",
    "collect_namelist_warnings",
    "validate_namelist",
    "validate_operational_namelist",
    "validate_supported_namelist",
]
