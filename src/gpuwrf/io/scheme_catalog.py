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
    RECOGNIZED_FAIL_CLOSED = "recognized_fail_closed"
    OUT_OF_SCOPE = "out_of_scope"

    @property
    def passes_namelist_check(self) -> bool:
        """True iff this status is *accepted* by ``validate_namelist``.

        ``IMPLEMENTED`` and ``REFERENCE_ONLY`` pass the namelist validator
        (REFERENCE_ONLY then fail-closes in the operational scan with a named
        reason). ``RECOGNIZED_FAIL_CLOSED`` and ``OUT_OF_SCOPE`` are rejected at
        the namelist layer so the user fails fast with a helpful message.
        """

        return self in (SupportStatus.IMPLEMENTED, SupportStatus.REFERENCE_ONLY)


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
    "ra_lw_physics": frozenset({0, 4}),
    "ra_sw_physics": frozenset({0, 4}),
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
    "ra_lw_physics": {
        1: (
            "Classic RRTM longwave passes its isolated WRF-savepoint gate "
            "(host-NumPy single-column kernel) but is not selectable by the "
            "operational GPU scan (post-0.9.0 jit/vmap rewrite + radiation "
            "dispatch).",
            "Use ra_lw_physics=4 (RRTMG, GPU-operational) for the operational LW path.",
        ),
    },
    "ra_sw_physics": {
        1: (
            "Dudhia shortwave passes its isolated WRF-savepoint gate but is not "
            "yet selectable by the operational GPU scan (post-0.9.0 radiation-"
            "family dispatch).",
            "Use ra_sw_physics=4 (RRTMG, GPU-operational) for the operational SW path.",
        ),
    },
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
    "ra_sw_physics": "Use ra_sw_physics=4 (RRTMG).",
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


__all__ = [
    "SupportStatus",
    "SchemeSupport",
    "OutOfScopeFeature",
    "OUT_OF_SCOPE_FEATURES",
    "OUT_OF_SCOPE_FEATURE_KEYS",
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
