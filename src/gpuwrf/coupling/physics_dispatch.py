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
* Each scheme records ``gpu_runnable``. KF (cu=1), BMJ (cu=2), Tiedtke (cu=6),
  and the jit/vmap'd microphysics / PBL / surface-layer / Noah-MP/Noah-classic
  kernels are GPU-runnable. Tiedtke (cu=6) is GPU-batched via
  ``cumulus_tiedtke_jax`` and scan-wired (``CU_SCAN_ADAPTERS[6]``), savepoint-
  gated against unmodified ``module_cu_tiedtke.F``
  (proofs/v060/tiedtke_gpubatch_savepoint_parity.json). Grell-Freitas (cu=3) is a
  faithful CPU-NumPy reference port -- selectable, parity-gated, but flagged
  ``gpu_runnable=False`` (GPU closure-ensemble batch TODO). New Tiedtke (cu=16)
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
    16: _mp_entry(16, "gpuwrf.physics.microphysics_wdm6", "wdm6_physics_tendency", gpu=True),
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
}

# --- Cumulus (cu_physics) ------------------------------------------------------
# KF (cu=1), BMJ (cu=2), and Tiedtke (cu=6) are jit/vmap'd operational GPU cumulus
# paths (scan-wired via CU_SCAN_ADAPTERS). Grell-Freitas (cu=3) is a FAITHFUL
# CPU-NumPy reference port -- selectable + parity gated but NOT yet jit/vmap'd, so
# gpu_runnable=False (GPU closure-ensemble batch TODO). New Tiedtke (cu=16) shares
# the Tiedtke kernel but is NOT separately savepoint-gated, so it stays
# gpu_runnable=False / fail-closed. All route through the combined R*CUTEN
# tendency + RAINCV/PRATEC family (S0 cugd_* correction: no inert cugd_* State
# carry for GF).
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
    3: SchemeEntry("cumulus", 3, CU_SCHEMES[3].name, "gpuwrf.physics.cumulus_grell_freitas",
                   "grell_freitas_step", "column_state", False,
                   reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                   writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
                   tendency_members=CUMULUS_TENDENCY_MEMBERS[3], accumulators=("rainc_acc",),
                   notes="CPU-NumPy reference port; GPU-batching (jit/vmap) TODO. cugd_* carry "
                   "DROPPED per S0 correction -- routed via R*CUTEN + RAINCV/PRATEC + shallow diags."),
    6: SchemeEntry("cumulus", 6, CU_SCHEMES[6].name, "gpuwrf.physics.cumulus_tiedtke_jax", "tiedtke_column_jax",
                   "column_state", True,
                   reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs"),
                   writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
                   tendency_members=CUMULUS_TENDENCY_MEMBERS[6], accumulators=("rainc_acc",),
                   notes="GPU-batched (jit/vmap) Tiedtke; scan-wired via CU_SCAN_ADAPTERS[6]; "
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
}

# --- Land surface (sf_surface_physics) -----------------------------------------
# sf_surface=4 Noah-MP is the EXISTING coupling.noahmp_surface_hook path (wired in
# operational_mode via namelist.use_noahmp). sf_surface=2 Noah classic exposes the
# lsm_noah_classic.sflx_step land step + a 4-layer land carry (S0 land carry).
_SURFACE_ENTRIES: dict[int, SchemeEntry] = {
    0: SchemeEntry("land_surface", 0, "disabled", "", "", "disabled", True),
    2: SchemeEntry("land_surface", 2, SURFACE_SCHEMES[2].name, "gpuwrf.physics.lsm_noah_classic",
                   "sflx_step", "land_step", True,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin", "soil_moisture", "mavail"),
                   carry_members=("flx4", "fvb", "fbur", "fgsn", "smcrel", "xlaidyn"),
                   notes="Owns a 4-layer (num_soil_layers=4) land carry; does NOT reinterpret the "
                   "2-D State.soil_moisture as a 4-layer field (S0 land carry rule)."),
    4: SchemeEntry("land_surface", 4, SURFACE_SCHEMES[4].name, "gpuwrf.coupling.noahmp_surface_hook",
                   "noahmp_surface_step", "state_adapter", True,
                   reads_state=("t_skin", "soil_moisture", "mavail"),
                   writes_state=("t_skin", "soil_moisture", "mavail"),
                   carry_members=("NoahMPLandState",)),
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
