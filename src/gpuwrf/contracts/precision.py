"""Precision registry for state fields and ADR-007 coupling boundaries.

`FP32_GATED` means ADR-007 authorizes FP32 storage only under the
field-specific validation gates. Locked mass, pressure, geopotential,
surface-stability, and accumulation fields remain FP64.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


FP64 = jnp.float64
FP32_GATED = jnp.float32
INT32 = jnp.int32


def force_fp64_island(*arrays):
    """Upcast cancellation-sensitive operator inputs to fp64 IN-OPERATOR (v0.20 S2).

    A genuine fp64 cancellation bracket -- the equation of state, the
    pressure-gradient brackets, the implicit w/phi vertical solve -- must compute
    in fp64 regardless of how its inputs are *stored*, so that a later fp32 storage
    downcast (the S3-S8 perturbation-authoritative work) cannot silently
    contaminate it. The upcast is applied at the operator boundary and is
    therefore CALLER-INDEPENDENT: the island is locked from inside.

    No-op for the ``fp64_default`` path: an already-fp64 array is returned
    UNCHANGED (Python identity -- no ``convert-element-type`` is emitted), so
    fp64_default stays bit-identical with zero extra HLO converts. For an fp32
    caller the array is widened to fp64 for the bracket arithmetic. ``None`` passes
    through. Returns a single value when one array is given, else a tuple.
    """

    def _up(value):
        if value is None:
            return None
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.dtype(dtype) == jnp.dtype(FP64):
            return value  # identity: no convert op, fp64_default bit-identical
        if hasattr(value, "astype"):
            return value.astype(FP64)
        return jnp.asarray(value, dtype=FP64)

    upcast = tuple(_up(value) for value in arrays)
    return upcast[0] if len(upcast) == 1 else upcast


class AcousticPrecisionMode(str, Enum):
    """Static acoustic precision contract labels.

    These labels are cache-key/report plumbing only. They do not change the
    state precision matrix unless a later ADR-backed implementation explicitly
    consumes the non-default mode.
    """

    FP64_DEFAULT = "fp64_default"
    MIXED_PERTURB_FP32 = "mixed_perturb_fp32"


DEFAULT_ACOUSTIC_PRECISION_MODE = AcousticPrecisionMode.FP64_DEFAULT.value
MIXED_PERTURB_FP32_ALIASES = frozenset({
    AcousticPrecisionMode.MIXED_PERTURB_FP32.value,
    "mixed_perturb_fp32_v020",
    "s4_mixed_perturb_fp32",
})
MIXED_PERTURB_FP32_STORAGE_FIELDS = frozenset({
    "p_perturbation",
    "ph_perturbation",
    "mu_perturbation",
    "w",
})


def normalize_acoustic_precision_mode(
    mode: str | AcousticPrecisionMode | None,
) -> AcousticPrecisionMode:
    """Return a validated acoustic precision mode enum."""

    if mode is None:
        return AcousticPrecisionMode.FP64_DEFAULT
    if isinstance(mode, AcousticPrecisionMode):
        return mode
    label = str(mode).strip().lower()
    if label in MIXED_PERTURB_FP32_ALIASES:
        return AcousticPrecisionMode.MIXED_PERTURB_FP32
    try:
        return AcousticPrecisionMode(label)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AcousticPrecisionMode)
        raise ValueError(
            f"unsupported acoustic_precision_mode {mode!r}; expected one of: {allowed}"
        ) from exc


def acoustic_precision_mode_label(mode: str | AcousticPrecisionMode | None) -> str:
    """Return the stable report/cache-key label for an acoustic precision mode."""

    return normalize_acoustic_precision_mode(mode).value


def is_mixed_perturb_fp32_mode(mode: str | AcousticPrecisionMode | None) -> bool:
    """Whether an opt-in mode stores acoustic perturbation carry in fp32."""

    return normalize_acoustic_precision_mode(mode) is AcousticPrecisionMode.MIXED_PERTURB_FP32


def mixed_perturb_storage_dtype(field: str, mode: str | AcousticPrecisionMode | None):
    """Return the opt-in mixed-mode storage dtype for a State field."""

    if is_mixed_perturb_fp32_mode(mode) and field in MIXED_PERTURB_FP32_STORAGE_FIELDS:
        return FP32_GATED
    return FP64


STATE_FIELD_ORDER = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    # v0.20 S1: legacy total aliases p/ph/mu removed from the State pytree (they
    # duplicated the totals). Kept in PRECISION_MATRIX below as harmless aliases so
    # any direct dtype_for("p") lookup still resolves, but dropped from the field
    # ORDER so it stays equal to State.__slots__ (the memory-audit invariant).
    "p_total",
    "p_perturbation",
    "ph_total",
    "ph_perturbation",
    "mu_total",
    "mu_perturbation",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "qke",
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    "t_skin",
    "soil_moisture",
    "xland",
    "lakemask",
    "mavail",
    "roughness_m",
    "lu_index",
    "rain_acc",
    "snow_acc",
    "graupel_acc",
    "ice_acc",
    "u_bdy",
    "v_bdy",
    "theta_bdy",
    "qv_bdy",
    "ph_bdy",
    "mu_bdy",
    "w_bdy",
    "p_bdy",
    "pb_bdy",
    "phb_bdy",
    "mub_bdy",
    # --- v0.6.0 S0 additive physics leaves (append-only; manager patch) ---
    # Appended at the END to preserve the pytree/byte order of every existing
    # leaf. Nc/Nn are WDM6 number concentrations; rainc_acc is the cumulus
    # precipitation accumulator (RAINC).
    "Nc",
    "Nn",
    "rainc_acc",
    # --- v0.15 MYNN SGS-cloud leaves (append-only) ---
    "qsq",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    # --- v0.17 ADR-032 graupel/hail (qh) substrate leaves (append-only) ---
    # Appended after the prior additive blocks so every existing leaf keeps its
    # pytree position. qh=hail mixing ratio (QHAIL), Nh=hail number (QNHAIL),
    # qvolg/qvolh=predicted-density graupel/hail particle volume
    # (QVGRAUPEL/QVHAIL).
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
    # --- v0.16 additive aerosol-aware Thompson (mp=28) leaves (append-only) ---
    # Appended after the v0.6.0 + v0.15 + v0.17 ADR-032 hail additions so every
    # existing leaf keeps its pytree position. nwfa/nifa are the WRF QNWFA/QNIFA
    # water-/ice-friendly aerosol number concentrations (kg^-1).
    "nwfa",
    "nifa",
    # --- v0.17 hail surface-precip accumulator (append-only historical tail) ---
    "hail_acc",
    # --- v0.22 standalone wrfbdy scalar boundary leaves (optional) ---
    "qc_bdy",
    "qr_bdy",
    "qi_bdy",
    "qs_bdy",
    "qg_bdy",
    "Ni_bdy",
    "Nr_bdy",
)


PRECISION_MATRIX = {
    # Dynamics / mass and acoustic boundary.
    "mu": (FP64, False),
    "p": (FP64, False),
    "p_total": (FP64, False),
    "p_perturbation": (FP64, False),
    "ph": (FP64, False),
    "ph_total": (FP64, False),
    "ph_perturbation": (FP64, False),
    "mu_total": (FP64, False),
    "mu_perturbation": (FP64, False),
    "pgeop": (FP64, False),
    "u": (FP32_GATED, True),
    "v": (FP32_GATED, True),
    "w": (FP64, False),
    "theta": (FP32_GATED, True),
    "qv": (FP32_GATED, True),
    # Thompson hydrometeor mass and number fields.
    "qc": (FP32_GATED, True),
    "qr": (FP32_GATED, True),
    "qi": (FP32_GATED, True),
    "qs": (FP32_GATED, True),
    "qg": (FP32_GATED, True),
    "Ni": (FP32_GATED, True),
    "Nr": (FP32_GATED, True),
    "Ns": (FP32_GATED, True),
    "Ng": (FP32_GATED, True),
    # MYNN turbulent kinetic energy.
    #
    # PRECISION-CONTRACT CHANGE (2026-06-04, qke-fp64-fix sprint, REVIEWED):
    # qke is promoted FP32_GATED -> FP64. At 1km (d03, steep Tenerife terrain,
    # dt=3s) the MYNN level-2.5 TKE budget (large production/dissipation
    # gradients) loses too much precision in fp32 and goes NON-FINITE after
    # forecast hour 1 (proofs/v090/d03_1km_validation.json: qke the SOLE
    # offending field, 3036 nonfinite cells; every other prognostic stayed
    # finite). qke is a diagnostic-style turbulence field OUTSIDE the conserved
    # mass / pressure / acoustic path (it never enters the fp64-locked mu/p/ph
    # accumulators), so promoting it to fp64 does NOT alter the gated-fp32
    # invariants and the d02 speedup is unchanged (qke is one small 3D field).
    # This is a precision-matrix promotion, NOT a clamp/fudge -- no physics
    # change, the existing WRF-faithful mym_predict qke cap (<=150) is untouched.
    #
    # The MYNN length-scale + TKE-budget INTERMEDIATES (el/elt/els/elf in
    # _mym_length_option1, the qke-weighted height integral, the _mym_predict_qke
    # tridiagonal budget) are promoted to fp64 IMPLICITLY: they are all functions
    # of qke (the column kernel carries tke = qke/2), so once qke is fp64 JAX
    # type-promotion widens every qke-touching intermediate to fp64. mynn_pbl.py
    # contains NO explicit float32 narrowing on this path (only int32 index
    # casts), and the column rho/dz are already fp64; the only remaining fp32
    # inputs are u/v/w/theta/qv (the bulk fields, deliberately kept FP32_GATED to
    # preserve the speedup). No source edit to mynn_pbl.py is needed.
    "qke": (FP64, False),
    # Surface-layer stability and flux handles.
    "ustar": (FP64, False),
    "theta_flux": (FP64, False),
    "qv_flux": (FP64, False),
    "tau_u": (FP64, False),
    "tau_v": (FP64, False),
    "rhosfc": (FP64, False),
    "fltv": (FP64, False),
    "t_skin": (FP64, False),
    "soil_moisture": (FP64, False),
    "xland": (FP32_GATED, True),
    "lakemask": (FP32_GATED, True),
    "mavail": (FP32_GATED, True),
    "roughness_m": (FP64, False),
    "lu_index": (INT32, False),
    # Accumulated precipitation diagnostics.
    "rain_acc": (FP64, False),
    "snow_acc": (FP64, False),
    "graupel_acc": (FP64, False),
    "ice_acc": (FP64, False),
    # Time-varying lateral-boundary forcing leaves.
    "u_bdy": (FP32_GATED, True),
    "v_bdy": (FP32_GATED, True),
    "w_bdy": (FP64, False),
    "theta_bdy": (FP32_GATED, True),
    "qv_bdy": (FP32_GATED, True),
    "p_bdy": (FP64, False),
    "pb_bdy": (FP64, False),
    "ph_bdy": (FP64, False),
    "phb_bdy": (FP64, False),
    "mu_bdy": (FP64, False),
    "mub_bdy": (FP64, False),
    "qc_bdy": (FP32_GATED, True),
    "qr_bdy": (FP32_GATED, True),
    "qi_bdy": (FP32_GATED, True),
    "qs_bdy": (FP32_GATED, True),
    "qg_bdy": (FP32_GATED, True),
    "Ni_bdy": (FP32_GATED, True),
    "Nr_bdy": (FP32_GATED, True),
    # --- v0.6.0 S0 additive physics leaves ---
    # WDM6 number concentrations follow the existing Thompson/Morrison number
    # species precision (FP32 gated). The cumulus precipitation accumulator is
    # FP64-locked like the grid-scale precip accumulators.
    "Nc": (FP32_GATED, True),
    "Nn": (FP32_GATED, True),
    "rainc_acc": (FP64, False),
    # --- v0.15 MYNN SGS-cloud leaves ---
    # qsq is the closure-2.6 prognostic total-water variance: it lives in the
    # same TKE-budget family as qke (FP64 after the qke-fp64-fix) and its
    # magnitudes are O(1e-8..1e-6), so it is FP64-locked. The SGS cloud
    # condensate/fraction are diagnostic radiation/buoyancy inputs recomputed
    # every MYNN step; FP32-gated like qc.
    "qsq": (FP64, False),
    "qc_bl": (FP32_GATED, True),
    "qi_bl": (FP32_GATED, True),
    "cldfra_bl": (FP32_GATED, True),
    # --- v0.17 ADR-032 graupel/hail (qh) substrate leaves ---
    # Hail mixing ratio / number / particle-volume follow the existing
    # hydrometeor + number-species precision class (FP32 gated, same as
    # qg/Ng/Nc). Bounded, non-conservation-critical scalar fields; they
    # downcast with the rest of the moisture family in a future fp32 run and
    # stay fp64 in the FP64-default operational mode.
    "qh": (FP32_GATED, True),
    "Nh": (FP32_GATED, True),
    "qvolg": (FP32_GATED, True),
    "qvolh": (FP32_GATED, True),
    # --- v0.16 additive aerosol-aware Thompson (mp=28) leaves ---
    # nwfa/nifa follow the existing hydrometeor/aerosol number-species
    # precision class (FP32 gated, same as Nc/Nn).
    "nwfa": (FP32_GATED, True),
    "nifa": (FP32_GATED, True),
    # Hail surface-precip accumulator: FP64-locked like every other precip
    # accumulator (rain/snow/graupel/ice), never gated -- accumulation fields
    # remain FP64 (ADR-007).
    "hail_acc": (FP64, False),
}


@dataclass(frozen=True)
class DTypeRegistry:
    """Encapsulates per-field dtype lookup reused by state allocation and tests."""

    defaults: tuple[tuple[str, object], ...]

    @classmethod
    def from_precision_matrix(cls) -> "DTypeRegistry":
        """Builds the M6 state-storage registry from the ADR-007 matrix."""

        return cls(tuple((field, PRECISION_MATRIX[field][0]) for field in STATE_FIELD_ORDER))

    @classmethod
    def fp64_defaults(cls) -> "DTypeRegistry":
        """Compatibility constructor for older callers that requested M3 defaults."""

        return cls.from_precision_matrix()

    def dtype_for(self, field: str):
        """Returns the dtype for one state field; guards against misspelled field names."""

        mapping = dict(self.defaults)
        if field in mapping:
            return mapping[field]
        # v0.20 S1: the legacy total aliases p/ph/mu were dropped from
        # STATE_FIELD_ORDER (no longer State leaves) but remain valid dtype
        # queries -- BaseState/BoundaryState/Tendencies still allocate
        # pressure/geopotential/mass buffers keyed by these names. Resolve them
        # from the precision matrix so those allocators are unchanged.
        if field in PRECISION_MATRIX:
            return PRECISION_MATRIX[field][0]
        raise KeyError(f"unknown state field {field!r}")


DEFAULT_DTYPES = DTypeRegistry.from_precision_matrix()
