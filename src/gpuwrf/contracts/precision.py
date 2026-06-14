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


class AcousticPrecisionMode(str, Enum):
    """Static acoustic precision contract labels.

    These labels are cache-key/report plumbing only. They do not change the
    state precision matrix unless a later ADR-backed implementation explicitly
    consumes the non-default mode.
    """

    FP64_DEFAULT = "fp64_default"
    MIXED_PERTURB_FP32 = "mixed_perturb_fp32"


DEFAULT_ACOUSTIC_PRECISION_MODE = AcousticPrecisionMode.FP64_DEFAULT.value


def normalize_acoustic_precision_mode(
    mode: str | AcousticPrecisionMode | None,
) -> AcousticPrecisionMode:
    """Return a validated acoustic precision mode enum."""

    if mode is None:
        return AcousticPrecisionMode.FP64_DEFAULT
    if isinstance(mode, AcousticPrecisionMode):
        return mode
    try:
        return AcousticPrecisionMode(str(mode))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AcousticPrecisionMode)
        raise ValueError(
            f"unsupported acoustic_precision_mode {mode!r}; expected one of: {allowed}"
        ) from exc


def acoustic_precision_mode_label(mode: str | AcousticPrecisionMode | None) -> str:
    """Return the stable report/cache-key label for an acoustic precision mode."""

    return normalize_acoustic_precision_mode(mode).value


STATE_FIELD_ORDER = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "p",
    "p_total",
    "p_perturbation",
    "ph",
    "ph_total",
    "ph_perturbation",
    "mu",
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
    # --- v0.16 additive aerosol-aware Thompson (mp=28) leaves (append-only) ---
    # Appended at the VERY END (after the v0.6.0 + v0.15 additions) so every
    # existing leaf keeps its pytree position. nwfa/nifa are the WRF
    # QNWFA/QNIFA water-/ice-friendly aerosol number concentrations (kg^-1).
    "nwfa",
    "nifa",
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
    # --- v0.16 additive aerosol-aware Thompson (mp=28) leaves ---
    # nwfa/nifa follow the existing hydrometeor/aerosol number-species
    # precision class (FP32 gated, same as Nc/Nn).
    "nwfa": (FP32_GATED, True),
    "nifa": (FP32_GATED, True),
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
        if field not in mapping:
            raise KeyError(f"unknown state field {field!r}")
        return mapping[field]


DEFAULT_DTYPES = DTypeRegistry.from_precision_matrix()
