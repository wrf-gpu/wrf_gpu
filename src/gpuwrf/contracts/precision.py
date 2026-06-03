"""Precision registry for state fields and ADR-007 coupling boundaries.

`FP32_GATED` means ADR-007 authorizes FP32 storage only under the
field-specific validation gates. Locked mass, pressure, geopotential,
surface-stability, and accumulation fields remain FP64.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


FP64 = jnp.float64
FP32_GATED = jnp.float32
INT32 = jnp.int32


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
    "qke": (FP32_GATED, True),
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
