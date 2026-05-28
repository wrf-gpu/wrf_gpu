"""Surface-layer interface and neutral bulk stub for the MYNN2.5 column fixture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jax import config
import jax.numpy as jnp

from gpuwrf.physics.mynn_constants import BULK_CD, BULK_CH, BULK_CQ, MIN_WIND, P608, R_D


config.update("jax_enable_x64", True)


class SurfaceLayerState(Protocol):
    """Minimum column state surface fields consumed by the surface-layer hook."""

    u: object
    v: object
    theta: object
    qv: object
    p: object


@dataclass(frozen=True)
class SurfaceFluxes:
    """Surface-layer flux contract consumed by MYNN.

    Scalar fluxes are kinematic and positive upward into the atmosphere.
    Momentum fluxes are kinematic components; for flow over fixed ground they
    are normally opposite-signed to the lowest model-level wind components.
    """

    ustar: object
    theta_flux: object
    qv_flux: object
    tau_u: object
    tau_v: object
    rhosfc: object
    fltv: object


def bulk_surface_fluxes(
    u0,
    v0,
    theta0,
    qv0,
    p0,
    *,
    surface_theta_delta=0.25,
    surface_qv_delta=1.0e-4,
) -> SurfaceFluxes:
    """Returns neutral bulk fluxes for the M5-S2 surface-layer placeholder."""

    wind = jnp.maximum(jnp.sqrt(u0 * u0 + v0 * v0), MIN_WIND)
    ustar = jnp.sqrt(BULK_CD) * wind
    theta_flux = BULK_CH * wind * surface_theta_delta
    qv_flux = BULK_CQ * wind * surface_qv_delta
    tau_u = -BULK_CD * wind * u0
    tau_v = -BULK_CD * wind * v0
    rhosfc = jnp.maximum(p0 / (R_D * (theta0 + P608 * qv0)), 1.0e-4)
    fltv = (1.0 + P608 * qv0) * theta_flux + P608 * theta0 * qv_flux
    return SurfaceFluxes(
        ustar=ustar,
        theta_flux=theta_flux,
        qv_flux=qv_flux,
        tau_u=tau_u,
        tau_v=tau_v,
        rhosfc=rhosfc,
        fltv=fltv,
    )


def surface_layer(state: SurfaceLayerState) -> SurfaceFluxes:
    """M6-S3 replacement hook for real Monin-Obukhov surface-layer coupling."""

    return bulk_surface_fluxes(
        state.u[..., 0],
        state.v[..., 0],
        state.theta[..., 0],
        state.qv[..., 0],
        state.p[..., 0],
    )
