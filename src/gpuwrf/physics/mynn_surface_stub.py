"""Bulk-formula surface-flux stub for the MYNN2.5 column fixture."""

from __future__ import annotations

from dataclasses import dataclass

from jax import config
import jax.numpy as jnp

from gpuwrf.physics.mynn_constants import BULK_CD, BULK_CH, BULK_CQ, MIN_WIND


config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SurfaceFluxes:
    """Container for diagnostic surface fluxes used by the column kernel."""

    ustar: object
    theta_flux: object
    qv_flux: object
    tau_u: object
    tau_v: object


def bulk_surface_fluxes(u0, v0, theta0, qv0, *, surface_theta_delta=0.25, surface_qv_delta=1.0e-4) -> SurfaceFluxes:
    """Returns neutral bulk fluxes for the M5-S2 surface-layer placeholder."""

    wind = jnp.maximum(jnp.sqrt(u0 * u0 + v0 * v0), MIN_WIND)
    ustar = jnp.sqrt(BULK_CD) * wind
    theta_flux = BULK_CH * wind * surface_theta_delta
    qv_flux = BULK_CQ * wind * surface_qv_delta
    tau_u = -BULK_CD * wind * u0
    tau_v = -BULK_CD * wind * v0
    return SurfaceFluxes(ustar=ustar, theta_flux=theta_flux, qv_flux=qv_flux, tau_u=tau_u, tau_v=tau_v)
