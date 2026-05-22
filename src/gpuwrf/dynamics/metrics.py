"""WRF map-factor accessors and metric helpers for c2 dycore modules."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec


config.update("jax_enable_x64", True)


def _first_time_variable(dataset: Dataset, name: str) -> jax.Array:
    """Loads one WRF Time-dependent variable during initialization only."""

    return jnp.asarray(np.asarray(dataset.variables[name][0], dtype=np.float64))


def load_wrfinput_metrics(path: str | Path) -> DycoreMetrics:
    """Loads WRF map factors and hybrid-eta coefficients from ``wrfinput``.

    This is an initialization-only helper. Timestep code receives the returned
    ``DycoreMetrics`` pytree as device-resident static grid data.
    """

    with Dataset(str(path)) as dataset:
        eta_levels = _first_time_variable(dataset, "ZNW")
        nz = int(dataset.dimensions["bottom_top"].size)
        return DycoreMetrics(
            msftx=_first_time_variable(dataset, "MAPFAC_MX"),
            msfty=_first_time_variable(dataset, "MAPFAC_MY"),
            msfux=_first_time_variable(dataset, "MAPFAC_UX"),
            msfuy=_first_time_variable(dataset, "MAPFAC_UY"),
            msfvx=_first_time_variable(dataset, "MAPFAC_VX"),
            msfvy=_first_time_variable(dataset, "MAPFAC_VY"),
            c1h=_first_time_variable(dataset, "C1H"),
            c2h=_first_time_variable(dataset, "C2H"),
            c3h=_first_time_variable(dataset, "C3H"),
            c4h=_first_time_variable(dataset, "C4H"),
            c1f=_first_time_variable(dataset, "C1F"),
            c2f=_first_time_variable(dataset, "C2F"),
            c3f=_first_time_variable(dataset, "C3F"),
            c4f=_first_time_variable(dataset, "C4F"),
            dn=_first_time_variable(dataset, "DN"),
            dnw=_first_time_variable(dataset, "DNW"),
            rdn=_first_time_variable(dataset, "RDN"),
            rdnw=_first_time_variable(dataset, "RDNW"),
            p_top=_first_time_variable(dataset, "P_TOP"),
            provenance=f"wrfinput:{Path(path)}:nz={nz}:eta={tuple(eta_levels.shape)}",
        )


def flat_metrics_for_grid(grid: GridSpec) -> DycoreMetrics:
    """Builds the analytic flat metric fixture matching a GridSpec."""

    return DycoreMetrics.flat(
        ny=grid.ny,
        nx=grid.nx,
        nz=grid.nz,
        eta_levels=grid.eta_levels,
        top_pressure_pa=grid.vertical.top_pressure_pa,
    )


@partial(jax.jit, static_argnames=())
def mass_metric_area(metrics: DycoreMetrics) -> jax.Array:
    """Returns the mass-point map-factor product used by WRF flux divergences."""

    return metrics.msftx * metrics.msfty


@partial(jax.jit, static_argnames=())
def u_metric_ratio(metrics: DycoreMetrics) -> jax.Array:
    """Returns the x-face metric ratio used by WRF u-momentum terms."""

    return metrics.msfux / metrics.msfuy


@partial(jax.jit, static_argnames=())
def v_metric_ratio(metrics: DycoreMetrics) -> jax.Array:
    """Returns the y-face metric ratio used by WRF v-momentum terms."""

    return metrics.msfvy / metrics.msfvx


@partial(jax.jit, static_argnames=())
def metric_minmax(metrics: DycoreMetrics) -> jax.Array:
    """Small JIT-safe summary used by proof scripts."""

    values = jnp.asarray(
        [
            jnp.min(metrics.msftx),
            jnp.max(metrics.msftx),
            jnp.min(metrics.msfty),
            jnp.max(metrics.msfty),
            jnp.min(metrics.msfux),
            jnp.max(metrics.msfux),
            jnp.min(metrics.msfvy),
            jnp.max(metrics.msfvy),
        ],
        dtype=jnp.float64,
    )
    return values
