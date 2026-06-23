"""WRF map-factor accessors and metric helpers for c2 dycore modules."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from functools import partial
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec


configure_jax_x64()


def _first_time_variable(dataset: Dataset, name: str) -> jax.Array:
    """Loads one WRF Time-dependent variable during initialization only."""

    return jnp.asarray(np.asarray(dataset.variables[name][0], dtype=np.float64))


def terrain_slope_metrics(terrain_height: jax.Array, dx_m: float, dy_m: float) -> tuple[jax.Array, ...]:
    """Derives mass- and face-point terrain slopes for well-balanced PGF terms."""

    terrain = jnp.asarray(terrain_height, dtype=jnp.float64)
    dx = jnp.asarray(dx_m, dtype=jnp.float64)
    dy = jnp.asarray(dy_m, dtype=jnp.float64)
    padded_x = jnp.pad(terrain, ((0, 0), (1, 1)), mode="edge")
    padded_y = jnp.pad(terrain, ((1, 1), (0, 0)), mode="edge")
    dzdx_u = (padded_x[:, 1:] - padded_x[:, :-1]) / dx
    dzdy_v = (padded_y[1:, :] - padded_y[:-1, :]) / dy
    dzdx = 0.5 * (dzdx_u[:, 1:] + dzdx_u[:, :-1])
    dzdy = 0.5 * (dzdy_v[1:, :] + dzdy_v[:-1, :])
    return dzdx, dzdy, dzdx_u, dzdy_v


# 2*Omega for the analytic Coriolis fallback (WRF EARTH_OMEGA, module_model_constants.F).
_TWO_OMEGA = 2.0 * 7.2921e-5


def _coriolis_metrics(
    dataset: Dataset, mass_shape: tuple[int, int]
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reads the WRF Coriolis metrics (F/E/SINALPHA/COSALPHA) from a wrfout/wrfinput.

    ``F = 2*Omega*sin(lat)`` and ``E = 2*Omega*cos(lat)`` are stored on mass points.
    If a field is absent we fall back to the analytic value from ``XLAT`` (F, E) or
    to the no-rotation default (sina=0, cosa=1), so the loader never silently drops
    the Coriolis force on a real case while still tolerating slim fixtures.
    """

    variables = dataset.variables
    if "XLAT" in variables:
        xlat = np.asarray(variables["XLAT"][0], dtype=np.float64)
        lat_rad = np.deg2rad(xlat)
    else:
        lat_rad = None

    def _read(name: str, analytic: np.ndarray | None, default: float) -> jax.Array:
        if name in variables:
            return jnp.asarray(np.asarray(variables[name][0], dtype=np.float64))
        if analytic is not None:
            return jnp.asarray(analytic)
        return jnp.full(mass_shape, default, dtype=jnp.float64)

    f_analytic = _TWO_OMEGA * np.sin(lat_rad) if lat_rad is not None else None
    e_analytic = _TWO_OMEGA * np.cos(lat_rad) if lat_rad is not None else None
    f = _read("F", f_analytic, 0.0)
    e = _read("E", e_analytic, 0.0)
    sina = _read("SINALPHA", None, 0.0)
    cosa = _read("COSALPHA", None, 1.0)
    return f, e, sina, cosa


def load_wrfinput_metrics(path: str | Path) -> DycoreMetrics:
    """Loads WRF map factors and hybrid-eta coefficients from ``wrfinput``.

    This is an initialization-only helper. Timestep code receives the returned
    ``DycoreMetrics`` pytree as device-resident static grid data.
    """

    with Dataset(str(path)) as dataset:
        eta_levels = _first_time_variable(dataset, "ZNW")
        terrain_height = _first_time_variable(dataset, "HGT")
        dzdx, dzdy, dzdx_u, dzdy_v = terrain_slope_metrics(
            terrain_height,
            float(getattr(dataset, "DX")),
            float(getattr(dataset, "DY")),
        )
        nz = int(dataset.dimensions["bottom_top"].size)
        f, e, sina, cosa = _coriolis_metrics(dataset, terrain_height.shape)
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
            cf1=_first_time_variable(dataset, "CF1"),
            cf2=_first_time_variable(dataset, "CF2"),
            cf3=_first_time_variable(dataset, "CF3"),
            fnm=_first_time_variable(dataset, "FNM"),
            fnp=_first_time_variable(dataset, "FNP"),
            dzdx=dzdx,
            dzdy=dzdy,
            dzdx_u=dzdx_u,
            dzdy_v=dzdy_v,
            f=f,
            e=e,
            sina=sina,
            cosa=cosa,
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
