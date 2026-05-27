"""Schaer et al. 2002 sinusoidal-terrain mountain-wave initial condition."""

from __future__ import annotations

import numpy as np

from .common import IdealizedCase, THETA0_K, dry_density, hydrostatic_pressure_from_height


def build_schaer_mountain_wave(
    *,
    nx: int = 401,
    nz: int = 121,
    dx_m: float = 250.0,
    dz_m: float = 250.0,
    domain_half_width_m: float = 50000.0,
    mountain_height_m: float = 250.0,
    envelope_half_width_m: float = 5000.0,
    wavelength_m: float = 4000.0,
    background_wind_m_s: float = 10.0,
    brunt_vaisala_s: float = 0.01,
) -> IdealizedCase:
    """Return the Schaer 2002 sinusoidal-terrain mountain-wave IC."""

    x = np.linspace(-float(domain_half_width_m), float(domain_half_width_m), int(nx), dtype=np.float64)
    z = np.arange(int(nz), dtype=np.float64) * float(dz_m)
    envelope = np.exp(-((x / float(envelope_half_width_m)) ** 2))
    terrain = float(mountain_height_m) * envelope * np.cos(np.pi * x / float(wavelength_m)) ** 2
    dhdx = np.gradient(terrain, x, edge_order=2)
    xx, zz = np.meshgrid(x, z)
    terrain_2d = np.broadcast_to(terrain[None, :], (int(nz), int(nx)))
    height_agl = np.maximum(zz - terrain_2d, 0.0)
    theta = THETA0_K * np.exp((float(brunt_vaisala_s) ** 2 / 9.80665) * height_agl)
    pressure = hydrostatic_pressure_from_height(height_agl)
    u = np.full_like(theta, float(background_wind_m_s), dtype=np.float64)
    arrays = {
        "x_m": x,
        "z_m": z,
        "terrain_m": terrain.astype(np.float64),
        "terrain_slope": dhdx.astype(np.float64),
        "theta_k": theta.astype(np.float64),
        "pressure_pa": pressure.astype(np.float64),
        "density_kg_m3": dry_density(pressure, theta).astype(np.float64),
        "u_m_s": u,
        "w_surface_linear_m_s": (float(background_wind_m_s) * dhdx).astype(np.float64),
    }
    return IdealizedCase(
        case_id="idealized-mountain-wave-schaer-2002",
        reference={
            "name": "Schaer et al. 2002 non-hydrostatic mountain-wave benchmark",
            "purpose": "terrain-following pressure-gradient and vertical wave propagation",
            "linear_oracle": "surface vertical velocity is U * dh/dx; full steady wave comparison is a separate runner.",
        },
        grid={"nx": int(nx), "nz": int(nz), "dx_m": float(dx_m), "dz_m": float(dz_m), "staggering": "2d-xz-mass"},
        parameters={
            "theta0_k": THETA0_K,
            "mountain_height_m": float(mountain_height_m),
            "envelope_half_width_m": float(envelope_half_width_m),
            "wavelength_m": float(wavelength_m),
            "background_wind_m_s": float(background_wind_m_s),
            "brunt_vaisala_s": float(brunt_vaisala_s),
        },
        arrays=arrays,
    )
