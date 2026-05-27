"""Straka et al. 1993 density-current initial condition."""

from __future__ import annotations

import numpy as np

from .common import IdealizedCase, THETA0_K, dry_density, hydrostatic_pressure_from_height


def build_density_current(
    *,
    nx: int = 257,
    nz: int = 65,
    dx_m: float = 100.0,
    dz_m: float = 100.0,
    center_x_m: float = 0.0,
    center_z_m: float = 3000.0,
    radius_x_m: float = 4000.0,
    radius_z_m: float = 2000.0,
    min_theta_perturbation_k: float = -15.0,
) -> IdealizedCase:
    """Return the canonical cold-block IC from Straka et al. 1993."""

    x = np.arange(int(nx), dtype=np.float64) * float(dx_m)
    z = np.arange(int(nz), dtype=np.float64) * float(dz_m)
    xx, zz = np.meshgrid(x, z)
    radius = np.sqrt(((xx - float(center_x_m)) / float(radius_x_m)) ** 2 + ((zz - float(center_z_m)) / float(radius_z_m)) ** 2)
    theta_prime = np.where(radius <= 1.0, 0.5 * float(min_theta_perturbation_k) * (1.0 + np.cos(np.pi * radius)), 0.0)
    theta = THETA0_K + theta_prime
    pressure = hydrostatic_pressure_from_height(zz)
    arrays = {
        "x_m": x,
        "z_m": z,
        "theta_perturbation_k": theta_prime.astype(np.float64),
        "theta_k": theta.astype(np.float64),
        "pressure_pa": pressure.astype(np.float64),
        "density_kg_m3": dry_density(pressure, theta).astype(np.float64),
        "u_m_s": np.zeros_like(theta, dtype=np.float64),
        "w_m_s": np.zeros_like(theta, dtype=np.float64),
    }
    return IdealizedCase(
        case_id="idealized-density-current-straka-1993",
        reference={
            "name": "Straka et al. 1993 density-current benchmark",
            "published_targets": {"front_speed_m_s": 33.0, "integration_s": 900, "dx_m": 100.0},
            "purpose": "cold-pool propagation and sharp-gradient handling",
        },
        grid={"nx": int(nx), "nz": int(nz), "dx_m": float(dx_m), "dz_m": float(dz_m), "staggering": "2d-xz-mass"},
        parameters={
            "theta0_k": THETA0_K,
            "min_theta_perturbation_k": float(min_theta_perturbation_k),
            "center_x_m": float(center_x_m),
            "center_z_m": float(center_z_m),
            "radius_x_m": float(radius_x_m),
            "radius_z_m": float(radius_z_m),
            "moisture": "dry",
        },
        arrays=arrays,
    )
