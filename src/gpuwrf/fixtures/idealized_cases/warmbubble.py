"""Bryan-Fritsch-style dry warm-bubble initial condition."""

from __future__ import annotations

import numpy as np

from .common import IdealizedCase, THETA0_K, centered_cosine_bump, dry_density, hydrostatic_pressure_from_height


def build_warmbubble(
    *,
    nx: int = 201,
    nz: int = 80,
    dx_m: float = 100.0,
    dz_m: float = 100.0,
    center_x_m: float | None = None,
    center_z_m: float = 2000.0,
    radius_x_m: float = 2000.0,
    radius_z_m: float = 2000.0,
    theta_perturbation_k: float = 2.0,
) -> IdealizedCase:
    """Return a dry warm-bubble IC matching the common Bryan-Fritsch setup.

    The perturbation is a compact cosine-squared ellipsoid in a neutral dry
    atmosphere. This builder emits IC arrays only; it does not claim a model
    integration or WRF parity result.
    """

    if center_x_m is None:
        center_x_m = 0.5 * (int(nx) - 1) * float(dx_m)
    x = np.arange(int(nx), dtype=np.float64) * float(dx_m)
    z = np.arange(int(nz), dtype=np.float64) * float(dz_m)
    xx, zz = np.meshgrid(x, z)
    radius = np.sqrt(((xx - float(center_x_m)) / float(radius_x_m)) ** 2 + ((zz - float(center_z_m)) / float(radius_z_m)) ** 2)
    theta_prime = float(theta_perturbation_k) * centered_cosine_bump(radius)
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
        case_id="idealized-warmbubble-bryan-fritsch-2002",
        reference={
            "name": "Bryan and Fritsch 2002 dry warm-bubble benchmark",
            "purpose": "buoyant response, acoustic stability, symmetry",
            "notes": "Compact dry thermal bubble; WRF em_quarter_ss is the stock-ARW reference path.",
        },
        grid={"nx": int(nx), "nz": int(nz), "dx_m": float(dx_m), "dz_m": float(dz_m), "staggering": "2d-xz-mass"},
        parameters={
            "theta0_k": THETA0_K,
            "theta_perturbation_k": float(theta_perturbation_k),
            "center_x_m": float(center_x_m),
            "center_z_m": float(center_z_m),
            "radius_x_m": float(radius_x_m),
            "radius_z_m": float(radius_z_m),
            "moisture": "dry",
        },
        arrays=arrays,
    )
