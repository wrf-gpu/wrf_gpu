"""Shared helpers for idealized publication-test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


GRAVITY_M_S2 = 9.80665
R_DRY_AIR = 287.05
CP_DRY_AIR = 1004.0
P0_PA = 100000.0
THETA0_K = 300.0


@dataclass(frozen=True)
class IdealizedCase:
    """Small, JSON-friendly container around NumPy initial-condition arrays."""

    case_id: str
    reference: dict[str, Any]
    grid: dict[str, Any]
    parameters: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def stats(self) -> dict[str, Any]:
        return {
            name: {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "min": float(np.nanmin(value)) if value.size else None,
                "max": float(np.nanmax(value)) if value.size else None,
                "finite": bool(np.all(np.isfinite(value))),
            }
            for name, value in sorted(self.arrays.items())
        }

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "reference": self.reference,
            "grid": self.grid,
            "parameters": self.parameters,
            "array_stats": self.stats(),
        }


def hydrostatic_pressure_from_height(z_m: np.ndarray, *, surface_pressure_pa: float = P0_PA) -> np.ndarray:
    """Isothermal dry hydrostatic base pressure used only for analytic ICs."""

    scale_height_m = R_DRY_AIR * THETA0_K / GRAVITY_M_S2
    return surface_pressure_pa * np.exp(-np.asarray(z_m, dtype=np.float64) / scale_height_m)


def dry_density(pressure_pa: np.ndarray, theta_k: np.ndarray) -> np.ndarray:
    """Dry ideal-gas density using potential temperature as the dry test temperature."""

    return np.asarray(pressure_pa, dtype=np.float64) / (R_DRY_AIR * np.asarray(theta_k, dtype=np.float64))


def centered_cosine_bump(radius: np.ndarray) -> np.ndarray:
    """Compact cosine-squared bump equal to zero outside radius <= 1."""

    r = np.asarray(radius, dtype=np.float64)
    return np.where(r <= 1.0, np.cos(0.5 * np.pi * r) ** 2, 0.0)
