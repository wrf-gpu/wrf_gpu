#!/usr/bin/env python3
"""Independent NumPy oracle for WRF nested scalar boundary RK cadence.

No production implementation is imported.  Equations are transcribed from
tracked WRF v4.7.1 ``module_bc_em.F:348-410,658-701``,
``share/module_bc.F:1244-1548``, and ``module_em.F:1653-1775``.
"""

from __future__ import annotations

import math
import numpy as np


SIDES = ("W", "E", "S", "N")
SIDE_INDEX = {name: index for index, name in enumerate(SIDES)}


def _strip(leaf: np.ndarray, side: str, distance: int, z_len: int, side_len: int):
    return leaf[SIDE_INDEX[side], distance, :z_len, :side_len]


def boundary_value(records: np.ndarray, lead_seconds: float, cadence_s: float) -> np.ndarray:
    nrec = int(records.shape[0])
    index = float(lead_seconds) / float(cadence_s)
    lower = min(max(int(math.floor(index)), 0), nrec - 1)
    upper = min(lower + 1, nrec - 1)
    alpha = min(max(index - lower, 0.0), 1.0)
    return (1.0 - alpha) * records[lower] + alpha * records[upper]


def boundary_rate(records: np.ndarray, lead_seconds: float, cadence_s: float) -> np.ndarray:
    if records.shape[0] < 2:
        return np.zeros_like(records[0])
    interval = min(
        max(int(math.ceil(float(lead_seconds) / float(cadence_s))) - 1, 0),
        int(records.shape[0]) - 2,
    )
    return (records[interval + 1] - records[interval]) / float(cadence_s)


def _ring_field(leaf: np.ndarray, z_len: int, ny: int, nx: int) -> np.ndarray:
    target = np.zeros((z_len, ny, nx), dtype=leaf.dtype)
    for distance in range(int(leaf.shape[1])):
        target[:, :, distance] = _strip(leaf, "W", distance, z_len, ny)
        target[:, :, nx - 1 - distance] = _strip(leaf, "E", distance, z_len, ny)
        target[:, distance, :] = _strip(leaf, "S", distance, z_len, nx)
        target[:, ny - 1 - distance, :] = _strip(leaf, "N", distance, z_len, nx)
    return target


def scalar_boundary_tendency(
    scalar: np.ndarray,
    mu_total: np.ndarray,
    c1h: np.ndarray,
    c2h: np.ndarray,
    records: np.ndarray,
    *,
    lead_seconds: float,
    cadence_s: float,
    dt_full: float,
    spec_zone: int,
    relax_zone: int,
    spec_exp: float = 0.0,
) -> np.ndarray:
    """Literal relax_bdy_scalar followed by spec_bdy_scalar."""

    q = np.asarray(scalar)
    nz, ny, nx = q.shape
    coupled = q * (
        np.asarray(c1h)[:, None, None] * np.asarray(mu_total)[None, :, :]
        + np.asarray(c2h)[:, None, None]
    )
    target_leaf = boundary_value(np.asarray(records), lead_seconds, cadence_s)
    target = _ring_field(target_leaf, nz, ny, nx)
    tendency = np.zeros_like(coupled)

    def residual(z, y, x):
        return target[z, y, x] - coupled[z, y, x]

    for distance in range(int(spec_zone), int(relax_zone)):
        loop = distance + 1
        linear = max(
            0.0,
            (float(spec_zone + relax_zone - loop) / float(relax_zone - 1))
            if relax_zone > 1
            else 0.0,
        )
        sponge = math.exp(-(loop - (spec_zone + 1)) * float(spec_exp))
        fcx = 0.1 / float(dt_full) * linear * sponge
        gcx = 1.0 / float(dt_full) / 50.0 * linear * sponge
        for z in range(nz):
            # Y sides own i=distance..nx-1-distance.
            for x in range(distance, nx - distance):
                xm1, xp1 = max(x - 1, 0), min(x + 1, nx - 1)
                y = distance
                r0 = residual(z, y, x)
                lap = (
                    residual(z, y, xm1)
                    + residual(z, y, xp1)
                    + residual(z, y - 1, x)
                    + residual(z, y + 1, x)
                    - 4.0 * r0
                )
                tendency[z, y, x] += fcx * r0 - gcx * lap
                y = ny - 1 - distance
                r0 = residual(z, y, x)
                lap = (
                    residual(z, y, xm1)
                    + residual(z, y, xp1)
                    + residual(z, y + 1, x)
                    + residual(z, y - 1, x)
                    - 4.0 * r0
                )
                tendency[z, y, x] += fcx * r0 - gcx * lap

            # X sides exclude the Y-owned corners.
            for y in range(distance + 1, ny - distance - 1):
                x = distance
                r0 = residual(z, y, x)
                lap = (
                    residual(z, y - 1, x)
                    + residual(z, y + 1, x)
                    + residual(z, y, x - 1)
                    + residual(z, y, x + 1)
                    - 4.0 * r0
                )
                tendency[z, y, x] += fcx * r0 - gcx * lap
                x = nx - 1 - distance
                r0 = residual(z, y, x)
                lap = (
                    residual(z, y - 1, x)
                    + residual(z, y + 1, x)
                    + residual(z, y, x + 1)
                    + residual(z, y, x - 1)
                    - 4.0 * r0
                )
                tendency[z, y, x] += fcx * r0 - gcx * lap

    # spec_bdytend overwrites the outer specified cells with bdy_tend; Y owns
    # corners and the later X loop is tangentially trimmed.
    rate = boundary_rate(np.asarray(records), lead_seconds, cadence_s)
    for distance in range(int(spec_zone)):
        x0, x1 = distance, nx - distance
        tendency[:, distance, x0:x1] = _strip(rate, "S", distance, nz, nx)[:, x0:x1]
        tendency[:, ny - 1 - distance, x0:x1] = _strip(rate, "N", distance, nz, nx)[:, x0:x1]
        y0, y1 = distance + 1, ny - distance - 1
        tendency[:, y0:y1, distance] = _strip(rate, "W", distance, nz, ny)[:, y0:y1]
        tendency[:, y0:y1, nx - 1 - distance] = _strip(rate, "E", distance, nz, ny)[:, y0:y1]
    return tendency


def rk_scalar_sequence(
    scalar_at_step_start: np.ndarray,
    mu_old: np.ndarray,
    mu_new_by_stage: tuple[np.ndarray, ...],
    c1h: np.ndarray,
    c2h: np.ndarray,
    frozen_boundary_tendency: np.ndarray,
    advection_by_stage: tuple[np.ndarray, ...],
    dt_rk_by_stage: tuple[float, ...],
    *,
    spec_zone: int,
    msfty: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Literal three-stage ``rk_update_scalar`` low-storage update.

    WRF applies ``msfty`` to raw scalar advection inside the non-specified
    rectangle, then adds the frozen ``sc_tend`` without that factor.
    """

    q0 = np.asarray(scalar_at_step_start)
    mass_old = np.asarray(c1h)[:, None, None] * np.asarray(mu_old)[None] + np.asarray(c2h)[:, None, None]
    outputs: list[np.ndarray] = []
    for mu_new, adv, dt_rk in zip(
        mu_new_by_stage,
        advection_by_stage,
        dt_rk_by_stage,
        strict=True,
    ):
        tendency = np.zeros_like(np.asarray(adv))
        sz = int(spec_zone)
        y1 = tendency.shape[-2] - sz
        x1 = tendency.shape[-1] - sz
        tendency[..., sz:y1, sz:x1] = (
            np.asarray(adv)[..., sz:y1, sz:x1]
            * np.asarray(msfty)[None, sz:y1, sz:x1]
        )
        tendency += np.asarray(frozen_boundary_tendency)
        mass_new = np.asarray(c1h)[:, None, None] * np.asarray(mu_new)[None] + np.asarray(c2h)[:, None, None]
        outputs.append((mass_old * q0 + float(dt_rk) * tendency) / mass_new)
    return tuple(outputs)


__all__ = [
    "boundary_rate",
    "boundary_value",
    "rk_scalar_sequence",
    "scalar_boundary_tendency",
]
