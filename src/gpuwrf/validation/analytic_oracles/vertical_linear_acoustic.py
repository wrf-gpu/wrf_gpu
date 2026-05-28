"""Closed-form vertical acoustic-gravity column modes.

Derivation for the oracle:

The reduced operator tested here is the 1-D, flat-column limit of the
split-explicit acoustic equations described in Skamarock et al. (2008),
WRF Technical Note, section 3.2. Strip horizontal advection, map factors,
moisture, terrain, and diffusion from the small-step equations and linearize
about a hydrostatic base state. With vertical velocity ``w``, pressure
perturbation ``p'``, and buoyancy ``b = g theta' / theta0``, the constant-
coefficient column equations are

    d_t w  = -(1 / rho0) d_z p' + b
    d_t p' = -rho0 c_s**2 d_z w
    d_t b  = -N**2 w

where ``c_s = sqrt(gamma R_d T_base)``. A normal mode
``exp(i k_z z - i omega t)`` gives

    omega**2 = c_s**2 k_z**2 + N**2.

The returned fields are the closed-form standing-wave solution for this
dispersion relation. This module intentionally does not integrate the
production scheme, so it can serve as a non-tautological oracle for WRF-shaped
vertical acoustic updates.
"""

from __future__ import annotations

import numpy as np


R_D_J_KG_K = 287.0
GAMMA_DRY_AIR = 1.4
GRAVITY_M_S2 = 9.80665


def vertical_acoustic_mode(
    n_levels: int,
    column_height_m: float,
    theta_base_K: float,
    brunt_vaisala_N_inv_s: float,
    wavelength_m: float,
    initial_amplitude_w_m_s: float,
    times_s: np.ndarray,
) -> dict:
    """Returns a closed-form 1-D vertical acoustic-gravity standing mode.

    ``theta_base_K`` is used as the reference temperature for the sound speed,
    corresponding to a flat reference-pressure column where potential
    temperature and temperature coincide. ``w`` and ``ph_perturbation`` are on
    vertical faces with shape ``(n_times, n_levels + 1)``. The theta
    perturbation is on mass levels with shape ``(n_times, n_levels)``.
    """

    if n_levels < 2:
        raise ValueError("n_levels must be at least 2")
    if column_height_m <= 0.0:
        raise ValueError("column_height_m must be positive")
    if theta_base_K <= 0.0:
        raise ValueError("theta_base_K must be positive")
    if brunt_vaisala_N_inv_s < 0.0:
        raise ValueError("brunt_vaisala_N_inv_s must be non-negative")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("times_s must be a one-dimensional array")

    z_faces = np.linspace(0.0, float(column_height_m), int(n_levels) + 1, dtype=np.float64)
    dz = float(column_height_m) / float(n_levels)
    z_mass = z_faces[:-1] + 0.5 * dz
    k_z = 2.0 * np.pi / float(wavelength_m)
    sound_speed_m_s = np.sqrt(GAMMA_DRY_AIR * R_D_J_KG_K * float(theta_base_K))
    omega_rad_s = np.sqrt((sound_speed_m_s * k_z) ** 2 + float(brunt_vaisala_N_inv_s) ** 2)
    period_s = 2.0 * np.pi / omega_rad_s
    decay_rate_inv_s = 0.0

    face_mode = np.sin(k_z * z_faces)
    mass_mode = np.sin(k_z * z_mass)
    max_abs = float(np.max(np.abs(face_mode)))
    if max_abs <= 0.0:
        raise ValueError("wavelength_m produces a zero face mode on this column")
    face_mode = face_mode / max_abs
    mass_mode = mass_mode / max_abs

    phase = omega_rad_s * times[:, None]
    amplitude = float(initial_amplitude_w_m_s) * np.exp(-decay_rate_inv_s * times[:, None])
    w = amplitude * np.cos(phase) * face_mode[None, :]
    vertical_displacement_m = (
        float(initial_amplitude_w_m_s)
        * np.exp(-decay_rate_inv_s * times[:, None])
        * np.sin(phase)
        / omega_rad_s
        * face_mode[None, :]
    )
    ph_perturbation = GRAVITY_M_S2 * vertical_displacement_m
    theta_perturbation = (
        -float(theta_base_K)
        * float(brunt_vaisala_N_inv_s) ** 2
        * float(initial_amplitude_w_m_s)
        * np.exp(-decay_rate_inv_s * times[:, None])
        * np.sin(phase)
        / (GRAVITY_M_S2 * omega_rad_s)
        * mass_mode[None, :]
    )

    return {
        "t": times,
        "z_w_m": z_faces,
        "z_theta_m": z_mass,
        "k_z_rad_m": k_z,
        "sound_speed_m_s": sound_speed_m_s,
        "omega_rad_s": omega_rad_s,
        "w": w,
        "ph_perturbation": ph_perturbation,
        "theta_perturbation": theta_perturbation,
        "period_s": period_s,
        "decay_rate_inv_s": decay_rate_inv_s,
    }
