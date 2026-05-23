"""Single-column MPAS acoustic-step oracle for ADR-023.

This module is intentionally NumPy-only. It does not call the local ADR-023
operator and it does not integrate the analytic R7 oracle. The recurrence below
is a reduced single-column port of MPAS-A 5.3
``src/core_atmosphere/dynamics/mpas_atm_time_integration.F``:

- lines 1589-1651: vertically implicit coefficient and Thomas-recursion setup
- lines 2038-2041: ``resm = (1 - epssm) / (1 + epssm)`` off-centering
- lines 2146-2169: old/new off-centered ``rs``, ``ts``, and ``rw_p`` RHS
- lines 2175-2182: upward sweep and downward back-substitution
- lines 2184-2193: Rayleigh block, kept as the documented zero-``dss`` no-op
- lines 2195-2208: ``wwAvg`` accumulation and ``rho_pp`` / ``rtheta_pp``
  perturbation reconstruction

Scenario setup uses the MPAS idealized squall-line/supercell warm perturbation
from ``src/core_init_atmosphere/mpas_init_atm_cases.F:1657-1690`` when
``scenario="warm_bubble_2km"``. ``scenario="stratified_rest"`` supplies the
hydrostatic zero-perturbation column that should remain motionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CP_D = 1004.0
R_D = 287.0
GRAVITY_M_S2 = 9.80665
THETA_BASE_K = 300.0
RHO_BASE_KG_M3 = 1.0
MPAS_OMEGA_TO_W_METRIC = 1.35


@dataclass(frozen=True)
class _ColumnSetup:
    z_faces_m: np.ndarray
    z_mass_m: np.ndarray
    dz_m: float
    theta_m: np.ndarray
    theta_perturbation: np.ndarray
    tend_rw: np.ndarray


@dataclass(frozen=True)
class _MpasCoefficients:
    cofrz: np.ndarray
    cofwr: np.ndarray
    cofwz: np.ndarray
    coftz: np.ndarray
    cofwt: np.ndarray
    a_tri: np.ndarray
    alpha_tri: np.ndarray
    gamma_tri: np.ndarray
    rdzw: np.ndarray
    zz: np.ndarray


def _one_based(size: int) -> np.ndarray:
    """Allocates a one-based work array so loop bounds match MPAS Fortran."""

    return np.zeros(int(size) + 2, dtype=np.float64)


def _warm_bubble_profile(z_mass_m: np.ndarray, column_height_m: float) -> np.ndarray:
    """Returns the MPAS idealized warm-bubble theta perturbation.

    MPAS source: ``mpas_init_atm_cases.F:1657-1690`` sets ``delt = 3``,
    ``radz = 1500``, ``zcent = 1500`` and applies
    ``delt*cos(.5*pii*rad)**2`` for ``rad < 1``. For this column-slice
    scenario the bubble center is moved to 2 km, matching the scenario name.
    """

    center_m = min(2000.0, 0.5 * float(column_height_m))
    radius_z_m = min(1500.0, max(1.0, 0.25 * float(column_height_m)))
    radius = np.abs((z_mass_m - center_m) / radius_z_m)
    return np.where(radius < 1.0, 3.0 * np.cos(0.5 * np.pi * radius) ** 2, 0.0)


def _build_column_setup(scenario: str, n_levels: int, column_height_m: float) -> _ColumnSetup:
    if n_levels < 3:
        raise ValueError("n_levels must be at least 3 for MPAS tridiagonal coefficients")
    if column_height_m <= 0.0:
        raise ValueError("column_height_m must be positive")

    z_faces = np.linspace(0.0, float(column_height_m), int(n_levels) + 1, dtype=np.float64)
    dz_m = float(column_height_m) / float(n_levels)
    z_mass = z_faces[:-1] + 0.5 * dz_m

    if scenario == "warm_bubble_2km":
        theta_perturbation = _warm_bubble_profile(z_mass, column_height_m)
    elif scenario == "stratified_rest":
        theta_perturbation = np.zeros(int(n_levels), dtype=np.float64)
    else:
        raise ValueError(f"unsupported MPAS column-slice scenario: {scenario!r}")

    theta_m = THETA_BASE_K + theta_perturbation
    buoyancy_mass = GRAVITY_M_S2 * theta_perturbation / THETA_BASE_K
    buoyancy_faces = np.concatenate(
        (
            buoyancy_mass[0:1],
            0.5 * (buoyancy_mass[:-1] + buoyancy_mass[1:]),
            buoyancy_mass[-1:],
        )
    )
    tend_rw = RHO_BASE_KG_M3 * 0.38 * buoyancy_faces
    return _ColumnSetup(z_faces, z_mass, dz_m, theta_m, theta_perturbation, tend_rw)


def _mpas_coefficients(setup: _ColumnSetup, dt_acoustic_s: float, epssm: float) -> _MpasCoefficients:
    """Builds MPAS vertical implicit coefficients.

    Literal source blocks:
    - ``mpas_atm_time_integration.F:1589-1597`` for ``dtseps`` and ``cofrz``.
    - ``mpas_atm_time_integration.F:1602-1624`` for ``cofwr``, ``cofwz``,
      ``coftz``, and ``cofwt``.
    - ``mpas_atm_time_integration.F:1627-1651`` for tridiagonal rows and the
      stored ``alpha_tri`` / ``gamma_tri`` forward-sweep factors.
    """

    n = setup.theta_m.size
    dtseps = 0.5 * float(dt_acoustic_s) * (1.0 + float(epssm))
    rcv = R_D / (CP_D - R_D)
    c2 = CP_D * rcv

    theta = _one_based(n)
    theta[1 : n + 1] = setup.theta_m
    zz = _one_based(n)
    zz[1 : n + 1] = 1.0
    exner = _one_based(n)
    exner[1 : n + 1] = 1.0
    exner_base = _one_based(n)
    exner_base[1 : n + 1] = 1.0
    rho_base = _one_based(n)
    rho_base[1 : n + 1] = RHO_BASE_KG_M3
    rtheta_base = _one_based(n)
    rtheta_base[1 : n + 1] = RHO_BASE_KG_M3 * THETA_BASE_K
    rtheta_pert = _one_based(n)
    rtheta_pert[1 : n + 1] = RHO_BASE_KG_M3 * setup.theta_perturbation
    qtotal = _one_based(n)

    rdzw = _one_based(n)
    rdzw[1 : n + 1] = 1.0 / setup.dz_m
    rdzu = _one_based(n)
    rdzu[2 : n + 1] = 1.0 / setup.dz_m
    fzm = _one_based(n)
    fzp = _one_based(n)
    fzm[2 : n + 1] = 0.5
    fzp[2 : n + 1] = 0.5

    cofrz = _one_based(n)
    cofwr = _one_based(n)
    cofwz = _one_based(n)
    coftz = _one_based(n + 1)
    cofwt = _one_based(n)
    a_tri = _one_based(n)
    b_tri = _one_based(n)
    c_tri = _one_based(n)
    alpha_tri = _one_based(n)
    gamma_tri = _one_based(n)

    for k in range(1, n + 1):
        cofrz[k] = dtseps * rdzw[k]

    for k in range(2, n + 1):
        z_face = fzm[k] * zz[k] + fzp[k] * zz[k - 1]
        exner_face = fzm[k] * exner[k] + fzp[k] * exner[k - 1]
        theta_face = fzm[k] * theta[k] + fzp[k] * theta[k - 1]
        cofwr[k] = 0.5 * dtseps * GRAVITY_M_S2 * z_face
        cofwz[k] = dtseps * c2 * z_face * rdzu[k] * exner_face
        coftz[k] = dtseps * theta_face

    coftz[1] = 0.0
    coftz[n + 1] = 0.0

    for k in range(1, n + 1):
        cofwt[k] = (
            0.5
            * dtseps
            * rcv
            * zz[k]
            * GRAVITY_M_S2
            * rho_base[k]
            * exner[k]
            / ((1.0 + qtotal[k]) * (rtheta_base[k] + rtheta_pert[k]) * exner_base[k])
        )

    a_tri[1] = 0.0
    b_tri[1] = 1.0
    c_tri[1] = 0.0
    gamma_tri[1] = 0.0
    alpha_tri[1] = 0.0

    for k in range(2, n + 1):
        a_tri[k] = (
            -cofwz[k] * coftz[k - 1] * rdzw[k - 1] * zz[k - 1]
            + cofwr[k] * cofrz[k - 1]
            - cofwt[k - 1] * coftz[k - 1] * rdzw[k - 1]
        )
        b_tri[k] = (
            1.0
            + cofwz[k] * (coftz[k] * rdzw[k] * zz[k] + coftz[k] * rdzw[k - 1] * zz[k - 1])
            - coftz[k] * (cofwt[k] * rdzw[k] - cofwt[k - 1] * rdzw[k - 1])
            + cofwr[k] * (cofrz[k] - cofrz[k - 1])
        )
        c_tri[k] = (
            -cofwz[k] * coftz[k + 1] * rdzw[k] * zz[k]
            - cofwr[k] * cofrz[k]
            + cofwt[k] * coftz[k + 1] * rdzw[k]
        )

    for k in range(2, n + 1):
        alpha_tri[k] = 1.0 / (b_tri[k] - a_tri[k] * gamma_tri[k - 1])
        gamma_tri[k] = c_tri[k] * alpha_tri[k]

    return _MpasCoefficients(cofrz, cofwr, cofwz, coftz, cofwt, a_tri, alpha_tri, gamma_tri, rdzw, zz)


def _record_state(
    *,
    step: int,
    dt_acoustic_s: float,
    rw_p: np.ndarray,
    rtheta_pp: np.ndarray,
    rho_pp: np.ndarray,
    ph_perturbation: np.ndarray,
    setup: _ColumnSetup,
    out: dict[str, np.ndarray],
) -> None:
    n = setup.theta_m.size
    out["t"][step] = float(step) * float(dt_acoustic_s)
    out["w"][step, :] = rw_p[1 : n + 2] / MPAS_OMEGA_TO_W_METRIC
    out["theta_perturbation"][step, :] = rtheta_pp[1 : n + 1] / RHO_BASE_KG_M3
    out["rho_perturbation"][step, :] = rho_pp[1 : n + 1]
    out["ph_perturbation"][step, :] = ph_perturbation[1 : n + 2]
    out["mu_perturbation"][step] = GRAVITY_M_S2 * setup.dz_m * float(np.sum(rho_pp[1 : n + 1]))


def mpas_column_slice(
    scenario: str,
    n_levels: int,
    column_height_m: float,
    dt_acoustic_s: float,
    n_substeps: int,
    epssm: float = 0.1,
) -> dict:
    """Returns a single-column MPAS-derived acoustic trajectory.

    The returned arrays have the contract shapes:
    ``t`` ``(n_substeps + 1,)``, ``w`` and ``ph_perturbation``
    ``(n_substeps + 1, n_levels + 1)``, ``theta_perturbation`` and
    ``rho_perturbation`` ``(n_substeps + 1, n_levels)``, and
    ``mu_perturbation`` ``(n_substeps + 1,)``.

    Equation block citations:
    - MPAS lines 2038-2041 define ``resm``.
    - MPAS lines 2146-2151 build ``rs`` and ``ts`` from old perturbations and
      off-centered ``rw_p`` divergence.
    - MPAS lines 2155-2157 accumulate old-side ``wwAvg``.
    - MPAS lines 2160-2169 build the implicit ``rw_p`` RHS.
    - MPAS lines 2175-2182 perform upward sweep and downward substitution.
    - MPAS lines 2184-2193 are represented with ``dss == 0`` so Rayleigh
      damping is an exact no-op in this validation slice.
    - MPAS lines 2195-2208 accumulate new-side ``wwAvg`` and reconstruct
      ``rho_pp`` and ``rtheta_pp`` from the solved ``rw_p``.
    """

    if dt_acoustic_s <= 0.0:
        raise ValueError("dt_acoustic_s must be positive")
    if n_substeps < 1:
        raise ValueError("n_substeps must be at least 1")
    if not (-1.0 < epssm < 1.0):
        raise ValueError("epssm must be between -1 and 1")

    setup = _build_column_setup(scenario, int(n_levels), float(column_height_m))
    coefficients = _mpas_coefficients(setup, float(dt_acoustic_s), float(epssm))
    n = setup.theta_m.size
    resm = (1.0 - float(epssm)) / (1.0 + float(epssm))

    rw_p = _one_based(n + 1)
    rho_pp = _one_based(n)
    rtheta_pp = _one_based(n)
    rtheta_pp[1 : n + 1] = RHO_BASE_KG_M3 * setup.theta_perturbation
    ph_perturbation = _one_based(n + 1)
    tend_rho = _one_based(n)
    tend_rt = _one_based(n)
    tend_rw = _one_based(n + 1)
    tend_rw[1 : n + 2] = setup.tend_rw
    ww_avg = _one_based(n + 1)
    dss = _one_based(n)
    rw = _one_based(n + 1)
    rw_save = _one_based(n + 1)

    out = {
        "t": np.zeros(int(n_substeps) + 1, dtype=np.float64),
        "w": np.zeros((int(n_substeps) + 1, n + 1), dtype=np.float64),
        "theta_perturbation": np.zeros((int(n_substeps) + 1, n), dtype=np.float64),
        "ph_perturbation": np.zeros((int(n_substeps) + 1, n + 1), dtype=np.float64),
        "mu_perturbation": np.zeros(int(n_substeps) + 1, dtype=np.float64),
        "rho_perturbation": np.zeros((int(n_substeps) + 1, n), dtype=np.float64),
    }
    _record_state(
        step=0,
        dt_acoustic_s=float(dt_acoustic_s),
        rw_p=rw_p,
        rtheta_pp=rtheta_pp,
        rho_pp=rho_pp,
        ph_perturbation=ph_perturbation,
        setup=setup,
        out=out,
    )

    for step in range(1, int(n_substeps) + 1):
        rw_prev = rw_p.copy()
        rs = _one_based(n)
        ts = _one_based(n)

        for k in range(1, n + 1):
            rs[k] = (
                rho_pp[k]
                + float(dt_acoustic_s) * tend_rho[k]
                + rs[k]
                - coefficients.cofrz[k] * resm * (rw_p[k + 1] - rw_p[k])
            )
            ts[k] = (
                rtheta_pp[k]
                + float(dt_acoustic_s) * tend_rt[k]
                + ts[k]
                - resm
                * coefficients.rdzw[k]
                * (coefficients.coftz[k + 1] * rw_p[k + 1] - coefficients.coftz[k] * rw_p[k])
            )

        for k in range(2, n + 1):
            ww_avg[k] = ww_avg[k] + 0.5 * (1.0 - float(epssm)) * rw_p[k]

        for k in range(2, n + 1):
            rw_p[k] = (
                rw_p[k]
                + float(dt_acoustic_s) * tend_rw[k]
                - coefficients.cofwz[k]
                * (
                    (coefficients.zz[k] * ts[k] - coefficients.zz[k - 1] * ts[k - 1])
                    + resm
                    * (
                        coefficients.zz[k] * rtheta_pp[k]
                        - coefficients.zz[k - 1] * rtheta_pp[k - 1]
                    )
                )
                - coefficients.cofwr[k] * ((rs[k] + rs[k - 1]) + resm * (rho_pp[k] + rho_pp[k - 1]))
                + coefficients.cofwt[k] * (ts[k] + resm * rtheta_pp[k])
                + coefficients.cofwt[k - 1] * (ts[k - 1] + resm * rtheta_pp[k - 1])
            )

        rw_p[1] = 0.0
        rw_p[n + 1] = 0.0

        for k in range(2, n + 1):
            rw_p[k] = (rw_p[k] - coefficients.a_tri[k] * rw_p[k - 1]) * coefficients.alpha_tri[k]

        for k in range(n, 0, -1):
            rw_p[k] = rw_p[k] - coefficients.gamma_tri[k] * rw_p[k + 1]

        for k in range(2, n + 1):
            z_face = 0.5 * (coefficients.zz[k] + coefficients.zz[k - 1])
            rho_face = RHO_BASE_KG_M3
            rw_p[k] = (
                rw_p[k]
                + (rw_save[k] - rw[k])
                - float(dt_acoustic_s) * dss[k] * z_face * rho_face * out["w"][step - 1, k - 1]
            ) / (1.0 + float(dt_acoustic_s) * dss[k]) - (rw_save[k] - rw[k])

        rw_p[1] = 0.0
        rw_p[n + 1] = 0.0

        for k in range(2, n + 1):
            ww_avg[k] = ww_avg[k] + 0.5 * (1.0 + float(epssm)) * rw_p[k]

        for k in range(1, n + 1):
            rho_pp[k] = rs[k] - coefficients.cofrz[k] * (rw_p[k + 1] - rw_p[k])
            rtheta_pp[k] = ts[k] - coefficients.rdzw[k] * (
                coefficients.coftz[k + 1] * rw_p[k + 1] - coefficients.coftz[k] * rw_p[k]
            )

        w_old = rw_prev[1 : n + 2] / MPAS_OMEGA_TO_W_METRIC
        w_new = rw_p[1 : n + 2] / MPAS_OMEGA_TO_W_METRIC
        ph_perturbation[1 : n + 2] = ph_perturbation[1 : n + 2] + GRAVITY_M_S2 * float(dt_acoustic_s) * (
            0.5 * (1.0 - float(epssm)) * w_old + 0.5 * (1.0 + float(epssm)) * w_new
        )

        _record_state(
            step=step,
            dt_acoustic_s=float(dt_acoustic_s),
            rw_p=rw_p,
            rtheta_pp=rtheta_pp,
            rho_pp=rho_pp,
            ph_perturbation=ph_perturbation,
            setup=setup,
            out=out,
        )

    return out
