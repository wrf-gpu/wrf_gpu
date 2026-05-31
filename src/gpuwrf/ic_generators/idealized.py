"""Idealized Skamarock warm-bubble and Straka density-current validation.

The builders in this module create dry 2-D x-z slabs, expand them into the
project's one-row C-grid ``State`` shape, and run the operational RK/acoustic
path through ``_physics_boundary_step`` with physics and boundaries disabled.
Artifacts are deliberately small JSON/Markdown/PPM files so the sprint proof
objects do not depend on optional plotting packages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace as dataclass_replace
from functools import partial
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _physics_boundary_step,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)

CaseName = Literal["warm_bubble", "density_current"]

# Use the SAME gravity the dycore ``advance_w`` solve uses (9.81), so the
# initial (mu, p, ph) column is in discrete hydrostatic balance with the
# acoustic solver and ``calc_p_rho`` diagnoses ~zero perturbation at rest.
GRAVITY_M_S2 = 9.81
R_DRY_AIR = 287.0
CP_DRY_AIR = 1004.0
CV_DRY_AIR = CP_DRY_AIR - R_DRY_AIR
P0_PA = 100000.0
THETA0_K = 300.0


def _alpha_dry(theta_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """Dry specific volume ``alt`` from the WRF EOS (matches acoustic_wrf).

    ``alt = (R_d/p0) * theta * (p/p0)^(cv/cp)``.
    """

    theta_k = np.asarray(theta_k, dtype=np.float64)
    pressure_pa = np.asarray(pressure_pa, dtype=np.float64)
    return (R_DRY_AIR / P0_PA) * theta_k * (pressure_pa / P0_PA) ** (-CV_DRY_AIR / CP_DRY_AIR)


def _uniform_z_hydrostatic_base(
    z_face_m: np.ndarray,
    theta0_k: float,
    *,
    p_surface_pa: float = P0_PA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build an eta-coordinate hydrostatic base column on uniform-z faces.

    A neutral (constant-θ) dry column is integrated in physical height so the
    z-grid stays uniform (which keeps the front/centroid diagnostics meaningful),
    while the returned eta levels make the WRF mass-coordinate metrics
    (``dnw = |Δeta|``) discretely consistent with that column.

    Returns ``(eta_levels, p_mass, ph_face, mu)``:

    * ``p_face(k)`` is integrated from ``dp = -g dz / alpha`` (hydrostatic) with
      ``alpha(z)`` the dry specific volume at the local pressure and ``θ0``.
    * ``mu = p_face(surface) - p_face(top)``; ``eta(k) = (p_face(k)-p_top)/mu``.
    * ``ph_face(k) = g * z_face(k)`` (flat terrain, uniform z).
    * ``p_mass(k) = 0.5*(p_face(k)+p_face(k+1))``.
    """

    z_face = np.asarray(z_face_m, dtype=np.float64)
    nz = int(z_face.size) - 1
    p_face = np.zeros(nz + 1, dtype=np.float64)
    p_face[0] = float(p_surface_pa)
    for k in range(nz):
        dz = z_face[k + 1] - z_face[k]
        # midpoint specific volume for a second-order hydrostatic integration.
        alpha_lo = _alpha_dry(np.array([theta0_k]), np.array([p_face[k]]))[0]
        p_mid = p_face[k] - 0.5 * GRAVITY_M_S2 * dz / alpha_lo
        alpha_mid = _alpha_dry(np.array([theta0_k]), np.array([p_mid]))[0]
        p_face[k + 1] = p_face[k] - GRAVITY_M_S2 * dz / alpha_mid
    p_top = float(p_face[-1])
    mu = float(p_face[0] - p_top)
    eta_levels = (p_face - p_top) / mu  # 1.0 at surface, 0.0 at top
    ph_face = GRAVITY_M_S2 * z_face
    p_mass = 0.5 * (p_face[:-1] + p_face[1:])
    return eta_levels, p_mass, ph_face, mu

REFERENCE_URLS = {
    "warm_bubble": [
        "https://www2.mmm.ucar.edu/people/skamarock/Papers/cv_20.pdf",
    ],
    "density_current": [
        "https://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html",
        "https://journals.ametsoc.org/view/journals/mwre/141/4/mwr-d-12-00144.1.xml",
    ],
}


@dataclass(frozen=True)
class NumpyIdealizedCase:
    """CPU-side analytic initial condition before JAX state conversion."""

    case_name: CaseName
    case_id: str
    x_m: np.ndarray
    z_m: np.ndarray
    z_face_m: np.ndarray
    theta_prime_k: np.ndarray
    theta_k: np.ndarray
    pressure_pa: np.ndarray
    ph_total_m2_s2: np.ndarray
    mu_base_pa: np.ndarray
    parameters: dict[str, Any]
    reference: dict[str, Any]
    snapshot_seconds: tuple[float, ...]
    end_seconds: float
    dt_s: float
    dx_m: float
    dz_m: float
    # Eta levels and model-top pressure for the WRF mass-coordinate metrics so the
    # IC is discretely hydrostatic (set by the eta-hydrostatic base builder).
    eta_levels: np.ndarray | None = None
    p_top_pa: float | None = None

    @property
    def nx(self) -> int:
        return int(self.x_m.size)

    @property
    def nz(self) -> int:
        return int(self.z_m.size)


@dataclass(frozen=True)
class IdealizedSetup:
    """Operational JAX setup for one idealized validation case."""

    numpy_case: NumpyIdealizedCase
    grid: GridSpec
    state: State
    namelist: OperationalNamelist
    device: str


@dataclass(frozen=True)
class IdealizedRunResult:
    """Summary returned by the runner and consumed by tests/proofs."""

    case_name: CaseName
    verdict: str
    status: str
    proof_json: Path
    proof_markdown: Path
    plot_paths: tuple[Path, ...]
    checks: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]


class IdealizedCaseBlocked(RuntimeError):
    """Raised when the requested GPU execution path is unavailable."""


def _stable_pressure(z_m: np.ndarray, theta0_k: float = THETA0_K) -> np.ndarray:
    """Hydrostatic pressure for a neutral dry potential-temperature profile."""

    exner = 1.0 - GRAVITY_M_S2 * np.asarray(z_m, dtype=np.float64) / (CP_DRY_AIR * float(theta0_k))
    exner = np.maximum(exner, 0.05)
    return P0_PA * exner ** (CP_DRY_AIR / R_DRY_AIR)


def _centered_cosine(radius: np.ndarray) -> np.ndarray:
    radius = np.asarray(radius, dtype=np.float64)
    return np.where(radius <= 1.0, 0.5 * (1.0 + np.cos(np.pi * radius)), 0.0)


def _stats(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    return {
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "finite": bool(np.all(np.isfinite(array))),
        "min": _safe_float(np.nanmin(array)) if array.size else None,
        "max": _safe_float(np.nanmax(array)) if array.size else None,
    }


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def build_warm_bubble_numpy() -> NumpyIdealizedCase:
    """Build the sprint-contracted Skamarock/Wicker rising-thermal IC."""

    dx_m = 250.0
    dz_m = 250.0
    nx = 80
    nz = 40
    x_m = (np.arange(nx, dtype=np.float64) + 0.5) * dx_m
    z_m = (np.arange(nz, dtype=np.float64) + 0.5) * dz_m
    z_face_m = np.arange(nz + 1, dtype=np.float64) * dz_m
    xx, zz = np.meshgrid(x_m, z_m)
    radius = np.sqrt(((xx - 10000.0) / 2000.0) ** 2 + ((zz - 2000.0) / 2000.0) ** 2)
    theta_prime = 2.0 * _centered_cosine(radius)
    theta_prime *= 2.0 / float(np.max(theta_prime))
    # Eta-hydrostatic neutral base (θ0=300) on the uniform-z faces.  Base p/ph
    # are shared by every column; the θ' bubble breaks balance via buoyancy only.
    eta_levels, p_mass_1d, ph_face_1d, mu = _uniform_z_hydrostatic_base(z_face_m, THETA0_K)
    p_top_pa = float(P0_PA - mu)
    pressure = np.broadcast_to(p_mass_1d[:, None], theta_prime.shape).copy()
    ph_total = np.broadcast_to(ph_face_1d[:, None], (nz + 1, nx)).copy()
    mu_base = np.ones((1, nx), dtype=np.float64) * mu
    return NumpyIdealizedCase(
        case_name="warm_bubble",
        case_id="skamarock-wicker-1998-rising-thermal-contract-f2",
        x_m=x_m,
        z_m=z_m,
        z_face_m=z_face_m,
        eta_levels=eta_levels.astype(np.float64),
        p_top_pa=p_top_pa,
        theta_prime_k=theta_prime.astype(np.float64),
        theta_k=(THETA0_K + theta_prime).astype(np.float64),
        pressure_pa=pressure.astype(np.float64),
        ph_total_m2_s2=ph_total.astype(np.float64),
        mu_base_pa=mu_base,
        parameters={
            "domain_width_m": 20000.0,
            "domain_height_m": 10000.0,
            "dx_m": dx_m,
            "dz_m": dz_m,
            "theta0_k": THETA0_K,
            "theta_perturbation_max_k": 2.0,
            "bubble_center_x_m": 10000.0,
            "bubble_center_z_m": 2000.0,
            "bubble_radius_m": 2000.0,
            "initial_u_v_w_m_s": 0.0,
        },
        reference={
            "description": "Dry rising thermal: coherent mushroom-like rise by 500-1000 s, bounded theta prime, O(10 m/s) vertical velocity, horizontal symmetry.",
            "sources": REFERENCE_URLS["warm_bubble"],
        },
        snapshot_seconds=(100.0, 250.0, 500.0),
        end_seconds=500.0,
        dt_s=0.1,
        dx_m=dx_m,
        dz_m=dz_m,
    )


def build_density_current_numpy() -> NumpyIdealizedCase:
    """Build the sprint-contracted Straka et al. 1993 density-current IC."""

    dx_m = 100.0
    dz_m = 100.0
    nx = 500
    nz = 60
    x_m = -25000.0 + (np.arange(nx, dtype=np.float64) + 0.5) * dx_m
    z_m = (np.arange(nz, dtype=np.float64) + 0.5) * dz_m
    z_face_m = np.arange(nz + 1, dtype=np.float64) * dz_m
    xx, zz = np.meshgrid(x_m, z_m)
    radius = np.sqrt((xx / 4000.0) ** 2 + ((zz - 3000.0) / 2000.0) ** 2)
    theta_prime = -15.0 * _centered_cosine(radius)
    theta_prime *= 15.0 / abs(float(np.min(theta_prime)))
    # Eta-hydrostatic neutral base (θ0=300) on the uniform-z faces.
    eta_levels, p_mass_1d, ph_face_1d, mu = _uniform_z_hydrostatic_base(z_face_m, THETA0_K)
    p_top_pa = float(P0_PA - mu)
    pressure = np.broadcast_to(p_mass_1d[:, None], theta_prime.shape).copy()
    ph_total = np.broadcast_to(ph_face_1d[:, None], (nz + 1, nx)).copy()
    mu_base = np.ones((1, nx), dtype=np.float64) * mu
    return NumpyIdealizedCase(
        case_name="density_current",
        case_id="straka-1993-density-current-contract-f2",
        x_m=x_m,
        z_m=z_m,
        z_face_m=z_face_m,
        eta_levels=eta_levels.astype(np.float64),
        p_top_pa=p_top_pa,
        theta_prime_k=theta_prime.astype(np.float64),
        theta_k=(THETA0_K + theta_prime).astype(np.float64),
        pressure_pa=pressure.astype(np.float64),
        ph_total_m2_s2=ph_total.astype(np.float64),
        mu_base_pa=mu_base,
        parameters={
            "domain_width_m": 50000.0,
            "domain_height_m": 6000.0,
            "dx_m": dx_m,
            "dz_m": dz_m,
            "theta0_k": THETA0_K,
            "theta_perturbation_min_k": -15.0,
            "bubble_center_x_m": 0.0,
            "bubble_center_z_m": 3000.0,
            "bubble_radius_x_m": 4000.0,
            "bubble_radius_z_m": 2000.0,
            "initial_u_v_w_m_s": 0.0,
        },
        reference={
            "description": "Cold-pool density current: about three rotors and front position near 15 km by 900 s on 100 m grids.",
            "sources": REFERENCE_URLS["density_current"],
        },
        snapshot_seconds=(900.0,),
        end_seconds=900.0,
        dt_s=0.1,
        dx_m=dx_m,
        dz_m=dz_m,
    )


def _device_inventory() -> dict[str, Any]:
    try:
        devices = jax.devices()
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return {"ok": False, "error": repr(exc), "devices": [], "gpu_devices": []}
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    return {
        "ok": True,
        "devices": [str(device) for device in devices],
        "gpu_devices": [str(device) for device in gpu_devices],
        "default_backend": jax.default_backend(),
    }


def _select_device(*, require_gpu: bool) -> jax.Device:
    inventory = _device_inventory()
    if not inventory.get("ok"):
        raise IdealizedCaseBlocked(f"JAX device discovery failed: {inventory.get('error')}")
    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    if gpus:
        return gpus[0]
    if require_gpu:
        raise IdealizedCaseBlocked("JAX GPU backend is not visible; idealized dycore runs require GPU")
    return jax.devices("cpu")[0]


# F7-B is fp64-correctness-only: build every idealized field in float64.  The
# fp32-gated operational matrix (ADR-007) loses the warm-bubble 2 K perturbation
# on a 300 K base and detonates the acoustic solve; perf downcast is F7-perf.
def _put(value: np.ndarray | float, field: str, device: jax.Device) -> jax.Array:
    del field
    return jax.device_put(jnp.asarray(value, dtype=jnp.float64), device)


def _zeros(shape: tuple[int, ...], field: str, device: jax.Device) -> jax.Array:
    del field
    return jax.device_put(jnp.zeros(shape, dtype=jnp.float64), device)


def _ones(shape: tuple[int, ...], field: str, device: jax.Device) -> jax.Array:
    del field
    return jax.device_put(jnp.ones(shape, dtype=jnp.float64), device)


def _make_grid(case: NumpyIdealizedCase, device: jax.Device) -> GridSpec:
    # Use the eta levels from the hydrostatic base builder so the WRF mass
    # metrics (dnw = |Δeta|) are discretely consistent with the IC column.
    eta_source = (
        jnp.asarray(case.eta_levels, dtype=jnp.float64)
        if case.eta_levels is not None
        else jnp.linspace(1.0, 0.0, case.nz + 1, dtype=jnp.float64)
    )
    eta_levels = jax.device_put(eta_source, device)
    top_pressure_pa = float(case.p_top_pa) if case.p_top_pa is not None else float(case.parameters["domain_height_m"])
    terrain_height = jax.device_put(jnp.zeros((1, case.nx), dtype=jnp.float64), device)
    projection = Projection("lambert", 0.0, 0.0, float(case.dx_m), float(case.dx_m), case.nx, 1)
    terrain = TerrainProvenance(
        source_path=f"idealized:{case.case_id}",
        sha256="analytic-f2",
        shape=(1, case.nx),
        units="m",
        projection_transform="flat-xz-slab",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", case.nz, float(case.parameters["domain_height_m"]), eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999,
        interpolation="linear",
        restart_compatible=False,
    )
    metrics = DycoreMetrics.flat(
        ny=1,
        nx=case.nx,
        nz=case.nz,
        eta_levels=eta_levels,
        top_pressure_pa=top_pressure_pa,
        provenance=f"analytic-f2-{case.case_name}",
    )
    # WRF idealized cases default to a PURE SIGMA coordinate (hybrid_opt=0):
    # c1h=c1f=1, c2h=c2f=0 (module_initialize_ideal.F).  DycoreMetrics.flat uses a
    # HYBRID c1f=eta which makes the top face dry mass (c1f[nz]*mut+c2f[nz]) vanish
    # and produces a singular calc_coef_w tridiagonal (gamma=inf).  Override to
    # pure sigma so the top-face mass is mut (nonzero), matching the f7a oracle and
    # the eta-hydrostatic IC builder (p_face = eta*mu + p_top).
    nz = case.nz
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    metrics = DycoreMetrics(
        msftx=metrics.msftx, msfty=metrics.msfty, msfux=metrics.msfux, msfuy=metrics.msfuy,
        msfvx=metrics.msfvx, msfvy=metrics.msfvy,
        c1h=one_h, c2h=zero_h, c3h=one_h, c4h=zero_h,
        c1f=one_f, c2f=zero_f, c3f=one_f, c4f=zero_f,
        dn=metrics.dn, dnw=metrics.dnw, rdn=metrics.rdn, rdnw=metrics.rdnw,
        cf1=metrics.cf1, cf2=metrics.cf2, cf3=metrics.cf3, fnm=metrics.fnm, fnp=metrics.fnp,
        dzdx=metrics.dzdx, dzdy=metrics.dzdy, dzdx_u=metrics.dzdx_u, dzdy_v=metrics.dzdy_v,
        # Idealized non-rotating frame: carry the f=e=sina=0, cosa=1 defaults so the
        # large-step Coriolis term is identically zero and the dycore gates stay
        # bit-identical to the f-free core.
        f=metrics.f, e=metrics.e, sina=metrics.sina, cosa=metrics.cosa,
        p_top=metrics.p_top, provenance=f"analytic-f2-{case.case_name}-pure-sigma",
    )
    return GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta_levels,
        terrain_height=terrain_height,
        metrics=metrics,
        halo_width=2,
        staggering="c-grid",
    )


def _make_tendencies(grid: GridSpec, device: jax.Device) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    return Tendencies(
        u=_zeros((nz, ny, nx + 1), "u", device),
        v=_zeros((nz, ny + 1, nx), "v", device),
        w=_zeros((nz + 1, ny, nx), "w", device),
        theta=_zeros((nz, ny, nx), "theta", device),
        qv=_zeros((nz, ny, nx), "qv", device),
        p=_zeros((nz, ny, nx), "p", device),
        ph=_zeros((nz + 1, ny, nx), "ph", device),
        mu=_zeros((ny, nx), "mu", device),
    )


def _make_state(case: NumpyIdealizedCase, grid: GridSpec, device: jax.Device) -> State:
    shapes = _state_field_shapes(grid)
    fields: dict[str, jax.Array] = {}
    for name, shape in shapes.items():
        fields[name] = _zeros(shape, name, device)

    theta = case.theta_k[:, None, :]
    pressure = case.pressure_pa[:, None, :]
    mu_total = np.broadcast_to(case.mu_base_pa, (1, case.nx)).copy()
    zero_mass = np.zeros((case.nz, 1, case.nx), dtype=np.float64)

    # WRF fixed-mass hydrostatic rebalance of the geopotential after the theta
    # perturbation (module_initialize_ideal.F:1103-1130 quarter_ss / :1278-1313
    # grav2d_x).  The column dry mass does NOT change (mu' = 0, verified below);
    # WRF perturbs theta, recomputes the full inverse density from the EOS, and
    # re-integrates ph_1/ph_2/ph0 hydrostatically at fixed mass:
    #
    #   alt_full = (R_d/p0)*(t_1+t0)*((p+pb)/p0)^cvpm           (:1113-1116)
    #   al       = alt_full - alb                                (:1117)
    #   phb(k+1) = phb(k) - dnw(k)*(c1h*mub+c2h)*alb(k)          (base, :982)
    #   ph'(k+1) = ph'(k) - dnw(k)*[ (c1h*mub+c2h + c1h*mu')*al(k)
    #                                + c1h*mu'*alb(k) ]          (:1124-1129)
    #
    # with ph'(1) = 0 at the lower boundary (:1056) and ph0 = phb + ph'.  This is
    # the ONLY change the bubble makes to the geometry; the buoyancy that lifts
    # the thermal is then the in-solver c2a*alt*t_2ave term in advance_w
    # (module_small_step_em.F:1486-1489), NOT a synthetic ph/p source.
    #
    # The JAX pure-sigma grid has c1h=1, c2h=0 and mu'=0, so (c1h*mub+c2h)=mu and
    # the recurrence collapses to ph'(k+1)=ph'(k) - dnw(k)*mu*al(k).  F7G uses the
    # WRF-SIGNED ``dnw=znw(k+1)-znw(k)`` (negative for normal eta), so the equation
    # is written literally as WRF (module_initialize_ideal.F:1124-1129 / :1308-1313)
    # ``ph'(k+1)=ph'(k) - wrf_dnw(k)*(...)`` -- NOT hidden behind abs(dnw).  This is
    # numerically identical to the previous +|dnw| form (since -wrf_dnw=+|dnw|) but
    # is now the exact discrete inverse of the WRF-signed calc_p_rho_phi diagnostic
    # the dycore applies (gpt-council-findings.md Q1; AC1 round-trip).
    eta = np.asarray(case.eta_levels, dtype=np.float64)
    wrf_dnw = eta[1:] - eta[:-1]  # (nz,) WRF-signed dnw (negative for normal eta)
    mu = float(case.mu_base_pa[0, 0])
    p_mass = case.pressure_pa  # (nz, nx) base hydrostatic mass-level pressure (p+pb)
    mass_h = mu  # c1h*mub + c2h with c1h=1, c2h=0, mub=mu (pure sigma)
    # Full inverse density alt = EOS(theta_full, p+pb) and base alb = EOS(theta0).
    alt_full = _alpha_dry(case.theta_k, p_mass)  # (nz, nx)
    alb = _alpha_dry(np.full(case.nz, THETA0_K), p_mass[:, 0])  # (nz,) neutral base
    al = alt_full - alb[:, None]  # (nz, nx) WRF grid%al = alt - alb (:1117)
    # Base geopotential phb from alb (WRF :982); ph'(1)=0 at the lower boundary.
    phb_col = np.zeros(case.nz + 1, dtype=np.float64)
    ph_pert_col = np.zeros((case.nz + 1, case.nx), dtype=np.float64)
    for k in range(case.nz):
        phb_col[k + 1] = phb_col[k] - wrf_dnw[k] * mass_h * alb[k]
        ph_pert_col[k + 1, :] = ph_pert_col[k, :] - wrf_dnw[k] * mass_h * al[k, :]
    ph_total = (phb_col[:, None] + ph_pert_col)[:, None, :]  # ph0 = phb + ph'
    ph_pert = ph_pert_col[:, None, :]

    # F7G start_em-equivalent post-init recompute (start_em.F:819-868): WRF derives
    # the diagnostic perturbation pressure ``grid%p`` from the rebalanced ph_1, the
    # WRF-signed calc_p_rho_phi al inverse, and the EOS BEFORE the first RK tendency
    # so that rk_tendency's once-per-stage pg_buoy_w(grid%p) has the real θ′ pressure
    # structure to act on.  Previously p_perturbation was left at 0, so the stage
    # grid%p (hence pg_buoy_w) was zero and the bubble was unforced.  With c1h=1,
    # c2h=0, mu'=0 (pure sigma fixed-mass) the al inverse reduces to
    #   al(k) = -rdnw(k)*(ph'(k+1)-ph'(k))/mu     [signed rdnw<0]
    # which by construction equals al_init = alt_full-alb (AC1 round-trip), and
    #   p'(k) = p0*((R_d*(t0+t')*qvf)/(p0*(al+alb)))^(cp/cv) - pb(k).
    wrf_rdnw = 1.0 / wrf_dnw  # (nz,) signed rdnw (negative)
    al_diag = -(wrf_rdnw[:, None] * (ph_pert_col[1:, :] - ph_pert_col[:-1, :])) / mass_h  # (nz, nx)
    alt_diag = al_diag + alb[:, None]  # full inverse density
    cpovcv = CP_DRY_AIR / CV_DRY_AIR
    pb_col = p_mass  # (nz, nx) base hydrostatic pressure (neutral θ0 column)
    p_full = P0_PA * ((R_DRY_AIR * case.theta_k) / (P0_PA * alt_diag)) ** cpovcv  # (nz, nx)
    p_pert = (p_full - pb_col)[:, None, :]  # (nz, 1, nx) WRF grid%p = p_total - pb
    p_total_field = (pb_col[:, None, :] + p_pert)  # already (nz,1,nx) via p_pert broadcast
    p_total_field = pb_col[:, None, :] + p_pert

    fields.update(
        {
            "theta": _put(theta, "theta", device),
            "p": _put(p_total_field, "p", device),
            "p_total": _put(p_total_field, "p_total", device),
            "p_perturbation": _put(p_pert, "p_perturbation", device),
            "ph": _put(ph_total, "ph", device),
            "ph_total": _put(ph_total, "ph_total", device),
            "ph_perturbation": _put(ph_pert, "ph_perturbation", device),
            "mu": _put(mu_total, "mu", device),
            "mu_total": _put(mu_total, "mu_total", device),
            "mu_perturbation": _put(np.zeros((1, case.nx), dtype=np.float64), "mu_perturbation", device),
            "Ni": _put(np.ones((case.nz, 1, case.nx), dtype=np.float64) * 1.0e5, "Ni", device),
            "Nr": _put(np.ones((case.nz, 1, case.nx), dtype=np.float64) * 1.0e5, "Nr", device),
            "xland": _ones((1, case.nx), "xland", device),
            "mavail": _put(np.ones((1, case.nx), dtype=np.float64) * 0.2, "mavail", device),
            "roughness_m": _put(np.ones((1, case.nx), dtype=np.float64) * 0.05, "roughness_m", device),
            "t_skin": _put(np.ones((1, case.nx), dtype=np.float64) * THETA0_K, "t_skin", device),
            "rhosfc": _ones((1, case.nx), "rhosfc", device),
        }
    )
    return State(**fields)


def _build_setup(case: NumpyIdealizedCase, *, require_gpu: bool = True) -> IdealizedSetup:
    device = _select_device(require_gpu=require_gpu)
    grid = _make_grid(case, device)
    state = _make_state(case, grid, device)
    tendencies = _make_tendencies(grid, device)
    # Straka et al. (1993) defines the reference solution with constant ν=75 m²/s
    # on u, v, θ; the dry rising-thermal benchmark uses weak numerical diffusion.
    const_nu = 75.0 if case.case_name == "density_current" else 0.0
    namelist = OperationalNamelist.from_grid(
        grid,
        tendencies=tendencies,
        metrics=grid.metrics,
        dt_s=float(case.dt_s),
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
        disable_guards=True,
        force_fp64=True,
        use_flux_advection=True,
        const_nu_m2_s=const_nu,
    )
    # Rigid lid + WRF top damping for the bounded idealized box: without the
    # rigid lid the open-top w accumulates a spurious top-face mode; WRF
    # idealized cases run with a rigid lid and an upper Rayleigh layer
    # (damp_opt=3, dampcoef=0.2, zdamp top ~3 km).  Cited WRF coefficients,
    # not a masking clamp.
    namelist = dataclass_replace(
        namelist,
        run_physics=False,
        run_boundary=False,
        top_lid=True,
        w_damping=1,
        damp_opt=3,
        dampcoef=0.2,
        zdamp=3000.0,
    )
    return IdealizedSetup(case, grid, state, namelist, str(device))


def build_warm_bubble_setup(*, require_gpu: bool = True) -> IdealizedSetup:
    """Return a JAX operational setup for the warm-bubble case."""

    return _build_setup(build_warm_bubble_numpy(), require_gpu=require_gpu)


def build_density_current_setup(*, require_gpu: bool = True) -> IdealizedSetup:
    """Return a JAX operational setup for the density-current case."""

    return _build_setup(build_density_current_numpy(), require_gpu=require_gpu)


@jax.jit
def _block_state_leaf(state: State) -> jax.Array:
    return jnp.sum(state.theta.astype(jnp.float64))


@jax.jit
def _initial_carry(state: State) -> OperationalCarry:
    # F7-B fp64-correctness: keep the IC in float64 (no operational fp32 downcast).
    return initial_operational_carry(_enforce_operational_precision(state, force_fp64=True))


@jax.jit
def _ready_carry(carry: OperationalCarry) -> jax.Array:
    return jnp.sum(carry.state.theta.astype(jnp.float64))


@partial(jax.jit, static_argnames=("start_step", "steps"))
def _run_segment_jit(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
) -> OperationalCarry:
    """Run a segment with no host callbacks inside the timestep loop."""

    indices = jnp.arange(int(start_step), int(start_step) + int(steps), dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index: jax.Array) -> tuple[OperationalCarry, None]:
        return _physics_boundary_step(scan_carry, namelist, step_index, run_radiation=False, debug=False), None

    next_carry, _ = jax.lax.scan(body, carry, indices)
    return next_carry


def _run_segment(carry: OperationalCarry, namelist: OperationalNamelist, *, start_step: int, steps: int) -> OperationalCarry:
    current = _run_segment_jit(carry, namelist, start_step=int(start_step), steps=int(steps))
    _ready_carry(current).block_until_ready()
    return current


def _snapshot(case: NumpyIdealizedCase, state: State, second: float) -> dict[str, Any]:
    theta = np.asarray(jax.device_get(state.theta[:, 0, :]), dtype=np.float64)
    theta_prime = theta - THETA0_K
    w_face = np.asarray(jax.device_get(state.w[:, 0, :]), dtype=np.float64)
    w_mass = 0.5 * (w_face[:-1] + w_face[1:])
    u_face = np.asarray(jax.device_get(state.u[:, 0, :]), dtype=np.float64)
    u_mass = 0.5 * (u_face[:, :-1] + u_face[:, 1:])
    mu_total = np.asarray(jax.device_get(state.mu_total[0, :]), dtype=np.float64)
    weight_positive = np.maximum(theta_prime, 0.0)
    weight_cold = np.maximum(-theta_prime, 0.0)

    def weighted_center(weight: np.ndarray, coords: np.ndarray, axis: int) -> float | None:
        total = float(np.sum(weight))
        if total <= 1.0e-12:
            return None
        shaped = coords[:, None] if axis == 0 else coords[None, :]
        return _safe_float(np.sum(weight * shaped) / total)

    cold_ground_mask = (theta_prime < -1.0) & (case.z_m[:, None] <= 1500.0)
    if np.any(cold_ground_mask):
        x_grid = np.broadcast_to(case.x_m[None, :], theta_prime.shape)
        front_position_m = float(np.max(np.abs(x_grid[cold_ground_mask])))
    else:
        front_position_m = None

    return {
        "second": float(second),
        "theta_prime_k": theta_prime,
        "w_mass_m_s": w_mass,
        "u_mass_m_s": u_mass,
        "finite": bool(
            np.all(np.isfinite(theta_prime))
            and np.all(np.isfinite(w_mass))
            and np.all(np.isfinite(u_mass))
            and np.all(np.isfinite(mu_total))
        ),
        "theta_prime_min_k": _safe_float(np.min(theta_prime)),
        "theta_prime_max_k": _safe_float(np.max(theta_prime)),
        "max_abs_w_m_s": _safe_float(np.max(np.abs(w_mass))),
        "max_abs_u_m_s": _safe_float(np.max(np.abs(u_mass))),
        "positive_theta_center_x_m": weighted_center(weight_positive, case.x_m, axis=1),
        "positive_theta_center_z_m": weighted_center(weight_positive, case.z_m, axis=0),
        "cold_theta_center_x_m": weighted_center(weight_cold, case.x_m, axis=1),
        "cold_theta_center_z_m": weighted_center(weight_cold, case.z_m, axis=0),
        "front_position_m": _safe_float(front_position_m),
        "mass_total_pa": _safe_float(np.sum(mu_total)),
        "theta_symmetry_linf_k": _safe_float(np.max(np.abs(theta_prime - theta_prime[:, ::-1]))),
        "w_symmetry_linf_m_s": _safe_float(np.max(np.abs(w_mass - w_mass[:, ::-1]))),
    }


def _json_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if not isinstance(value, np.ndarray)}


def _check(value: float | None, threshold: str, passed: bool | None) -> dict[str, Any]:
    return {"value": value, "threshold": threshold, "passed": passed}


def _relative_mass_drift(snapshots: Sequence[dict[str, Any]]) -> float | None:
    masses = [item.get("mass_total_pa") for item in snapshots if item.get("mass_total_pa") is not None]
    if len(masses) < 2:
        return None
    initial = float(masses[0])
    if abs(initial) <= 1.0e-30:
        return None
    return _safe_float(max(abs((float(mass) - initial) / initial) for mass in masses))


def _rotor_proxy_count(snapshot: dict[str, Any], x_m: np.ndarray, z_m: np.ndarray) -> int:
    w = np.asarray(snapshot["w_mass_m_s"], dtype=np.float64)
    front = snapshot.get("front_position_m")
    if front is None or not np.isfinite(front):
        return 0
    z_mask = (z_m >= 400.0) & (z_m <= 2600.0)
    x_mask = (x_m >= 0.0) & (x_m <= float(front))
    if np.count_nonzero(z_mask) < 3 or np.count_nonzero(x_mask) < 7:
        return 0
    profile = np.mean(w[z_mask][:, x_mask], axis=0)
    if profile.size < 7:
        return 0
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(profile, kernel, mode="same")
    scale = max(float(np.max(np.abs(smooth))), 1.0e-12)
    extrema = 0
    for idx in range(1, smooth.size - 1):
        is_peak = smooth[idx] > smooth[idx - 1] and smooth[idx] > smooth[idx + 1]
        is_trough = smooth[idx] < smooth[idx - 1] and smooth[idx] < smooth[idx + 1]
        if (is_peak or is_trough) and abs(float(smooth[idx])) >= 0.15 * scale and abs(float(smooth[idx])) >= 0.5:
            extrema += 1
    return int(extrema)


def _evaluate_warm(
    case: NumpyIdealizedCase,
    initial_snapshot: dict[str, Any],
    snapshots: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    final = snapshots[-1]
    initial_center_z = initial_snapshot.get("positive_theta_center_z_m")
    rise = None
    if final.get("positive_theta_center_z_m") is not None and initial_center_z is not None:
        rise = _safe_float(float(final["positive_theta_center_z_m"]) - float(initial_center_z))
    drift = None
    if final.get("positive_theta_center_x_m") is not None:
        drift = _safe_float(abs(float(final["positive_theta_center_x_m"]) - 10000.0))
    mass_drift = _relative_mass_drift([initial_snapshot, *snapshots])
    checks = {
        "all_snapshots_finite": _check(1.0 if all(item["finite"] for item in snapshots) else 0.0, "all snapshot arrays finite", all(item["finite"] for item in snapshots)),
        "theta_prime_max_500s": _check(final.get("theta_prime_max_k"), "0.5 <= max(theta prime) <= 2.5 K", final.get("theta_prime_max_k") is not None and 0.5 <= float(final["theta_prime_max_k"]) <= 2.5),
        "max_abs_w_500s": _check(final.get("max_abs_w_m_s"), "1 <= max(|w|) <= 30 m/s", final.get("max_abs_w_m_s") is not None and 1.0 <= float(final["max_abs_w_m_s"]) <= 30.0),
        "thermal_rise_500s": _check(rise, "positive-theta centroid rises by at least 500 m", rise is not None and rise >= 500.0),
        "horizontal_drift_500s": _check(drift, "positive-theta centroid drift <= 250 m", drift is not None and drift <= 250.0),
        "relative_mass_drift": _check(mass_drift, "max relative dry-column mass drift <= 1e-8", mass_drift is not None and mass_drift <= 1.0e-8),
    }
    return checks, "PASS" if all(bool(row["passed"]) for row in checks.values()) else "FAIL"


def _evaluate_density(
    case: NumpyIdealizedCase,
    initial_snapshot: dict[str, Any],
    snapshots: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    final = snapshots[-1]
    mass_drift = _relative_mass_drift([initial_snapshot, *snapshots])
    rotor_proxy = _rotor_proxy_count(final, case.x_m, case.z_m)
    front = final.get("front_position_m")
    checks = {
        "all_snapshots_finite": _check(1.0 if all(item["finite"] for item in snapshots) else 0.0, "all snapshot arrays finite", all(item["finite"] for item in snapshots)),
        "theta_prime_min_900s": _check(final.get("theta_prime_min_k"), "-25 <= min(theta prime) <= -5 K", final.get("theta_prime_min_k") is not None and -25.0 <= float(final["theta_prime_min_k"]) <= -5.0),
        "max_abs_w_900s": _check(final.get("max_abs_w_m_s"), "1 <= max(|w|) <= 50 m/s", final.get("max_abs_w_m_s") is not None and 1.0 <= float(final["max_abs_w_m_s"]) <= 50.0),
        "front_position_900s": _check(front, "|front position - 15000 m| <= 2000 m", front is not None and abs(float(front) - 15000.0) <= 2000.0),
        "rotor_count_proxy_900s": _check(float(rotor_proxy), "2 <= rotor proxy count <= 4", 2 <= rotor_proxy <= 4),
        "relative_mass_drift": _check(mass_drift, "max relative dry-column mass drift <= 1e-8", mass_drift is not None and mass_drift <= 1.0e-8),
    }
    return checks, "PASS" if all(bool(row["passed"]) for row in checks.values()) else "FAIL"


def _rgb_heatmap(field: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1.0e-12:
        scaled = np.zeros_like(arr)
    else:
        scaled = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    red = np.clip(2.0 * scaled, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - scaled), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * scaled - 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return np.asarray(np.round(255.0 * rgb), dtype=np.uint8)


def _write_ppm(path: Path, field: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _rgb_heatmap(np.asarray(field)[::-1, :], vmin=vmin, vmax=vmax)
    height, width, _ = image.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def _case_paths(case_name: CaseName, proof_dir: Path) -> tuple[Path, Path]:
    if case_name == "warm_bubble":
        return proof_dir / "skamarock_bubble_diagnostics.json", proof_dir / "skamarock_bubble_verdict.md"
    return proof_dir / "straka_density_current_diagnostics.json", proof_dir / "straka_density_current_verdict.md"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _markdown_for_result(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    lines = [
        f"# {payload['title']}",
        "",
        f"Verdict: {payload['verdict']}",
        f"Status: {payload['status']}",
        "",
        "## Checks",
        "",
        "| Check | Value | Threshold | Passed |",
        "| --- | ---: | --- | --- |",
    ]
    for name, row in checks.items():
        lines.append(f"| {name} | {row.get('value')} | {row.get('threshold')} | {row.get('passed')} |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Device: {payload.get('device')}",
            f"- CPU affinity: {payload.get('cpu_affinity')}",
            f"- Timesteps: {payload.get('timesteps')}",
            f"- Snapshot seconds: {payload.get('snapshot_seconds')}",
            f"- Plots: {', '.join(payload.get('plots', [])) if payload.get('plots') else 'none'}",
            "",
            "## References",
            "",
        ]
    )
    for source in payload.get("reference", {}).get("sources", []):
        lines.append(f"- {source}")
    lines.extend(["", payload.get("interpretation", "")])
    return "\n".join(lines).rstrip() + "\n"


def _cpu_affinity() -> list[int] | str:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return "unavailable"


def _blocked_result(case: NumpyIdealizedCase, proof_dir: Path, reason: str) -> IdealizedRunResult:
    proof_json, proof_md = _case_paths(case.case_name, proof_dir)
    plot = proof_dir / "plots" / f"{case.case_name}_initial_theta_prime.ppm"
    _write_ppm(plot, case.theta_prime_k, vmin=-15.0 if case.case_name == "density_current" else 0.0, vmax=2.0 if case.case_name == "warm_bubble" else 0.0)
    checks = {
        "gpu_available": _check(0.0, "JAX GPU backend visible", False),
        "initial_condition_finite": _check(1.0 if np.all(np.isfinite(case.theta_prime_k)) else 0.0, "analytic IC arrays finite", bool(np.all(np.isfinite(case.theta_prime_k)))),
    }
    payload = {
        "schema": "f2_idealized_case_verdict",
        "schema_version": 1,
        "title": "Skamarock warm bubble verdict" if case.case_name == "warm_bubble" else "Straka density current verdict",
        "case_name": case.case_name,
        "case_id": case.case_id,
        "verdict": "BLOCKED",
        "status": "BLOCKED_GPU_UNAVAILABLE",
        "blocked_reason": reason,
        "device_inventory": _device_inventory(),
        "device": None,
        "cpu_affinity": _cpu_affinity(),
        "parameters": case.parameters,
        "reference": case.reference,
        "timesteps": 0,
        "snapshot_seconds": list(case.snapshot_seconds),
        "checks": checks,
        "initial_condition_stats": {
            "theta_prime_k": _stats(case.theta_prime_k),
            "theta_k": _stats(case.theta_k),
            "pressure_pa": _stats(case.pressure_pa),
        },
        "plots": [str(plot)],
        "interpretation": "The analytic initial condition was generated, but the dycore integration was not run because no JAX GPU backend was visible.",
    }
    _write_json(proof_json, payload)
    proof_md.write_text(_markdown_for_result(payload), encoding="utf-8")
    return IdealizedRunResult(case.case_name, "BLOCKED", "BLOCKED_GPU_UNAVAILABLE", proof_json, proof_md, (plot,), checks, payload)


def _run_case(case: NumpyIdealizedCase, *, proof_dir: Path, require_gpu: bool = True) -> IdealizedRunResult:
    try:
        setup = _build_setup(case, require_gpu=require_gpu)
    except IdealizedCaseBlocked as exc:
        return _blocked_result(case, proof_dir, str(exc))

    proof_json, proof_md = _case_paths(case.case_name, proof_dir)
    carry = _initial_carry(setup.state)
    _ready_carry(carry).block_until_ready()
    initial_snapshot = _snapshot(case, carry.state, 0.0)
    snapshots: list[dict[str, Any]] = []
    previous_step = 0
    plot_paths: list[Path] = []
    for second in case.snapshot_seconds:
        target_step = int(round(float(second) / float(case.dt_s)))
        carry = _run_segment(carry, setup.namelist, start_step=previous_step + 1, steps=target_step - previous_step)
        previous_step = target_step
        snapshot = _snapshot(case, carry.state, float(second))
        snapshots.append(snapshot)
        plot_path = proof_dir / "plots" / f"{case.case_name}_theta_prime_{int(second)}s.ppm"
        vmin = -15.0 if case.case_name == "density_current" else 0.0
        vmax = 2.0 if case.case_name == "warm_bubble" else 0.0
        _write_ppm(plot_path, snapshot["theta_prime_k"], vmin=vmin, vmax=vmax)
        plot_paths.append(plot_path)

    if case.case_name == "warm_bubble":
        checks, verdict = _evaluate_warm(case, initial_snapshot, snapshots)
        title = "Skamarock warm bubble verdict"
        if verdict == "PASS":
            interpretation = "The warm bubble rose coherently with bounded theta prime, active vertical motion, symmetry, and conserved dry mass under the declared checks."
        else:
            interpretation = "The warm-bubble run failed at least one declared check; this is an honest dycore-correctness failure, not a reference pass."
    else:
        checks, verdict = _evaluate_density(case, initial_snapshot, snapshots)
        title = "Straka density current verdict"
        if verdict == "PASS":
            interpretation = "The density current matched the declared front-position, rotor-proxy, bounded-theta, active-motion, and mass checks."
        else:
            interpretation = "The density-current run failed at least one declared check; this is an honest dycore-correctness failure, not a reference pass."

    payload = {
        "schema": "f2_idealized_case_verdict",
        "schema_version": 1,
        "title": title,
        "case_name": case.case_name,
        "case_id": case.case_id,
        "verdict": verdict,
        "status": "RAN_TO_COMPLETION",
        "device_inventory": _device_inventory(),
        "device": setup.device,
        "cpu_affinity": _cpu_affinity(),
        "parameters": case.parameters,
        "reference": case.reference,
        "timesteps": int(round(case.end_seconds / case.dt_s)),
        "dt_s": case.dt_s,
        "snapshot_seconds": list(case.snapshot_seconds),
        "checks": checks,
        "initial_snapshot": _json_snapshot(initial_snapshot),
        "snapshots": [_json_snapshot(snapshot) for snapshot in snapshots],
        "plots": [str(path) for path in plot_paths],
        "interpretation": interpretation,
    }
    _write_json(proof_json, payload)
    proof_md.write_text(_markdown_for_result(payload), encoding="utf-8")
    return IdealizedRunResult(case.case_name, verdict, "RAN_TO_COMPLETION", proof_json, proof_md, tuple(plot_paths), checks, payload)


def run_warm_bubble_case(*, proof_dir: Path | str = Path("proofs/f2"), require_gpu: bool = True) -> IdealizedRunResult:
    """Run or block-record the F2 warm-bubble case."""

    return _run_case(build_warm_bubble_numpy(), proof_dir=Path(proof_dir), require_gpu=require_gpu)


def run_density_current_case(*, proof_dir: Path | str = Path("proofs/f2"), require_gpu: bool = True) -> IdealizedRunResult:
    """Run or block-record the F2 density-current case."""

    return _run_case(build_density_current_numpy(), proof_dir=Path(proof_dir), require_gpu=require_gpu)


def _write_correctness_summary(proof_dir: Path, results: Sequence[IdealizedRunResult]) -> Path:
    path = proof_dir / "dycore_correctness_summary.md"
    statuses = {result.case_name: result.status for result in results}
    verdicts = {result.case_name: result.verdict for result in results}
    if any(result.status.startswith("BLOCKED") for result in results):
        paragraph = (
            "Given the F2 outcomes, the most likely structural bug cannot be newly isolated from these idealized cases because "
            "the required JAX GPU backend was unavailable and both dycore integrations were blocked before timestep execution. "
            "The standing dycore-review hypothesis therefore remains the best-supported explanation: operational correctness is most at risk from "
            "large-step advection/RK coupling defects, acoustic dry-mass carry errors, and theta mass-decoupling mistakes. "
            "This summary is intentionally not a physics pass/fail claim."
        )
    elif any(result.verdict == "FAIL" for result in results):
        paragraph = (
            "Given the warm-bubble and density-current outcomes, the most likely structural bug is a missing or ineffective transport/buoyancy pathway in the "
            "operational RK/acoustic dycore: the reference cases require coherent vertical motion, lateral cold-pool propagation, and dry-mass consistency, "
            "so failures in rise/front/rotor checks point first to advection/RK coupling and then to acoustic mass/theta coupling. "
            "This is diagnostic evidence only; it does not repair the protected dycore files."
        )
    else:
        paragraph = (
            "Given the warm-bubble and density-current outcomes, no new structural dycore bug is indicated by these two idealized checks; the remaining risk is "
            "whether this behavior survives true WRF fixtures and broader Canary L2/L3 ensembles."
        )
    text = (
        "# F2 Dycore Correctness Summary\n\n"
        f"Case statuses: {statuses}. Case verdicts: {verdicts}.\n\n"
        f"{paragraph}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_all_idealized_cases(*, proof_dir: Path | str = Path("proofs/f2"), require_gpu: bool = True) -> list[IdealizedRunResult]:
    """Run both F2 cases and write the required one-paragraph summary."""

    root = Path(proof_dir)
    results = [
        run_warm_bubble_case(proof_dir=root, require_gpu=require_gpu),
        run_density_current_case(proof_dir=root, require_gpu=require_gpu),
    ]
    _write_correctness_summary(root, results)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("warm_bubble", "density_current", "all"), default="all")
    parser.add_argument("--proof-dir", type=Path, default=Path("proofs/f2"))
    parser.add_argument("--allow-cpu", action="store_true", help="debug only: run on CPU if no GPU is visible")
    args = parser.parse_args(argv)

    require_gpu = not bool(args.allow_cpu)
    if args.case == "warm_bubble":
        results = [run_warm_bubble_case(proof_dir=args.proof_dir, require_gpu=require_gpu)]
        _write_correctness_summary(args.proof_dir, results)
    elif args.case == "density_current":
        results = [run_density_current_case(proof_dir=args.proof_dir, require_gpu=require_gpu)]
        _write_correctness_summary(args.proof_dir, results)
    else:
        results = run_all_idealized_cases(proof_dir=args.proof_dir, require_gpu=require_gpu)

    print(json.dumps({result.case_name: {"verdict": result.verdict, "status": result.status} for result in results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
