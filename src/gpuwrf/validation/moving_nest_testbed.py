"""Tiny flat-terrain 3-D nested operational pair for the v0.23 G2 CPU gates.

Builds a REAL two-domain operational tree (full RK3 + acoustic dycore, live
nested forcedown, optional two-way feedback) on grids small enough to compile
and run on CPU in seconds:

* neutral hydrostatically balanced base columns (the F7 idealized recipe from
  :mod:`gpuwrf.ic_generators.idealized`, generalized to 3-D ``ny > 1``);
* flat terrain, pure-sigma metrics (the WRF ``hybrid_opt=0`` idealized default);
* uniform ambient wind plus an optional analytic theta anomaly placed by
  PHYSICAL position, so the parent and child ICs describe the same atmosphere
  through the WRF cell-centered nest registration.

No GPU, no fixtures, no masking -- this is the moving-nest oracle bed, not a
physics benchmark (``run_physics=False`` keeps the full dynamical core).
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Callable

import numpy as np

import jax.numpy as jnp

from gpuwrf.contracts.grid import (
    BCMetadata,
    DomainHierarchy,
    DomainNest,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.ic_generators.idealized import (
    CP_DRY_AIR,
    CV_DRY_AIR,
    GRAVITY_M_S2,
    P0_PA,
    R_DRY_AIR,
    THETA0_K,
    _alpha_dry,
    _uniform_z_hydrostatic_base,
)
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist


def build_flat_grid(*, nx: int, ny: int, nz: int, dx_m: float, z_top_m: float = 8000.0) -> GridSpec:
    """Flat-terrain pure-sigma :class:`GridSpec` (3-D generalization of F7)."""

    z_face = np.linspace(0.0, float(z_top_m), int(nz) + 1)
    eta_levels_np, _p_mass, _ph_face, mu0 = _uniform_z_hydrostatic_base(z_face, THETA0_K)
    eta_levels = jnp.asarray(eta_levels_np, dtype=jnp.float64)
    # Hydrostatic column top pressure: mu = p_sfc - p_top with p_sfc = P0.
    p_top = float(P0_PA - mu0)
    terrain_height = jnp.zeros((int(ny), int(nx)), dtype=jnp.float64)
    projection = Projection("lambert", 0.0, 0.0, float(dx_m), float(dx_m), int(nx), int(ny))
    terrain = TerrainProvenance(
        source_path="idealized:g2-moving-nest",
        sha256="analytic-g2",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="flat-3d",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", int(nz), float(z_top_m), eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999,
        interpolation="linear",
        restart_compatible=False,
    )
    metrics = DycoreMetrics.flat(
        ny=int(ny),
        nx=int(nx),
        nz=int(nz),
        eta_levels=eta_levels,
        top_pressure_pa=p_top,
        provenance="analytic-g2-moving-nest",
    )
    # Pure sigma (WRF idealized hybrid_opt=0): see ic_generators/idealized.py --
    # DycoreMetrics.flat's hybrid c1f=eta zeroes the top-face dry mass and makes
    # the vertical solver singular on these columns.
    nz_i = int(nz)
    one_h = jnp.ones((nz_i,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz_i,), dtype=jnp.float64)
    one_f = jnp.ones((nz_i + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz_i + 1,), dtype=jnp.float64)
    metrics = DycoreMetrics(
        msftx=metrics.msftx, msfty=metrics.msfty, msfux=metrics.msfux, msfuy=metrics.msfuy,
        msfvx=metrics.msfvx, msfvy=metrics.msfvy,
        c1h=one_h, c2h=zero_h, c3h=one_h, c4h=zero_h,
        c1f=one_f, c2f=zero_f, c3f=one_f, c4f=zero_f,
        dn=metrics.dn, dnw=metrics.dnw, rdn=metrics.rdn, rdnw=metrics.rdnw,
        cf1=metrics.cf1, cf2=metrics.cf2, cf3=metrics.cf3, fnm=metrics.fnm, fnp=metrics.fnp,
        dzdx=metrics.dzdx, dzdy=metrics.dzdy, dzdx_u=metrics.dzdx_u, dzdy_v=metrics.dzdy_v,
        f=metrics.f, e=metrics.e, sina=metrics.sina, cosa=metrics.cosa,
        p_top=metrics.p_top, provenance="analytic-g2-moving-nest-pure-sigma",
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


def cell_center_coords_m(
    *, n: int, dx_m: float, parent_dx_m: float | None = None,
    parent_start_1based: int | None = None, ratio: int | None = None,
) -> np.ndarray:
    """Physical coordinates of cell centers along one axis.

    Root domain: ``x_i = i * dx``.  Nested child: the WRF cell-centered
    registration ``x_i = ((start-1) + (i - ratio//2)/ratio) * parent_dx`` --
    the SAME registration the SINT forcedown gathers with, so a parent field
    and its child rendition describe one continuous atmosphere.
    """

    idx = np.arange(int(n), dtype=np.float64)
    if parent_start_1based is None:
        return idx * float(dx_m)
    assert ratio is not None and parent_dx_m is not None
    return (float(parent_start_1based - 1) + (idx - float(int(ratio) // 2)) / float(ratio)) * float(
        parent_dx_m
    )


def build_neutral_state(
    grid: GridSpec,
    *,
    z_top_m: float = 8000.0,
    u0_m_s: float = 0.0,
    v0_m_s: float = 0.0,
    theta_anomaly: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    x_coords_m: np.ndarray | None = None,
    y_coords_m: np.ndarray | None = None,
) -> State:
    """Hydrostatically balanced neutral 3-D state with optional theta anomaly.

    ``theta_anomaly(x2d, y2d)`` returns a (ny, nx) horizontal anomaly [K] applied
    to every mass level; the geopotential and diagnostic pressure perturbations
    are re-integrated at fixed column mass exactly as WRF's idealized init
    (``module_initialize_ideal.F``; see ic_generators/idealized.py:479-543).
    """

    nz, ny, nx = int(grid.nz), int(grid.ny), int(grid.nx)
    z_face = np.linspace(0.0, float(z_top_m), nz + 1)
    eta, p_mass_1d, ph_face_1d, mu0 = _uniform_z_hydrostatic_base(z_face, THETA0_K)

    xs = x_coords_m if x_coords_m is not None else np.arange(nx) * float(grid.projection.dx_m)
    ys = y_coords_m if y_coords_m is not None else np.arange(ny) * float(grid.projection.dy_m)
    x2d, y2d = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))

    anom = (
        np.zeros((ny, nx), dtype=np.float64)
        if theta_anomaly is None
        else np.asarray(theta_anomaly(x2d, y2d), dtype=np.float64)
    )
    theta = THETA0_K + np.broadcast_to(anom, (nz, ny, nx)).copy()

    pb = np.broadcast_to(p_mass_1d[:, None, None], (nz, ny, nx)).copy()
    alt_full = _alpha_dry(theta, pb)
    alb = _alpha_dry(np.full(nz, THETA0_K), p_mass_1d)
    al = alt_full - alb[:, None, None]

    wrf_dnw = eta[1:] - eta[:-1]
    ph_pert = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    for k in range(nz):
        ph_pert[k + 1] = ph_pert[k] - wrf_dnw[k] * mu0 * al[k]
    phb = np.broadcast_to((GRAVITY_M_S2 * z_face)[:, None, None], (nz + 1, ny, nx)).copy()
    ph_total = phb + ph_pert

    cpovcv = CP_DRY_AIR / CV_DRY_AIR
    p_full = P0_PA * ((R_DRY_AIR * theta) / (P0_PA * alt_full)) ** cpovcv
    p_pert = p_full - pb

    mu_total = np.full((ny, nx), mu0, dtype=np.float64)

    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    fields.update(
        theta=jnp.asarray(theta),
        u=jnp.full((nz, ny, nx + 1), float(u0_m_s), dtype=jnp.float64),
        v=jnp.full((nz, ny + 1, nx), float(v0_m_s), dtype=jnp.float64),
        w=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        p=jnp.asarray(p_full),
        p_total=jnp.asarray(p_full),
        p_perturbation=jnp.asarray(p_pert),
        ph=jnp.asarray(ph_total),
        ph_total=jnp.asarray(ph_total),
        ph_perturbation=jnp.asarray(ph_pert),
        mu=jnp.asarray(mu_total),
        mu_total=jnp.asarray(mu_total),
        mu_perturbation=jnp.zeros((ny, nx), dtype=jnp.float64),
        Ni=jnp.full((nz, ny, nx), 1.0e5, dtype=jnp.float64),
        Nr=jnp.full((nz, ny, nx), 1.0e5, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float64),
        mavail=jnp.full((ny, nx), 0.2, dtype=jnp.float64),
        roughness_m=jnp.full((ny, nx), 0.05, dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), THETA0_K, dtype=jnp.float64),
        rhosfc=jnp.ones((ny, nx), dtype=jnp.float64),
    )
    return State(**fields)


def _zero_tendencies(grid: GridSpec) -> Tendencies:
    """CPU-safe zero tendencies (``Tendencies.zeros`` insists on a GPU device)."""

    nz, ny, nx = int(grid.nz), int(grid.ny), int(grid.nx)
    return Tendencies(
        u=jnp.zeros((nz, ny, nx + 1), dtype=jnp.float64),
        v=jnp.zeros((nz, ny + 1, nx), dtype=jnp.float64),
        w=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        theta=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        qv=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        p=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        ph=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        mu=jnp.zeros((ny, nx), dtype=jnp.float64),
    )


def build_domain_namelist(
    grid: GridSpec,
    *,
    dt_s: float,
    is_child: bool,
    parent_dt_s: float | None = None,
    acoustic_substeps: int = 4,
) -> OperationalNamelist:
    namelist = OperationalNamelist.from_grid(
        grid,
        tendencies=_zero_tendencies(grid),
        metrics=grid.metrics,
        dt_s=float(dt_s),
        acoustic_substeps=int(acoustic_substeps),
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
        disable_guards=True,
        force_fp64=True,
        use_flux_advection=True,
    )
    namelist = dataclass_replace(
        namelist,
        run_physics=False,
        run_boundary=bool(is_child),
        top_lid=True,
        w_damping=1,
        damp_opt=3,
        dampcoef=0.2,
        zdamp=3000.0,
    )
    if is_child:
        assert parent_dt_s is not None
        namelist = with_live_child_boundary_config(namelist, parent_dt_s=float(parent_dt_s))
    return namelist


def build_nested_pair(
    *,
    parent_nx: int = 30,
    parent_ny: int = 27,
    child_nx: int = 24,
    child_ny: int = 21,
    nz: int = 8,
    ratio: int = 3,
    i_start: int = 5,
    j_start: int = 5,
    parent_dx_m: float = 3000.0,
    dt_s: float = 6.0,
    z_top_m: float = 8000.0,
    u0_m_s: float = 0.0,
    v0_m_s: float = 0.0,
    theta_anomaly: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    feedback_enabled: bool = False,
) -> DomainTree:
    """Full two-domain operational tree with a consistent parent/child IC."""

    child_dx = float(parent_dx_m) / float(ratio)
    parent_grid = build_flat_grid(nx=parent_nx, ny=parent_ny, nz=nz, dx_m=parent_dx_m, z_top_m=z_top_m)
    child_grid = build_flat_grid(nx=child_nx, ny=child_ny, nz=nz, dx_m=child_dx, z_top_m=z_top_m)

    parent_state = build_neutral_state(
        parent_grid, z_top_m=z_top_m, u0_m_s=u0_m_s, v0_m_s=v0_m_s, theta_anomaly=theta_anomaly
    )
    child_state = build_neutral_state(
        child_grid,
        z_top_m=z_top_m,
        u0_m_s=u0_m_s,
        v0_m_s=v0_m_s,
        theta_anomaly=theta_anomaly,
        x_coords_m=cell_center_coords_m(
            n=child_nx, dx_m=child_dx, parent_dx_m=parent_dx_m,
            parent_start_1based=i_start, ratio=ratio,
        ),
        y_coords_m=cell_center_coords_m(
            n=child_ny, dx_m=child_dx, parent_dx_m=parent_dx_m,
            parent_start_1based=j_start, ratio=ratio,
        ),
    )

    parent_nl = build_domain_namelist(parent_grid, dt_s=dt_s, is_child=False)
    child_nl = build_domain_namelist(
        child_grid, dt_s=dt_s / float(ratio), is_child=True, parent_dt_s=dt_s
    )

    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", int(ratio), int(i_start), int(j_start), feedback=feedback_enabled),),
    )
    bundles = {
        "d01": DomainBundle("d01", parent_state, parent_nl, grid=parent_grid),
        "d02": DomainBundle("d02", child_state, child_nl, grid=child_grid),
    }
    return DomainTree.from_domains(hierarchy, bundles, feedback_enabled=feedback_enabled)


def gaussian_low(
    *, x0_m: float, y0_m: float, radius_m: float, amplitude_k: float = -2.0
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A cold (negative-theta) circular anomaly -- the trackable 'vortex'."""

    def anomaly(x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
        r2 = (x2d - float(x0_m)) ** 2 + (y2d - float(y0_m)) ** 2
        return float(amplitude_k) * np.exp(-r2 / (2.0 * float(radius_m) ** 2))

    return anomaly
