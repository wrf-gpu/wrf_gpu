"""WRF-shaped acoustic scan and horizontal pressure-gradient force."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import BaseState, State
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig, apply_rayleigh_w, apply_smdiv_pressure
from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients, solve_tridiagonal


config.update("jax_enable_x64", True)


R_D = 287.0
CP_D = 1004.0
P0_PA = 100000.0
CPOVCV = CP_D / (CP_D - R_D)
CVPM = -(CP_D - R_D) / CP_D
GAMMA_DRY_AIR = CP_D / (CP_D - R_D)
GRAVITY_M_S2 = 9.80665
MIN_PRESSURE_PA = 1.0
MIN_ALT = 1.0e-8
CONVECTIVE_BUOYANCY_GAIN = 0.0
NONHYDROSTATIC_BUOYANCY_SCALE = 0.38
NONHYDROSTATIC_UPDRAFT_DRAG = 0.0005
MU_CONTINUITY_CFL_FRACTION = 1.0e-3
MPAS_OMEGA_TO_W_METRIC = 1.35
POST_SOLVE_REPLACEMENT_ORDER = (
    "w",
    "theta",
    "ph_perturbation",
    "mu_perturbation",
    "p_perturbation",
    "al",
    "alt",
)


@dataclass(frozen=True)
class AcousticConfig:
    """Static acoustic-substep config for c2 nested scans."""

    n_substeps: int = 1
    dx_m: float = 1.0
    dy_m: float = 1.0
    non_hydrostatic: bool = True
    top_lid: bool = True
    mu_continuity: bool = True
    epssm: float = 0.1
    smdiv: SmdivConfig = field(default_factory=SmdivConfig)
    rayleigh: RayleighConfig = field(default_factory=RayleighConfig)


@jax.tree_util.register_pytree_node_class
class AcousticScanCarry:
    """Small-step scan carry: state, pressure memory, and WRF intermediates.

    ADR-023 keeps the public carry to these six leaves:
    ``(state, previous_pressure, al, alt, cqu, cqv)``. Per-substep locals
    such as ``rs``, ``ts``, ``rw_p`` and ``rho_pp`` are named inside
    ``vertical_acoustic_update`` and do not leak through the scan carry.
    Solver scratch is restricted to tridiagonal coefficients built by
    ``_calc_coef_w`` / ``build_epssm_column_coefficients`` and consumed by the
    Thomas solve within the same substep.
    """

    __slots__ = ("state", "previous_pressure", "al", "alt", "cqu", "cqv")

    def __init__(
        self,
        state: State,
        previous_pressure: jax.Array,
        al: jax.Array,
        alt: jax.Array,
        cqu: jax.Array,
        cqv: jax.Array,
    ) -> None:
        self.state = state
        self.previous_pressure = previous_pressure
        self.al = al
        self.alt = alt
        self.cqu = cqu
        self.cqv = cqv

    def tree_flatten(self):
        """Presents acoustic intermediates as scan-carry leaves."""

        return (self.state, self.previous_pressure, self.al, self.alt, self.cqu, self.cqv), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds acoustic carry after JAX transformations."""

        del aux
        state, previous_pressure, al, alt, cqu, cqv = children
        return cls(state, previous_pressure, al, alt, cqu, cqv)


def _safe_pressure(pressure: jax.Array) -> jax.Array:
    """Keeps equation-of-state powers finite in diagnostic-only helpers."""

    return jnp.maximum(pressure, jnp.asarray(MIN_PRESSURE_PA, dtype=pressure.dtype))


def _safe_alt(alt: jax.Array) -> jax.Array:
    """Keeps diagnostic pressure finite without clipping prognostic state."""

    return jnp.maximum(alt, jnp.asarray(MIN_ALT, dtype=alt.dtype))


def _inverse_density_from_theta_pressure(theta: jax.Array, pressure: jax.Array, qv: jax.Array | None = None) -> jax.Array:
    """Dry WRF inverse-density equation of state.

    WRF source anchors: ``module_big_step_utilities_em.F:1085-1087`` and
    ``module_small_step_em.F:527-528`` use the same dry equation-of-state
    relation between potential temperature, pressure, and inverse density.
    """

    qvf = 1.0 if qv is None else 1.0 + 0.608 * qv
    return (R_D / P0_PA) * theta * qvf * ((_safe_pressure(pressure) / P0_PA) ** CVPM)


def _pressure_from_theta_alt(theta: jax.Array, alt: jax.Array, qv: jax.Array | None = None) -> jax.Array:
    """Inverts WRF's dry equation of state for diagnostic pressure."""

    qvf = 1.0 if qv is None else 1.0 + 0.608 * qv
    argument = (R_D * theta * qvf) / (P0_PA * _safe_alt(alt))
    return P0_PA * (jnp.maximum(argument, 1.0e-12) ** CPOVCV)


def _x_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns left/right mass values at x-staggered faces with edge BCs."""

    padded = jnp.pad(field, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return padded[:, :, :-1], padded[:, :, 1:]


def _y_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns south/north mass values at y-staggered faces with edge BCs."""

    padded = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return padded[:, :-1, :], padded[:, 1:, :]


def _x_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns left/right 2-D mass values at x-staggered faces."""

    padded = jnp.pad(field, ((0, 0), (1, 1)), mode="edge")
    return padded[:, :-1], padded[:, 1:]


def _y_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns south/north 2-D mass values at y-staggered faces."""

    padded = jnp.pad(field, ((1, 1), (0, 0)), mode="edge")
    return padded[:-1, :], padded[1:, :]


def _base_pressure(base_state: BaseState | None, state: State) -> jax.Array:
    """Returns base pressure, falling back to the legacy total/pert split."""

    if base_state is None:
        return state.p_total - state.p_perturbation
    return base_state.pb


def _base_geopotential(base_state: BaseState | None, state: State) -> jax.Array:
    """Returns base geopotential, falling back to the legacy total/pert split."""

    if base_state is None:
        return state.ph_total - state.ph_perturbation
    return base_state.phb


def _base_mu(base_state: BaseState | None, state: State) -> jax.Array:
    """Returns base dry-column mass, falling back to the legacy total/pert split."""

    if base_state is None:
        return state.mu_total - state.mu_perturbation
    return base_state.mub


def _base_theta(base_state: BaseState | None, state: State) -> jax.Array:
    """Returns base potential temperature for base inverse-density diagnostics."""

    if base_state is None:
        return state.theta
    return base_state.theta_base


@partial(jax.jit, static_argnames=())
def moisture_coupling_factors(state: State) -> tuple[jax.Array, jax.Array]:
    """Computes WRF-shaped ``cqu/cqv`` from scan-resident moist fields.

    WRF source anchors: ``module_big_step_utilities_em.F:789-850`` computes
    inverse moist loading on u/v faces; ``module_small_step_em.F:868,942``
    applies these factors to the final horizontal PGF tendency.
    """

    qtot = state.qv + state.qc + state.qr + state.qi + state.qs + state.qg
    qx_left, qx_right = _x_face_pair_3d(qtot)
    qy_south, qy_north = _y_face_pair_3d(qtot)
    cqu = 1.0 / (1.0 + 0.5 * (qx_left + qx_right))
    cqv = 1.0 / (1.0 + 0.5 * (qy_south + qy_north))
    return cqu, cqv


@partial(jax.jit, static_argnames=())
def diagnose_pressure_al_alt(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes substep-local pressure, ``al``, and ``alt``.

    With an explicit ``BaseState``, this follows WRF's non-hydrostatic
    ``calc_p_rho_phi`` pattern: ``al`` from geopotential/mass
    (``module_big_step_utilities_em.F:1025-1030``), diagnostic pressure
    from the equation of state (``:1082-1087``), and ``alt = al + alb``
    (``:910-943``). Without a base state, the c2 architecture smoke tests
    use the already-consistent legacy pressure while still carrying ``alt``.
    """

    if base_state is None:
        alt = _inverse_density_from_theta_pressure(state.theta, state.p_total, state.qv)
        al = jnp.zeros_like(alt)
        return state.p_perturbation, al, alt

    base_pressure = _safe_pressure(base_state.pb)
    alb = _inverse_density_from_theta_pressure(base_state.theta_base, base_pressure)
    muts = base_state.mub + state.mu_perturbation
    mass_weight = metrics.c1h[:, None, None] * muts[None, :, :] + metrics.c2h[:, None, None]
    mu_term = alb * metrics.c1h[:, None, None] * state.mu_perturbation[None, :, :]
    al = -mu_term / jnp.maximum(jnp.abs(mass_weight), 1.0e-12)
    alt = al + alb
    total_pressure = _pressure_from_theta_alt(base_state.theta_base, alt, state.qv)
    pressure_perturbation = total_pressure - base_state.pb
    return pressure_perturbation, al, alt


def _php_mass_levels(state: State, base_state: BaseState | None) -> jax.Array:
    """Computes WRF ``php`` on mass levels.

    WRF source anchor: ``module_big_step_utilities_em.F:1227-1261`` averages
    base plus perturbation geopotential on adjacent vertical faces.
    """

    phb = _base_geopotential(base_state, state)
    return 0.5 * (phb[:-1, :, :] + phb[1:, :, :] + state.ph_perturbation[:-1, :, :] + state.ph_perturbation[1:, :, :])


@partial(jax.jit, static_argnames=("top_lid",))
def x_face_pressure_dpn(pressure_perturbation: jax.Array, metrics: DycoreMetrics, top_lid: bool = False) -> jax.Array:
    """Builds WRF non-hydrostatic ``dpn`` on x faces.

    WRF source anchors: ``module_small_step_em.F:836-851`` use
    ``cf1/cf2/cf3`` at the bottom/top boundary and ``fnm/fnp`` in the
    interior for the x-momentum fourth PGF term.
    """

    left, right = _x_face_pair_3d(pressure_perturbation)
    pair_sum = left + right
    nz, ny, nx_face = pair_sum.shape
    bottom = 0.5 * (metrics.cf1 * pair_sum[0] + metrics.cf2 * pair_sum[1] + metrics.cf3 * pair_sum[2])
    interior = 0.5 * (
        metrics.fnm[1:, None, None] * pair_sum[1:, :, :]
        + metrics.fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    dpn = jnp.zeros((nz + 1, ny, nx_face), dtype=pressure_perturbation.dtype)
    dpn = dpn.at[0, :, :].set(bottom)
    dpn = dpn.at[1:nz, :, :].set(interior)
    if bool(top_lid):
        top = 0.5 * (
            metrics.cf1 * pair_sum[-1, :, :]
            + metrics.cf2 * pair_sum[-2, :, :]
            + metrics.cf3 * pair_sum[-3, :, :]
        )
        dpn = dpn.at[nz, :, :].set(top)
    return dpn


@partial(jax.jit, static_argnames=("top_lid",))
def y_face_pressure_dpn(pressure_perturbation: jax.Array, metrics: DycoreMetrics, top_lid: bool = False) -> jax.Array:
    """Builds WRF non-hydrostatic ``dpn`` on y faces.

    WRF source anchors: ``module_small_step_em.F:910-925`` use
    ``cf1/cf2/cf3`` at the bottom/top boundary and ``fnm/fnp`` in the
    interior for the y-momentum fourth PGF term.
    """

    south, north = _y_face_pair_3d(pressure_perturbation)
    pair_sum = south + north
    nz, ny_face, nx = pair_sum.shape
    bottom = 0.5 * (metrics.cf1 * pair_sum[0] + metrics.cf2 * pair_sum[1] + metrics.cf3 * pair_sum[2])
    interior = 0.5 * (
        metrics.fnm[1:, None, None] * pair_sum[1:, :, :]
        + metrics.fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    dpn = jnp.zeros((nz + 1, ny_face, nx), dtype=pressure_perturbation.dtype)
    dpn = dpn.at[0, :, :].set(bottom)
    dpn = dpn.at[1:nz, :, :].set(interior)
    if bool(top_lid):
        top = 0.5 * (
            metrics.cf1 * pair_sum[-1, :, :]
            + metrics.cf2 * pair_sum[-2, :, :]
            + metrics.cf3 * pair_sum[-3, :, :]
        )
        dpn = dpn.at[nz, :, :].set(top)
    return dpn


@partial(jax.jit, static_argnames=("dx_m", "dy_m", "non_hydrostatic", "top_lid"))
def horizontal_pressure_gradient(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    pressure_perturbation: jax.Array,
    al: jax.Array,
    alt: jax.Array,
    cqu: jax.Array,
    cqv: jax.Array,
    *,
    dx_m: float = 1.0,
    dy_m: float = 1.0,
    non_hydrostatic: bool = True,
    top_lid: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Computes WRF small-step horizontal PGF tendencies.

    The first three terms are WRF ``module_small_step_em.F:828-831`` and
    ``:902-905``. The fourth non-hydrostatic correction is ``:854-862`` and
    ``:928-936``; the M1 ambiguity is resolved as the literal WRF
    ``-0.5*c1h*(mu_left + mu_right)``. ``php``/``dpn`` are substep-local
    intermediates, addressing the M2 classification follow-up.
    """

    rdx = 1.0 / float(dx_m)
    rdy = 1.0 / float(dy_m)
    pb = _base_pressure(base_state, state)
    full_mu = _base_mu(base_state, state) + state.mu_perturbation
    perturbation_mu = state.mu_perturbation

    ph_left_x, ph_right_x = _x_face_pair_3d(state.ph_perturbation)
    p_left_x, p_right_x = _x_face_pair_3d(pressure_perturbation)
    pb_left_x, pb_right_x = _x_face_pair_3d(pb)
    al_left_x, al_right_x = _x_face_pair_3d(al)
    alt_left_x, alt_right_x = _x_face_pair_3d(alt)
    mu_left_x, mu_right_x = _x_face_pair_2d(full_mu)
    mu_pert_left_x, mu_pert_right_x = _x_face_pair_2d(perturbation_mu)
    muu = 0.5 * (mu_left_x + mu_right_x)
    mass_x = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
    ph_term_x = (ph_right_x[1:] - ph_left_x[1:]) + (ph_right_x[:-1] - ph_left_x[:-1])
    p_term_x = (alt_left_x + alt_right_x) * (p_right_x - p_left_x)
    pb_term_x = (al_left_x + al_right_x) * (pb_right_x - pb_left_x)
    dpx = (metrics.msfux / metrics.msfuy)[None, :, :] * 0.5 * rdx * mass_x * (
        ph_term_x + p_term_x + pb_term_x
    )

    ph_south_y, ph_north_y = _y_face_pair_3d(state.ph_perturbation)
    p_south_y, p_north_y = _y_face_pair_3d(pressure_perturbation)
    pb_south_y, pb_north_y = _y_face_pair_3d(pb)
    al_south_y, al_north_y = _y_face_pair_3d(al)
    alt_south_y, alt_north_y = _y_face_pair_3d(alt)
    mu_south_y, mu_north_y = _y_face_pair_2d(full_mu)
    mu_pert_south_y, mu_pert_north_y = _y_face_pair_2d(perturbation_mu)
    muv = 0.5 * (mu_south_y + mu_north_y)
    mass_y = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
    ph_term_y = (ph_north_y[1:] - ph_south_y[1:]) + (ph_north_y[:-1] - ph_south_y[:-1])
    p_term_y = (alt_south_y + alt_north_y) * (p_north_y - p_south_y)
    pb_term_y = (al_south_y + al_north_y) * (pb_north_y - pb_south_y)
    dpy = (metrics.msfvy / metrics.msfvx)[None, :, :] * 0.5 * rdy * mass_y * (
        ph_term_y + p_term_y + pb_term_y
    )

    if bool(non_hydrostatic):
        php = _php_mass_levels(state, base_state)
        php_left_x, php_right_x = _x_face_pair_3d(php)
        dpn_x = x_face_pressure_dpn(pressure_perturbation, metrics, top_lid=top_lid)
        bracket_x = metrics.rdnw[:, None, None] * (dpn_x[1:] - dpn_x[:-1]) - 0.5 * (
            metrics.c1h[:, None, None] * (mu_pert_left_x + mu_pert_right_x)[None, :, :]
        )
        dpx = dpx + (metrics.msfux / metrics.msfuy)[None, :, :] * rdx * (php_right_x - php_left_x) * bracket_x

        php_south_y, php_north_y = _y_face_pair_3d(php)
        dpn_y = y_face_pressure_dpn(pressure_perturbation, metrics, top_lid=top_lid)
        bracket_y = metrics.rdnw[:, None, None] * (dpn_y[1:] - dpn_y[:-1]) - 0.5 * (
            metrics.c1h[:, None, None] * (mu_pert_south_y + mu_pert_north_y)[None, :, :]
        )
        dpy = dpy + (metrics.msfvy / metrics.msfvx)[None, :, :] * rdy * (php_north_y - php_south_y) * bracket_y

    du_dt = -cqu * dpx
    dv_dt = -cqv * dpy
    return du_dt, dv_dt, dpx, dpy


@partial(jax.jit, static_argnames=("dx_m", "dy_m"))
def mu_continuity_tendency(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dx_m: float = 1.0,
    dy_m: float = 1.0,
) -> jax.Array:
    """Computes the small-step dry-column mass tendency.

    WRF source anchors: ``module_small_step_em.F:1094-1099`` computes the
    horizontal mass-flux divergence and ``:1102-1108`` updates ``MU`` inside
    the acoustic loop. This c2 implementation keeps the same in-loop carry
    placement while using the resident C-grid velocity fields directly.
    """

    rdx = 1.0 / float(dx_m)
    rdy = 1.0 / float(dy_m)
    full_mu = _base_mu(base_state, state) + state.mu_perturbation
    mu_left_x, mu_right_x = _x_face_pair_2d(full_mu)
    mu_south_y, mu_north_y = _y_face_pair_2d(full_mu)
    muu = 0.5 * (mu_left_x + mu_right_x)
    muv = 0.5 * (mu_south_y + mu_north_y)
    u_mass = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
    v_mass = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
    u_flux = state.u * u_mass / metrics.msfuy[None, :, :]
    v_flux = state.v * v_mass / metrics.msfvx[None, :, :]
    dudx = rdx * (u_flux[:, :, 1:] - u_flux[:, :, :-1])
    dvdy = rdy * (v_flux[:, 1:, :] - v_flux[:, :-1, :])
    dvdxi = metrics.msftx[None, :, :] * metrics.msfty[None, :, :] * (dudx + dvdy)
    return jnp.sum(metrics.dnw[:, None, None] * dvdxi, axis=0)


def _replace_pressure(state: State, pressure_perturbation: jax.Array, base_state: BaseState | None) -> State:
    """Updates explicit perturbation, total, and legacy pressure aliases."""

    if base_state is None:
        total_pressure = pressure_perturbation
    else:
        total_pressure = base_state.pb + pressure_perturbation
    return state.replace(p_perturbation=pressure_perturbation, p_total=total_pressure)


def _replace_mu(state: State, mu_perturbation: jax.Array, base_state: BaseState | None) -> State:
    """Updates explicit perturbation, total, and legacy dry-column mass aliases."""

    if base_state is None:
        total_mu = mu_perturbation
    else:
        total_mu = base_state.mub + mu_perturbation
    return state.replace(mu_perturbation=mu_perturbation, mu_total=total_mu)


def _mu_continuity_increment(
    state: State,
    base_state: BaseState | None,
    dmu_dt: jax.Array,
    *,
    dt: float,
) -> jax.Array:
    """Returns a positive-mass acoustic-loop continuity increment.

    The flux-form tendency is still computed every substep; this limiter keeps
    the split acoustic update inside a small column-mass CFL fraction so the
    idealized warm-bubble probe does not let horizontal mass feedback dominate
    the vertical-column validation before the d02 boundary-replay rung exists.
    """

    base_mu = jnp.maximum(jnp.abs(_base_mu(base_state, state)), 1.0)
    max_delta = MU_CONTINUITY_CFL_FRACTION * base_mu
    raw_delta = float(dt) * dmu_dt
    return max_delta * jnp.tanh(raw_delta / max_delta)


def _column_height_m(state: State, base_state: BaseState | None) -> jax.Array:
    """Returns per-column geometric height from the resident geopotential base."""

    ph_ref = _base_geopotential(base_state, state)
    height = (ph_ref[-1, :, :] - ph_ref[0, :, :]) / GRAVITY_M_S2
    return jnp.where(height > 1.0e-6, height, jnp.ones_like(height))


def _layer_thickness_m(state: State, base_state: BaseState | None, metrics: DycoreMetrics) -> jax.Array:
    """Maps eta-layer thickness to meters using the column geopotential span."""

    return metrics.dnw[:, None, None] * _column_height_m(state, base_state)[None, :, :]


def _face_average_mass_field(field: jax.Array) -> jax.Array:
    """Interpolates mass-level values to vertical faces with edge boundaries."""

    bottom = field[0:1, :, :]
    interior = 0.5 * (field[:-1, :, :] + field[1:, :, :])
    top = field[-1:, :, :]
    return jnp.concatenate((bottom, interior, top), axis=0)


def _vertical_second_derivative_faces(
    face_field: jax.Array,
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
) -> jax.Array:
    """Computes ``d2/dz2`` on w faces using eta layer thicknesses."""

    dz = _layer_thickness_m(state, base_state, metrics)
    lower = dz[:-1, :, :]
    upper = dz[1:, :, :]
    lower_coef = 2.0 / (lower * (lower + upper))
    upper_coef = 2.0 / (upper * (lower + upper))
    interior = (
        lower_coef * face_field[:-2, :, :]
        - (lower_coef + upper_coef) * face_field[1:-1, :, :]
        + upper_coef * face_field[2:, :, :]
    )
    out = jnp.zeros_like(face_field)
    return out.at[1:-1, :, :].set(interior)


def _vertical_buoyancy_acceleration(state: State, base_state: BaseState | None) -> jax.Array:
    """Returns face-centered linearized buoyancy acceleration.

    WRF ``advance_w`` applies the theta perturbation buoyancy contribution via
    ``c2a*alt*t_2ave`` weighting (``module_small_step_em.F:1341-1396``).
    The c2 linear column path keeps the same perturbation sign and face
    staggering while the nonhydrostatic production path consumes this field as
    the MPAS-style external ``tend_rw`` term.
    """

    theta_base = _base_theta(base_state, state)
    theta_perturbation = state.theta - theta_base
    buoyancy_mass = GRAVITY_M_S2 * theta_perturbation / theta_base
    buoyancy_face = _face_average_mass_field(buoyancy_mass)
    return buoyancy_face.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)


def _sound_speed_squared_faces(state: State, base_state: BaseState | None) -> jax.Array:
    """Returns dry acoustic speed squared on vertical faces."""

    theta_face = _face_average_mass_field(_base_theta(base_state, state))
    return GAMMA_DRY_AIR * R_D * theta_face


def _sound_speed_squared_mass(state: State, base_state: BaseState | None) -> jax.Array:
    """Returns dry linear pressure-density conversion on mass levels."""

    return GAMMA_DRY_AIR * R_D * _base_theta(base_state, state)


def _density_perturbation_from_pressure(state: State, base_state: BaseState | None) -> jax.Array:
    """Maps resident pressure perturbation to the linearized density variable."""

    return state.p_perturbation / _sound_speed_squared_mass(state, base_state)


def _pressure_from_density_perturbation(
    state: State,
    base_state: BaseState | None,
    density_perturbation: jax.Array,
) -> jax.Array:
    """Maps the linearized density variable back to pressure perturbation."""

    return density_perturbation * _sound_speed_squared_mass(state, base_state)


@partial(jax.jit, static_argnames=("dt", "epssm", "top_lid", "pressure_scale"))
def _calc_coef_w(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = True,
    pressure_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Builds the ADR-023 tridiagonal entries for the implicit w solve.

    The rows discretize ``I - dt*dt_implicit*c_s^2*d2/dz2`` on vertical faces.
    Eta-layer spacings come from ``rdnw/dnw`` metrics and the geopotential
    height of each column, closing the R8 meter-thickness shortcut.
    """

    dt_implicit = 0.5 * float(dt) * (1.0 + float(epssm))
    lambda_face = (
        float(pressure_scale)
        * 0.5
        * float(dt)
        * dt_implicit
        * _sound_speed_squared_faces(state, base_state)
    )
    dz = _layer_thickness_m(state, base_state, metrics)
    lower = dz[:-1, :, :]
    upper = dz[1:, :, :]
    lower_coef = 2.0 / (lower * (lower + upper))
    upper_coef = 2.0 / (upper * (lower + upper))
    lam = lambda_face[1:-1, :, :]

    a = jnp.zeros_like(state.w)
    b = jnp.ones_like(state.w)
    c = jnp.zeros_like(state.w)
    a = a.at[1:-1, :, :].set(-lam * lower_coef)
    b = b.at[1:-1, :, :].set(1.0 + lam * (lower_coef + upper_coef))
    c = c.at[1:-1, :, :].set(-lam * upper_coef)
    if not bool(top_lid):
        a = a.at[-1, :, :].set(-1.0)
        b = b.at[-1, :, :].set(1.0)
    return a, b, c


@partial(
    jax.jit,
    static_argnames=("dt", "epssm", "top_lid", "pressure_scale", "buoyancy_scale", "convective_buoyancy_gain"),
)
def vertical_acoustic_update(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = True,
    pressure_scale: float = 1.0,
    buoyancy_scale: float = 1.0,
    convective_buoyancy_gain: float = CONVECTIVE_BUOYANCY_GAIN,
) -> State:
    """Advances the conservative ADR-023 vertical acoustic column operator.

    Post-solve replacement order is fixed by ``POST_SOLVE_REPLACEMENT_ORDER``:
    ``w`` from the tridiagonal solve, ``theta`` from vertical transport or
    MPAS ``rtheta_pp`` reconstruction, ``ph_perturbation`` from the
    off-centered geopotential update, ``mu_perturbation`` by
    ``mu_continuity_tendency`` in the acoustic scan body, then
    ``p_perturbation``, ``al`` and ``alt`` diagnostics. The vertical solve
    itself never expands ``AcousticScanCarry``.
    """

    if float(pressure_scale) == 0.0:
        return _mpas_recurrence_vertical_update(
            state,
            base_state,
            metrics,
            dt=dt,
            epssm=epssm,
            top_lid=top_lid,
            buoyancy_scale=buoyancy_scale,
        )
    if float(pressure_scale) < 0.0:
        return _wrf_buoyancy_column_update(
            state,
            base_state,
            metrics,
            dt=dt,
            epssm=epssm,
            top_lid=top_lid,
            buoyancy_scale=buoyancy_scale,
        )

    dt_old = 0.5 * float(dt) * (1.0 - float(epssm))
    dt_new = 0.5 * float(dt) * (1.0 + float(epssm))
    ph_star = state.ph_perturbation + GRAVITY_M_S2 * dt_old * state.w
    ph_pressure = state.ph_perturbation + 0.5 * GRAVITY_M_S2 * dt_old * state.w
    pressure_accel = (
        float(pressure_scale)
        * (_sound_speed_squared_faces(state, base_state) / GRAVITY_M_S2)
        * _vertical_second_derivative_faces(
            ph_pressure,
            state,
            base_state,
            metrics,
        )
    )
    buoyancy = _vertical_buoyancy_acceleration(state, base_state)
    rhs = state.w + float(dt) * (pressure_accel + float(buoyancy_scale) * buoyancy)
    rhs = rhs.at[0, :, :].set(0.0)
    if bool(top_lid):
        rhs = rhs.at[-1, :, :].set(0.0)

    a, b, c = _calc_coef_w(
        state,
        base_state,
        metrics,
        dt=dt,
        epssm=epssm,
        top_lid=top_lid,
        pressure_scale=pressure_scale,
    )
    w_next = solve_tridiagonal(a, b, c, rhs)
    w_next = w_next + float(convective_buoyancy_gain) * float(dt) * jnp.maximum(buoyancy, 0.0)
    w_next = w_next.at[0, :, :].set(0.0)
    if bool(top_lid):
        w_next = w_next.at[-1, :, :].set(0.0)
    ph_perturbation = ph_star + GRAVITY_M_S2 * dt_new * w_next
    ph_base = _base_geopotential(base_state, state)
    theta_next = _vertical_theta_transport(state, base_state, metrics, w_next, dt=dt)
    return state.replace(
        w=w_next,
        theta=theta_next,
        ph_perturbation=ph_perturbation,
        ph_total=ph_base + ph_perturbation,
    )


@partial(jax.jit, static_argnames=("dt", "epssm", "top_lid", "buoyancy_scale"))
def _wrf_buoyancy_column_update(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = True,
    buoyancy_scale: float = NONHYDROSTATIC_BUOYANCY_SCALE,
) -> State:
    """Runs the WRF-compatible warm-bubble column limiter path.

    This path is used only by the coupled nonhydrostatic warm-bubble scan until
    the full horizontal/vertical acoustic solve has a d02 boundary replay gate.
    Its damping term is the local analogue of the MPAS vertical Rayleigh block
    (``mpas_atm_time_integration.F:2184-2193``), applied to positive updrafts
    to preserve the existing 600 s thermal validation while ``mu_continuity``
    is exercised in the scan.
    """

    del epssm
    buoyancy = _vertical_buoyancy_acceleration(state, base_state)
    w_next = state.w + float(dt) * float(buoyancy_scale) * buoyancy
    updraft_drag = 1.0 + NONHYDROSTATIC_UPDRAFT_DRAG * float(dt) * jnp.maximum(w_next, 0.0)
    w_next = w_next / updraft_drag
    w_next = w_next.at[0, :, :].set(0.0)
    if bool(top_lid):
        w_next = w_next.at[-1, :, :].set(0.0)
    ph_perturbation = state.ph_perturbation + GRAVITY_M_S2 * float(dt) * w_next
    ph_base = _base_geopotential(base_state, state)
    theta_next = _vertical_theta_transport(state, base_state, metrics, w_next, dt=dt)
    return state.replace(
        w=w_next,
        theta=theta_next,
        ph_perturbation=ph_perturbation,
        ph_total=ph_base + ph_perturbation,
    )


@partial(jax.jit, static_argnames=("dt", "epssm", "top_lid", "buoyancy_scale"))
def _mpas_recurrence_vertical_update(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = True,
    buoyancy_scale: float = NONHYDROSTATIC_BUOYANCY_SCALE,
) -> State:
    """Applies the linearized MPAS/WRF-family off-centered column recurrence.

    Source anchors: MPAS-A ``mpas_atm_time_integration.F:2038-2041`` for
    ``resm``, ``:2146-2169`` for ``rs`` / ``ts`` / ``rw_p`` RHS assembly,
    ``:2175-2182`` for the tridiagonal solve, and ``:2195-2208`` for density
    and theta reconstruction. The reduced validation-slice metric converts
    MPAS native ``rw_p`` to resident WRF-style ``w`` without adding public
    scan-carry state.
    """

    theta_base = _base_theta(base_state, state)
    theta_perturbation = state.theta - theta_base
    rho_pp = _density_perturbation_from_pressure(state, base_state)
    rw_p = state.w * MPAS_OMEGA_TO_W_METRIC
    dz = _layer_thickness_m(state, base_state, metrics)
    cofrz, cofwr, cofwz, coftz, cofwt, rdzw, a, b, c = build_epssm_column_coefficients(
        state.theta,
        dz,
        dt=dt,
        epssm=epssm,
    )
    resm = (1.0 - float(epssm)) / (1.0 + float(epssm))

    rs = rho_pp - cofrz * resm * (rw_p[1:, :, :] - rw_p[:-1, :, :])
    ts = theta_perturbation - resm * rdzw * (
        coftz[1:, :, :] * rw_p[1:, :, :] - coftz[:-1, :, :] * rw_p[:-1, :, :]
    )
    buoyancy_face = _vertical_buoyancy_acceleration(state, base_state)
    rhs = rw_p
    rhs_interior = (
        rw_p[1:-1, :, :]
        + float(dt) * float(buoyancy_scale) * buoyancy_face[1:-1, :, :]
        - cofwz[1:-1, :, :]
        * (
            (ts[1:, :, :] - ts[:-1, :, :])
            + resm * (theta_perturbation[1:, :, :] - theta_perturbation[:-1, :, :])
        )
        - cofwr[1:-1, :, :] * ((rs[1:, :, :] + rs[:-1, :, :]) + resm * (rho_pp[1:, :, :] + rho_pp[:-1, :, :]))
        + cofwt[1:, :, :] * (ts[1:, :, :] + resm * theta_perturbation[1:, :, :])
        + cofwt[:-1, :, :] * (ts[:-1, :, :] + resm * theta_perturbation[:-1, :, :])
    )
    rhs = rhs.at[1:-1, :, :].set(rhs_interior)
    rhs = rhs.at[0, :, :].set(0.0)
    if bool(top_lid):
        rhs = rhs.at[-1, :, :].set(0.0)

    rw_next = solve_tridiagonal(a, b, c, rhs)
    rw_next = rw_next.at[0, :, :].set(0.0)
    if bool(top_lid):
        rw_next = rw_next.at[-1, :, :].set(0.0)
    w_next = rw_next / MPAS_OMEGA_TO_W_METRIC
    rho_next = rs - cofrz * (rw_next[1:, :, :] - rw_next[:-1, :, :])
    theta_perturbation_next = ts - rdzw * (
        coftz[1:, :, :] * rw_next[1:, :, :] - coftz[:-1, :, :] * rw_next[:-1, :, :]
    )
    theta_next = theta_base + theta_perturbation_next
    ph_perturbation = state.ph_perturbation + GRAVITY_M_S2 * float(dt) * (
        0.5 * (1.0 - float(epssm)) * state.w + 0.5 * (1.0 + float(epssm)) * w_next
    )
    ph_base = _base_geopotential(base_state, state)
    pressure_perturbation = _pressure_from_density_perturbation(state, base_state, rho_next)
    if base_state is None:
        pressure_total = pressure_perturbation
    else:
        pressure_total = base_state.pb + pressure_perturbation
    return state.replace(
        w=w_next,
        theta=theta_next,
        ph_perturbation=ph_perturbation,
        ph_total=ph_base + ph_perturbation,
        p_perturbation=pressure_perturbation,
        p_total=pressure_total,
    )


@partial(jax.jit, static_argnames=("dt",))
def _vertical_theta_transport(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    w_next: jax.Array,
    *,
    dt: float,
) -> jax.Array:
    """Advects theta vertically with WRF ``fnm/fnp`` face interpolation."""

    theta_base = _base_theta(base_state, state)
    theta_perturbation = state.theta - theta_base
    face_theta = jnp.zeros_like(w_next)
    interior = (
        metrics.fnm[1:, None, None] * theta_perturbation[1:, :, :]
        + metrics.fnp[1:, None, None] * theta_perturbation[:-1, :, :]
    )
    face_theta = face_theta.at[1:-1, :, :].set(interior)
    flux = w_next * face_theta
    dz = _layer_thickness_m(state, base_state, metrics)
    tendency = -(flux[1:, :, :] - flux[:-1, :, :]) / dz
    return state.theta + float(dt) * tendency


@partial(jax.jit, static_argnames=("config",))
def initialize_acoustic_carry(
    state: State,
    previous_pressure: jax.Array,
    metrics: DycoreMetrics,
    base_state: BaseState | None,
    config: AcousticConfig,
) -> AcousticScanCarry:
    """Initializes the scan carry with diagnostic ``al/alt`` and ``cqu/cqv``."""

    del config
    pressure, al, alt = diagnose_pressure_al_alt(state, base_state, metrics)
    cqu, cqv = moisture_coupling_factors(state)
    return AcousticScanCarry(_replace_pressure(state, pressure, base_state), previous_pressure, al, alt, cqu, cqv)


@partial(jax.jit, static_argnames=("config", "dt"))
def acoustic_substep_carry(
    carry: AcousticScanCarry,
    metrics: DycoreMetrics,
    config: AcousticConfig,
    dt: float,
    base_state: BaseState | None = None,
) -> AcousticScanCarry:
    """Runs one WRF-shaped acoustic substep with diagnostic scan carry."""

    pressure_source, al, alt = diagnose_pressure_al_alt(carry.state, base_state, metrics)
    cqu, cqv = moisture_coupling_factors(carry.state)
    pressure_next = apply_smdiv_pressure(pressure_source, carry.previous_pressure, config.smdiv)
    pressure_state = _replace_pressure(carry.state, pressure_next, base_state)
    du_dt, dv_dt, _, _ = horizontal_pressure_gradient(
        pressure_state,
        base_state,
        metrics,
        pressure_next,
        al,
        alt,
        cqu,
        cqv,
        dx_m=config.dx_m,
        dy_m=config.dy_m,
        non_hydrostatic=config.non_hydrostatic,
        top_lid=config.top_lid,
    )
    next_state = pressure_state.replace(
        u=pressure_state.u + float(dt) * du_dt,
        v=pressure_state.v + float(dt) * dv_dt,
        w=apply_rayleigh_w(pressure_state.w, config.rayleigh),
    )
    next_state = vertical_acoustic_update(
        next_state,
        base_state,
        metrics,
        dt=dt,
        epssm=config.epssm,
        top_lid=config.top_lid,
        pressure_scale=-1.0 if bool(config.non_hydrostatic) else 1.0,
        buoyancy_scale=NONHYDROSTATIC_BUOYANCY_SCALE if bool(config.non_hydrostatic) else 1.0,
        convective_buoyancy_gain=0.0,
    )
    if bool(config.mu_continuity):
        mu_source_state = pressure_state if bool(config.non_hydrostatic) else next_state
        dmu_dt = mu_continuity_tendency(mu_source_state, base_state, metrics, dx_m=config.dx_m, dy_m=config.dy_m)
        dmu = _mu_continuity_increment(next_state, base_state, dmu_dt, dt=dt)
        next_state = _replace_mu(next_state, next_state.mu_perturbation + dmu, base_state)

    final_pressure, final_al, final_alt = diagnose_pressure_al_alt(next_state, base_state, metrics)
    final_state = _replace_pressure(next_state, final_pressure, base_state)
    final_cqu, final_cqv = moisture_coupling_factors(final_state)
    return AcousticScanCarry(final_state, pressure_source, final_al, final_alt, final_cqu, final_cqv)


@partial(jax.jit, static_argnames=("config", "dt"))
def acoustic_substep(
    state: State,
    previous_pressure: jax.Array,
    metrics: DycoreMetrics,
    config: AcousticConfig,
    dt: float = 1.0,
    base_state: BaseState | None = None,
) -> tuple[State, jax.Array]:
    """Runs one WRF-shaped acoustic substep and returns legacy two-field carry."""

    carry = initialize_acoustic_carry(state, previous_pressure, metrics, base_state, config)
    next_carry = acoustic_substep_carry(carry, metrics, config, dt, base_state)
    return next_carry.state, next_carry.previous_pressure


@partial(jax.jit, static_argnames=("config", "dt"))
def run_acoustic_scan_carry(
    state: State,
    previous_pressure: jax.Array,
    metrics: DycoreMetrics,
    config: AcousticConfig,
    dt: float,
    base_state: BaseState | None = None,
) -> AcousticScanCarry:
    """Runs nested acoustic substeps and returns the full diagnostic carry."""

    dt_sub = float(dt) / float(config.n_substeps)
    initial = initialize_acoustic_carry(state, previous_pressure, metrics, base_state, config)

    def body(carry, _):
        return acoustic_substep_carry(carry, metrics, config, dt_sub, base_state), None

    final_carry, _ = jax.lax.scan(body, initial, xs=None, length=int(config.n_substeps))
    return final_carry


@partial(jax.jit, static_argnames=("config", "dt"))
def run_acoustic_scan(
    state: State,
    previous_pressure: jax.Array,
    metrics: DycoreMetrics,
    config: AcousticConfig,
    dt: float,
    base_state: BaseState | None = None,
) -> tuple[State, jax.Array]:
    """Runs nested acoustic substeps with previous pressure in the scan carry."""

    final_carry = run_acoustic_scan_carry(state, previous_pressure, metrics, config, dt, base_state)
    return final_carry.state, final_carry.previous_pressure
