"""WRF-shaped acoustic scan and horizontal pressure-gradient force."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass, field
from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.precision import force_fp64_island
from gpuwrf.contracts.state import BaseState, State
# WRF/MPAS source anchors for the imported damping hooks:
# module_small_step_em.F:548-563, :1559-1569 and
# mpas_atm_time_integration.F:2184-2192.
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig, apply_rayleigh_w, apply_smdiv_pressure
from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients, solve_tridiagonal


configure_jax_x64()


R_D = 287.0
R_V = 461.6
# WRF share/module_model_constants.F:41 rvovrd = r_v/r_d (~1.6084). The EOS
# carries ONE qv factor (module_big_step_utilities_em.F use_theta_m branch):
#   use_theta_m=1 (WRF default; production State.theta = MOIST theta_m):
#       qvf = 1, because theta_m = theta_dry*(1+rvovrd*qv) already carries the
#       moisture factor (pass qv=None below; matches the bit-exact start_em
#       transcription d02_replay._wrf_live_nest_start_domain_perturb_init).
#   use_theta_m=0 (DRY theta callers only): qvf = 1 + rvovrd*qv
#       (module_big_step_utilities_em.F:1064,1118,1140).
# NEVER 1 + p608*qv: p608 = rvovrd-1 belongs to the virtual-temperature/
# moist-density convention, not WRF's dry-alpha_d EOS (v0.14 h1 root cause).
RVOVRD = R_V / R_D
# WRF share/module_model_constants.F:20 ``cp = 7.*r_d/2.`` = 1004.5 EXACTLY.
# v0.14 acoustic-substep root-cause (proofs/v014/switzerland_acoustic_substep_
# blocker.json): the previous ``CP_D = 1004.0`` shifted CVPM by +1.42e-4 and
# CPOVCV by +2.79e-4, putting EVERY dycore EOS evaluation (alt, the diagnostic
# p inversion, c2a = cpovcv*(pb+p)/alt, the implicit-w coefficients) off WRF by
# a one-signed ~1e-4*ln(p/p0) relative error: the recomputed ``alt`` deviated
# from the WRF-carried ``grid%alt`` by mean -9.7e-4 / max 5.0e-3 m3/kg at a
# bit-identical state (collapses to 2.8e-7 with the WRF value).
CP_D = 7.0 * R_D / 2.0
P0_PA = 100000.0
CPOVCV = CP_D / (CP_D - R_D)
CVPM = -(CP_D - R_D) / CP_D
GAMMA_DRY_AIR = CP_D / (CP_D - R_D)
# WRF share/module_model_constants.F:17 ``g = 9.81`` EXACTLY (not the SI
# standard 9.80665).  The previous 9.80665 here made calc_coef_w build the
# implicit-w solve coefficients with a different gravity than the advance_w
# RHS terms (core/advance_w.py GRAVITY_M_S2 = 9.81), breaking the exact
# implicit cancellation WRF has -- a one-signed 3.4e-4 inconsistency INSIDE
# the same tridiagonal solve, every acoustic substep.
GRAVITY_M_S2 = 9.81
MIN_PRESSURE_PA = 1.0
MIN_ALT = 1.0e-8
CONVECTIVE_BUOYANCY_GAIN = 0.0
# Slice-only compatibility: WRF module_small_step_em.F:1451-1489 and MPAS
# mpas_atm_time_integration.F:2160-2169 do not define a production scalar gain.
SLICE_ONLY_MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38
MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = SLICE_ONLY_MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE
SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE = 1.0
TEMPORARY_MU_CONTINUITY_CFL_FRACTION = 1.0e-3
# Legacy diagnostic import compatibility only. The operator uses
# ``_mpas_w_metric_faces`` instead of this scalar.
MPAS_OMEGA_TO_W_METRIC = 1.0
POST_SOLVE_REPLACEMENT_ORDER = (
    "w",
    "theta",
    "ph_perturbation",
    "mu_perturbation",
    "p_perturbation",
    "al",
    "alt",
)
_SHARDED_HALO_CONTEXT: tuple[object, int] | None = None


@dataclass(frozen=True)
class AcousticConfig:
    """Static acoustic-substep config for c2 nested scans.

    WRF source anchors: ``module_small_step_em.F:548-563`` for ``smdiv``
    pressure memory and ``module_small_step_em.F:1559-1569`` for top
    Rayleigh damping on vertical velocity.
    """

    n_substeps: int = 1
    dx_m: float = 1.0
    dy_m: float = 1.0
    non_hydrostatic: bool = True
    top_lid: bool = True
    mu_continuity: bool = True
    epssm: float = 0.1
    # WRF source anchor: module_small_step_em.F:548-563.
    smdiv: SmdivConfig = field(default_factory=SmdivConfig)
    # WRF/MPAS source anchors: module_small_step_em.F:1559-1569 and
    # mpas_atm_time_integration.F:2184-2192.
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
    """Keeps WRF equation-of-state diagnostics finite outside the state carry."""

    return jnp.maximum(alt, jnp.asarray(MIN_ALT, dtype=alt.dtype))


def _inverse_density_from_theta_pressure(theta: jax.Array, pressure: jax.Array, qv: jax.Array | None = None) -> jax.Array:
    """WRF inverse-density equation of state (dry alpha_d).

    WRF source anchors: ``module_big_step_utilities_em.F:1085-1087`` and
    ``module_small_step_em.F:527-528``. ``theta`` convention: production
    ``State.theta`` is MOIST ``theta_m`` (``use_theta_m=1``) -> pass ``qv=None``
    (``qvf=1``). Pass ``qv`` ONLY with genuinely DRY theta (``use_theta_m=0``
    form, ``qvf=1+rvovrd*qv``); the two forms are the same WRF EOS.
    """

    # v0.20 S2 intrinsic fp64-island lock: the WRF equation of state is the
    # canonical cancellation bracket (small inverse-density / pressure residuals of
    # large theta*pressure powers). Widen the EOS inputs to fp64 IN-OPERATOR so an
    # fp32 storage downcast cannot contaminate the diagnosis. No-op (bit-identical)
    # on fp64_default.
    theta, pressure, qv = force_fp64_island(theta, pressure, qv)
    qvf = 1.0 if qv is None else 1.0 + RVOVRD * qv
    return (R_D / P0_PA) * theta * qvf * ((_safe_pressure(pressure) / P0_PA) ** CVPM)


def _pressure_from_theta_alt(theta: jax.Array, alt: jax.Array, qv: jax.Array | None = None) -> jax.Array:
    """Inverts WRF's equation of state for diagnostic pressure.

    Same ``theta``/``qv`` convention as
    :func:`_inverse_density_from_theta_pressure`: moist ``theta_m`` callers
    (production ``State.theta``) pass ``qv=None``; dry-theta callers pass ``qv``.
    """

    # v0.20 S2 intrinsic fp64-island lock (EOS pressure inversion). No-op
    # (bit-identical) on fp64_default; protects an fp32 caller's diagnostic.
    theta, alt, qv = force_fp64_island(theta, alt, qv)
    qvf = 1.0 if qv is None else 1.0 + RVOVRD * qv
    argument = (R_D * theta * qvf) / (P0_PA * _safe_alt(alt))
    return P0_PA * (jnp.maximum(argument, 1.0e-12) ** CPOVCV)


def _maybe_sharded_x_edge_pair(
    field: jax.Array,
    left: jax.Array,
    right: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Make x-face pairs match global-domain edge padding under opt-in x sharding."""

    context = _SHARDED_HALO_CONTEXT
    if context is None:
        return left, right
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return left, right
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("acoustic_wrf sharded edge-pair correction supports x-axis decomposition only")
    h = int(width)
    owned = int(field.shape[-1]) - 2 * h
    if owned < 1:
        raise ValueError("haloed x field has no owned cells")
    rank = jax.lax.axis_index(str(sharding.axis_name))
    start = rank * owned
    global_nx = owned * int(sharding.resolved_partitions())
    west_face = h
    east_face = h + owned
    is_first = start == 0
    is_last = start + owned == global_nx
    left = left.at[..., west_face].set(jnp.where(is_first, right[..., west_face], left[..., west_face]))
    right = right.at[..., east_face].set(jnp.where(is_last, left[..., east_face], right[..., east_face]))
    return left, right


def _x_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns left/right mass values at x-staggered faces with edge BCs."""

    padded = jnp.pad(field, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return _maybe_sharded_x_edge_pair(field, padded[:, :, :-1], padded[:, :, 1:])


def _y_face_pair_3d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns south/north mass values at y-staggered faces with edge BCs."""

    padded = jnp.pad(field, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return padded[:, :-1, :], padded[:, 1:, :]


def _x_face_pair_2d(field: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns left/right 2-D mass values at x-staggered faces."""

    padded = jnp.pad(field, ((0, 0), (1, 1)), mode="edge")
    return _maybe_sharded_x_edge_pair(field, padded[:, :-1], padded[:, 1:])


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


@partial(jax.jit, static_argnames=("hypsometric_opt",))
def diagnose_pressure_al_alt(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    hypsometric_opt: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes substep-local pressure, ``al``, and ``alt``.

    With an explicit ``BaseState``, this follows WRF's non-hydrostatic
    ``calc_p_rho_phi`` pattern: ``al`` from geopotential/mass
    (``module_big_step_utilities_em.F:1025-1062``), diagnostic pressure
    from the equation of state (``:1082-1087``), and ``alt = al + alb``
    (``:910-943``). Without a base state, the c2 architecture smoke tests
    use the already-consistent legacy pressure while still carrying ``alt``.

    ``hypsometric_opt`` selects the WRF ``calc_p_rho_phi`` specific-volume
    relation (Registry.EM_COMMON:2285, **WRF default 2**):

    * ``1`` -- linear ``dp = mut*d(eta)`` form (``:1027-1030``).  This was the
      ONLY form implemented before v0.14 and is kept as the function default so
      idealized callers (whose generators carry placeholder ``c3f/c4f``) are
      byte-unchanged.
    * ``2`` -- LOG-pressure-thickness form ``al = dZ/(p*dLOG(p)) - alb``
      (``:1043-1062``), the WRF v4 registry DEFAULT every real case runs with.
      v0.14 root cause (proofs/v014/switzerland_hpg_native_face_fix.json): the
      h36 Switzerland CPU live ``alt`` matches this form to fp32 roundoff
      (~1.4e-6 rel) while the linear form is off one-signed by ~4.2e-4 mean /
      6.2e-4 max rel, horizontally modulated by terrain ``muts`` -- a spurious
      large-step PGF over the Alps (the d01 h36 mass-venting blocker).
    """

    if base_state is None:
        # State.theta is moist theta_m (use_theta_m=1) -> WRF EOS qvf=1.
        alt = _inverse_density_from_theta_pressure(state.theta, state.p_total)
        al = jnp.zeros_like(alt)
        return state.p_perturbation, al, alt

    # Base inverse density ``alb`` must be the EXACT discrete inverse density the
    # base geopotential ``phb`` was hydrostatically integrated from at init
    # (WRF dyn_em/module_initialize_real.F:3817:
    #   phb(k+1) = phb(k) - dnw(k)*(c1h(k)*mub + c2h(k))*alb(k)).
    # Recover it by inverting that relation from the base state's OWN phb/mub
    # rather than recomputing it from a base potential temperature, because the
    # operational path historically rebuilds ``theta_base`` as a CONSTANT 300 K
    # (operational_mode._theta_base_offset, _refresh_grid_p_from_finished), which
    # makes the recomputed ``alb`` disagree with the discrete ``alb`` the loaded
    # ``phb`` carries -- by up to ~35 % aloft (300 K vs the realistic t0+t_init
    # base profile that reaches ~465 K near the lid).  That mismatch put the loaded
    # IC out of the dycore's discrete hydrostatic balance and drove the steady
    # ~+2.6 kPa diagnostic perturbation-pressure / Exner-T2 offset on BOTH d02
    # (force_geopotential=True) and d03.  Inverting phb makes ``alb`` exact and
    # caller-agnostic.  See .agent/reviews/2026-06-01-opus-pressure-drift-rootcause.md.
    base_pressure = _safe_pressure(base_state.pb)
    mass_h_base = metrics.c1h[:, None, None] * base_state.mub[None, :, :] + metrics.c2h[:, None, None]
    dphb = base_state.phb[1:, :, :] - base_state.phb[:-1, :, :]
    denom_alb = metrics.dnw[:, None, None] * mass_h_base
    safe_denom_alb = jnp.where(
        jnp.abs(denom_alb) > 1.0e-12, denom_alb, jnp.asarray(1.0e-12, dtype=denom_alb.dtype)
    )
    alb = -dphb / safe_denom_alb
    muts = base_state.mub + state.mu_perturbation
    mass_weight = metrics.c1h[:, None, None] * muts[None, :, :] + metrics.c2h[:, None, None]
    safe_mass = jnp.where(jnp.abs(mass_weight) > 1.0e-12, mass_weight, jnp.asarray(1.0e-12, dtype=mass_weight.dtype))
    # WRF calc_p_rho_phi non-hydrostatic al (module_big_step_utilities_em.F:1029):
    #   al = -1/(c1*muts+c2) * ( alb*c1*mu' + rdnw*(ph'(k+1)-ph'(k)) )
    # The rdnw*(ph'(k+1)-ph'(k)) geopotential term was previously DROPPED, so a
    # hydrostatically rebalanced bubble (mu'=0, ph'!=0) produced al==0 and a dead
    # p_perturbation -- the warm/cold thermal never developed a perturbation
    # pressure.  Restore it (F7F).
    ph_pert = state.ph_perturbation.astype(alb.dtype)
    if int(hypsometric_opt) == 2:
        # WRF calc_p_rho_phi hypsometric_opt=2 (module_big_step_utilities_em.F:
        # 1043-1062, the v4 Registry DEFAULT): the pressure depth dp = mut*d(eta)
        # is replaced by p*dLOG(p) on the DRY reference column
        #   pfu = c3f(k+1)*muts + c4f(k+1) + ptop   (upper face)
        #   pfd = c3f(k  )*muts + c4f(k  ) + ptop   (lower face)
        #   phm = c3h(k  )*muts + c4h(k  ) + ptop   (half level)
        #   al  = (ph(k+1)-ph(k)+phb(k+1)-phb(k)) / phm / LOG(pfd/pfu) - alb
        # CRITICAL (native-face proof, proofs/v014/switzerland_hpg_native_face_fix
        # .json::wrf_faces): the subtracted ``alb`` must ALSO be the LOG-form base
        # (real.exe integrates PHB from alb with the SAME relation under opt 2);
        # subtracting the linear reconstruction leaves the one-signed hypso bias
        # inside ``al`` and corrupts the dominant HPG ``pb*al`` face term.  With
        # the log-base alb the live WRF ``al`` is reproduced to ~1.7e-6 rel.
        p_top = jnp.reshape(metrics.p_top, ()).astype(alb.dtype)
        muts_col = muts[None, :, :]
        mub_col = base_state.mub[None, :, :]

        def _log_alpha(dph: jax.Array, mass_col: jax.Array) -> jax.Array:
            pfu = metrics.c3f[1:, None, None] * mass_col + metrics.c4f[1:, None, None] + p_top
            pfd = metrics.c3f[:-1, None, None] * mass_col + metrics.c4f[:-1, None, None] + p_top
            phm = metrics.c3h[:, None, None] * mass_col + metrics.c4h[:, None, None] + p_top
            return dph / phm / jnp.log(pfd / pfu)

        dph_tot = (base_state.phb[1:, :, :] + ph_pert[1:, :, :]) - (
            base_state.phb[:-1, :, :] + ph_pert[:-1, :, :]
        )
        alb_used = _log_alpha(base_state.phb[1:, :, :] - base_state.phb[:-1, :, :], mub_col)
        al = _log_alpha(dph_tot, muts_col) - alb_used
    else:
        mu_term = alb * metrics.c1h[:, None, None] * state.mu_perturbation[None, :, :]
        geo_term = metrics.rdnw[:, None, None] * (ph_pert[1:, :, :] - ph_pert[:-1, :, :])
        al = -(mu_term + geo_term) / safe_mass
        alb_used = alb
    alt = al + alb_used
    # WRF diagnostic perturbation pressure (module_big_step_utilities_em.F:1083-1087)
    # uses the FULL theta (t0+theta') in the nonlinear EOS, not the base theta:
    #   p = p0*( Rd*(t0+theta')*qvf/(p0*(al+alb)) )^cpovcv - pb
    # State.theta is moist theta_m (use_theta_m=1) -> qvf=1, matching WRF's
    # use_theta_m branch and the bit-exact start_em init transcription
    # (d02_replay.py qvf=1); rk_addtend_dry/_acoustic_core_state already use
    # the same qvf=1 form.
    total_pressure = _pressure_from_theta_alt(state.theta, alt)
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

    WRF applies ``dpxy`` to mass-coupled small-step momentum
    (``small_step_prep`` at ``module_small_step_em.F:238-254``; update at
    ``:868`` and ``:942``). This routine returns velocity tendencies, so the
    final pressure-gradient force is decoupled by the same face dry-column mass
    used in the WRF ``dpxy`` construction.
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

    safe_mass_x = jnp.maximum(jnp.abs(mass_x), jnp.asarray(1.0e-12, dtype=dpx.dtype))
    safe_mass_y = jnp.maximum(jnp.abs(mass_y), jnp.asarray(1.0e-12, dtype=dpy.dtype))
    du_dt = -cqu * dpx / safe_mass_x
    dv_dt = -cqv * dpy / safe_mass_y
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
    """Bounded mass update for the pre-d02 public scan path.

    DEFER to post-S2.1 sprint pending real Gen2 baseline. WRF updates
    ``MU/MUTS/MUAVE/ww`` without a tanh mass cap in
    ``module_small_step_em.F:1102-1119``; MPAS advances ``rho_pp/rw_p/wwAvg``
    without a dry-column tanh cap in ``mpas_atm_time_integration.F:2146-2199``.
    The bounded return is preserved only because the pre-d02 warm-bubble public
    scan becomes nonfinite without it; S2.1 must replace or ratify it.
    """

    base_mu = jnp.maximum(jnp.abs(_base_mu(base_state, state)), 1.0)
    max_delta = TEMPORARY_MU_CONTINUITY_CFL_FRACTION * base_mu
    raw_delta = float(dt) * dmu_dt
    # DEFER to post-S2.1: not WRF/MPAS source-backed; see
    # module_small_step_em.F:1102-1119 and mpas_atm_time_integration.F:2146-2199.
    return max_delta * jnp.tanh(raw_delta / max_delta)


def _column_height_m(state: State, base_state: BaseState | None) -> jax.Array:
    """Returns per-column geometric height from the resident geopotential base."""

    ph_ref = _base_geopotential(base_state, state)
    height = (ph_ref[-1, :, :] - ph_ref[0, :, :]) / GRAVITY_M_S2
    return jnp.where(height > 1.0e-6, height, jnp.ones_like(height))


def _layer_thickness_m(state: State, base_state: BaseState | None, metrics: DycoreMetrics) -> jax.Array:
    """Maps eta-layer thickness to meters using the column geopotential span."""

    return metrics.dnw[:, None, None] * _column_height_m(state, base_state)[None, :, :]


def _mpas_w_metric_faces(state: State, base_state: BaseState | None, metrics: DycoreMetrics) -> jax.Array:
    """Returns a per-face MPAS ``rw``/WRF ``w`` geometry metric.

    MPAS source anchors: ``mpas_atm_time_integration.F:2491-2495`` recovers
    diagnosed ``w`` by dividing ``rw`` by face ``zz`` geometry, and
    ``:5575-5585`` initializes ``rw`` from ``rho_zz``, ``w``, and ``zz``.
    This helper replaces the former synthetic column constant with a resident
    per-column/per-level metric from dry-column mass and eta geometry.
    """

    if str(metrics.provenance).startswith("analytic"):
        from gpuwrf.validation.mpas_oracles.mpas_column_slice import (
            MPAS_OMEGA_TO_W_METRIC as slice_oracle_omega_to_w_metric,
        )

        return jnp.ones_like(state.w) * jnp.asarray(slice_oracle_omega_to_w_metric, dtype=state.w.dtype)

    dry_column_mass = jnp.maximum(jnp.abs(_base_mu(base_state, state) + state.mu_perturbation), 1.0) / GRAVITY_M_S2
    dz_deta = _layer_thickness_m(state, base_state, metrics) / metrics.dnw[:, None, None]
    mass_geometry = dry_column_mass[None, :, :] * dz_deta
    reference = jnp.mean(mass_geometry, axis=0, keepdims=True)
    mass_metric = mass_geometry / jnp.maximum(jnp.abs(reference), 1.0e-12)
    return _face_average_mass_field(mass_metric)


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


def calc_coef_w_wrf_coefficients(
    mut: jax.Array,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = False,
    cqw: jax.Array | None = None,
    c2a: jax.Array | None = None,
    gravity: float = GRAVITY_M_S2,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns WRF ``calc_coef_w`` ``a``, ``alpha`` and ``gamma`` coefficients.

    Source: WRF ``dyn_em/module_small_step_em.F:624-649``. The M6B0-R
    coefficient fixture has ``cqw=1`` and ``c2a=1`` because its CPU-path
    extractor isolated the hybrid-coordinate coefficient construction.
    """

    mut = jnp.asarray(mut, dtype=jnp.float64)
    nz = int(metrics.c1h.shape[0])
    field_shape = (nz + 1,) + tuple(mut.shape)
    mass_shape = (nz,) + tuple(mut.shape)
    cqw = jnp.ones(field_shape, dtype=mut.dtype) if cqw is None else jnp.asarray(cqw, dtype=mut.dtype)
    c2a = jnp.ones(mass_shape, dtype=mut.dtype) if c2a is None else jnp.asarray(c2a, dtype=mut.dtype)

    mass_h = metrics.c1h[:, None, None] * mut[None, :, :] + metrics.c2h[:, None, None]
    mass_f = metrics.c1f[:, None, None] * mut[None, :, :] + metrics.c2f[:, None, None]
    rdn = metrics.rdn[:, None, None]
    rdnw = metrics.rdnw[:, None, None]

    cof = (0.5 * float(dt) * float(gravity) * (1.0 + float(epssm))) ** 2
    lid_flag = 0.0 if bool(top_lid) else 1.0
    a = jnp.zeros(field_shape, dtype=mut.dtype)
    alpha = jnp.ones(field_shape, dtype=mut.dtype)
    gamma = jnp.zeros(field_shape, dtype=mut.dtype)

    # WRF lines 624-627: lower boundary row, top lower diagonal, gamma seed.
    # WRF :626 uses c1f(kde-1) for the top a row; WRF :646 uses c1f(kde) for
    # the top b row. The two denominators differ whenever c1f[nz-1] != c1f[nz],
    # so split them explicitly.
    top_denom_a = mass_h[nz - 1] * mass_f[nz - 1]
    top_denom_b = mass_h[nz - 1] * mass_f[nz]
    a = a.at[1, :, :].set(0.0)
    a = a.at[nz, :, :].set(-2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1] * lid_flag / top_denom_a)
    gamma = gamma.at[0, :, :].set(0.0)

    # WRF lines 629-633: lower diagonal on interior w faces.
    for kk in range(2, nz):
        k = kk - 1
        denom = mass_h[k] * mass_f[k]
        a = a.at[kk, :, :].set(-cqw[kk] * cof * rdn[kk] * rdnw[kk - 1] * c2a[kk - 1] / denom)

    # WRF lines 635-642: diagonal/upper diagonal followed by Thomas forward sweep.
    for k in range(1, nz):
        denom_upper = mass_h[k] * mass_f[k]
        denom_lower = mass_h[k - 1] * mass_f[k]
        denom_c = mass_h[k] * mass_f[k + 1]
        b = 1.0 + cqw[k] * cof * rdn[k] * (
            rdnw[k] * c2a[k] / denom_upper + rdnw[k - 1] * c2a[k - 1] / denom_lower
        )
        c = -cqw[k] * cof * rdn[k] * rdnw[k] * c2a[k] / denom_c
        alpha_k = 1.0 / (b - a[k] * gamma[k - 1])
        alpha = alpha.at[k, :, :].set(alpha_k)
        gamma = gamma.at[k, :, :].set(c * alpha_k)

    # WRF lines 644-649: top row diagonal and gamma closure.
    b_top = 1.0 + 2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1] / top_denom_b
    alpha = alpha.at[nz, :, :].set(1.0 / (b_top - a[nz] * gamma[nz - 1]))
    gamma = gamma.at[nz, :, :].set(0.0)
    return a, alpha, gamma


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

    if float(pressure_scale) <= 0.0:
        return _mpas_recurrence_vertical_update(
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
def _mpas_recurrence_vertical_update(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = True,
    buoyancy_scale: float = SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE,
) -> State:
    """Applies the linearized MPAS/WRF-family off-centered column recurrence.

    Source anchors: MPAS-A ``mpas_atm_time_integration.F:2038-2041`` for
    ``resm``, ``:2146-2169`` for ``rs`` / ``ts`` / ``rw_p`` RHS assembly,
    ``:2175-2182`` for the tridiagonal solve, and ``:2195-2208`` for density
    and theta reconstruction. MPAS ``:2491-2495`` and ``:5575-5585`` define
    the geometry metric used to convert between native ``rw`` and diagnosed
    ``w`` without adding public scan-carry state.
    """

    theta_base = _base_theta(base_state, state)
    theta_perturbation = state.theta - theta_base
    rho_pp = _density_perturbation_from_pressure(state, base_state)
    mpas_w_metric = _mpas_w_metric_faces(state, base_state, metrics)
    rw_p = state.w * mpas_w_metric
    dz = _layer_thickness_m(state, base_state, metrics)
    cofrz, cofwr, cofwz, coftz, cofwt, rdzw, a, b, c = build_epssm_column_coefficients(
        state.theta,
        dz,
        dt=dt,
        epssm=epssm,
        theta_coefficient=theta_base,
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
    w_next = rw_next / mpas_w_metric
    rho_next = rs - cofrz * (rw_next[1:, :, :] - rw_next[:-1, :, :])
    theta_perturbation_next = ts - rdzw * (
        coftz[1:, :, :] * rw_next[1:, :, :] - coftz[:-1, :, :] * rw_next[:-1, :, :]
    )
    theta_next = theta_base + theta_perturbation_next
    ph_perturbation = state.ph_perturbation + GRAVITY_M_S2 * float(dt) * (
        0.5 * (1.0 - float(epssm)) * state.w + 0.5 * (1.0 + float(epssm)) * w_next
    )
    ph_base = _base_geopotential(base_state, state)
    if base_state is None:
        pressure_perturbation = state.p_perturbation
        pressure_total = state.p_total
    else:
        pressure_perturbation = _pressure_from_density_perturbation(state, base_state, rho_next)
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

    pressure, al, alt = diagnose_pressure_al_alt(state, base_state, metrics)
    cqu, cqv = moisture_coupling_factors(state)
    keep_resident_pressure = bool(config.non_hydrostatic) and base_state is not None
    pressure_state = state if keep_resident_pressure else _replace_pressure(state, pressure, base_state)
    return AcousticScanCarry(pressure_state, previous_pressure, al, alt, cqu, cqv)


@partial(jax.jit, static_argnames=("config", "dt"))
def acoustic_substep_carry(
    carry: AcousticScanCarry,
    metrics: DycoreMetrics,
    config: AcousticConfig,
    dt: float,
    base_state: BaseState | None = None,
) -> AcousticScanCarry:
    """Runs one WRF-shaped acoustic substep with diagnostic scan carry."""

    pressure_diag, al, alt = diagnose_pressure_al_alt(carry.state, base_state, metrics)
    keep_resident_pressure = bool(config.non_hydrostatic) and base_state is not None
    pressure_source = carry.state.p_perturbation if keep_resident_pressure else pressure_diag
    cqu, cqv = moisture_coupling_factors(carry.state)
    # WRF source anchor: module_small_step_em.F:548-563 applies smdiv pressure
    # memory as p = p + smdiv*(p-pm1).
    pressure_next = apply_smdiv_pressure(pressure_source, carry.previous_pressure, config.smdiv)
    pressure_state = carry.state if keep_resident_pressure else _replace_pressure(carry.state, pressure_next, base_state)
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
        # WRF source anchor: module_small_step_em.F:1559-1569 applies the
        # top-layer Rayleigh ramp to w; MPAS :2184-2192 damps rw_p implicitly.
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
        buoyancy_scale=(
            SLICE_ONLY_MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE
            if bool(config.non_hydrostatic) and str(metrics.provenance).startswith("analytic")
            else SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE
        ),
        convective_buoyancy_gain=0.0,
    )
    if bool(config.mu_continuity):
        mu_source_state = pressure_state if bool(config.non_hydrostatic) else next_state
        dmu_dt = mu_continuity_tendency(mu_source_state, base_state, metrics, dx_m=config.dx_m, dy_m=config.dy_m)
        dmu = _mu_continuity_increment(next_state, base_state, dmu_dt, dt=dt)
        next_state = _replace_mu(next_state, next_state.mu_perturbation + dmu, base_state)

    final_pressure, final_al, final_alt = diagnose_pressure_al_alt(next_state, base_state, metrics)
    final_state = next_state if keep_resident_pressure else _replace_pressure(next_state, final_pressure, base_state)
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
