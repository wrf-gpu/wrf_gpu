"""WRF-shaped small-step preparation for the operational RK acoustic path.

The source routine is WRF ``dyn_em/module_small_step_em.F:125-285``.
It saves the RK reference family, records the per-stage save arrays, builds
the small-step mass work fields, and prepares coupled perturbation work arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import State
from gpuwrf.dynamics.acoustic_wrf import CPOVCV, diagnose_pressure_al_alt, moisture_coupling_factors


THETA_BASE_OFFSET_K = 300.0
_SHARDED_HALO_CONTEXT: tuple[object, int] | None = None


def _base_mu(state: State) -> jax.Array:
    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def _maybe_sharded_u_face_average(field: jax.Array, face: jax.Array) -> jax.Array:
    context = _SHARDED_HALO_CONTEXT
    if context is None:
        return face
    sharding, width = context
    if not bool(getattr(sharding, "enabled", False)):
        return face
    if getattr(sharding, "axis", "x") != "x":
        raise NotImplementedError("small-step prep sharded face average supports x-axis decomposition only")
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
    face = face.at[:, west_face].set(jnp.where(is_first, field[:, h], face[:, west_face]))
    face = face.at[:, east_face].set(jnp.where(is_last, field[:, h + owned - 1], face[:, east_face]))
    return face


def _u_face_average_2d(field: jax.Array) -> jax.Array:
    west = field[:, :1]
    east = field[:, -1:]
    interior = 0.5 * (field[:, :-1] + field[:, 1:])
    return _maybe_sharded_u_face_average(field, jnp.concatenate((west, interior, east), axis=1))


def _v_face_average_2d(field: jax.Array) -> jax.Array:
    south = field[:1, :]
    north = field[-1:, :]
    interior = 0.5 * (field[:-1, :] + field[1:, :])
    return jnp.concatenate((south, interior, north), axis=0)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SmallStepPrepState:
    """Arrays prepared by WRF ``small_step_prep`` for one RK stage."""

    rk_step: int
    dt_rk: float
    entry_state: State
    theta_offset: jax.Array
    u_1: jax.Array
    v_1: jax.Array
    w_1: jax.Array
    theta_1: jax.Array
    theta_1_total: jax.Array
    ph_1: jax.Array
    mu_1: jax.Array
    u_save: jax.Array
    v_save: jax.Array
    w_save: jax.Array
    t_save: jax.Array
    ph_save: jax.Array
    mu_save: jax.Array
    ww_save: jax.Array
    mub: jax.Array
    mut: jax.Array
    muu: jax.Array
    muv: jax.Array
    muts: jax.Array
    muus: jax.Array
    muvs: jax.Array
    mu_work: jax.Array
    u_work: jax.Array
    v_work: jax.Array
    w_work: jax.Array
    theta_work: jax.Array
    ph_work: jax.Array
    c2a: jax.Array
    al: jax.Array
    alt: jax.Array
    pb: jax.Array
    cqu: jax.Array
    cqv: jax.Array
    c1h: jax.Array
    c2h: jax.Array
    c1f: jax.Array
    c2f: jax.Array
    rdnw: jax.Array
    msfuy: jax.Array
    msfvx: jax.Array
    msfty: jax.Array
    # WRF ``php`` = full geopotential at mass (pressure) points, built ONCE per RK
    # stage by ``calc_php`` in ``rk_step_prep`` (module_em.F:181;
    # module_big_step_utilities_em.F:1227-1266) from the STAGE-ENTRY ``grid%ph_2``
    # (= absolute perturbation geopotential ph') and the base ``grid%phb``.  WRF
    # holds it STAGE-CONSTANT and passes it INTENT(IN) to ``advance_uv`` every
    # acoustic substep (solve_em.F:1282; advance_uv :861/:935 4th PGF term).  WRF
    # does NOT recompute ``php`` from the live, substep-updated ``grid%ph_2``; the
    # live ``ph_2`` instead drives the SEPARATE first-3-terms gradient (:828-831).
    # Carrying this frozen array removes the split-explicit violation where the JAX
    # acoustic core re-diagnosed php from live ``state.ph`` each substep
    # (acoustic.py advance_uv_wrf).
    php: jax.Array

    def replace(self, **updates) -> "SmallStepPrepState":
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        children = (
            self.entry_state,
            self.theta_offset,
            self.u_1,
            self.v_1,
            self.w_1,
            self.theta_1,
            self.theta_1_total,
            self.ph_1,
            self.mu_1,
            self.u_save,
            self.v_save,
            self.w_save,
            self.t_save,
            self.ph_save,
            self.mu_save,
            self.ww_save,
            self.mub,
            self.mut,
            self.muu,
            self.muv,
            self.muts,
            self.muus,
            self.muvs,
            self.mu_work,
            self.u_work,
            self.v_work,
            self.w_work,
            self.theta_work,
            self.ph_work,
            self.c2a,
            self.al,
            self.alt,
            self.pb,
            self.cqu,
            self.cqv,
            self.c1h,
            self.c2h,
            self.c1f,
            self.c2f,
            self.rdnw,
            self.msfuy,
            self.msfvx,
            self.msfty,
            self.php,
        )
        return children, (int(self.rk_step), float(self.dt_rk))

    @classmethod
    def tree_unflatten(cls, aux, children):
        rk_step, dt_rk = aux
        return cls(int(rk_step), float(dt_rk), *children)


def small_step_prep_wrf(
    state: State,
    rk_step: int,
    dt_rk: float,
    *,
    metrics: DycoreMetrics,
    reference_state: State | None = None,
    ww: jax.Array | None = None,
) -> SmallStepPrepState:
    """Prepare one WRF acoustic small-step stage.

    Source: WRF ``dyn_em/module_small_step_em.F:125-285``.  The operational
    path keeps ``mu`` in the existing physical-mu convention used by
    ``advance_mu_t_wrf``; ``mu_work`` and ``muts`` carry the WRF work delta so
    ``muts - mut == mu - mu_save`` remains explicit across substeps.
    """

    reference = state if reference_state is None else reference_state
    theta_offset = jnp.asarray(THETA_BASE_OFFSET_K, dtype=state.theta.dtype)
    theta_ref = jnp.asarray(reference.theta) - theta_offset
    theta_cur = jnp.asarray(state.theta) - theta_offset

    # WRF mass semantics (module_small_step_em.F:172-215; solve_em.F passes
    # ``grid%mut`` = full stage-entry dry mass and ``grid%muts`` as the work-total
    # denominator into calc_p_rho/advance_w):
    #   ``mub``       = MUB  (base dry mass, INTENT(IN) ``mub``)
    #   ``mut``       = grid%mut = MUB + MU_current  (full *current* dry mass;
    #                   calculate_full in rk_step_prep, module_em.F:184-187)
    #   ``mu_work``   = small-step work ``MU_2`` after small_step_prep:
    #                     RK1 -> 0                (:187-190)
    #                     else -> MU_ref - MU_cur (:213-214)
    #   ``muts``      = MUB + MU_ref = mut + mu_work  (full *stage/reference* dry
    #                   mass; WRF MUTS = MUB+MU_2(RK1) / MUB+MU_1(else),
    #                   :172-175, :196-199).
    # Previously JAX set ``mut = MUB`` which fed calc_p_rho/calc_coef_w/advance_w
    # the wrong (base) total mass and broke the acoustic restoring loop.
    mub = _base_mu(state)
    mu_current = jnp.asarray(state.mu_perturbation)
    mu_ref = jnp.asarray(reference.mu_perturbation)
    mu_save = mu_current
    mu_work = jnp.zeros_like(mu_current) if int(rk_step) == 1 else mu_ref - mu_current
    mut = mub + mu_current
    muts = mut + mu_work

    # ``muu/muv`` = face averages of the full *current* dry mass ``mut`` (WRF
    # ``grid%muu/muv`` from calculate_full); ``muus/muvs`` = face averages of the
    # full *stage* dry mass ``muts`` (WRF :172-207, RK1 sets muus=muu, but for the
    # rest/balanced RK1 state reference==state so the two are identical anyway).
    muu = _u_face_average_2d(mut)
    muv = _v_face_average_2d(mut)
    muus = _u_face_average_2d(muts)
    muvs = _v_face_average_2d(muts)

    mass_u_ref = metrics.c1h[:, None, None] * muus[None, :, :] + metrics.c2h[:, None, None]
    mass_u_cur = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
    mass_v_ref = metrics.c1h[:, None, None] * muvs[None, :, :] + metrics.c2h[:, None, None]
    mass_v_cur = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
    mass_h_ref = metrics.c1h[:, None, None] * muts[None, :, :] + metrics.c2h[:, None, None]
    mass_h_cur = metrics.c1h[:, None, None] * mut[None, :, :] + metrics.c2h[:, None, None]
    mass_f_ref = metrics.c1f[:, None, None] * muts[None, :, :] + metrics.c2f[:, None, None]
    mass_f_cur = metrics.c1f[:, None, None] * mut[None, :, :] + metrics.c2f[:, None, None]

    u_work = (mass_u_ref * reference.u - mass_u_cur * state.u) / metrics.msfuy[None, :, :]
    v_work = (mass_v_ref * reference.v - mass_v_cur * state.v) / metrics.msfvx[None, :, :]
    theta_work = mass_h_ref * theta_ref - mass_h_cur * theta_cur
    w_work = (mass_f_ref * reference.w - mass_f_cur * state.w) / metrics.msfty[None, :, :]
    ph_work = reference.ph_perturbation - state.ph_perturbation

    p_pert, al, alt = diagnose_pressure_al_alt(state, None, metrics)
    pb = jnp.asarray(state.p_total) - jnp.asarray(state.p_perturbation)
    c2a = CPOVCV * (pb + p_pert) / jnp.maximum(jnp.abs(alt), jnp.asarray(1.0e-12, dtype=alt.dtype))
    cqu, cqv = moisture_coupling_factors(state)

    # WRF ``calc_php`` (rk_step_prep, module_em.F:181;
    # module_big_step_utilities_em.F:1259) builds the full geopotential at mass
    # points ONCE per RK stage from the STAGE-ENTRY geopotential ``grid%ph_2``
    # (absolute perturbation ph') and the base ``grid%phb``:
    #     php(k) = 0.5*(phb(k)+phb(k+1)+ph'(k)+ph'(k+1)) ,  k=1..nz .
    # ``state`` is the stage-entry state (= WRF ``grid%ph_2`` at rk_step_prep
    # time), so ``state.ph_perturbation`` is the WRF-faithful stage-entry ph'.
    # This frozen array is threaded into the acoustic core and used UNCHANGED by
    # advance_uv's 4th PGF term for every substep (replacing the per-substep
    # re-diagnosis from the live work-geopotential).
    phb_full = jnp.asarray(state.ph_total) - jnp.asarray(state.ph_perturbation)
    ph_full = jnp.asarray(state.ph_perturbation)
    php = 0.5 * (phb_full[:-1, :, :] + phb_full[1:, :, :] + ph_full[:-1, :, :] + ph_full[1:, :, :])

    ww_save = jnp.zeros_like(state.w) if ww is None else jnp.asarray(ww)
    return SmallStepPrepState(
        rk_step=int(rk_step),
        dt_rk=float(dt_rk),
        entry_state=state,
        theta_offset=theta_offset,
        u_1=jnp.asarray(reference.u),
        v_1=jnp.asarray(reference.v),
        w_1=jnp.asarray(reference.w),
        theta_1=theta_ref,
        theta_1_total=jnp.asarray(reference.theta),
        ph_1=jnp.asarray(reference.ph_perturbation),
        mu_1=mu_ref,
        u_save=jnp.asarray(state.u),
        v_save=jnp.asarray(state.v),
        w_save=jnp.asarray(state.w),
        t_save=theta_cur,
        ph_save=jnp.asarray(state.ph_perturbation),
        mu_save=mu_save,
        ww_save=ww_save,
        mub=mub,
        mut=mut,
        muu=muu,
        muv=muv,
        muts=muts,
        muus=muus,
        muvs=muvs,
        mu_work=mu_work,
        u_work=u_work,
        v_work=v_work,
        w_work=w_work,
        theta_work=theta_work,
        ph_work=ph_work,
        c2a=c2a,
        al=al,
        alt=alt,
        pb=pb,
        cqu=cqu,
        cqv=cqv,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        c1f=metrics.c1f,
        c2f=metrics.c2f,
        rdnw=metrics.rdnw,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msfty=metrics.msfty,
        php=php,
    )


__all__ = ["SmallStepPrepState", "small_step_prep_wrf"]
