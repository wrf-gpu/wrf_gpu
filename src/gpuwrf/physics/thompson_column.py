"""JAX Thompson-column source/sink subset for M5-S1."""

from __future__ import annotations

from functools import partial
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.thompson_constants import CP, EPS, HGFR, LFUS, LSUB, R1, R2, R_D, RV, T_0, XM0I
from gpuwrf.physics.thompson_saturation import (
    cp_inverse,
    latent_heat_vaporization,
    saturation_mixing_ratio_ice,
    saturation_mixing_ratio_liquid,
)


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class ThompsonColumnState:
    """Pytree for a batch of independent Thompson columns on mass levels."""

    __slots__ = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T", "p", "rho")

    def __init__(self, qv, qc, qr, qi, qs, qg, Ni, Nr, T, p, rho) -> None:
        self.qv = qv
        self.qc = qc
        self.qr = qr
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.Ni = Ni
        self.Nr = Nr
        self.T = T
        self.p = p
        self.rho = rho

    def replace(self, **updates) -> "ThompsonColumnState":
        """Returns a same-layout pytree with explicit field updates."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents column arrays as JAX leaves for JIT and scan transforms."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the column state after JAX transformations."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for cache/debug tests."""

        if not isinstance(other, ThompsonColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside JIT; never used in the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


def _leaves(state: ThompsonColumnState) -> Iterable[jax.Array]:
    """Centralizes leaf iteration for equality, hashing, and byte accounting."""

    return (getattr(state, name) for name in ThompsonColumnState.__slots__)


def density_from_pressure_temperature(p, T, qv):
    """Matches mp_gt_driver's rho diagnostic at module_mp_thompson.F.pre line 1270."""

    return 0.622 * p / (R_D * T * (qv + 0.622))


def _clip_species(state: ThompsonColumnState) -> ThompsonColumnState:
    """Mirrors Thompson's non-negative hydrometeor preamble and qv floor."""

    return state.replace(
        qv=jnp.maximum(state.qv, 1.0e-10),
        qc=jnp.maximum(state.qc, 0.0),
        qr=jnp.maximum(state.qr, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qg=jnp.maximum(state.qg, 0.0),
        Ni=jnp.maximum(state.Ni, 0.0),
        Nr=jnp.maximum(state.Nr, 0.0),
    )


def _finish(state: ThompsonColumnState) -> ThompsonColumnState:
    """Applies WRF-style small-species floors and recomputes density."""

    qv = jnp.maximum(state.qv, 1.0e-10)
    qc = jnp.where(state.qc <= R1, 0.0, jnp.maximum(state.qc, 0.0))
    qr = jnp.where(state.qr <= R1, 0.0, jnp.maximum(state.qr, 0.0))
    qi = jnp.where(state.qi <= R1, 0.0, jnp.maximum(state.qi, 0.0))
    qs = jnp.where(state.qs <= R1, 0.0, jnp.maximum(state.qs, 0.0))
    qg = jnp.where(state.qg <= R1, 0.0, jnp.maximum(state.qg, 0.0))
    Ni = jnp.where(qi <= R1, 0.0, jnp.maximum(state.Ni, 0.0))
    Nr = jnp.where(qr <= R1, 0.0, jnp.maximum(state.Nr, 0.0))
    T = jnp.maximum(state.T, 50.0)
    return state.replace(qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, Ni=Ni, Nr=Nr, T=T, rho=density_from_pressure_temperature(state.p, T, qv))


def _saturation_adjustment(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Implements the 3-iteration Thompson cloud condensation adjustment."""

    del dt
    qvs = saturation_mixing_ratio_liquid(state.p, state.T)
    lvap = latent_heat_vaporization(state.T)
    ocp = cp_inverse(state.qv)
    lvt2 = lvap * lvap * ocp / RV / (state.T * state.T)
    clap = (state.qv - qvs) / (1.0 + lvt2 * qvs)
    for _ in range(3):
        expo = jnp.exp(lvt2 * clap)
        fcd = qvs * expo - state.qv + clap
        dfcd = qvs * lvt2 * expo + 1.0
        clap = clap - fcd / dfcd
    ssatw = state.qv / qvs - 1.0
    active = (ssatw > EPS) | ((ssatw < -EPS) & (state.qc > 0.0))
    clap = jnp.where(active, clap, 0.0)
    clap = jnp.where(clap < 0.0, jnp.maximum(clap, -state.qc), jnp.minimum(clap, state.qv - 1.0e-10))
    return state.replace(
        qv=state.qv - clap,
        qc=state.qc + clap,
        T=state.T + lvap * ocp * clap,
    )


def _warm_rain(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Moves cloud water to rain and evaporates rain while conserving total water."""

    autoconv_source = jnp.maximum(state.qc - 1.0e-4, 0.0)
    autoconv = jnp.minimum(state.qc, autoconv_source * (1.0 - jnp.exp(-float(dt) / 900.0)))
    accretion = jnp.minimum(state.qc - autoconv, 0.18 * state.qr * (1.0 - jnp.exp(-float(dt) / 300.0)))
    transfer = jnp.maximum(0.0, autoconv + accretion)
    nr_gain = autoconv / jnp.maximum(4.0 / 3.0 * jnp.pi * 1000.0 * (80.0e-6) ** 3, R2)
    state = state.replace(qc=state.qc - transfer, qr=state.qr + transfer, Nr=state.Nr + nr_gain)

    qvs = saturation_mixing_ratio_liquid(state.p, state.T)
    deficit = jnp.maximum(qvs - state.qv, 0.0)
    lvap = latent_heat_vaporization(state.T)
    ocp = cp_inverse(state.qv)
    evap = jnp.minimum(state.qr, jnp.minimum(0.20 * deficit, state.qr * (1.0 - jnp.exp(-float(dt) / 900.0))))
    nr_loss = jnp.where(state.qr > 0.0, state.Nr * evap / jnp.maximum(state.qr, R1), 0.0)
    return state.replace(
        qv=state.qv + evap,
        qr=state.qr - evap,
        Nr=jnp.maximum(0.0, state.Nr - nr_loss),
        T=state.T - lvap * ocp * evap,
    )


def _ice_sources(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Handles ice deposition/sublimation, freezing, melting, and phase partitioning."""

    del dt
    ocp = cp_inverse(state.qv)
    lvap = latent_heat_vaporization(state.T)
    lfus2 = LSUB - lvap

    qi_melt = jnp.where(state.T > T_0, state.qi, 0.0)
    state = state.replace(qc=state.qc + qi_melt, qi=state.qi - qi_melt, Ni=jnp.where(qi_melt > 0.0, 0.0, state.Ni), T=state.T - LFUS * ocp * qi_melt)

    qc_freeze = jnp.where(state.T < HGFR, state.qc, 0.0)
    state = state.replace(qc=state.qc - qc_freeze, qi=state.qi + qc_freeze, Ni=state.Ni + qc_freeze / XM0I, T=state.T + lfus2 * ocp * qc_freeze)

    rain_freeze_fraction = jnp.clip((HGFR - state.T) / 40.0, 0.0, 1.0)
    rain_freeze = state.qr * rain_freeze_fraction
    state = state.replace(qr=state.qr - rain_freeze, qg=state.qg + rain_freeze, Nr=state.Nr * (1.0 - rain_freeze_fraction), T=state.T + lfus2 * ocp * rain_freeze)

    warm_fraction = jnp.clip((state.T - T_0) / 20.0, 0.0, 1.0)
    qs_melt = state.qs * warm_fraction
    qg_melt = state.qg * warm_fraction
    state = state.replace(qs=state.qs - qs_melt, qg=state.qg - qg_melt, qr=state.qr + qs_melt + qg_melt, T=state.T - LFUS * ocp * (qs_melt + qg_melt))

    qvsi = saturation_mixing_ratio_ice(state.p, state.T)
    supersat = jnp.maximum(state.qv - qvsi, 0.0)
    existing_ice = state.qi + state.qs + state.qg
    deposition = jnp.minimum(state.qv - 1.0e-10, 0.25 * supersat)
    deposition = jnp.where(state.T < T_0, deposition, 0.0)
    ice_weight = jnp.where(existing_ice > R1, state.qi / jnp.maximum(existing_ice, R1), 1.0)
    snow_weight = jnp.where(existing_ice > R1, state.qs / jnp.maximum(existing_ice, R1), 0.0)
    graupel_weight = jnp.where(existing_ice > R1, state.qg / jnp.maximum(existing_ice, R1), 0.0)
    state = state.replace(
        qv=state.qv - deposition,
        qi=state.qi + deposition * ice_weight,
        qs=state.qs + deposition * snow_weight,
        qg=state.qg + deposition * graupel_weight,
        Ni=state.Ni + deposition * ice_weight / XM0I,
        T=state.T + LSUB * ocp * deposition,
    )

    qvsi = saturation_mixing_ratio_ice(state.p, state.T)
    subsat = jnp.maximum(qvsi - state.qv, 0.0)
    existing_ice = state.qi + state.qs + state.qg
    sublimation = jnp.minimum(existing_ice, 0.25 * subsat)
    sublimation = jnp.where(state.T < T_0, sublimation, 0.0)
    ice_weight = jnp.where(existing_ice > R1, state.qi / jnp.maximum(existing_ice, R1), 0.0)
    snow_weight = jnp.where(existing_ice > R1, state.qs / jnp.maximum(existing_ice, R1), 0.0)
    graupel_weight = jnp.where(existing_ice > R1, state.qg / jnp.maximum(existing_ice, R1), 0.0)
    return state.replace(
        qv=state.qv + sublimation,
        qi=state.qi - sublimation * ice_weight,
        qs=state.qs - sublimation * snow_weight,
        qg=state.qg - sublimation * graupel_weight,
        Ni=jnp.maximum(0.0, state.Ni - state.Ni * sublimation * ice_weight / jnp.maximum(state.qi, R1)),
        T=state.T - LSUB * ocp * sublimation,
    )


def _debug_checks(state: ThompsonColumnState, debug: bool) -> ThompsonColumnState:
    """Threads zero-production-cost debug assertions through the public kernel."""

    qv = assert_finite(state.qv, "thompson.qv", enabled=debug)
    qc = assert_physical_bounds(state.qc, 0.0, 1.0, "thompson.qc", enabled=debug)
    qr = assert_physical_bounds(state.qr, 0.0, 1.0, "thompson.qr", enabled=debug)
    qi = assert_physical_bounds(state.qi, 0.0, 1.0, "thompson.qi", enabled=debug)
    qs = assert_physical_bounds(state.qs, 0.0, 1.0, "thompson.qs", enabled=debug)
    qg = assert_physical_bounds(state.qg, 0.0, 1.0, "thompson.qg", enabled=debug)
    Ni = assert_physical_bounds(state.Ni, 0.0, 1.0e12, "thompson.Ni", enabled=debug)
    Nr = assert_physical_bounds(state.Nr, 0.0, 1.0e12, "thompson.Nr", enabled=debug)
    T = assert_physical_bounds(state.T, 50.0, 400.0, "thompson.T", enabled=debug)
    p = assert_physical_bounds(state.p, 1.0, 120000.0, "thompson.p", enabled=debug)
    rho = assert_finite(state.rho, "thompson.rho", enabled=debug)
    return state.replace(qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, Ni=Ni, Nr=Nr, T=T, p=p, rho=rho)


def _step_thompson_column_impl(state: ThompsonColumnState, dt: float, debug: bool) -> ThompsonColumnState:
    """Runs the fused source/sink Thompson column body with sedimentation removed."""

    state = _clip_species(state)
    state = _debug_checks(state, debug)
    state = _saturation_adjustment(state, dt)
    state = _ice_sources(state, dt)
    state = _warm_rain(state, dt)
    state = _finish(state)
    return _debug_checks(state, debug)


@partial(jax.jit, static_argnames=("dt", "debug"))
def step_thompson_column(state: ThompsonColumnState, dt: float, *, debug: bool = False) -> ThompsonColumnState:
    """Advances one Thompson source/sink column step under one JAX program."""

    return _step_thompson_column_impl(state, dt, debug)
