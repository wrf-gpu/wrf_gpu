"""Betts-Miller-Janjic (WRF ``cu_physics=2``) cumulus adapter surface.

This module implements a pure-JAX, jit/vmap-traceable BMJ-style column endpoint
for the v0.6.0 cumulus interface.  The committed parity report is the authority
on whether the implementation is WRF-faithful for the pristine
``module_cu_bmj.F`` savepoints; do not infer parity from the presence of this
module alone.

WRF BMJ differs from KF/GF/Tiedtke mass-flux schemes: it adjusts temperature and
water vapor toward post-convective reference profiles.  The driver returns only
``RTHCUTEN`` and ``RQVCUTEN`` plus cumulus precipitation and diagnostics.
"""

from __future__ import annotations

from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import (
    PhysicsCarry,
    PhysicsDiagnostics,
    PhysicsStepResult,
    PhysicsTendency,
)

config.update("jax_enable_x64", True)


CP_DRY = 1004.5
G = 9.81
P0 = 100000.0
R_D_OVER_CP = 287.0 / 1004.5
TREL = 2400.0
EPSQ = 1.0e-12
PQ0 = 379.90516
A2 = 17.2693882
A3 = 273.16
A4 = 35.86
D608 = 0.608
AVGEFI = 0.6


def _specific_from_mixing(qv: jax.Array) -> jax.Array:
    return jnp.maximum(EPSQ, qv / (1.0 + qv))


def _mixing_from_specific(q: jax.Array) -> jax.Array:
    return q / jnp.maximum(1.0 - q, 1.0e-12)


def _sat_specific(T: jax.Array, p: jax.Array) -> jax.Array:
    """WRF BMJ saturation specific-humidity expression."""

    denom = jnp.maximum(T - A4, 1.0e-6)
    qsat = PQ0 / jnp.maximum(p, 1.0) * jnp.exp(A2 * (T - A3) / denom)
    return jnp.maximum(qsat, EPSQ)


def _last_true_index(mask: jax.Array) -> jax.Array:
    idx = jnp.arange(mask.shape[0], dtype=jnp.int32)
    return jnp.max(jnp.where(mask, idx, jnp.zeros_like(idx)))


def _safe_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask_f = mask.astype(jnp.float64)
    return jnp.sum(values * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


@partial(jax.jit, static_argnames=("stepcu",))
def _bmj_column_arrays(
    temperature: jax.Array,
    qv: jax.Array,
    pressure: jax.Array,
    dz: jax.Array,
    rho: jax.Array,
    pi_exner: jax.Array,
    dt: float,
    *,
    stepcu: int = 1,
    xland: float = 1.0,
    cldefi: float = AVGEFI,
) -> tuple[jax.Array, ...]:
    """Run the traceable BMJ column kernel and return raw array leaves.

    Inputs follow WRF driver order: bottom-up mass levels, ``qv`` as mixing
    ratio, pressure in Pa, and ``pi_exner`` such that
    ``RTHCUTEN = dTdt / pi``.
    """

    del rho  # BMJDRV uses rho only to form DPRS; this approximation uses p/dz.
    T = jnp.asarray(temperature, jnp.float64)
    qv_mix = jnp.asarray(qv, jnp.float64)
    p = jnp.asarray(pressure, jnp.float64)
    dz = jnp.asarray(dz, jnp.float64)
    pi = jnp.asarray(pi_exner, jnp.float64)
    dt_f = jnp.asarray(dt, jnp.float64)
    stepcu_f = jnp.asarray(stepcu, jnp.float64)
    cld0 = jnp.asarray(cldefi, jnp.float64)
    nz = T.shape[0]

    idx = jnp.arange(nz, dtype=jnp.int32)
    z_top = jnp.cumsum(dz)
    z = z_top - 0.5 * dz
    q_spec = _specific_from_mixing(qv_mix)
    qsat = _sat_specific(T, p)
    rh = q_spec / jnp.maximum(qsat, EPSQ)
    theta = T / jnp.maximum(pi, 1.0e-12)
    thetae = theta * jnp.exp((2.683e6 / CP_DRY) * q_spec / jnp.maximum(T, 180.0))

    n_low = max(3, nz // 6)
    n_mid = max(n_low + 2, nz // 2)
    low_mask = idx < n_low
    mid_mask = (idx >= n_low) & (idx < n_mid)
    rh_low = _safe_mean(rh, low_mask)
    thetae_low = jnp.max(jnp.where(low_mask, thetae, -1.0e9))
    thetae_mid = jnp.min(jnp.where(mid_mask, thetae, 1.0e9))
    instability = thetae_low - thetae_mid

    psfc = p[0]
    deep_depth_pa = 20000.0 * psfc / 101300.0
    base_idx = jnp.asarray(1, jnp.int32)
    p_base = p[base_idx]
    deep_top_idx = _last_true_index((p_base - p) >= deep_depth_pa)
    shallow_top_idx = _last_true_index((p_base - p) >= 10000.0)
    deep_top_idx = jnp.maximum(deep_top_idx, base_idx + 2)
    shallow_top_idx = jnp.maximum(shallow_top_idx, base_idx + 2)
    shallow_top_idx = jnp.minimum(shallow_top_idx, jnp.asarray(max(3, nz // 3), jnp.int32))

    is_deep = (rh_low > 0.68) & (instability > 1.0) & (deep_top_idx > base_idx + 1)
    is_shallow = (~is_deep) & (rh_low > 0.62) & (instability > -2.0) & (shallow_top_idx > base_idx + 1)
    active = is_deep | is_shallow
    top_idx = jnp.where(is_deep, deep_top_idx, shallow_top_idx)
    cloud_mask = active & (idx >= base_idx) & (idx <= top_idx)

    z_base = z[base_idx]
    lapse = jnp.where(is_deep, 0.0057, 0.0045)
    tref = T[base_idx] - lapse * (z - z_base)
    tref = jnp.where(is_deep & (T < 273.16), jnp.maximum(tref, T - 8.0), tref)
    tref = jnp.where(is_shallow, jnp.maximum(tref, T - 1.0), tref)
    qsat_ref = _sat_specific(tref, p)
    target_rh = jnp.where(is_deep, 0.80, 0.72)
    qref_spec = jnp.minimum(q_spec, target_rh * qsat_ref)
    qref_mix = _mixing_from_specific(qref_spec)

    eff = jnp.where(is_deep, jnp.maximum(cld0, 0.2), 1.0)
    dtdt = jnp.where(cloud_mask, (tref - T) / TREL * eff, 0.0)
    dqdt_spec = jnp.where(cloud_mask, (qref_spec - q_spec) / TREL * eff, 0.0)
    rthcuten = dtdt / jnp.maximum(pi, 1.0e-12)
    rqvcuten = dqdt_spec / jnp.maximum((1.0 - q_spec) ** 2, 1.0e-12)

    dp = jnp.maximum(p - jnp.concatenate([p[1:], p[-1:]]), 0.0)
    water_sink = jnp.maximum(q_spec - qref_spec, 0.0)
    raincv = jnp.where(
        is_deep,
        jnp.sum(jnp.where(cloud_mask, water_sink * dp / G, 0.0)) * 1.0e3 / stepcu_f,
        0.0,
    )
    pratec = raincv / jnp.maximum(stepcu_f * dt_f, 1.0)
    cldefi_next = jnp.where(is_deep, jnp.maximum(0.2, jnp.minimum(1.0, cld0)), AVGEFI * (xland - 1.0) + (1.0 - (xland - 1.0)))
    cutop = jnp.where(active, top_idx + 1, 1).astype(jnp.float64)
    cubot = jnp.where(active, base_idx + 1, nz + 1).astype(jnp.float64)

    return (
        rthcuten,
        rqvcuten,
        raincv,
        pratec,
        cutop,
        cubot,
        cldefi_next,
        is_deep.astype(jnp.int32),
        is_shallow.astype(jnp.int32),
    )


def step_bmj_column(
    temperature: jax.Array,
    qv: jax.Array,
    pressure: jax.Array,
    dz: jax.Array,
    rho: jax.Array,
    pi_exner: jax.Array,
    dt: float,
    *,
    stepcu: int = 1,
    xland: float = 1.0,
    cldefi: float = AVGEFI,
) -> PhysicsStepResult:
    """Run one BMJ column and return the frozen physics-interface payload.

    The numeric work is delegated to ``_bmj_column_arrays`` so the actual column
    kernel has an explicit JIT-compatible array ABI.  The returned tendency keys
    match BMJDRV.
    """

    (
        rthcuten,
        rqvcuten,
        raincv,
        pratec,
        cutop,
        cubot,
        cldefi_next,
        is_deep,
        is_shallow,
    ) = _bmj_column_arrays(
        temperature,
        qv,
        pressure,
        dz,
        rho,
        pi_exner,
        dt,
        stepcu=stepcu,
        xland=xland,
        cldefi=cldefi,
    )
    tendency = PhysicsTendency(
        state_tendencies={
            "theta": rthcuten,
            "qv": rqvcuten,
        },
        accumulator_increments={"rainc_acc": raincv},
        diagnostics={
            "rthcuten": rthcuten,
            "rqvcuten": rqvcuten,
            "raincv": raincv,
            "pratec": pratec,
            "cutop": cutop,
            "cubot": cubot,
            "cldefi": cldefi_next,
            "trigger_deep": is_deep,
            "trigger_shallow": is_shallow,
        },
    )
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus={"cldefi": cldefi_next}),
        diagnostics=PhysicsDiagnostics(cumulus=tendency.diagnostics),
    )


def initial_bmj_cldefi(shape) -> jax.Array:
    """WRF BMJINIT default cloud efficiency, ``AVGEFI=(EFIMN+1)/2``."""

    return jnp.full(shape, AVGEFI, dtype=jnp.float64)


__all__ = ["AVGEFI", "initial_bmj_cldefi", "step_bmj_column"]
