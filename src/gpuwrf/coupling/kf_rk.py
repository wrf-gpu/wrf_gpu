"""WRF KF source lifetime between the column driver and the RK step.

The operational carry keeps the existing uncoupled KF rate tuple at step
boundaries.  A coupled copy belongs to one RK step and is decoupled by final
mass after advance_ppt, matching solve_em.F:3649/3669.
"""
from __future__ import annotations

import os

import jax.numpy as jnp

KF_MOIST_SPECIES = ("qv", "qc", "qr", "qi", "qs")


def kf_tend_rk_enabled() -> bool:
    """Trace-time opt-in, covered by the live GPUWRF E80 environment key."""
    return os.environ.get("GPUWRF_KF_TEND_RK_WRF", "0") == "1"


def couple_kf_rates(rates, mu_total, metrics):
    """calculate_phy_tend: couple each of six REAL KF profiles with entry mut."""
    dtype = jnp.asarray(rates[0]).dtype
    mass = (jnp.asarray(metrics.c1h, dtype)[:, None, None]
            * jnp.asarray(mu_total, dtype)[None]
            + jnp.asarray(metrics.c2h, dtype)[:, None, None])
    return tuple(mass * jnp.asarray(rate, dtype) for rate in rates[:6]) + (rates[6],)


def kf_add_a2a(rate, *, specified_or_nested: bool, periodic_x: bool = False):
    """add_a2a: omit the outer mass-grid ring for specified/nested domains."""
    if not specified_or_nested:
        return rate
    x = slice(None) if periodic_x else slice(1, -1)
    return jnp.zeros_like(rate).at[..., 1:-1, x].set(rate[..., 1:-1, x])


def kf_scalar_tendencies(coupled_rates, *, specified_or_nested: bool, periodic_x: bool = False):
    """Unscaled moist_tend sources; advection alone receives msfty in RK."""
    return {name: kf_add_a2a(rate, specified_or_nested=specified_or_nested,
                           periodic_x=periodic_x)
            for name, rate in zip(KF_MOIST_SPECIES, coupled_rates[1:6], strict=True)}


def finish_kf_rates(coupled_rates, nca, final_mu_total, metrics, dt):
    """advance_ppt then phy_prep_part2, after all RK consumers used the rates."""
    dtype = jnp.asarray(coupled_rates[0]).dtype
    dt = jnp.asarray(dt, dtype)
    nca = jnp.asarray(nca, dtype)
    # WRF NINT on a positive countdown; consumption precedes this clear.
    clear = (nca > 0) & (jnp.floor(nca / dt + jnp.asarray(.5, dtype)) <= 1)
    used = tuple(jnp.where(clear[None], jnp.asarray(0, dtype), rate)
                 for rate in coupled_rates[:6])
    final_mass = (jnp.asarray(metrics.c1h, dtype)[:, None, None]
                  * jnp.asarray(final_mu_total, dtype)[None]
                  + jnp.asarray(metrics.c2h, dtype)[:, None, None])
    rates = tuple(rate / final_mass for rate in used) + (coupled_rates[6],)
    next_nca = jnp.where(nca > 0, nca - dt, nca)
    rain_increment = dt * jnp.asarray(coupled_rates[6], dtype)
    return rates, next_nca, rain_increment
