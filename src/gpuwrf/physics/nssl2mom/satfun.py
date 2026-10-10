"""Saturation lookup tables of nssl_2mom_init (module lines 1569-1584), evaluated at the use site.

WRF builds ``tabqvs/tabqis/dtabqvs/dtabqis(1:nqsat)`` (nqsat = 1,000,001) at init and indexes
them with ``ltemq = Int((T - 163.15)/fqsat + 1.5)`` clamped to [1, nqsat].  The port evaluates the
same REAL expressions for the requested index instead of materialising the 4 x 1e6 table
(iqvsopt = 1: Bolton formulation for water).
"""

from __future__ import annotations

import jax.numpy as jnp


def ltemq_index(temp, C, R):
    """``ltemq = Min(nqsat, Max(1, Int((temp-163.15)/fqsat + 1.5)))`` (driver line 2988)."""
    temp = jnp.asarray(temp, R)
    l = jnp.trunc((temp - 163.15) / jnp.asarray(C.fqsat, R) + 1.5).astype(jnp.int32)
    return jnp.minimum(C.nqsat, jnp.maximum(1, l))


def _temq(l, C, R):
    return jnp.asarray(163.15, R) + (l - 1).astype(R) * jnp.asarray(C.fqsat, R)


def tabqvs(l, C, R):
    t = _temq(l, C, R)
    return jnp.exp(jnp.asarray(C.cawbolton, R) * (t - 273.15) / (t - jnp.asarray(C.cbwbolton, R)))


def tabqis(l, C, R):
    t = _temq(l, C, R)
    return jnp.exp(jnp.asarray(C.cai, R) * (t - 273.15) / (t - jnp.asarray(C.cbi, R)))


def dtabqvs(l, C, R):
    t = _temq(l, C, R)
    a, b = jnp.asarray(C.cawbolton, R), jnp.asarray(C.cbwbolton, R)
    return ((-a * (-273.15 + t)) / (t - b) ** 2 + a / (t - b)) * tabqvs(l, C, R)


def dtabqis(l, C, R):
    t = _temq(l, C, R)
    a, b = jnp.asarray(C.cai, R), jnp.asarray(C.cbi, R)
    return ((-a * (-273.15 + t)) / (t - b) ** 2 + a / (t - b)) * tabqis(l, C, R)
