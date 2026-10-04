"""WRF REAL work behind a default-off Noah-MP precision gate.

Pristine module_sf_noahmplsm.F declares scoped physics REAL. Its sole explicit
DOUBLE local (S_NODE, :9623) belongs to GROUNDWATER, excluded by opt_run=3.
Integer categories, static Python controls and None remain unchanged.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp


def native_real_enabled():
    """Trace-time flag; clear caller JIT caches after changing it."""
    return os.environ.get("GPUWRF_NOAHMP_NATIVE_REAL", "0") == "1"


def real_dtype():
    return jnp.float32 if native_real_enabled() else jnp.float64


def real_tree(tree):
    """Preserve the legacy object exactly while the flag is off."""
    if not native_real_enabled():
        return tree

    def cast(value):
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.floating):
            return jnp.asarray(value, dtype=jnp.float32)
        return value

    return jax.tree_util.tree_map(cast, tree)


def real_scalar(value):
    """Type scalar-only arithmetic as REAL; retain legacy weak literals off."""
    return jnp.asarray(value, jnp.float32) if native_real_enabled() else value
