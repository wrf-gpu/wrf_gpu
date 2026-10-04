"""WRF REAL storage for dycore-owned leaves behind a default-off flag."""
import os

import jax
import jax.numpy as jnp


STATE_REAL_FIELDS = frozenset((
    "u", "v", "w", "theta", "p_total", "p_perturbation", "ph_total", "ph_perturbation",
    "mu_total", "mu_perturbation", "qv", "qc", "qr", "qi", "qs", "qg",
    "Ni", "Nr", "Ns", "Ng", "qke",
))
SCRATCH_REAL_FIELDS = (
    "t_2ave", "ww", "mudf", "muave", "muts", "ph_tend", "u_save", "v_save",
    "w_save", "t_save", "ph_save", "mu_save", "ww_save",
)


# GPUWRF_CARRY_REAL_ALL: every remaining float carry leaf is WRF REAL too (surface/PBL State
# leaves, held Noah/RRTMG radiation, KF carry and held rates). WRF has no DOUBLE carry leaf.
REAL_ALL_CARRY_FIELDS = ("noahmp_rad", "cumulus_carry", "cumulus_tendencies", "radiation_diagnostics")


def enabled():
    return os.environ.get("GPUWRF_DYN_CARRY_FP32", "0") == "1"


def real_all_enabled():
    return enabled() and os.environ.get("GPUWRF_CARRY_REAL_ALL", "0") == "1"


def real_dtype(default):
    """REAL under GPUWRF_CARRY_REAL_ALL, else ``default`` (the historical glue dtype)."""
    return jnp.float32 if real_all_enabled() else default


def real_tree(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) and x.dtype != jnp.float32 else x,
        tree,
    )


def like(tree, reference):
    """``tree`` cast leaf-wise to the dtypes of ``reference`` (lax.cond/loop carry seams)."""
    return jax.tree.map(
        lambda x, r: x.astype(r.dtype) if hasattr(x, "dtype") and x.dtype != r.dtype else x,
        tree, reference,
    )


def real_carry(carry):
    """Seed: the non-State carry leaves of REAL_ALL_CARRY_FIELDS as REAL (no-op when the flag is off)."""
    if not real_all_enabled():
        return carry
    return carry.replace(**{name: real_tree(getattr(carry, name)) for name in REAL_ALL_CARRY_FIELDS})


def real_state(state):
    real_all = real_all_enabled()
    updates = {}
    for name in state.__slots__:
        value = getattr(state, name)
        if (real_all or name in STATE_REAL_FIELDS or name.endswith("_bdy")) and hasattr(value, "dtype"):
            if jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != jnp.float32:
                updates[name] = value.astype(jnp.float32)
    return state.replace(_cast=False, **updates)


def real_scratch(carry):
    return carry.replace(**{name: jnp.asarray(getattr(carry, name), jnp.float32)
                            for name in SCRATCH_REAL_FIELDS})
