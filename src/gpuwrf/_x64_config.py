"""Central JAX x64 bootstrap policy.

The production default remains fp64.  The only exception is an explicit audit
mode that needs to prove a global-x64-off arm is real.
"""

from __future__ import annotations

import os

from jax import config


_FALSE = {"", "0", "false", "off", "no"}
_GLOBAL_FP32_MODES = {
    "aggressive",
    "aggressive_global",
    "aggressive_global_fp32",
    "global_fp32",
    "jax_no_x64",
    "no_x64",
}


def fp32_mode_label() -> str:
    """Return the normalized opt-in fp32/audit mode label."""

    return os.environ.get("GPUWRF_FP32_MODE", "").strip().lower()


def should_force_jax_x64() -> bool:
    """Whether this process should force JAX x64 on at import time."""

    mode = fp32_mode_label()
    if mode in _GLOBAL_FP32_MODES:
        return False
    # Compatibility backdoor for one-off dtype audits.
    allow_x64_false = os.environ.get("GPUWRF_ALLOW_JAX_X64_FALSE", "").strip().lower()
    if allow_x64_false not in _FALSE:
        return False
    return True


def configure_jax_x64() -> bool:
    """Apply the package x64 policy and return the resulting force decision."""

    force = should_force_jax_x64()
    if force:
        config.update("jax_enable_x64", True)
    return force


__all__ = ["configure_jax_x64", "fp32_mode_label", "should_force_jax_x64"]
