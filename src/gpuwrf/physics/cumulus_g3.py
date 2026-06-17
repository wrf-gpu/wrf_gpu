"""v0.17 fail-closed marker for WRF ``cu_physics=5`` Grell-3D ensemble.

Grell-3D calls ``phys/module_cu_g3.F:G3DRV`` and shares some historical data
structures with Grell-Freitas, but it is not the same source path as the
operational ``cu_physics=3`` GF endpoint. Until the G3 source-specific oracle
and JAX column endpoint are green, this module remains a named, non-operational
marker only.
"""

from __future__ import annotations


def step_grell3_column(*args, **kwargs):
    """Fail closed for the unported Grell-3D column endpoint."""

    raise NotImplementedError(
        "cu_physics=5 Grell-3D has no parity-proven JAX endpoint yet; "
        "run proofs/v017/run_cu_kfgrell_parity.py for the current RED proof."
    )


__all__ = ["step_grell3_column"]
