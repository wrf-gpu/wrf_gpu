"""v0.17 fail-closed marker for WRF ``cu_physics=93`` Grell-Devenyi.

``cu_physics=93`` calls the unmodified WRF source path
``phys/module_cu_gd.F:GRELLDRV``. It is not equivalent to the operational
Grell-Freitas ``cu_physics=3`` endpoint, so this module exposes a named
entrypoint for dispatch/proof metadata but refuses execution until a real
traceable JAX column port is implemented and parity-gated.
"""

from __future__ import annotations


def step_grell_devenyi_column(*args, **kwargs):
    """Fail closed for the unported Grell-Devenyi column endpoint."""

    raise NotImplementedError(
        "cu_physics=93 Grell-Devenyi has no parity-proven JAX endpoint yet; "
        "run proofs/v017/run_cu_kfgrell_parity.py for the current RED proof."
    )


__all__ = ["step_grell_devenyi_column"]
