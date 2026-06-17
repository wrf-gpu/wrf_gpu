"""v0.17 candidate wrapper for WRF ``cu_physics=99`` previous Kain-Fritsch.

WRF option 99 calls ``phys/module_cu_kf.F:KFCPS``. That is a distinct source
path from the already-green KF-eta implementation used by ``cu_physics=1``
(``phys/module_cu_kfeta.F``). This module intentionally exposes only a RED
candidate endpoint that reuses the KF-eta JAX family for measurement against the
real old-KF oracle. It must not be added to ``CU_SCAN_ADAPTERS`` until the
source-specific parity gate passes.
"""

from __future__ import annotations

from gpuwrf.physics.cumulus_kf import step_kf_column


def step_previous_kf_column(*args, **kwargs):
    """Run the current KF-eta-family candidate for old-KF parity experiments.

    The signature is deliberately identical to :func:`step_kf_column`. The
    wrapper exists so dispatch/proof metadata can name ``cu_physics=99`` without
    implying that the old-KF WRF source path is green.
    """

    return step_kf_column(*args, **kwargs)


__all__ = ["step_previous_kf_column"]
