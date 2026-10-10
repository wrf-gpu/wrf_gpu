"""WRF ``cu_physics=5`` Grell-3D ensemble (``phys/module_cu_g3.F``).

The line-faithful JAX port lives in :mod:`gpuwrf.physics._grell_cup_jax`
(``G3DRV`` + ``CUP_enss_3d`` + ``conv_grell_spread3d``); the operational scan
calls it through :func:`gpuwrf.coupling.scan_adapters.g3_adapter`.  It is
machine-precision vs a pristine-WRF multi-column tile oracle
(``proofs/v034/oracle/cumulus_grell``) -- CPU-oracle-qualified; GPU and
coupled-forecast qualification pending.
"""

from __future__ import annotations

from gpuwrf.physics._grell_cup_jax import g3_cup_enss_3d_column, g3drv_tile


def step_grell3_column(**tile_inputs):
    """Run WRF G3DRV + conv_grell_spread3d on one tile (see ``g3drv_tile``)."""

    return g3drv_tile(**tile_inputs)


__all__ = ["step_grell3_column", "g3drv_tile", "g3_cup_enss_3d_column"]
