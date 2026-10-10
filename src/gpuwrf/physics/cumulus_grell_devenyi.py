"""WRF ``cu_physics=93`` Grell-Devenyi ensemble (``phys/module_cu_gd.F``).

The line-faithful JAX port lives in :mod:`gpuwrf.physics._grell_cup_jax`
(``GRELLDRV`` + ``CUP_enss`` + ``neg_check``); the operational scan calls it
through :func:`gpuwrf.coupling.scan_adapters.gd_adapter`.  It is
machine-precision vs a pristine-WRF multi-column tile oracle
(``proofs/v034/oracle/cumulus_grell``) -- CPU-oracle-qualified; GPU and
coupled-forecast qualification pending.  Distinct WRF source path from the
cu=3 Grell-Freitas endpoint (``module_cu_gf_*``), which is untouched.
"""

from __future__ import annotations

from gpuwrf.physics._grell_cup_jax import gd_cup_enss_column, grelldrv_tile


def step_grell_devenyi_column(**tile_inputs):
    """Run WRF GRELLDRV on one tile (see ``grelldrv_tile``)."""

    return grelldrv_tile(**tile_inputs)


__all__ = ["step_grell_devenyi_column", "grelldrv_tile", "gd_cup_enss_column"]
